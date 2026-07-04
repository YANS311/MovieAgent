"""
多路召回模块 (Multi-Channel Recall)
================================================
从多个数据源并行召回候选电影，实现"宽进"策略。

召回路径:
  A. 向量语义召回 (Vector Recall)     - FAISS/BGE 语义相似度
  B. 内容特征召回 (Content Recall)    - Genre/Actor 共现匹配
  C. 模型推理召回 (Model Recall)      - SKB-FMLP 预测分
  D. 知识图谱召回 (KG Recall)         - Neo4j 拓扑遍历
  E. 热门兜底召回 (Hot Recall)        - 全局热度排序

输入: 用户对象 或 查询文本
输出: List[Dict[movie_id, score, source]]
================================================
"""

import time
import numpy as np
from collections import Counter
from django.core.cache import cache
from django.db.models import Q, Count, Max

from myapp.models import Movie, UserRating, Collect, Genre, Actor, Rec


# ── 内部辅助：获取用户时间感知画像 ──────────────────────────────────────────
def _get_user_liked_ids(user, limit=50):
    """
    获取用户近期正向交互（评分>=6.0 + 收藏），按时间倒序去重
    """
    recent_ratings = list(
        UserRating.objects.filter(user=user, score__gte=6.0)
        .order_by('-comment_time')
        .values_list('movie_id', flat=True)[:limit]
    )
    recent_collects = list(
        Collect.objects.filter(user=user)
        .order_by('-collect_time')
        .values_list('movie_id', flat=True)[:limit]
    )
    return list(dict.fromkeys(recent_ratings + recent_collects))


# =============================================================
# 路径 A: 向量语义召回 (Vector Semantic Recall)
# =============================================================
def vector_recall(query_text, excluded_ids=None, k=60, rag_resources=None):
    """
    用用户画像文本或查询文本在 FAISS 向量库做相似度检索。

    Args:
        query_text: 检索文本（用户画像/自然语言查询）
        excluded_ids: 排除的电影ID列表
        k: 召回数量
        rag_resources: RAG资源字典 {"vectorstore": FAISS实例}

    Returns:
        List[Tuple[int, float]] - [(movie_id, similarity_score), ...]
    """
    if excluded_ids is None:
        excluded_ids = []

    results = []
    vectorstore = (rag_resources or {}).get("vectorstore")
    if not vectorstore:
        return results

    try:
        docs_scores = vectorstore.similarity_search_with_score(
            query_text, k=k + len(excluded_ids)
        )
        seen = set(excluded_ids)
        for doc, dist_score in docs_scores:
            mid = (
                doc.metadata.get('id')
                or doc.metadata.get('movie_id')
                or doc.metadata.get('mid')
            )
            if not mid:
                continue
            mid = int(mid)
            if mid in seen:
                continue
            seen.add(mid)
            # FAISS L2 距离转相似度（归一化向量 dist≈2*(1-cos_sim)）
            sim = max(0.0, 1.0 - float(dist_score) / 2.0)
            results.append({'movie_id': mid, 'score': sim, 'source': 'vector'})
            if len(results) >= k:
                break
    except Exception as e:
        print(f"[Recall/Vector] 异常: {e}")

    return results


# =============================================================
# 路径 B: 内容特征召回 (Content-Based Recall)
# =============================================================
def content_recall(user, excluded_ids=None, k=60):
    """
    基于 Genre + Actor 共现匹配的内容召回。
    对冷启动用户自动降级为热门召回。

    Returns:
        List[Dict] - [{'movie_id': int, 'score': float, 'source': 'content'}, ...]
    """
    if excluded_ids is None:
        excluded_ids = []

    liked_ids = _get_user_liked_ids(user, limit=100)
    liked_set = set(liked_ids)

    pref_genres = list(
        Movie.genres.through.objects
        .filter(movie_id__in=liked_ids)
        .values_list('genre_id', flat=True)
    )
    pref_actors = list(
        Actor.objects.filter(movie__id__in=liked_ids)
        .annotate(c=Count('id'))
        .order_by('-c')
        .values_list('id', flat=True)[:50]
    )

    # 冷启动降级：热门
    if not pref_genres and not pref_actors:
        hot = list(
            Movie.objects.exclude(id__in=excluded_ids)
            .order_by('-vote_count', '-score')
            .values_list('id', 'vote_count')[:k]
        )
        max_vc = hot[0][1] if hot else 1
        return [{'movie_id': mid, 'score': vc / max_vc, 'source': 'hot'} for mid, vc in hot]

    candidates = (
        Movie.objects.filter(vote_count__gt=50)
        .exclude(id__in=excluded_ids + liked_ids)
        .annotate(
            match_score=(
                Count('genres', filter=Q(genres__in=pref_genres), distinct=True) * 2
                + Count('actors', filter=Q(actors__in=pref_actors), distinct=True) * 3
            )
        )
        .filter(match_score__gt=0)
        .order_by('-match_score')
        .values_list('id', 'match_score')[:k]
    )
    items = list(candidates)
    if not items:
        return []
    max_s = items[0][1] or 1
    return [{'movie_id': mid, 'score': score / max_s, 'source': 'content'} for mid, score in items]


# =============================================================
# 路径 C: 模型推理召回 (Model-Based Recall)
# =============================================================
def model_recall(user, k=60):
    """
    从 Rec 表读取 SKB-FMLP / DeepFM 预测分。

    Returns:
        List[Dict] - [{'movie_id': int, 'score': float, 'source': 'model'}, ...]
    """
    recs = (
        Rec.objects.filter(user=user)
        .order_by('-rating')
        .values_list('movie_id', 'rating')[:k]
    )
    items = list(recs)
    if not items:
        return []
    max_r = items[0][1] or 1.0
    return [{'movie_id': mid, 'score': (r or 0) / max_r, 'source': 'model'} for mid, r in items]


# =============================================================
# 路径 D: 知识图谱召回 (Knowledge Graph Recall)
# =============================================================
def kg_recall(user, neo_graph=None, k=30):
    """
    利用 Neo4j 知识图谱进行拓扑遍历召回。
    从用户高分电影出发，通过导演/类型关系发现新电影。

    Returns:
        List[Dict] - [{'movie_id': int, 'score': float, 'source': 'kg'}, ...]
    """
    if neo_graph is None:
        return []

    liked_ids = list(
        UserRating.objects.filter(user=user, score__gte=7.5)
        .order_by('-comment_time')
        .values_list('movie_id', flat=True)[:15]
    )
    if not liked_ids:
        return []

    try:
        cypher = """
        MATCH (h:Movie)<-[:DIRECTED_BY]-(d:Person)-[:DIRECTED_BY]->(m:Movie)
        WHERE h.mid IN $hist_mids AND NOT m.mid IN $hist_mids AND h <> m
        RETURN DISTINCT m.mid AS mid, m.title AS title, d.name AS director, count(*) AS weight
        ORDER BY weight DESC
        LIMIT $limit
        """
        rows = neo_graph.run(cypher, hist_mids=liked_ids, limit=k).data()
        results = []
        for r in rows:
            mid = r.get('mid')
            if mid:
                results.append({
                    'movie_id': int(mid),
                    'score': min(1.0, r.get('weight', 1) / 3.0),
                    'source': 'kg'
                })
        return results
    except Exception as e:
        print(f"[Recall/KG] 异常: {e}")
        return []


# =============================================================
# 路径 E: 热门兜底召回 (Popularity-Based Recall)
# =============================================================
def hot_recall(excluded_ids=None, k=60):
    """
    基于全局热度的兜底召回。
    """
    if excluded_ids is None:
        excluded_ids = []

    hot = list(
        Movie.objects.exclude(id__in=excluded_ids)
        .order_by('-vote_count', '-score')
        .values_list('id', 'vote_count')[:k]
    )
    max_vc = hot[0][1] if hot else 1
    return [{'movie_id': mid, 'score': vc / max_vc, 'source': 'hot'} for mid, vc in hot]


# =============================================================
# RRF 融合 (Reciprocal Rank Fusion)
# =============================================================
def rrf_merge(channels, k_rrf=60, weights=None):
    """
    Reciprocal Rank Fusion 多路召回融合。

    Args:
        channels: List[List[Dict]] - 每路召回结果列表
        k_rrf: RRF 参数（通常60）
        weights: 每路权重

    Returns:
        List[Tuple[int, float]] - [(movie_id, rrf_score), ...] 已按降序
    """
    if weights is None:
        weights = [1.0] * len(channels)

    score_map = {}
    for channel, w in zip(channels, weights):
        for rank_idx, item in enumerate(channel):
            mid = item['movie_id']
            score_map[mid] = score_map.get(mid, 0.0) + w / (k_rrf + rank_idx + 1)

    merged = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return merged  # [(movie_id, rrf_score), ...]


# =============================================================
# 主入口：多路召回融合
# =============================================================
def multi_channel_recall(user, query_text=None, top_k=30,
                          neo_graph=None, rag_resources=None,
                          weights=None, force_refresh=False):
    """
    多路召回主函数。

    Args:
        user: Django User 对象
        query_text: 查询文本（可选，用于向量召回）
        top_k: 最终返回数量
        neo_graph: Neo4j 图实例（可选）
        rag_resources: RAG资源字典（可选）
        weights: 各路权重 [vector, content, model, kg]
        force_refresh: 是否强制刷新缓存

    Returns:
        Tuple[List[Dict], Dict] - (召回结果列表, 统计信息)
    """
    t_start = time.time()

    # 缓存
    cache_key = f"multi_recall_{user.id}_{hash(query_text or '')}"
    if not force_refresh:
        cached = cache.get(cache_key)
        if cached:
            return cached

    # 排除已看过的电影
    liked_ids = _get_user_liked_ids(user, limit=200)
    liked_set = set(liked_ids)

    # 构建用户画像文本（用于向量召回）
    if not query_text:
        query_text = _build_profile_query(user, liked_ids)

    # ── 并行执行多路召回 ──
    vec_results = vector_recall(query_text, liked_ids, k=80, rag_resources=rag_resources)
    cb_results = content_recall(user, liked_ids, k=80)
    model_results = model_recall(user, k=80)
    kg_results = kg_recall(user, neo_graph=neo_graph, k=30)

    stats = {
        'vector': len(vec_results),
        'content': len(cb_results),
        'model': len(model_results),
        'kg': len(kg_results),
        'query_text': query_text[:100],
    }

    # ── RRF 融合 ──
    if weights is None:
        weights = [1.2, 1.0, 1.5, 0.8]  # vector:content:model:kg

    channels = [vec_results, cb_results, model_results, kg_results]
    merged = rrf_merge(channels, k_rrf=60, weights=weights)

    # 过滤已看过的，取 top_k
    final_results = []
    for mid, rrf_score in merged:
        if mid in liked_set:
            continue
        final_results.append({'movie_id': mid, 'score': rrf_score, 'source': 'rrf_merged'})
        if len(final_results) >= top_k:
            break

    # 绝对兜底
    if not final_results:
        final_results = hot_recall(liked_ids, k=top_k)
        stats['fallback'] = 'hot'

    stats['total_recall'] = len(final_results)
    stats['latency_ms'] = int((time.time() - t_start) * 1000)

    result = (final_results, stats)
    cache.set(cache_key, result, 1800)  # 缓存30分钟
    return result


# ── 辅助：构建用户画像查询文本 ──────────────────────────────────────────
def _build_profile_query(user, liked_ids, top_n_genres=5, top_n_actors=5):
    """
    将用户历史行为转化为自然语言查询文本（用于向量召回）
    """
    genre_counter = Counter(
        Movie.genres.through.objects.filter(movie_id__in=liked_ids)
        .values_list('genre__name', flat=True)
    )
    top_genres = [g for g, _ in genre_counter.most_common(top_n_genres)]

    actor_counter = Counter(
        Actor.objects.filter(movie__id__in=liked_ids)
        .values_list('name', flat=True)
    )
    top_actors = [a for a, _ in actor_counter.most_common(top_n_actors)]

    parts = []
    if top_genres:
        parts.append(f"喜欢的电影类型：{'、'.join(top_genres)}")
    if top_actors:
        parts.append(f"喜爱的演员：{'、'.join(top_actors)}")

    # 人口属性
    from myapp.models import UserInfo
    occ_dict = dict(UserInfo.OCCUPATION_CHOICES)
    if getattr(user, 'occupation', None) is not None:
        parts.append(f"职业：{occ_dict.get(user.occupation, '其他')}")
    if getattr(user, 'age', None):
        parts.append(f"年龄：{user.age}岁")

    # 近期高分电影
    recent_high = list(
        Movie.objects.filter(
            id__in=liked_ids,
            userrating__user=user,
            userrating__score__gte=8.0
        ).values_list('title', flat=True)[:5]
    )
    if recent_high:
        parts.append(f"近期高分电影：{'、'.join(recent_high)}")

    return "；".join(parts) if parts else "优质电影 高评分 经典"