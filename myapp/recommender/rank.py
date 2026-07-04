"""
精排模块 (Precision Ranking)
================================================
对召回阶段的候选电影进行精细化排序。
实现多种精排策略：
  1. WeightedScoreRank   - 加权评分排序
  2. DeepModelRank       - 深度模型打分（SKB-FMLP/DeepFM）
  3. DiversityAwareRank  - 多样性感知排序（MMR变体）

输入: 召回候选列表 + 用户特征
输出: 精排后的有序列表
================================================
"""

import time
import numpy as np
from collections import defaultdict
from myapp.models import Movie, UserRating, Genre, Actor


# =============================================================
# 策略 1: 加权评分精排 (Weighted Score Ranking)
# =============================================================
def weighted_score_rank(candidates, user=None, weights=None):
    """
    综合多维度特征的加权评分排序。
    
    特征维度：
      - 预测分 (recall score)
      - 电影质量分 (IMDb Bayesian average)
      - 用户偏好匹配度
      - 时效性衰减
    
    Args:
        candidates: List[Dict] 召回结果
        user: 用户对象
        weights: 各维度权重 dict
    
    Returns:
        List[Dict] 精排后的候选列表
    """
    if weights is None:
        weights = {
            'recall_score': 0.4,    # 召回阶段的分数
            'quality': 0.3,         # 电影质量
            'preference': 0.2,      # 偏好匹配
            'freshness': 0.1,       # 时效性
        }
    
    if not candidates:
        return []
    
    # 批量获取电影信息
    movie_ids = [c['movie_id'] for c in candidates]
    movies = Movie.objects.filter(id__in=movie_ids).prefetch_related('genres', 'actors', 'directors')
    movie_map = {m.id: m for m in movies}
    
    # 计算全局平均分（贝叶斯平均）
    global_avg = Movie.objects.aggregate(avg=models.Avg('score'))['avg'] or 7.0
    C = 1000  # 最小投票阈值
    
    # 获取用户偏好（如果有）
    pref_genres = set()
    pref_actors = set()
    if user:
        liked_ids = list(
            UserRating.objects.filter(user=user, score__gte=7.0)
            .values_list('movie_id', flat=True)[:50]
        )
        pref_genres = set(
            Movie.genres.through.objects.filter(movie_id__in=liked_ids)
            .values_list('genre__name', flat=True)
        )
        pref_actors = set(
            Actor.objects.filter(movie__id__in=liked_ids)
            .values_list('name', flat=True)
        )
    
    # 精排打分
    scored = []
    for c in candidates:
        mid = c['movie_id']
        movie = movie_map.get(mid)
        if not movie:
            continue
        
        # 1. 召回分
        recall_s = c.get('score', 0.0)
        
        # 2. 质量分（贝叶斯平均）
        vc = movie.vote_count or 0
        ms = float(movie.score) if movie.score else 0.0
        quality_s = (vc / (vc + C)) * ms + (C / (vc + C)) * global_avg
        quality_s = quality_s / 10.0  # 归一化到0-1
        
        # 3. 偏好匹配分
        pref_s = 0.0
        if pref_genres or pref_actors:
            movie_genres = set(g.name for g in movie.genres.all())
            movie_actors = set(a.name for a in movie.actors.all())
            genre_overlap = len(movie_genres & pref_genres) / max(len(movie_genres), 1)
            actor_overlap = len(movie_actors & pref_actors) / max(len(movie_actors), 1)
            pref_s = genre_overlap * 0.6 + actor_overlap * 0.4
        
        # 4. 时效性
        freshness_s = 0.5
        if movie.date:
            year = movie.date.year
            freshness_s = min(1.0, max(0.0, (year - 2000) / 26.0))
        
        # 加权求和
        final_score = (
            weights['recall_score'] * recall_s +
            weights['quality'] * quality_s +
            weights['preference'] * pref_s +
            weights['freshness'] * freshness_s
        )
        
        scored.append({
            'movie_id': mid,
            'score': final_score,
            'source': c.get('source', ''),
            'detail_scores': {
                'recall': round(recall_s, 4),
                'quality': round(quality_s, 4),
                'preference': round(pref_s, 4),
                'freshness': round(freshness_s, 4),
            }
        })
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    return scored


# =============================================================
# 策略 2: 深度模型精排 (Deep Model Ranking)
# =============================================================
def deep_model_rank(candidates, user, model_cache=None, top_k=30):
    """
    使用 SKB-FMLP / DeepFM 模型进行精排打分。
    
    Args:
        candidates: 召回候选列表
        user: 用户对象
        model_cache: 全局模型缓存字典 {'model': ..., 'meta': ...}
        top_k: 返回数量
    
    Returns:
        List[Dict] 精排结果
    """
    if not model_cache or not model_cache.get('model'):
        # 模型未加载，降级到加权评分
        return weighted_score_rank(candidates, user)[:top_k]
    
    try:
        model = model_cache['model']
        meta = model_cache['meta']
        
        lbe_user = meta['lbe_user']
        lbe_movie = meta['lbe_movie']
        feature_store = meta['feature_store']
        SEQ_LEN = meta['SEQ_LEN']
        DIM = meta['UNIFIED_EMBED_DIM']
        
        # 构建用户序列
        user_history = list(
            UserRating.objects.filter(user=user)
            .order_by('comment_time')
            .values_list('movie_id', flat=True)
        )
        hist_enc = [lbe_movie.transform([str(m)])[0] + 1 for m in user_history if str(m) in lbe_movie.classes_]
        
        if len(hist_enc) == 0:
            hist_padded = np.zeros(SEQ_LEN, dtype=np.int32)
        else:
            hist_padded = np.pad(
                hist_enc[-SEQ_LEN:],
                (0, max(0, SEQ_LEN - len(hist_enc))),
                'constant'
            ) if len(hist_enc) < SEQ_LEN else np.array(hist_enc[-SEQ_LEN:], dtype=np.int32)
        
        # 用户编码
        u_str = str(user.id)
        u_idx = lbe_user.transform([u_str])[0] + 1 if u_str in lbe_user.classes_ else 0
        
        # 构建候选集特征
        movie_ids = [c['movie_id'] for c in candidates]
        enc_ids = []
        valid_candidates = []
        
        for c in candidates:
            mid_str = str(c['movie_id'])
            if mid_str in lbe_movie.classes_:
                enc_ids.append(lbe_movie.transform([mid_str])[0] + 1)
                valid_candidates.append(c)
        
        if not enc_ids:
            return weighted_score_rank(candidates, user)[:top_k]
        
        N = len(enc_ids)
        import torch
        
        infer_input = {
            'user_id': np.full(N, u_idx, dtype=np.int32),
            'movie_id': np.array(enc_ids, dtype=np.int32),
            'hist_movie_id': np.tile(hist_padded, (N, 1)),
            'sl': np.full(N, min(len(hist_enc), SEQ_LEN), dtype=np.int32)
        }
        
        # 图谱特征
        valid_mask = np.array([str(c['movie_id']) in lbe_movie.classes_ for c in candidates])
        if 'genres_matrix' in feature_store:
            full_genres = feature_store['genres_matrix']
            full_actors = feature_store['actors_matrix']
            infer_input['genres'] = full_genres[valid_mask][:N]
            infer_input['actors'] = full_actors[valid_mask][:N]
        if 'directors_matrix' in feature_store:
            infer_input['directors'] = feature_store['directors_matrix'][valid_mask][:N]
        
        rag_b = feature_store['rag_matrix'][valid_mask][:N]
        for i in range(DIM):
            infer_input[f'rag_{i}'] = rag_b[:, i]
        
        with torch.no_grad():
            preds = model.predict(infer_input, batch_size=512).flatten()
        
        for i, c in enumerate(valid_candidates):
            c['score'] = float(preds[i])
            c['source'] = 'deep_model'
        
        valid_candidates.sort(key=lambda x: x['score'], reverse=True)
        return valid_candidates[:top_k]
    
    except Exception as e:
        print(f"[Rank/DeepModel] 异常，降级到加权排序: {e}")
        return weighted_score_rank(candidates, user)[:top_k]


# =============================================================
# 策略 3: MMR 多样性排序 (Maximal Marginal Relevance)
# =============================================================
def mmr_diversity_rank(candidates, alpha=0.7, top_k=15):
    """
    基于 MMR (Maximal Marginal Relevance) 的多样性排序。
    平衡相关性与结果多样性，避免"信息茧房"。
    
    Args:
        candidates: 精排候选列表（需包含 score 字段）
        alpha: 相关性权重（0.0=纯多样性, 1.0=纯相关性）
        top_k: 返回数量
    
    Returns:
        List[Dict] 多样性排序后的结果
    """
    if len(candidates) <= top_k:
        return candidates
    
    selected = []
    remaining = list(range(len(candidates)))
    
    # 第一个：选最高分
    best_idx = max(remaining, key=lambda i: candidates[i].get('score', 0))
    selected.append(best_idx)
    remaining.remove(best_idx)
    
    while len(selected) < top_k and remaining:
        best_score = -float('inf')
        best_idx = -1
        
        for i in remaining:
            relevance = candidates[i].get('score', 0)
            
            # 与已选集合的最大相似度（使用 source 多样性作为代理）
            max_sim = 0.0
            for j in selected:
                # 同来源扣分
                if candidates[i].get('source') == candidates[j].get('source'):
                    sim = 0.5
                else:
                    sim = 0.1
                max_sim = max(max_sim, sim)
            
            mmr_score = alpha * relevance - (1 - alpha) * max_sim
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i
        
        if best_idx >= 0:
            selected.append(best_idx)
            remaining.remove(best_idx)
    
    return [candidates[i] for i in selected]


# =============================================================
# 主入口：精排
# =============================================================
def precision_rank(candidates, user=None, strategy='weighted',
                    model_cache=None, top_k=30, alpha=0.7):
    """
    精排主函数。
    
    Args:
        candidates: 召回阶段的候选列表
        user: 用户对象
        strategy: 'weighted' | 'deep' | 'mmr'
        model_cache: 深度模型缓存
        top_k: 返回数量
        alpha: MMR 参数
    
    Returns:
        Tuple[List[Dict], Dict] - (精排结果, 统计信息)
    """
    t_start = time.time()
    
    if not candidates:
        return [], {'strategy': strategy, 'count': 0, 'latency_ms': 0}
    
    if strategy == 'deep':
        results = deep_model_rank(candidates, user, model_cache, top_k)
    elif strategy == 'mmr':
        weighted = weighted_score_rank(candidates, user)
        results = mmr_diversity_rank(weighted, alpha=alpha, top_k=top_k)
    else:
        results = weighted_score_rank(candidates, user)[:top_k]
    
    stats = {
        'strategy': strategy,
        'input_count': len(candidates),
        'output_count': len(results),
        'latency_ms': int((time.time() - t_start) * 1000),
    }
    
    return results, stats


# 需要导入 models
from django.db import models as django_models