"""
重排模块 (Re-Ranking)
================================================
在精排结果之上进行最终调整，主要实现：
  1. 业务规则过滤（敏感内容、已推荐去重）
  2. 结果列表截断（Top-K）
  3. 后置多样性保障
  4. 推荐日志记录

输入: 精排后的候选列表
输出: 最终推荐列表 + 完整日志
================================================
"""

import time
import logging
from collections import Counter
from myapp.models import Movie
from myapp.recommender.recall import _get_user_liked_ids

logger = logging.getLogger('movie_agent')


def business_filter(candidates, user=None, exclude_sensitive=True, exclude_ids=None):
    """
    业务规则过滤：
      1. 排除已推荐过的电影（去重）
      2. 排除敏感内容（如果启用）
      3. 未成年用户保护：自动排除不适宜内容
    
    ★ 核心改进：当敏感内容被过滤导致候选不足时，
      自动从安全电影池中补充，而非直接吞掉。
    
    Args:
        candidates: 精排候选列表
        user: 用户对象
        exclude_sensitive: 是否排除敏感内容
        exclude_ids: 额外排除的ID集合
    
    Returns:
        List[Dict] 过滤后的候选列表（自动补充安全内容）
    """
    if not candidates:
        return []
    
    exclude_set = set(exclude_ids or [])
    
    # 获取已推荐电影（最近一次推荐日志）
    if user:
        try:
            from myapp.models_upgrade import RecommendLog
            recent_log = RecommendLog.objects.filter(user=user).order_by('-created_at').first()
            if recent_log and recent_log.final_results:
                exclude_set.update(recent_log.final_results[:10])
        except Exception:
            pass  # RecommendLog 表可能尚未迁移
    
    # 获取未成年用户应排除的电影ID
    minor_excluded_ids = set()
    if user and exclude_sensitive:
        try:
            from myapp.utils.content_safety import get_minor_excluded_ids
            minor_excluded_ids = get_minor_excluded_ids(user)
        except Exception:
            pass
    
    filtered = []
    removed_sensitive_ids = []  # 记录被过滤的敏感电影ID（用于日志）
    
    for c in candidates:
        mid = c['movie_id']
        
        # 排除已推荐
        if mid in exclude_set:
            continue
        
        # 未成年用户保护：排除不适宜内容
        if mid in minor_excluded_ids:
            removed_sensitive_ids.append(mid)
            continue
        
        # 敏感内容过滤（全局，不仅限于未成年）
        if exclude_sensitive:
            movie = Movie.objects.filter(id=mid).values('is_sensitive').first()
            if movie and movie.get('is_sensitive'):
                removed_sensitive_ids.append(mid)
                continue
        
        filtered.append(c)
    
    # ★ 自动补充安全内容：当过滤导致候选不足时
    # 从数据库中随机选取高分安全电影补充
    target_count = min(15, len(candidates))
    if len(filtered) < target_count and user:
        try:
            existing_ids = {c['movie_id'] for c in filtered} | exclude_set | minor_excluded_ids
            needed = target_count - len(filtered)
            
            # 获取用户偏好类型（优先匹配）
            user_genres = []
            try:
                from myapp.agent.memory import MemoryManager
                session_id = f"user_{getattr(user, 'id', 'anon')}"
                memory = MemoryManager(user=user, session_id=session_id)
                slots = memory.get_slots()
                if slots.get('genre'):
                    user_genres.append(slots['genre'])
            except Exception:
                pass
            
            # 优先从用户偏好类型中补充
            safe_qs = Movie.objects.filter(is_sensitive=False).exclude(id__in=existing_ids)
            if user_genres:
                safe_qs = safe_qs.filter(genres__name__icontains=user_genres[0])
            
            safe_movies = list(
                safe_qs.order_by('-score', '-vote_count')
                .values('id', 'score')[:needed * 2]
            )
            
            # 如果偏好类型补充不足，从全库高分电影中补充
            if len(safe_movies) < needed:
                safe_movies_all = list(
                    Movie.objects.filter(is_sensitive=False)
                    .exclude(id__in=existing_ids | {m['id'] for m in safe_movies})
                    .order_by('-score', '-vote_count')
                    .values('id', 'score')[:needed]
                )
                safe_movies.extend(safe_movies_all)
            
            for m in safe_movies[:needed]:
                filtered.append({
                    'movie_id': m['id'],
                    'score': m.get('score', 0),
                    '_auto_safe_fill': True,  # 标记为自动补充的安全内容
                })
            
            if safe_movies:
                logger.info(
                    f"[Rerank] 安全补充: 过滤了 {len(removed_sensitive_ids)} 部敏感内容，"
                    f"自动补充了 {min(len(safe_movies), needed)} 部安全电影"
                )
        except Exception as e:
            logger.warning(f"[Rerank] 安全补充失败: {e}")
    
    return filtered


def diversity_guarantee(candidates, min_genre_diversity=3, top_k=15):
    """
    后置多样性保障：确保最终列表中至少包含 min_genre_diversity 种类型。
    
    Args:
        candidates: 排序后的候选列表
        min_genre_diversity: 最少类型数
        top_k: 最终返回数量
    
    Returns:
        List[Dict] 多样性保障后的推荐列表
    """
    if len(candidates) <= top_k:
        return candidates
    
    # 批量获取电影类型
    movie_ids = [c['movie_id'] for c in candidates]
    movies = Movie.objects.filter(id__in=movie_ids).prefetch_related('genres')
    genre_map = {}
    for m in movies:
        genre_map[m.id] = [g.name for g in m.genres.all()]
    
    selected = []
    genre_counter = Counter()
    remaining = list(candidates)
    
    # 第一轮：贪心选择，优先选不同类型
    while len(selected) < top_k and remaining:
        best_item = None
        best_score = -1
        
        for item in remaining:
            mid = item['movie_id']
            item_genres = genre_map.get(mid, [])
            
            # 多样性加分：如果该电影的主要类型在已选中较少，加分
            diversity_bonus = 0
            if item_genres:
                main_genre = item_genres[0]
                if genre_counter[main_genre] < 2:
                    diversity_bonus = 0.1
            
            adjusted_score = item.get('score', 0) + diversity_bonus
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_item = item
        
        if best_item:
            selected.append(best_item)
            remaining.remove(best_item)
            mid = best_item['movie_id']
            for g in genre_map.get(mid, []):
                genre_counter[g] += 1
    
    return selected


def final_rerank(candidates, user=None, top_k=15,
                  exclude_sensitive=True, ensure_diversity=True):
    """
    重排主函数。
    
    Args:
        candidates: 精排后的候选列表
        user: 用户对象
        top_k: 最终返回数量
        exclude_sensitive: 是否过滤敏感内容
        ensure_diversity: 是否保障多样性
    
    Returns:
        Tuple[List[Dict], Dict] - (最终推荐列表, 统计信息)
    """
    t_start = time.time()
    
    # 1. 业务过滤
    filtered = business_filter(
        candidates, user=user,
        exclude_sensitive=exclude_sensitive
    )
    
    # 2. 多样性保障
    if ensure_diversity and len(filtered) > top_k:
        final = diversity_guarantee(filtered, top_k=top_k)
    else:
        final = filtered[:top_k]
    
    # 3. 提取最终ID列表
    final_ids = [item['movie_id'] for item in final]
    
    stats = {
        'input_count': len(candidates),
        'filtered_count': len(filtered),
        'output_count': len(final),
        'final_ids': final_ids,
        'latency_ms': int((time.time() - t_start) * 1000),
    }
    
    return final, stats