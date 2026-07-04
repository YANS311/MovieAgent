"""
离线评估模块 (Offline Evaluation)
================================================
提供推荐系统离线评估指标，支持论文实验数据采集。

支持指标:
  - AUC  (Area Under ROC Curve)   - 排序质量
  - NDCG (Normalized Discounted Cumulative Gain) - 排序精度
  - HR@K (Hit Rate at K)          - 命中率
  - MRR  (Mean Reciprocal Rank)   - 平均倒数排名
  - Precision@K / Recall@K        - 精确率/召回率
  - Coverage                      - 覆盖率
  - Diversity                     - 多样性
  - Novelty                       - 新颖性

论文实验接口:
  /evaluate/auc
  /evaluate/ndcg
  /evaluate/hr
  /evaluate/mrr
================================================
"""

import time
import math
import random
import numpy as np
from collections import defaultdict
from django.db.models import Count, Avg
from myapp.models import Movie, UserRating, Rec, UserInfo


# =============================================================
# 基础指标计算
# =============================================================

def calc_auc(y_true, y_score):
    """
    计算 AUC (Area Under ROC Curve)
    使用 Mann-Whitney U 统计量的等价形式。
    
    Args:
        y_true: 真实标签列表 (0/1)
        y_score: 预测分数列表
    
    Returns:
        float: AUC 值
    """
    pos_scores = [s for t, s in zip(y_true, y_score) if t == 1]
    neg_scores = [s for t, s in zip(y_true, y_score) if t == 0]
    
    if not pos_scores or not neg_scores:
        return 0.5
    
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    
    concordant = 0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                concordant += 1
            elif p == n:
                concordant += 0.5
    
    return concordant / (n_pos * n_neg)


def calc_ndcg(relevance_list, k=10):
    """
    计算 NDCG@K (Normalized Discounted Cumulative Gain)
    
    Args:
        relevance_list: 按推荐顺序排列的相关性列表 (0或1)
        k: 截断位置
    
    Returns:
        float: NDCG@K 值
    """
    # DCG
    dcg = 0.0
    for i, rel in enumerate(relevance_list[:k]):
        dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
    
    # IDCG (理想排序)
    ideal = sorted(relevance_list[:k], reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal):
        idcg += rel / math.log2(i + 2)
    
    return dcg / idcg if idcg > 0 else 0.0


def calc_hr_at_k(recommended_ids, ground_truth_ids, k=10):
    """
    计算 HR@K (Hit Rate at K)
    只要 Top-K 中有一部命中就算一次 hit。
    
    Args:
        recommended_ids: 推荐列表 (已排序)
        ground_truth_ids: 用户真正喜欢的电影ID集合
        k: 截断位置
    
    Returns:
        float: 0 或 1
    """
    rec_set = set(recommended_ids[:k])
    gt_set = set(ground_truth_ids)
    return 1.0 if rec_set & gt_set else 0.0


def calc_mrr(recommended_ids, ground_truth_ids):
    """
    计算 MRR (Mean Reciprocal Rank)
    第一部命中电影的倒数排名。
    
    Args:
        recommended_ids: 推荐列表 (已排序)
        ground_truth_ids: 用户真正喜欢的电影ID集合
    
    Returns:
        float: 1/rank 或 0
    """
    gt_set = set(ground_truth_ids)
    for i, mid in enumerate(recommended_ids):
        if mid in gt_set:
            return 1.0 / (i + 1)
    return 0.0


def calc_precision_recall_at_k(recommended_ids, ground_truth_ids, k=10):
    """
    计算 Precision@K 和 Recall@K
    
    Returns:
        Tuple[float, float] - (precision, recall)
    """
    rec_top_k = set(recommended_ids[:k])
    gt_set = set(ground_truth_ids)
    
    hits = len(rec_top_k & gt_set)
    precision = hits / k if k > 0 else 0.0
    recall = hits / len(gt_set) if gt_set else 0.0
    
    return precision, recall


# =============================================================
# 系统级指标
# =============================================================

def calc_coverage(all_recommendations, total_movie_count):
    """
    计算推荐覆盖率 (Coverage)
    所有用户被推荐过的电影占总电影库的比例。
    
    Args:
        all_recommendations: List[List[int]] - 每个用户的推荐列表
        total_movie_count: 电影库总电影数
    
    Returns:
        float: 覆盖率 0-1
    """
    unique_recommended = set()
    for rec_list in all_recommendations:
        unique_recommended.update(rec_list)
    return len(unique_recommended) / total_movie_count if total_movie_count > 0 else 0.0


def calc_diversity(recommended_ids, movie_genre_map):
    """
    计算推荐多样性 (Intra-List Diversity)
    推荐列表中电影对之间的平均类型距离 (1 - Jaccard相似度)。
    
    Args:
        recommended_ids: 推荐列表
        movie_genre_map: Dict[int, Set[str]] - {movie_id: set(genres)}
    
    Returns:
        float: 多样性 0-1
    """
    pairs = 0
    total_distance = 0.0
    n = len(recommended_ids)
    
    for i in range(n):
        for j in range(i + 1, n):
            g_i = movie_genre_map.get(recommended_ids[i], set())
            g_j = movie_genre_map.get(recommended_ids[j], set())
            if g_i or g_j:
                jaccard = len(g_i & g_j) / len(g_i | g_j) if len(g_i | g_j) > 0 else 1.0
                total_distance += (1 - jaccard)
                pairs += 1
    
    return total_distance / pairs if pairs > 0 else 0.0


def calc_novelty(recommended_ids, movie_popularity_map):
    """
    计算推荐新颖性 (Novelty)
    推荐电影的平均"反流行度" (-log2(popularity))。
    
    Args:
        recommended_ids: 推荐列表
        movie_popularity_map: Dict[int, float] - {movie_id: popularity_ratio}
    
    Returns:
        float: 新颖性分数
    """
    if not recommended_ids:
        return 0.0
    
    novelty_sum = 0.0
    count = 0
    for mid in recommended_ids:
        pop = movie_popularity_map.get(mid, 0.001)
        novelty_sum += -math.log2(max(pop, 0.0001))
        count += 1
    
    return novelty_sum / count if count > 0 else 0.0


# =============================================================
# 完整评估流程
# =============================================================

def run_full_evaluation(sample_users=50, k_values=None):
    """
    执行完整的离线评估，输出论文所需的全部指标。
    
    Args:
        sample_users: 采样用户数量
        k_values: 评估的K值列表
    
    Returns:
        Dict: 包含所有评估指标的字典
    """
    if k_values is None:
        k_values = [5, 10, 15, 20]
    
    t_start = time.time()
    
    # 获取有足够评分的用户
    active_users = list(
        UserInfo.objects.annotate(
            rating_count=Count('userrating')
        ).filter(
            rating_count__gte=10,
            is_staff=False
        ).values_list('id', flat=True)[:sample_users]
    )
    
    if not active_users:
        return {'error': '没有足够的活跃用户进行评估'}
    
    # 电影流行度（用于计算新颖性）
    total_ratings = UserRating.objects.count() or 1
    popularity_map = {}
    pop_data = Movie.objects.annotate(
        r_count=Count('userrating')
    ).values_list('id', 'r_count')
    for mid, rc in pop_data:
        popularity_map[mid] = rc / total_ratings
    
    # 电影类型映射（用于计算多样性）
    genre_map = {}
    for movie in Movie.objects.prefetch_related('genres').all():
        genre_map[movie.id] = set(g.name for g in movie.genres.all())
    
    total_movies = Movie.objects.count()
    
    # ── 逐用户评估 ──
    metrics = defaultdict(list)
    all_rec_ids = []
    
    for user_id in active_users:
        # 获取用户的正样本（高分电影）
        user_positive = set(
            UserRating.objects.filter(
                user_id=user_id, score__gte=7.5
            ).values_list('movie_id', flat=True)
        )
        
        if len(user_positive) < 3:
            continue
        
        # 获取推荐列表
        rec_ids = list(
            Rec.objects.filter(user_id=user_id)
            .order_by('-rating')
            .values_list('movie_id', flat=True)[:max(k_values)]
        )
        
        if not rec_ids:
            continue
        
        all_rec_ids.append(rec_ids)
        
        # 二元相关性
        y_true = [1 if mid in user_positive else 0 for mid in rec_ids]
        y_score = [1.0] * len(rec_ids)  # 简化：使用排名作为分数
        
        # AUC
        if len(set(y_true)) > 1:
            auc = calc_auc(y_true, list(range(len(rec_ids), 0, -1)))
            metrics['auc'].append(auc)
        
        # 各K值指标
        for k in k_values:
            # NDCG@K
            ndcg = calc_ndcg(y_true, k=k)
            metrics[f'ndcg@{k}'].append(ndcg)
            
            # HR@K
            hr = calc_hr_at_k(rec_ids, user_positive, k=k)
            metrics[f'hr@{k}'].append(hr)
            
            # Precision@K, Recall@K
            prec, rec = calc_precision_recall_at_k(rec_ids, user_positive, k=k)
            metrics[f'precision@{k}'].append(prec)
            metrics[f'recall@{k}'].append(rec)
        
        # MRR
        mrr = calc_mrr(rec_ids, user_positive)
        metrics['mrr'].append(mrr)
    
    # ── 汇总 ──
    results = {}
    for metric_name, values in metrics.items():
        if values:
            results[metric_name] = {
                'mean': round(np.mean(values), 4),
                'std': round(np.std(values), 4),
                'count': len(values),
            }
    
    # 系统级指标
    results['coverage'] = calc_coverage(all_rec_ids, total_movies)
    
    # 多样性和新颖性（取所有用户的平均）
    diversity_scores = []
    novelty_scores = []
    for rec_ids in all_rec_ids:
        diversity_scores.append(calc_diversity(rec_ids[:10], genre_map))
        novelty_scores.append(calc_novelty(rec_ids[:10], popularity_map))
    
    if diversity_scores:
        results['diversity'] = {
            'mean': round(np.mean(diversity_scores), 4),
            'std': round(np.std(diversity_scores), 4),
        }
    if novelty_scores:
        results['novelty'] = {
            'mean': round(np.mean(novelty_scores), 4),
            'std': round(np.std(novelty_scores), 4),
        }
    
    results['evaluation_meta'] = {
        'total_users_evaluated': len(active_users),
        'total_movies': total_movies,
        'k_values': k_values,
        'elapsed_seconds': round(time.time() - t_start, 2),
    }
    
    return results


# =============================================================
# 单用户快速评估
# =============================================================

def evaluate_single_user(user, k=10):
    """
    单用户推荐质量评估。
    
    Returns:
        Dict: 该用户的评估指标
    """
    # 留一法：取最后一部高分电影作为测试集
    user_ratings = list(
        UserRating.objects.filter(user=user, score__gte=7.0)
        .order_by('comment_time')
        .values_list('movie_id', flat=True)
    )
    
    if len(user_ratings) < 5:
        return {'error': '用户评分数据不足'}
    
    test_item = user_ratings[-1]  # 最后一部作为测试
    ground_truth = {test_item}
    
    # 获取推荐列表
    rec_ids = list(
        Rec.objects.filter(user=user)
        .order_by('-rating')
        .values_list('movie_id', flat=True)[:k]
    )
    
    if not rec_ids:
        return {'error': '无推荐结果'}
    
    y_true = [1 if mid in ground_truth else 0 for mid in rec_ids]
    
    return {
        'hr@k': calc_hr_at_k(rec_ids, ground_truth, k=k),
        'mrr': calc_mrr(rec_ids, ground_truth),
        'ndcg@k': calc_ndcg(y_true, k=k),
        'test_item': test_item,
        'rec_list_length': len(rec_ids),
    }