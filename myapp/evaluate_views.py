"""
离线评估视图
================================================
论文实验数据接口：
  /evaluate/           - 评估总览页
  /evaluate/auc        - AUC指标接口
  /evaluate/ndcg       - NDCG指标接口
  /evaluate/hr         - HR@K指标接口
  /evaluate/mrr        - MRR指标接口
  /evaluate/full       - 完整评估（JSON）
================================================
"""

import json
from django.http import JsonResponse
from django.shortcuts import render
from django.contrib.auth.decorators import login_required, user_passes_test


def admin_required(function):
    """管理员权限装饰器"""
    return user_passes_test(
        lambda u: u.is_authenticated and u.is_staff,
        login_url='login_user'
    )(function)


# =============================================================
# 评估总览页
# =============================================================
@admin_required
def evaluate_index(request):
    """
    评估总览页 - 展示系统评估概览
    """
    from myapp.models_upgrade import RecommendLog, AgentTrace, UserFeedback
    from myapp.models import Rec, UserRating, UserInfo, Movie
    from django.db.models import Count, Avg
    
    # 统计数据
    stats = {
        'total_users': UserInfo.objects.filter(is_staff=False).count(),
        'total_movies': Movie.objects.count(),
        'total_ratings': UserRating.objects.count(),
        'total_recs': Rec.objects.count(),
        'total_traces': AgentTrace.objects.count() if hasattr(AgentTrace, 'objects') else 0,
        'total_feedbacks': UserFeedback.objects.count() if hasattr(UserFeedback, 'objects') else 0,
    }
    
    context = {
        'stats': stats,
        'page_title': '系统评估总览',
    }
    return render(request, 'evaluate_index.html', context)


# =============================================================
# AUC 接口
# =============================================================
@admin_required
def evaluate_auc(request):
    """
    AUC 评估接口
    GET 参数: sample_users (采样用户数, 默认50)
    """
    from myapp.recommender.evaluate import run_full_evaluation
    
    sample_users = int(request.GET.get('sample_users', 50))
    
    results = run_full_evaluation(sample_users=sample_users, k_values=[10])
    auc_data = results.get('auc', {'mean': 0, 'std': 0, 'count': 0})
    
    return JsonResponse({
        'metric': 'AUC',
        'value': auc_data.get('mean', 0),
        'std': auc_data.get('std', 0),
        'sample_count': auc_data.get('count', 0),
        'status': 'success',
    })


# =============================================================
# NDCG 接口
# =============================================================
@admin_required
def evaluate_ndcg(request):
    """
    NDCG@K 评估接口
    GET 参数: k (截断位置, 默认10), sample_users (采样用户数, 默认50)
    """
    from myapp.recommender.evaluate import run_full_evaluation
    
    k = int(request.GET.get('k', 10))
    sample_users = int(request.GET.get('sample_users', 50))
    
    results = run_full_evaluation(sample_users=sample_users, k_values=[k])
    ndcg_data = results.get(f'ndcg@{k}', {'mean': 0, 'std': 0, 'count': 0})
    
    return JsonResponse({
        'metric': f'NDCG@{k}',
        'value': ndcg_data.get('mean', 0),
        'std': ndcg_data.get('std', 0),
        'sample_count': ndcg_data.get('count', 0),
        'status': 'success',
    })


# =============================================================
# HR@K 接口
# =============================================================
@admin_required
def evaluate_hr(request):
    """
    HR@K 评估接口
    GET 参数: k (截断位置, 默认10), sample_users
    """
    from myapp.recommender.evaluate import run_full_evaluation
    
    k = int(request.GET.get('k', 10))
    sample_users = int(request.GET.get('sample_users', 50))
    
    results = run_full_evaluation(sample_users=sample_users, k_values=[k])
    hr_data = results.get(f'hr@{k}', {'mean': 0, 'std': 0, 'count': 0})
    
    return JsonResponse({
        'metric': f'HR@{k}',
        'value': hr_data.get('mean', 0),
        'std': hr_data.get('std', 0),
        'sample_count': hr_data.get('count', 0),
        'status': 'success',
    })


# =============================================================
# MRR 接口
# =============================================================
@admin_required
def evaluate_mrr(request):
    """
    MRR 评估接口
    GET 参数: sample_users
    """
    from myapp.recommender.evaluate import run_full_evaluation
    
    sample_users = int(request.GET.get('sample_users', 50))
    
    results = run_full_evaluation(sample_users=sample_users, k_values=[10])
    mrr_data = results.get('mrr', {'mean': 0, 'std': 0, 'count': 0})
    
    return JsonResponse({
        'metric': 'MRR',
        'value': mrr_data.get('mean', 0),
        'std': mrr_data.get('std', 0),
        'sample_count': mrr_data.get('count', 0),
        'status': 'success',
    })


# =============================================================
# 完整评估接口
# =============================================================
@admin_required
def evaluate_full(request):
    """
    完整评估接口 - 一次运行所有指标
    GET 参数: sample_users (默认50)
    
    返回所有论文所需指标的 JSON
    """
    from myapp.recommender.evaluate import run_full_evaluation
    
    sample_users = int(request.GET.get('sample_users', 50))
    
    results = run_full_evaluation(
        sample_users=sample_users,
        k_values=[5, 10, 15, 20]
    )
    
    return JsonResponse({
        'status': 'success',
        'results': results,
    })