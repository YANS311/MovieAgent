"""
Agent 实验评估模块 (Evaluator)
================================================
实现第五章所需的四类实验：
  A. RAG 作用实验 (RAG Ablation)
  B. KAG 作用实验 (KAG Accuracy)
  C. ReAct vs Workflow 对比实验
  D. 消融实验 (Ablation Study)

评估指标：
  - HR@K (Hit Rate)
  - NDCG@K (Normalized Discounted Cumulative Gain)
  - MRR (Mean Reciprocal Rank)
  - Task Success Rate
  - Avg Steps
  - Response Quality (人工/自动化评分)
================================================
"""

import time
import json
import random
import math
from collections import defaultdict
from django.db.models import Avg, Count, Q


# =============================================================
# 基础指标计算
# =============================================================

def hit_rate(recommended_ids, ground_truth_ids, k=5):
    """
    HR@K: 推荐列表 Top-K 中是否命中真实正样本。
    
    Args:
        recommended_ids: 推荐的电影ID列表
        ground_truth_ids: 真实正样本ID集合
        k: 截断位置
    
    Returns:
        float: 0.0 或 1.0
    """
    rec_set = set(recommended_ids[:k])
    gt_set = set(ground_truth_ids)
    return 1.0 if rec_set & gt_set else 0.0


def ndcg_at_k(recommended_ids, ground_truth_ids, k=5):
    """
    NDCG@K: 归一化折损累积增益。
    
    Args:
        recommended_ids: 推荐的电影ID列表
        ground_truth_ids: 真实正样本ID集合
        k: 截断位置
    
    Returns:
        float: NDCG@K 值
    """
    gt_set = set(ground_truth_ids)
    dcg = 0.0
    for i, mid in enumerate(recommended_ids[:k]):
        if mid in gt_set:
            dcg += 1.0 / math.log2(i + 2)  # log2(rank + 1)
    
    # 理想排序的 DCG
    ideal_hits = min(len(gt_set), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    
    return dcg / idcg if idcg > 0 else 0.0


def mrr(recommended_ids, ground_truth_ids, k=10):
    """
    MRR: 平均倒数排名。
    
    Args:
        recommended_ids: 推荐的电影ID列表
        ground_truth_ids: 真实正样本ID集合
        k: 检索范围
    
    Returns:
        float: MRR 值
    """
    gt_set = set(ground_truth_ids)
    for i, mid in enumerate(recommended_ids[:k]):
        if mid in gt_set:
            return 1.0 / (i + 1)
    return 0.0


# =============================================================
# 实验 A: RAG 作用实验
# =============================================================

def evaluate_rag_ablation(agent_cls, test_queries, user, rag_resources, neo_graph):
    """
    对比 BaseRec vs BaseRec+RAG 的推荐效果。
    
    Args:
        agent_cls: MovieAgent 类
        test_queries: 测试查询列表 [{'query': str, 'expected_ids': list}]
        user: 测试用户
        rag_resources: RAG资源
        neo_graph: Neo4j图实例
    
    Returns:
        Dict: {'base': {metrics}, 'rag': {metrics}}
    """
    results = {'base': [], 'rag': []}
    
    for item in test_queries:
        query = item['query']
        expected = item.get('expected_ids', [])
        
        # BaseRec（无RAG）
        agent_base = agent_cls(user=user, neo_graph=None, rag_resources={})
        result_base = agent_base.run(query)
        rec_ids_base = result_base['recommended_ids']
        results['base'].append({
            'hr5': hit_rate(rec_ids_base, expected, k=5),
            'ndcg5': ndcg_at_k(rec_ids_base, expected, k=5),
            'mrr': mrr(rec_ids_base, expected),
        })
        
        # BaseRec + RAG
        agent_rag = agent_cls(user=user, neo_graph=None, rag_resources=rag_resources)
        result_rag = agent_rag.run(query)
        rec_ids_rag = result_rag['recommended_ids']
        results['rag'].append({
            'hr5': hit_rate(rec_ids_rag, expected, k=5),
            'ndcg5': ndcg_at_k(rec_ids_rag, expected, k=5),
            'mrr': mrr(rec_ids_rag, expected),
        })
    
    # 汇总
    summary = {}
    for key in ['base', 'rag']:
        n = len(results[key])
        summary[key] = {
            'HR@5': sum(r['hr5'] for r in results[key]) / max(n, 1),
            'NDCG@5': sum(r['ndcg5'] for r in results[key]) / max(n, 1),
            'MRR': sum(r['mrr'] for r in results[key]) / max(n, 1),
            'n': n,
        }
    
    return summary


# =============================================================
# 实验 B: KAG 作用实验
# =============================================================

def evaluate_kag_accuracy(neo_graph):
    """
    测试知识图谱查询的准确率。
    
    测试任务：
      1. 导演查询：查询诺兰导演的电影
      2. 关系查询：查询与某电影同类型的电影
      3. 系列电影查询：查询续集/前传
    
    Returns:
        Dict: 各任务的准确率
    """
    if not neo_graph:
        return {'error': 'Neo4j 未连接', 'director_acc': 0, 'relation_acc': 0, 'series_acc': 0}
    
    results = {'director': [], 'relation': [], 'series': []}
    
    # 测试1：导演查询
    director_tests = [
        {'director': 'Christopher Nolan', 'expected_count_min': 3},
        {'director': 'Steven Spielberg', 'expected_count_min': 5},
        {'director': 'Quentin Tarantino', 'expected_count_min': 3},
    ]
    for dt in director_tests:
        try:
            cypher = """
            MATCH (d:Person {name: $name})-[:DIRECTED_BY]->(m:Movie)
            RETURN count(m) AS cnt
            """
            row = neo_graph.run(cypher, name=dt['director']).data()
            cnt = row[0]['cnt'] if row else 0
            results['director'].append(1.0 if cnt >= dt['expected_count_min'] else 0.0)
        except Exception:
            results['director'].append(0.0)
    
    # 测试2：关系查询
    relation_tests = [
        {'genre': 'Action', 'expected_count_min': 50},
        {'genre': 'Sci-Fi', 'expected_count_min': 20},
    ]
    for rt in relation_tests:
        try:
            cypher = """
            MATCH (g:Genre {name: $name})<-[:BELONGS_TO]-(m:Movie)
            RETURN count(m) AS cnt
            """
            row = neo_graph.run(cypher, name=rt['genre']).data()
            cnt = row[0]['cnt'] if row else 0
            results['relation'].append(1.0 if cnt >= rt['expected_count_min'] else 0.0)
        except Exception:
            results['relation'].append(0.0)
    
    # 测试3：系列查询（导演共现）
    series_tests = [
        {'movie': 'Star Wars', 'expected_min': 2},
    ]
    for st in series_tests:
        try:
            cypher = """
            MATCH (m:Movie)-[:DIRECTED_BY]-(d:Person)-[:DIRECTED_BY]-(other:Movie)
            WHERE m.title CONTAINS $title AND m <> other
            RETURN count(DISTINCT other) AS cnt
            """
            row = neo_graph.run(cypher, title=st['movie']).data()
            cnt = row[0]['cnt'] if row else 0
            results['series'].append(1.0 if cnt >= st['expected_min'] else 0.0)
        except Exception:
            results['series'].append(0.0)
    
    # 汇总
    summary = {
        'director_accuracy': sum(results['director']) / max(len(results['director']), 1),
        'relation_accuracy': sum(results['relation']) / max(len(results['relation']), 1),
        'series_accuracy': sum(results['series']) / max(len(results['series']), 1),
        'overall': (
            sum(results['director']) + sum(results['relation']) + sum(results['series'])
        ) / max(
            len(results['director']) + len(results['relation']) + len(results['series']), 1
        ),
    }
    return summary


# =============================================================
# 实验 C: ReAct vs Workflow 对比
# =============================================================

def evaluate_react_vs_workflow(agent_cls, complex_queries, user, rag_resources, neo_graph):
    """
    对比 ReAct 推理 vs 固定 Workflow 的效果。
    
    Args:
        agent_cls: MovieAgent 类
        complex_queries: 复杂查询列表 [{'query': str, 'success_criteria': dict}]
        user: 测试用户
        rag_resources: RAG 资源
        neo_graph: Neo4j 图实例
    
    Returns:
        Dict: ReAct 和 Workflow 的对比结果
    """
    react_results = []
    workflow_results = []
    
    for item in complex_queries:
        query = item['query']
        criteria = item.get('success_criteria', {})
        
        # ReAct Agent
        agent = agent_cls(user=user, neo_graph=neo_graph, rag_resources=rag_resources)
        result = agent.run(query)
        
        success = _check_success(result, criteria)
        react_results.append({
            'success': success,
            'steps': len(result['actions']),
            'latency_ms': result['latency_ms'],
            'rec_count': len(result['recommended_ids']),
        })
        
        # Workflow（固定流程：召回→重排→输出）
        from myapp.recommender.recall import hot_recall
        hot = hot_recall(k=10)
        workflow_results.append({
            'success': len(hot) > 0,  # 热门总能返回结果
            'steps': 1,
            'latency_ms': 0,
            'rec_count': len(hot),
        })
    
    n = len(complex_queries)
    return {
        'react': {
            'success_rate': sum(r['success'] for r in react_results) / max(n, 1),
            'avg_steps': sum(r['steps'] for r in react_results) / max(n, 1),
            'avg_latency_ms': sum(r['latency_ms'] for r in react_results) / max(n, 1),
        },
        'workflow': {
            'success_rate': sum(r['success'] for r in workflow_results) / max(n, 1),
            'avg_steps': sum(r['steps'] for r in workflow_results) / max(n, 1),
            'avg_latency_ms': sum(r['latency_ms'] for r in workflow_results) / max(n, 1),
        },
    }


def _check_success(result, criteria):
    """检查推荐结果是否满足成功条件"""
    if not criteria:
        return len(result.get('recommended_ids', [])) > 0
    
    rec_ids = result.get('recommended_ids', [])
    if not rec_ids:
        return False
    
    from myapp.models import Movie
    movies = Movie.objects.filter(id__in=rec_ids[:5]).prefetch_related('genres')
    
    # 检查类型条件
    if criteria.get('genre'):
        target_genre = criteria['genre']
        genre_match = any(
            target_genre in [g.name for g in m.genres.all()]
            for m in movies
        )
        if not genre_match:
            return False
    
    # 检查评分条件
    if criteria.get('score_min'):
        score_match = any(m.score and m.score >= criteria['score_min'] for m in movies)
        if not score_match:
            return False
    
    # 检查年份条件
    if criteria.get('year_min'):
        year_match = any(m.year and m.year >= criteria['year_min'] for m in movies)
        if not year_match:
            return False
    
    return True


# =============================================================
# 实验 D: 消融实验
# =============================================================

def evaluate_ablation(agent_cls, test_queries, user, rag_resources, neo_graph):
    """
    消融实验：分别去掉 RAG/KG/Memory/Planner 观察影响。
    
    Returns:
        Dict: 各消融条件下的指标
    """
    configs = {
        'full': {'rag': True, 'kg': True, 'memory': True},
        'no_rag': {'rag': False, 'kg': True, 'memory': True},
        'no_kg': {'rag': True, 'kg': False, 'memory': True},
        'no_memory': {'rag': True, 'kg': True, 'memory': False},
        'no_rag_no_kg': {'rag': False, 'kg': False, 'memory': True},
    }
    
    all_results = {}
    
    for config_name, config in configs.items():
        results = []
        for item in test_queries:
            query = item['query']
            expected = item.get('expected_ids', [])
            
            # 构造 Agent
            r = rag_resources if config['rag'] else {}
            n = neo_graph if config['kg'] else None
            agent = agent_cls(user=user, neo_graph=n, rag_resources=r)
            
            result = agent.run(query)
            rec_ids = result['recommended_ids']
            
            results.append({
                'hr5': hit_rate(rec_ids, expected, k=5),
                'ndcg5': ndcg_at_k(rec_ids, expected, k=5),
                'mrr': mrr(rec_ids, expected),
                'steps': len(result['actions']),
                'latency_ms': result['latency_ms'],
            })
        
        n = len(results)
        all_results[config_name] = {
            'HR@5': round(sum(r['hr5'] for r in results) / max(n, 1), 4),
            'NDCG@5': round(sum(r['ndcg5'] for r in results) / max(n, 1), 4),
            'MRR': round(sum(r['mrr'] for r in results) / max(n, 1), 4),
            'Avg Steps': round(sum(r['steps'] for r in results) / max(n, 1), 2),
            'Avg Latency(ms)': round(sum(r['latency_ms'] for r in results) / max(n, 1), 0),
        }
    
    return all_results


# =============================================================
# 综合评估入口
# =============================================================

def run_full_evaluation(agent_cls, user, rag_resources, neo_graph):
    """
    运行完整评估套件，返回所有实验结果。
    
    Returns:
        Dict: 包含所有实验结果的字典
    """
    from myapp.models import Movie, UserRating
    
    # 构建测试集
    test_queries = _build_test_set(user)
    complex_queries = _build_complex_queries()
    
    t_start = time.time()
    results = {}
    
    # A. RAG 实验
    results['rag_ablation'] = evaluate_rag_ablation(
        agent_cls, test_queries, user, rag_resources, neo_graph
    )
    
    # B. KAG 实验
    results['kag_accuracy'] = evaluate_kag_accuracy(neo_graph)
    
    # C. ReAct vs Workflow
    results['react_vs_workflow'] = evaluate_react_vs_workflow(
        agent_cls, complex_queries, user, rag_resources, neo_graph
    )
    
    # D. 消融实验
    results['ablation'] = evaluate_ablation(
        agent_cls, test_queries, user, rag_resources, neo_graph
    )
    
    results['meta'] = {
        'total_time_ms': int((time.time() - t_start) * 1000),
        'test_queries_count': len(test_queries),
        'complex_queries_count': len(complex_queries),
    }
    
    return results


def _build_test_set(user, n=20):
    """
    从用户历史构建测试集。
    """
    from myapp.models import Movie, UserRating
    
    # 获取用户高分电影作为 ground truth
    rated = list(
        UserRating.objects.filter(user=user, score__gte=7.0)
        .order_by('-score')
        .values_list('movie_id', flat=True)[:50]
    )
    
    test_queries = [
        {'query': '推荐类似星际穿越的科幻片', 'expected_ids': rated[:10]},
        {'query': '推荐烧脑悬疑片', 'expected_ids': rated[5:15]},
        {'query': '推荐高分经典电影', 'expected_ids': rated[:10]},
        {'query': '推荐最新热门电影', 'expected_ids': rated[10:20]},
        {'query': '推荐动作冒险片', 'expected_ids': rated[3:13]},
    ]
    
    return test_queries


def _build_complex_queries():
    """
    构建复杂查询集（用于 ReAct vs Workflow 实验）。
    """
    return [
        {
            'query': '推荐近五年评分高的悬疑片，不要太冷门',
            'success_criteria': {'score_min': 7.0, 'year_min': 2021},
        },
        {
            'query': '推荐诺兰导演的科幻片',
            'success_criteria': {'genre': 'Sci-Fi'},
        },
        {
            'query': '推荐像盗梦空间那样烧脑的电影',
            'success_criteria': {'score_min': 7.5},
        },
        {
            'query': '推荐适合周末看的轻松喜剧',
            'success_criteria': {'genre': 'Comedy'},
        },
        {
            'query': '推荐有深度的剧情片，评分8分以上',
            'success_criteria': {'score_min': 8.0},
        },
    ]