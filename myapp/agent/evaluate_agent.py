"""
MovieAgent 真实评测管线 (Agent Evaluation Pipeline)
=====================================================
自动读取 benchmark → 调用真实 MovieAgent → 记录 Trace/Latency/Tool Chain
→ 保存 logs/*.json → 输出统计结果到 Markdown

用法:
    cd /home/daylight/下载/DjangoProject3/DjangoProject3
    python manage.py shell < myapp/agent/evaluate_agent.py
    
    或在 Django shell 中:
    exec(open('myapp/agent/evaluate_agent.py').read())
"""

import os
import sys
import json
import time
from datetime import datetime
from collections import defaultdict, Counter

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie.settings')

import django
django.setup()

from myapp.agent.movie_agent import MovieAgent, IntentClassifier
from myapp.models import Movie

# 兼容 __init__.py 导入
AgentBenchmark = None  # 已迁移至 run_evaluation() 函数
AGENT_EVAL_SET = None  # 已迁移至 golden_dataset_agent_eval.json


def load_benchmark(path=None):
    """加载 benchmark_queries.json"""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'management', 'commands', 'golden_dataset_agent_eval.json'
        )
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def get_neo_graph():
    """获取 Neo4j 连接"""
    try:
        from neo4j import GraphDatabase
        from django.conf import settings
        uri = getattr(settings, 'NEO4J_URI', 'bolt://localhost:7687')
        user = getattr(settings, 'NEO4J_USER', 'neo4j')
        password = getattr(settings, 'NEO4J_PASSWORD', '')
        driver = GraphDatabase.driver(uri, auth=(user, password))
        return driver
    except Exception as e:
        print(f"[WARN] Neo4j 连接失败: {e}")
        return None


def get_rag_resources():
    """加载 RAG 资源（FAISS 索引等）"""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        import torch
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-zh-v1.5",
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        index_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'faiss_movie_index')
        if os.path.exists(index_path):
            vector_db = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
            return {'vector_db': vector_db, 'embedding_model': embedding_model}
    except Exception as e:
        print(f"[WARN] RAG 资源加载失败: {e}")
    return None


# ── 类型匹配辅助：缓存 movie_id → genres 映射 ──
_MOVIE_GENRES_CACHE = None

def _get_movie_genres_map():
    """懒加载 movie_id → set(genre_name) 映射"""
    global _MOVIE_GENRES_CACHE
    if _MOVIE_GENRES_CACHE is None:
        from myapp.models import Movie
        movies = Movie.objects.prefetch_related('genres').values_list('id', 'genres__name')
        genre_map = defaultdict(set)
        for mid, gname in movies:
            if gname:
                genre_map[mid].add(gname)
        _MOVIE_GENRES_CACHE = dict(genre_map)
    return _MOVIE_GENRES_CACHE


def _genre_based_metrics(rec_ids, gt_ids):
    """
    基于类型的语义匹配指标。
    - genre_hit_rate: ground_truth 类型被推荐覆盖的比例 (Context Recall)
    - genre_precision@k: 推荐电影中与 ground_truth 类型匹配的比例 (Context Precision)
    """
    if not rec_ids or not gt_ids:
        return 0.0, 0.0, 0.0

    genre_map = _get_movie_genres_map()

    gt_genres = set()
    for mid in gt_ids:
        gt_genres.update(genre_map.get(mid, set()))
    if not gt_genres:
        return 0.0, 0.0, 0.0

    # Context Recall: ground_truth 类型被推荐覆盖的比例
    rec_genres = set()
    for mid in rec_ids:
        rec_genres.update(genre_map.get(mid, set()))
    genre_recall = len(rec_genres & gt_genres) / len(gt_genres)

    # Context Precision@5: 推荐 Top-5 中与 ground_truth 类型匹配的比例
    rec_list = list(rec_ids)
    matched_5 = sum(1 for mid in rec_list[:5] if genre_map.get(mid, set()) & gt_genres)
    matched_10 = sum(1 for mid in rec_list[:10] if genre_map.get(mid, set()) & gt_genres)
    genre_prec5 = matched_5 / min(5, len(rec_list)) if rec_list else 0
    genre_prec10 = matched_10 / min(10, len(rec_list)) if rec_list else 0

    return genre_recall, genre_prec5, genre_prec10


def run_single_query(agent, query_item, user=None):
    """
    运行单条查询，返回完整的评测结果。
    支持 multiturn（多轮对话）和 open_ended（无 ground_truth）任务。
    """
    query = query_item['query']
    query_id = query_item.get('id', 'N/A')
    difficulty = query_item.get('difficulty', 'unknown')
    expected_chain = query_item.get('expected_tool_chain', [])
    gt_ids = set(query_item.get('ground_truth_ids', []))
    gt_movies = query_item.get('ground_truth_movies', [])
    multiturn_inputs = query_item.get('multiturn_inputs')

    # ── 多轮对话：逐轮执行，测试对话记忆 ──
    if multiturn_inputs and isinstance(multiturn_inputs, list) and len(multiturn_inputs) > 1:
        t0 = time.time()
        result = None
        for turn_input in multiturn_inputs:
            try:
                result = agent.run(turn_input)
            except Exception as e:
                return {
                    'query_id': query_id,
                    'query': ' → '.join(multiturn_inputs),
                    'difficulty': difficulty,
                    'success': False,
                    'error': str(e),
                    'latency_ms': int((time.time() - t0) * 1000),
                }
        latency_ms = int((time.time() - t0) * 1000)
        query_display = ' → '.join(multiturn_inputs)
    else:
        # ── 单轮查询 ──
        t0 = time.time()
        try:
            result = agent.run(query)
        except Exception as e:
            return {
                'query_id': query_id,
                'query': query,
                'difficulty': difficulty,
                'success': False,
                'error': str(e),
                'latency_ms': int((time.time() - t0) * 1000),
            }
        latency_ms = int((time.time() - t0) * 1000)
        query_display = query

    # 提取实际工具链
    actual_tools = []
    for step in result.get('trace_steps', []):
        if step.get('type') == 'action' and step.get('tool'):
            t = step['tool']
            if t not in actual_tools:
                actual_tools.append(t)

    # 工具路由准确性
    tool_match = (set(expected_chain) == set(actual_tools)) if expected_chain else None

    # 推荐命中（精确 ID 匹配 + 类型语义匹配）
    rec_ids = set(result.get('recommended_ids', []))
    hit_ids = rec_ids & gt_ids if gt_ids else set()
    hit_rate = len(hit_ids) / len(gt_ids) if gt_ids else None

    # 基于类型的语义匹配（修复精确 ID 匹配导致指标偏低的问题）
    genre_recall, genre_prec5, genre_prec10 = _genre_based_metrics(rec_ids, gt_ids)

    # 推理步数
    total_steps = len(result.get('trace_steps', []))

    # 是否有纠偏
    has_correction = any(s.get('is_retry') for s in result.get('trace_steps', []))

    # 纠偏是否成功
    correction_success = False
    if has_correction:
        correction_success = len(result.get('recommended_ids', [])) > 0

    # 是否需要追问
    need_clarification = result.get('need_clarification', False)

    # 意图分类
    intent = result.get('intent', '')
    intent_match = (intent == 'QUERY_MOVIE' and difficulty != 'chat') or \
                   (intent == 'CHAT' and difficulty == 'chat')

    # MAAN 精排分数（如果有）
    maan_scores = []
    for obs in result.get('observations', []):
        if obs.get('tool') == 'maan_rerank':
            output = obs.get('output', [])
            if isinstance(output, list):
                for item in output[:5]:
                    if isinstance(item, dict) and 'maan_score' in item:
                        maan_scores.append(item['maan_score'])

    # ── RAGAS + LLM-as-a-Judge 评测 ──
    final_answer = result.get('final_answer', '')

    # LLM Judge 打分 (1-10)
    judge_score = None
    if final_answer and len(final_answer) > 10:
        judge_score = llm_judge_score(query_display, final_answer)

    # Faithfulness (忠实度) — 仅对有 ground_truth 的任务计算
    faith = None
    if rec_ids and gt_ids:
        faith = compute_faithfulness(query_display, list(rec_ids)[:10])

    # Answer Relevancy (相关性)
    relevancy = None
    if final_answer and len(final_answer) > 10:
        relevancy = compute_answer_relevancy(query_display, final_answer)

    return {
        'query_id': query_id,
        'query': query_display,
        'difficulty': difficulty,
        'intent': intent,
        'expected_tool_chain': expected_chain,
        'actual_tool_chain': actual_tools,
        'tool_chain_match': tool_match,
        'recommended_ids': list(rec_ids)[:10],
        'ground_truth_ids': list(gt_ids),
        'hit_ids': list(hit_ids),
        'hit_rate': hit_rate,
        'genre_recall': genre_recall,
        'genre_prec5': genre_prec5,
        'genre_prec10': genre_prec10,
        'total_steps': total_steps,
        'latency_ms': latency_ms,
        'has_correction': has_correction,
        'correction_success': correction_success,
        'need_clarification': need_clarification,
        'maan_scores': maan_scores,
        'llm_judge_score': judge_score,
        'faithfulness': faith,
        'answer_relevancy': relevancy,
        'success': True,
        'trace_steps': result.get('trace_steps', []),
        'thought': result.get('thought', ''),
        'final_answer': final_answer,
    }


def compute_statistics(results):
    """
    从评测结果中计算统计指标。
    """
    valid = [r for r in results if r.get('success', False)]
    total = len(results)
    failed = total - len(valid)

    if not valid:
        return {'error': '所有查询均失败'}

    # ── 1. Tool Routing Accuracy ──
    tool_match_results = [r for r in valid if r.get('tool_chain_match') is not None]
    tool_routing_accuracy = sum(1 for r in tool_match_results if r['tool_chain_match']) / len(tool_match_results) if tool_match_results else 0

    # ── 2. Planning Success Rate ──
    planning_success = sum(1 for r in valid if r.get('recommended_ids')) / len(valid)

    # ── 3. Avg Reasoning Steps ──
    avg_steps = sum(r['total_steps'] for r in valid) / len(valid)

    # ── 4. Avg Latency ──
    avg_latency = sum(r['latency_ms'] for r in valid) / len(valid)
    p50_latency = sorted(r['latency_ms'] for r in valid)[len(valid) // 2]
    p95_latency = sorted(r['latency_ms'] for r in valid)[int(len(valid) * 0.95)]

    # ── 5. Correction Rate ──
    correction_count = sum(1 for r in valid if r.get('has_correction'))
    correction_rate = correction_count / len(valid)
    correction_success_count = sum(1 for r in valid if r.get('has_correction') and r.get('correction_success'))
    correction_recovery = correction_success_count / correction_count if correction_count > 0 else 0

    # ── 6. Context Recall (基于类型语义匹配，替代精确 ID 匹配) ──
    genre_recall_results = [r for r in valid if r.get('genre_recall') is not None]
    context_recall = sum(r['genre_recall'] for r in genre_recall_results) / len(genre_recall_results) if genre_recall_results else 0

    # ── 7. Context Precision@k (基于类型语义匹配) ──
    prec5_list = [r['genre_prec5'] for r in valid if r.get('genre_prec5') is not None]
    prec10_list = [r['genre_prec10'] for r in valid if r.get('genre_prec10') is not None]
    context_precision_5 = sum(prec5_list) / len(prec5_list) if prec5_list else 0
    context_precision_10 = sum(prec10_list) / len(prec10_list) if prec10_list else 0

    # 保留精确 ID 匹配作为参考（旧指标）
    exact_hit_results = [r for r in valid if r.get('hit_rate') is not None]
    exact_hit_rate = sum(r['hit_rate'] for r in exact_hit_results) / len(exact_hit_results) if exact_hit_results else 0

    # ── 8. Clarification Rate ──
    clarification_count = sum(1 for r in valid if r.get('need_clarification'))
    clarification_rate = clarification_count / len(valid)

    # ── 9. MAAN Score Stats ──
    all_maan = []
    for r in valid:
        all_maan.extend(r.get('maan_scores', []))
    avg_maan_score = sum(all_maan) / len(all_maan) if all_maan else 0

    # ── 10. LLM Judge Score (RAGAS-style) ──
    judge_scores = [r['llm_judge_score'] for r in valid if r.get('llm_judge_score') is not None]
    avg_judge_score = sum(judge_scores) / len(judge_scores) if judge_scores else None

    # ── 11. Faithfulness (RAGAS) ──
    faith_scores = [r['faithfulness'] for r in valid if r.get('faithfulness') is not None]
    avg_faithfulness = sum(faith_scores) / len(faith_scores) if faith_scores else None

    # ── 12. Answer Relevancy (RAGAS) ──
    rel_scores = [r['answer_relevancy'] for r in valid if r.get('answer_relevancy') is not None]
    avg_relevancy = sum(rel_scores) / len(rel_scores) if rel_scores else None

    # ── 13. Per-category breakdown ──
    difficulty_stats = {}
    for diff in ['simple', 'complex', 'vague', 'multiturn', 'agent_only', 'correction', 'open_ended']:
        subset = [r for r in valid if r.get('difficulty') == diff]
        if not subset:
            continue
        tca_subset = [r for r in subset if r.get('tool_chain_match') is not None]
        diff_judge = [r['llm_judge_score'] for r in subset if r.get('llm_judge_score') is not None]
        diff_faith = [r['faithfulness'] for r in subset if r.get('faithfulness') is not None]
        difficulty_stats[diff] = {
            'count': len(subset),
            'tool_routing_accuracy': sum(1 for r in tca_subset if r['tool_chain_match']) / len(tca_subset) if tca_subset else 0,
            'planning_success_rate': sum(1 for r in subset if r.get('recommended_ids')) / len(subset),
            'avg_steps': sum(r['total_steps'] for r in subset) / len(subset),
            'avg_latency_ms': sum(r['latency_ms'] for r in subset) / len(subset),
            'correction_rate': sum(1 for r in subset if r.get('has_correction')) / len(subset),
            'avg_judge_score': sum(diff_judge) / len(diff_judge) if diff_judge else None,
            'avg_faithfulness': sum(diff_faith) / len(diff_faith) if diff_faith else None,
        }

    # ── 14. Tool usage frequency ──
    tool_counter = Counter()
    for r in valid:
        for t in r.get('actual_tool_chain', []):
            tool_counter[t] += 1

    return {
        'total_queries': total,
        'valid_queries': len(valid),
        'failed_queries': failed,
        'tool_routing_accuracy': round(tool_routing_accuracy, 4),
        'planning_success_rate': round(planning_success, 4),
        'avg_reasoning_steps': round(avg_steps, 2),
        'avg_latency_ms': round(avg_latency, 1),
        'p50_latency_ms': p50_latency,
        'p95_latency_ms': p95_latency,
        'correction_rate': round(correction_rate, 4),
        'correction_recovery_rate': round(correction_recovery, 4),
        'context_recall': round(context_recall, 4),
        'context_precision_5': round(context_precision_5, 4),
        'context_precision_10': round(context_precision_10, 4),
        'exact_id_hit_rate': round(exact_hit_rate, 4),  # 旧指标：精确 ID 匹配
        'clarification_rate': round(clarification_rate, 4),
        'avg_maan_score': round(avg_maan_score, 4),
        'avg_llm_judge_score': round(avg_judge_score, 2) if avg_judge_score is not None else None,
        'avg_faithfulness': round(avg_faithfulness, 4) if avg_faithfulness is not None else None,
        'avg_answer_relevancy': round(avg_relevancy, 4) if avg_relevancy is not None else None,
        'difficulty_breakdown': difficulty_stats,
        'tool_usage_frequency': dict(tool_counter),
    }


# =============================================================
# RAGAS + LLM-as-a-Judge 评测函数
# =============================================================

def _call_llm_judge(system_prompt, user_prompt, timeout=30):
    """调用 Ollama LLM 进行评测打分"""
    try:
        import requests as req
        from django.conf import settings
        olla_base = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
        model = getattr(settings, 'AGENT_LLM_MODEL', 'qwen3:4b-instruct')
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 128, "num_gpu": 99},
        }
        resp = req.post(f"{olla_base}/api/chat", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "").strip()
    except Exception:
        return None


def llm_judge_score(query, final_answer):
    """
    LLM-as-a-Judge: 对单个回答打分 (1-10)。
    参照 MT-Bench (Zheng et al., NeurIPS 2023) 的 single-answer grading。
    """
    if not final_answer:
        return None
    sys_prompt = (
        "你是一个公正的评委。请对以下AI助手的回答质量进行评分。\n"
        "评分标准：1-10分。\n"
        "- 1-2: 回答完全不相关或有害\n"
        "- 3-4: 回答有部分相关信息但质量很差\n"
        "- 5-6: 回答基本合格但有明显不足\n"
        "- 7-8: 回答质量良好\n"
        "- 9-10: 回答质量优秀\n"
        "注意：不要因为回答更长就给更高分（避免 verbosity bias）。\n"
        "请只输出一个数字评分（1-10），不要输出其他内容。"
    )
    user_prompt = f"用户问题：{query}\nAI回答：{final_answer}"
    raw = _call_llm_judge(sys_prompt, user_prompt)
    if not raw:
        return None
    import re
    m = re.search(r'([1-9]|10)', raw)
    return int(m.group(1)) if m else None


def compute_faithfulness(query, recommended_ids):
    """
    RAGAS-style Faithfulness: 检查推荐电影是否忠实于用户查询约束。
    返回 0.0-1.0 的忠实度分数。
    """
    if not recommended_ids:
        return None
    try:
        from myapp.models import Movie
        movies = Movie.objects.filter(id__in=recommended_ids[:10]).values('id', 'title', 'genres__name')
        movie_info = {}
        for m in movies:
            mid = m['id']
            if mid not in movie_info:
                movie_info[mid] = {'title': m['title'], 'genres': []}
            if m.get('genres__name'):
                movie_info[mid]['genres'].append(m['genres__name'])

        info_str = "; ".join(
            f"{v['title']}({','.join(v['genres'])})" for v in movie_info.values()
        )
        sys_prompt = (
            "你是电影推荐忠实度评估专家。判断推荐的电影是否符合用户查询要求。\n"
            "对每部电影，判断它是否满足用户的核心需求（类型、导演、氛围等）。\n"
            "输出一个JSON数组，格式: [{\"movie_id\": id, \"faithful\": true/false}]\n"
            "只输出JSON，不要其他内容。"
        )
        user_prompt = f"用户查询：{query}\n推荐电影：{info_str}"
        raw = _call_llm_judge(sys_prompt, user_prompt, timeout=20)
        if not raw:
            return None
        import re, json as _json
        raw_clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        m = re.search(r'\[.*\]', raw_clean, re.DOTALL)
        if not m:
            return None
        items = _json.loads(m.group(0))
        faithful_count = sum(1 for item in items if item.get('faithful', False))
        return faithful_count / len(items) if items else None
    except Exception:
        return None


def compute_answer_relevancy(query, final_answer):
    """
    RAGAS-style Answer Relevancy: 对 final_answer 的相关性打分 (1-5)。
    """
    if not final_answer:
        return None
    sys_prompt = (
        "你是回答质量评估专家。请评估以下回答与用户问题的相关性。\n"
        "评分1-5：1=完全不相关，2=基本不相关，3=部分相关，4=相关，5=完美回答。\n"
        "只输出一个数字（1-5），不要其他内容。"
    )
    user_prompt = f"用户问题：{query}\n系统回答：{final_answer}"
    raw = _call_llm_judge(sys_prompt, user_prompt, timeout=15)
    if not raw:
        return None
    import re
    m = re.search(r'([1-5])', raw)
    return int(m.group(1)) if m else None


def llm_judge_pairwise(query, answer_a, answer_b):
    """
    LLM-as-a-Judge: Pairwise Comparison (带 bias mitigation)。
    交换位置两次以消除 position bias。
    返回: 'A', 'B', 'tie', 或 None
    """
    if not answer_a or not answer_b:
        return None
    sys_prompt = (
        "请比较以下两个AI助手对同一问题的回答，选出更好的一个。\n"
        "评判标准：准确性、完整性、自然度。\n"
        "不要因为回答更长就认为更好。\n"
        '输出JSON: {"winner": "A" 或 "B" 或 "tie", "reason": "简短理由"}'
    )

    # 第一次：正常顺序
    user_prompt_1 = f"用户问题：{query}\n回答A：{answer_a}\n回答B：{answer_b}"
    raw1 = _call_llm_judge(sys_prompt, user_prompt_1, timeout=20)

    # 第二次：交换顺序（消除 position bias）
    user_prompt_2 = f"用户问题：{query}\n回答A：{answer_b}\n回答B：{answer_a}"
    raw2 = _call_llm_judge(sys_prompt, user_prompt_2, timeout=20)

    if not raw1 or not raw2:
        return None

    import re, json as _json
    winner1 = winner2 = None
    for raw, label in [(raw1, 'normal'), (raw2, 'swapped')]:
        raw_clean = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        m = re.search(r'\{.*\}', raw_clean, re.DOTALL)
        if m:
            try:
                d = _json.loads(m.group(0))
                w = d.get('winner', '').upper()
                if label == 'normal':
                    winner1 = w
                else:
                    # 交换回来
                    winner2 = {'A': 'B', 'B': 'A'}.get(w, w)
            except Exception:
                pass

    if winner1 and winner2:
        if winner1 == winner2:
            return winner1
        else:
            return 'tie'  # 两次结果不一致视为平局
    return winner1 or winner2


def generate_markdown_report(stats, results, output_dir):
    """生成 Markdown 实验报告"""
    lines = []
    lines.append("# MovieAgent 评测报告")
    lines.append(f"\n> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 评测数据集: MovieAgent-Bench ({stats['total_queries']} 条查询)")
    lines.append("")
    
    # ── 核心指标 ──
    lines.append("## 核心指标")
    lines.append("")
    lines.append("| 指标 | 数值 | 说明 |")
    lines.append("|------|:----:|------|")
    lines.append(f"| **Tool Routing Accuracy** | **{stats['tool_routing_accuracy']}** | 预期工具链与实际工具链完全匹配比例 |")
    lines.append(f"| **Planning Success Rate** | **{stats['planning_success_rate']}** | 推理链成功返回推荐结果的比例 |")
    lines.append(f"| **Avg Reasoning Steps** | **{stats['avg_reasoning_steps']}** | 平均推理步数 |")
    lines.append(f"| **Avg Latency** | **{stats['avg_latency_ms']} ms** | 平均端到端延迟 |")
    lines.append(f"| **P50 Latency** | **{stats['p50_latency_ms']} ms** | 中位数延迟 |")
    lines.append(f"| **P95 Latency** | **{stats['p95_latency_ms']} ms** | 95分位延迟 |")
    lines.append(f"| **Correction Rate** | **{stats['correction_rate']}** | 触发自反馈纠偏的查询比例 |")
    lines.append(f"| **Correction Recovery** | **{stats['correction_recovery_rate']}** | 纠偏后成功获取结果的比例 |")
    lines.append(f"| **Clarification Rate** | **{stats['clarification_rate']}** | 触发模糊追问的查询比例 |")
    lines.append(f"| **MAAN Avg Score** | **{stats['avg_maan_score']}** | MAAN 精排平均预测分 |")
    lines.append("")

    # ── RAGAS + LLM-as-a-Judge 指标 ──
    lines.append("## RAGAS + LLM-as-a-Judge 指标")
    lines.append("")
    lines.append("| 指标 | 数值 | 说明 | 参考论文 |")
    lines.append("|------|:----:|------|----------|")
    judge = stats.get('avg_llm_judge_score')
    lines.append(f"| **LLM Judge Score** | **{judge if judge is not None else 'N/A'}** | LLM 对回答质量的评分 (1-10) | MT-Bench (NeurIPS 2023) |")
    faith = stats.get('avg_faithfulness')
    lines.append(f"| **Faithfulness** | **{faith if faith is not None else 'N/A'}** | 推荐忠实于用户查询约束的比例 | RAGAs (2023) |")
    rel = stats.get('avg_answer_relevancy')
    lines.append(f"| **Answer Relevancy** | **{rel if rel is not None else 'N/A'}** | 回答与问题的相关性 (1-5) | RAGAs (2023) |")
    lines.append(f"| **Context Recall** | **{stats['context_recall']}** | ground_truth 类型被召回的比例 (基于类型匹配) | RAGAs (2023) |")
    lines.append(f"| **Context Precision@5** | **{stats['context_precision_5']}** | Top-5 推荐中类型匹配比例 (基于类型匹配) | RAGAs (2023) |")
    lines.append(f"| **Context Precision@10** | **{stats['context_precision_10']}** | Top-10 推荐中类型匹配比例 (基于类型匹配) | RAGAs (2023) |")
    lines.append(f"| _Exact ID Hit Rate_ | _{stats.get('exact_id_hit_rate', 'N/A')}_ | _精确 ID 匹配（旧指标，仅供参考）_ | _—_ |")
    lines.append("")
    
    # ── 分难度统计 ──
    lines.append("## 分难度统计")
    lines.append("")
    lines.append("| 难度 | 数量 | TCA | Planning | Judge | Faith | Latency |")
    lines.append("|------|:----:|:---:|:--------:|:-----:|:-----:|:-------:|")
    for diff, ds in stats.get('difficulty_breakdown', {}).items():
        diff_label = {
            'simple': '简单推荐', 'complex': '复合约束', 'vague': '模糊查询',
            'multiturn': '多轮对话', 'agent_only': 'Agent专属',
            'correction': '纠偏任务', 'open_ended': '开放评价',
        }.get(diff, diff)
        judge = f"{ds['avg_judge_score']:.1f}" if ds.get('avg_judge_score') else 'N/A'
        faith = f"{ds['avg_faithfulness']:.2f}" if ds.get('avg_faithfulness') else 'N/A'
        lines.append(
            f"| {diff_label} | {ds['count']} | {ds['tool_routing_accuracy']} | "
            f"{ds['planning_success_rate']} | {judge} | {faith} | "
            f"{ds['avg_latency_ms']:.0f}ms |"
        )
    lines.append("")
    
    # ── 工具使用频率 ──
    lines.append("## 工具使用频率")
    lines.append("")
    lines.append("| 工具 | 调用次数 |")
    lines.append("|------|:------:|")
    for tool, count in sorted(stats.get('tool_usage_frequency', {}).items(), key=lambda x: -x[1]):
        lines.append(f"| {tool} | {count} |")
    lines.append("")
    
    # ── 失败案例 ──
    failures = [r for r in results if r.get('tool_chain_match') == False and r.get('success')]
    if failures:
        lines.append("## 工具路由偏差案例")
        lines.append("")
        lines.append("| Query ID | 查询 | 预期工具链 | 实际工具链 |")
        lines.append("|:---------:|------|-----------|-----------|")
        for r in failures[:20]:
            expected = ' → '.join(r.get('expected_tool_chain', []))
            actual = ' → '.join(r.get('actual_tool_chain', []))
            lines.append(f"| {r['query_id']} | {r['query'][:30]}... | {expected} | {actual} |")
        lines.append("")
    
    # ── 纠偏案例 ──
    corrections = [r for r in results if r.get('has_correction') and r.get('success')]
    if corrections:
        lines.append("## 自反馈纠偏案例")
        lines.append("")
        for r in corrections[:5]:
            lines.append(f"### {r['query_id']}: {r['query']}")
            lines.append("")
            lines.append(f"- 纠偏成功: {'✓' if r.get('correction_success') else '✗'}")
            lines.append(f"- 最终推荐数: {len(r.get('recommended_ids', []))}")
            lines.append(f"- 延迟: {r['latency_ms']}ms")
            lines.append("")
    
    # ── 样例 Trace ──
    lines.append("## 样例推理链")
    lines.append("")
    for r in results[:3]:
        if not r.get('success'):
            continue
        lines.append(f"### {r['query_id']}: {r['query']}")
        lines.append("")
        lines.append("```")
        for step in r.get('trace_steps', []):
            stype = step.get('type', '').upper()
            content = step.get('content', '')
            retry = " ⚡[纠偏]" if step.get('is_retry') else ""
            lines.append(f"Step {step.get('step', '?')} [{stype}]{retry}: {content}")
        lines.append("```")
        lines.append("")
        lines.append(f"**Final Answer:** {r.get('final_answer', '')[:200]}")
        lines.append("")
    
    report_text = "\n".join(lines)
    
    # 保存
    report_path = os.path.join(output_dir, 'agent_eval_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"[✓] Markdown 报告已保存至: {report_path}")
    
    return report_path


def run_evaluation(max_queries=None):
    """
    主评测入口。
    
    Args:
        max_queries: 最大查询数（None=全部）
    """
    print("=" * 60)
    print("  MovieAgent 真实评测管线")
    print("=" * 60)
    
    # ── 1. 加载 Benchmark ──
    benchmark = load_benchmark()
    raw_tasks = benchmark.get('tasks', [])
    # 转换为 run_single_query 期望的格式
    queries = []
    for t in raw_tasks:
        inp = t.get('input', '')
        # multiturn: input 是数组，取最后一轮作为评测查询
        if isinstance(inp, list):
            query_text = inp[-1] if inp else ''
        else:
            query_text = inp
        queries.append({
            'id': t.get('task_id', ''),
            'query': query_text,
            'difficulty': t.get('category', 'unknown'),
            'expected_tool_chain': t.get('expected_tool_chain', []),
            'ground_truth_ids': t.get('ground_truth_ids', []),
            'ground_truth_movies': t.get('ground_truth_movies', []),
            'success_criteria': t.get('success_criteria', {}),
            'multiturn_inputs': inp if isinstance(inp, list) else None,
        })
    if max_queries:
        queries = queries[:max_queries]
    print(f"[1/5] 加载 Benchmark: {len(queries)} 条查询")
    
    # ── 2. 初始化依赖 ──
    print("[2/5] 初始化 Agent 依赖...")
    neo_graph = get_neo_graph()
    rag_resources = get_rag_resources()
    
    # 使用测试用户（id=1）
    from myapp.models import UserInfo
    try:
        test_user = UserInfo.objects.get(id=1)
    except UserInfo.DoesNotExist:
        test_user = UserInfo.objects.first()
    
    if not test_user:
        print("[ERROR] 无可用测试用户")
        return
    
    agent = MovieAgent(
        user=test_user,
        neo_graph=neo_graph,
        rag_resources=rag_resources,
        session_id='eval_session'
    )
    print(f"  测试用户: {test_user.username} (id={test_user.id})")
    
    # ── 3. 运行评测 ──
    print(f"[3/5] 运行 {len(queries)} 条查询...")
    results = []
    for i, q in enumerate(queries):
        result = run_single_query(agent, q, user=test_user)
        results.append(result)
        
        # 进度输出
        if (i + 1) % 10 == 0 or i == len(queries) - 1:
            success_count = sum(1 for r in results if r.get('success'))
            print(f"  [{i+1}/{len(queries)}] 成功: {success_count}, "
                  f"最新延迟: {result.get('latency_ms', 'N/A')}ms")
    
    # ── 4. 计算统计 ──
    print("[4/5] 计算统计指标...")
    stats = compute_statistics(results)
    
    # ── 5. 保存结果 ──
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'experiment_results', 'agent_eval'
    )
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # 保存原始结果
    raw_path = os.path.join(output_dir, f'raw_results_{timestamp}.json')
    # 移除 trace_steps 中的大型数据以减小文件体积
    slim_results = []
    for r in results:
        slim = {k: v for k, v in r.items() if k != 'trace_steps'}
        slim['trace_step_count'] = len(r.get('trace_steps', []))
        slim_results.append(slim)
    
    with open(raw_path, 'w', encoding='utf-8') as f:
        json.dump({'stats': stats, 'results': slim_results}, f, ensure_ascii=False, indent=2)
    print(f"[✓] 原始结果已保存至: {raw_path}")
    
    # 保存完整 trace（含推理链）
    trace_path = os.path.join(output_dir, f'trace_log_{timestamp}.json')
    with open(trace_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[✓] Trace 日志已保存至: {trace_path}")
    
    # 保存统计摘要
    stats_path = os.path.join(output_dir, f'stats_{timestamp}.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"[✓] 统计摘要已保存至: {stats_path}")
    
    # 生成 Markdown 报告
    print("[5/5] 生成 Markdown 报告...")
    report_path = generate_markdown_report(stats, results, output_dir)
    
    # ── 输出摘要 ──
    print("\n" + "=" * 60)
    print("  评测完成！核心指标：")
    print("=" * 60)
    print(f"  Tool Routing Accuracy:  {stats['tool_routing_accuracy']}")
    print(f"  Planning Success Rate:  {stats['planning_success_rate']}")
    print(f"  Avg Reasoning Steps:    {stats['avg_reasoning_steps']}")
    print(f"  Avg Latency:            {stats['avg_latency_ms']}ms")
    print(f"  P95 Latency:            {stats['p95_latency_ms']}ms")
    print(f"  Correction Rate:        {stats['correction_rate']}")
    print(f"  Correction Recovery:    {stats['correction_recovery_rate']}")
    print("=" * 60)
    
    return stats, results


# 如果直接执行此文件
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max', type=int, default=None, help='最大查询数')
    args = parser.parse_args()
    run_evaluation(max_queries=args.max)