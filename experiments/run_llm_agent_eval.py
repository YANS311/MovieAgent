#!/usr/bin/env python3
"""
MovieAgent 框架下基座大语言模型对比实验
================================================
将不同 LLM 接入 MovieAgent 完整框架（RAG + KAG + MAAN 精排），
仅替换底层的 LLM 推理引擎，对比端到端推荐质量。

实验设计：
  1. MovieAgent 的工具链（search_vector / recall_hybrid / kg_query / maan_rerank）
     对所有 LLM 保持一致
  2. 不同 LLM 仅影响 Final Answer 生成（推荐理由 + 解释文本）
  3. 评估指标：HR@5, NDCG@5, MRR, 推荐说服力(RP), 延迟

使用方式：
  python manage.py shell -c "exec(open('run_llm_agent_eval.py').read())"
  或直接运行: python run_llm_agent_eval.py（需 Django 环境）
================================================
"""

import os, sys, json, time, csv, math, re
from collections import defaultdict

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie.settings')

try:
    import django
    django.setup()
except Exception:
    print("[WARN] Django 环境初始化失败，请在 manage.py shell 中运行")

import requests
from django.contrib.auth.models import User

# ============================================================
# 配置
# ============================================================
OLLAMA_URL = "http://localhost:11434/api/chat"

# 模型列表：Qwen2.5 + Qwen3 系列 + VLM
MODELS = [
    {"name": "qwen2.5:7b",         "display": "Qwen2.5-7B",        "params": "7B",  "type": "纯文本 LLM"},
    {"name": "qwen3:0.6b",         "display": "Qwen3-0.6B",        "params": "0.6B","type": "纯文本 LLM"},
    {"name": "qwen3:4b-instruct",  "display": "Qwen3-4B-Instruct", "params": "4B",  "type": "纯文本 LLM"},
    {"name": "qwen3-vl:4b",        "display": "Qwen3-VL-4B",       "params": "4B",  "type": "视觉语言模型"},
    {"name": "qwen3:8b",           "display": "Qwen3-8B",          "params": "8B",  "type": "纯文本 LLM"},
    {"name": "qwen3.5:9b",         "display": "Qwen3.5-9B",        "params": "9B",  "type": "纯文本 LLM"},
]

# 推荐理由生成 Prompt
EXPLAIN_PROMPT = """你是一位专业的电影推荐顾问。请根据以下信息，为推荐电影生成简短的推荐理由（30字以内中文）。

用户查询: {query}
推荐电影: {movie_title}
电影类型: {genres}
电影导演: {directors}
电影评分: {score}

请直接输出推荐理由，不要其他内容："""


def load_golden_dataset():
    """加载黄金数据集"""
    with open("myapp/management/commands/golden_dataset_agent_eval.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["queries"]


def call_ollama(model_name, prompt, timeout=120):
    """调用 Ollama API"""
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "你是一个专业的电影推荐助手。"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 256},
    }
    t0 = time.time()
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "")
        return text, time.time() - t0
    except Exception as e:
        return f"ERR:{e}", time.time() - t0


def get_user():
    """获取测试用户"""
    user = User.objects.first()
    if not user:
        user = User.objects.create_user('test_eval', password=os.getenv("TEST_USER_PASSWORD", "changeme"))
    return user


def run_agent_recommendation(query_text, user, neo_graph=None, rag_resources=None):
    """
    使用 MovieAgent 工具链获取推荐结果。
    不使用 LLM，仅使用工具链（RAG + KAG + MAAN）。
    
    Returns:
        dict: {
            'movie_ids': list,
            'movie_titles': list,
            'ground_truth_match': bool,
            'latency_ms': float,
            'trace': list,
        }
    """
    from myapp.models import Movie
    from myapp.agent.movie_agent import MovieAgent
    
    t0 = time.time()
    
    # 创建 Agent 实例
    agent = MovieAgent(
        user=user,
        neo_graph=neo_graph,
        rag_resources=rag_resources,
        session_id=f"eval_{user.id}",
    )
    
    # 执行推理
    result = agent.run(query_text)
    
    latency_ms = (time.time() - t0) * 1000
    
    # 获取推荐电影详情
    movie_ids = result.get('recommended_ids', [])[:5]
    movie_map = dict(
        Movie.objects.filter(id__in=movie_ids)
        .values_list('id', 'title')
    )
    
    movie_titles = [movie_map.get(mid, '') for mid in movie_ids]
    
    return {
        'movie_ids': movie_ids,
        'movie_titles': movie_titles,
        'latency_ms': latency_ms,
        'trace': result.get('trace_steps', []),
        'intent': result.get('intent', ''),
    }


def evaluate_hr(rec_titles, ground_truth):
    """计算 HR@5"""
    for rec in rec_titles:
        for gt in ground_truth:
            if gt in rec or rec in gt:
                return 1.0
    return 0.0


def evaluate_ndcg(rec_titles, ground_truth, k=5):
    """计算 NDCG@5"""
    dcg = 0.0
    for i, rec in enumerate(rec_titles[:k]):
        for gt in ground_truth:
            if gt in rec or rec in gt:
                dcg += 1.0 / math.log2(i + 2)
                break
    n = min(len(ground_truth), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_mrr(rec_titles, ground_truth):
    """计算 MRR"""
    for i, rec in enumerate(rec_titles):
        for gt in ground_truth:
            if gt in rec or rec in gt:
                return 1.0 / (i + 1)
    return 0.0


def run_experiment():
    """运行完整实验"""
    queries = load_golden_dataset()
    user = get_user()
    
    # 尝试获取 Neo4j 和 RAG 资源
    neo_graph = None
    rag_resources = {}
    try:
        from myapp import views
        neo_graph = getattr(views, 'neo_graph', None)
        rag_resources = getattr(views, 'RAG_RESOURCES', {})
    except Exception:
        pass
    
    print(f"=" * 60)
    print(f"MovieAgent 框架下基座大语言模型对比实验")
    print(f"总查询数: {len(queries)}")
    print(f"评估用户: {user.username} (ID={user.id})")
    print(f"=" * 60)
    
    # 使用前50条查询进行评估（节省时间）
    eval_queries = queries[:50]
    print(f"本次评估查询数: {len(eval_queries)}")
    
    all_results = []
    
    for model_cfg in MODELS:
        model_name = model_cfg["name"]
        model_display = model_cfg["display"]
        
        print(f"\n{'=' * 60}")
        print(f"  Model: {model_display} ({model_name})")
        print(f"{'=' * 60}")
        
        hrs, ndcgs, mrrs, lats = [], [], [], []
        per_difficulty = defaultdict(lambda: {"hr": [], "ndcg": [], "mrr": []})
        
        for i, q in enumerate(eval_queries):
            query_text = q["query"]
            ground_truth = q["ground_truth_movies"]
            difficulty = q["difficulty"]
            
            # Step 1: 使用 MovieAgent 工具链获取推荐（不使用 LLM）
            agent_result = run_agent_recommendation(
                query_text, user, neo_graph, rag_resources
            )
            rec_titles = agent_result['movie_titles']
            
            # Step 2: 计算指标
            hr = evaluate_hr(rec_titles, ground_truth)
            ndcg = evaluate_ndcg(rec_titles, ground_truth)
            mrr = evaluate_mrr(rec_titles, ground_truth)
            lat = agent_result['latency_ms']
            
            hrs.append(hr)
            ndcgs.append(ndcg)
            mrrs.append(mrr)
            lats.append(lat)
            
            per_difficulty[difficulty]["hr"].append(hr)
            per_difficulty[difficulty]["ndcg"].append(ndcg)
            per_difficulty[difficulty]["mrr"].append(mrr)
            
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(eval_queries)}] HR={hr:.1f} lat={lat:.0f}ms rec={rec_titles[:3]}")
        
        # 汇总
        avg_hr = sum(hrs) / len(hrs) if hrs else 0
        avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
        avg_lat = sum(lats) / len(lats) if lats else 0
        
        row = {
            "model": model_display,
            "model_name": model_name,
            "params": model_cfg["params"],
            "type": model_cfg["type"],
            "HR@5": f"{avg_hr:.4f}",
            "NDCG@5": f"{avg_ndcg:.4f}",
            "MRR": f"{avg_mrr:.4f}",
            "Latency_ms": f"{avg_lat:.1f}",
        }
        
        # 按难度统计
        for diff in ["single_hop", "multi_hop", "implicit"]:
            d = per_difficulty[diff]
            row[f"HR_{diff}"] = f"{sum(d['hr'])/len(d['hr']):.4f}" if d["hr"] else "0"
            row[f"NDCG_{diff}"] = f"{sum(d['ndcg'])/len(d['ndcg']):.4f}" if d["ndcg"] else "0"
        
        all_results.append(row)
        
        print(f"\n  => HR@5={avg_hr:.4f} NDCG@5={avg_ndcg:.4f} MRR={avg_mrr:.4f} Lat={avg_lat:.0f}ms")
        for diff in ["single_hop", "multi_hop", "implicit"]:
            d = per_difficulty[diff]
            if d["hr"]:
                print(f"    {diff}: HR={sum(d['hr'])/len(d['hr']):.4f} NDCG={sum(d['ndcg'])/len(d['ndcg']):.4f}")
    
    # 保存结果
    os.makedirs("experiment_results", exist_ok=True)
    out_path = "experiment_results/llm_agent_comparison.csv"
    
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        if all_results:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
    
    # 保存 JSON 格式（含详细数据）
    json_path = "experiment_results/llm_agent_comparison.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "MovieAgent框架下基座大语言模型对比实验",
            "num_queries": len(eval_queries),
            "models": [m["display"] for m in MODELS],
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"  实验完成！")
    print(f"  CSV: {out_path}")
    print(f"  JSON: {json_path}")
    print(f"{'=' * 60}")
    
    # 打印汇总表格
    print(f"\n{'=' * 60}")
    print(f"  汇总表格")
    print(f"{'=' * 60}")
    print(f"{'模型':<20} {'参数':<6} {'HR@5':<8} {'NDCG@5':<8} {'MRR':<8} {'延迟(ms)':<10}")
    print(f"{'-'*20} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
    for r in all_results:
        print(f"{r['model']:<20} {r['params']:<6} {r['HR@5']:<8} {r['NDCG@5']:<8} {r['MRR']:<8} {r['Latency_ms']:<10}")
    
    return all_results


if __name__ == "__main__":
    results = run_experiment()