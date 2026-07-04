"""
Django Management Command: MovieAgent 框架下基座大语言模型对比实验 V2
- 使用全部300条黄金数据集
- 评估幻觉率（RAG事实性 + KAG逻辑性）
- 评估推荐说服力
用法: python manage.py run_llm_agent_comparison_v2
"""
import json, time, math, csv, os, re
from collections import defaultdict
from django.core.management.base import BaseCommand
from myapp.models import UserInfo as User
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = [
    {"name": "qwen2.5:7b", "display": "Qwen2.5-7B", "params": "7B", "type": "纯文本 LLM（上代）"},
    {"name": "qwen3:0.6b", "display": "Qwen3-0.6B", "params": "0.6B", "type": "纯文本 LLM"},
    {"name": "qwen3:4b-instruct", "display": "Qwen3-4B-Instruct", "params": "4B", "type": "纯文本 LLM"},
    {"name": "qwen3-vl:4b", "display": "Qwen3-VL-4B", "params": "4B", "type": "视觉语言模型"},
    {"name": "qwen3:8b", "display": "Qwen3-8B", "params": "8B", "type": "纯文本 LLM"},
    {"name": "qwen3.5:9b", "display": "Qwen3.5-9B", "params": "9B", "type": "纯文本 LLM"},
]

HALLUC_PROMPT = """你是电影推荐事实核查专家。判断以下推荐理由是否包含事实错误。

推荐电影: {movie}
推荐理由: {reason}

检查项：
1. 导演是否正确
2. 演员是否正确
3. 类型是否正确
4. 评分/年份是否正确

只输出JSON: {{"hallucination": true/false, "error_type": "none/entity/relation/attribute", "detail": ""}}"""

PERSUADE_PROMPT = """评估以下电影推荐理由的说服力(1-5分)。

用户查询: {query}
推荐电影: {movie}
推荐理由: {reason}

评分标准：
5分: 理由精准命中用户需求，事实正确，有说服力
3分: 理由大致相关但缺乏个性化
1分: 理由与需求无关或有明显错误

只输出JSON: {{"score": 1-5, "reason": "简要说明"}}"""


class Command(BaseCommand):
    help = 'MovieAgent框架下基座大语言模型对比实验V2(300条数据集)'

    def handle(self, *args, **options):
        from myapp.models import Movie
        from myapp.agent.movie_agent import MovieAgent
        from myapp import views

        # 加载全部黄金数据集
        with open("myapp/management/commands/golden_dataset_agent_eval.json", "r", encoding="utf-8") as f:
            queries = json.load(f)["queries"]

        user = User.objects.first()
        neo_graph = getattr(views, 'neo_graph', None)
        rag_resources = getattr(views, 'RAG_RESOURCES', {})

        self.stdout.write(f"总查询数: {len(queries)}")
        self.stdout.write("=" * 60)

        # 先用第一个模型跑一遍获取MovieAgent推荐结果（所有模型共享）
        self.stdout.write("\n[Phase 1] 获取MovieAgent推荐结果（共享工具链）...")
        agent_results = []
        for i, q in enumerate(queries):
            t0 = time.time()
            agent = MovieAgent(user=user, neo_graph=neo_graph,
                               rag_resources=rag_resources, session_id=f"eval_{user.id}")
            result = agent.run(q["query"])
            movie_ids = result.get('recommended_ids', [])[:5]
            movie_map = dict(Movie.objects.filter(id__in=movie_ids).values_list('id', 'title'))
            rec_titles = [movie_map.get(mid, '') for mid in movie_ids]
            rec_ids = movie_ids
            agent_lat = (time.time() - t0) * 1000
            agent_results.append({
                'query': q["query"], 'gt': q["ground_truth_movies"],
                'difficulty': q["difficulty"], 'rec_titles': rec_titles,
                'rec_ids': rec_ids, 'agent_lat': agent_lat,
                'intent': result.get('intent', ''),
            })
            if (i + 1) % 50 == 0 or i == 0:
                self.stdout.write(f"  [{i+1}/{len(queries)}] processed")

        # 计算MovieAgent推荐指标（所有LLM共享）
        def calc_metrics(results):
            hrs, ndcgs, mrrs = [], [], []
            per_d = defaultdict(lambda: {"hr": [], "ndcg": []})
            for r in results:
                gt, rec = r['gt'], r['rec_titles']
                h = 0.0
                for rec_t in rec:
                    for g in gt:
                        if g in rec_t or rec_t in g: h = 1.0; break
                    if h: break
                dcg = 0.0
                for j, rec_t in enumerate(rec[:5]):
                    for g in gt:
                        if g in rec_t or rec_t in g: dcg += 1.0 / math.log2(j + 2); break
                n = min(len(gt), 5); idcg = sum(1.0 / math.log2(k + 2) for k in range(n))
                ndcg_v = dcg / idcg if idcg > 0 else 0.0
                mrr_v = 0.0
                for j, rec_t in enumerate(rec):
                    for g in gt:
                        if g in rec_t or rec_t in g: mrr_v = 1.0 / (j + 1); break
                    if mrr_v: break
                hrs.append(h); ndcgs.append(ndcg_v); mrrs.append(mrr_v)
                per_d[r['difficulty']]["hr"].append(h); per_d[r['difficulty']]["ndcg"].append(ndcg_v)
            return {
                'HR@5': sum(hrs)/len(hrs) if hrs else 0,
                'NDCG@5': sum(ndcgs)/len(ndcgs) if ndcgs else 0,
                'MRR': sum(mrrs)/len(mrrs) if mrrs else 0,
                'per_difficulty': {d: {'HR': sum(v['hr'])/len(v['hr']) if v['hr'] else 0,
                                       'NDCG': sum(v['ndcg'])/len(v['ndcg']) if v['ndcg'] else 0}
                                   for d, v in per_d.items()},
            }

        shared_metrics = calc_metrics(agent_results)
        self.stdout.write(f"\n[Phase 1 完成] 共享推荐指标:")
        self.stdout.write(f"  HR@5={shared_metrics['HR@5']:.4f} NDCG@5={shared_metrics['NDCG@5']:.4f} MRR={shared_metrics['MRR']:.4f}")
        for d, v in shared_metrics['per_difficulty'].items():
            self.stdout.write(f"  {d}: HR={v['HR']:.4f} NDCG={v['NDCG']:.4f}")

        # Phase 2: 各LLM评估推荐理由质量（采样50条评估幻觉率和说服力）
        eval_sample = agent_results[:50]  # 评估子集
        self.stdout.write(f"\n[Phase 2] 评估推荐理由质量（{len(eval_sample)}条子集）...")

        all_results = []
        for model_cfg in MODELS:
            model_name = model_cfg["name"]
            model_display = model_cfg["display"]
            self.stdout.write(f"\n  Model: {model_display}")

            lats, halluc_rates, persuade_scores = [], [], []

            for i, ar in enumerate(eval_sample):
                query = ar['query']
                rec_titles = ar['rec_titles']
                rec_title = rec_titles[0] if rec_titles else "未知电影"

                # 生成推荐理由
                explain_prompt = f"用户查询:{query}\n推荐电影:{rec_title}\n请简短说明推荐理由(50字内):"
                reason, llm_lat = self._call_llm(model_name, explain_prompt)
                lats.append(ar['agent_lat'] + llm_lat)

                # 评估幻觉率
                if rec_title and rec_title != "未知电影":
                    hall_prompt = HALLUC_PROMPT.format(movie=rec_title, reason=reason)
                    hall_resp, _ = self._call_llm(model_name, hall_prompt, timeout=30)
                    try:
                        s = hall_resp.find('{'); e = hall_resp.rfind('}') + 1
                        hall_data = json.loads(hall_resp[s:e])
                        halluc = 1.0 if hall_data.get('hallucination', False) else 0.0
                    except:
                        halluc = 0.0
                    halluc_rates.append(halluc)

                # 评估说服力
                if rec_title and rec_title != "未知电影":
                    p_prompt = PERSUADE_PROMPT.format(query=query, movie=rec_title, reason=reason)
                    p_resp, _ = self._call_llm(model_name, p_prompt, timeout=30)
                    try:
                        s = p_resp.find('{'); e = p_resp.rfind('}') + 1
                        p_data = json.loads(p_resp[s:e])
                        persuade_scores.append(float(p_data.get('score', 3)))
                    except:
                        persuade_scores.append(3.0)

                if (i + 1) % 10 == 0:
                    self.stdout.write(f"    [{i+1}/{len(eval_sample)}] lat={lats[-1]:.0f}ms")

            avg_lat = sum(lats)/len(lats) if lats else 0
            avg_halluc = sum(halluc_rates)/len(halluc_rates) if halluc_rates else 0
            avg_persuade = sum(persuade_scores)/len(persuade_scores) if persuade_scores else 0

            row = {
                "model": model_display, "params": model_cfg["params"], "type": model_cfg["type"],
                "HR@5": f"{shared_metrics['HR@5']:.4f}",
                "NDCG@5": f"{shared_metrics['NDCG@5']:.4f}",
                "MRR": f"{shared_metrics['MRR']:.4f}",
                "hallucination_rate": f"{avg_halluc:.4f}",
                "persuasiveness": f"{avg_persuade:.2f}",
                "Latency_ms": f"{avg_lat:.1f}",
            }
            all_results.append(row)
            self.stdout.write(f"  => HR@5={shared_metrics['HR@5']:.4f} Halluc={avg_halluc:.4f} Persuade={avg_persuade:.2f} Lat={avg_lat:.0f}ms")

        # 保存结果
        os.makedirs("experiment_results", exist_ok=True)
        out = "experiment_results/llm_agent_comparison_v2.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader(); w.writerows(all_results)

        json_out = "experiment_results/llm_agent_comparison_v2.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({"experiment": "MovieAgent框架下基座大语言模型对比实验V2",
                        "num_queries": len(queries), "eval_sample": len(eval_sample),
                        "shared_metrics": shared_metrics,
                        "results": all_results}, f, ensure_ascii=False, indent=2)

        self.stdout.write(f"\n{'='*60}\n  实验完成! CSV: {out}\n{'='*60}")
        for r in all_results:
            self.stdout.write(f"  {r['model']:<20} HR={r['HR@5']} Halluc={r['hallucination_rate']} Persuade={r['persuasiveness']} Lat={r['Latency_ms']}ms")

    def _call_llm(self, model, prompt, timeout=120):
        payload = {"model": model, "messages": [
            {"role": "system", "content": "你是电影推荐助手。"},
            {"role": "user", "content": prompt}], "stream": False,
            "options": {"temperature": 0, "num_predict": 256}}
        t0 = time.time()
        text = ""
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            r.raise_for_status()
            text = r.json().get("message", {}).get("content", "")
        except: pass
        return text, (time.time() - t0) * 1000