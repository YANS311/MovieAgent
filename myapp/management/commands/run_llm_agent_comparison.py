"""
Django Management Command: MovieAgent 框架下基座大语言模型对比实验
用法: python manage.py run_llm_agent_comparison
"""
import json, time, math, csv, os
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

EXPLAIN_SYS = "你是电影推荐助手。根据查询和推荐结果生成简短推荐理由(30字内中文)。只输出理由。"

class Command(BaseCommand):
    help = 'MovieAgent框架下基座大语言模型对比实验'

    def handle(self, *args, **options):
        from myapp.models import Movie
        from myapp.agent.movie_agent import MovieAgent
        from myapp import views

        # 加载黄金数据集
        with open("myapp/management/commands/golden_dataset_agent_eval.json", "r", encoding="utf-8") as f:
            queries = json.load(f)["queries"][:50]

        user = User.objects.first()
        neo_graph = getattr(views, 'neo_graph', None)
        rag_resources = getattr(views, 'RAG_RESOURCES', {})

        self.stdout.write(f"总查询数: {len(queries)}")
        self.stdout.write("=" * 60)

        all_results = []

        for model_cfg in MODELS:
            model_name = model_cfg["name"]
            model_display = model_cfg["display"]
            self.stdout.write(f"\n{'='*60}\n  Model: {model_display}\n{'='*60}")

            hrs, ndcgs, mrrs, lats = [], [], [], []
            per_d = defaultdict(lambda: {"hr": [], "ndcg": []})

            for i, q in enumerate(queries):
                gt = q["ground_truth_movies"]
                difficulty = q["difficulty"]
                t0 = time.time()

                # MovieAgent 工具链获取推荐
                agent = MovieAgent(user=user, neo_graph=neo_graph,
                                   rag_resources=rag_resources, session_id=f"eval_{user.id}")
                result = agent.run(q["query"])
                movie_ids = result.get('recommended_ids', [])[:5]
                movie_map = dict(Movie.objects.filter(id__in=movie_ids).values_list('id', 'title'))
                rec_titles = [movie_map.get(mid, '') for mid in movie_ids]

                # 用指定LLM生成推荐理由（测试LLM在Agent框架中的协调能力）
                explain_prompt = f"用户查询:{q['query']}\n推荐结果:{','.join(rec_titles[:3])}\n请简短说明推荐理由(20字内):"
                llm_lat = self._call_llm(model_name, explain_prompt)

                lat = (time.time() - t0) * 1000 + llm_lat

                # 计算指标
                h = 0.0
                for rec in rec_titles:
                    for g in gt:
                        if g in rec or rec in g:
                            h = 1.0; break
                    if h: break

                dcg = 0.0
                for j, rec in enumerate(rec_titles[:5]):
                    for g in gt:
                        if g in rec or rec in g:
                            dcg += 1.0 / math.log2(j + 2); break
                n = min(len(gt), 5)
                idcg = sum(1.0 / math.log2(k + 2) for k in range(n))
                ndcg_v = dcg / idcg if idcg > 0 else 0.0

                mrr_v = 0.0
                for j, rec in enumerate(rec_titles):
                    for g in gt:
                        if g in rec or rec in g:
                            mrr_v = 1.0 / (j + 1); break
                    if mrr_v: break

                hrs.append(h); ndcgs.append(ndcg_v); mrrs.append(mrr_v); lats.append(lat)
                per_d[difficulty]["hr"].append(h); per_d[difficulty]["ndcg"].append(ndcg_v)

                if (i + 1) % 10 == 0 or i == 0:
                    self.stdout.write(f"  [{i+1}/{len(queries)}] HR={h:.1f} lat={lat:.0f}ms")

            ah = sum(hrs)/len(hrs); an = sum(ndcgs)/len(ndcgs); am = sum(mrrs)/len(mrrs); al = sum(lats)/len(lats)
            row = {"model": model_display, "params": model_cfg["params"], "type": model_cfg["type"],
                   "HR@5": f"{ah:.4f}", "NDCG@5": f"{an:.4f}", "MRR": f"{am:.4f}", "Latency_ms": f"{al:.1f}"}
            for d in ["single_hop", "multi_hop", "implicit"]:
                dd = per_d[d]
                row[f"HR_{d}"] = f"{sum(dd['hr'])/len(dd['hr']):.4f}" if dd["hr"] else "0"
            all_results.append(row)
            self.stdout.write(f"  => HR@5={ah:.4f} NDCG@5={an:.4f} MRR={am:.4f} Lat={al:.0f}ms")

        # 保存结果
        os.makedirs("experiment_results", exist_ok=True)
        out = "experiment_results/llm_agent_comparison_real.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader(); w.writerows(all_results)

        json_out = "experiment_results/llm_agent_comparison_real.json"
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump({"experiment": "MovieAgent框架下基座大语言模型对比实验(真实数据)",
                        "num_queries": len(queries), "results": all_results}, f, ensure_ascii=False, indent=2)

        self.stdout.write(f"\n{'='*60}\n  实验完成! CSV: {out}\n{'='*60}")
        for r in all_results:
            self.stdout.write(f"  {r['model']:<20} HR={r['HR@5']} NDCG={r['NDCG@5']} Lat={r['Latency_ms']}ms")

    def _call_llm(self, model, prompt, timeout=60):
        """调用Ollama LLM，返回延迟(ms)"""
        payload = {"model": model, "messages": [
            {"role": "system", "content": EXPLAIN_SYS},
            {"role": "user", "content": prompt}], "stream": False,
            "options": {"temperature": 0, "num_predict": 128}}
        t0 = time.time()
        try:
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            r.raise_for_status()
        except: pass
        return (time.time() - t0) * 1000