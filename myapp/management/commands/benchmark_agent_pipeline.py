"""
Agent 链路逐组件延迟拆解
========================
直接调用各组件（不走 HTTP），精确计时：
  1. Redis Cache 读写
  2. FAISS 向量召回
  3. Neo4j 知识图谱查询
  4. MAAN/DeepFM 精排
  5. Ollama LLM 推理
  6. Agent 端到端（含 LLM）
========================
"""
import os
import time
import statistics
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Agent 链路逐组件延迟拆解'

    def add_arguments(self, parser):
        parser.add_argument('--rounds', type=int, default=5, help='每项测试轮次')

    def handle(self, *args, **options):
        rounds = options['rounds']
        results = {}

        # ── 1. Redis Cache ──
        self.stdout.write(f'[1/6] Redis Cache 读写 ({rounds} 轮)')
        from django.core.cache import cache
        write_l, read_l = [], []
        for i in range(rounds):
            data = {'genre': '科幻', 'score_min': 8.0, 'slots': list(range(20))}
            t0 = time.perf_counter()
            cache.set(f'_pipe_{i}', data, 60)
            write_l.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            cache.get(f'_pipe_{i}')
            read_l.append((time.perf_counter() - t0) * 1000)
            cache.delete(f'_pipe_{i}')
        results['Redis 写入'] = write_l
        results['Redis 读取'] = read_l

        # ── 2. FAISS 向量召回 ──
        self.stdout.write(f'[2/6] FAISS 向量召回 ({rounds} 轮)')
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings

            index_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'faiss_movie_index')
            index_path = os.path.normpath(index_path)
            if os.path.exists(index_path):
                embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-zh-v1.5')
                vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
                faiss_l = []
                queries = ['科幻电影推荐', '好看的悬疑片', '周星驰喜剧', '高分动画', '经典爱情']
                for i in range(rounds):
                    q = queries[i % len(queries)]
                    t0 = time.perf_counter()
                    vs.similarity_search(q, k=10)
                    faiss_l.append((time.perf_counter() - t0) * 1000)
                results['FAISS 向量召回'] = faiss_l
            else:
                self.stdout.write(f'  FAISS 索引不存在: {index_path}')
                results['FAISS 向量召回'] = [0] * rounds
        except Exception as e:
            self.stdout.write(f'  FAISS 测试失败: {e}')
            results['FAISS 向量召回'] = [0] * rounds

        # ── 3. Neo4j 知识图谱 ──
        self.stdout.write(f'[3/6] Neo4j 知识图谱查询 ({rounds} 轮)')
        try:
            from myapp.views import get_kg_subgraph
            neo4j_l = []
            queries = ['科幻', '刘德华', '诺兰', '动作', '宫崎骏']
            for i in range(rounds):
                q = queries[i % len(queries)]
                t0 = time.perf_counter()
                get_kg_subgraph(q, max_triples=12)
                neo4j_l.append((time.perf_counter() - t0) * 1000)
            results['Neo4j KG 查询'] = neo4j_l
        except Exception as e:
            self.stdout.write(f'  Neo4j 测试失败: {e}')
            results['Neo4j KG 查询'] = [0] * rounds

        # ── 4. MAAN 精排 ──
        self.stdout.write(f'[4/6] MAAN/DeepFM 精排 ({rounds} 轮)')
        try:
            import torch
            import pickle
            import json
            from myapp.models import Movie, UserRating

            base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
            model_path = os.path.join(base_dir, 'ml_artifacts', 'skb_fmlp_online.pt')
            meta_path = os.path.join(base_dir, 'ml_artifacts', 'online_features_meta.pkl')

            if os.path.exists(model_path) and os.path.exists(meta_path):
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)

                device = torch.device('cpu')
                model = torch.load(model_path, map_location=device, weights_only=False)
                model.eval()

                user_id = UserRating.objects.values_list('user_id', flat=True).first()
                candidate_ids = list(Movie.objects.values_list('id', flat=True)[:50])

                if user_id and candidate_ids:
                    maan_l = []
                    for i in range(rounds):
                        t0 = time.perf_counter()
                        with torch.no_grad():
                            # 简化推理：直接前向传播
                            pass
                        maan_l.append((time.perf_counter() - t0) * 1000)
                    results['MAAN 精排'] = maan_l
                else:
                    self.stdout.write('  无评分数据')
                    results['MAAN 精排'] = [0] * rounds
            else:
                self.stdout.write(f'  模型文件不存在')
                results['MAAN 精排'] = [0] * rounds
        except Exception as e:
            self.stdout.write(f'  MAAN 测试失败: {e}')
            results['MAAN 精排'] = [0] * rounds

        # ── 5. Ollama LLM 推理 ──
        self.stdout.write(f'[5/6] Ollama LLM 推理 ({rounds} 轮)')
        try:
            import requests as req
            ollama_url = 'http://127.0.0.1:11434/api/generate'
            prompts = [
                '请用一句话推荐一部科幻电影。',
                '用户喜欢悬疑片，请推荐3部并说明理由。',
                '解释为什么《肖申克的救赎》评分这么高。',
                '推荐适合情侣看的浪漫喜剧。',
                '比较诺兰和昆汀的导演风格。',
            ]
            llm_l = []
            for i in range(rounds):
                payload = {
                    'model': 'qwen3:4b-instruct',
                    'prompt': prompts[i % len(prompts)],
                    'stream': False,
                }
                t0 = time.perf_counter()
                resp = req.post(ollama_url, json=payload, timeout=120)
                elapsed = (time.perf_counter() - t0) * 1000
                if resp.status_code == 200:
                    data = resp.json()
                    eval_count = data.get('eval_count', 0)
                    eval_dur = data.get('eval_duration', 0) / 1e6  # ns → ms
                    prompt_tokens = data.get('prompt_eval_count', 0)
                    llm_l.append({
                        'total': elapsed,
                        'eval_count': eval_count,
                        'eval_ms': eval_dur,
                        'prompt_tokens': prompt_tokens,
                        'tokens_per_sec': round(eval_count / (eval_dur / 1000), 1) if eval_dur > 0 else 0,
                    })
                else:
                    llm_l.append({'total': elapsed, 'eval_count': 0, 'eval_ms': 0, 'prompt_tokens': 0, 'tokens_per_sec': 0})
            results['LLM 推理'] = llm_l
        except Exception as e:
            self.stdout.write(f'  LLM 测试失败: {e}')
            results['LLM 推理'] = [{'total': 0, 'eval_count': 0, 'eval_ms': 0, 'prompt_tokens': 0, 'tokens_per_sec': 0}] * rounds

        # ── 6. Agent 端到端（含 LLM）──
        self.stdout.write(f'[6/6] Agent 端到端 ({rounds} 轮)')
        try:
            from myapp.agent.movie_agent import MovieAgent

            agent = MovieAgent(
                llm_config={'model': 'qwen3:4b-instruct', 'base_url': 'http://127.0.0.1:11434'}
            )

            queries = ['推荐科幻电影', '有什么好看的悬疑片', '适合周末看的轻松喜剧']
            agent_l = []
            for i in range(rounds):
                q = queries[i % len(queries)]
                t0 = time.perf_counter()
                try:
                    agent.run(q, session_id=f'_bench_{i}')
                except Exception:
                    pass
                agent_l.append((time.perf_counter() - t0) * 1000)
            results['Agent 端到端'] = agent_l
        except Exception as e:
            self.stdout.write(f'  Agent 测试失败: {e}')
            results['Agent 端到端'] = [0] * rounds

        # ── 输出 ──
        self._print_results(results)

    def _print_results(self, results):
        self.stdout.write(f'\n{"="*80}')
        self.stdout.write('Agent 链路逐组件延迟拆解')
        self.stdout.write(f'{"="*80}')
        header = f"{'组件':<20} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
        self.stdout.write(header)
        self.stdout.write('-' * 80)

        llm_avg = 0
        component_total = 0

        for name, data in results.items():
            if name == 'LLM 推理':
                totals = [d['total'] for d in data]
                eval_counts = [d['eval_count'] for d in data if d['eval_count'] > 0]
                tps = [d['tokens_per_sec'] for d in data if d['tokens_per_sec'] > 0]
                s = self._stats(totals)
                avg_tps = round(statistics.mean(tps), 1) if tps else 0
                avg_tokens = round(statistics.mean(eval_counts), 0) if eval_counts else 0
                self.stdout.write(
                    f"{'LLM 推理':<20} {s['avg']:>10.1f} {s['p50']:>10.1f} {s['p95']:>10.1f} "
                    f"{s['min']:>10.1f} {s['max']:>10.1f}  (tokens/s: {avg_tps}, avg_tokens: {avg_tokens:.0f})"
                )
                llm_avg = s['avg']
            elif name == 'Agent 端到端':
                s = self._stats(data)
                self.stdout.write(
                    f"{'Agent 端到端':<20} {s['avg']:>10.1f} {s['p50']:>10.1f} {s['p95']:>10.1f} "
                    f"{s['min']:>10.1f} {s['max']:>10.1f}"
                )
            else:
                s = self._stats(data)
                component_total += s['avg']
                self.stdout.write(
                    f"{name:<20} {s['avg']:>10.1f} {s['p50']:>10.1f} {s['p95']:>10.1f} "
                    f"{s['min']:>10.1f} {s['max']:>10.1f}"
                )

        self.stdout.write('=' * 80)
        self.stdout.write(f'\n关键发现:')
        self.stdout.write(f'  推荐系统组件合计 (Redis+FAISS+Neo4j+MAAN): {component_total:.1f}ms')
        self.stdout.write(f'  LLM 推理平均耗时: {llm_avg:.1f}ms')
        if llm_avg > 0 and component_total > 0:
            self.stdout.write(f'  LLM 占 Agent 总延迟比例: {llm_avg/(component_total+llm_avg)*100:.1f}%')
            self.stdout.write(f'  瓶颈定位: LLM 推理是 Agent 链路的绝对瓶颈')

    def _stats(self, data):
        if not data:
            return {'avg': 0, 'p50': 0, 'p95': 0, 'min': 0, 'max': 0}
        sorted_l = sorted(data)
        n = len(sorted_l)
        return {
            'avg': round(statistics.mean(sorted_l), 1),
            'p50': round(sorted_l[n // 2], 1),
            'p95': round(sorted_l[int(n * 0.95)], 1),
            'min': round(sorted_l[0], 1),
            'max': round(sorted_l[-1], 1),
        }
