"""
性能基准测试：统计各功能模块的端到端延迟
================================================
运行方式:
  python manage.py benchmark_latency
  python manage.py benchmark_latency --rounds 5
================================================
"""

import os
import sys
import time
import statistics
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model

User = get_user_model()


class Command(BaseCommand):
    help = 'MovieAgent 性能基准测试'

    def add_arguments(self, parser):
        parser.add_argument('--rounds', type=int, default=3, help='每项测试轮次')

    def handle(self, *args, **options):
        rounds = options['rounds']
        user = User.objects.first()
        if not user:
            self.stderr.write('No user found')
            return

        # 加载资源
        self.stdout.write('加载 RAG 资源...')
        rag_resources = self._load_rag()
        self.stdout.write('加载 Neo4j...')
        neo_graph = self._load_neo()

        results = {}

        # ── 1. Agent 端到端（简单查询）──
        self.stdout.write('\n[1/5] Agent 端到端（简单查询）')
        queries_simple = ['推荐科幻片', '推荐喜剧电影', '推荐悬疑片']
        results['Agent_简单查询'] = self._bench_agent(user, rag_resources, neo_graph, queries_simple, rounds)

        # ── 2. Agent 端到端（复杂查询）──
        self.stdout.write('[2/5] Agent 端到端（复杂查询）')
        queries_complex = ['类似《星际穿越》的科幻片', '评分8分以上的悬疑片', '诺兰导演的科幻电影']
        results['Agent_复杂查询'] = self._bench_agent(user, rag_resources, neo_graph, queries_complex, rounds)

        # ── 3. Agent 端到端（多轮对话）──
        self.stdout.write('[3/5] Agent 端到端（多轮对话）')
        results['Agent_多轮对话'] = self._bench_multiturn(user, rag_resources, neo_graph, rounds)

        # ── 4. RAG 向量召回 ──
        self.stdout.write('[4/5] RAG 向量召回')
        results['RAG_向量召回'] = self._bench_rag(rag_resources, queries_simple + queries_complex, rounds)

        # ── 5. MAAN 精排 ──
        self.stdout.write('[5/5] MAAN 精排')
        results['MAAN_精排'] = self._bench_maan(user, rag_resources, queries_simple, rounds)

        # 输出结果
        self._print_table(results)

    def _load_rag(self):
        from django.conf import settings
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS
        index_path = os.path.join(settings.BASE_DIR, 'faiss_movie_index')
        if not os.path.exists(index_path):
            return {}
        local_bge_path = os.path.join(settings.BASE_DIR, 'local_models', 'bge-small-zh-v1.5')
        snapshot_dir = os.path.join(local_bge_path, 'snapshots')
        model_name = 'BAAI/bge-small-zh-v1.5'
        if os.path.isdir(snapshot_dir):
            for h in os.listdir(snapshot_dir):
                candidate = os.path.join(snapshot_dir, h)
                if os.path.exists(os.path.join(candidate, 'config.json')):
                    model_name = candidate
                    break
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        return {'vectorstore': FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True), 'embeddings': embeddings}

    def _load_neo(self):
        try:
            from py2neo import Graph
            from django.conf import settings
            return Graph(getattr(settings, 'NEO4J_URI', 'bolt://localhost:7687'),
                         auth=(getattr(settings, 'NEO4J_USER', 'neo4j'), getattr(settings, 'NEO4J_PASSWORD', '')))
        except Exception:
            return None

    def _bench_agent(self, user, rag, neo, queries, rounds):
        from myapp.agent.movie_agent import MovieAgent
        latencies = []
        for _ in range(rounds):
            for q in queries:
                agent = MovieAgent(user=user, rag_resources=rag, neo_graph=neo)
                t0 = time.time()
                agent.run(q)
                latencies.append((time.time() - t0) * 1000)
        return self._stats(latencies)

    def _bench_multiturn(self, user, rag, neo, rounds):
        from myapp.agent.movie_agent import MovieAgent
        turns = ['推荐科幻片', '要硬科幻', '评分高一点', '不要太老的']
        latencies = []
        for _ in range(rounds):
            agent = MovieAgent(user=user, rag_resources=rag, neo_graph=neo)
            for q in turns:
                t0 = time.time()
                agent.run(q)
                latencies.append((time.time() - t0) * 1000)
        return self._stats(latencies)

    def _bench_rag(self, rag, queries, rounds):
        if not rag or not rag.get('vectorstore'):
            return {'avg': 0, 'p50': 0, 'p95': 0, 'min': 0, 'max': 0}
        vs = rag['vectorstore']
        latencies = []
        for _ in range(rounds):
            for q in queries:
                t0 = time.time()
                vs.similarity_search(q, k=10)
                latencies.append((time.time() - t0) * 1000)
        return self._stats(latencies)

    def _bench_maan(self, user, rag, queries, rounds):
        from myapp.agent.movie_agent import MovieAgent, MAANRerankTool
        latencies = []
        for _ in range(rounds):
            for q in queries:
                agent = MovieAgent(user=user, rag_resources=rag)
                # 先召回候选
                vs = rag.get('vectorstore')
                if not vs:
                    continue
                docs = vs.similarity_search(q, k=20)
                candidates = [{'movie_id': d.metadata.get('movie_id', 0), 'title': d.metadata.get('title', '')} for d in docs]
                tool = MAANRerankTool()
                t0 = time.time()
                tool.execute(candidates=candidates, user=user, top_k=5)
                latencies.append((time.time() - t0) * 1000)
        return self._stats(latencies)

    def _stats(self, latencies):
        if not latencies:
            return {'avg': 0, 'p50': 0, 'p95': 0, 'min': 0, 'max': 0, 'n': 0}
        sorted_l = sorted(latencies)
        n = len(sorted_l)
        return {
            'avg': round(statistics.mean(sorted_l), 1),
            'p50': round(sorted_l[n // 2], 1),
            'p95': round(sorted_l[int(n * 0.95)], 1),
            'min': round(sorted_l[0], 1),
            'max': round(sorted_l[-1], 1),
            'n': n,
        }

    def _print_table(self, results):
        self.stdout.write('\n' + '=' * 80)
        self.stdout.write('MovieAgent 性能基准测试结果')
        self.stdout.write('=' * 80)
        header = f"{'功能模块':<20} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'N':>6}"
        self.stdout.write(header)
        self.stdout.write('-' * 80)
        for name, s in results.items():
            row = f"{name:<20} {s['avg']:>10.1f} {s['p50']:>10.1f} {s['p95']:>10.1f} {s['min']:>10.1f} {s['max']:>10.1f} {s['n']:>6}"
            self.stdout.write(row)
        self.stdout.write('=' * 80)
