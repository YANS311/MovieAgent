"""
高并发压力测试：分类测试轻量接口与 Agent 接口
================================================
运行方式:
  python manage.py load_test
  python manage.py load_test --users 50 --requests 5
================================================
"""

import time
import statistics
import concurrent.futures
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'MovieAgent 高并发压力测试（分类测试）'

    def add_arguments(self, parser):
        parser.add_argument('--users', type=int, default=20, help='并发用户数')
        parser.add_argument('--requests', type=int, default=5, help='每用户请求数')

    def handle(self, *args, **options):
        import requests

        n_users = options['users']
        n_requests = options['requests']
        base = 'http://127.0.0.1:8000'

        # 登录
        session = requests.Session()
        try:
            login_page = session.get(f'{base}/login/', timeout=10)
            csrf = session.cookies.get('csrftoken', '')
            resp = session.post(f'{base}/login/', data={
                'username': 'benchmark_user', 'password': 'bench123',
                'csrfmiddlewaretoken': csrf,
            }, headers={'Referer': f'{base}/login/'}, timeout=10, allow_redirects=False)
            self.stdout.write(f'登录状态: {resp.status_code}')
        except Exception as e:
            self.stdout.write(f'登录失败: {e}')

        # ── 测试 1: 轻量接口 ──
        from myapp.models import Movie
        movie_id = Movie.objects.values_list('id', flat=True).first() or 1

        light_endpoints = [
            ('首页', f'{base}/'),
            ('排行榜', f'{base}/rank/'),
            ('电影详情页', f'{base}/movie/{movie_id}/'),
            ('推荐页', f'{base}/recommend/'),
        ]

        self.stdout.write('')
        self.stdout.write('=' * 70)
        self.stdout.write(f'测试 1: 轻量接口并发 ({n_users} 用户 x {n_requests} 请求)')
        self.stdout.write('=' * 70)

        for name, url in light_endpoints:
            results = self._run_concurrent(session, url, n_users, n_requests, method='GET')
            self._print_result(name, results, n_users)

        # ── 测试 2: Agent API ──
        self.stdout.write('')
        self.stdout.write('=' * 70)
        self.stdout.write(f'测试 2: Agent API 并发 (5 用户 x 2 请求)')
        self.stdout.write('=' * 70)

        queries = ['推荐科幻片', '推荐喜剧电影', '推荐悬疑片', '要刺激的', '评分高的']
        agent_results = self._run_concurrent(
            session, f'{base}/agent/api/', 5, 2,
            method='POST', queries=queries
        )
        self._print_result('Agent API', agent_results, 5)

    def _run_concurrent(self, session, url, n_users, n_requests, method='GET', queries=None):
        results = []
        errors = 0

        def single_request(user_id, req_id):
            t0 = time.time()
            try:
                if method == 'POST' and queries:
                    q = queries[(user_id + req_id) % len(queries)]
                    resp = session.post(url, data={'msg': q}, timeout=60)
                else:
                    resp = session.get(url, timeout=30)
                latency = (time.time() - t0) * 1000
                return {
                    'user_id': user_id, 'req_id': req_id,
                    'status': resp.status_code,
                    'latency_ms': round(latency, 1),
                }
            except Exception as e:
                latency = (time.time() - t0) * 1000
                return {
                    'user_id': user_id, 'req_id': req_id,
                    'status': 0, 'latency_ms': round(latency, 1),
                    'error': str(e),
                }

        t_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_users) as executor:
            futures = []
            for u in range(n_users):
                for r in range(n_requests):
                    futures.append(executor.submit(single_request, u, r))
            for f in concurrent.futures.as_completed(futures):
                result = f.result()
                results.append(result)

        t_total = (time.time() - t_start) * 1000
        latencies = [r['latency_ms'] for r in results if r['status'] == 200]
        errors = sum(1 for r in results if r['status'] != 200)

        return {
            'total': len(results),
            'success': len(latencies),
            'errors': errors,
            'latencies': latencies,
            't_total': t_total,
        }

    def _print_result(self, name, data, n_users):
        latencies = data['latencies']
        if not latencies:
            self.stdout.write(f'  {name}: 全部失败')
            return

        sorted_l = sorted(latencies)
        n = len(sorted_l)
        success_rate = n / data['total'] * 100
        throughput = data['total'] / data['t_total'] * 1000

        self.stdout.write(f'  {name}:')
        self.stdout.write(f'    成功率: {success_rate:.0f}%  |  吞吐: {throughput:.1f} req/s  |  '
                         f'Avg: {statistics.mean(sorted_l):.0f}ms  |  '
                         f'P50: {sorted_l[n//2]:.0f}ms  |  '
                         f'P95: {sorted_l[int(n*0.95)]:.0f}ms  |  '
                         f'Max: {sorted_l[-1]:.0f}ms')
