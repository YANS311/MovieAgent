"""
系统级性能基准测试：覆盖全部核心业务接口
================================================
运行方式:
  python manage.py system_benchmark
  python manage.py system_benchmark --rounds 5
================================================
"""

import time
import statistics
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'MovieAgent 系统级性能基准测试'

    def add_arguments(self, parser):
        parser.add_argument('--rounds', type=int, default=5, help='每项测试轮次')

    def handle(self, *args, **options):
        import requests

        rounds = options['rounds']
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
            self.stdout.write(f'  登录状态: {resp.status_code}')
            # 验证登录
            check = session.get(f'{base}/agent/chat/', timeout=10)
            self.stdout.write(f'  认证验证: {check.status_code}')
        except Exception as e:
            self.stdout.write(f'  登录失败: {e}')

        # 获取一个有效的电影 ID
        movie_id = self._get_movie_id(session, base)

        # 定义测试接口
        endpoints = [
            ('首页', 'GET', f'{base}/', None),
            ('排行榜', 'GET', f'{base}/rank/', None),
            ('电影详情页', 'GET', f'{base}/movie/{movie_id}/', None),
            ('个性化推荐', 'GET', f'{base}/recommend/', None),
            ('搜索结果', 'GET', f'{base}/search/?q=科幻', None),
            ('视觉搜索', 'GET', f'{base}/search/visual/?q=科幻海报', None),
            ('知识图谱页面', 'GET', f'{base}/agent/kg/', None),
            ('Agent对话页面', 'GET', f'{base}/agent/chat/', None),
            ('推荐API(传统)', 'POST', f'{base}/recommend/explain/', {'movie_id': movie_id}),
            ('Agent API', 'POST_FORM', f'{base}/agent/api/', {'msg': '推荐科幻片'}),
            ('知识图谱API', 'GET', f'{base}/agent/kg/query/?q=星际穿越&mode=query', None),
        ]

        results = {}
        for name, method, url, data in endpoints:
            self.stdout.write(f'  测试: {name}...')
            latencies = []
            for _ in range(rounds):
                t0 = time.time()
                try:
                    if method == 'GET':
                        resp = session.get(url, timeout=60)
                    elif method == 'POST_FORM':
                        resp = session.post(url, data=data, timeout=60)
                    else:
                        resp = session.post(url, json=data, timeout=60)
                    latency = (time.time() - t0) * 1000
                    if resp.status_code == 200:
                        latencies.append(latency)
                except Exception:
                    pass
            if latencies:
                results[name] = self._stats(latencies)

        self._print_table(results)

    def _get_movie_id(self, session, base):
        try:
            from myapp.models import Movie
            m = Movie.objects.first()
            return m.id if m else 1
        except Exception:
            return 1

    def _stats(self, latencies):
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
        self.stdout.write('')
        self.stdout.write('=' * 80)
        self.stdout.write('表 6-7 核心业务接口响应时间统计')
        self.stdout.write('=' * 80)
        header = f"{'接口':<16} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10}"
        self.stdout.write(header)
        self.stdout.write('-' * 80)
        for name, s in results.items():
            row = f"{name:<16} {s['avg']:>10.1f} {s['p50']:>10.1f} {s['p95']:>10.1f} {s['min']:>10.1f} {s['max']:>10.1f}"
            self.stdout.write(row)
        self.stdout.write('=' * 80)
