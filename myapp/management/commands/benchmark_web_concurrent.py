"""
Web 接口高并发压测（不含 Agent LLM 接口）
==========================================
测试首页/排行榜/详情页/推荐页/搜索/KG 在 100/500/1000 并发下的表现
==========================================
"""
import time
import statistics
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Web 接口高并发压测（不含 Agent）'

    def add_arguments(self, parser):
        parser.add_argument('--base-url', default='http://127.0.0.1:8000', help='服务地址')
        parser.add_argument('--rounds', type=int, default=3, help='每级并发重复轮次')

    def handle(self, *args, **options):
        base = options['base_url']
        rounds = options['rounds']
        session = requests.Session()

        # 登录
        login_url = f'{base}/login/'
        csrf = session.get(login_url)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(csrf.text, 'html.parser')
        token = soup.find('input', {'name': 'csrfmiddlewaretoken'})['value']
        session.post(login_url, data={
            'csrfmiddlewaretoken': token,
            'username': 'benchmark_user',
            'password': 'testpass123',
        }, allow_redirects=True)

        endpoints = {
            '首页': '/',
            '排行榜': '/rank/',
            '电影详情': '/movie/1/',
            '推荐页': '/recommend/',
            '搜索': '/search/?q=科幻',
            '知识图谱': '/agent/kg/',
        }

        concurrency_levels = [10, 50, 100, 200, 500]

        all_results = {}

        for c_level in concurrency_levels:
            self.stdout.write(f'\n{"="*70}')
            self.stdout.write(f'并发数: {c_level}')
            self.stdout.write(f'{"="*70}')

            for name, path in endpoints.items():
                url = f'{base}{path}'
                latencies = []
                errors = 0

                for r in range(rounds):
                    def fetch(u=url):
                        try:
                            t0 = time.perf_counter()
                            resp = session.get(u, timeout=30)
                            return (time.perf_counter() - t0) * 1000, resp.status_code
                        except Exception:
                            return -1, 0

                    with ThreadPoolExecutor(max_workers=c_level) as pool:
                        futures = [pool.submit(fetch) for _ in range(c_level)]
                        for f in as_completed(futures):
                            lat, status = f.result()
                            if lat > 0 and 200 <= status < 400:
                                latencies.append(lat)
                            else:
                                errors += 1

                if latencies:
                    sorted_l = sorted(latencies)
                    n = len(sorted_l)
                    result = {
                        'avg': round(statistics.mean(sorted_l), 1),
                        'p50': round(sorted_l[n // 2], 1),
                        'p95': round(sorted_l[int(n * 0.95)], 1),
                        'max': round(sorted_l[-1], 1),
                        'success': n,
                        'fail': errors,
                        'qps': round(n / (sum(sorted_l) / 1000 / n), 1) if sum(sorted_l) > 0 else 0,
                    }
                    all_results[(name, c_level)] = result
                    self.stdout.write(
                        f'  {name:<12} 成功:{n:>4}  失败:{errors:>3}  '
                        f'Avg:{result["avg"]:>7.1f}ms  P50:{result["p50"]:>7.1f}ms  '
                        f'P95:{result["p95"]:>7.1f}ms  QPS:{result["qps"]:>6.1f}'
                    )
                else:
                    all_results[(name, c_level)] = {'avg': 0, 'p50': 0, 'p95': 0, 'max': 0, 'success': 0, 'fail': errors, 'qps': 0}
                    self.stdout.write(f'  {name:<12} 全部失败 ({errors})')

        # 汇总表
        self.stdout.write(f'\n{"="*90}')
        self.stdout.write('表 6-X Web 接口高并发性能汇总')
        self.stdout.write(f'{"="*90}')
        header = f"{'接口':<12} {'并发':>6} {'成功率':>8} {'QPS':>8} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10}"
        self.stdout.write(header)
        self.stdout.write('-' * 90)
        for (name, c_level), r in all_results.items():
            total = r['success'] + r['fail']
            rate = f"{r['success']/total*100:.0f}%" if total > 0 else "0%"
            self.stdout.write(
                f"{name:<12} {c_level:>6} {rate:>8} {r['qps']:>8.1f} "
                f"{r['avg']:>10.1f} {r['p50']:>10.1f} {r['p95']:>10.1f}"
            )
        self.stdout.write('=' * 90)
