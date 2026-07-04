"""
Redis 缓存基准测试：量化 Cache 加速效果
================================================
运行方式:
  python manage.py benchmark_redis
  python manage.py benchmark_redis --rounds 100
================================================
"""

import time
import statistics
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Redis 缓存基准测试（GET/SET 微基准 + ORM vs Cache 对比）'

    def add_arguments(self, parser):
        parser.add_argument('--rounds', type=int, default=100, help='每项测试轮次')

    def handle(self, *args, **options):
        from django.core.cache import cache

        rounds = options['rounds']

        self.stdout.write('=' * 70)
        self.stdout.write('Redis 缓存基准测试')
        self.stdout.write('=' * 70)

        results = {}

        # ── 1. Redis SET 延迟 ──
        self.stdout.write(f'\n[1/6] Redis SET 延迟 ({rounds} 轮)')
        latencies = []
        for i in range(rounds):
            t0 = time.perf_counter()
            cache.set(f'_bench_set_{i}', {'data': 'x' * 100, 'idx': i}, 60)
            latencies.append((time.perf_counter() - t0) * 1000)
        results['Redis SET'] = self._stats(latencies)

        # ── 2. Redis GET 延迟（命中）──
        self.stdout.write(f'[2/6] Redis GET 延迟-命中 ({rounds} 轮)')
        # 预写入
        for i in range(rounds):
            cache.set(f'_bench_get_{i}', {'data': 'x' * 100, 'idx': i}, 60)
        latencies = []
        for i in range(rounds):
            t0 = time.perf_counter()
            cache.get(f'_bench_get_{i}')
            latencies.append((time.perf_counter() - t0) * 1000)
        results['Redis GET (命中)'] = self._stats(latencies)

        # ── 3. Redis GET 延迟（未命中）──
        self.stdout.write(f'[3/6] Redis GET 延迟-未命中 ({rounds} 轮)')
        latencies = []
        for i in range(rounds):
            t0 = time.perf_counter()
            cache.get(f'_bench_miss_{i}')
            latencies.append((time.perf_counter() - t0) * 1000)
        results['Redis GET (未命中)'] = self._stats(latencies)

        # ── 4. ORM 单条查询延迟 ──
        self.stdout.write(f'[4/6] ORM 单条查询延迟 ({rounds} 轮)')
        try:
            from myapp.models import Movie
            movie_ids = list(Movie.objects.values_list('id', flat=True)[:rounds])
            if movie_ids:
                latencies = []
                for mid in movie_ids:
                    t0 = time.perf_counter()
                    list(Movie.objects.filter(id=mid).values('id', 'title', 'score'))
                    latencies.append((time.perf_counter() - t0) * 1000)
                results['ORM 单条查询'] = self._stats(latencies)
            else:
                results['ORM 单条查询'] = {'avg': 0, 'p50': 0, 'p95': 0, 'min': 0, 'max': 0, 'n': 0}
        except Exception as e:
            self.stdout.write(f'  ORM 查询失败: {e}')
            results['ORM 单条查询'] = {'avg': 0, 'p50': 0, 'p95': 0, 'min': 0, 'max': 0, 'n': 0}

        # ── 5. Cache 读取 vs ORM 读取（同一数据）──
        self.stdout.write(f'[5/6] Cache vs ORM 对比 ({rounds} 轮)')
        try:
            from myapp.models import Movie
            test_movies = list(Movie.objects.values('id', 'title', 'score')[:20])
            cache_key = '_bench_movie_list'

            # ORM 直读
            orm_latencies = []
            for _ in range(rounds):
                t0 = time.perf_counter()
                list(Movie.objects.values('id', 'title', 'score')[:20])
                orm_latencies.append((time.perf_counter() - t0) * 1000)

            # Cache 读取
            cache.set(cache_key, test_movies, 300)
            cache_latencies = []
            for _ in range(rounds):
                t0 = time.perf_counter()
                cache.get(cache_key)
                cache_latencies.append((time.perf_counter() - t0) * 1000)

            results['ORM 列表查询'] = self._stats(orm_latencies)
            results['Cache 列表查询'] = self._stats(cache_latencies)
        except Exception as e:
            self.stdout.write(f'  对比测试失败: {e}')

        # ── 6. Agent Memory 槽位读写 ──
        self.stdout.write(f'[6/6] Agent Memory 槽位读写 ({rounds} 轮)')
        write_latencies = []
        read_latencies = []
        for i in range(rounds):
            key = f'_bench_memory_{i}'
            slots = {'genre': '科幻', 'score_min': 8.0, 'year_min': 2020}
            t0 = time.perf_counter()
            cache.set(key, slots, 3600)
            write_latencies.append((time.perf_counter() - t0) * 1000)

            t0 = time.perf_counter()
            cache.get(key)
            read_latencies.append((time.perf_counter() - t0) * 1000)
        results['Memory 写入'] = self._stats(write_latencies)
        results['Memory 读取'] = self._stats(read_latencies)

        # 清理
        for i in range(rounds):
            cache.delete(f'_bench_set_{i}')
            cache.delete(f'_bench_get_{i}')
            cache.delete(f'_bench_memory_{i}')
        cache.delete('_bench_movie_list')

        self._print_table(results)

        # 计算加速比
        if results.get('ORM 列表查询', {}).get('avg', 0) > 0:
            orm_avg = results['ORM 列表查询']['avg']
            cache_avg = results.get('Cache 列表查询', {}).get('avg', 1)
            speedup = orm_avg / cache_avg if cache_avg > 0 else 0
            self.stdout.write(f'\n加速比: ORM({orm_avg:.3f}ms) / Cache({cache_avg:.3f}ms) = {speedup:.0f}x')

    def _stats(self, latencies):
        if not latencies:
            return {'avg': 0, 'p50': 0, 'p95': 0, 'min': 0, 'max': 0, 'n': 0}
        sorted_l = sorted(latencies)
        n = len(sorted_l)
        return {
            'avg': round(statistics.mean(sorted_l), 4),
            'p50': round(sorted_l[n // 2], 4),
            'p95': round(sorted_l[int(n * 0.95)], 4),
            'min': round(sorted_l[0], 4),
            'max': round(sorted_l[-1], 4),
            'n': n,
        }

    def _print_table(self, results):
        self.stdout.write('\n' + '=' * 80)
        self.stdout.write('Redis 缓存基准测试结果')
        self.stdout.write('=' * 80)
        header = f"{'场景':<20} {'Avg(ms)':>10} {'P50(ms)':>10} {'P95(ms)':>10} {'Min(ms)':>10} {'Max(ms)':>10} {'N':>6}"
        self.stdout.write(header)
        self.stdout.write('-' * 80)
        for name, s in results.items():
            row = f"{name:<20} {s['avg']:>10.4f} {s['p50']:>10.4f} {s['p95']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f} {s['n']:>6}"
            self.stdout.write(row)
        self.stdout.write('=' * 80)
