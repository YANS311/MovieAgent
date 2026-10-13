"""P2 stdlib/AST/Fake tests. No app setup, real services or model imports."""

import ast
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
BASELINE = 'ccbc04d9e2e3208c906a32fba7da8dd4c7b24d25'


def load_file(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclass metadata refers to its declaring module.
    spec.loader.exec_module(module)
    return module


class P2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bench = load_file('p2_microbench', 'benchmarks/redis_cache_microbench.py')
        cls.metrics_module = load_file('p2_cache_metrics', 'myapp/utils/cache_metrics.py')
        cls.keys = load_file('p2_cache_keys', 'myapp/utils/cache_keys.py')
        cls.access = load_file('p2_cache_access', 'myapp/utils/cache_access.py')

    def observer(self, enabled=True, **kwargs):
        metrics = self.metrics_module.CacheMetrics(enabled=enabled, **kwargs)
        backend = self.bench.FakeCache()
        proxy = self.metrics_module.ObservedBusinessCache(backend, metrics, 'hybrid')
        return metrics, backend, proxy

    def hydrate(self, count=30):
        result = self.bench.synthetic_result(count)
        scope = self.bench.production_functions()
        manager = self.bench.FakeManager(result[0])
        fn = self.bench.restore_function(scope, manager)
        return result, scope['build_payload'](result), fn, manager

    def test_new_helpers_and_benchmark_only_import_stdlib(self):
        for name in ('myapp/utils/cache_metrics.py', 'benchmarks/redis_cache_microbench.py',
                     'tests/test_redis_cache_p2_isolated.py'):
            tree = ast.parse((ROOT / name).read_text())
            for n in ast.walk(tree):
                if isinstance(n, ast.ImportFrom):
                    self.assertIn(n.module.split('.')[0], sys.stdlib_module_names)
                if isinstance(n, ast.Import):
                    for alias in n.names:
                        self.assertIn(alias.name.split('.')[0], sys.stdlib_module_names)

    def test_observation_default_disabled_has_no_clock_or_samples(self):
        clock = Mock(side_effect=RuntimeError('must not time disabled path'))
        metrics = self.metrics_module.CacheMetrics(clock=clock)
        self.assertFalse(metrics.enabled)
        self.assertEqual(metrics.measure('hybrid', 'read', lambda: 42), 42)
        metrics.event('hybrid', 'hit')
        metrics.payload_bytes('hybrid', b'private')
        clock.assert_not_called()
        self.assertEqual(metrics.snapshot(), {'events': {}, 'samples': {}})

    def test_hit_miss_bypass_and_read_write_latency(self):
        metrics, backend, proxy = self.observer()
        self.assertIsNone(proxy.get('missing'))
        self.assertTrue(proxy.set('k', False, 1800))
        self.assertIs(proxy.get('k'), False)
        self.assertIsNone(proxy.get(None))
        self.assertFalse(proxy.set(None, 3))
        events = metrics.snapshot()['events']
        self.assertEqual(events, {('hybrid', 'miss'): 1, ('hybrid', 'hit'): 1, ('hybrid', 'bypass'): 2})
        self.assertEqual(backend.calls['get'], 2)
        self.assertEqual(len(metrics.samples[('hybrid', 'read')]), 2)
        self.assertEqual(len(metrics.samples[('hybrid', 'write')]), 1)

    def test_default_argument_and_stored_none_keep_backend_semantics(self):
        metrics, backend, proxy = self.observer()
        sentinel = object()
        self.assertIs(proxy.get('missing', sentinel), sentinel)
        backend.set('none', None)
        self.assertIsNone(proxy.get('none', sentinel))
        self.assertEqual(metrics.events[('hybrid', 'miss')], 2)

    def test_backend_read_exception_preserved_and_business_helper_falls_back(self):
        metrics, backend, proxy = self.observer()
        backend.get = Mock(side_effect=ConnectionError('redacted'))
        with self.assertRaises(ConnectionError):
            proxy.get('k')
        with self.assertLogs('movie_agent.cache', level='WARNING'):
            self.assertEqual(self.access.business_cache_get(proxy, 'k', 9), 9)
        self.assertEqual(metrics.events[('hybrid', 'error')], 2)
        self.assertEqual(backend.get.call_count, 2)

    def test_write_error_and_false_result_preserve_cache_access_fallback(self):
        metrics, backend, proxy = self.observer()
        backend.set = Mock(side_effect=ConnectionError('redacted'))
        with self.assertLogs('movie_agent.cache', level='WARNING'):
            self.assertFalse(self.access.business_cache_set(proxy, 'k', {}, 1800))
        backend.set = Mock(return_value=False)
        self.assertIs(proxy.set('k', 2), False)
        self.assertEqual(metrics.events[('hybrid', 'error')], 2)

    def test_broken_observer_does_not_repeat_or_break_backend_calls(self):
        broken = Mock()
        broken._start.side_effect = RuntimeError()
        broken._finish.side_effect = RuntimeError()
        broken.event.side_effect = RuntimeError()
        backend = self.bench.FakeCache()
        proxy = self.metrics_module.ObservedBusinessCache(backend, broken, 'hybrid')
        self.assertTrue(proxy.set('k', 7))
        self.assertEqual(proxy.get('k'), 7)
        self.assertEqual(backend.calls, {'get': 1, 'set': 1, 'add': 0, 'delete': 0})

    def test_timer_failure_preserves_function_result_and_business_exception(self):
        metrics, _, _ = self.observer(clock=Mock(side_effect=RuntimeError()))
        fn = Mock(return_value='ok')
        self.assertEqual(metrics.measure('hybrid', 'orm_rehydration', fn), 'ok')
        fn.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'business'):
            metrics.measure('hybrid', 'orm_rehydration', Mock(side_effect=ValueError('business')))

    def test_broken_timer_methods_cannot_affect_business_result(self):
        metrics, _, _ = self.observer()
        metrics._start = Mock(side_effect=RuntimeError())
        metrics._finish = Mock(side_effect=RuntimeError())
        fn = Mock(return_value=17)
        self.assertEqual(metrics.measure('hybrid', 'generation_lookup', fn), 17)
        fn.assert_called_once()

    def test_samples_are_bounded_and_labels_are_low_cardinality(self):
        metrics, _, _ = self.observer(max_samples=3)
        for i in range(10):
            metrics.event('private-user-' + str(i), 'hit')
            metrics.payload_bytes('private-session-' + str(i), b'x' * i)
        metrics.event('hybrid', 'private-event')
        self.assertEqual(metrics.events, {('other', 'hit'): 10})
        self.assertEqual(list(metrics.samples[('other', 'serialization_bytes')]), [7, 8, 9])

    def test_snapshot_never_contains_keys_profiles_queries_or_session_tokens(self):
        metrics, backend, proxy = self.observer()
        private = 'SECRET-session-query-profile-user'
        proxy.set(private, {'profile_text': private})
        proxy.get(private)
        metrics.payload_bytes('hybrid', private.encode())
        self.assertNotIn('SECRET', repr(metrics.snapshot()))
        self.assertEqual(backend.data[private], {'profile_text': private})

    def test_payload_size_uses_existing_blob_only_and_memoryview_bytes(self):
        metrics, _, _ = self.observer()
        unsafe = Mock()
        metrics.payload_bytes('hybrid', unsafe)
        self.assertEqual(unsafe.mock_calls, [])
        blob = memoryview(b'12345678').cast('I')
        metrics.payload_bytes('hybrid', blob)
        self.assertEqual(list(metrics.samples[('hybrid', 'serialization_bytes')]), [8])

    def test_generation_lookup_measurement_preserves_cold_and_hot_call_counts(self):
        metrics, backend, _ = self.observer()
        first = metrics.measure('generation', 'generation_lookup',
                                self.keys.hybrid_recommendation_cache_key, backend, 7, 30)
        self.assertEqual(backend.calls, {'get': 4, 'add': 2, 'set': 0, 'delete': 0})
        backend.calls = dict.fromkeys(backend.calls, 0)
        second = metrics.measure('generation', 'generation_lookup',
                                 self.keys.hybrid_recommendation_cache_key, backend, 7, 30)
        self.assertEqual(first, second)
        self.assertEqual(backend.calls, {'get': 2, 'add': 0, 'set': 0, 'delete': 0})

    def test_invalidation_changes_all_top_k_for_one_user_only(self):
        backend = self.bench.FakeCache()
        old = [self.keys.hybrid_recommendation_cache_key(backend, 7, k) for k in (30, 100)]
        other = self.keys.hybrid_recommendation_cache_key(backend, 8, 30)
        self.keys.invalidate_recommendation_cache(backend, 7)
        self.assertTrue(all(key != self.keys.hybrid_recommendation_cache_key(backend, 7, k)
                            for key, k in zip(old, (30, 100))))
        self.assertEqual(other, self.keys.hybrid_recommendation_cache_key(backend, 8, 30))

    def test_generation_failure_still_bypasses_existing_keys(self):
        backend = Mock()
        backend.get.side_effect = ConnectionError()
        with self.assertLogs('movie_agent.cache', level='WARNING'):
            self.assertIsNone(self.keys.hybrid_recommendation_cache_key(backend, 7, 30))
        backend.get.assert_called_once()

    def test_light_payload_roundtrip_order_scores_relations_and_batch_restore(self):
        for count in (30, 100, 500):
            with self.subTest(count=count):
                result, payload, hydrate, manager = self.hydrate(count)
                actual = hydrate(pickle.loads(pickle.dumps(payload, protocol=5)))
                self.assertEqual(self.bench.result_snapshot(actual), self.bench.result_snapshot(result))
                self.assertEqual(manager.query_calls, 1)
                self.assertEqual(manager.prefetch_calls, 1)
                self.assertEqual(manager.relations, ('genres', 'actors', 'directors'))

    def test_light_payload_contains_no_synthetic_objects(self):
        _, payload, _, _ = self.hydrate()
        def plain(value):
            if isinstance(value, dict): return all(plain(k) and plain(v) for k, v in value.items())
            if isinstance(value, list): return all(plain(v) for v in value)
            return type(value) in (int, float, str, bool, type(None))
        self.assertTrue(plain(payload))
        self.assertEqual(payload['schema'], 'hybrid_rec_payload_v1')

    def test_missing_movie_rejects_complete_payload_instead_of_partial_result(self):
        _, payload, hydrate, manager = self.hydrate()
        del manager.rows[1]
        self.assertIsNone(hydrate(payload))
        self.assertEqual(manager.query_calls, 1)

    def test_invalid_payload_is_rejected_before_query(self):
        _, payload, hydrate, manager = self.hydrate()
        for invalid in (float('nan'), float('inf'), True, 10 ** 1000, 'score'):
            bad = {**payload, 'items': [{'movie_id': 1, 'rrf_score': invalid}]}
            self.assertIsNone(hydrate(bad))
        self.assertIsNone(hydrate({**payload, 'schema': 'old'}))
        self.assertEqual(manager.query_calls, 0)

    def test_database_error_still_propagates(self):
        _, payload, hydrate, manager = self.hydrate()
        manager.filter = Mock(side_effect=RuntimeError('database'))
        with self.assertRaisesRegex(RuntimeError, 'database'):
            hydrate(payload)

    def test_candidate_is_functionally_equivalent_but_not_applied(self):
        source = (ROOT / 'myapp/views.py').read_text()
        self.assertEqual(source.count('round(float(scores[movie.id]), 4)'), 1)
        candidate = source.replace('round(float(scores[movie.id]), 4)', 'round(scores[movie.id], 4)')
        old = self.bench.production_functions(source)
        new = self.bench.production_functions(candidate)
        result = self.bench.synthetic_result(30)
        payload = old['build_payload'](result)
        payload['items'][0]['rrf_score'] = 2  # finite int and unrounded float retain rounding.
        payload['items'][1]['rrf_score'] = .12345678
        before = self.bench.restore_function(old, self.bench.FakeManager(result[0]))(payload)
        after = self.bench.restore_function(new, self.bench.FakeManager(result[0]))(payload)
        self.assertEqual(self.bench.result_snapshot(before), self.bench.result_snapshot(after))

    def test_production_cache_algorithms_and_helpers_match_frozen_baseline(self):
        paths = ['myapp/views.py', 'myapp/utils/cache_access.py', 'myapp/utils/cache_keys.py',
                 'myapp/services/cache_mutations.py', 'myapp/recommender/recall.py',
                 'myapp/recommender/rerank.py', 'myapp/services/agent_service.py',
                 'myapp/agent/movie_agent.py']
        for path in paths:
            expected = subprocess.check_output(['git', 'show', BASELINE + ':' + path], cwd=ROOT)
            self.assertEqual((ROOT / path).read_bytes(), expected, path)

    def test_benchmark_limits_and_nearest_rank_percentiles(self):
        self.assertEqual(self.bench.percentiles([1000, 2000, 3000, 4000]),
                         {'p50_us': 2.0, 'p95_us': 4.0, 'p99_us': 4.0})
        for args in ((121, 5, 20), (20, 11, 20), (20, 5, 31)):
            with self.assertRaises(ValueError): self.bench.run(*args)
        with self.assertRaises(ValueError): self.bench.compare_candidate(121)

    def test_original_test_count_and_legacy_anchor_preserved(self):
        paths = list((ROOT / 'tests').glob('test_redis_cache*isolated.py'))
        original = [p for p in paths if p.name != 'test_redis_cache_p2_isolated.py']
        count = sum(sum(isinstance(n, ast.FunctionDef) and n.name.startswith('test_')
                        for n in ast.walk(ast.parse(p.read_text()))) for p in original)
        self.assertEqual(count, 81)
        legacy = (ROOT / 'tests/test_redis_cache_p01_isolated.py').read_text()
        self.assertNotIn('HEAD:myapp/recommender/rerank.py', legacy)
        self.assertIn('d78390dede4a0371a1e7991f06568d8d390fbdbd', legacy)


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(os.environ.get(k) != '1' for k in
            ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')) or not sys.flags.no_site:
        raise RuntimeError('Explicit CPU-only -S environment required')
    forbidden = {'torch', 'transformers', 'sentence_transformers', 'faiss', 'tensorflow', 'numpy',
                 'django', 'redis', 'pymysql', 'MySQLdb', 'neo4j'}
    class Guard:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in forbidden:
                raise RuntimeError('Forbidden isolated test import: ' + fullname)
    sys.meta_path.insert(0, Guard())
    if any(name.split('.')[0] in forbidden for name in sys.modules):
        raise RuntimeError('Unsafe module already imported')
    unittest.main(verbosity=2)
