"""CPU-only compatibility tests for the lightweight hybrid recommendation cache.

Only stdlib, FakeCache, and AST-selected functions are executed. No Django app,
models, database, Redis, model stack, or AppConfig.ready() is imported.
"""

import ast
import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p0_fixture', ROOT / 'tests/test_redis_cache_p0_isolated.py')
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)
keys = p0.keys


class HybridCacheP1BTests(unittest.TestCase):
    def setUp(self):
        fixture = p0.CacheP0Tests()
        fixture.setUp()
        self.cache = fixture.cache
        self.ns = fixture.hybrid_function()
        self.recommend = self.ns['hybrid_recall_recommend']
        self.user = types.SimpleNamespace(id=7)

    def snapshot(self, result):
        movies, profile_text, stats = result
        return ([(movie.id, movie.rrf_score) for movie in movies], profile_text, stats)

    def test_cold_and_hot_paths_preserve_return_contract_order_scores_and_fields(self):
        cold = self.recommend(self.user, top_k=3)
        key = keys.hybrid_recommendation_cache_key(self.cache, self.user.id, 3)
        payload = self.cache.get(key)
        self.assertEqual(payload['schema'], 'hybrid_rec_payload_v1')
        self.assertEqual([item['movie_id'] for item in payload['items']], [3, 4, 2])
        self.assertEqual([item['rrf_score'] for item in payload['items']],
                         [movie.rrf_score for movie in cold[0]])
        self.assertEqual(payload['profile_text'], cold[1])
        self.assertNotIn('profile_text', payload['source_stats'])
        self.assertEqual({**payload['source_stats'], 'profile_text': payload['profile_text']}, cold[2])
        self.assertTrue(all(not isinstance(value, types.SimpleNamespace)
                            for item in payload['items'] for value in item.values()))
        self.assertEqual(self.cache.timeouts[key], 1800)

        hot = self.recommend(self.user, top_k=3)
        self.assertEqual(self.snapshot(hot), self.snapshot(cold))
        for movies, _, _ in (cold, hot):
            for movie in movies:
                self.assertEqual((movie.title, movie.summary, movie.score),
                                 (f'Movie{movie.id}', f'Summary{movie.id}', 8.0))
                self.assertEqual([genre.name for genre in movie.genres.all()[:3]], ['科幻'])
        self.assertEqual(self.ns['Movie'].objects.filter.call_count, 2)
        self.assertEqual(self.ns['prefetch_related'].call_count, 2)
        self.assertEqual(self.ns['prefetch_related'].call_args_list[-1].args,
                         ('genres', 'actors', 'directors'))
        self.assertEqual(self.ns['load_rag_resources'].call_count, 1)

    def test_top_k_and_user_ids_are_isolated(self):
        key_a = keys.hybrid_recommendation_cache_key(self.cache, 7, 3)
        key_b = keys.hybrid_recommendation_cache_key(self.cache, 7, 4)
        key_other = keys.hybrid_recommendation_cache_key(self.cache, 8, 3)
        self.assertNotEqual(key_a, key_b)
        self.assertNotEqual(key_a, key_other)

    def test_generation_backend_failure_disables_cache_lookup(self):
        unavailable = Mock()
        unavailable.get.side_effect = ConnectionError('generation unavailable')
        self.assertIsNone(keys.hybrid_recommendation_cache_key(unavailable, 7, 3))
        self.assertIsNone(keys.catalog_business_cache_key(unavailable, 'catalog:test'))

    def test_generation_update_makes_old_payload_unreachable(self):
        old_key = keys.hybrid_recommendation_cache_key(self.cache, 7, 3)
        self.cache.set(old_key, {'schema': 'hybrid_rec_payload_v1', 'items': [],
                                 'profile_text': 'old', 'source_stats': {}}, 1800)
        keys.invalidate_recommendation_cache(self.cache, 7)
        new_key = keys.hybrid_recommendation_cache_key(self.cache, 7, 3)
        self.assertNotEqual(old_key, new_key)
        self.assertIsNone(self.cache.get(new_key))

    def test_deleted_cached_movie_invalidates_payload_and_recomputes(self):
        cold = self.recommend(self.user, top_k=2)
        self.assertEqual([movie.id for movie in cold[0]], [3, 4])
        self.ns['available_ids'].remove(4)
        recomputed = self.recommend(self.user, top_k=2)
        self.assertEqual([movie.id for movie in recomputed[0]], [3])
        self.assertEqual(self.ns['load_rag_resources'].call_count, 2)
        key = keys.hybrid_recommendation_cache_key(self.cache, 7, 2)
        payload = self.cache.get(key)
        self.assertEqual([item['movie_id'] for item in payload['items']], [3])
        cached_again = self.recommend(self.user, top_k=2)
        self.assertEqual(self.snapshot(cached_again), self.snapshot(recomputed))

    def test_old_orm_tuple_and_wrong_schema_are_never_returned(self):
        new_key = keys.hybrid_recommendation_cache_key(self.cache, 7, 2)
        self.cache.set(new_key, ([types.SimpleNamespace(id=1)], 'stale', {}), 1800)
        result = self.recommend(self.user, top_k=2)
        self.assertEqual(result[1], '科幻画像')
        self.assertEqual(self.ns['load_rag_resources'].call_count, 1)
        self.assertEqual(self.cache.get(new_key)['schema'], 'hybrid_rec_payload_v1')

        old_family_key = keys.user_business_cache_key(self.cache, 7, 'hybrid_rec', 'top_k:2')
        self.assertNotEqual(new_key, old_family_key)

    def test_redis_entry_read_and_write_failures_still_return_recommendations(self):
        class PartialOfflineCache(p0.FakeCache):
            def get(inner_self, key, default=None):
                if key.startswith('hybrid_rec_payload:'):
                    raise ConnectionError('cache read unavailable')
                return super(PartialOfflineCache, inner_self).get(key, default)

            def set(inner_self, key, value, timeout=None):
                if key.startswith('hybrid_rec_payload:'):
                    raise ConnectionError('cache write unavailable')
                return super(PartialOfflineCache, inner_self).set(key, value, timeout)

        offline = PartialOfflineCache()
        ns = p0.CacheP0Tests()
        ns.setUp()
        fixture = ns.hybrid_function()
        fixture['cache'] = offline
        fn = fixture['hybrid_recall_recommend']
        for _ in range(2):
            result = fn(types.SimpleNamespace(id=7), top_k=2)
            self.assertEqual([movie.id for movie in result[0]], [3, 4])
        self.assertEqual(fixture['load_rag_resources'].call_count, 2)

    def test_restore_helper_rejects_malformed_fields_before_orm_query(self):
        helper = self.ns['_restore_hybrid_cache_payload']
        before = self.ns['Movie'].objects.filter.call_count
        malformed = {'schema': 'hybrid_rec_payload_v1', 'items': [{'movie_id': '3', 'rrf_score': 0.1}],
                     'profile_text': 'profile', 'source_stats': {}}
        self.assertIsNone(helper(malformed))
        self.assertEqual(self.ns['Movie'].objects.filter.call_count, before)


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(
            os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('Explicit CPU-only environment required')
    forbidden = ('torch', 'transformers', 'sentence_transformers', 'faiss', 'tensorflow', 'numpy')
    class BlockModelImports:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in forbidden:
                raise RuntimeError(f'Forbidden model import: {fullname}')
    sys.meta_path.insert(0, BlockModelImports())
    if any(name.split('.')[0] in forbidden for name in sys.modules):
        raise RuntimeError('Model module already imported')
    unittest.main(verbosity=2)
