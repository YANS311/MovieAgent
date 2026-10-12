"""P1-A consistency tests: stdlib, AST extraction and fake backends only.

Do not import Django settings, myapp.views, models, or AppConfig: ready() warms
the recommendation stack. This suite does not test real SessionMiddleware.
"""

import ast
from contextlib import contextmanager
import functools
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import types
import unittest
import unittest.mock
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p0_fixture', ROOT / 'tests/test_redis_cache_p0_isolated.py')
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)
keys = p0.keys
access = p0.access


class FakeResponse(dict):
    def __init__(self, payload, status=200):
        super().__init__(payload)
        self.status_code = status
        self.content = json.dumps(payload).encode()

    def get(self, key, default=None):
        return 'application/json' if key == 'Content-Type' else default


class FakeTransaction:
    """Enough atomic/on_commit semantics to prove callback and rollback order."""
    class Rollback(Exception):
        pass

    def __init__(self):
        self.frames = []
        self.callbacks = []
        self.rollback_flags = []

    @contextmanager
    def atomic(self):
        snapshot = list(self.callbacks)
        self.frames.append(snapshot)
        self.rollback_flags.append(False)
        try:
            yield
            if self.rollback_flags[-1]:
                raise self.Rollback()
        except BaseException as exc:
            del self.callbacks[len(snapshot):]
            self.frames.pop()
            self.rollback_flags.pop()
            if isinstance(exc, self.Rollback):
                return
            raise
        else:
            self.frames.pop()
            self.rollback_flags.pop()
            if not self.frames:
                callbacks, self.callbacks = self.callbacks, []
                for callback in callbacks:
                    callback()

    def set_rollback(self, value):
        self.rollback_flags[-1] = value

    def on_commit(self, callback, using=None):
        if self.frames:
            self.callbacks.append(callback)
        else:
            callback()


def mutation_helpers(cache, tx, logger=None):
    p0.assert_production_cache_bindings('myapp/services/cache_mutations.py')
    tree = ast.parse((ROOT / 'myapp/services/cache_mutations.py').read_text())
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {
                 'schedule_user_invalidation', 'schedule_catalog_invalidation',
                 'partial_commit_response', 'recommendation_write_view', 'invalidation_recovery_requests'}]
    ns = {
        'cache': cache, 'transaction': tx, 'JsonResponse': FakeResponse,
        'wraps': functools.wraps, 'json': json, 'logger': logger or Mock(),
        'invalidate_recommendation_cache': keys.invalidate_recommendation_cache,
        'invalidate_catalog_cache': keys.invalidate_catalog_cache,
        'RecommendationCacheInvalidationError': keys.RecommendationCacheInvalidationError,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'cache_mutations', 'exec'), ns)
    return ns


class P1ATests(unittest.TestCase):
    def setUp(self):
        self.cache = p0.FakeCache()
        self.tx = FakeTransaction()

    def test_view_wrapper_runs_user_invalidation_only_after_commit(self):
        ns = mutation_helpers(self.cache, self.tx)
        calls = []
        def view(request):
            ns['schedule_user_invalidation'](request.user.id)
            calls.append('database-write')
            return FakeResponse({'status': 'ok'})
        wrapped = ns['recommendation_write_view'](view)
        result = wrapped(types.SimpleNamespace(method='POST', user=types.SimpleNamespace(id=17)))
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(calls, ['database-write'])
        self.assertIn('hybrid_rec:v2:17:generation', self.cache.data)
        self.assertEqual(self.tx.callbacks, [])

    def test_outer_rollback_discards_invalidation_callback(self):
        ns = mutation_helpers(self.cache, self.tx)
        wrapped = ns['recommendation_write_view'](
            lambda request: (ns['schedule_user_invalidation'](17), FakeResponse({'status': 'ok'}))[1])
        with self.assertRaises(RuntimeError):
            with self.tx.atomic():
                wrapped(types.SimpleNamespace(method='POST'))
                self.assertEqual(self.cache.data, {})
                raise RuntimeError('outer rollback')
        self.assertEqual(self.cache.data, {})
        self.assertEqual(self.tx.callbacks, [])

    def test_error_json_marks_transaction_for_rollback(self):
        ns = mutation_helpers(self.cache, self.tx)
        def view(request):
            ns['schedule_user_invalidation'](17)
            return FakeResponse({'status': 'error'})
        ns['recommendation_write_view'](view)(types.SimpleNamespace(method='POST'))
        self.assertEqual(self.cache.data, {})
        self.assertEqual(self.tx.callbacks, [])

    def test_committed_redis_failure_is_partial_success_and_not_mutation_retry(self):
        offline = Mock()
        offline.set.side_effect = ConnectionError('offline')
        tx = FakeTransaction()
        ns = mutation_helpers(offline, tx)
        wrapped = ns['recommendation_write_view'](
            lambda request: (ns['schedule_user_invalidation'](17), FakeResponse({'status': 'ok'}))[1])
        response = wrapped(types.SimpleNamespace(method='POST'))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['status'], 'partial_success')
        self.assertTrue(response['database_committed'])
        self.assertFalse(response['cache_invalidated'])
        self.assertFalse(response['retry_mutation'])
        self.assertEqual(response['recovery'], 'retry_cache_invalidation_only')

    def test_outer_transaction_failure_is_translated_by_committed_cache_middleware(self):
        ns = mutation_helpers(self.cache, self.tx)
        tree = ast.parse((ROOT / 'myapp/middleware.py').read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'CommittedCacheFailureMiddleware')
        scope = {'partial_commit_response': ns['partial_commit_response'],
                 'RecommendationCacheInvalidationError': keys.RecommendationCacheInvalidationError}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'CommittedCacheFailureMiddleware', 'exec'), scope)
        middleware = scope['CommittedCacheFailureMiddleware'](lambda request: None)
        error = keys.RecommendationCacheInvalidationError('redis down', user_id=17)
        response = middleware.process_exception(object(), error)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['status'], 'partial_success')
        self.assertFalse(response['retry_mutation'])

    def test_retry_endpoint_retries_only_cache_and_checks_user_scope(self):
        endpoint = p0.extract('myapp/agent_views.py', {'retry_cache_invalidation'}, {
            'cache': self.cache, 'JsonResponse': FakeResponse,
            'invalidate_recommendation_cache': keys.invalidate_recommendation_cache,
            'invalidate_catalog_cache': keys.invalidate_catalog_cache,
            'RecommendationCacheInvalidationError': keys.RecommendationCacheInvalidationError,
        })['retry_cache_invalidation']
        self.cache.set('hybrid_rec_17', 'legacy')
        self.cache.set('hybrid_rec_22', 'other-user')
        own = types.SimpleNamespace(id=17, is_staff=False)
        response = endpoint(types.SimpleNamespace(user=own, POST={'scope': 'user'}))
        self.assertEqual(response['status'], 'ok')
        self.assertFalse(response['database_mutation'])
        self.assertNotIn('hybrid_rec_17', self.cache.data)
        self.assertIn('hybrid_rec_22', self.cache.data)
        denied = endpoint(types.SimpleNamespace(user=own, POST={'scope': 'user', 'user_id': '22'}))
        self.assertEqual(denied.status_code, 403)

    def test_nested_savepoint_rollback_discards_only_its_callback(self):
        ns = mutation_helpers(self.cache, self.tx)
        with self.tx.atomic():
            ns['schedule_user_invalidation'](1)
            try:
                with self.tx.atomic():
                    ns['schedule_user_invalidation'](2)
                    raise ValueError('savepoint rollback')
            except ValueError:
                pass
            self.assertEqual(len(self.tx.callbacks), 1)
        self.assertIn('hybrid_rec:v2:1:generation', self.cache.data)
        self.assertNotIn('hybrid_rec:v2:2:generation', self.cache.data)

    def test_all_top_k_keys_change_for_only_the_mutated_user(self):
        before = {k: keys.hybrid_recommendation_cache_key(self.cache, 4, k) for k in (5, 20, 50)}
        other = keys.hybrid_recommendation_cache_key(self.cache, 9, 20)
        for key in before.values():
            self.cache.set(key, 'cached', 1800)
        self.cache.set(other, 'other-user', 1800)
        keys.invalidate_recommendation_cache(self.cache, 4)
        after = {k: keys.hybrid_recommendation_cache_key(self.cache, 4, k) for k in before}
        self.assertTrue(all(before[k] != after[k] for k in before))
        self.assertEqual(self.cache.get(other), 'other-user')

    def test_multi_recall_key_tracks_user_query_top_k_weights_and_invalidation(self):
        base = keys.multi_recall_cache_key(self.cache, 4, '科幻', 10, [1.0, 0.8])
        self.assertNotEqual(base, keys.multi_recall_cache_key(self.cache, 4, '科幻', 20, [1.0, 0.8]))
        self.assertNotEqual(base, keys.multi_recall_cache_key(self.cache, 4, '剧情', 10, [1.0, 0.8]))
        self.assertNotEqual(base, keys.multi_recall_cache_key(self.cache, 4, '科幻', 10, [1.0, 0.9]))
        other = keys.multi_recall_cache_key(self.cache, 9, '科幻', 10, [1.0, 0.8])
        keys.invalidate_recommendation_cache(self.cache, 4)
        self.assertNotEqual(base, keys.multi_recall_cache_key(self.cache, 4, '科幻', 10, [1.0, 0.8]))
        self.assertEqual(other, keys.multi_recall_cache_key(self.cache, 9, '科幻', 10, [1.0, 0.8]))

    def test_multi_channel_recall_cache_failure_falls_back_to_recall(self):
        offline = Mock()
        offline.get.side_effect = ConnectionError('redis down')
        offline.add.side_effect = ConnectionError('redis down')
        offline.set.side_effect = ConnectionError('redis down')
        hot = Mock(return_value=[{'movie_id': 6, 'score': 1.0}])
        ns = {
            'time': __import__('time'), 'cache': offline,
            'multi_recall_cache_key': keys.multi_recall_cache_key,
            'business_cache_get': access.business_cache_get,
            'business_cache_set': access.business_cache_set,
            '_get_user_liked_ids': Mock(return_value=[]), 'vector_recall': Mock(return_value=[]),
            'content_recall': Mock(return_value=[]), 'model_recall': Mock(return_value=[]),
            'kg_recall': Mock(return_value=[]), 'rrf_merge': lambda *args, **kwargs: [],
            'hot_recall': hot,
        }
        fn = p0.extract('myapp/recommender/recall.py', {'multi_channel_recall'}, ns)['multi_channel_recall']
        result = fn(types.SimpleNamespace(id=4), query_text='科幻', top_k=1)
        self.assertEqual(result[0], [{'movie_id': 6, 'score': 1.0}])
        hot.assert_called_once_with([], k=1)

    def test_query_rewriter_cache_failure_uses_rule_path(self):
        source = ast.parse((ROOT / 'myapp/recommender/query_rewriter.py').read_text())
        cls = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == 'QueryRewriter')
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'rewrite')
        method.decorator_list = []
        ns = {'cache': Mock(), 'query_rewrite_cache_key': keys.query_rewrite_cache_key,
              'business_cache_get': access.business_cache_get,
              'business_cache_set': access.business_cache_set}
        ns['cache'].get.side_effect = ConnectionError('redis down')
        ns['cache'].add.side_effect = ConnectionError('redis down')
        ns['cache'].set.side_effect = ConnectionError('redis down')
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'QueryRewriter.rewrite', 'exec'), ns)
        instance = types.SimpleNamespace(use_llm=False, _get_llm=Mock(return_value=None),
                                         _rule_based_rewrite=Mock(return_value='科幻 太空'))
        self.assertEqual(ns['rewrite'](instance, '科幻'), '科幻 太空')
        instance._rule_based_rewrite.assert_called_once_with('科幻')

    def test_catalog_change_rotates_business_keys_but_leaves_session_keys(self):
        base = 'kg_subgraph:test'
        old = keys.catalog_business_cache_key(self.cache, base)
        self.cache.set(old, 'graph', 1800)
        self.cache.set('django.contrib.sessions.cache:session-id', 'auth-session', 3600)
        keys.invalidate_catalog_cache(self.cache)
        self.assertNotEqual(old, keys.catalog_business_cache_key(self.cache, base))
        self.assertEqual(self.cache.get('django.contrib.sessions.cache:session-id'), 'auth-session')

    def test_business_cache_failure_falls_back_and_never_returns_empty_as_cache_result(self):
        ns = p0.CacheP0Tests()
        ns.setUp()
        fixture = ns.hybrid_function()
        fixture['cache'] = Mock()
        fixture['cache'].get.side_effect = ConnectionError('redis down')
        fixture['cache'].add.side_effect = ConnectionError('redis down')
        fixture['cache'].set.side_effect = ConnectionError('redis down')
        fn = fixture['hybrid_recall_recommend']
        result = fn(types.SimpleNamespace(id=7), top_k=2)
        self.assertEqual([m.id for m in result[0]], [3, 4])
        self.assertEqual(fixture['load_rag_resources'].call_count, 1)

    def test_kg_cache_read_write_failure_returns_computed_graph(self):
        offline = Mock()
        offline.get.side_effect = ConnectionError('redis down')
        offline.add.side_effect = ConnectionError('redis down')
        offline.set.side_effect = ConnectionError('redis down')
        graph = Mock()
        graph.run.return_value.data.return_value = [
            {'mid': 1, 'title': 'One', 'director': 'Director', 'genres': ['Sci-Fi']}]
        fn = p0.extract('myapp/views.py', {'get_kg_subgraph'}, {
            'cache': offline, 'kg_subgraph_cache_key': keys.kg_subgraph_cache_key,
            'catalog_business_cache_key': keys.catalog_business_cache_key,
            'business_cache_get': access.business_cache_get,
            'business_cache_set': access.business_cache_set,
            'neo_graph': graph, 'logger': Mock()})['get_kg_subgraph']
        self.assertIn('One', fn('topic'))
        # The KG routine follows three query branches for its bounded result.
        self.assertGreater(graph.run.call_count, 0)

    def test_intent_cache_failure_uses_parser_instead_of_failing_closed(self):
        source = (ROOT / 'myapp/agent/movie_agent.py').read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'LLMIntentParser')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'parse')
        cache = Mock()
        cache.get.side_effect = ConnectionError('redis down')
        cache.add.side_effect = ConnectionError('redis down')
        cache.set.side_effect = ConnectionError('redis down')
        llm = Mock()
        llm.invoke.return_value.content = '{"intent":"recommend"}'
        ns = {'cache': cache, 'intent_cache_key': keys.intent_cache_key,
              'business_cache_get': access.business_cache_get,
              'business_cache_set': access.business_cache_set,
              'json': json, 're': re, 'logger': Mock(), 'ChatPromptTemplate': Mock(),
              'StrOutputParser': Mock(), 'llm': llm}
        class Parser:
            SYSTEM_PROMPT = 'system'
            VALID_SORT = {'rating', 'release_date', 'hot'}
            _normalize = staticmethod(lambda value: value)
            _map_tags_to_zh = staticmethod(lambda value: value if isinstance(value, list) else [])
        django = p0.fake_module('django')
        django_core = p0.fake_module('django.core')
        django_cache = p0.fake_module('django.core.cache', cache=cache)
        modules = {'django': django, 'django.core': django_core, 'django.core.cache': django_cache}
        method.decorator_list = []
        exec(compile(ast.Module(body=[method], type_ignores=[]), 'LLMIntentParser.parse', 'exec'), ns)
        with unittest.mock.patch.dict(sys.modules, modules):
            result = ns['parse'](Parser, '推荐科幻',
                                 lambda *_: '{"query_text":"推荐科幻","tags":["sci-fi"],"sort_by":"hot"}')
        self.assertEqual(result['query_text'], '推荐科幻')
        self.assertEqual(result['tags'], ['sci-fi'])

    def test_feedback_entrypoints_schedule_user_cache_invalidation(self):
        # P0.1 suite executes the three feedback views against FakeDB/FakeCache.
        # Here verify the remaining source-defined write entrances are wired.
        source = (ROOT / 'myapp/views.py').read_text()
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        expected = ('score_movie', 'movie_detail', 'add_collect', 'remove_collect',
                    'ajax_collect', 'center', 'admin_comment_delete', 'admin_user_add',
                    'admin_user_edit', 'admin_user_delete')
        for name in expected:
            with self.subTest(view=name):
                node = functions[name]
                calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)]
                self.assertTrue(any(isinstance(c.func, ast.Name) and
                                    c.func.id in {'schedule_user_invalidation', 'schedule_catalog_invalidation'}
                                    for c in calls), name)
        agent_source = ast.parse((ROOT / 'myapp/agent_views.py').read_text())
        agent_functions = {node.name: node for node in agent_source.body
                           if isinstance(node, ast.FunctionDef)}
        for name in ('recommend_feedback_view', 'admin_user_feedback_delete',
                     'admin_user_profile_edit', 'admin_user_profile_delete'):
            with self.subTest(agent_view=name):
                calls = [n for n in ast.walk(agent_functions[name]) if isinstance(n, ast.Call)]
                allowed = ({'schedule_user_invalidation', 'invalidate_recommendation_cache'}
                           if name == 'recommend_feedback_view' else {'schedule_user_invalidation'})
                self.assertTrue(any(isinstance(c.func, ast.Name) and
                                    c.func.id in allowed for c in calls), name)

    def test_admin_and_view_sources_do_not_invoke_global_cache_clear(self):
        checked = ('myapp/admin.py', 'myapp/views.py', 'myapp/agent_views.py',
                   'myapp/services/cache_mutations.py')
        for relative in checked:
            tree = ast.parse((ROOT / relative).read_text(), filename=relative)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    attr = node.func.attr if isinstance(node.func, ast.Attribute) else None
                    is_global_cache_clear = (attr == 'clear' and isinstance(node.func, ast.Attribute)
                                             and isinstance(node.func.value, ast.Name)
                                             and node.func.value.id == 'cache')
                    self.assertFalse(is_global_cache_clear, f'{relative}:{node.lineno} global clear')

    def test_cache_helpers_treat_preferences_as_strict_and_business_as_optional(self):
        offline = Mock()
        offline.get.side_effect = ConnectionError('offline')
        self.assertEqual(access.business_cache_get(offline, 'business-key', 'fallback'), 'fallback')
        with self.assertRaises(access.MemoryCacheUnavailable):
            access.memory_cache_operation(offline, 'get', 'preference-key')
        fixture = p0.CacheP0Tests()
        fixture.setUp()
        fixture.cache = offline
        manager_class = fixture.memory_class()
        memory = manager_class(session_id='real-session')
        with self.assertRaises(access.MemoryCacheUnavailable):
            memory.get_slots()


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
