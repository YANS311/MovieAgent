"""CPU-only tests: stdlib + AST-extracted production code + FakeCache.

Never import myapp.agent, myapp.views, Django settings, or real ORM/models.
Run only this file, not Django setup/test discovery across the repository.
"""

import ast
import builtins
import copy
from contextlib import nullcontext
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
import symtable
import types
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('isolated_cache_keys', ROOT / 'myapp/utils/cache_keys.py')
keys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(keys)
access_spec = importlib.util.spec_from_file_location('isolated_cache_access', ROOT / 'myapp/utils/cache_access.py')
access = importlib.util.module_from_spec(access_spec)
access_spec.loader.exec_module(access)


def assert_production_cache_bindings(path, source=None):
    """Check actual production bindings before injecting any fake dependency."""
    modules = ('myapp.utils.cache_keys', 'myapp.utils.cache_access', 'myapp.services.cache_mutations')
    exports = {}
    for module in modules:
        tree = ast.parse((ROOT / (module.replace('.', '/') + '.py')).read_text())
        exports[module] = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    known = set().union(*exports.values())
    source = (ROOT / path).read_text() if source is None else source
    tree = ast.parse(source, filename=path)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in exports:
            for alias in node.names:
                if alias.name not in exports[node.module]:
                    raise AssertionError(f'{path}:{node.lineno}: unknown cache import {alias.name}')
    table = symtable.symtable(source, path, 'exec')
    bound = {symbol.get_name() for symbol in table.get_symbols()
             if symbol.is_assigned() or symbol.is_imported() or symbol.is_namespace()}
    missing = set()
    def inspect(scope):
        for symbol in scope.get_symbols():
            if symbol.is_global() and symbol.is_referenced() and symbol.get_name() in known:
                if symbol.get_name() not in bound and not hasattr(builtins, symbol.get_name()):
                    missing.add(symbol.get_name())
        for child in scope.get_children():
            inspect(child)
    inspect(table)
    if missing:
        raise AssertionError(f'{path}: unbound production cache symbols {sorted(missing)}')


def extract(path, names, namespace):
    """Execute selected definitions only, without module imports/initializers."""
    assert_production_cache_bindings(path)
    for name in ('business_cache_get', 'business_cache_set', 'memory_cache_operation', 'MemoryCacheUnavailable'):
        namespace.setdefault(name, getattr(access, name))
    namespace.setdefault('business_cache_delete', getattr(access, 'business_cache_delete'))
    for name in ('catalog_business_cache_key', 'user_business_cache_key',
                 'multi_recall_cache_key', 'query_rewrite_cache_key'):
        namespace.setdefault(name, getattr(keys, name))
    if 'partial_commit_response' not in namespace:
        helper_tree = ast.parse((ROOT / 'myapp/services/cache_mutations.py').read_text())
        helpers = [n for n in helper_tree.body if isinstance(n, ast.FunctionDef)
                   and n.name in {'partial_commit_response', 'invalidation_recovery_requests'}]
        exec(compile(ast.Module(body=helpers, type_ignores=[]), 'partial_commit_response', 'exec'), namespace)
    tree = ast.parse((ROOT / path).read_text(), filename=path)
    selected = [node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Assign))
                and (getattr(node, 'name', None) in names
                     or (isinstance(node, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)))]
    for node in selected:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []  # HTTP decorators require a live Django app.
    exec(compile(ast.Module(body=selected, type_ignores=[]), path, 'exec'), namespace)
    return namespace


class FakeCache:
    def __init__(self):
        self.data = {}
        self.timeouts = {}
        self.deleted = []

    def get(self, key, default=None):
        return copy.deepcopy(self.data.get(key, default))

    def set(self, key, value, timeout=None):
        self.data[key] = copy.deepcopy(value)
        self.timeouts[key] = timeout
        return True

    def add(self, key, value, timeout=None):
        if key in self.data:
            return False
        return self.set(key, value, timeout)

    def delete(self, key):
        self.deleted.append(key)
        self.data.pop(key, None)


def fake_module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


class CacheP0Tests(unittest.TestCase):
    def setUp(self):
        self.cache = FakeCache()
        self.logger = Mock()

    def memory_class(self):
        ns = {'cache': self.cache, 'memory_cache_key': keys.memory_cache_key,
              're': re, 'logger': self.logger}
        return extract('myapp/agent/memory.py', {'MemoryManager', 'SLOT_PATTERNS'}, ns)['MemoryManager']

    def test_stable_intent_key_across_processes_and_hash_seeds(self):
        text = '推荐科幻电影'
        expected = 'intent:v2:' + hashlib.sha256(text.encode()).hexdigest()
        self.assertEqual(keys.intent_cache_key('  ' + text + '\n'), expected)
        self.assertNotIn(text, expected)
        code = ('import runpy; '
                f'm=runpy.run_path({str(ROOT / "myapp/utils/cache_keys.py")!r}); '
                f'print(m["intent_cache_key"]({text!r}))')
        for seed in ('1', '987'):
            result = subprocess.check_output([sys.executable, '-S', '-c', code],
                                             env={**os.environ, 'PYTHONHASHSEED': seed}, text=True)
            self.assertEqual(result.strip(), expected)

    def test_sessions_users_and_anonymous_sessions_are_isolated(self):
        Memory = self.memory_class()
        for user in (None, types.SimpleNamespace(id=1)):
            a = Memory(user=user, session_id='real-session-a')
            b = Memory(user=user, session_id='real-session-b')
            a.update_slots({'genre': '科幻'})
            self.assertEqual(b.get_slots(), {})
            self.assertEqual(Memory(user=user, session_id='real-session-a').get_slots(), {'genre': '科幻'})
            a.clear_slots()
        a = Memory(user=types.SimpleNamespace(id=1), session_id='shared-id')
        b = Memory(user=types.SimpleNamespace(id=2), session_id='shared-id')
        a.update_slots({'genre': '动作'})
        self.assertEqual(b.get_slots(), {})

    def test_missing_or_legacy_session_uses_instance_memory_only(self):
        Memory = self.memory_class()
        for sid in (None, '', ' ', 'default', 'user_1', 'user_anon', 'user_None'):
            a = Memory(user=types.SimpleNamespace(id=1), session_id=sid)
            b = Memory(user=types.SimpleNamespace(id=1), session_id=sid)
            a.update_slots({'genre': '科幻'})
            a.update_slots({'director': 'Nolan'})
            self.assertEqual(a.get_slots(), {'genre': '科幻', 'director': 'Nolan'})
            self.assertEqual(b.get_slots(), {})
            a.clear_slots()
            self.assertEqual(a.get_slots(), {})
        self.assertEqual(self.cache.data, {})

    def test_dictionary_branch_preserves_merge_rules_and_persists(self):
        Memory = self.memory_class()
        a = Memory(session_id='session-a')
        a.update_slots({'genre': '科幻', 'score_min': 8.0})
        a.update_slots({'genre': '', 'score_min': None, 'year_min': 0})
        expected = {'genre': '科幻', 'score_min': 8.0, 'year_min': 0}
        self.assertEqual(Memory(session_id='session-a').get_slots(), expected)
        self.assertEqual(self.cache.timeouts[a._cache_key], 3600)
        a.update_slots('nolan')
        self.assertEqual(Memory(session_id='session-a').get_slots()['director'], 'Christopher Nolan')

    def test_session_key_creation_and_session_failure_propagation(self):
        get_key = extract('myapp/agent_views.py', {'_get_session_key'}, {})['_get_session_key']
        session = types.SimpleNamespace(session_key=None)
        session.create = Mock(side_effect=lambda: setattr(session, 'session_key', 'real-key'))
        request = types.SimpleNamespace(session=session)
        self.assertEqual(get_key(request), 'real-key')
        self.assertEqual(get_key(request), 'real-key')
        session.create.assert_called_once()
        session.session_key = None
        session.create.side_effect = ConnectionError('session backend offline')
        with self.assertRaises(ConnectionError):
            get_key(request)

    def test_chat_service_uses_same_session_identity_as_memory_views(self):
        from typing import Optional, Dict, Any, Generator
        ns = {'Optional': Optional, 'Dict': Dict, 'Any': Any, 'Generator': Generator}
        Service = extract('myapp/services/agent_chat_service.py', {'AgentChatService'}, ns)['AgentChatService']
        user = types.SimpleNamespace(id=9)
        service = Service(user, 'real-key')
        self.assertEqual(service.session_id, 'user_9_real-key')
        self.assertEqual(keys.memory_cache_key(service.session_id, user.id),
                         keys.memory_cache_key('user_9_real-key', user.id))
        self.assertIsNone(Service(user).session_id)
        self.assertIsNone(Service(user, 'default').session_id)

    def test_kg_key_normalization_preserves_id_types_and_limit(self):
        a = keys.kg_subgraph_cache_key('科幻', [3, 1, 3], 8)
        self.assertEqual(a, keys.kg_subgraph_cache_key('科幻', (1, 3), 8))
        self.assertEqual(keys.kg_subgraph_cache_key('科幻', None, 8),
                         keys.kg_subgraph_cache_key('科幻', [], 8))
        self.assertNotEqual(a, keys.kg_subgraph_cache_key('科幻', [1, 3], 12))
        self.assertNotEqual(a, keys.kg_subgraph_cache_key('科幻', ['1', '3'], 8))

    def test_kg_actual_function_does_not_reuse_different_max_triples(self):
        rows = [{'mid': i, 'title': f'Movie{i}', 'director': 'Director', 'genres': []}
                for i in range(1, 7)]
        graph = Mock()
        graph.run.return_value.data.return_value = rows
        ns = {'cache': self.cache, 'kg_subgraph_cache_key': keys.kg_subgraph_cache_key,
              'neo_graph': graph, 'logger': self.logger}
        fn = extract('myapp/views.py', {'get_kg_subgraph'}, ns)['get_kg_subgraph']
        small = fn('topic', max_triples=2)
        large = fn('topic', max_triples=5)
        self.assertEqual(len(small.splitlines()), 2)
        self.assertEqual(len(large.splitlines()), 5)
        self.assertEqual(fn('topic', max_triples=2), small)
        self.assertEqual(graph.run.call_count, 2)
        self.assertTrue(all(t == 1800 for t in self.cache.timeouts.values()))

    def hybrid_function(self):
        movies = {
            i: types.SimpleNamespace(
                id=i, title=f'Movie{i}', summary=f'Summary{i}', score=8.0,
                genres=types.SimpleNamespace(all=Mock(return_value=[types.SimpleNamespace(name='科幻')])),
                actors=types.SimpleNamespace(all=Mock(return_value=[])),
                directors=types.SimpleNamespace(all=Mock(return_value=[])),
            )
            for i in (1, 2, 3, 4)
        }
        available_ids = set(movies)
        manager = Mock()
        prefetch_related = Mock(side_effect=lambda *args, ids=None: [
            movies[i] for i in reversed(ids or []) if i in available_ids])
        manager.filter.side_effect = lambda id__in: types.SimpleNamespace(
            prefetch_related=lambda *args: prefetch_related(*args, ids=id__in))
        ns = {'cache': self.cache, 'hybrid_recommendation_cache_key': keys.hybrid_recommendation_cache_key,
              'Movie': types.SimpleNamespace(objects=manager),
              'math': __import__('math'), 'business_cache_delete': access.business_cache_delete,
              'build_user_profile_text': Mock(return_value=('科幻画像', [1])),
              'load_rag_resources': Mock(),
              'vector_recall': Mock(return_value=[(1, 1), (3, .9), (2, .8)]),
              'content_recall': Mock(return_value=[(2, .9), (4, .8)]),
              'model_recall': Mock(return_value=[(3, .9), (4, .8)])}
        extract('myapp/views.py', {'_HYBRID_CACHE_SCHEMA', '_restore_hybrid_cache_payload',
                                   'rrf_merge', 'hybrid_recall_recommend'}, ns)
        ns['prefetch_related'] = prefetch_related
        ns['available_ids'] = available_ids
        return ns

    def test_hybrid_cache_structure_order_scores_and_top_k(self):
        ns = self.hybrid_function()
        fn = ns['hybrid_recall_recommend']
        user = types.SimpleNamespace(id=7)
        first = fn(user, top_k=2)
        snapshot = lambda result: ([(m.id, m.rrf_score) for m in result[0]], result[1], result[2])
        second = fn(user, top_k=2)
        self.assertIsInstance(first, tuple)
        self.assertIsInstance(first[0], list)
        self.assertEqual(snapshot(first), snapshot(second))
        self.assertEqual([m.id for m in second[0]], [3, 4])
        self.assertEqual(ns['load_rag_resources'].call_count, 1)
        bigger = fn(user, top_k=3)
        self.assertEqual([m.id for m in bigger[0]], [3, 4, 2])
        self.assertEqual(ns['load_rag_resources'].call_count, 2)
        self.assertEqual(snapshot(fn(user, top_k=2, force_refresh=True)), snapshot(first))
        self.assertEqual(ns['load_rag_resources'].call_count, 3)
        self.assertTrue(all(t == 1800 for t in self.cache.timeouts.values()))

    def test_generation_eviction_does_not_resurrect_old_entries(self):
        key = keys.hybrid_recommendation_cache_key(self.cache, 1, 8)
        self.cache.set(key, 'old-result', 1800)
        generation_key = next(k for k in self.cache.data if k.endswith(':generation'))
        self.cache.delete(generation_key)
        self.assertNotEqual(keys.hybrid_recommendation_cache_key(self.cache, 1, 8), key)

    def test_inflight_old_generation_write_stays_unreachable(self):
        old = keys.hybrid_recommendation_cache_key(self.cache, 1, 8)
        keys.invalidate_recommendation_cache(self.cache, 1)
        self.cache.set(old, 'stale-recommendations', 1800)
        new = keys.hybrid_recommendation_cache_key(self.cache, 1, 8)
        self.assertNotEqual(old, new)
        self.assertIsNone(self.cache.get(new))

    def feedback_function(self, name, callback_list):
        feedback = Mock()
        feedback.objects.filter.return_value.delete.return_value = (1, {})
        modules = {'myapp': fake_module('myapp'),
                   'myapp.models_upgrade': fake_module('myapp.models_upgrade', UserFeedback=feedback)}
        movie = types.SimpleNamespace(id=3, title='Movie3')
        manager = Mock()
        manager.get.return_value = movie
        rec = Mock()
        rec.objects.filter.return_value.exclude.return_value.select_related.return_value.order_by.return_value.first.return_value = None
        ns = {'Movie': types.SimpleNamespace(objects=manager, DoesNotExist=LookupError),
              'Rec': rec, 'get_excluded_movie_ids': Mock(return_value=[]),
              'JsonResponse': lambda data, **kwargs: data,
              'cache': self.cache, 'invalidate_recommendation_cache': keys.invalidate_recommendation_cache,
              # Registration-only fixture assumes an existing outer transaction.
              'transaction': types.SimpleNamespace(atomic=nullcontext, on_commit=callback_list.append),
              'RecommendationCacheInvalidationError': keys.RecommendationCacheInvalidationError,
              'logger': self.logger}
        fn = extract('myapp/views.py', {name}, ns)[name]
        return fn, feedback, modules

    def test_feedback_invalidates_all_top_k_after_commit_only_for_this_user(self):
        for name in ('ajax_exclude_add', 'ajax_exclude_remove'):
            callbacks = []
            fn, feedback, modules = self.feedback_function(name, callbacks)
            before = [keys.hybrid_recommendation_cache_key(self.cache, 7, k) for k in (8, 30)]
            other = keys.hybrid_recommendation_cache_key(self.cache, 8, 8)
            self.cache.set('hybrid_rec_7', 'legacy')
            self.cache.set('cb_recs_7', 'legacy')
            request = types.SimpleNamespace(method='POST', user=types.SimpleNamespace(id=7),
                                            POST={'movie_id': '3'})
            with patch.dict(sys.modules, modules):
                self.assertEqual(fn(request)['status'], 'ok')
            self.assertEqual(len(callbacks), 1)
            self.assertEqual(keys.hybrid_recommendation_cache_key(self.cache, 7, 8), before[0])
            callbacks[0]()  # simulate successful outer transaction commit
            for k, old in zip((8, 30), before):
                self.assertNotEqual(keys.hybrid_recommendation_cache_key(self.cache, 7, k), old)
            self.assertEqual(keys.hybrid_recommendation_cache_key(self.cache, 8, 8), other)
            self.assertNotIn('hybrid_rec_7', self.cache.data)
            self.assertNotIn('cb_recs_7', self.cache.data)

    def test_failed_feedback_write_does_not_schedule_invalidation(self):
        callbacks = []
        fn, feedback, modules = self.feedback_function('ajax_exclude_add', callbacks)
        feedback.objects.create.side_effect = RuntimeError('database write failed')
        request = types.SimpleNamespace(method='POST', user=types.SimpleNamespace(id=7),
                                        POST={'movie_id': '3'})
        with patch.dict(sys.modules, modules), self.assertRaises(RuntimeError):
            fn(request)
        self.assertEqual(callbacks, [])

    def test_intent_cache_failure_retains_existing_business_fallback(self):
        ns = {'intent_cache_key': keys.intent_cache_key, 're': re, 'json': json, 'logger': self.logger}
        Parser = extract('myapp/agent/movie_agent.py', {'LLMIntentParser'}, ns)['LLMIntentParser']
        offline = Mock()
        offline.get.side_effect = ConnectionError('cache offline')
        offline.set.side_effect = ConnectionError('cache offline')
        modules = {'django': fake_module('django'), 'django.core': fake_module('django.core'),
                   'django.core.cache': fake_module('django.core.cache', cache=offline)}
        llm = Mock(return_value=json.dumps({'query_text': '科幻', 'tags': ['sci-fi'], 'sort_by': 'rating'}))
        with patch.dict(sys.modules, modules):
            result = Parser.parse('推荐科幻', llm)
        self.assertEqual(result['tags'], ['科幻'])
        self.assertEqual(result['sort_by'], 'rating')
        llm.assert_called_once()
        offline.set.assert_called_once()


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(
            os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('Run with explicit CPU-only environment shown in the audit report')
    forbidden = ('torch', 'transformers', 'sentence_transformers', 'faiss', 'tensorflow', 'numpy')
    class BlockModelImports:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in forbidden:
                raise RuntimeError(f'Forbidden model import: {fullname}')
    sys.meta_path.insert(0, BlockModelImports())
    if any(name.split('.')[0] in forbidden for name in sys.modules):
        raise RuntimeError('Model module already imported')
    unittest.main(verbosity=2)
