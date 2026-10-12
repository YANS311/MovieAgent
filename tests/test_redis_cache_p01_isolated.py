"""P0.1 stdlib tests. All Django/ORM/Agent dependencies are replaced by fakes.

Session cookie and real transaction integration are deliberately not claimed.
"""

import ast
import copy
from contextlib import contextmanager
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('p0_fixture', ROOT / 'tests/test_redis_cache_p0_isolated.py')
p0 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p0)
keys = p0.keys


class FakeTransaction:
    """Model savepoint rollback and callbacks at the outermost commit only."""
    def __init__(self, state):
        self.state = state
        self.frames = []
        self.callbacks = []

    @contextmanager
    def atomic(self):
        snapshot = copy.deepcopy(self.state)
        callback_count = len(self.callbacks)
        self.frames.append(snapshot)
        try:
            yield
        except BaseException:
            self.state[:] = snapshot
            del self.callbacks[callback_count:]
            self.frames.pop()
            raise
        else:
            self.frames.pop()
        if not self.frames:
            callbacks, self.callbacks = self.callbacks, []
            # DB state has committed; callback failure cannot undo it.
            for callback in callbacks:
                callback()

    def on_commit(self, callback):
        if self.frames:
            self.callbacks.append(callback)
        else:
            callback()


class Response(dict):
    def __init__(self, data, status=200, **kwargs):
        super().__init__(data)
        self.status_code = status


class FakeFeedbackDB:
    def __init__(self, rows):
        self.rows = rows
        self.fail_create = False

    def filter(self, **kwargs):
        def delete():
            selected = [r for r in self.rows if all(r.get(k) == v for k, v in kwargs.items())]
            self.rows[:] = [r for r in self.rows if r not in selected]
            return len(selected), {}
        return types.SimpleNamespace(delete=delete)

    def create(self, **kwargs):
        if self.fail_create:
            raise RuntimeError('database create failed')
        self.rows.append(kwargs)
        return kwargs


class P01Tests(unittest.TestCase):
    def setUp(self):
        self.cache = p0.FakeCache()
        self.user = types.SimpleNamespace(id=7)
        self.movie = types.SimpleNamespace(id=3, title='Movie3')
        self.logger = Mock()

    def stream(self, session_key):
        events = []
        session = types.SimpleNamespace(session_key=session_key, modified=False)
        def create():
            events.append('session-create')
            session.session_key = 'new-real-session'
            session.modified = True
        session.create = Mock(side_effect=create)
        service = Mock()
        chunks = [{'type': 'chunk', 'content': 'hello'}, {'type': 'done', 'recommended_ids': []}]
        service.stream_chat.return_value = iter(chunks)
        Service = Mock(return_value=service)
        def response(iterator, **kwargs):
            events.append(('response-created', session.session_key, session.modified))
            return types.SimpleNamespace(streaming_content=iterator)
        ns = {'JsonResponse': Response, 'StreamingHttpResponse': response,
              '_sanitize_input': lambda text: text, 'json': json,
              'time': types.SimpleNamespace(sleep=Mock()), 'logger': self.logger}
        p0.extract('myapp/agent_views.py', {'_get_session_key', '_sse_data', 'agent_stream_view'}, ns)
        modules = {'myapp': p0.fake_module('myapp'),
                   'myapp.services': p0.fake_module('myapp.services'),
                   'myapp.services.agent_chat_service': p0.fake_module(
                       'myapp.services.agent_chat_service', AgentChatService=Service)}
        request = types.SimpleNamespace(session=session, method='POST', user=self.user, POST={'msg': 'hello'})
        return ns, request, events, Service, modules, chunks

    def test_sse_first_session_exists_and_is_modified_before_response(self):
        ns, request, events, Service, modules, chunks = self.stream(None)
        response = ns['agent_stream_view'](request)
        self.assertEqual(events, ['session-create', ('response-created', 'new-real-session', True)])
        Service.assert_not_called()
        with patch.dict(sys.modules, modules):
            self.assertEqual(list(response.streaming_content), [ns['_sse_data'](c) for c in chunks])
        request.session.create.assert_called_once()
        Service.assert_called_once_with(self.user, 'new-real-session')

    def test_sse_existing_session_reused_without_creation(self):
        ns, request, events, Service, modules, chunks = self.stream('existing-session')
        response = ns['agent_stream_view'](request)
        self.assertEqual(events, [('response-created', 'existing-session', False)])
        with patch.dict(sys.modules, modules):
            list(response.streaming_content)
        request.session.create.assert_not_called()
        Service.assert_called_once_with(self.user, 'existing-session')

    def test_sse_session_failure_occurs_before_response_construction(self):
        ns, request, events, Service, modules, chunks = self.stream(None)
        request.session.create.side_effect = ConnectionError('session backend offline')
        with self.assertRaises(ConnectionError):
            ns['agent_stream_view'](request)
        self.assertEqual(events, [])
        Service.assert_not_called()

    def feedback(self, name, cache=None):
        rows = [{'user': self.user, 'movie': self.movie, 'feedback_type': 'dislike', 'source': 'old'}]
        db = FakeFeedbackDB(rows)
        tx = FakeTransaction(rows)
        manager = Mock()
        manager.get.return_value = self.movie
        rec = Mock()
        rec.objects.filter.return_value.exclude.return_value.select_related.return_value.order_by.return_value.first.return_value = None
        ns = {'Movie': types.SimpleNamespace(objects=manager, DoesNotExist=LookupError),
              'Rec': rec, 'get_excluded_movie_ids': lambda user: [],
              'get_object_or_404': lambda *args, **kwargs: self.movie,
              'JsonResponse': Response, 'logger': self.logger,
              'cache': cache if cache is not None else self.cache, 'transaction': tx,
              'invalidate_recommendation_cache': keys.invalidate_recommendation_cache,
              'RecommendationCacheInvalidationError': keys.RecommendationCacheInvalidationError}
        path = 'myapp/agent_views.py' if name == 'recommend_feedback_view' else 'myapp/views.py'
        fn = p0.extract(path, {name}, ns)[name]
        modules = {'myapp': p0.fake_module('myapp'),
                   'myapp.models_upgrade': p0.fake_module('myapp.models_upgrade',
                                                        UserFeedback=types.SimpleNamespace(objects=db))}
        request = types.SimpleNamespace(method='POST', user=self.user,
                                        POST={'movie_id': '3', 'feedback_type': 'dislike'})
        return fn, request, db, tx, modules

    def test_feedback_success_invalidates_all_limits_for_three_endpoints(self):
        for name in ('ajax_exclude_add', 'ajax_exclude_remove', 'recommend_feedback_view'):
            with self.subTest(endpoint=name):
                oldkeys = [keys.hybrid_recommendation_cache_key(self.cache, 7, k) for k in (8, 30)]
                for key in oldkeys:
                    self.cache.set(key, 'old-recommendations', 1800)
                other = keys.hybrid_recommendation_cache_key(self.cache, 8, 8)
                self.cache.set(other, 'other-user', 1800)
                self.cache.set('cb_recs_7', 'legacy')
                fn, request, db, tx, modules = self.feedback(name)
                with patch.dict(sys.modules, modules):
                    result = fn(request)
                self.assertEqual(result.status_code, 200)
                for k, old in zip((8, 30), oldkeys):
                    new = keys.hybrid_recommendation_cache_key(self.cache, 7, k)
                    self.assertNotEqual(new, old)
                    self.assertIsNone(self.cache.get(new))
                self.assertEqual(self.cache.get(other), 'other-user')
                self.assertNotIn('cb_recs_7', self.cache.data)

    def test_failed_create_restores_deleted_dislike_and_does_not_invalidate(self):
        fn, request, db, tx, modules = self.feedback('ajax_exclude_add')
        oldrows = copy.deepcopy(db.rows)
        before = keys.hybrid_recommendation_cache_key(self.cache, 7, 8)
        db.fail_create = True
        with patch.dict(sys.modules, modules), self.assertRaises(RuntimeError):
            fn(request)
        self.assertEqual(db.rows, oldrows)
        self.assertEqual(tx.callbacks, [])
        self.assertEqual(keys.hybrid_recommendation_cache_key(self.cache, 7, 8), before)

    def test_outer_rollback_restores_db_and_discards_invalidation(self):
        for name in ('ajax_exclude_add', 'ajax_exclude_remove', 'recommend_feedback_view'):
            with self.subTest(endpoint=name):
                fn, request, db, tx, modules = self.feedback(name)
                oldrows = copy.deepcopy(db.rows)
                before = keys.hybrid_recommendation_cache_key(self.cache, 7, 8)
                with patch.dict(sys.modules, modules), self.assertRaises(RuntimeError):
                    with tx.atomic():
                        fn(request)
                        self.assertEqual(keys.hybrid_recommendation_cache_key(self.cache, 7, 8), before)
                        raise RuntimeError('outer transaction failed')
                self.assertEqual(db.rows, oldrows)
                self.assertEqual(tx.callbacks, [])
                self.assertEqual(keys.hybrid_recommendation_cache_key(self.cache, 7, 8), before)

    def test_nested_commit_invalidates_only_at_outermost_exit(self):
        fn, request, db, tx, modules = self.feedback('ajax_exclude_add')
        before = keys.hybrid_recommendation_cache_key(self.cache, 7, 8)
        with patch.dict(sys.modules, modules), tx.atomic():
            fn(request)
            self.assertEqual(keys.hybrid_recommendation_cache_key(self.cache, 7, 8), before)
            self.assertEqual(len(tx.callbacks), 1)
        self.assertNotEqual(keys.hybrid_recommendation_cache_key(self.cache, 7, 8), before)

    def test_redis_failure_returns_explicit_partial_commit_error(self):
        for name in ('ajax_exclude_add', 'ajax_exclude_remove', 'recommend_feedback_view'):
            with self.subTest(endpoint=name):
                offline = Mock()
                offline.set.side_effect = ConnectionError('redis write failed')
                fn, request, db, tx, modules = self.feedback(name, offline)
                with patch.dict(sys.modules, modules):
                    result = fn(request)
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result['status'], 'partial_success')
                self.assertFalse(result['retry_mutation'])
                self.assertTrue(result['database_committed'])
                self.assertFalse(result['cache_invalidated'])
                self.assertEqual(tx.frames, [])
                if name == 'ajax_exclude_remove':
                    self.assertEqual(db.rows, [])
                else:
                    self.assertEqual(db.rows[-1]['source'], 'recommend_page')

    def test_nested_postcommit_failure_propagates_without_db_rollback(self):
        offline = Mock()
        offline.set.side_effect = ConnectionError('redis write failed')
        fn, request, db, tx, modules = self.feedback('ajax_exclude_add', offline)
        with patch.dict(sys.modules, modules), self.assertRaises(keys.RecommendationCacheInvalidationError):
            with tx.atomic():
                fn(request)
        self.assertEqual(db.rows[-1]['source'], 'recommend_page')

    def test_invalidation_reports_false_write_and_delete_failure(self):
        for failing_method in ('set', 'delete'):
            backend = Mock()
            if failing_method == 'set':
                backend.set.return_value = False
            else:
                backend.delete.side_effect = ConnectionError('delete failed')
            with self.assertRaises(keys.RecommendationCacheInvalidationError):
                keys.invalidate_recommendation_cache(backend, 7)

    def test_admin_clear_is_disabled_without_cache_or_model_mutation(self):
        model_cache = {'existing-model': object()}
        backend = Mock()
        ns = {'cache': backend, 'MODEL_CACHE': model_cache, 'JsonResponse': Response}
        fn = p0.extract('myapp/views.py', {'admin_clear_cache'}, ns)['admin_clear_cache']
        result = fn(types.SimpleNamespace(method='POST'))
        self.assertEqual(result.status_code, 409)
        self.assertEqual(result['code'], 'cache_clear_disabled')
        self.assertIs(ns['MODEL_CACHE'], model_cache)
        self.assertEqual(backend.mock_calls, [])

    def filter_fixture(self, legacy=False):
        class Query:
            def __init__(self, rows):
                self.rows = rows
            def filter(self, **kwargs):
                rows = self.rows
                for field, value in kwargs.items():
                    if field == 'genres__name__icontains':
                        rows = [r for r in rows if value in r['genre']]
                    else:
                        rows = [r for r in rows if r.get(field) == value]
                return Query(rows)
            def exclude(self, id__in):
                return Query([r for r in self.rows if r['id'] not in id__in])
            def order_by(self, *args):
                return Query(sorted(self.rows, key=lambda r: (-r['score'], -r['vote_count'])))
            def values(self, *args):
                return Query([{k: r[k] for k in args} for r in self.rows])
            def first(self):
                return self.rows[0] if self.rows else None
            def __getitem__(self, key):
                return self.rows[key]
        rows = [{'id': 1, 'score': 8, 'vote_count': 20, 'genre': '科幻', 'is_sensitive': True},
                {'id': 2, 'score': 7, 'vote_count': 20, 'genre': '剧情', 'is_sensitive': False},
                {'id': 3, 'score': 8, 'vote_count': 20, 'genre': '科幻', 'is_sensitive': False},
                {'id': 4, 'score': 9, 'vote_count': 20, 'genre': '剧情', 'is_sensitive': False}]
        ns = {'Movie': types.SimpleNamespace(objects=Query(rows)), 'logger': self.logger, 'time': time}
        if legacy:
            source = subprocess.check_output(['git', 'show', 'HEAD:myapp/recommender/rerank.py'],
                                             cwd=ROOT, text=True)
            tree = ast.parse(source)
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'business_filter')
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'legacy-business-filter', 'exec'), ns)
        else:
            p0.extract('myapp/recommender/rerank.py', {'business_filter', 'final_rerank'}, ns)
        log = Mock()
        log.objects.filter.return_value.order_by.return_value.first.return_value = None
        Memory = Mock(return_value=types.SimpleNamespace(get_slots=lambda: {'genre': '科幻'}))
        modules = {'myapp': p0.fake_module('myapp'),
                   'myapp.models_upgrade': p0.fake_module('myapp.models_upgrade', RecommendLog=log),
                   'myapp.utils': p0.fake_module('myapp.utils'),
                   'myapp.utils.content_safety': p0.fake_module('myapp.utils.content_safety', get_minor_excluded_ids=lambda user: set()),
                   'myapp.agent': p0.fake_module('myapp.agent'),
                   'myapp.agent.memory': p0.fake_module('myapp.agent.memory', MemoryManager=Memory)}
        return ns, modules, Memory

    def test_business_filter_matches_legacy_genre_fill_without_memory_lookup(self):
        candidates = [{'movie_id': 1, 'score': .9}, {'movie_id': 2, 'score': .8}]
        old, modules, Memory = self.filter_fixture(legacy=True)
        with patch.dict(sys.modules, modules):
            expected = old['business_filter'](candidates, user=self.user)
        new, modules, Memory = self.filter_fixture()
        with patch.dict(sys.modules, modules):
            result = new['business_filter'](candidates, user=self.user, memory_slots={'genre': '科幻'})
        self.assertEqual(result, expected)
        self.assertEqual([c['movie_id'] for c in result], [2, 3])
        Memory.assert_not_called()

    def test_business_filter_missing_slots_keeps_safe_fallback_without_other_session(self):
        ns, modules, Memory = self.filter_fixture()
        with patch.dict(sys.modules, modules):
            result = ns['business_filter']([{'movie_id': 1}, {'movie_id': 2}], user=self.user)
        self.assertEqual([c['movie_id'] for c in result], [2, 4])
        Memory.assert_not_called()

    def test_final_rerank_propagates_slots_and_preserves_truncation(self):
        ns, modules, Memory = self.filter_fixture()
        with patch.dict(sys.modules, modules):
            result, stats = ns['final_rerank']([{'movie_id': 1}, {'movie_id': 2}], user=self.user,
                                              top_k=2, memory_slots={'genre': '科幻'})
        self.assertEqual([c['movie_id'] for c in result], [2, 3])
        self.assertEqual(stats['final_ids'], [2, 3])

    def test_current_session_slots_produce_distinct_genre_fill(self):
        Memory = p0.extract('myapp/agent/memory.py', {'MemoryManager', 'SLOT_PATTERNS'},
                            {'cache': self.cache, 'memory_cache_key': keys.memory_cache_key,
                             're': p0.re, 'logger': self.logger})['MemoryManager']
        a = Memory(user=self.user, session_id='session-a')
        b = Memory(user=self.user, session_id='session-b')
        a.update_slots({'genre': '科幻'})
        b.update_slots({'genre': '剧情'})
        ns, modules, ignored = self.filter_fixture()
        with patch.dict(sys.modules, modules):
            results = [ns['business_filter']([{'movie_id': 1}, {'movie_id': 2}], user=self.user,
                                             memory_slots=m.get_slots()) for m in (a, b)]
        self.assertEqual([[c['movie_id'] for c in result] for result in results], [[2, 3], [2, 4]])

    def test_rerank_tool_forwards_slots(self):
        tree = ast.parse((ROOT / 'myapp/agent/movie_agent.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RerankTool')
        execute = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'execute')
        ns = {}
        exec(compile(ast.Module(body=[execute], type_ignores=[]), 'RerankTool.execute', 'exec'), ns)
        final = Mock(return_value=([{'movie_id': 3}], {}))
        modules = {'myapp': p0.fake_module('myapp'),
                   'myapp.recommender': p0.fake_module('myapp.recommender'),
                   'myapp.recommender.rerank': p0.fake_module('myapp.recommender.rerank', final_rerank=final)}
        candidates, slots = [{'movie_id': 3}], {'genre': '科幻'}
        with patch.dict(sys.modules, modules):
            ns['execute'](types.SimpleNamespace(name='rerank'), candidates, self.user, 2, slots)
        final.assert_called_once_with(candidates, user=self.user, top_k=2, memory_slots=slots)

    def test_movie_and_react_action_forward_own_memory_slots(self):
        from typing import Tuple
        for path, class_name in (('myapp/agent/movie_agent.py', 'MovieAgent'),
                                 ('myapp/agent/react_agent.py', 'ReActAgent')):
            with self.subTest(agent=class_name):
                tree = ast.parse((ROOT / path).read_text())
                cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
                fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_act')
                ns = {'time': time, 'Tuple': Tuple}
                exec(compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec'), ns)
                tool = Mock()
                tool.execute.return_value = {'output': [{'movie_id': 3}], 'count': 1}
                memory = types.SimpleNamespace(get_slots=Mock(return_value={'genre': '科幻'}))
                agent = types.SimpleNamespace(tools={'rerank': tool}, user=self.user, memory=memory,
                                               _build_enhanced_query=lambda *args: '', _build_query=lambda *args: '')
                args = (agent, 'rerank', 'hello', [{'movie_id': 3}])
                result = ns['_act'](*args, {}) if class_name == 'ReActAgent' else ns['_act'](*args)
                self.assertNotIn('error', result[1])
                self.assertEqual(tool.execute.call_args.kwargs['memory_slots'], {'genre': '科幻'})
                memory.get_slots.assert_called_once()

    def test_workflow_rerank_call_and_diversity_algorithm_unchanged(self):
        tree = ast.parse((ROOT / 'myapp/agent/workflow_agent.py').read_text())
        branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                      and ast.unparse(n.test) == "tool_name == 'rerank'")
        calls = [n for n in ast.walk(branch.body[0]) if isinstance(n, ast.Call)]
        self.assertTrue(any(k.arg == 'memory_slots' and ast.unparse(k.value) == 'self.memory.get_slots()'
                            for call in calls for k in call.keywords))
        before = ast.parse(subprocess.check_output(['git', 'show', 'HEAD:myapp/recommender/rerank.py'],
                                                    cwd=ROOT, text=True))
        after = ast.parse((ROOT / 'myapp/recommender/rerank.py').read_text())
        pick = lambda t: next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == 'diversity_guarantee')
        self.assertEqual(ast.dump(pick(before)), ast.dump(pick(after)))


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
