"""Blocker regressions: stdlib + AST + FakeCache/transactions, no app imports."""

import ast
import functools
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


def fixture_file(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tests' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p0 = fixture_file('test_redis_cache_p0_isolated')
p01 = fixture_file('test_redis_cache_p01_isolated')
keys = p0.keys


class Response(dict):
    def __init__(self, data, status=200):
        super().__init__(data)
        self.status_code = status
        self.content = json.dumps(data).encode()

    def get(self, key, default=None):
        if key == 'Content-Type':
            return 'application/json'
        return super().get(key, default)


class SelectiveCache(p0.FakeCache):
    def __init__(self, failing=()):
        super().__init__()
        self.failing = set(failing)
        self.attempted = []

    def set(self, key, value, timeout=None):
        if key.startswith('hybrid_rec:v2:') and key.endswith(':generation'):
            user_id = int(key.split(':')[2])
            self.attempted.append(user_id)
            if user_id in self.failing:
                raise ConnectionError('private-backend-details')
        return super().set(key, value, timeout)


def mutation_namespace(cache, tx):
    path = 'myapp/services/cache_mutations.py'
    tree = ast.parse((ROOT / path).read_text())
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    ns = {'cache': cache, 'transaction': tx, 'JsonResponse': Response,
          'wraps': functools.wraps, 'json': json, 'logger': Mock()}
    for name in ('invalidate_recommendation_cache', 'invalidate_recommendation_caches',
                 'invalidate_catalog_cache', 'RecommendationCacheInvalidationError'):
        ns[name] = getattr(keys, name)
    return p0.extract(path, names, ns)


class Record:
    objects = Mock()

    def __init__(self, user_id):
        self.user_id = user_id
        self.pk = user_id


class AdminBase:
    """Database writes are plain list writes guarded by FakeTransaction."""
    def __init__(self, rows):
        self.rows = rows

    def save_model(self, request, obj, form, change):
        self.rows.append(obj.user_id)

    def delete_model(self, request, obj):
        self.rows.remove(obj.user_id)

    def delete_queryset(self, request, queryset):
        self.rows[:] = []

    def save_related(self, *args):
        pass

    def changelist_view(self, request):
        if getattr(request, 'bulk', False):
            self.delete_queryset(request, types.SimpleNamespace(
                values_list=lambda *args, **kwargs: list(self.rows)))
        else:
            for owner in request.owners:
                self.save_model(request, Record(owner), None, False)
        if getattr(request, 'db_failure', False):
            raise RuntimeError('database rollback')
        return Response({'status': 'ok'})


class BlockerTests(unittest.TestCase):
    def admin(self, failing=(), rows=None):
        rows = [] if rows is None else rows
        backend = SelectiveCache(failing)
        tx = p01.FakeTransaction(rows)
        ns = mutation_namespace(backend, tx)
        Admin = type('IsolatedAdmin', (ns['RecommendationCacheAdminMixin'], AdminBase), {})
        return Admin(rows), backend, tx, ns

    def retry(self, backend):
        return p0.extract('myapp/agent_views.py', {'retry_cache_invalidation'}, {
            'cache': backend, 'JsonResponse': Response, 'json': json,
            'invalidate_recommendation_cache': keys.invalidate_recommendation_cache,
            'invalidate_recommendation_caches': keys.invalidate_recommendation_caches,
            'invalidate_catalog_cache': keys.invalidate_catalog_cache,
            'RecommendationCacheInvalidationError': keys.RecommendationCacheInvalidationError,
            'invalidation_recovery_requests': mutation_namespace(backend, p01.FakeTransaction([]))[
                'invalidation_recovery_requests'],
        })['retry_cache_invalidation']

    def request(self, staff=True, **post):
        return types.SimpleNamespace(method='POST', user=types.SimpleNamespace(id=7, is_staff=staff),
                                     POST=post)

    def test_all_changed_production_cache_references_have_real_bindings(self):
        paths = subprocess.check_output([
            'git', 'diff', 'd78390dede4a0371a1e7991f06568d8d390fbdbd', '--name-only'
        ], cwd=ROOT, text=True).splitlines()
        paths += ['myapp/utils/cache_keys.py', 'myapp/utils/cache_access.py', 'myapp/services/cache_mutations.py']
        for path in paths:
            if path.endswith('.py'):
                with self.subTest(path=path):
                    p0.assert_production_cache_bindings(path)

    def test_static_checker_rejects_the_previous_missing_delete_import(self):
        source = (ROOT / 'myapp/views.py').read_text().replace(
            'from myapp.utils.cache_access import business_cache_get, business_cache_set, business_cache_delete',
            'from myapp.utils.cache_access import business_cache_get, business_cache_set')
        with self.assertRaisesRegex(AssertionError, 'business_cache_delete'):
            p0.assert_production_cache_bindings('myapp/views.py', source=source)

    def test_static_checker_rejects_nonexistent_import_export(self):
        with self.assertRaisesRegex(AssertionError, 'unknown cache import'):
            p0.assert_production_cache_bindings('fake.py',
                                               'from myapp.utils.cache_access import missing_cache_export')

    def test_bad_payload_uses_the_delete_declared_by_production_import(self):
        fixture = p0.CacheP0Tests()
        fixture.setUp()
        ns = fixture.hybrid_function()
        ns.pop('business_cache_delete')
        tree = ast.parse((ROOT / 'myapp/views.py').read_text())
        # Resolve only imports actually declared by views.py, never an automatic fallback.
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == 'myapp.utils.cache_access':
                for alias in node.names:
                    ns[alias.asname or alias.name] = getattr(p0.access, alias.name)
        self.assertIn('business_cache_delete', ns)
        key = keys.hybrid_recommendation_cache_key(fixture.cache, 7, 2)
        fixture.cache.set(key, ('obsolete',), 1800)
        result = ns['hybrid_recall_recommend'](types.SimpleNamespace(id=7), top_k=2)
        self.assertEqual([m.id for m in result[0]], [3, 4])
        self.assertIn(key, fixture.cache.deleted)

    def test_malformed_score_and_stats_trigger_safe_rejection(self):
        fixture = p0.CacheP0Tests()
        fixture.setUp()
        ns = fixture.hybrid_function()
        payload = {'schema': 'hybrid_rec_payload_v1', 'items': [{'movie_id': 3, 'rrf_score': 10 ** 1000}],
                   'profile_text': 'profile', 'source_stats': {'vector': 1, 'content': 1, 'model': 1}}
        self.assertIsNone(ns['_restore_hybrid_cache_payload'](payload))
        payload['items'][0]['rrf_score'] = 0.1
        payload['source_stats']['vector'] = 'bad'
        self.assertIsNone(ns['_restore_hybrid_cache_payload'](payload))

    def test_batch_attempts_later_users_and_records_all_failed_scopes(self):
        backend = SelectiveCache((1, 3))
        backend.data['hybrid_rec:v2:99:generation'] = 'unrelated'
        with self.assertRaises(keys.RecommendationCacheInvalidationError) as caught:
            keys.invalidate_recommendation_caches(backend, [3, 1, 2, 1])
        self.assertEqual(backend.attempted, [1, 2, 3])
        self.assertEqual([f['user_id'] for f in caught.exception.failures], [1, 3])
        self.assertTrue(all(f['operation'] == 'generation_set' for f in caught.exception.failures))
        self.assertTrue(all(f['error_type'] == 'ConnectionError' for f in caught.exception.failures))
        self.assertNotIn('private-backend-details', json.dumps(caught.exception.failures))
        self.assertEqual(backend.data['hybrid_rec:v2:99:generation'], 'unrelated')

    def test_list_editable_callbacks_continue_after_first_failure(self):
        admin, backend, tx, ns = self.admin(failing=(1, 3))
        response = admin.changelist_view(types.SimpleNamespace(method='POST', owners=[1, 2, 3]))
        self.assertEqual(backend.attempted, [1, 2, 3])
        self.assertEqual(admin.rows, [1, 2, 3])  # database committed despite callback failure
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['status'], 'partial_success')
        self.assertTrue(response['database_committed'])
        self.assertFalse(response['retry_mutation'])
        self.assertEqual(response['retry_requests'], [{'scope': 'users', 'user_ids': [1, 3]}])

    def test_bulk_delete_attempts_every_owner_once(self):
        admin, backend, tx, ns = self.admin(failing=(1,), rows=[1, 2, 2, 3])
        response = admin.changelist_view(types.SimpleNamespace(method='POST', bulk=True))
        self.assertEqual(admin.rows, [])
        self.assertEqual(backend.attempted, [1, 2, 3])
        self.assertEqual([f['user_id'] for f in response['failed_scopes']], [1])

    def test_save_owner_change_invalidates_previous_and_new_owner(self):
        admin, backend, tx, ns = self.admin()
        Record.objects.filter.return_value.values_list.return_value.first.return_value = 1
        admin.save_model(self.request(), Record(2), None, True)
        self.assertEqual(backend.attempted, [1, 2])

    def test_database_failure_rolls_back_writes_and_all_callbacks(self):
        admin, backend, tx, ns = self.admin()
        with self.assertRaises(RuntimeError):
            admin.changelist_view(types.SimpleNamespace(method='POST', owners=[1, 2], db_failure=True))
        self.assertEqual(admin.rows, [])
        self.assertEqual(backend.attempted, [])
        self.assertEqual(tx.callbacks, [])

    def test_outer_transaction_rollback_discards_admin_callbacks(self):
        admin, backend, tx, ns = self.admin()
        with self.assertRaises(RuntimeError):
            with tx.atomic():
                admin.changelist_view(types.SimpleNamespace(method='POST', owners=[1, 2]))
                self.assertEqual(backend.attempted, [])
                raise RuntimeError('outer rollback')
        self.assertEqual(admin.rows, [])
        self.assertEqual(backend.attempted, [])

    def test_outer_commit_attempts_all_users_before_reporting_failure(self):
        admin, backend, tx, ns = self.admin(failing=(1,))
        with self.assertRaises(keys.RecommendationCacheInvalidationError) as caught:
            with tx.atomic():
                admin.changelist_view(types.SimpleNamespace(method='POST', owners=[1, 2]))
                self.assertEqual(backend.attempted, [])
        self.assertEqual(admin.rows, [1, 2])
        self.assertEqual(backend.attempted, [1, 2])
        self.assertEqual([f['user_id'] for f in caught.exception.failures], [1])

    def test_nested_savepoint_rollback_discards_only_rolled_back_owner(self):
        admin, backend, tx, ns = self.admin()
        with tx.atomic():
            try:
                with tx.atomic():
                    admin.save_model(self.request(), Record(1), None, False)
                    raise RuntimeError('savepoint rollback')
            except RuntimeError:
                pass
            admin.save_model(self.request(), Record(2), None, False)
        self.assertEqual(admin.rows, [2])
        self.assertEqual(backend.attempted, [2])

    def test_recovery_retries_only_recorded_users_without_database_mutations(self):
        backend = SelectiveCache()
        fn = self.retry(backend)
        response = fn(self.request(scope='users', user_ids='[1,3,1]'))
        self.assertEqual(response['status'], 'ok')
        self.assertFalse(response['database_mutation'])
        self.assertEqual(backend.attempted, [1, 3])

    def test_batch_recovery_failure_lists_only_remaining_scopes(self):
        backend = SelectiveCache((1,))
        response = self.retry(backend)(self.request(scope='users', user_ids='[1,2,3]'))
        self.assertEqual(backend.attempted, [1, 2, 3])
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response['database_mutation'])
        self.assertFalse(response['retry_mutation'])
        self.assertEqual(response['retry_requests'], [{'scope': 'user', 'user_id': 1}])

    def test_non_staff_can_recover_their_single_partial_failure(self):
        backend = SelectiveCache()
        ns = mutation_namespace(backend, p01.FakeTransaction([]))
        response = ns['partial_commit_response'](
            keys.RecommendationCacheInvalidationError('failed', user_id=7))
        request_data = response['retry_requests'][0]
        self.assertEqual(request_data, {'scope': 'user', 'user_id': 7})
        retry_result = self.retry(backend)(self.request(staff=False, **request_data))
        self.assertEqual(retry_result['status'], 'ok')
        self.assertEqual(backend.attempted, [7])

    def test_staff_catalog_retry_preserves_sessions(self):
        backend = SelectiveCache()
        backend.data['session-test'] = 'identity'
        response = self.retry(backend)(self.request(scope='catalog'))
        self.assertEqual(response['status'], 'ok')
        self.assertFalse(response['database_mutation'])
        self.assertIn('recommendation_catalog:v1:generation', backend.data)
        self.assertEqual(backend.data['session-test'], 'identity')

    def test_non_staff_cannot_batch_retry_or_invalidate_other_users_or_catalog(self):
        backend = SelectiveCache()
        fn = self.retry(backend)
        for post in ({'scope': 'users', 'user_ids': '[7,8]'},
                     {'scope': 'user', 'user_id': '8'}, {'scope': 'catalog'}):
            with self.subTest(post=post):
                self.assertEqual(fn(self.request(staff=False, **post)).status_code, 403)
        self.assertEqual(backend.attempted, [])

    def test_batch_retry_rejects_invalid_ids_before_any_invalidation(self):
        backend = SelectiveCache()
        fn = self.retry(backend)
        for value in ('[]', '[0]', '[true]', '[-1]', '["1"]', '{}', 'not-json'):
            self.assertEqual(fn(self.request(scope='users', user_ids=value)).status_code, 400)
        self.assertEqual(backend.attempted, [])

    def test_large_failure_report_splits_authorized_recovery_into_bounded_batches(self):
        ns = mutation_namespace(SelectiveCache(), p01.FakeTransaction([]))
        requests = ns['invalidation_recovery_requests']([
            {'scope': 'user', 'user_id': i} for i in range(1, 202)])
        self.assertEqual([len(r['user_ids']) for r in requests], [100, 100, 1])

    def test_state_mutation_routes_have_post_auth_and_csrf_design(self):
        required = {
            'myapp/views.py': ('add_collect', 'remove_collect', 'admin_comment_delete',
                              'ajax_exclude_add', 'ajax_exclude_remove', 'chat_clear_history'),
            'myapp/agent_views.py': ('recommend_feedback_view', 'retry_cache_invalidation',
                                    'agent_api_view', 'agent_stream_view', 'ajax_clear_chat',
                                    'ajax_new_chat', 'ajax_edit_last', 'ajax_summarize_profile', 'ajax_agent_debug'),
        }
        for path, names in required.items():
            tree = ast.parse((ROOT / path).read_text())
            functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
            for name in names:
                decorators = [ast.unparse(d) for d in functions[name].decorator_list]
                with self.subTest(view=name):
                    self.assertIn('require_POST', decorators)
                    self.assertIn('login_required', decorators)
                    self.assertNotIn('csrf_exempt', decorators)
        self.assertIn('if not request.user.is_staff', ast.unparse(functions['retry_cache_invalidation']))
        settings = (ROOT / 'movie/settings.py').read_text()
        self.assertIn('django.middleware.csrf.CsrfViewMiddleware', settings)

    def test_async_chat_keeps_coroutine_and_csrf_and_post_guard(self):
        tree = ast.parse((ROOT / 'myapp/views.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'ajax_chat')
        self.assertTrue(any(isinstance(n, ast.If) and "request.method != 'POST'" in ast.unparse(n.test)
                            for n in function.body))
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and ast.unparse(n.targets[0]) == 'ajax_chat.csrf_exempt')
        self.assertIs(assignment.value.value, False)

    def test_changed_frontend_delete_calls_send_post_and_csrf(self):
        admin_template = (ROOT / 'templates/admin_comments.html').read_text()
        self.assertIn('method="post" action="{% url \'admin_comment_delete\' rating.id %}"', admin_template)
        self.assertIn('{% csrf_token %}', admin_template)
        chat_template = (ROOT / 'templates/chat.html').read_text()
        self.assertIn('$.post("{% url \'chat_clear_history\' %}"', chat_template)
        self.assertIn("csrfmiddlewaretoken: '{{ csrf_token }}'", chat_template)


if __name__ == '__main__':
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or any(
            os.environ.get(k) != '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise RuntimeError('Explicit CPU-only environment required')
    class BlockModelImports:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in {'torch', 'transformers', 'sentence_transformers', 'faiss', 'numpy', 'tensorflow'}:
                raise RuntimeError(f'Forbidden import: {fullname}')
    sys.meta_path.insert(0, BlockModelImports())
    unittest.main(verbosity=2)
