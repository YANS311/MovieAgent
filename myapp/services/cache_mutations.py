"""Transaction boundaries and truthful partial-commit responses; no model imports."""

from functools import wraps
import logging
import json

from django.core.cache import cache
from django.db import transaction
from django.http import JsonResponse

from myapp.utils.cache_keys import (
    invalidate_recommendation_cache, invalidate_catalog_cache,
    invalidate_recommendation_caches,
    RecommendationCacheInvalidationError,
)

logger = logging.getLogger('movie_agent.cache')


def schedule_user_invalidation(user_id, using=None):
    if user_id is not None:
        callback = lambda: invalidate_recommendation_cache(cache, user_id)
        if using is None:
            transaction.on_commit(callback)
        else:
            transaction.on_commit(callback, using=using)


def schedule_catalog_invalidation():
    transaction.on_commit(lambda: invalidate_catalog_cache(cache))


def schedule_users_invalidation(user_ids):
    owners = tuple(sorted(set(user_ids) - {None}))
    if owners:
        transaction.on_commit(lambda: invalidate_recommendation_caches(cache, owners))


def invalidation_recovery_requests(failures):
    owners = sorted({failure['user_id'] for failure in failures if failure['scope'] == 'user'})
    requests = ([{'scope': 'user', 'user_id': owners[0]}] if len(owners) == 1 else
                [{'scope': 'users', 'user_ids': owners[i:i + 100]}
                 for i in range(0, len(owners), 100)])
    if any(failure['scope'] == 'catalog' for failure in failures):
        requests.append({'scope': 'catalog'})
    return requests


def _raise_collected_failures(collector):
    if collector['failures']:
        failures = collector['failures']
        scopes = {f['scope'] for f in failures}
        scope = 'users' if scopes == {'user'} else 'catalog' if scopes == {'catalog'} else 'multiple'
        raise RecommendationCacheInvalidationError(
            'Committed admin write, cache invalidation incomplete', scope=scope, failures=failures
        )


def partial_commit_response(exc):
    # HTTP 200 avoids automatic retry of an already committed mutation.
    logger.error('Committed mutation invalidation incomplete scope=%s user_id=%s', exc.scope, exc.user_id)
    return JsonResponse({
        'status': 'partial_success', 'database_committed': True,
        'cache_invalidated': False, 'retry_mutation': False,
        'recovery': 'retry_cache_invalidation_only', 'scope': exc.scope, 'user_id': exc.user_id,
        'failed_scopes': exc.failures,
        'retry_requests': invalidation_recovery_requests(exc.failures),
        'msg': '数据已保存，但缓存失效未完成；请勿重复提交，请仅重试缓存失效。',
    })


def recommendation_write_view(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if request.method != 'POST':
            return view(request, *args, **kwargs)
        try:
            with transaction.atomic():
                response = view(request, *args, **kwargs)
                # Legacy views can catch DB errors and return JSON status=error.
                # Do not commit earlier writes when that happens.
                failed = response.status_code >= 400
                if response.get('Content-Type', '').startswith('application/json'):
                    failed = failed or json.loads(response.content).get('status') == 'error'
                if failed:
                    transaction.set_rollback(True)
            return response
        except RecommendationCacheInvalidationError as exc:
            return partial_commit_response(exc)
    return wrapped


class RecommendationCacheAdminMixin:
    """Default Django admin writes are covered without loading AppConfig.ready."""
    cache_scope = 'user'
    cache_owner_field = 'user_id'

    def _owners(self, obj, include_previous=False):
        owners = {getattr(obj, self.cache_owner_field, None)}
        if include_previous and obj.pk and self.cache_owner_field == 'user_id':
            owners.add(type(obj).objects.filter(pk=obj.pk).values_list('user_id', flat=True).first())
        return owners - {None}

    def _schedule(self, owners, request):
        collector = getattr(request, '_admin_cache_invalidation', None)
        if collector is None:
            if self.cache_scope == 'catalog':
                schedule_catalog_invalidation()
            else:
                schedule_users_invalidation(owners)
            return
        owners = set(owners) - {None}

        def invalidate():
            try:
                if self.cache_scope == 'catalog':
                    if collector['catalog_attempted']:
                        return
                    collector['catalog_attempted'] = True
                    invalidate_catalog_cache(cache)
                else:
                    pending = owners - collector['attempted']
                    collector['attempted'].update(pending)
                    invalidate_recommendation_caches(cache, pending)
            except RecommendationCacheInvalidationError as exc:
                # Keep executing later callbacks; the final callback reports all
                # failures after every committed operation has been attempted.
                collector['failures'].extend(exc.failures)
                logger.error('Committed admin cache failure scopes=%s', exc.failures)

        transaction.on_commit(invalidate)

    def save_model(self, request, obj, form, change):
        with transaction.atomic():
            owners = self._owners(obj, include_previous=change) if self.cache_scope == 'user' else set()
            super().save_model(request, obj, form, change)
            if self.cache_scope == 'user':
                owners.update(self._owners(obj))
            self._schedule(owners, request)

    def delete_model(self, request, obj):
        with transaction.atomic():
            owners = self._owners(obj) if self.cache_scope == 'user' else set()
            super().delete_model(request, obj)
            self._schedule(owners, request)

    def delete_queryset(self, request, queryset):
        with transaction.atomic():
            owners = set(queryset.values_list(self.cache_owner_field, flat=True)) if self.cache_scope == 'user' else set()
            super().delete_queryset(request, queryset)
            self._schedule(owners - {None}, request)

    def save_related(self, request, form, formsets, change):
        super().save_related(request, form, formsets, change)
        if self.cache_scope == 'catalog':
            self._schedule(set(), request)

    def _write_view(self, view, request, *args, **kwargs):
        @recommendation_write_view
        def run(request, *args, **kwargs):
            if request.method != 'POST' or hasattr(request, '_admin_cache_invalidation'):
                return view(request, *args, **kwargs)
            collector = {'failures': [], 'attempted': set(), 'catalog_attempted': False}
            request._admin_cache_invalidation = collector
            try:
                response = view(request, *args, **kwargs)
                transaction.on_commit(lambda: _raise_collected_failures(collector))
                return response
            finally:
                del request._admin_cache_invalidation
        return run(request, *args, **kwargs)

    def changeform_view(self, request, *args, **kwargs):
        return self._write_view(super().changeform_view, request, *args, **kwargs)

    def delete_view(self, request, *args, **kwargs):
        return self._write_view(super().delete_view, request, *args, **kwargs)

    def changelist_view(self, request, *args, **kwargs):
        return self._write_view(super().changelist_view, request, *args, **kwargs)
