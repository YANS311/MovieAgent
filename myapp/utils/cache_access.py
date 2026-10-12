"""Only recomputable business cache operations may fail open. No Session use."""

import logging

logger = logging.getLogger('movie_agent.cache')


def business_cache_get(backend, key, default=None):
    if key is None:
        return default
    try:
        return backend.get(key, default)
    except Exception as exc:
        logger.warning('Business cache unavailable operation=get error_type=%s', type(exc).__name__)
        return default


def business_cache_set(backend, key, value, timeout):
    if key is None:
        return False
    try:
        result = backend.set(key, value, timeout)
        if result is False:
            logger.warning('Business cache write rejected')
        return result is not False
    except Exception as exc:
        logger.warning('Business cache unavailable operation=set error_type=%s', type(exc).__name__)
        return False


def business_cache_delete(backend, key):
    if key is None:
        return False
    try:
        result = backend.delete(key)
        if result is False:
            logger.warning('Business cache delete rejected')
        return result is not False
    except Exception as exc:
        logger.warning('Business cache unavailable operation=delete error_type=%s', type(exc).__name__)
        return False


class MemoryCacheUnavailable(RuntimeError):
    """Persisted preference state cannot be trusted; do not reset it silently."""


def memory_cache_operation(backend, operation, key, *args):
    try:
        result = getattr(backend, operation)(key, *args)
        if operation == 'set' and result is False:
            raise RuntimeError('Memory write rejected')
        return result
    except Exception as exc:
        logger.error('Preference memory unavailable operation=%s error_type=%s', operation, type(exc).__name__)
        raise MemoryCacheUnavailable('会话偏好缓存不可用；本次偏好操作未确认完成。') from exc
