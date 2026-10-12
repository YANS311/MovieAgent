"""Versioned business cache keys; no Django/model imports or global cache access."""

import hashlib
import json
import logging
import uuid

RECOMMENDATION_TTL = 1800
logger = logging.getLogger('movie_agent.cache')


class RecommendationCacheInvalidationError(RuntimeError):
    """DB commit succeeded, but recommendation invalidation is incomplete."""

    def __init__(self, message, user_id=None, scope='user', operation='unknown',
                 error_type='unknown', failures=None):
        super().__init__(message)
        self.user_id = user_id
        self.scope = scope
        self.failures = list(failures) if failures is not None else [{
            'scope': scope, 'user_id': user_id, 'operation': operation,
            'error_type': error_type,
        }]


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def intent_cache_key(user_input):
    return f"intent:v2:{_digest(user_input.strip())}"


def query_rewrite_cache_key(query):
    return f"query_rewrite:v2:{_digest(query.strip())}"


def memory_cache_key(session_id, user_id=None):
    # Legacy user-only fallbacks are not session identifiers. Keep such callers
    # local to their MemoryManager instance rather than sharing Redis state.
    if not isinstance(session_id, str) or not session_id.strip():
        return None
    if session_id in {'default', 'user_anon', 'user_None', f'user_{user_id}'}:
        return None
    identity = json.dumps([user_id, session_id], ensure_ascii=False, separators=(',', ':'))
    return f"agent_memory:v2:{_digest(identity)}"


def kg_subgraph_cache_key(topic_keyword, user_history_mids, max_triples):
    # Neo4j IN is order/duplicate insensitive, but int and str IDs can have
    # different query semantics. Preserve types and leave query inputs intact.
    history = sorted({json.dumps(mid, ensure_ascii=False, separators=(',', ':'))
                      for mid in (user_history_mids or [])})
    history_json = json.dumps(history, ensure_ascii=False, separators=(',', ':'))
    return f"kg_subgraph:v4:{_digest(topic_keyword)}:{_digest(history_json)}:{max_triples}"


def _recommendation_generation_key(user_id):
    return f"hybrid_rec:v2:{user_id}:generation"


def _generation(cache_backend, generation_key):
    generation = cache_backend.get(generation_key)
    if generation is None:
        # add chooses one winner for concurrent misses. Random generations also
        # prevent old entries becoming reachable after token eviction/expiry.
        cache_backend.add(generation_key, uuid.uuid4().hex, RECOMMENDATION_TTL)
        generation = cache_backend.get(generation_key)
    return generation


def catalog_business_cache_key(cache_backend, base_key):
    try:
        generation = _generation(cache_backend, 'recommendation_catalog:v1:generation')
        return f'{base_key}:catalog:{generation}' if generation is not None else None
    except Exception as exc:
        logger.warning('Business cache generation unavailable error_type=%s', type(exc).__name__)
        return None


def user_business_cache_key(cache_backend, user_id, family, parameter=''):
    try:
        generation = _generation(cache_backend, _recommendation_generation_key(user_id))
        if generation is None:
            return None
        return catalog_business_cache_key(cache_backend, f'{family}:v3:{user_id}:{generation}:{parameter}')
    except Exception as exc:
        logger.warning('Business cache generation unavailable error_type=%s', type(exc).__name__)
        return None


def hybrid_recommendation_cache_key(cache_backend, user_id, top_k):
    # Separate lightweight payloads from the prior ORM-object cache family.
    return user_business_cache_key(cache_backend, user_id, 'hybrid_rec_payload', f'top_k:{top_k}')


def multi_recall_cache_key(cache_backend, user_id, query_text, top_k, weights):
    normalized_weights = [float(weight) for weight in weights] if weights is not None else None
    parameters = json.dumps(
        [query_text or '', top_k, normalized_weights], ensure_ascii=False,
        separators=(',', ':'), sort_keys=True,
    )
    return user_business_cache_key(
        cache_backend, user_id, 'multi_recall', f'params:{_digest(parameters)}'
    )


def invalidate_catalog_cache(cache_backend):
    """Catalog changes affect every user's candidates; no keys are deleted."""
    try:
        if cache_backend.set('recommendation_catalog:v1:generation', uuid.uuid4().hex,
                             RECOMMENDATION_TTL) is False:
            raise RuntimeError('Catalog generation write rejected')
    except Exception as exc:
        raise RecommendationCacheInvalidationError('Catalog committed, invalidation incomplete',
                                                   scope='catalog', operation='generation_set',
                                                   error_type=type(exc).__name__) from exc


def invalidate_recommendation_cache(cache_backend, user_id):
    """Call after DB commit; failures propagate, never imply DB rollback.

    This is not a distributed transaction. A failed Redis write can leave old
    entries reachable until expiry; callers must report the partial failure.
    """
    operation = 'generation_set'
    try:
        written = cache_backend.set(_recommendation_generation_key(user_id), uuid.uuid4().hex,
                                    RECOMMENDATION_TTL)
        if written is False:
            raise RuntimeError('Recommendation generation write rejected')
        # Remove this user's legacy entries only; superseded v2 entries expire.
        for family in ('hybrid_rec', 'cb_recs', 'agent_recall'):
            operation = f'legacy_delete:{family}'
            cache_backend.delete(f"{family}_{user_id}")
    except Exception as exc:
        raise RecommendationCacheInvalidationError(
            'Data committed, recommendation cache invalidation incomplete', user_id=user_id,
            operation=operation, error_type=type(exc).__name__,
        ) from exc


def invalidate_recommendation_caches(cache_backend, user_ids):
    """Attempt every affected user; report all failures without raw secrets."""
    failures = []
    for user_id in sorted(set(user_ids) - {None}):
        try:
            invalidate_recommendation_cache(cache_backend, user_id)
        except RecommendationCacheInvalidationError as exc:
            failures.extend(exc.failures)
    if failures:
        for failure in failures:
            logger.error('Cache invalidation failed scope=%s user_id=%s operation=%s error_type=%s',
                         failure['scope'], failure['user_id'], failure['operation'], failure['error_type'])
        raise RecommendationCacheInvalidationError(
            'Data committed, affected user invalidations incomplete', scope='users', failures=failures
        )
