"""Versioned Redis/memory model recommendation cache with strict isolation.

Prevents:
- Cross-model cache collisions (PAMK vs UKMAP vs PAMK-AV)
- Cross-user cache sharing
- Stale rankings across model checkpoint or feature updates
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
from typing import Any, Dict, Optional, Tuple

from django.conf import settings
from django.core.cache import caches

logger = logging.getLogger('movie_agent.model_cache')


class JSONSerializer:
    def dumps(self, value: Any) -> bytes:
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()

    def loads(self, value: bytes | str) -> Any:
        return json.loads(value)


def enabled() -> bool:
    return getattr(settings, 'TASK84_MODEL_CACHE_ENABLED', False)


def digest(value: Any) -> str:
    return hashlib.sha256(JSONSerializer().dumps(value)).hexdigest()


def database_scope() -> str:
    db = settings.DATABASES['default']
    return digest([db['ENGINE'], str(db['NAME']), db.get('HOST', ''), str(db.get('PORT', ''))])[:16]


def model_cache_key(
    model_name: str,
    checkpoint_sha: str,
    feature_version: str,
    user_context: str,
    top_k: int,
    candidate_scope: str = 'FULL_TRAIN_UNSEEN',
) -> Tuple[str, str]:
    """Generates a strictly partitioned cache key and owner scope."""
    db_scope = database_scope()
    clean_model = model_name.replace(' ', '_').replace('-', '_').lower()
    ckpt_prefix = digest(checkpoint_sha or 'none')
    feat_ver = digest(feature_version or 'v1')
    user_scope = digest([user_context])[:16]
    scope_hash = digest([candidate_scope])[:8]

    owner_token = digest([clean_model, ckpt_prefix, feat_ver, user_context])[:24]
    key = f'rec_model:{clean_model}:{db_scope}:{ckpt_prefix}:{feat_ver}:{user_scope}:{top_k}:{scope_hash}'
    return key, owner_token


def get_cache_backend():
    alias = getattr(settings, 'TASK84_MODEL_CACHE_ALIAS', 'default')
    try:
        return caches[alias]
    except Exception:
        return caches['default']


def read_model_cache(
    model_name: str,
    checkpoint_sha: str,
    feature_version: str,
    user_context: str,
    top_k: int,
    candidate_scope: str = 'FULL_TRAIN_UNSEEN',
) -> Tuple[Optional[Dict[str, Any]], str]:
    if not enabled():
        return None, 'DISABLED'
    key, owner_token = model_cache_key(model_name, checkpoint_sha, feature_version, user_context, top_k, candidate_scope)
    try:
        raw = get_cache_backend().get(key)
        if raw is None:
            return None, 'MISS'
        envelope = JSONSerializer().loads(raw) if isinstance(raw, (bytes, str)) else raw
        if not isinstance(envelope, dict):
            return None, 'INVALID_PAYLOAD'
        if envelope.get('owner_token') != owner_token or envelope.get('key_identity') != digest(key):
            return None, 'TOKEN_MISMATCH'
        if envelope.get('model_name') != model_name or envelope.get('checkpoint_sha') != checkpoint_sha:
            return None, 'MODEL_MISMATCH'
        payload = deepcopy(envelope.get('payload'))
        if isinstance(payload, dict):
            payload['cache_hit'] = True
            payload['cached_at'] = envelope.get('computed_at')
        return payload, 'HIT'
    except Exception as exc:
        logger.warning('model_cache operation=get error_type=%s', type(exc).__name__)
        return None, 'UNAVAILABLE'


def write_model_cache(
    model_name: str,
    checkpoint_sha: str,
    feature_version: str,
    user_context: str,
    top_k: int,
    payload: Dict[str, Any],
    candidate_scope: str = 'FULL_TRAIN_UNSEEN',
    ttl: int = 120,
) -> str:
    if not enabled():
        return 'DISABLED'
    key, owner_token = model_cache_key(model_name, checkpoint_sha, feature_version, user_context, top_k, candidate_scope)
    try:
        computed_at = datetime.now(timezone.utc).isoformat()
        envelope = {
            'schema': 1,
            'model_name': model_name,
            'checkpoint_sha': checkpoint_sha,
            'feature_version': feature_version,
            'owner_token': owner_token,
            'key_identity': digest(key),
            'computed_at': computed_at,
            'payload': payload,
        }
        serialized = JSONSerializer().dumps(envelope)
        if len(serialized) > 2 * 1024 * 1024:
            return 'SKIPPED_TOO_LARGE'
        effective_ttl = max(1, min(int(ttl), getattr(settings, 'TASK84_MODEL_CACHE_MAX_TTL', 900)))
        get_cache_backend().set(key, serialized, timeout=effective_ttl)
        return 'STORED'
    except Exception as exc:
        logger.warning('model_cache operation=set error_type=%s', type(exc).__name__)
        return 'UNAVAILABLE'


def invalidate_model_cache(
    model_name: str,
    checkpoint_sha: str = '',
    user_context: str = '',
) -> bool:
    """Best-effort invalidation for test and management commands."""
    try:
        backend = get_cache_backend()
        if hasattr(backend, 'delete_pattern'):
            pattern = f'rec_model:{model_name.lower()}*'
            backend.delete_pattern(pattern)
            return True
        return False
    except Exception:
        return False
