"""Recomputable ordinary-account cache; no research result/permission caching."""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import logging
import uuid
from django.apps import apps
from django.conf import settings
from django.core.cache import caches
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.db.models.signals import post_save, post_delete, m2m_changed

logger = logging.getLogger('movie_agent.personal_cache')


class JSONSerializer:
    def dumps(self, value):
        return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()

    def loads(self, value):
        return json.loads(value)


def enabled():
    return getattr(settings, 'TASK63_CACHE_ENABLED', False)


def digest(value):
    return hashlib.sha256(JSONSerializer().dumps(value)).hexdigest()


def database_scope():
    db = settings.DATABASES['default']
    return digest([db['ENGINE'], str(db['NAME']), db.get('HOST', ''), str(db.get('PORT', '')),
                   settings.AUTH_USER_MODEL])[:24]


def catalogue_generation():
    from .models import CatalogueCacheState
    label = getattr(settings, 'TASK60_MOVIE_MODEL', 'myapp.Movie')
    state, _ = CatalogueCacheState.objects.get_or_create(catalogue_model=label)
    return label, str(state.generation)


def invalidate_catalogue(using='default'):
    """Call after bulk imports/QuerySet.update, which do not emit save signals."""
    from .models import CatalogueCacheState
    label = getattr(settings, 'TASK60_MOVIE_MODEL', 'myapp.Movie')
    CatalogueCacheState.objects.using(using).update_or_create(
        catalogue_model=label, defaults={'generation': uuid.uuid4()})


def connect_catalogue_signals():
    movie = apps.get_model(getattr(settings, 'TASK60_MOVIE_MODEL', 'myapp.Movie'))

    def changed(sender, using='default', **kwargs):
        if enabled():
            transaction.on_commit(lambda: invalidate_catalogue(using), using=using)

    for signal in (post_save, post_delete):
        signal.connect(changed, sender=movie, weak=False,
                       dispatch_uid=f'task63:{signal}:{movie._meta.label_lower}')
    if any(f.name == 'genres' and f.many_to_many for f in movie._meta.get_fields()):
        relation = movie._meta.get_field('genres')
        for signal in (post_save, post_delete):
            signal.connect(changed, sender=relation.related_model, weak=False,
                           dispatch_uid=f'task63:{signal}:{relation.related_model._meta.label_lower}')

        def genres_changed(sender, action, using='default', **kwargs):
            if action in ('post_add', 'post_remove', 'post_clear'):
                changed(sender, using=using)

        m2m_changed.connect(genres_changed, sender=relation.remote_field.through,
                            weak=False, dispatch_uid='task63:movie-genres:' + movie._meta.label_lower)


def backend():
    return caches[getattr(settings, 'TASK63_CACHE_ALIAS', 'recommendation')]


def read(key):
    try:
        value = backend().get(key)
        return value, 'HIT' if value is not None else 'MISS'
    except Exception as exc:
        # Never log keys, Redis URLs, payloads or exception text.
        logger.warning('personal_cache operation=get error_type=%s', type(exc).__name__)
        return None, 'UNAVAILABLE'


def write(key, value, ttl):
    try:
        # Bound cache memory; JSON primitives only, no ORM/user/session objects.
        if len(JSONSerializer().dumps(value)) > 4*1024*1024:
            return 'SKIPPED_SIZE'
        result = backend().set(key, value, timeout=max(1, min(int(ttl), 900)))
        return 'REJECTED' if result is False else 'STORED'
    except Exception as exc:
        logger.warning('personal_cache operation=set error_type=%s', type(exc).__name__)
        return 'UNAVAILABLE'


def catalogue_key(label, generation):
    return f'public:catalogue:v1:{database_scope()}:{digest(label)[:16]}:{generation}'


def cached_catalogue(loader):
    from .page_catalogue import MovieCard
    label, generation = catalogue_generation()
    key = catalogue_key(label, generation)
    cached, _ = read(key)
    if isinstance(cached, dict) and cached.get('scope') == 'public_movie_metadata' and isinstance(cached.get('cards'), list):
        try:
            return [MovieCard(**{**row, 'genres': tuple(row['genres'])}) for row in cached['cards']]
        except (KeyError, TypeError, ValueError):
            pass
    cards = loader()
    write(key, {'scope': 'public_movie_metadata', 'cards': [asdict(c) for c in cards]},
          getattr(settings, 'TASK63_CATALOGUE_TTL', 300))
    return cards


def recommendation_key(user, preference, label, generation):
    owner = digest([settings.AUTH_USER_MODEL, str(user.pk)])[:24]
    preference_identity = None if preference is None else [
        preference.genres, preference.movie_ids, preference.catalogue_model,
        preference.updated_at.isoformat()]
    revision = digest(preference_identity)
    key = f'ordinary:genre-v1:{database_scope()}:{owner}:{digest(label)[:16]}:{generation}:{revision}'
    return key, owner


def recommendation(user, preference, loader, ranker):
    if not user.is_authenticated or (preference is not None and preference.user_id != user.pk):
        raise PermissionDenied
    label, generation = catalogue_generation()
    key, owner = recommendation_key(user, preference, label, generation)
    cached, status = read(key)
    valid = (isinstance(cached, dict) and cached.get('owner_scope') == owner and
             cached.get('key_identity') == digest(key) and cached.get('schema') == 1 and
             isinstance(cached.get('computed_at'), str) and
             isinstance(cached.get('provenance'), dict))
    if valid:
        data = cached['provenance']
        valid = (data.get('identity_mode') == 'DJANGO_ACCOUNT' and data.get('source') == 'METADATA_COLD_START'
                 and data.get('research_profile_id') is None and data.get('model_name') is None
                 and data.get('checkpoint_identity') is None
                 and isinstance(data.get('results'), list) and len(data['results']) <= 20)
    if valid:
        provenance = deepcopy(cached['provenance'])
        provenance['cache'] = {'result': 'HIT', 'computed_at': cached['computed_at'], 'schema': 1}
        return provenance
    if status == 'HIT':
        status = 'REJECTED'
    _, provenance = ranker(loader(), preference)
    computed_at = datetime.now(timezone.utc).isoformat()
    envelope = {'schema': 1, 'owner_scope': owner, 'key_identity': digest(key),
                'computed_at': computed_at, 'provenance': deepcopy(provenance)}
    stored = write(key, envelope, getattr(settings, 'TASK63_RECOMMENDATION_TTL', 120))
    provenance['cache'] = {'result': status, 'write': stored, 'computed_at': computed_at, 'schema': 1}
    return provenance
