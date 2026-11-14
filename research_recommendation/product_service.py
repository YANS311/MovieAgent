"""Authenticated product ranking. Accounts and scientific identities remain distinct."""
import hashlib
import threading
import uuid
from collections import Counter
from django.db import transaction, DatabaseError
from django.db.models import Q
from myapp.models import Movie, UserRating, Collect
from .models import (ResearchDataset, ResearchUser, ResearchMovie, ResearchTrainInteraction,
                     ResearchProfile, ProfileGrant, RecommendationTrace, MovieIDBridge,
                     UserPreference, MovieInteraction)
from .contracts import RecommendationError, ModelSpec, M3L, identifier
from .pamk_online import PAMKOnlineBackend
from .pamk_identity import CHECKPOINT_SHA256

_backend = None
_backend_lock = threading.Lock()
from .product_mapping import MAPPING_VERSION, normalized_title, business_identity_verified


def dataset():
    d = ResearchDataset.objects.filter(protocol=M3L, complete=True).first()
    if not d:
        raise RecommendationError('RESEARCH_TRAIN_IMPORT_NOT_READY', 503)
    return d


def backend(d):
    global _backend
    with _backend_lock:
        if _backend is None or _backend.closed or (_backend.process and _backend.process.poll() is not None):
            _backend = PAMKOnlineBackend(ModelSpec('PAMK-Net', M3L, CHECKPOINT_SHA256,
                MAPPING_VERSION, d.manifest['assets']['train_context_csr']['sha256'],
                d.manifest['catalogue_sha256']), admitted_user=None)
        return _backend


def business_seen(user):
    return set(UserRating.objects.filter(user=user).values_list('movie_id', flat=True)) | set(
        Collect.objects.filter(user=user).values_list('movie_id', flat=True)) | set(
        MovieInteraction.objects.filter(user=user, catalogue_model='myapp.Movie', kind='watched').values_list('movie_id', flat=True))


def valid_grant(user, profile_id):
    try:
        pid = uuid.UUID(str(profile_id))
    except (ValueError, TypeError, AttributeError):
        raise RecommendationError('INVALID_PROFILE_ID') from None
    grant = ProfileGrant.objects.select_related('profile').filter(user=user, profile_id=pid,
        is_active=True, profile__is_active=True, profile__demo_only=True, profile__dataset=M3L).first()
    if not grant:
        raise RecommendationError('PROFILE_GRANT_REQUIRED', 403)
    return grant


def recommend(user, payload):
    if not user.is_authenticated:
        raise RecommendationError('AUTHENTICATION_REQUIRED', 401)
    if type(payload) is not dict or set(payload) - {'top_k', 'profile_id', 'model_name'}:
        raise RecommendationError('INVALID_REQUEST_FIELDS')
    k = payload.get('top_k', 10)
    if type(k) is not int or not 1 <= k <= 100:
        raise RecommendationError('INVALID_TOP_K')
    requested_model = payload.get('model_name', 'PAMK-Net')
    if not isinstance(requested_model,str):
        raise RecommendationError('INVALID_MODEL_NAME')
    if requested_model not in {'PAMK-Net','PAMK-Inductive-v1','PAMK-HistoryMean-v1'}:
        raise RecommendationError('MODEL_NOT_AVAILABLE', 503)
    profile_id = payload.get('profile_id')
    if profile_id is None:
        grants = list(ProfileGrant.objects.filter(user=user, is_active=True, profile__is_active=True,
            profile__demo_only=True, profile__dataset=M3L).values_list('profile_id', flat=True)[:2])
        if len(grants) == 1:
            profile_id = grants[0]
        elif len(grants) > 1:
            raise RecommendationError('EXPLICIT_PROFILE_REQUIRED', 409)
    seen = business_seen(user)
    if profile_id is None:
        from django.conf import settings
        explicit_inductive = requested_model in {'PAMK-Inductive-v1','PAMK-HistoryMean-v1'}
        if explicit_inductive and not getattr(settings, 'PAMK_INDUCTIVE_AVAILABLE', False):
            raise RecommendationError('INDUCTIVE_NOT_ENABLED',503)
        result = None
        if explicit_inductive or getattr(settings, 'PAMK_INDUCTIVE_ENABLED', False):
            from .inductive_service import recommend_inductive
            method = ('mean' if requested_model=='PAMK-HistoryMean-v1' else 'learned') if explicit_inductive else getattr(settings,'PAMK_INDUCTIVE_STRATEGY','mean')
            result = recommend_inductive(user,k,seen,explicit_inductive,encoding_method=method)
        if result is None:
            result = cold_start(user,k,seen)
            if explicit_inductive or getattr(settings, 'PAMK_INDUCTIVE_ENABLED', False):
                result['reason'] = 'NO_TRUSTED_MAPPED_POSITIVE_HISTORY'
    else:
        if requested_model in {'PAMK-Inductive-v1','PAMK-HistoryMean-v1'}:
            raise RecommendationError('INDUCTIVE_ACCOUNT_SCOPE_REQUIRED',409)
        grant = valid_grant(user, profile_id)
        p = grant.profile
        d = dataset()
        uid = int(identifier(p.canonical_user_id))
        if p.train_manifest_sha256 != d.manifest['assets']['train_context_csr']['sha256'] or not ResearchUser.objects.filter(dataset=d, canonical_user_id=uid, train_count__gt=0).exists():
            raise RecommendationError('TRAIN_PROFILE_NOT_VERIFIED', 409)
        excluded = list(MovieIDBridge.objects.filter(source_dataset=M3L, mapping_version=MAPPING_VERSION,
            mapping_status='VERIFIED_UNIQUE', django_movie_id__in=seen).values_list('canonical_movie_id', flat=True))
        instance = backend(d)
        try:
            receipt = instance.call(str(uid), k, exclude_canonical_ids=[int(x) for x in excluded])
        except RecommendationError as error:
            if error.code not in {'MODEL_INFERENCE_UNAVAILABLE', 'MODEL_WORKER_UNAVAILABLE', 'MODEL_SESSION_EXPIRED'}:
                raise
            # Read-only scoring has no external side effects. Retry at most once.
            instance.close()
            valid_grant(user, profile_id)
            receipt = backend(d).call(str(uid), k, exclude_canonical_ids=[int(x) for x in excluded])
        valid_grant(user, profile_id)  # Recheck after inference: no stale cached authorization.
        ids = [int(x['canonical_movie_id']) for x in receipt['items']]
        # SQL Train rows are independently checked against the worker's returned IDs.
        if ResearchTrainInteraction.objects.filter(dataset=d, canonical_user_id=uid, canonical_movie_id__in=ids).exists():
            raise RecommendationError('TRAIN_HISTORY_FILTER_MISMATCH', 502)
        items = map_items(receipt['items'], d)
        result = {'source':'PAMK_RESEARCH_PROFILE', 'model_name':'PAMK-Net', 'dataset':M3L,
            'research_demo_only':True, 'identity_link_verified':False,
            'profile_id':str(p.pk), 'canonical_user_id':str(uid),
            'checkpoint_sha256':CHECKPOINT_SHA256, 'mapping_version':MAPPING_VERSION,
            'recommendations':items, 'worker':{a:v for a,v in receipt.items() if a != 'items'},
            'personalization_scope':'Authorized research Train identity; not proof of account identity'}
    result['trace_id'] = str(uuid.uuid4())
    result['NEW_TEST_ACCESS'] = 0
    result['business_context_sha256'] = hashlib.sha256(','.join(map(str, sorted(seen))).encode()).hexdigest()
    try:
        RecommendationTrace.objects.create(id=result['trace_id'], requester=user, outcome=result['source'], provenance=result)
    except DatabaseError:
        raise RecommendationError('TRACE_PERSISTENCE_FAILED', 503) from None
    return result


def cold_start(user, k, seen):
    pref = UserPreference.objects.filter(user=user, catalogue_model='myapp.Movie').first()
    weights = Counter(pref.genres if pref else [])
    positive = list(UserRating.objects.filter(user=user, score__gte=4).values_list('movie_id', flat=True))
    positive += list(Collect.objects.filter(user=user).values_list('movie_id', flat=True))
    if pref:
        positive += pref.movie_ids
    for m in Movie.objects.filter(pk__in=positive).prefetch_related('genres'):
        weights.update(g.name for g in m.genres.all())
    rows = []
    for m in Movie.objects.exclude(pk__in=seen).prefetch_related('genres'):
        affinity = sum(weights[g.name] for g in m.genres.all())
        rows.append((affinity, float(m.score or 0), m.vote_count, -m.pk, m))
    rows.sort(key=lambda x:x[:4], reverse=True)
    return {'source':'METADATA_PREFERENCE' if weights else 'METADATA_COLD_START',
        'model_name':None, 'checkpoint_sha256':None, 'canonical_user_id':None,
        'research_demo_only':False, 'reason':'MODEL_ADAPTATION_DISABLED',
        'business_history_count':len(seen), 'recommendations':[
            {'rank':i,'django_movie_id':m.pk, 'canonical_movie_id':None,'movie_title':m.title,
             'detail_url':f'/movie/{m.pk}/', 'mapping_status':'BUSINESS_CATALOGUE',
             'score':affinity, 'business_rating':rating}
            for i,(affinity,rating,_,__,m) in enumerate(rows[:k],1)]}


def map_items(raw_items,d):
    ids = [int(x['canonical_movie_id']) for x in raw_items]
    metadata = {x.canonical_movie_id:x for x in ResearchMovie.objects.filter(dataset=d, canonical_movie_id__in=ids)}
    bridge_rows = list(MovieIDBridge.objects.filter(source_dataset=M3L,
        mapping_version=MAPPING_VERSION, canonical_movie_id__in=[str(i) for i in ids]))
    bridge_counts = Counter(int(x.canonical_movie_id) for x in bridge_rows)
    bridges = {int(x.canonical_movie_id):x for x in bridge_rows}
    movies = Movie.objects.in_bulk([b.django_movie_id for b in bridges.values() if b.django_movie_id is not None])
    items = []
    for rank, raw in enumerate(raw_items, 1):
        cid = int(raw['canonical_movie_id']); rm = metadata[cid]; b = bridges.get(cid)
        status = b.mapping_status if b else 'UNMAPPED'
        if b and (bridge_counts[cid] != 1 or b.evidence.get('catalogue_sha256') != d.manifest['catalogue_sha256'] or b.source_movie_id != rm.source_movie_id):
            status = 'AMBIGUOUS'
        movie = movies.get(b.django_movie_id) if b and status == 'VERIFIED_UNIQUE' else None
        if status == 'VERIFIED_UNIQUE' and movie is None:
            status = 'UNMAPPED'
        if movie and not business_identity_verified(b,rm,movie):
            movie = None; status = 'AMBIGUOUS'
        items.append(dict(raw, rank=rank, movie_title=rm.title, source_movie_id=rm.source_movie_id,
            mapping_status=status, django_movie_id=movie.pk if movie else None,
            detail_url=f'/movie/{movie.pk}/' if movie else None,
            research_detail_url=f'/research/movies/{cid}/'))
    return items
