"""Unified Recommender Adapter for Chapter 5: Multi-model switching, strict identity resolution, and caching.

Supports:
- Models: UKMAP v2.2, PAMK-Net, PAMK-AV
- User Identity Categories:
  1. CANONICAL_M3L_BENCHMARK: Verified M3L canonical benchmark users
  2. BUSINESS_DEMO_GRANTED: Django users with explicit ProfileGrant for demonstration
  3. BUSINESS_UNMAPPED: Business accounts without M3L canonical mapping (clean cold-start fallback)
  4. NEW_USER_COLD_START: Zero-history new users (Train-only popularity / genre fallback)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

from django.conf import settings
from django.contrib.auth.models import AnonymousUser
from django.db import DatabaseError

from .contracts import (
    M3L,
    ML1M,
    MODEL_DATASETS,
    ModelSpec,
    PAMKAdapter,
    PAMKAVAdapter,
    RecommendationError,
    RecommendationItem,
    RecommendRequest,
    RecommendResponse,
    SKBHIFAdapter,
    UKMAPAdapter,
)
from .model_cache import digest, read_model_cache, write_model_cache
from .models import (
    ProfileGrant,
    RecommendationTrace,
    ResearchProfile,
    UserPreference,
)
from .page_catalogue import PageCatalogue
from .personal import catalogue as load_personal_catalogue, rank as rank_personal
from .services import DjangoMovieCatalogue, MappingService, ProfileService

logger = logging.getLogger('movie_agent.unified_adapter')

SUPPORTED_MODELS = ('UKMAP', 'PAMK-Net', 'PAMK-AV')


class UserIdentityCategory(str, Enum):
    CANONICAL_M3L_BENCHMARK = 'CANONICAL_M3L_BENCHMARK'
    BUSINESS_DEMO_GRANTED = 'BUSINESS_DEMO_GRANTED'
    BUSINESS_UNMAPPED = 'BUSINESS_UNMAPPED'
    NEW_USER_COLD_START = 'NEW_USER_COLD_START'


@dataclass(frozen=True)
class IdentityResolution:
    category: UserIdentityCategory
    profile: Optional[ResearchProfile]
    message: str
    can_execute_model: bool


class UnifiedRecommenderAdapter:
    """Central facade for model selection, identity resolution, caching, and fallback."""

    def __init__(
        self,
        adapters: Optional[Dict[str, Any]] = None,
        mapping_service: Optional[MappingService] = None,
        profile_manifest: Optional[Any] = None,
    ):
        self.adapters = adapters or {}
        self.mapping_service = mapping_service
        self.profile_manifest = profile_manifest

    def resolve_user_identity(
        self,
        user: Any,
        requested_profile_id: Optional[str] = None,
    ) -> IdentityResolution:
        """Strictly categorizes user and prohibits unauthorized M3L vector borrowing."""
        if user is None or isinstance(user, AnonymousUser) or not getattr(user, 'is_authenticated', False):
            raise RecommendationError('AUTHENTICATION_REQUIRED', 401)

        # 1. Check for explicit, authorized ProfileGrant
        if requested_profile_id:
            try:
                profile = ProfileService.authorized(user, requested_profile_id)
                cat = UserIdentityCategory.BUSINESS_DEMO_GRANTED
                return IdentityResolution(
                    category=cat,
                    profile=profile,
                    message='Authorized research profile verified via explicit ProfileGrant.',
                    can_execute_model=True,
                )
            except RecommendationError:
                raise
        else:
            # Check if user has any active grant
            grant = ProfileGrant.objects.filter(user=user, is_active=True, profile__is_active=True).first()
            if grant is not None:
                profile = grant.profile
                return IdentityResolution(
                    category=UserIdentityCategory.BUSINESS_DEMO_GRANTED,
                    profile=profile,
                    message='Active ProfileGrant located for user.',
                    can_execute_model=True,
                )

        # 2. Check preferences for business account
        pref = UserPreference.objects.filter(user=user).first()
        has_pref = pref and (pref.genres or pref.movie_ids)
        if has_pref:
            return IdentityResolution(
                category=UserIdentityCategory.BUSINESS_UNMAPPED,
                profile=None,
                message='Regular business user with preferences but NO canonical M3L mapping. Route to metadata cold start.',
                can_execute_model=False,
            )

        # 3. New user cold start
        return IdentityResolution(
            category=UserIdentityCategory.NEW_USER_COLD_START,
            profile=None,
            message='Zero-history new user without M3L mapping. Route to non-personalized catalog exploration.',
            can_execute_model=False,
        )

    def execute_cold_start_fallback(
        self,
        user: Any,
        top_k: int,
        resolution: IdentityResolution,
    ) -> Dict[str, Any]:
        """Provides verified metadata-based cold-start ranking without faking model identities."""
        pref = UserPreference.objects.filter(user=user).first()
        cards = load_personal_catalogue()
        results, provenance = rank_personal(cards, pref)
        top_results = results[:top_k]

        recommendations = []
        for i, r in enumerate(top_results):
            movie = r['movie']
            recommendations.append({
                'canonical_movie_id': str(movie.id),
                'django_movie_id': movie.id,
                'rank': i + 1,
                'score': float(r['score']),
                'movie_title': movie.title,
                'mapping_status': 'VERIFIED_UNIQUE' if movie.id else 'UNMAPPED',
                'matched_genres': r.get('matched_genres', []),
                'reason': r.get('reason', ''),
            })

        return {
            'model_name': 'METADATA_COLD_START_FALLBACK',
            'dataset': 'BUSINESS_CATALOGUE',
            'checkpoint_identity': 'none',
            'profile_source': {
                'mode': resolution.category.value,
                'user_id': str(user.pk),
                'resolution_note': resolution.message,
                'demo_only': False,
            },
            'recommendations': recommendations,
            'inference_mode': 'RULE_BASED_COLD_START',
            'mapping_version': 'direct_business_catalogue_v1',
            'candidate_scope': f'BUSINESS_CATALOGUE:{len(cards)}',
            'source_reference': provenance.get('strategy', 'METADATA_COLD_START'),
            'output_artifact_identity': None,
            'mapping_summary': {
                'VERIFIED_UNIQUE': len(recommendations),
                'AMBIGUOUS': 0,
                'UNMAPPED': 0,
            },
            'telemetry': None,
        }

    def recommend(
        self,
        user: Any,
        top_k: int = 20,
        model_version: str = 'PAMK-Net',
        profile_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Unified entry point for recommendation serving across all 4 user classes and 3 models."""
        rid = request_id or str(uuid.uuid4())

        # 1. Identity resolution
        resolution = self.resolve_user_identity(user, profile_id)
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise RecommendationError('INVALID_TOP_K')
        if model_version not in SUPPORTED_MODELS:
            raise RecommendationError('UNSUPPORTED_MODEL', 400)
        if model_version == 'PAMK-AV':
            raise RecommendationError('PAMK_AV_INFERENCE_PENDING_VERIFICATION', 503)
        if not resolution.can_execute_model:
            logger.info('User %s routed to cold start: %s', user.pk, resolution.category)
            payload = self.execute_cold_start_fallback(user, top_k, resolution)
            payload['request_id'] = rid
            return payload

        # 2. Model version verification
        if model_version not in SUPPORTED_MODELS:
            raise RecommendationError('UNSUPPORTED_MODEL', 400)

        profile = resolution.profile
        assert profile is not None

        # Verify dataset matches model
        if profile.dataset != MODEL_DATASETS.get(model_version):
            raise RecommendationError('PROFILE_DATASET_MISMATCH', 409)

        if self.profile_manifest is not None:
            self.profile_manifest.verify(profile)

        adapter = self.adapters.get(model_version)
        if adapter is None:
            raise RecommendationError('REAL_MODEL_NOT_VERIFIED', 503)
        if self.mapping_service is not None:
            mapping = self.mapping_service
            if (mapping.manifest.dataset != adapter.spec.dataset
                    or mapping.manifest.identity != adapter.spec.catalogue_manifest_identity
                    or mapping.version != adapter.spec.mapping_version):
                raise RecommendationError('MAPPING_NOT_CONFIGURED', 503)

        # 3. Model Cache check
        ckpt_sha = getattr(adapter.spec, 'checkpoint_identity', '')
        # Both the research history and the actual feature/export version must be
        # content-addressed by the operator. A mapping label alone is insufficient.
        feature_version = digest([
            adapter.spec.train_manifest_identity, adapter.spec.catalogue_manifest_identity,
            adapter.spec.mapping_version, getattr(settings, 'TASK84_FEATURE_VERSION', 'UNVERSIONED'),
        ])
        user_ctx = digest([str(user.pk), str(profile.pk), profile.canonical_user_id,
                           profile.train_manifest_sha256, profile.profile_source])
        candidate_scope = 'FULL_TRAIN_UNSEEN'
        cached_result, cache_status = read_model_cache(
            model_name=model_version,
            checkpoint_sha=ckpt_sha,
            feature_version=feature_version,
            user_context=user_ctx,
            top_k=top_k,
            candidate_scope=candidate_scope,
        )
        if cached_result is not None:
            if self.mapping_service is not None:
                current = self.mapping_service.resolve_many([
                    row['canonical_movie_id'] for row in cached_result['recommendations']])
                fields = ('django_movie_id', 'movie_title', 'mapping_status')
                if any(any(row.get(k) != current[row['canonical_movie_id']].get(k) for k in fields)
                       for row in cached_result['recommendations']):
                    raise RecommendationError('CACHED_MOVIE_MAPPING_CHANGED', 409)
            cached_result['request_id'] = rid
            cached_result['cache_status'] = cache_status
            self._record_trace(user, rid, cached_result)
            return cached_result

        # 4. Model execution
        batch = adapter.infer(profile, top_k)

        # 5. Mapping resolution
        if self.mapping_service is not None:
            mapped = self.mapping_service.resolve_many([p.canonical_movie_id for p in batch.predictions])
        else:
            mapped = {
                p.canonical_movie_id: {
                    'django_movie_id': None,
                    'movie_title': None,
                    'mapping_status': 'UNMAPPED',
                }
                for p in batch.predictions
            }

        # Check for telemetry from backend receipt
        backend_receipt = getattr(adapter.backend, 'last_receipt', None)
        telemetry_map = {}
        if isinstance(backend_receipt, dict) and 'items' in backend_receipt:
            for item in backend_receipt['items']:
                if 'canonical_movie_id' in item and 'telemetry' in item:
                    telemetry_map[item['canonical_movie_id']] = item['telemetry']

        recommendations = []
        for i, p in enumerate(batch.predictions):
            item_data = dict(
                canonical_movie_id=p.canonical_movie_id,
                rank=i + 1,
                score=p.score,
                **mapped.get(p.canonical_movie_id, {
                    'django_movie_id': None,
                    'movie_title': None,
                    'mapping_status': 'UNMAPPED',
                })
            )
            if p.canonical_movie_id in telemetry_map:
                item_data['telemetry'] = telemetry_map[p.canonical_movie_id]
            recommendations.append(item_data)

        payload = {
            'model_name': model_version,
            'dataset': profile.dataset,
            'checkpoint_identity': batch.checkpoint_identity,
            'profile_source': {
                'mode': 'RESEARCH_DATASET_PROFILE',
                'identity_category': resolution.category.value,
                'profile_id': str(profile.id),
                'dataset': profile.dataset,
                'canonical_user_id': profile.canonical_user_id,
                'reference': profile.profile_source,
                'train_manifest_sha256': profile.train_manifest_sha256,
                'demo_only': profile.demo_only,
            },
            'recommendations': recommendations,
            'inference_mode': batch.inference_mode,
            'mapping_version': getattr(adapter.spec, 'mapping_version', 'v1'),
            'candidate_scope': batch.candidate_scope,
            'catalogue_manifest_identity': getattr(adapter.spec, 'catalogue_manifest_identity', None),
            'source_reference': batch.source_reference,
            'output_artifact_identity': batch.output_artifact_identity,
            'mapping_summary': {
                s: sum(x.get('mapping_status') == s for x in recommendations)
                for s in ('VERIFIED_UNIQUE', 'AMBIGUOUS', 'UNMAPPED')
            },
            'request_id': rid,
            'cache_status': cache_status,
        }

        # Persist evidence before admitting a result to the cache or caller.
        self._record_trace(user, rid, payload)
        # 6. Write cache
        write_model_cache(
            model_name=model_version,
            checkpoint_sha=ckpt_sha,
            feature_version=feature_version,
            user_context=user_ctx,
            top_k=top_k,
            payload=payload,
            candidate_scope=candidate_scope,
        )

        return payload

    @staticmethod
    def _record_trace(user, rid, payload):
        from .explanation import PROVENANCE_KEYS, result_snapshot
        provenance = {key: payload.get(key) for key in PROVENANCE_KEYS}
        provenance.update(recommendation_count=len(payload['recommendations']),
                          explanation_results=result_snapshot(payload),
                          cache_status=payload.get('cache_status'))
        try:
            RecommendationTrace.objects.create(
                id=uuid.UUID(rid) if isinstance(rid, str) else rid,
                requester=user,
                outcome='SUCCESS',
                provenance=provenance,
            )
        except DatabaseError:
            raise RecommendationError('TRACE_PERSISTENCE_FAILED', 503) from None
