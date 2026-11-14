"""Strict, CPU-only contracts. IDs are scoped by dataset, never business PKs."""
from __future__ import annotations
from dataclasses import dataclass
import math
import re
from typing import TypedDict

ML1M = 'ML1M_CH3_V1'
M3L = 'M3L_AUTHOR_ALIGNED_V1'
MODEL_DATASETS = {
    'SKB-HIF': ML1M,
    'UKMAP': M3L,
    'PAMK-Net': M3L,
    'PAMK-AV': M3L,
}
STATUSES = ('VERIFIED_UNIQUE', 'AMBIGUOUS', 'UNMAPPED')
# The legacy token is retained for existing operator adapters; only the new
# explicit REAL_ONLINE token is used by the verified Task58 checkpoint backend.
MODES = ('REALTIME_MODEL_INFERENCE', 'REAL_ONLINE_INFERENCE',
         'OFFLINE_MODEL_INFERENCE', 'TEST_DOUBLE')


class RecommendationItem(TypedDict):
    canonical_movie_id: str
    django_movie_id: int | None
    rank: int
    score: float
    movie_title: str | None
    mapping_status: str


class RecommendResponse(TypedDict):
    model_name: str
    dataset: str
    checkpoint_identity: str
    profile_source: dict
    recommendations: list[RecommendationItem]
    inference_mode: str
    mapping_version: str
    catalogue_manifest_identity: str
    candidate_scope: str
    source_reference: str
    output_artifact_identity: str | None
    mapping_summary: dict[str, int]


class RecommendationError(Exception):
    def __init__(self, code, status=400):
        self.code, self.status = code, status
        super().__init__(code)


def identifier(value):
    # Accept only canonical decimal spelling; zero is valid, bool is not.
    if type(value) is int:
        value = str(value)
    if type(value) is not str or not re.fullmatch(r'0|[1-9][0-9]{0,18}', value):
        raise RecommendationError('INVALID_CANONICAL_ID')
    return value


def digest(value):
    if type(value) is not str or not re.fullmatch(r'[a-f0-9]{64}', value):
        raise RecommendationError('INVALID_ARTIFACT_IDENTITY', 503)
    return value


@dataclass(frozen=True)
class RecommendRequest:
    model_name: str
    profile_id: str
    top_k: int

    @classmethod
    def parse(cls, value):
        import uuid
        if type(value) is not dict or set(value) != {'model_name', 'profile_id', 'top_k'}:
            raise RecommendationError('INVALID_REQUEST_FIELDS')
        if type(value['model_name']) is not str or value['model_name'] not in MODEL_DATASETS:
            raise RecommendationError('UNSUPPORTED_MODEL')
        if type(value['top_k']) is not int or not 1 <= value['top_k'] <= 100:
            raise RecommendationError('INVALID_TOP_K')
        try:
            if type(value['profile_id']) is not str:
                raise ValueError
            pid = str(uuid.UUID(value['profile_id']))
        except (ValueError, AttributeError):
            raise RecommendationError('INVALID_PROFILE_ID') from None
        return cls(value['model_name'], pid, value['top_k'])


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dataset: str
    checkpoint_identity: str
    mapping_version: str
    train_manifest_identity: str | None = None
    catalogue_manifest_identity: str | None = None

    def __post_init__(self):
        if MODEL_DATASETS.get(self.name) != self.dataset:
            raise RecommendationError('MODEL_DATASET_MISMATCH', 503)
        digest(self.checkpoint_identity)
        digest(self.train_manifest_identity)
        digest(self.catalogue_manifest_identity)
        if type(self.mapping_version) is not str or not 1 <= len(self.mapping_version) <= 100:
            raise RecommendationError('INVALID_MAPPING_VERSION', 503)


@dataclass(frozen=True)
class Prediction:
    canonical_movie_id: str
    score: float

    def __post_init__(self):
        object.__setattr__(self, 'canonical_movie_id', identifier(self.canonical_movie_id))
        if type(self.score) not in (float, int) or not math.isfinite(self.score):
            raise RecommendationError('INVALID_MODEL_SCORE', 502)


@dataclass(frozen=True)
class InferenceBatch:
    model_name: str
    dataset: str
    checkpoint_identity: str
    canonical_user_id: str
    predictions: tuple
    inference_mode: str
    candidate_scope: str
    source_reference: str
    output_artifact_identity: str | None = None
    train_manifest_identity: str | None = None
    catalogue_manifest_identity: str | None = None


class CheckpointAdapter:
    """Backend boundary; no torch import, checkpoint loading or fallback scores.

    A future real backend must return an InferenceBatch with provenance. Injecting
    a test backend only validates contracts, never authenticates model inference.
    """
    model_name = None

    def __init__(self, spec, backend=None):
        if spec.name != self.model_name:
            raise RecommendationError('ADAPTER_MODEL_MISMATCH', 503)
        self.spec, self.backend = spec, backend

    def infer(self, profile, top_k):
        if profile.dataset != self.spec.dataset:
            raise RecommendationError('PROFILE_DATASET_MISMATCH', 409)
        if self.backend is None:
            raise RecommendationError('REAL_MODEL_NOT_VERIFIED', 503)
        batch = self.backend.infer(profile, top_k)
        if not isinstance(batch, InferenceBatch):
            raise RecommendationError('INVALID_MODEL_RESPONSE', 502)
        if (batch.model_name, batch.dataset, batch.checkpoint_identity,
                batch.canonical_user_id) != (self.spec.name, self.spec.dataset,
                self.spec.checkpoint_identity, profile.canonical_user_id):
            raise RecommendationError('MODEL_PROVENANCE_MISMATCH', 502)
        if (batch.train_manifest_identity != self.spec.train_manifest_identity
                or batch.catalogue_manifest_identity != self.spec.catalogue_manifest_identity):
            raise RecommendationError('MODEL_MANIFEST_MISMATCH', 502)
        if (batch.inference_mode not in MODES
                or type(batch.source_reference) is not str or not 1 <= len(batch.source_reference) <= 250
                or type(batch.candidate_scope) is not str or not 1 <= len(batch.candidate_scope) <= 250):
            raise RecommendationError('MISSING_MODEL_PROVENANCE', 502)
        if batch.inference_mode == 'OFFLINE_MODEL_INFERENCE':
            digest(batch.output_artifact_identity)
        if type(batch.predictions) is not tuple or len(batch.predictions) > top_k:
            raise RecommendationError('INVALID_MODEL_TOP_K', 502)
        seen = set()
        last = float('inf')
        for pred in batch.predictions:
            if not isinstance(pred, Prediction) or pred.canonical_movie_id in seen or pred.score > last:
                raise RecommendationError('INVALID_MODEL_RANKING', 502)
            seen.add(pred.canonical_movie_id)
            last = pred.score
        return batch


class SKBHIFAdapter(CheckpointAdapter):
    model_name = 'SKB-HIF'


class UKMAPAdapter(CheckpointAdapter):
    model_name = 'UKMAP'


class PAMKAdapter(CheckpointAdapter):
    model_name = 'PAMK-Net'


@dataclass(frozen=True)
class FeatureManifest:
    modality: str
    dimension: int
    encoder: str
    coverage_ratio: float
    missing_policy: str
    feature_sha256: str


class PAMKAVAdapter(CheckpointAdapter):
    model_name = 'PAMK-AV'
    feature_manifests: dict[str, FeatureManifest] = {
        'text': FeatureManifest('text', 384, 'MiniLM-L6-v2', 1.0, 'FAIL_CLOSED', 'a4de49d050839fb58cc75eb2f4381a35faa4f9493fa9729f4a32d0d8faa96ba4'),
        'image': FeatureManifest('image', 512, 'CLIP-ViT-B/32', 1.0, 'FAIL_CLOSED', '94070bbb5826e2ac420e7ef2213b0a227c65c80ef11e8766bf3293e9200f3517'),
        'kg': FeatureManifest('kg', 14, 'TransE/RelationEvidence', 1.0, 'FAIL_CLOSED', '3fecfca2faa9f49e8fbe1a8bb4cfd3e6d53d9768fc0cb69b62e193c7730d134b'),
        'audio': FeatureManifest('audio', 128, 'VGGish', 0.982, 'MASK_LOGIT_PENALTY_20', 'pending_production_export'),
        'video': FeatureManifest('video', 768, 'MViT-B-24', 0.958, 'MASK_LOGIT_PENALTY_20', 'pending_production_export'),
    }
    status = 'FEATURE_MANIFEST_DEFINED_INFERENCE_PENDING_GO_NO_GO'

    def infer(self, profile, top_k):
        if self.backend is None:
            raise RecommendationError('PAMK_AV_INFERENCE_PENDING_VERIFICATION', 503)
        return super().infer(profile, top_k)

