"""Operator configuration for pinned manifests; real backends remain unplugged.

No dynamic application/model imports, dataset scanning or checkpoint loading.
"""
from django.conf import settings
from .contracts import (
    ModelSpec,
    SKBHIFAdapter,
    UKMAPAdapter,
    PAMKAdapter,
    PAMKAVAdapter,
    RecommendationError,
)
from .manifests import MovieCatalogueManifest, TrainProfileManifest
from .services import DjangoMovieCatalogue, MappingService, RecommendationService


def configured_service():
    profiles, mappings, adapters = {}, {}, {}
    adapter_types = {
        'SKB-HIF': SKBHIFAdapter,
        'UKMAP': UKMAPAdapter,
        'PAMK-Net': PAMKAdapter,
        'PAMK-AV': PAMKAVAdapter,
    }
    for entry in getattr(settings, 'TASK57_MODEL_REGISTRY', ()):
        spec = ModelSpec(entry['model_name'], entry['dataset'], entry['checkpoint_identity'],
                         entry['mapping_version'], entry['profile_manifest_sha256'],
                         entry['catalogue_manifest_sha256'])
        if spec.name in adapters:
            raise RecommendationError('DUPLICATE_MODEL_REGISTRY', 503)
        train = TrainProfileManifest.load(entry['profile_manifest_path'], entry['profile_manifest_sha256'])
        movies = MovieCatalogueManifest.load(entry['catalogue_manifest_path'], entry['catalogue_manifest_sha256'])
        if train.dataset != spec.dataset or movies.dataset != spec.dataset:
            raise RecommendationError('MANIFEST_DATASET_MISMATCH', 503)
        profiles[spec.dataset] = train
        mappings[spec.name] = MappingService(movies, spec.mapping_version, DjangoMovieCatalogue())
        adapter_type = adapter_types.get(spec.name)
        if adapter_type is None:
            raise RecommendationError('UNSUPPORTED_MODEL', 503)
        # Deliberately no real backend: returns REAL_MODEL_NOT_VERIFIED, never guesses.
        adapters[spec.name] = adapter_type(spec)
    return RecommendationService(adapters, profiles, mappings)
