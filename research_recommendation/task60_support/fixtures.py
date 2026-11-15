"""TEST_DOUBLE scores only. Never imports a model, torch, data arrays or worker."""
from dataclasses import replace
import json
import os
from pathlib import Path
from .models import Movie
from research_recommendation.contracts import M3L, ModelSpec, UKMAPAdapter
from research_recommendation.manifests import MovieCatalogueManifest, TrainProfileManifest
from research_recommendation.services import DjangoMovieCatalogue, MappingService, RecommendationService
from research_recommendation.test_support.fixtures import SPECS, TRAINS, CATALOGUES, SyntheticBackend


class Catalogue(DjangoMovieCatalogue):
    model_label = 'task60_snapshot.Movie'


def test_service():
    from research_recommendation.contracts import SKBHIFAdapter
    adapters, mappings = {}, {}
    for name, spec in SPECS.items():
        adapters[name] = {'UKMAP': UKMAPAdapter, 'SKB-HIF': SKBHIFAdapter}[name](spec, SyntheticBackend(spec))
        mappings[name] = MappingService(CATALOGUES[spec.dataset], spec.mapping_version, Catalogue())
    return RecommendationService(adapters, TRAINS, mappings)


class EmptyBackend(SyntheticBackend):
    def infer(self, profile, top_k):
        return replace(super().infer(profile, top_k), predictions=())


def empty_service():
    app = test_service()
    for a in app.adapters.values():
        a.backend = EmptyBackend(a.spec)
    return app


def unavailable_service():
    app = test_service()
    for a in app.adapters.values():
        a.backend = None
    return app


def demo_service():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('TASK60_CPU_ONLY_REQUIRED')
    base = Path(__file__).resolve().parents[2] / 'docs/task58/evidence'
    cfg = json.loads((base / 'RUNTIME_CONFIG.json').read_text())
    spec = ModelSpec('UKMAP', M3L, 'b'*64, 'task60-TEST_DOUBLE-only-v1',
                     cfg['profile_manifest_sha256'], cfg['catalogue_manifest_sha256'])
    train = TrainProfileManifest.load(base / 'TRAIN_PROFILE.json', spec.train_manifest_identity)
    movies = MovieCatalogueManifest.load(base / 'MOVIE_CATALOGUE.json', spec.catalogue_manifest_identity)
    return RecommendationService({'UKMAP': UKMAPAdapter(spec, SyntheticBackend(spec))}, {M3L: train},
                                 {'UKMAP': MappingService(movies, spec.mapping_version, Catalogue())})
