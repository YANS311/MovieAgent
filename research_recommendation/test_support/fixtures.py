"""Test-only source declarations and backend; NOT real checkpoint evidence."""
from types import MappingProxyType
from research_recommendation.contracts import (
    ML1M, M3L, ModelSpec, Prediction, InferenceBatch, SKBHIFAdapter, UKMAPAdapter,
)
from research_recommendation.manifests import TrainProfileManifest, MovieCatalogueManifest
from research_recommendation.services import DjangoMovieCatalogue, MappingService, RecommendationService

SPECS = {'SKB-HIF': ModelSpec('SKB-HIF', ML1M, 'a'*64, 'synthetic-ml1m-v1', 'c'*64, 'e'*64),
         'UKMAP': ModelSpec('UKMAP', M3L, 'b'*64, 'synthetic-m3l-v1', 'd'*64, 'f'*64)}
TRAINS = {ML1M: TrainProfileManifest(ML1M, 'c'*64, 'SYNTHETIC_TRAIN_FIXTURE_ML1M', frozenset({'0','7'})),
          M3L: TrainProfileManifest(M3L, 'd'*64, 'SYNTHETIC_TRAIN_FIXTURE_M3L', frozenset({'0','7'}))}
CATALOGUES = {d: MovieCatalogueManifest(d, ('e' if d==ML1M else 'f')*64, 'movielens:movieId',
                                      MappingProxyType({'0':'100','1':'101','2':'102'})) for d in TRAINS}


class SyntheticBackend:
    def __init__(self, spec):
        self.spec = spec

    def infer(self, profile, top_k):
        return InferenceBatch(self.spec.name, self.spec.dataset, self.spec.checkpoint_identity,
                              profile.canonical_user_id,
                              tuple(Prediction(str(i), 3.0-i) for i in range(min(top_k,3))),
                              'TEST_DOUBLE', 'SYNTHETIC_THREE_ITEM_POOL',
                              'TEST_DOUBLE_ONLY_NOT_REAL_MODEL_INFERENCE', None,
                              self.spec.train_manifest_identity, self.spec.catalogue_manifest_identity)


class SyntheticCatalogue(DjangoMovieCatalogue):
    model_label = 'task57_fixtures.SyntheticMovie'


def test_service():
    adapters, mappings = {}, {}
    for name, spec in SPECS.items():
        adapters[name] = {'SKB-HIF':SKBHIFAdapter,'UKMAP':UKMAPAdapter}[name](spec, SyntheticBackend(spec))
        mappings[name] = MappingService(CATALOGUES[spec.dataset], spec.mapping_version, SyntheticCatalogue())
    return RecommendationService(adapters, TRAINS, mappings)


def no_backend_service():
    app = test_service()
    for adapter in app.adapters.values():
        adapter.backend = None
    return app


def broken_service():
    raise RuntimeError('synthetic secret/path must never appear in response')
