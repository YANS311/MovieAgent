"""Pinned real-output bundle reader; no bundle or invented outputs are shipped."""
from .contracts import InferenceBatch, Prediction, RecommendationError
from .manifests import read_pinned_json


class OfflineOutputBackend:
    def __init__(self, path, sha256, spec, train_manifest_identity):
        self.value = read_pinned_json(path, sha256)
        self.sha256, self.spec = sha256, spec
        v = self.value
        if (v.get('schema') != 'task57.offline_inference.v1'
                or v.get('inference_mode') != 'OFFLINE_MODEL_INFERENCE'
                or v.get('model_name') != spec.name or v.get('dataset') != spec.dataset
                or v.get('checkpoint_identity') != spec.checkpoint_identity
                or v.get('train_manifest_sha256') != train_manifest_identity
                or v.get('train_manifest_sha256') != spec.train_manifest_identity
                or v.get('catalogue_manifest_sha256') != spec.catalogue_manifest_identity
                or not v.get('source_reference') or not v.get('candidate_scope')
                or type(v.get('profiles')) is not dict):
            raise RecommendationError('OFFLINE_PROVENANCE_MISMATCH', 503)

    def infer(self, profile, top_k):
        if profile.train_manifest_sha256 != self.value['train_manifest_sha256']:
            raise RecommendationError('OFFLINE_PROVENANCE_MISMATCH', 503)
        values = self.value['profiles'].get(profile.canonical_user_id)
        if type(values) is not list or len(values) > 1000:
            raise RecommendationError('OFFLINE_PROFILE_UNAVAILABLE', 503)
        predictions = []
        for v in values:
            if type(v) is not dict or set(v) != {'canonical_movie_id', 'score'}:
                raise RecommendationError('INVALID_OFFLINE_OUTPUT', 503)
            predictions.append(Prediction(v['canonical_movie_id'], v['score']))
        # Validate the entire stored order before slicing; never silently re-rank.
        if len({p.canonical_movie_id for p in predictions}) != len(predictions) or any(
                a.score < b.score for a, b in zip(predictions, predictions[1:])):
            raise RecommendationError('INVALID_OFFLINE_OUTPUT', 503)
        if len(predictions) < top_k:
            raise RecommendationError('OFFLINE_TOP_K_UNAVAILABLE', 409)
        return InferenceBatch(self.spec.name, self.spec.dataset, self.spec.checkpoint_identity,
                              profile.canonical_user_id, tuple(predictions[:top_k]),
                              'OFFLINE_MODEL_INFERENCE', self.value['candidate_scope'],
                              self.value['source_reference'], self.sha256,
                              self.spec.train_manifest_identity, self.spec.catalogue_manifest_identity)
