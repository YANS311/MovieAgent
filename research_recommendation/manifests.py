"""Explicit operator-pinned metadata. Never reads Train/Test interaction arrays.

Hashes establish artifact identity, not the truth of an operator's declaration.
Real manifests require a separate provenance review before production enabling.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

from .contracts import MODEL_DATASETS, RecommendationError, digest, identifier


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RecommendationError('DUPLICATE_JSON_KEY')
        result[key] = value
    return result


def read_pinned_json(path, expected_sha256, max_bytes=2_000_000):
    digest(expected_sha256)
    # Only an explicit configured file, no directory traversal or discovery.
    p = Path(path)
    with p.open('rb') as f:
        data = f.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise RecommendationError('MANIFEST_TOO_LARGE', 503)
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise RecommendationError('MANIFEST_IDENTITY_MISMATCH', 503)
    try:
        value = json.loads(data, object_pairs_hook=unique_object)
    except (ValueError, UnicodeError):
        raise RecommendationError('INVALID_MANIFEST_JSON', 503) from None
    if type(value) is not dict:
        raise RecommendationError('INVALID_MANIFEST', 503)
    return value


@dataclass(frozen=True)
class TrainProfileManifest:
    dataset: str
    identity: str
    source_reference: str
    users: frozenset

    @classmethod
    def load(cls, path, sha):
        v = read_pinned_json(path, sha)
        if (v.get('schema') != 'task57.train_profiles.v1'
                or v.get('evidence_kind') != 'REGISTERED_TRAIN_PROFILE_MEMBERSHIP'
                or v.get('dataset') not in MODEL_DATASETS.values()
                or type(v.get('source_reference')) is not str or not v['source_reference']
                or type(v.get('canonical_user_ids')) is not list
                or not 1 <= len(v['canonical_user_ids']) <= 100_000):
            raise RecommendationError('INVALID_TRAIN_PROFILE_MANIFEST', 503)
        ids = [identifier(x) for x in v['canonical_user_ids']]
        if len(ids) != len(set(ids)):
            raise RecommendationError('DUPLICATE_TRAIN_PROFILE', 503)
        digest(v.get('train_identity'))
        return cls(v['dataset'], sha, v['source_reference'], frozenset(ids))

    def verify(self, profile):
        if (profile.dataset != self.dataset or not profile.demo_only
                or profile.train_manifest_sha256 != self.identity
                or profile.profile_source != self.source_reference
                or identifier(profile.canonical_user_id) not in self.users):
            raise RecommendationError('TRAIN_PROFILE_NOT_VERIFIED', 409)


@dataclass(frozen=True)
class MovieCatalogueManifest:
    dataset: str
    identity: str
    source_namespace: str
    movies: object

    @classmethod
    def load(cls, path, sha):
        v = read_pinned_json(path, sha)
        if (v.get('schema') != 'task57.movie_catalogue.v1'
                or v.get('dataset') not in MODEL_DATASETS.values()
                or v.get('source_namespace') != 'movielens:movieId'
                or type(v.get('movies')) is not list or not 1 <= len(v['movies']) <= 100_000):
            raise RecommendationError('INVALID_MOVIE_CATALOGUE_MANIFEST', 503)
        movies = {}
        external = set()
        for item in v['movies']:
            if type(item) is not dict or set(item) != {'canonical_movie_id', 'source_movie_id'}:
                raise RecommendationError('INVALID_MOVIE_CATALOGUE_MANIFEST', 503)
            canonical, source = identifier(item['canonical_movie_id']), identifier(item['source_movie_id'])
            if canonical in movies or source in external:
                raise RecommendationError('AMBIGUOUS_MOVIE_CATALOGUE', 503)
            movies[canonical] = source
            external.add(source)
        return cls(v['dataset'], sha, v['source_namespace'], MappingProxyType(movies))
