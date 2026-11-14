"""Explicit datasets, grants and mapping evidence; never guesses an identity."""
from dataclasses import dataclass
from django.apps import apps
from django.db import transaction

from .contracts import MODEL_DATASETS, RecommendationError, RecommendResponse, identifier
from .models import MovieIDBridge, ResearchProfile


@dataclass(frozen=True)
class MovieRecord:
    django_movie_id: int
    title: str
    movielens_id: str | None


class DjangoMovieCatalogue:
    """Production Movie resolved lazily; safe to import without myapp.ready()."""
    model_label = 'myapp.Movie'

    def get_many(self, ids):
        model = apps.get_model(self.model_label)
        return {row['id']: MovieRecord(row['id'], row['title'], row['movielens_id'])
                for row in model.objects.filter(pk__in=ids).values('id', 'title', 'movielens_id')}


class MappingService:
    def __init__(self, manifest, mapping_version, catalogue):
        self.manifest, self.version, self.catalogue = manifest, mapping_version, catalogue

    @transaction.atomic
    def record(self, canonical_movie_id, source_movie_id, django_movie_id, evidence):
        """Explicit reviewed link import; no title matching or numeric PK inference.

        Intended for an authorized offline version build, not a public API.
        Conflicting rows remain visible and resolution becomes AMBIGUOUS.
        """
        canonical, source = identifier(canonical_movie_id), identifier(source_movie_id)
        if canonical not in self.manifest.movies or self.manifest.movies[canonical] != source:
            raise RecommendationError('CANONICAL_SOURCE_MISMATCH', 409)
        if django_movie_id is not None and (type(django_movie_id) is not int or django_movie_id <= 0):
            raise RecommendationError('INVALID_DJANGO_MOVIE_ID')
        status = 'UNMAPPED'
        if django_movie_id is not None:
            details = self.catalogue.get_many({django_movie_id})
            movie = details.get(django_movie_id)
            if movie is not None and movie.movielens_id == source:
                if (type(evidence) is dict and evidence.get('catalogue_sha256') == self.manifest.identity
                        and evidence.get('source_namespace') == self.manifest.source_namespace
                        and type(evidence.get('review_reference')) is str
                        and evidence['review_reference'].strip()):
                    status = 'VERIFIED_UNIQUE'
            elif movie is not None:
                raise RecommendationError('BUSINESS_MOVIE_SOURCE_CONFLICT', 409)
        existing = MovieIDBridge.objects.filter(source_dataset=self.manifest.dataset, mapping_version=self.version)
        if existing.filter(canonical_movie_id=canonical).exists() or (django_movie_id is not None and
                existing.filter(django_movie_id=django_movie_id).exists()):
            status = 'AMBIGUOUS'
        row = MovieIDBridge(source_dataset=self.manifest.dataset, source_movie_id=source,
                            canonical_movie_id=canonical, django_movie_id=django_movie_id,
                            match_method='EXACT_EXTERNAL_ID_REVIEWED', evidence=evidence,
                            mapping_status=status, mapping_version=self.version)
        row.full_clean()
        row.save()
        return row

    def resolve_many(self, ids):
        ids = [identifier(x) for x in ids]
        if any(x not in self.manifest.movies for x in ids):
            raise RecommendationError('MODEL_ITEM_NOT_IN_CATALOGUE', 502)
        base = MovieIDBridge.objects.filter(source_dataset=self.manifest.dataset, mapping_version=self.version)
        rows = list(base.filter(canonical_movie_id__in=ids).order_by('pk')[:1001])
        if len(rows) > 1000:
            raise RecommendationError('MAPPING_EVIDENCE_TOO_LARGE', 409)
        grouped = {x: [] for x in ids}
        for row in rows:
            grouped[row.canonical_movie_id].append(row)
        business_ids = {r.django_movie_id for r in rows if r.django_movie_id is not None}
        # Reverse collisions are checked across the entire dataset+version.
        reverse = {}
        reverse_rows = list(base.filter(django_movie_id__in=business_ids)[:1001])
        if len(reverse_rows) > 1000:
            raise RecommendationError('MAPPING_EVIDENCE_TOO_LARGE', 409)
        for row in reverse_rows:
            reverse.setdefault(row.django_movie_id, set()).add(row.canonical_movie_id)
        details = self.catalogue.get_many(business_ids) if business_ids else {}
        result = {}
        for canonical, candidates in grouped.items():
            item = {'django_movie_id': None, 'movie_title': None, 'mapping_status': 'UNMAPPED'}
            if candidates:
                if (len(candidates) != 1 or any(r.mapping_status == 'AMBIGUOUS' for r in candidates)
                        or any(len(reverse.get(r.django_movie_id, ())) > 1 for r in candidates)):
                    item['mapping_status'] = 'AMBIGUOUS'
                else:
                    row = candidates[0]
                    evidence = row.evidence
                    valid = (row.mapping_status == 'VERIFIED_UNIQUE'
                             and row.match_method == 'EXACT_EXTERNAL_ID_REVIEWED'
                             and row.source_movie_id == self.manifest.movies[canonical]
                             and type(evidence) is dict
                             and evidence.get('catalogue_sha256') == self.manifest.identity
                             and evidence.get('source_namespace') == self.manifest.source_namespace
                             and type(evidence.get('review_reference')) is str
                             and bool(evidence['review_reference'].strip()))
                    if valid and row.django_movie_id in details:
                        movie = details[row.django_movie_id]
                        if movie.movielens_id != row.source_movie_id:
                            raise RecommendationError('BUSINESS_MOVIE_SOURCE_CONFLICT', 409)
                        item.update(django_movie_id=row.django_movie_id, movie_title=movie.title,
                                    mapping_status='VERIFIED_UNIQUE')
                    elif valid:
                        raise RecommendationError('MAPPED_MOVIE_NOT_FOUND', 409)
            result[canonical] = item
        return result


class ProfileService:
    @staticmethod
    def register(manifest, canonical_user_id, display_name):
        canonical = identifier(canonical_user_id)
        if canonical not in manifest.users:
            raise RecommendationError('TRAIN_PROFILE_NOT_VERIFIED', 409)
        profile = ResearchProfile(dataset=manifest.dataset, canonical_user_id=canonical,
                                  display_name=display_name, profile_source=manifest.source_reference,
                                  train_manifest_sha256=manifest.identity, demo_only=True)
        profile.full_clean()
        profile.save()
        return profile

    @staticmethod
    def authorized(user, profile_id):
        # No automatic grant, staff bypass, user.pk mapping or unknown-user row.
        profile = ResearchProfile.objects.filter(pk=profile_id, is_active=True,
                  grants__user=user, grants__is_active=True).first()
        if profile is None:
            raise RecommendationError('PROFILE_NOT_AVAILABLE', 404)
        return profile


class RecommendationService:
    def __init__(self, adapters, profile_manifests, mapping_services):
        self.adapters, self.profile_manifests, self.mapping_services = adapters, profile_manifests, mapping_services

    def verify_profile(self, profile):
        manifest = self.profile_manifests.get(profile.dataset)
        if manifest is None:
            raise RecommendationError('TRAIN_MANIFEST_NOT_CONFIGURED', 503)
        manifest.verify(profile)

    def recommend(self, user, request) -> RecommendResponse:
        profile = ProfileService.authorized(user, request.profile_id)
        if profile.dataset != MODEL_DATASETS[request.model_name]:
            raise RecommendationError('PROFILE_DATASET_MISMATCH', 409)
        self.verify_profile(profile)
        adapter = self.adapters.get(request.model_name)
        if adapter is None:
            raise RecommendationError('REAL_MODEL_NOT_VERIFIED', 503)
        mapping = self.mapping_services.get(request.model_name)
        if (mapping is None or mapping.manifest.dataset != adapter.spec.dataset
                or mapping.version != adapter.spec.mapping_version
                or mapping.manifest.identity != adapter.spec.catalogue_manifest_identity
                or profile.train_manifest_sha256 != adapter.spec.train_manifest_identity):
            raise RecommendationError('MAPPING_NOT_CONFIGURED', 503)
        batch = adapter.infer(profile, request.top_k)
        mapped = mapping.resolve_many([p.canonical_movie_id for p in batch.predictions])
        recommendations = [dict(canonical_movie_id=p.canonical_movie_id, rank=i+1, score=p.score,
                                **mapped[p.canonical_movie_id]) for i, p in enumerate(batch.predictions)]
        return {
            'model_name': adapter.spec.name, 'dataset': profile.dataset,
            'checkpoint_identity': batch.checkpoint_identity,
            'profile_source': {'mode': 'RESEARCH_DATASET_PROFILE', 'profile_id': str(profile.id),
                               'dataset': profile.dataset, 'reference': profile.profile_source,
                               'train_manifest_sha256': profile.train_manifest_sha256, 'demo_only': True},
            'recommendations': recommendations, 'inference_mode': batch.inference_mode,
            'mapping_version': mapping.version, 'candidate_scope': batch.candidate_scope,
            'catalogue_manifest_identity': mapping.manifest.identity,
            'source_reference': batch.source_reference,
            'output_artifact_identity': batch.output_artifact_identity,
            'mapping_summary': {s: sum(x['mapping_status'] == s for x in recommendations)
                                for s in ('VERIFIED_UNIQUE', 'AMBIGUOUS', 'UNMAPPED')},
        }


def unconfigured_service():
    # A working API must fail closed until real assets are explicitly reviewed.
    return RecommendationService({}, {}, {})
