import uuid
from django.conf import settings
from django.db import models

from .contracts import ML1M, M3L, STATUSES

DATASETS = [(ML1M, ML1M), (M3L, M3L)]


class MovieIDBridge(models.Model):
    source_dataset = models.CharField(max_length=40, choices=DATASETS)
    source_movie_id = models.CharField(max_length=32)
    canonical_movie_id = models.CharField(max_length=32)
    django_movie_id = models.PositiveBigIntegerField(null=True, blank=True)
    match_method = models.CharField(max_length=64)
    evidence = models.JSONField(default=dict, blank=True)
    mapping_status = models.CharField(max_length=24, choices=[(x, x) for x in STATUSES])
    mapping_version = models.CharField(max_length=100)

    class Meta:
        indexes = [models.Index(fields=['source_dataset', 'mapping_version', 'canonical_movie_id'], name='t57_bridge_lookup')]
        constraints = [models.CheckConstraint(condition=models.Q(mapping_status__in=STATUSES), name='t57_bridge_status')]
        # Multiple conflicting rows are retained as evidence; resolver fails closed.


class ResearchProfile(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    dataset = models.CharField(max_length=40, choices=DATASETS)
    canonical_user_id = models.CharField(max_length=32)
    display_name = models.CharField(max_length=100)
    profile_source = models.CharField(max_length=250)
    train_manifest_sha256 = models.CharField(max_length=64)
    demo_only = models.BooleanField(default=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['dataset', 'canonical_user_id', 'train_manifest_sha256'], name='t57_unique_profile'),
            models.CheckConstraint(condition=models.Q(demo_only=True), name='t57_demo_only'),
        ]


class ProfileGrant(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    profile = models.ForeignKey(ResearchProfile, on_delete=models.CASCADE, related_name='grants')
    is_active = models.BooleanField(default=True)

    class Meta:
        constraints = [models.UniqueConstraint(fields=['user', 'profile'], name='t57_unique_grant')]


class RecommendationTrace(models.Model):
    id = models.UUIDField(primary_key=True, editable=False)
    requester = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    outcome = models.CharField(max_length=64)
    provenance = models.JSONField(default=dict)


class UserPreference(models.Model):
    """Ordinary account preferences; never a research dataset identity."""
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    genres = models.JSONField(default=list, blank=True)
    # Business movie PKs in the explicitly configured catalogue, not M3L IDs.
    movie_ids = models.JSONField(default=list, blank=True)
    catalogue_model = models.CharField(max_length=100)
    updated_at = models.DateTimeField(auto_now=True)


class PersonalRecommendationTrace(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    provenance = models.JSONField(default=dict)


class MovieInteraction(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    # Catalogue PK + model label supports both existing business and snapshot models.
    movie_id = models.PositiveBigIntegerField()
    catalogue_model = models.CharField(max_length=100)
    kind = models.CharField(max_length=16, choices=[('view', '查看详情'), ('watched', '看过'), ('interested', '感兴趣')])
    trace = models.ForeignKey(PersonalRecommendationTrace, null=True, blank=True,
                              on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['user', 'created_at'], name='t62_user_interaction')]


class CatalogueCacheState(models.Model):
    """Durable generation survives Redis outages and prevents stale resurrection."""
    catalogue_model = models.CharField(max_length=100, unique=True)
    generation = models.UUIDField(default=uuid.uuid4, editable=False)


class ResearchDataset(models.Model):
    """Import generation and immutable Train asset identities, separate from accounts."""
    protocol = models.CharField(max_length=40, unique=True)
    manifest = models.JSONField(default=dict)
    imported_pairs = models.PositiveBigIntegerField(default=0)
    complete = models.BooleanField(default=False)


class ResearchUser(models.Model):
    dataset = models.ForeignKey(ResearchDataset, on_delete=models.CASCADE)
    canonical_user_id = models.PositiveIntegerField()
    train_count = models.PositiveIntegerField(default=0)
    class Meta:
        constraints = [models.UniqueConstraint(fields=['dataset', 'canonical_user_id'], name='ch5_canonical_user')]


class ResearchMovie(models.Model):
    dataset = models.ForeignKey(ResearchDataset, on_delete=models.CASCADE)
    canonical_movie_id = models.PositiveIntegerField()
    source_movie_id = models.CharField(max_length=32)
    title = models.CharField(max_length=500)
    metadata = models.JSONField(default=dict)
    class Meta:
        constraints = [models.UniqueConstraint(fields=['dataset', 'canonical_movie_id'], name='ch5_canonical_movie')]


class ResearchTrainInteraction(models.Model):
    """Train-only canonical pairs. No FK to Django users or business movie IDs."""
    dataset = models.ForeignKey(ResearchDataset, on_delete=models.CASCADE)
    canonical_user_id = models.PositiveIntegerField()
    canonical_movie_id = models.PositiveIntegerField()
    class Meta:
        constraints = [models.UniqueConstraint(fields=['dataset', 'canonical_user_id', 'canonical_movie_id'], name='ch5_train_pair')]

# Registration is explicit; these models store only account-owned online data.
from .models_completion import UnifiedMovieFeedback, ProductAgentSession, ProductAgentTurn, ProductToolCall, PendingPreferenceAction
