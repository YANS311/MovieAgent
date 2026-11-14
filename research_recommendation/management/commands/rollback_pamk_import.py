"""Explicit rollback for the independent acceptance database only; dry run by default."""
from django.conf import settings
from django.core.management.base import BaseCommand,CommandError
from django.db import transaction
from research_recommendation.models import ResearchDataset,ResearchProfile,ProfileGrant,MovieIDBridge
from research_recommendation.contracts import M3L
from research_recommendation.product_mapping import MAPPING_VERSION
class Command(BaseCommand):
    help='Dry-run or remove this import from an independent ch5_pamk_* database, never business rows.'
    def add_arguments(self,p):
        p.add_argument('--confirm-train-sha256',required=True)
        p.add_argument('--apply',action='store_true')
    def handle(self,*args,**o):
        name=settings.DATABASES['default']['NAME']
        if not name.startswith(('ch5_pamk_','test_ch5_pamk_')):raise CommandError('INDEPENDENT_DATABASE_ONLY')
        d=ResearchDataset.objects.get(protocol=M3L)
        if o['confirm_train_sha256']!=d.manifest['assets']['train_context_csr']['sha256']:raise CommandError('TRAIN_SHA_CONFIRMATION_MISMATCH')
        profiles=ResearchProfile.objects.filter(dataset=M3L,train_manifest_sha256=o['confirm_train_sha256'])
        self.stdout.write(f'DB={name} users={d.researchuser_set.count()} movies={d.researchmovie_set.count()} pairs={d.researchtraininteraction_set.count()} apply={o["apply"]}')
        if not o['apply']:return
        with transaction.atomic():
            ProfileGrant.objects.filter(profile__in=profiles).update(is_active=False)
            profiles.update(is_active=False)
            MovieIDBridge.objects.filter(source_dataset=M3L,mapping_version=MAPPING_VERSION).delete()
            d.delete()
        self.stdout.write('ROLLED_BACK_IMPORT_ONLY; source assets and business accounts/movies/ratings preserved')
