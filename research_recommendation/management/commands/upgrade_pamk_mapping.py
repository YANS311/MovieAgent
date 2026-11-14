"""Add a versioned deterministic mapping; never promote old AMBIGUOUS rows."""
from pathlib import Path
import json,hashlib,collections
from django.core.management.base import BaseCommand,CommandError
from django.db import transaction
from django.conf import settings
from research_recommendation.models import MovieIDBridge,ResearchMovie,ResearchDataset
from research_recommendation.product_mapping import TMDB_SOURCE_SHA256,business_identity_verified
from research_recommendation.contracts import M3L
from myapp.models import Movie

class Command(BaseCommand):
    def add_arguments(self,p):p.add_argument('--candidates',required=True)
    def handle(self,*a,**opts):
        if settings.DATABASES['default']['NAME']!='ch5_pamk_inductive_20261113':raise CommandError('ISOLATED_DATABASE_ONLY')
        old=list(MovieIDBridge.objects.filter(source_dataset=M3L,mapping_version='ch5-exact-year-v1'));new_version='ch5-exact-tmdb-v2'
        if len(old)!=9031:raise CommandError('BASE_MAPPING_INCOMPLETE')
        original={int(b.canonical_movie_id):b for b in old};rows=json.loads(Path(opts['candidates']).read_text())['approved_unmapped_candidates']
        metadata_path=Path('/home/daylight/下载/DjangoProject3/DjangoProject3/data/m3l_full/kg/full_tmdb_metadata_cache.json')
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest()!=TMDB_SOURCE_SHA256:raise CommandError('METADATA_SOURCE_CHANGED')
        metadata=json.loads(metadata_path.read_text());d=ResearchDataset.objects.get(protocol=M3L);catalogue=ResearchMovie.objects.in_bulk(field_name='id');bycid={r.canonical_movie_id:r for r in catalogue.values()}
        replacements={};used={b.django_movie_id for b in old if b.mapping_status=='VERIFIED_UNIQUE'}
        for candidate in rows:
            cid=candidate['canonical_movie_id'];b=original[cid];rm=bycid[cid]
            if b.mapping_status!='UNMAPPED':raise CommandError('AMBIGUOUS_OR_VERIFIED_PROMOTION_FORBIDDEN')
            tm=metadata[rm.source_movie_id]
            for k,value in [('tmdb_id',str(tm['id'])),('tmdb_title',tm.get('title')),('tmdb_original_title',tm.get('original_title')),('tmdb_release_date',tm.get('release_date')),('poster_filename',(tm.get('poster_path') or '').split('/')[-1])]:
                if candidate[k]!=value:raise CommandError('CANDIDATE_METADATA_CONFLICT')
            movie=Movie.objects.get(pk=candidate['django_movie_id'])
            evidence=dict(candidate,catalogue_sha256=d.manifest['catalogue_sha256'],namespace='TMDB:movie_id (legacy business imdb_id field)',previous_status='UNMAPPED')
            new=MovieIDBridge(source_dataset=M3L,source_movie_id=rm.source_movie_id,canonical_movie_id=str(cid),django_movie_id=movie.pk,mapping_status='VERIFIED_UNIQUE',mapping_version=new_version,match_method='EXACT_TMDB_CORROBORATED',evidence=evidence)
            if movie.pk in used or not business_identity_verified(new,rm,movie):raise CommandError('LIVE_IDENTITY_CONFLICT')
            used.add(movie.pk);replacements[cid]=new
        result=[]
        for b in old:
            cid=int(b.canonical_movie_id)
            if cid in replacements:result.append(replacements[cid])
            else:result.append(MovieIDBridge(source_dataset=b.source_dataset,source_movie_id=b.source_movie_id,canonical_movie_id=b.canonical_movie_id,django_movie_id=b.django_movie_id,match_method=b.match_method,evidence=b.evidence,mapping_status=b.mapping_status,mapping_version=new_version))
        with transaction.atomic():
            existing=MovieIDBridge.objects.filter(source_dataset=M3L,mapping_version=new_version)
            if existing.exists():
                if existing.count()!=9031:raise CommandError('PARTIAL_EXISTING_MAPPING_VERSION')
            else:MovieIDBridge.objects.bulk_create(result,batch_size=1000)
        self.stdout.write(json.dumps({'version':new_version,'new_verified':len(replacements),'old_ambiguous_unchanged':sum(b.mapping_status=='AMBIGUOUS' for b in old),'counts':dict(collections.Counter(b.mapping_status for b in result))}))
