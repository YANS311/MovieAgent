"""Resumable, idempotent Train-only import, preserving all canonical identities."""
import hashlib
import json
from pathlib import Path
import numpy as np
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction, connection
from research_recommendation.models import ResearchDataset, ResearchMovie, ResearchUser, ResearchTrainInteraction, MovieIDBridge
from research_recommendation.contracts import M3L
from research_recommendation.product_mapping import MAPPING_VERSION, normalized_title
from myapp.models import Movie


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

class Command(BaseCommand):
    help='Import pinned Train CSR and canonical metadata; no validation/test labels, no account grants.'
    def add_arguments(self,p):
        p.add_argument('--assets',required=True)
        p.add_argument('--stop-after-users',type=int,default=69878)
    def handle(self,*args,**opts):
        root=Path(opts['assets']);m=json.loads((root/'train_manifest.json').read_text())
        if m['protocol']!=M3L or m['NEW_TEST_ACCESS']!=0:raise CommandError('INVALID_PROTOCOL')
        a=m['assets']['train_context_csr']
        if sha(a['path'])!=a['sha256'] or sha(m['catalogue_source'])!=m['catalogue_sha256']:raise CommandError('ASSET_SHA_MISMATCH')
        d,created=ResearchDataset.objects.get_or_create(protocol=M3L,defaults={'manifest':m})
        if d.manifest!=m:raise CommandError('EXISTING_GENERATION_DIFFERS')
        users=json.loads((root/'research_users.json').read_text());movies=json.loads((root/'research_movies.json').read_text())
        # Independently verify the prepared metadata against the pinned source CSV.
        import csv
        with Path(m['catalogue_source']).open() as f:
            original={int(float(x['item_idx'])):x for x in csv.DictReader(f) if x['node_type']=='Movie'}
        if len(movies)!=9031 or len({r['canonical_movie_id'] for r in movies})!=9031:raise CommandError('BAD_MOVIE_CATALOGUE')
        for row in movies:
            x=original[row['canonical_movie_id']]
            if row['source_movie_id']!=str(int(float(x['movieId']))) or row['title']!=x['title']:raise CommandError('METADATA_PREPARATION_MISMATCH')
        ResearchMovie.objects.bulk_create([ResearchMovie(dataset=d,**r) for r in movies],ignore_conflicts=True,batch_size=1000)
        z=np.load(a['path']);indices=z['indices'];indptr=z['indptr'];shape=z['shape']
        if tuple(shape)!=(69878,9031) or int(indptr[-1])!=m['train_unique_pairs']:raise CommandError('BAD_TRAIN_SHAPE')
        for u in users:
            uid=u['canonical_user_id']
            if not 0<=uid<69878 or u['train_count']!=int(indptr[uid+1]-indptr[uid]):raise CommandError('BAD_TRAIN_MEMBERSHIP')
        ResearchUser.objects.bulk_create([ResearchUser(dataset=d,**r) for r in users],ignore_conflicts=True,batch_size=2000)
        start=int(d.manifest.get('resume_user',0)) # Cursor is separately stored below; immutable manifest never edited.
        # imported_pairs always marks a whole user chunk; recover cursor by count of completed users.
        start=0
        if d.imported_pairs:
            positions=np.where(indptr==d.imported_pairs)[0]
            if not len(positions):raise CommandError('CORRUPT_IMPORT_CURSOR')
            start=int(positions[-1])
        stop=min(69878,opts['stop_after_users'])
        table=ResearchTrainInteraction._meta.db_table
        for first in range(start,stop,500):
            end=min(first+500,stop)
            with transaction.atomic():
                with connection.cursor() as cur:
                    buf=[]
                    for uid in range(first,end):
                        for mid in indices[indptr[uid]:indptr[uid+1]]:
                            buf.append((d.pk,uid,int(mid)))
                            if len(buf)==10000:
                                cur.executemany(f'INSERT IGNORE INTO `{table}` (dataset_id,canonical_user_id,canonical_movie_id) VALUES (%s,%s,%s)',buf);buf=[]
                    if buf:cur.executemany(f'INSERT IGNORE INTO `{table}` (dataset_id,canonical_user_id,canonical_movie_id) VALUES (%s,%s,%s)',buf)
                d.imported_pairs=int(indptr[end]);d.save(update_fields=['imported_pairs'])
            if end%5000==0 or end==stop:self.stdout.write(f'TRAIN_IMPORT users={end} pairs={d.imported_pairs}',ending='\n');self.stdout.flush()
        if stop<69878:return
        count=ResearchTrainInteraction.objects.filter(dataset=d).count()
        if count!=m['train_unique_pairs']:raise CommandError(f'PAIR_COUNT_MISMATCH:{count}')
        # Exact external identity AND exact title; no fuzzy matching, no guessed IDs.
        business={str(x.movielens_id):x for x in Movie.objects.exclude(movielens_id__isnull=True)}
        bridges=[];counts={'VERIFIED_UNIQUE':0,'AMBIGUOUS':0,'UNMAPPED':0}
        for r in movies:
            b=business.get(r['source_movie_id']);status='UNMAPPED';pk=None
            if b:
                status='VERIFIED_UNIQUE' if normalized_title(b.title)==normalized_title(r['title']) else 'AMBIGUOUS'
                if status=='VERIFIED_UNIQUE':pk=b.pk
            counts[status]+=1
            bridges.append(MovieIDBridge(source_dataset=M3L,source_movie_id=r['source_movie_id'],canonical_movie_id=str(r['canonical_movie_id']),django_movie_id=pk,
                mapping_status=status,mapping_version=MAPPING_VERSION,match_method='EXACT_ID_TITLE_TRAILING_YEAR_ONLY',
                evidence={'catalogue_sha256':m['catalogue_sha256'],'business_title':b.title if b else None,'source_namespace':'movielens:movieId','no_fuzzy_matching':True}))
        with transaction.atomic():
            existing=MovieIDBridge.objects.filter(source_dataset=M3L,mapping_version=MAPPING_VERSION)
            if existing.exists():
                if existing.count()!=9031:raise CommandError('INCOMPLETE_EXISTING_BRIDGES')
            else:MovieIDBridge.objects.bulk_create(bridges,batch_size=1000)
            d.complete=True;d.save(update_fields=['complete'])
        self.stdout.write(json.dumps({'status':'COMPLETE','pairs':count,'mapping':counts,'NEW_TEST_ACCESS':0}))
