"""Business-owned positive history → trained encoder; no research identity assignment."""
import hashlib,json
from pathlib import Path
from django.conf import settings
from django.core.cache import cache
from myapp.models import UserRating,Collect
from .models import UserPreference,MovieIDBridge,MovieInteraction
from .contracts import RecommendationError,M3L
from .product_mapping import MAPPING_VERSION

NAME='PAMK-Inductive-v1'
from .encoding_policy import policy

def manifest():
    m=json.loads(Path(settings.PAMK_INDUCTIVE_MANIFEST).read_text())
    if m['version']!=NAME or m['NEW_TEST_ACCESS']!=0:raise RecommendationError('INDUCTIVE_MANIFEST_INVALID',503)
    return m

def business_context(user,seen):
    # Positive feedback means explicit rating >=4 or a collection; low scores
    # exclude the movie, never become fabricated positive histories.
    ratings=list(UserRating.objects.filter(user=user).order_by('movie_id').values_list('movie_id','score'))
    collects=list(Collect.objects.filter(user=user).order_by('collect_time','id').values_list('movie_id',flat=True))
    rating_values=dict(ratings)
    positive=[mid for mid,score in UserRating.objects.filter(user=user).order_by('comment_time','id').values_list('movie_id','score') if score is not None and score>=4]+collects
    pref=UserPreference.objects.filter(user=user,catalogue_model='myapp.Movie').first()
    if pref:positive+=pref.movie_ids
    positive=[mid for mid in positive if rating_values.get(mid) is None or rating_values[mid]>=4]
    if not positive:
        return None,[],[],hashlib.sha256(b'empty').hexdigest(),0
    rows=list(MovieIDBridge.objects.filter(source_dataset=M3L,mapping_version=MAPPING_VERSION,mapping_status='VERIFIED_UNIQUE',django_movie_id__in=set(positive)|seen))
    by_business={}
    for b in rows:
        by_business.setdefault(b.django_movie_id,[]).append(b)
    # Verify all source IDs/titles and catalogue provenance live, not just status.
    from .product_service import dataset, map_items
    d=dataset()
    canonical=[b.canonical_movie_id for b in rows]
    mapped=map_items([{'canonical_movie_id':cid,'score':0.,'original_model_rank':i+1} for i,cid in enumerate(canonical)],d)
    trusted={x['django_movie_id']:int(x['canonical_movie_id']) for x in mapped if x['mapping_status']=='VERIFIED_UNIQUE' and len(by_business.get(x['django_movie_id'],[]))==1}
    history=[]
    for mid in positive:
        cid=trusted.get(mid)
        if cid is not None:
            if cid in history:history.remove(cid)
            history.append(cid)
    history=history[-20:]
    exclusions=sorted({trusted[mid] for mid in seen if mid in trusted}|set(history))
    evidence={'ratings':ratings,'collections':collects,'preference_genres':pref.genres if pref else [],'preference_movie_ids':pref.movie_ids if pref else [],'seen':sorted(seen),'history':history,
        'mapping_evidence':[(b.canonical_movie_id,b.source_movie_id,b.django_movie_id,b.mapping_status,b.evidence) for b in sorted(rows,key=lambda b:(b.canonical_movie_id,b.pk))]}
    identity=hashlib.sha256(json.dumps(evidence,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return d,history,exclusions,identity,len(set(positive))-len({mid for mid in positive if mid in trusted})

def recommend_inductive(user,k,seen,explicit=False,encoding_method="learned"):
    from .product_service import backend,map_items
    d,h,exclude,context,unmapped=business_context(user,seen)
    if not h:
        return None
    m=manifest();chosen=policy(encoding_method)
    key='pamk:inductive:'+hashlib.sha256(json.dumps([user.pk,context,k,chosen['encoding_identity'],m['base_sha256'],d.manifest,MAPPING_VERSION],sort_keys=True).encode()).hexdigest()
    try:receipt=cache.get(key);cache_status='HIT' if receipt else 'MISS'
    except Exception:receipt=None;cache_status='BYPASS'
    invoked=receipt is None
    if receipt is None:
        instance=backend(d)
        try:receipt=instance.call('business:'+str(user.pk),k,history=h,context_identity=context,adapter_identity=chosen['encoding_identity'],encoding_method=encoding_method,exclude_canonical_ids=exclude)
        except RecommendationError as err:
            if err.code not in {'MODEL_INFERENCE_UNAVAILABLE','MODEL_WORKER_UNAVAILABLE','MODEL_SESSION_EXPIRED'}:raise
            instance.close()
            receipt=backend(d).call('business:'+str(user.pk),k,history=h,context_identity=context,adapter_identity=chosen['encoding_identity'],encoding_method=encoding_method,exclude_canonical_ids=exclude)
        if receipt['context_identity']!=context:raise RecommendationError('CONTEXT_PROVENANCE_MISMATCH',502)
        try:cache.set(key,receipt,120)
        except Exception:cache_status='BYPASS'
    if receipt.get('encoding_identity')!=chosen['encoding_identity'] or receipt.get('user')!='business:'+str(user.pk):raise RecommendationError('CACHE_PROVENANCE_MISMATCH',502)
    ids={int(x['canonical_movie_id']) for x in receipt['items']}
    if ids.intersection(exclude):raise RecommendationError('BUSINESS_HISTORY_FILTER_MISMATCH',502)
    result={'source':chosen['source'],'model_name':chosen['model_name'],'adapter_version':chosen['version'],'adapter_checkpoint_sha256':chosen['encoder_checkpoint_sha256'],'encoding_method':encoding_method,'encoding_identity':chosen['encoding_identity'],'checkpoint_sha256':m['base_sha256'],
        'canonical_user_id':None,'research_demo_only':False,'identity_mode':'DJANGO_ACCOUNT','history_count':len(h),'unmapped_positive_feedback_count':unmapped,
        'context_identity':context,'cache_status':cache_status,'model_invoked':invoked,'mapping_version':MAPPING_VERSION,
        'recommendations':map_items(receipt['items'],d),'worker':{a:v for a,v in receipt.items() if a!='items'},
        'limitations':'New adapter, not identical to native Chapter 4 PAMK. Current internal holdout is weaker than mean-history PAMK baseline.'}
    return result
