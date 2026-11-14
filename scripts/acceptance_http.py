"""Real TCP HTTP, MySQL, Redis and GPU acceptance. Never mocks model scoring."""
from pathlib import Path
import os,sys,json,time,uuid,secrets,statistics,argparse
from dotenv import dotenv_values
p=argparse.ArgumentParser();p.add_argument('--env',required=True);p.add_argument('--base',default='http://127.0.0.1:8770');p.add_argument('--accounts',required=True,help='Private credentials for existing synthetic research accounts');a=p.parse_args();env={k:v for k,v in dotenv_values(a.env).items() if v is not None};os.environ.update(env,DJANGO_SETTINGS_MODULE='movie.release_settings',NEW_TEST_ACCESS='0');S=Path(__file__).resolve().parents[1];sys.path.insert(0,str(S));import django;django.setup()
from myapp.models import UserInfo,Movie,UserRating,Collect
from research_recommendation.models import ProfileGrant,RecommendationTrace,ResearchTrainInteraction,MovieIDBridge
from research_recommendation.pamk_identity import CHECKPOINT_SHA256
import requests
O=Path(env['PAMK_EVIDENCE_DIR']);O.mkdir(parents=True,exist_ok=True);checks=[];logs=[];BASE=a.base

def ck(n,x,detail=None):checks.append({'case':n,'pass':bool(x),'detail':detail});print(('PASS ' if x else 'FAIL ')+n,flush=True)
def sess():s=requests.Session();s.trust_env=False;return s
def new(label):
 s=sess();s.get(BASE+'/register/');pw=secrets.token_urlsafe(22);name='final_'+label+'_'+uuid.uuid4().hex[:8];r=s.post(BASE+'/register/',data={'username':name,'email':name+'@example.invalid','password1':pw,'password2':pw,'csrfmiddlewaretoken':s.cookies['csrftoken']});assert r.status_code==200 and r.url.endswith('/recommend/')
 return s,UserInfo.objects.get(username=name),{'username':name,'password':pw}
def call(s,payload,label):
 t=time.monotonic();r=s.post(BASE+'/api/recommendations/',json=payload,headers={'X-CSRFToken':s.cookies.get('csrftoken','')},timeout=180)
 try:b=r.json()
 except ValueError:b={'raw':r.text[:300]}
 logs.append({'case':label,'payload':payload,'status':r.status_code,'http_seconds':time.monotonic()-t,'response':b});return r.status_code,b
def feedback(s,mid,kind='rating',score=5):return s.post(BASE+f'/movie/{mid}/interaction/',data={'kind':kind,'score':score,'csrfmiddlewaretoken':s.cookies['csrftoken']})
sa,ua,aa=new('animation');sb,ub,ab=new('thriller');sc,uc,ac=new('edge')
private={'animation':aa,'thriller':ab,'edge':ac};cred=Path(a.env).parent/'final_accounts.json';cred.write_text(json.dumps(private));cred.chmod(0o600)
for s,u in [(sa,ua),(sb,ub),(sc,uc)]:
 st,b=call(s,{},'empty_'+str(u.pk));ck('empty_cold_source',st==200 and b['model_name'] is None and b['source']=='METADATA_COLD_START');ck('no_borrowed_research_identity',u.user_ID is None and not ProfileGrant.objects.filter(user=u).exists())
verified=list(MovieIDBridge.objects.filter(mapping_version=env['PAMK_MAPPING_VERSION'],mapping_status='VERIFIED_UNIQUE').values_list('django_movie_id',flat=True))
ma=list(Movie.objects.filter(pk__in=verified,genres__name='动画').distinct().order_by('id')[:5]);mb=list(Movie.objects.filter(pk__in=verified,genres__name='惊悚').exclude(pk__in=[m.pk for m in ma]).distinct().order_by('id')[:5]);assert len(ma)==len(mb)==5
for s,movies in [(sa,ma[:3]),(sb,mb[:3])]:
 for m in movies:ck('http_real_rating_'+str(m.pk),feedback(s,m.pk).status_code==200)
st,ra=call(sa,{'top_k':20},'default_mean_a');ck('default_mean_gpu',st==200 and ra.get('source')=='PAMK_HISTORY_MEAN_PROFILE',ra if st!=200 else ra['worker'])
if st!=200:(O/'http_failure.json').write_text(json.dumps(logs));raise SystemExit(1)
st,rb=call(sb,{'top_k':20},'default_mean_b');ck('second_own_mean_gpu',st==200 and rb['source']=='PAMK_HISTORY_MEAN_PROFILE');ck('distinct_vectors',ra['worker']['user_vector_sha256']!=rb['worker']['user_vector_sha256']);ck('distinct_rankings',[x['canonical_movie_id'] for x in ra['recommendations']]!=[x['canonical_movie_id'] for x in rb['recommendations']]);ck('mean_not_claimed_trained_encoder',ra['adapter_checkpoint_sha256'] is None and ra['encoding_method']=='mean' and ra['worker']['encoder_call_count']==0)
for b,u,seen in [(ra,ua,ma[:3]),(rb,ub,mb[:3])]:
 ck('cuda_real_receipt',b['worker']['device']=='cuda:0' and b['worker']['inference_count']>0 and b['worker']['gpu_forward_ms']>0 and b['worker']['candidate_count']==9028)
 ck('base_identity',b['checkpoint_sha256']==CHECKPOINT_SHA256)
 ck('no_research_claim',b['canonical_user_id'] is None and not b['research_demo_only'])
 ck('trace_persisted',RecommendationTrace.objects.filter(pk=b['trace_id'],requester=u).exists())
 ck('seen_filtered',not {m.pk for m in seen}.intersection(x['django_movie_id'] for x in b['recommendations']))
 ck('finite_sorted_full_k',len(b['recommendations'])==20 and all(__import__('math').isfinite(x['score']) for x in b['recommendations']) and all(b['recommendations'][i]['score']>=b['recommendations'][i+1]['score'] for i in range(19)))
 ck('no_fake_movie_link',all(x['detail_url'] is None and x['django_movie_id'] is None for x in b['recommendations'] if x['mapping_status']!='VERIFIED_UNIQUE'))
st,hit=call(sa,{'top_k':20},'cache_hit');ck('cache_hit_no_gpu',st==200 and not hit['model_invoked'] and hit['cache_status']=='HIT')
st,repk=call(sa,{'top_k':10},'representation_hit');ck('cached_representation_real_score',st==200 and repk['model_invoked'] and repk['worker']['representation_cache_hit'])
feedback(sa,ma[3].pk);st,changed=call(sa,{'top_k':20},'new_positive');ck('new_history_invalidation',st==200 and changed['model_invoked'] and changed['history_count']==4 and changed['worker']['user_vector_sha256']!=ra['worker']['user_vector_sha256'] and changed['context_identity']!=ra['context_identity']);ck('new_score_change',changed['worker']['score_digest']!=ra['worker']['score_digest'])
feedback(sa,ma[3].pk,score=4.5);st,dup=call(sa,{'top_k':10},'duplicate_rating');ck('duplicate_one_row',UserRating.objects.filter(user=ua,movie=ma[3]).count()==1 and dup['history_count']==4)
st,learned=call(sa,{'top_k':20,'model_name':'PAMK-Inductive-v1'},'explicit_learned');ck('learned_switch_real_gpu',st==200 and learned['model_name']=='PAMK-Inductive-v1' and learned['encoding_method']=='learned' and learned['model_invoked'] and learned['worker']['encoder_call_count']>0 and learned['adapter_checkpoint_sha256']==json.loads(Path(env['PAMK_INDUCTIVE_MANIFEST']).read_text())['sha256']);ck('strategy_cache_isolation',learned['encoding_identity']!=changed['encoding_identity'] and learned['worker']['user_vector_sha256']!=changed['worker']['user_vector_sha256'])
feedback(sc,mb[0].pk,score=1);st,low=call(sc,{},'only_low_score');ck('low_score_no_fake_positive',st==200 and low['model_name'] is None)
unmapped=Movie.objects.exclude(pk__in=verified).order_by('id').first();feedback(sc,unmapped.pk);st,um=call(sc,{},'only_unmapped_positive');ck('unmapped_honest_cold',st==200 and um['model_name'] is None and um['reason']=='NO_TRUSTED_MAPPED_POSITIVE_HISTORY')
feedback(sc,mb[1].pk,kind='collect');st,short=call(sc,{},'one_valid_collection');ck('single_collection_gpu',st==200 and short['model_name']=='PAMK-HistoryMean-v1' and short['history_count']==1 and Collect.objects.filter(user=uc,movie=mb[1]).exists())
ck('invalid_movie_404',feedback(sa,999999999).status_code==404)
# A movie deleted before request resolution has the same missing-object contract.
fixture=Movie.objects.create(title='Disposable isolated acceptance fixture',score=1);deleted=fixture.pk;fixture.delete();ck('deleted_movie_404',feedback(sa,deleted).status_code==404)
ck('cross_trace_404',sb.get(BASE+'/recommendation-traces/'+ra['trace_id']+'/').status_code==404)
for path in ['/','/depot/','/recommend/','/preferences/',f'/movie/{ma[0].pk}/']:
 ck('page_'+path,sa.get(BASE+path).status_code==200)
for status in ['UNMAPPED','AMBIGUOUS']:
 item=next((x for x in ra['recommendations'] if x['mapping_status']==status),None)
 if item:ck('research_detail_'+status,sa.get(BASE+item['research_detail_url']).status_code==200)
accounts=json.loads(Path(a.accounts).read_text());native=accounts['pamk_short'];sn=sess();sn.get(BASE+'/login/');rr=sn.post(BASE+'/login/',data={'username':native['username'],'password':native['password'],'csrfmiddlewaretoken':sn.cookies['csrftoken']});ck('native_login',rr.status_code==200 and rr.url.endswith('/recommend/'))
for k in [100,10,11,12,13]:
 st,b=call(sn,{'top_k':k},'native_'+str(k));seen=set(ResearchTrainInteraction.objects.filter(canonical_user_id=native['canonical_user_id']).values_list('canonical_movie_id',flat=True));ck('native_gpu_'+str(k),st==200 and b['source']=='PAMK_RESEARCH_PROFILE' and b['worker']['device']=='cuda:0');ck('native_train_zero_'+str(k),st==200 and not seen.intersection(int(x['canonical_movie_id']) for x in b['recommendations']) and b['worker']['canonical_zero_eligible']==(0 not in seen));ck('native_correct_identity_'+str(k),b['checkpoint_sha256']==CHECKPOINT_SHA256 and b['research_demo_only'] and not b['identity_link_verified'])
for k in [14,15,16,17,18]:
 st,b=call(sa,{'top_k':k},'mean_warm_'+str(k));ck('warm_mean_gpu_'+str(k),st==200 and b['model_invoked'] and b['encoding_method']=='mean')
anon=sess();anon.get(BASE+'/login/');st,b=call(anon,{},'anonymous');ck('anonymous_401',st==401)
for payload,code,label in [({'canonical_user_id':'0'},400,'forged_user'),({'profile_id':native['profile_id']},403,'cross_profile'),({'model_name':'PAMK-AV'},503,'pamk_av'),({'model_name':[]},400,'invalid_type')]:st,b=call(sa,payload,label);ck(label,st==code)
summary={'pass':sum(c['pass'] for c in checks),'fail':sum(not c['pass'] for c in checks),'checks':checks,'gpu_calls':sum(r['response'].get('model_invoked',r['response'].get('source')=='PAMK_RESEARCH_PROFILE') for r in logs),'source_checkout':str(S),'accounts_private_file':str(cred),'feedback_movies':{'animation':[m.pk for m in ma],'thriller':[m.pk for m in mb]},'NEW_TEST_ACCESS':0}
(O/'http_requests.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2));(O/'http_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='checks'},ensure_ascii=False));assert not summary['fail']
