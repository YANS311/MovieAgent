from pathlib import Path
from dotenv import dotenv_values
import os,sys,json,time,uuid,secrets,argparse
p=argparse.ArgumentParser();p.add_argument('--base',default='http://127.0.0.1:8770');p.add_argument('--env',required=True);a=p.parse_args()
config=Path(a.env).resolve();values={k:v for k,v in dotenv_values(config).items() if v is not None};O=Path(values['PAMK_EVIDENCE_DIR']).parent;S=Path(__file__).resolve().parents[1];os.environ.update(values,DJANGO_SETTINGS_MODULE='movie.release_settings',NEW_TEST_ACCESS='0');sys.path.insert(0,str(S));import django;django.setup()
from myapp.models import UserInfo,UserRating
from research_recommendation.models import UnifiedMovieFeedback,ResearchMovie,ProductToolCall,RecommendationTrace,ProfileGrant,PendingPreferenceAction
from research_recommendation.unified_catalogue import catalogue
from research_recommendation.inductive_service import business_context
from research_recommendation.product_service import business_seen
import requests
checks=[];records=[];B=a.base

def ck(name,value,detail=None):checks.append({'case':name,'pass':bool(value),'detail':detail});print(('PASS ' if value else 'FAIL ')+name,flush=True)
def session():s=requests.Session();s.trust_env=False;return s
def request(s,path,payload=None,label=None):
 t=time.monotonic();r=s.get(B+path,timeout=120) if payload is None else s.post(B+path,json=payload,headers={'X-CSRFToken':s.cookies.get('csrftoken','')},timeout=180)
 try:b=r.json()
 except ValueError:b={'html':r.text[:100]}
 records.append({'case':label or path,'method':'GET' if payload is None else 'POST','payload':payload,'status':r.status_code,'seconds':time.monotonic()-t,'response':b});return r.status_code,b

def new(label):
 s=session();s.get(B+'/register/');name='completion_'+label+'_'+uuid.uuid4().hex[:8];pw=secrets.token_urlsafe(22);r=s.post(B+'/register/',data={'username':name,'email':name+'@example.invalid','password1':pw,'password2':pw,'csrfmiddlewaretoken':s.cookies['csrftoken']});assert r.status_code==200 and r.url.endswith('/recommend/')
 return s,UserInfo.objects.get(username=name),{'username':name,'password':pw}
def feedback(s,cid,action,rating=None):
 data={'identity':f'm3l:{cid}','action':action}
 if rating is not None:data['rating']=rating
 return request(s,'/api/catalog/feedback/',data,'feedback_'+action+'_'+str(cid))
sa,ua,aa=new('a');sb,ub,ab=new('b');sc,uc,ac=new('c');private=config.parent/'completion_accounts.json';private.write_text(json.dumps({'a':aa,'b':ab,'c':ac}));private.chmod(0o600)
try:
 anon=session();anon.get(B+'/login/');st,b=request(anon,'/api/agent/',{'message':'推荐电影'});ck('anonymous_agent_401',st==401);st,b=request(anon,'/api/catalog/feedback/',{'identity':'m3l:0','action':'like'});ck('anonymous_feedback_401',st==401)
 st,ids=request(sa,'/api/catalog/identities/');ck('all9031_queryable_identity_manifest',st==200 and ids['count']==9031 and [x['canonical_movie_id'] for x in ids['identities']]==list(range(9031)))
 cards=catalogue();canonical=[x for x in cards if x['canonical_movie_id'] is not None];ck('no_duplicate_identity',len(cards)==len({x['identity'] for x in cards}));ck('no_duplicate_trusted_business',len([x['django_movie_id'] for x in canonical if x['mapping_status']=='VERIFIED_UNIQUE'])==len({x['django_movie_id'] for x in canonical if x['mapping_status']=='VERIFIED_UNIQUE'}));ck('catalogue_counts',len(cards)==12701 and len(canonical)==9031)
 st,zero=request(sa,'/api/catalog/resolve/?identity=m3l:0');ck('canonical_zero_real_resolver',st==200 and zero['canonical_movie_id']==0 and zero['identity']=='m3l:0')
 st,alias=request(sa,'/api/catalog/resolve/?identity=business:'+str(zero['django_movie_id']));ck('business_alias_same_identity',st==200 and alias['identity']=='m3l:0')
 un=[x for x in canonical if x['mapping_status']=='UNMAPPED'];amb=[x for x in canonical if x['mapping_status']=='AMBIGUOUS'];x,y=un[0]['canonical_movie_id'],un[1]['canonical_movie_id'];z=un[2]['canonical_movie_id']
 for item in [un[0],amb[0]]:
  st,b=request(sa,item['detail_url']);ck('unmapped_or_ambiguous_detail_'+str(item['canonical_movie_id']),st==200)
  ck('no_false_business_link_'+str(item['canonical_movie_id']),item['django_movie_id'] is None and item['detail_url'].startswith('/research/movies/'))
 st,b=request(sa,'/api/catalog/?scope=research&sort=canonical&page_size=50');ck('canonical_pagination_zero',st==200 and b['total']==9031 and b['items'][0]['canonical_movie_id']==0)
 st,b=request(sa,'/api/catalog/?scope=research&sort=canonical&page_size=50&page=181');ck('canonical_last_page',st==200 and b['items'][-1]['canonical_movie_id']==9030)
 for s,cid in [(sa,x),(sb,y)]:st,b=feedback(s,cid,'rating',5);ck('real_canonical_rating_'+str(cid),st==200 and b['model_history_eligible'])
 ck('unmapped_feedback_no_fake_business_id',UnifiedMovieFeedback.objects.filter(user=ua,research_movie__canonical_movie_id=x,business_movie__isnull=True).exists() and not UserRating.objects.filter(user=ua).exists())
 st,ra=request(sa,'/api/recommendations/',{'top_k':7},'direct_canonical_a_gpu');st2,rb=request(sb,'/api/recommendations/',{'top_k':7},'direct_canonical_b_gpu');ck('unmapped_history_real_pamk_gpu',st==st2==200 and ra.get('model_invoked') and rb.get('model_invoked') and ra['worker']['device']=='cuda:0' and ra['history_count']==rb['history_count']==1)
 if st!=200 or st2!=200:raise RuntimeError('GPU gate failed')
 ck('different_histories_different_vectors',ra['worker']['user_vector_sha256']!=rb['worker']['user_vector_sha256']);ck('own_identity_no_research_user',ua.user_ID is None and ub.user_ID is None and not ProfileGrant.objects.filter(user__in=[ua,ub]).exists() and ra['canonical_user_id'] is None)
 feedback(sa,0,'rating',5);st,r2=request(sa,'/api/recommendations/',{'top_k':7},'after_zero_feedback');ck('zero_in_history_and_filtered',business_context(ua,business_seen(ua))[1][-1]==0 and not {0,x}.intersection(v['canonical_movie_id'] for v in r2['recommendations']));ck('feedback_real_input_score_change',st==200 and r2['model_invoked'] and r2['context_identity']!=ra['context_identity'] and r2['worker']['user_vector_sha256']!=ra['worker']['user_vector_sha256'] and r2['worker']['score_digest']!=ra['worker']['score_digest'])
 st,b=request(sa,'/api/catalog/feedback/',{'identity':'business:'+str(zero['django_movie_id']),'action':'rating','rating':4.5});ck('alias_rating_one_canonical_row',st==200 and UnifiedMovieFeedback.objects.filter(user=ua,research_movie__canonical_movie_id=0).count()==1 and b['identity']=='m3l:0')
 feedback(sc,z,'collect');st,cr=request(sc,'/api/recommendations/',{'top_k':4});ck('canonical_collection_gpu',st==200 and cr['model_invoked'] and cr['history_count']==1)
 feedback(sc,z,'uncollect');st,cr=request(sc,'/api/recommendations/',{});ck('uncollect_removes_positive_history',st==200 and cr['model_name'] is None and not UnifiedMovieFeedback.objects.get(user=uc,research_movie__canonical_movie_id=z).collected)
 feedback(sa,x,'dislike');st,r3=request(sa,'/api/recommendations/',{'top_k':7});ck('dislike_excluded_not_positive',st==200 and r3['history_count']==1 and not {0,x}.intersection(v['canonical_movie_id'] for v in r3['recommendations']));ck('other_user_feedback_unchanged',UnifiedMovieFeedback.objects.get(user=ub,research_movie__canonical_movie_id=y).rating==5)
 for data,status in [({'identity':'m3l:999999','action':'like'},404),({'identity':'m3l:-1','action':'like'},400),({'identity':'m3l:0','action':'rating','rating':6},400),({'identity':'m3l:0','action':'like','user_id':ub.pk},400)]:st,b=request(sa,'/api/catalog/feedback/',data);ck('invalid_feedback_'+str(data),st==status)
 st,b=request(sa,'/history/');ck('personal_history_page',st==200)
 # Actual model-selected native tool calls; no deterministic planner injected.
 def agent(message,session_id=None,**extra):return request(sa,'/api/agent/',{'message':message,**({'session_id':session_id} if session_id else {}),**extra},'agent_'+message)
 st,search=agent('搜索2020年前后的科幻电影。');ck('real_llm_search_native_tool',st==200 and search.get('planning_mode')=='OLLAMA_NATIVE_TOOL_CALLS' and any(c['tool']=='search_movies' and c['outcome']=='PASS' for c in search.get('tool_calls',[])),search)
 if st!=200:raise RuntimeError('real LLM search failed')
 sid=search['session_id'];st,rec=agent('根据我喜欢的电影推荐一些作品。',sid);rc=next((c['result'] for c in rec.get('tool_calls',[]) if c['tool']=='recommend_movies'),{});ck('agent_real_http_gpu_trace',st==200 and rc.get('model_invoked') and rc.get('worker',{}).get('device')=='cuda:0' and rc['source']=='PAMK_HISTORY_MEAN_PROFILE',rec)
 if not rc.get('recommendations'):raise RuntimeError('Agent recommendation gate failed')
 target=rc['recommendations'][0];identity='m3l:'+str(target['canonical_movie_id']);st,ex=agent('为什么给我推荐这部电影？',sid,trace_id=rc['trace_id'],selected_identity=identity);exp=next((c['result'] for c in ex.get('tool_calls',[]) if c['tool']=='explain_recommendation'),{});ck('agent_trace_factual_explanation',st==200 and exp.get('facts',{}).get('recorded_score')==target['score'] and exp.get('trace_id')==rc['trace_id'] and exp.get('explanation_kind')=='RECORDED_FACTS_NOT_CAUSAL',ex)
 count=UnifiedMovieFeedback.objects.filter(user=ua).count();st,proposal=agent('我喜欢电影 '+identity+'，请准备保存喜欢偏好。',sid,selected_identity=identity);pending=next((c['result'] for c in proposal.get('tool_calls',[]) if c['tool']=='update_preferences'),{});ck('agent_proposal_no_unconfirmed_write',st==200 and pending.get('requires_confirmation') and UnifiedMovieFeedback.objects.filter(user=ua).count()==count,proposal)
 if not pending.get('pending_action_id'):raise RuntimeError('Agent proposal gate failed')
 pid=pending['pending_action_id'];st,b=request(sb,'/api/agent/',{'message':'confirm','session_id':sid,'confirm_action_id':pid,'confirmed':True},'cross_account_pending');ck('cross_account_session_rejected',st==404)
 st,confirm=request(sa,'/api/agent/',{'message':'确认保存本人偏好','session_id':sid,'confirm_action_id':pid,'confirmed':True},'confirmed_write');ck('agent_actual_confirmed_preference',st==200 and PendingPreferenceAction.objects.get(pk=pid).applied and UnifiedMovieFeedback.objects.get(user=ua,research_movie__canonical_movie_id=target['canonical_movie_id']).preference=='like')
 stamp=UnifiedMovieFeedback.objects.get(user=ua,research_movie__canonical_movie_id=target['canonical_movie_id']).updated_at;st,again=request(sa,'/api/agent/',{'message':'重复确认','session_id':sid,'confirm_action_id':pid,'confirmed':True},'idempotent_confirmation');ck('confirmation_idempotent',st==200 and UnifiedMovieFeedback.objects.get(user=ua,research_movie__canonical_movie_id=target['canonical_movie_id']).updated_at==stamp)
 st,after=agent('根据我喜欢的电影推荐一些作品。',sid);ar=next((c['result'] for c in after.get('tool_calls',[]) if c['tool']=='recommend_movies'),{});ck('agent_write_changes_actual_pamk_input',st==200 and ar.get('model_invoked') and ar['context_identity']!=rc['context_identity'] and ar['worker']['user_vector_sha256']!=rc['worker']['user_vector_sha256'] and ar['worker']['score_digest']!=rc['worker']['score_digest'])
 ck('tool_execution_ids_persisted',all(ProductToolCall.objects.filter(pk=c['tool_call_id']).exists() for response in [search,rec,ex,proposal,confirm,again,after] for c in response.get('tool_calls',[])))
 st,b=request(sa,'/api/agent/',{'message':'推荐电影','user_id':ub.pk});ck('agent_cannot_take_other_user_argument',st==400)
 st,b=request(sb,'/api/agent/',{'message':'解释推荐','trace_id':rc['trace_id']});ck('agent_cross_trace_rejected',st==404)
except Exception as error:
 records.append({'failure':repr(error)});raise
finally:
 out={'checks':checks,'pass':sum(c['pass'] for c in checks),'fail':sum(not c['pass'] for c in checks),'requests':records,'canonical_test_ids':{'unmapped_a':x if 'x' in locals() else None,'unmapped_b':y if 'y' in locals() else None},'base':B,'NEW_TEST_ACCESS':0};(O/'evidence/completion_http.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('SUMMARY',out['pass'],out['fail'],flush=True)
assert not out['fail']
