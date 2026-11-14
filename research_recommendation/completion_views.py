"""Unified catalogue / feedback / native-tool Agent HTTP boundaries."""
import json,time
from urllib.parse import urlencode
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import render,redirect
from django.views.decorators.http import require_POST,require_GET
from django.views.decorators.cache import never_cache
from myapp.models import Genre
from .models import ResearchMovie,UnifiedMovieFeedback,RecommendationTrace
from .contracts import RecommendationError,M3L
from .unified_catalogue import search_movies,resolve,catalogue
from .unified_feedback import write_feedback,feedback_for,state

def response_error(error):return JsonResponse({'error':error.code},status=error.status)
def args_search(request):
 allowed={'q','query','genre','year_from','year_to','sort','scope','page','page_size'}
 if set(request.GET)-allowed:raise RecommendationError('INVALID_SEARCH_FIELDS')
 data={k:request.GET[k] for k in request.GET};genre=data.get('genre','')
 if len(genre)>100 or genre.isdecimal() and len(genre)>10:raise RecommendationError('INVALID_GENRE')
 if genre.isdecimal():genre=Genre.objects.filter(pk=int(genre)).values_list('name',flat=True).first() or ''
 try:
  return {'query':data.get('q',data.get('query','')),'genre':genre,'year_from':int(data['year_from']) if data.get('year_from') else None,'year_to':int(data['year_to']) if data.get('year_to') else None,'sort':data.get('sort','score'),'scope':data.get('scope','all'),'page':int(data.get('page') or 1),'page_size':int(data.get('page_size') or 20)}
 except ValueError:raise RecommendationError('INVALID_SEARCH_ARGUMENTS') from None

@require_GET
def catalogue_api(request):
 try:return JsonResponse(search_movies(**args_search(request)),json_dumps_params={'ensure_ascii':False,'allow_nan':False})
 except RecommendationError as e:return response_error(e)

@require_GET
def identities_api(request):
 rows=list(ResearchMovie.objects.filter(dataset__protocol=M3L,dataset__complete=True).order_by('canonical_movie_id').values('canonical_movie_id','source_movie_id'))
 return JsonResponse({'protocol':M3L,'count':len(rows),'identities':[{'identity':'m3l:'+str(x['canonical_movie_id']),'canonical_movie_id':x['canonical_movie_id'],'source_movie_id':x['source_movie_id'],'url':'/research/movies/'+str(x['canonical_movie_id'])+'/'} for x in rows]})

@require_GET
def resolve_api(request):
 try:return JsonResponse(resolve(request.GET.get('identity')).card,json_dumps_params={'ensure_ascii':False,'allow_nan':False})
 except RecommendationError as e:return response_error(e)

def unified_browse(request):
 try:result=search_movies(**args_search(request))
 except RecommendationError as e:return response_error(e)
 query=request.GET.copy();query.pop('page',None);filters=query.urlencode();result.update(has_previous=result['page']>1,has_next=result['page']<result['pages'],previous_page=result['page']-1,next_page=result['page']+1)
 return render(request,'pamk/unified_browse.html',{'results':result,'q':request.GET.get('q',''),'genres':Genre.objects.order_by('name'),'genre':request.GET.get('genre',''),'sort':request.GET.get('sort','score'),'scope':request.GET.get('scope','all'),'filter_query':filters})

@never_cache
@login_required
def canonical_detail(request,canonical_id):
 try:film=resolve('m3l:'+str(canonical_id))
 except RecommendationError as e:return response_error(e)
 if film.business:return redirect(film.card['detail_url'])
 feedback=feedback_for(request.user,film)
 return render(request,'pamk/canonical_detail.html',{'film':film.card,'feedback':feedback,'state':state(feedback) if feedback else {}})

@never_cache
@login_required
@require_POST
def canonical_interaction(request,canonical_id):
 try:
  score=float(request.POST['score']) if request.POST.get('kind')=='rating' else None
  result=write_feedback(request.user,'m3l:'+str(canonical_id),request.POST.get('kind'),score)
  return redirect('/research/movies/'+str(canonical_id)+'/')
 except (ValueError,KeyError):return JsonResponse({'error':'INVALID_SCORE'},status=400)
 except RecommendationError as e:return response_error(e)

@never_cache
@require_POST
def feedback_api(request):
 if not request.user.is_authenticated:return JsonResponse({'error':'AUTHENTICATION_REQUIRED'},status=401)
 try:
  if len(request.body)>4096:raise RecommendationError('REQUEST_TOO_LARGE',413)
  data=json.loads(request.body)
  if type(data) is not dict or set(data)-{'identity','action','rating'}:raise RecommendationError('INVALID_FEEDBACK_FIELDS')
  r=write_feedback(request.user,data.get('identity'),data.get('action'),data.get('rating'))
  return JsonResponse(r,json_dumps_params={'ensure_ascii':False,'allow_nan':False})
 except (ValueError,UnicodeDecodeError):return JsonResponse({'error':'INVALID_JSON'},status=400)
 except RecommendationError as e:return response_error(e)

@never_cache
@login_required
def history(request):
 from myapp.models import UserRating,Collect
 rows=[];seen=set()
 for feedback in UnifiedMovieFeedback.objects.filter(user=request.user).select_related('research_movie','business_movie').order_by('-updated_at')[:100]:
  identity='m3l:'+str(feedback.research_movie.canonical_movie_id) if feedback.research_movie else 'business:'+str(feedback.business_movie_id)
  try:film=resolve(identity)
  except RecommendationError:continue
  rows.append({'film':film.card,'state':state(feedback)});seen.add(film.identity)
 # Keep earlier business feedback visible without duplicating trusted aliases.
 legacy=list(UserRating.objects.filter(user=request.user).order_by('-comment_time').values('movie_id','score','comment_time')[:100])
 collects=set(Collect.objects.filter(user=request.user).values_list('movie_id',flat=True))
 for r in legacy:
  try:film=resolve('business:'+str(r['movie_id']))
  except RecommendationError:continue
  if film.identity in seen:continue
  rows.append({'film':film.card,'state':{'rating':r['score'],'collected':r['movie_id'] in collects,'preference':'','watched':False,'updated_at':r['comment_time'].isoformat()}});seen.add(film.identity)
 for mid in sorted(collects):
  if len(rows)>=100:break
  try:film=resolve('business:'+str(mid))
  except RecommendationError:continue
  if film.identity in seen:continue
  rows.append({'film':film.card,'state':{'rating':None,'collected':True,'preference':'','watched':False,'updated_at':''}});seen.add(film.identity)
 rows.sort(key=lambda x:x['state']['updated_at'],reverse=True)
 return render(request,'pamk/history.html',{'rows':rows[:100]})

@never_cache
@require_POST
def agent_api(request):
 if not request.user.is_authenticated:return JsonResponse({'error':'AUTHENTICATION_REQUIRED'},status=401)
 from myapp.agent.product_agent import run_agent
 try:
  if len(request.body)>8192:raise RecommendationError('REQUEST_TOO_LARGE',413)
  payload=json.loads(request.body);r=run_agent(request.user,payload)
  return JsonResponse(r,json_dumps_params={'ensure_ascii':False,'allow_nan':False})
 except (ValueError,UnicodeDecodeError):return JsonResponse({'error':'INVALID_JSON'},status=400)
 except RecommendationError as e:return response_error(e)

@require_GET
@login_required
def tools_api(request):
 from myapp.agent.product_agent import registry
 return JsonResponse({'tools':[s.to_dict() for s in registry().list_all()],'planner':'OLLAMA_NATIVE_TOOL_CALLS'})
