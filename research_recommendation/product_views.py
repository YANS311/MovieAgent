"""Actual product routes using native accounts, movies and persistent traces."""
import json
import time
from urllib.parse import urlencode
from pathlib import Path
from django.conf import settings
from django.http import Http404
from django.views.static import serve
from .page_catalogue import PageCatalogue
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.db import transaction
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_http_methods, require_POST
from django.views.decorators.cache import never_cache
from django import forms
from myapp.models import Movie, Genre, UserRating, Collect
from .models import ResearchMovie, ProfileGrant, UserPreference, MovieInteraction, RecommendationTrace
from .contracts import RecommendationError
from .product_service import recommend

class RegisterForm(UserCreationForm):
    email = forms.EmailField()
    class Meta:
        model = get_user_model()
        fields = ('username', 'email')

@require_http_methods(['GET','POST'])
def account_login(request):
    form = AuthenticationForm(request, data=request.POST or None)
    if request.method == 'POST' and form.is_valid():
        login(request, form.get_user())
        return redirect('/recommend/')
    return render(request, 'pamk/account.html', {'form':form,'title':'用户登录'})

@require_http_methods(['GET','POST'])
def register(request):
    form = RegisterForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save(); login(request,user)
        return redirect('/recommend/')
    return render(request, 'pamk/account.html', {'form':form,'title':'注册账号'})

@require_POST
def account_logout(request):
    logout(request);return redirect('/login/')

def display_movies(rows):
    rows=list(rows)
    for movie in rows:
        movie.display_poster=PageCatalogue.card(movie).poster_url
    return rows


def home(request):
    qs=Movie.objects.all()
    popular=list(qs.order_by('-vote_count','id')[:8])
    high=list(qs.filter(score__isnull=False).order_by('-score','-vote_count','id')[:8])
    groups=[]
    for title,unit,items,field in [('热门电影','评分人数',popular,'vote_count'),('高分佳作','目录评分',high,'score')]:
        maximum=max([float(getattr(m,field) or 0) for m in items]+[1])
        groups.append({'title':title,'unit':unit,'rows':[{'movie':m,'value':getattr(m,field),'width':round(100*float(getattr(m,field) or 0)/maximum,1)} for m in items]})
    return render(request,'pamk/home.html',{'movie_count':qs.count(),'genre_count':Genre.objects.count(),
        'my_rating_count':UserRating.objects.filter(user=request.user).count() if request.user.is_authenticated else 0,
        'featured':display_movies(qs.prefetch_related('genres').filter(score__isnull=False).order_by('-vote_count','id')[:10]),'chart_groups':groups})


def rankings(request):
    mode=request.GET.get('mode','popular')
    choices={'popular':('-vote_count','热门电影榜'),'score':('-score','高分电影榜'),'recent':('-date','新片上映榜')}
    if mode not in choices:mode='popular'
    order,heading=choices[mode]
    rows=Movie.objects.prefetch_related('genres').order_by(order,'-vote_count','id')[:20]
    return render(request,'pamk/rankings.html',{'mode':mode,'heading':heading,'rows':display_movies(rows)})


def browse(request):
    from .completion_views import unified_browse
    return unified_browse(request)


@login_required
def assistant(request):
    initial={'trace_id':request.GET.get('trace_id'),'selected_identity':request.GET.get('identity')}
    if initial['trace_id']:
        try:
            from uuid import UUID
            tid=UUID(initial['trace_id'])
        except ValueError:return JsonResponse({'error':'INVALID_TRACE_ID'},status=400)
        if not RecommendationTrace.objects.filter(pk=tid,requester=request.user).exists():return JsonResponse({'error':'TRACE_NOT_FOUND'},status=404)
    return render(request,'pamk/assistant.html',{'agent_initial':initial})


def local_poster(request,filename):
    # The isolated release only serves public image files from an explicitly configured directory.
    if Path(filename).name!=filename or Path(filename).suffix.lower() not in ('.jpg','.jpeg','.png','.webp'):raise Http404
    return serve(request,filename,document_root=settings.PAMK_POSTER_ROOT)


def add_display_metadata(result):
    ids=[x.get('django_movie_id') for x in result['recommendations'] if x.get('django_movie_id') is not None]
    movies={m.pk:m for m in Movie.objects.filter(pk__in=ids).prefetch_related('genres')}
    research={r.canonical_movie_id:r for r in ResearchMovie.objects.filter(dataset__protocol='M3L_AUTHOR_ALIGNED_V1',canonical_movie_id__in=[x['canonical_movie_id'] for x in result['recommendations'] if x.get('canonical_movie_id') is not None])}
    from .unified_catalogue import _metadata,resolve
    for item in result['recommendations']:
        cid=item.get('canonical_movie_id');movie=movies.get(item.get('django_movie_id'))
        if cid is not None:item['identity']='m3l:'+str(cid)
        else:item['identity']=resolve('business:'+str(item['django_movie_id'])).identity
        if movie:
            card=PageCatalogue.card(movie)
            item.update(poster_url=card.poster_url,genres=list(card.genres),summary=card.summary[:240],year=movie.date.year if movie.date else None)
        else:
            meta=_metadata(research[cid]) if cid in research else {'genres':[],'year':None,'summary':''}
            item.update(poster_url=None,**meta)
    return result

@never_cache
@login_required
def recommendation_page(request):
    grants=ProfileGrant.objects.filter(user=request.user,is_active=True,profile__is_active=True).select_related('profile')
    return render(request,'pamk/recommend.html',{'grants':grants})

@require_POST
@never_cache
def recommendation_api(request):
    if not request.user.is_authenticated:
        return JsonResponse({'error':'AUTHENTICATION_REQUIRED'},status=401)
    start=time.monotonic()
    try:
        if len(request.body)>8192:raise RecommendationError('REQUEST_TOO_LARGE',413)
        data=json.loads(request.body)
        result=add_display_metadata(recommend(request.user,data))
        result['server_total_seconds']=time.monotonic()-start
        return JsonResponse(result,json_dumps_params={'ensure_ascii':False,'allow_nan':False})
    except (json.JSONDecodeError,UnicodeDecodeError):
        return JsonResponse({'error':'INVALID_JSON'},status=400)
    except RecommendationError as e:
        return JsonResponse({'error':e.code},status=e.status)

@never_cache
@login_required
def movie_detail(request,pk):
    movie=get_object_or_404(Movie.objects.prefetch_related('genres','directors','regions'),pk=pk)
    display_movies([movie])
    rating=UserRating.objects.filter(user=request.user,movie=movie).values_list('score',flat=True).first()
    from .unified_catalogue import resolve
    from .unified_feedback import feedback_for
    film=resolve('business:'+str(pk));feedback=feedback_for(request.user,film)
    return render(request,'pamk/detail.html',{'movie':movie,'my_rating':rating,'collected':Collect.objects.filter(user=request.user,movie=movie).exists(),'identity':film.identity,'feedback':feedback})

@never_cache
@login_required
def research_detail(request,canonical_id):
    from .completion_views import canonical_detail
    return canonical_detail(request,canonical_id)

@never_cache
@login_required
@require_POST
def interaction(request,pk):
    from .unified_feedback import write_feedback
    try:
        kind=request.POST.get('kind');score=float(request.POST['score']) if kind=='rating' else None
        write_feedback(request.user,'business:'+str(pk),kind,score)
    except (ValueError,KeyError):return JsonResponse({'error':'INVALID_SCORE'},status=400)
    except RecommendationError as e:return JsonResponse({'error':e.code},status=e.status)
    return redirect(f'/movie/{pk}/')

@never_cache
@login_required
@require_http_methods(['GET','POST'])
def preferences(request):
    genres=list(Genre.objects.order_by('name').values_list('name',flat=True))
    if request.method=='POST':
        selected=request.POST.getlist('genres')
        if len(selected)>30 or any(g not in genres for g in selected):return JsonResponse({'error':'INVALID_GENRE'},status=400)
        UserPreference.objects.update_or_create(user=request.user,defaults={'genres':selected,'catalogue_model':'myapp.Movie','movie_ids':[]})
        return redirect('/recommend/')
    pref=UserPreference.objects.filter(user=request.user).first()
    return render(request,'pamk/preferences.html',{'genres':genres,'selected':pref.genres if pref else []})

@never_cache
@login_required
def trace(request,trace_id):
    obj=get_object_or_404(RecommendationTrace,pk=trace_id,requester=request.user)
    return JsonResponse(obj.provenance,json_dumps_params={'ensure_ascii':False})

@never_cache
@login_required
def trace_page(request,trace_id):
    obj=get_object_or_404(RecommendationTrace,pk=trace_id,requester=request.user)
    return render(request,'pamk/trace.html',{'p':obj.provenance,'trace_id':str(obj.pk),'provenance_json':json.dumps(obj.provenance,ensure_ascii=False,indent=2)})
