"""Actual product routes using native accounts, movies and persistent traces."""
import json
import time
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

def browse(request):
    q=request.GET.get('q','')[:200]
    rows=Movie.objects.filter(title__icontains=q).order_by('-score','-vote_count','id')
    return render(request,'pamk/browse.html',{'page':Paginator(rows,24).get_page(request.GET.get('page')), 'q':q})

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
        result=recommend(request.user,data)
        result['server_total_seconds']=time.monotonic()-start
        return JsonResponse(result,json_dumps_params={'ensure_ascii':False,'allow_nan':False})
    except (json.JSONDecodeError,UnicodeDecodeError):
        return JsonResponse({'error':'INVALID_JSON'},status=400)
    except RecommendationError as e:
        return JsonResponse({'error':e.code},status=e.status)

@never_cache
@login_required
def movie_detail(request,pk):
    movie=get_object_or_404(Movie,pk=pk)
    return render(request,'pamk/detail.html',{'movie':movie})

@never_cache
@login_required
def research_detail(request,canonical_id):
    movie=get_object_or_404(ResearchMovie,canonical_movie_id=canonical_id,dataset__protocol='M3L_AUTHOR_ALIGNED_V1')
    return render(request,'pamk/detail.html',{'research_movie':movie})

@never_cache
@login_required
@require_POST
def interaction(request,pk):
    movie=get_object_or_404(Movie,pk=pk)
    kind=request.POST.get('kind')
    if kind=='rating':
        try:score=float(request.POST['score'])
        except (ValueError,KeyError):return JsonResponse({'error':'INVALID_SCORE'},status=400)
        if not 0.5<=score<=5:return JsonResponse({'error':'INVALID_SCORE'},status=400)
        UserRating.objects.update_or_create(user=request.user,movie=movie,defaults={'score':score})
    elif kind=='collect':Collect.objects.get_or_create(user=request.user,movie=movie)
    elif kind=='watched':MovieInteraction.objects.create(user=request.user,movie_id=pk,catalogue_model='myapp.Movie',kind=kind)
    else:return JsonResponse({'error':'INVALID_INTERACTION'},status=400)
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
