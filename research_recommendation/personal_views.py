import json
from uuid import UUID
from functools import wraps
from django.contrib.auth.decorators import login_required
from django.db import transaction, DatabaseError
from django.http import Http404, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_GET, require_http_methods, require_POST
from .models import UserPreference, PersonalRecommendationTrace, MovieInteraction
from .page_catalogue import PageCatalogue
from .page_views import context
from .personal import RegistrationForm, PreferenceForm, catalogue, catalogue_label, rank


def storage_available(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        try:
            return view(request, *args, **kwargs)
        except (DatabaseError, LookupError):
            return render(request, 'research_recommendation/error.html', context(request,
                heading='个人推荐服务暂不可用', message='目录或存储尚未就绪，请稍后重试。'), status=503)
    return wrapped


@never_cache
@require_http_methods(['GET', 'POST'])
@storage_available
def register(request):
    if request.user.is_authenticated:
        return redirect('research_pages:preferences')
    form = RegistrationForm(request.POST if request.method == 'POST' else None)
    if request.method == 'POST' and form.is_valid():
        form.save()
        return redirect('/cinema/login/?next=/cinema/personal/preferences/')
    return render(request, 'research_recommendation/register.html', context(request, form=form))


@never_cache
@login_required(login_url='research_pages:login')
@require_http_methods(['GET', 'POST'])
@storage_available
def preferences(request):
    cards = catalogue()
    current = UserPreference.objects.filter(user=request.user).first()
    initial = {'genres': current.genres, 'movie_ids': ','.join(map(str, current.movie_ids))} if current and current.catalogue_model == catalogue_label() else {}
    form = PreferenceForm(request.POST if request.method == 'POST' else None, cards=cards, initial=initial)
    if request.method == 'POST' and form.is_valid():
        UserPreference.objects.update_or_create(user=request.user, defaults={
            **form.cleaned_data, 'catalogue_model': catalogue_label()})
        return redirect('research_pages:personal')
    return render(request, 'research_recommendation/preferences.html', context(request,
        form=form, selected_movies=[c for c in cards if current and str(c.id) in initial.get('movie_ids', '').split(',')]))


@never_cache
@login_required(login_url='research_pages:login')
@require_http_methods(['GET', 'POST'])
@storage_available
def recommendations(request):
    current = UserPreference.objects.filter(user=request.user).first()
    trace = None
    results = []
    if request.method == 'POST':
        from . import personal_cache
        if personal_cache.enabled():
            provenance = personal_cache.recommendation(request.user, current, catalogue, rank)
        else:
            results, provenance = rank(catalogue(), current)
        trace = PersonalRecommendationTrace.objects.create(user=request.user, provenance=provenance)
        return redirect('research_pages:personal_trace', trace_id=trace.pk)
    return render(request, 'research_recommendation/personal.html', context(request,
        preference=current if current and current.catalogue_model == catalogue_label() else None,
        results=results, trace=trace,
        recent_traces=PersonalRecommendationTrace.objects.filter(user=request.user).order_by('-created_at')[:10]))


@never_cache
@login_required(login_url='research_pages:login')
@require_GET
@storage_available
def trace_detail(request, trace_id):
    trace = get_object_or_404(PersonalRecommendationTrace, pk=trace_id, user=request.user)
    data = trace.provenance
    ids = [r['movie_id'] for r in data['results']]
    cards = PageCatalogue().get_many(ids) if data['catalogue_model'] == catalogue_label() else {}
    results = [{'movie': cards[r['movie_id']], 'score': r['score'],
                'reason': '匹配类型：' + '、'.join(r['matched_genres']) if r['matched_genres'] else '目录探索：暂无类型匹配'}
               for r in data['results'] if r['movie_id'] in cards]
    return render(request, 'research_recommendation/personal.html', context(request,
        preference=UserPreference.objects.filter(user=request.user).first(), results=results, trace=trace,
        trace_json=json.dumps(data, ensure_ascii=False, indent=2)))


def owned_trace(request, movie_id):
    value = request.POST.get('personal_trace') if request.method == 'POST' else request.GET.get('personal_trace')
    if not value:
        return None
    try:
        key = UUID(value)
    except (ValueError, TypeError, AttributeError):
        raise Http404
    trace = get_object_or_404(PersonalRecommendationTrace, pk=key, user=request.user)
    if trace.provenance['catalogue_model'] != catalogue_label() or movie_id not in [r['movie_id'] for r in trace.provenance['results']]:
        raise Http404
    return trace


@never_cache
@login_required(login_url='research_pages:login')
@require_POST
@storage_available
def interaction(request, movie_id):
    kind = request.POST.get('kind')
    if kind not in ('watched', 'interested'):
        return HttpResponseBadRequest('无效交互类型')
    if PageCatalogue().get(movie_id) is None:
        raise Http404
    trace = owned_trace(request, movie_id)
    with transaction.atomic():
        if kind == 'interested':
            UserPreference.objects.get_or_create(user=request.user, defaults={'catalogue_model': catalogue_label()})
            preference = UserPreference.objects.select_for_update().get(user=request.user)
            if preference.catalogue_model != catalogue_label():
                preference.movie_ids = []
                preference.genres = []
                preference.catalogue_model = catalogue_label()
            if movie_id not in preference.movie_ids:
                if len(preference.movie_ids) >= 20:
                    return HttpResponseBadRequest('最多选择20部电影，请先更新偏好。')
                preference.movie_ids = sorted([*preference.movie_ids, movie_id])
                preference.save()
        MovieInteraction.objects.create(user=request.user, movie_id=movie_id, catalogue_model=catalogue_label(), kind=kind, trace=trace)
    return redirect('research_pages:interactions')


@never_cache
@login_required(login_url='research_pages:login')
@require_GET
@storage_available
def interactions(request):
    rows = list(MovieInteraction.objects.filter(user=request.user).order_by('-created_at', '-pk')[:50])
    cards = PageCatalogue().get_many([r.movie_id for r in rows if r.catalogue_model == catalogue_label()])
    return render(request, 'research_recommendation/interactions.html', context(request,
        entries=[{'record': r, 'movie': cards.get(r.movie_id) if r.catalogue_model == catalogue_label() else None} for r in rows]))
