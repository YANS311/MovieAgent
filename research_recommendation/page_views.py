"""Django template pages. Recommendation writes/scoring stay in the existing API."""
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode
from django.conf import settings
from django.contrib.auth.views import LoginView, LogoutView
from django.http import Http404, JsonResponse
from django.shortcuts import redirect, render
from django.urls import NoReverseMatch, reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_GET
from django.views.decorators.cache import never_cache

from .page_catalogue import PageCatalogue
from . import views as api


def context(request, **extra):
    try:
        agent_url = reverse('agent_chat')
    except NoReverseMatch:
        agent_url = reverse('research_pages:assistant')
    return dict(cpu_demo=getattr(settings, 'TASK60_CPU_DEMO', False), agent_url=agent_url, **extra)


@require_GET
def home(request):
    try:
        page, cards = PageCatalogue().browse()
        return render(request, 'research_recommendation/home.html', context(request, movies=cards[:6], movie_count=page.paginator.count))
    except Exception:
        return render(request, 'research_recommendation/error.html', context(request,
            heading='影片库暂不可用', message='电影数据服务尚未连接，请稍后再试。'), status=503)


@require_GET
def movies(request):
    query = request.GET.get('q', '').strip()
    if len(query) > 100:
        return render(request, 'research_recommendation/error.html', context(request,
            heading='检索内容过长', message='请将检索词控制在100个字符以内。'), status=400)
    try:
        page, cards = PageCatalogue().browse(query, request.GET.get('page', '1'))
        return render(request, 'research_recommendation/movies.html', context(request, movies=cards, page=page, query=query))
    except Exception:
        return render(request, 'research_recommendation/error.html', context(request,
            heading='影片库暂不可用', message='未返回替代电影数据。'), status=503)


@never_cache
@require_GET
def detail(request, movie_id):
    try:
        movie = PageCatalogue().get(movie_id)
    except Exception:
        return render(request, 'research_recommendation/error.html', context(request,
            heading='电影详情暂不可用', message='请稍后再试。'), status=503)
    if movie is None:
        raise Http404('电影不存在')
    from .multimodal_details import details_for
    from .page_catalogue import safe_poster
    extended = details_for(movie)
    trace_context = None
    selected_trace = request.GET.get('trace_id') or request.GET.get('personal_trace')
    if selected_trace:
        if not request.user.is_authenticated:
            return redirect(reverse('research_pages:login') + '?' + urlencode({'next': request.get_full_path()}))
        from .trace_pages import detail_trace_context
        from .contracts import RecommendationError
        try:
            trace_context = detail_trace_context(request.user, selected_trace, movie_id)
        except RecommendationError as exc:
            return render(request, 'research_recommendation/error.html', context(request,
                heading='推荐来源不可用', message='当前账号、研究画像授权或电影与推荐记录的关联未通过核验。'), status=exc.status)
    trace = None
    if request.user.is_authenticated:
        from .personal_views import owned_trace
        from .personal import catalogue_label
        from .models import MovieInteraction
        trace = owned_trace(request, movie_id)
        MovieInteraction.objects.create(user=request.user, movie_id=movie_id,
            catalogue_model=catalogue_label(), kind='view', trace=trace)
    return render(request, 'research_recommendation/detail.html', context(request, movie=movie, personal_trace=trace,
        extended=extended, trace_context=trace_context,
        detail_poster_url=None if movie.is_sensitive else (safe_poster(extended.get('local_poster_url')) or movie.poster_url or safe_poster(extended.get('poster_url')))))


@require_GET
def recommend(request):
    if not request.user.is_authenticated:
        return redirect(reverse('research_pages:login') + '?' + urlencode({'next': request.path}))
    result = api.profiles(request)
    payload = json.loads(result.content)
    profiles = payload.get('profiles', []) if result.status_code == 200 else []
    return render(request, 'research_recommendation/recommend.html', context(request,
        profiles=profiles, profile_error=None if result.status_code == 200 else '研究画像服务暂不可用。'),
        status=200 if result.status_code == 200 else result.status_code)


@require_GET
def card_metadata(request):
    # Public movie metadata only, bounded to one Top-K response. No model imports.
    parts = request.GET.get('ids', '').split(',')
    if not 1 <= len(parts) <= 100 or any(not p.isdecimal() or not 1 <= len(p) <= 18 for p in parts):
        return JsonResponse({'error': 'INVALID_MOVIE_IDS'}, status=400)
    try:
        cards = PageCatalogue().get_many([int(x) for x in parts])
        return JsonResponse({'movies': {str(k): {'genres': list(v.genres), 'poster_url': v.poster_url,
            'detail_url': reverse('research_pages:detail', args=[k])} for k, v in cards.items()}})
    except Exception:
        return JsonResponse({'error': 'MOVIE_METADATA_UNAVAILABLE'}, status=503)


@require_GET
def historical_trace(request):
    if not request.user.is_authenticated:
        return redirect(reverse('research_pages:login'))
    # Read-only opt-in metadata, separate from the recommendation endpoint.
    configured = getattr(settings, 'TASK60_HISTORICAL_TRACE', None)
    if not configured:
        raise Http404('未配置历史验收记录')
    path, expected = configured
    try:
        with Path(path).open('rb') as stream:
            data = stream.read(100_001)
        if len(data) > 100_000 or hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('HISTORICAL_IDENTITY_MISMATCH')
        payload = json.loads(data)
        if not isinstance(payload, dict) or not isinstance(payload.get('recommendations'), list):
            raise ValueError('HISTORICAL_SCHEMA_MISMATCH')
    except (OSError, ValueError, UnicodeError):
        return render(request, 'research_recommendation/error.html', context(request,
            heading='历史证据暂不可用', message='文件缺失、格式或身份未通过核验；未显示替代记录。'), status=503)
    # A historical demo record is not a bypass around current research grants.
    from .models import ResearchProfile
    allowed = getattr(settings, 'TASK60_HISTORY_PROFILE', None)
    profile = None if allowed is None else ResearchProfile.objects.filter(dataset=allowed[0],
        canonical_user_id=allowed[1], is_active=True, grants__user=request.user, grants__is_active=True).first()
    if profile is None:
        return render(request, 'research_recommendation/error.html', context(request,
            heading='研究画像权限不足（403）', message='当前账号无权查看该研究画像的历史演示记录。'), status=403)
    try:
        api.service().verify_profile(profile)
    except Exception:
        return render(request, 'research_recommendation/error.html', context(request,
            heading='研究画像暂不可用', message='历史记录未通过当前画像验证。'), status=503)
    # Historical profile IDs must not be offered as current grants or API inputs.
    return render(request, 'research_recommendation/history.html', context(request, historical=payload))


class PageLogin(LoginView):
    template_name = 'research_recommendation/login.html'
    redirect_authenticated_user = True

    def get_default_redirect_url(self):
        if getattr(settings, 'TASK62_NATIVE_AUTH', True):
            return reverse('research_pages:personal')
        return super().get_default_redirect_url()

    def dispatch(self, request, *args, **kwargs):
        if not getattr(settings, 'TASK62_NATIVE_AUTH', True) and getattr(settings, 'TASK60_USE_EXISTING_AUTH', True):
            try:
                original = reverse('login_user')
            except NoReverseMatch:
                original = None
            if original:
                target = request.GET.get('next', reverse('research_pages:home'))
                if not url_has_allowed_host_and_scheme(target, {request.get_host()}, require_https=request.is_secure()):
                    target = reverse('research_pages:home')
                return redirect(original + '?' + urlencode({'next': target}))
        return super().dispatch(request, *args, **kwargs)

    def get_context_data(self, **kwargs):
        return context(self.request, **super().get_context_data(**kwargs))


class PageLogout(LogoutView):
    next_page = 'research_pages:home'


@require_GET
def assistant(request):
    return render(request, 'research_recommendation/error.html', context(request,
        heading='智能助手入口', message='原系统的Agent功能保持原状。本轮独立页面演示未启用Agent，推荐功能请进入研究推荐。'))
