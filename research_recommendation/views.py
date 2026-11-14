"""Authenticated JSON API. No views/Agent import and no automatic user mapping."""
import json
import logging
import time
import uuid
from django.conf import settings
from django.core.exceptions import RequestDataTooBig
from django.db import DatabaseError
from django.http import JsonResponse
from django.utils.module_loading import import_string
from django.views.decorators.csrf import csrf_protect

from .contracts import RecommendRequest, RecommendationError
from .manifests import unique_object
from .models import RecommendationTrace, ResearchProfile
from .services import unconfigured_service

logger = logging.getLogger('movie_agent.research_recommendation')


def response(payload, request_id, status=200):
    result = JsonResponse(dict(payload, request_id=request_id, trace_id=payload.get('trace_id', request_id)), status=status,
                          json_dumps_params={'allow_nan': False})
    result['X-Request-ID'] = request_id
    result['Cache-Control'] = 'no-store'
    return result


def service():
    # Operator-owned factory only. No dotted path from a request is accepted.
    factory = getattr(settings, 'TASK57_SERVICE_FACTORY', None)
    return import_string(factory)() if factory else unconfigured_service()


def guard(request, method, rid):
    if not request.user.is_authenticated:
        return response({'error': 'AUTHENTICATION_REQUIRED'}, rid, 401)
    if request.method != method:
        result = response({'error': 'METHOD_NOT_ALLOWED'}, rid, 405)
        result['Allow'] = method
        return result
    return None


def csrf_failure(request, reason=''):
    # No reason/body/cookie/session logged, and no unauthenticated SQL writes.
    return response({'error': 'CSRF_FAILED'}, str(uuid.uuid4()), 403)


@csrf_protect
def recommend(request):
    rid = str(uuid.uuid4())
    denied = guard(request, 'POST', rid)
    if denied is not None:
        return denied
    started = time.monotonic()
    provenance, outcome, status = {}, 'INTERNAL_ERROR', 500
    try:
        if request.content_type != 'application/json':
            raise RecommendationError('JSON_CONTENT_TYPE_REQUIRED', 415)
        raw = request.body
        if len(raw) > 8192:
            raise RecommendationError('REQUEST_TOO_LARGE', 413)
        try:
            data = json.loads(raw, object_pairs_hook=unique_object)
        except (ValueError, UnicodeError):
            raise RecommendationError('INVALID_JSON') from None
        parsed = RecommendRequest.parse(data)
        payload = service().recommend(request.user, parsed)
        provenance = {k: payload[k] for k in ('model_name', 'dataset', 'checkpoint_identity',
                      'profile_source', 'inference_mode', 'mapping_version', 'candidate_scope',
                      'source_reference', 'output_artifact_identity', 'mapping_summary',
                      'catalogue_manifest_identity')}
        provenance['recommendation_count'] = len(payload['recommendations'])
        from .explanation import result_snapshot
        provenance['explanation_results'] = result_snapshot(payload)
        outcome, status = 'SUCCESS', 200
    except RequestDataTooBig:
        outcome, status, payload = 'REQUEST_TOO_LARGE', 413, {'error': 'REQUEST_TOO_LARGE'}
    except RecommendationError as exc:
        outcome, status, payload = exc.code, exc.status, {'error': exc.code}
    except DatabaseError:
        outcome, status, payload = 'RECOMMENDATION_STORAGE_UNAVAILABLE', 503, {'error': 'RECOMMENDATION_STORAGE_UNAVAILABLE'}
    except Exception:
        # Never log the exception text: it can contain paths, SQL or credentials.
        outcome, status, payload = 'RECOMMENDATION_DEPENDENCY_FAILED', 503, {'error': 'RECOMMENDATION_DEPENDENCY_FAILED'}
    try:
        RecommendationTrace.objects.create(id=rid, requester=request.user, outcome=outcome,
                                           provenance=provenance)
    except DatabaseError:
        outcome, status, payload = 'TRACE_PERSISTENCE_FAILED', 503, {'error': 'TRACE_PERSISTENCE_FAILED'}
    try:
        logger.info('research_recommendation request_id=%s outcome=%s elapsed_ms=%.3f',
                    rid, outcome, max(0, time.monotonic()-started)*1000)
    except Exception:
        pass
    return response(payload, rid, status)


def profiles(request):
    rid = str(uuid.uuid4())
    denied = guard(request, 'GET', rid)
    if denied is not None:
        return denied
    try:
        result = []
        app = service()
        for profile in ResearchProfile.objects.filter(is_active=True, grants__user=request.user,
                                                       grants__is_active=True).order_by('id')[:101]:
            if len(result) >= 100:
                raise RecommendationError('PROFILE_LIST_LIMIT', 409)
            app.verify_profile(profile)
            result.append({'id': str(profile.id), 'dataset': profile.dataset,
                           'canonical_user_id': profile.canonical_user_id,
                           'display_name': profile.display_name,
                           'profile_source': profile.profile_source,
                           'identity_mode': 'RESEARCH_DATASET_PROFILE', 'demo_only': True})
        return response({'profiles': result}, rid)
    except RecommendationError as exc:
        return response({'error': exc.code}, rid, exc.status)
    except Exception:
        return response({'error': 'PROFILE_DEPENDENCY_UNAVAILABLE'}, rid, 503)


def trace_detail(request, trace_id):
    rid = str(uuid.uuid4())
    denied = guard(request, 'GET', rid)
    if denied is not None:
        return denied
    try:
        trace = RecommendationTrace.objects.filter(pk=trace_id, requester=request.user).first()
        if trace is None:
            return response({'error': 'TRACE_NOT_AVAILABLE'}, rid, 404)
        return response({'trace_id': str(trace.id), 'outcome': trace.outcome,
                         'provenance': trace.provenance}, rid)
    except DatabaseError:
        return response({'error': 'TRACE_STORAGE_UNAVAILABLE'}, rid, 503)
