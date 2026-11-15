"""Small JSON/CSRF endpoint. Client supplies selectors, never model provenance."""
import json
import uuid
from django.core.exceptions import RequestDataTooBig
from django.db import DatabaseError
from django.views.decorators.csrf import csrf_protect

from .contracts import RecommendationError
from .manifests import unique_object
from .views import guard,response


def explanation_tool():
    from myapp.services.trace_explanation import TraceExplanationTool
    return TraceExplanationTool()


@csrf_protect
def explain(request):
    rid=str(uuid.uuid4())
    denied=guard(request,'POST',rid)
    if denied is not None:return denied
    try:
        if request.content_type!='application/json':raise RecommendationError('JSON_CONTENT_TYPE_REQUIRED',415)
        raw=request.body
        if len(raw)>8192:raise RecommendationError('REQUEST_TOO_LARGE',413)
        try:data=json.loads(raw,object_pairs_hook=unique_object)
        except (ValueError,UnicodeError):raise RecommendationError('INVALID_JSON') from None
        if (type(data) is not dict or not {'trace_id','movie_id'}<=set(data)
                or set(data)-{'trace_id','movie_id','question'}):raise RecommendationError('INVALID_REQUEST_FIELDS')
        try:
            if type(data['trace_id']) is not str:raise ValueError
            tid=uuid.UUID(data['trace_id'])
        except (ValueError,TypeError,AttributeError):raise RecommendationError('INVALID_TRACE_ID') from None
        movie=data['movie_id'];question=data.get('question','')
        if type(movie) is not int or not 1<=movie<2**63:raise RecommendationError('INVALID_MOVIE_ID')
        if type(question) is not str or len(question)>1000:raise RecommendationError('INVALID_QUESTION')
        value=explanation_tool().execute(user=request.user,trace_id=tid,movie_id=movie,question=question)
        return response(value,rid)
    except RequestDataTooBig:return response({'error':'REQUEST_TOO_LARGE'},rid,413)
    except RecommendationError as exc:return response({'error':exc.code},rid,exc.status)
    except DatabaseError:return response({'error':'EXPLANATION_STORAGE_UNAVAILABLE'},rid,503)
    except Exception:return response({'error':'EXPLANATION_AGENT_UNAVAILABLE'},rid,503)
