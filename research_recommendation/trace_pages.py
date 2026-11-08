"""Read-only presentation of existing owner-bound recommendation facts."""
from urllib.parse import urlencode
from django.http import Http404
from django.shortcuts import redirect,render
from django.urls import reverse
from django.views.decorators.http import require_GET
from django.views.decorators.cache import never_cache
from .contracts import RecommendationError
from .explanation import build_context,archived_results,ranked_results,sha_json
from .models import RecommendationTrace
from .services import ProfileService
from .page_catalogue import PageCatalogue,safe_poster
from .multimodal_details import details_for
from .pamk_online import PAMK_SHA

FULL_SHA='73c7d95868005dae1a461d367a5afa0d3550c1a8379a9125764c4334dab2c7d6'
VERIFIED_CHECKPOINTS={'UKMAP':FULL_SHA,'PAMK-Net':PAMK_SHA}

def detail_trace_context(user,trace_id,movie_id):
    data=build_context(user,trace_id,movie_id).as_data()
    data['original_score_text']=repr(data['original_score'])
    if data['source_type']=='RESEARCH_MODEL_RESULT':
        trace=RecommendationTrace.objects.get(pk=data['trace_id'],requester=user)
        data.update(recorded_at=trace.created_at,source_reference=trace.provenance.get('source_reference'),
            inference_mode=trace.provenance.get('inference_mode'),
            record_url=reverse('research_pages:recommendation_record',args=[trace.pk]),
            json_url=reverse('research_recommendation:trace',args=[trace.pk]))
    return data

@never_cache
@require_GET
def recommendation_record(request,trace_id):
    from .page_views import context
    if not request.user.is_authenticated:
        return redirect(reverse('research_pages:login')+'?'+urlencode({'next':request.get_full_path()}))
    trace=RecommendationTrace.objects.filter(pk=trace_id,requester=request.user).first()
    if trace is None:raise Http404('推荐记录不可用')
    try:
        p=trace.provenance
        if (trace.outcome!='SUCCESS' or p.get('model_name') not in VERIFIED_CHECKPOINTS
                or p.get('checkpoint_identity')!=VERIFIED_CHECKPOINTS.get(p.get('model_name'))
                or p.get('inference_mode') not in ('REAL_ONLINE_INFERENCE','OFFLINE_MODEL_INFERENCE','REALTIME_MODEL_INFERENCE')):
            raise RecommendationError('VERIFIED_MODEL_TRACE_REQUIRED',409)
        profile=ProfileService.authorized(request.user,p['profile_source']['profile_id'])
        snapshot=p.get('explanation_results')
        if snapshot:
            if snapshot.get('sha256')!=sha_json(snapshot.get('value')):raise RecommendationError('TRACE_EVIDENCE_MISMATCH',409)
            rows=ranked_results(snapshot['value']['recommendations'])
        else:rows=archived_results(trace,profile)[0]
        # Reuse the strict existing context builder for current grants and each linked identity.
        items=[]
        for row in rows:
            item={**row,'original_score_text':repr(row['score'])}
            if row.get('mapping_status')=='VERIFIED_UNIQUE':
                data=build_context(request.user,trace.pk,row['django_movie_id']).as_data()
                card=PageCatalogue().get(row['django_movie_id']);extra=details_for(card)
                item.update(poster_url=None if card.is_sensitive else (safe_poster(extra.get('local_poster_url')) or card.poster_url),
                    genres=card.genres,detail_url=reverse('research_pages:detail',args=[card.id])+'?'+urlencode({'trace_id':trace.pk}))
            items.append(item)
        if not items:raise RecommendationError('TRACE_ITEM_RESULTS_UNAVAILABLE',409)
    except RecommendationError as exc:
        return render(request,'research_recommendation/error.html',context(request,heading='推荐记录不可用',
            message='当前授权、原始结果或电影身份未通过核验，未生成替代推荐。'),status=exc.status)
    except (KeyError,TypeError,ValueError):
        return render(request,'research_recommendation/error.html',context(request,heading='推荐记录不可用',
            message='记录不完整，未生成替代推荐。'),status=409)
    return render(request,'research_recommendation/record.html',context(request,items=items,
        recommendation_trace=trace,provenance=p,original_count=len(rows),
        display_model='UKMAP Full' if p['model_name']=='UKMAP' else 'PAMK-Net'))
