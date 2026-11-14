"""Latest account-owned feedback. Aliases share one canonical row; legacy writes are mirrored."""
import math
from django.contrib.auth import get_user_model
from django.db import transaction
from myapp.models import UserRating,Collect
from .models import UnifiedMovieFeedback,MovieIDBridge
from .contracts import RecommendationError,M3L
from .product_mapping import MAPPING_VERSION
from .unified_catalogue import resolve

def feedback_for(user,film):
    key={'research_movie':film.research} if film.research else {'business_movie':film.business}
    return UnifiedMovieFeedback.objects.filter(user=user,**key).first()

def state(row):
    return {'rating':row.rating,'collected':row.collected,'preference':row.preference,'watched':row.watched,'updated_at':row.updated_at.isoformat()}

def positive(row):
    if row.preference=='dislike':return False
    if row.preference=='like':return True
    if row.rating is not None and row.rating<4:return False
    return row.collected or row.rating is not None and row.rating>=4

def has_interaction(row):return row.rating is not None or row.collected or bool(row.preference) or row.watched

@transaction.atomic
def write_feedback(user,identity,action,rating=None):
    if not user.is_authenticated:raise RecommendationError('AUTHENTICATION_REQUIRED',401)
    if action not in ('rating','collect','uncollect','like','dislike','clear_preference','watched'):raise RecommendationError('INVALID_INTERACTION')
    if action=='rating' and (type(rating) not in (int,float) or not math.isfinite(rating) or not .5<=rating<=5):raise RecommendationError('INVALID_SCORE')
    get_user_model().objects.select_for_update().get(pk=user.pk)
    film=resolve(identity);key={'research_movie':film.research} if film.research else {'business_movie':film.business}
    row,_=UnifiedMovieFeedback.objects.select_for_update().get_or_create(user=user,**key)
    if action=='rating':row.rating=float(rating);row.preference=''
    elif action in ('collect','uncollect'):row.collected=action=='collect'
    elif action=='watched':row.watched=True
    else:row.preference='' if action=='clear_preference' else action
    row.save()
    if film.business:
        if action=='rating':UserRating.objects.update_or_create(user=user,movie=film.business,defaults={'score':rating})
        elif action=='collect':Collect.objects.get_or_create(user=user,movie=film.business)
        elif action=='uncollect':Collect.objects.filter(user=user,movie=film.business).delete()
    return {'identity':film.identity,'movie_title':film.card['movie_title'],'state':state(row),'model_history_eligible':film.research is not None,'cache_invalidation':'CONTEXT_VERSION_CHANGED'}

def canonical_exclusions(user):
    return {r.research_movie.canonical_movie_id for r in UnifiedMovieFeedback.objects.filter(user=user,research_movie__isnull=False).select_related('research_movie') if has_interaction(r)}

def mapped_business_ids(ids):
    from .product_service import dataset,map_items
    rows=list(MovieIDBridge.objects.filter(source_dataset=M3L,mapping_version=MAPPING_VERSION,mapping_status='VERIFIED_UNIQUE',django_movie_id__in=ids))
    from collections import Counter
    counts=Counter(b.django_movie_id for b in rows)
    mapped=map_items([{'canonical_movie_id':x.canonical_movie_id} for x in rows],dataset()) if rows else []
    return {x['django_movie_id']:int(x['canonical_movie_id']) for x in mapped if x['mapping_status']=='VERIFIED_UNIQUE' and counts[x['django_movie_id']]==1}


def business_feedback_seen(user):
    from .unified_catalogue import projected_research
    rows=list(UnifiedMovieFeedback.objects.filter(user=user).select_related('research_movie'))
    direct={r.business_movie_id for r in rows if r.business_movie_id is not None and has_interaction(r)}
    research=[r.research_movie for r in rows if r.research_movie is not None and has_interaction(r)]
    if research:direct|={f.business.pk for f in projected_research(research) if f.business is not None}
    return direct
