"""Canonical research identities plus unmatched business identities, never guessed aliases."""
from dataclasses import dataclass
from collections import Counter
import math,re
from django.conf import settings
from django.db.models import Q
from myapp.models import Movie
from .models import ResearchMovie,MovieIDBridge
from .contracts import RecommendationError,M3L
from .page_catalogue import PageCatalogue
from .product_mapping import MAPPING_VERSION

@dataclass
class Film:
    identity:str
    research:object
    business:object
    card:dict

def parse_identity(value):
    if not isinstance(value,str) or not re.fullmatch(r'(m3l|business):(0|[1-9][0-9]{0,9})',value):raise RecommendationError('INVALID_MOVIE_IDENTITY')
    ns,n=value.split(':');return ns,int(n)

def _metadata(rm):
    m=rm.metadata;g=m.get('genres',[])
    if isinstance(g,str):g=[x.strip() for x in re.split(r'[|,]',g) if x.strip()]
    if not isinstance(g,list):g=[]
    genres=[x for x in g if isinstance(x,str) and len(x)<=100 and x!='(no genres listed)'][:12]
    try:y=int(float(m.get('year')));y=y if 1880<=y<=2100 else None
    except (ValueError,TypeError,OverflowError):y=None
    return {'year':y,'genres':genres,'summary':''}

def _business(movie):
    c=PageCatalogue.card(movie)
    return dict(movie_title=movie.title,poster_url=c.poster_url,summary=c.summary[:1200],genres=list(c.genres),year=movie.date.year if movie.date else None,business_rating=float(movie.score) if movie.score is not None else None,vote_count=movie.vote_count)

def projected_research(rows):
    from .product_service import dataset,map_items
    rows=list(rows)
    if not rows:return []
    d=dataset();mapped=map_items([{'canonical_movie_id':x.canonical_movie_id} for x in rows],d)
    # A business entity cannot silently represent multiple distinct canonical IDs.
    bids=[x['django_movie_id'] for x in mapped if x['django_movie_id'] is not None]
    inverse=Counter(MovieIDBridge.objects.filter(source_dataset=M3L,mapping_version=MAPPING_VERSION,mapping_status='VERIFIED_UNIQUE',django_movie_id__in=bids).values_list('django_movie_id',flat=True))
    movies={m.pk:m for m in Movie.objects.filter(pk__in=bids).prefetch_related('genres')}
    out=[]
    for rm,x in zip(rows,mapped):
        bid=x['django_movie_id'];movie=movies.get(bid) if bid is not None and inverse[bid]==1 else None
        if bid is not None and movie is None:x.update(django_movie_id=None,detail_url=None,mapping_status='AMBIGUOUS')
        card={**x,**_metadata(rm),'identity':f'm3l:{rm.canonical_movie_id}','research_title':rm.title,'poster_url':None,'business_rating':None,'vote_count':0,'detail_url':f'/research/movies/{rm.canonical_movie_id}/'}
        card.pop('rank',None)  # Catalogue position is not a model ranking score.
        if movie:card.update(_business(movie),detail_url=f'/movie/{movie.pk}/')
        out.append(Film(card['identity'],rm,movie,card))
    return out

def resolve(identity):
    ns,n=parse_identity(identity)
    if ns=='m3l':
        rm=ResearchMovie.objects.filter(dataset__protocol=M3L,dataset__complete=True,canonical_movie_id=n).first()
        if rm is None:raise RecommendationError('MOVIE_NOT_FOUND',404)
        return projected_research([rm])[0]
    movie=Movie.objects.prefetch_related('genres').filter(pk=n).first()
    if movie is None:raise RecommendationError('MOVIE_NOT_FOUND',404)
    rows=list(MovieIDBridge.objects.filter(source_dataset=M3L,mapping_version=MAPPING_VERSION,mapping_status='VERIFIED_UNIQUE',django_movie_id=n)[:2])
    if len(rows)==1:
        rm=ResearchMovie.objects.filter(dataset__protocol=M3L,dataset__complete=True,canonical_movie_id=rows[0].canonical_movie_id).first()
        if rm:
            film=projected_research([rm])[0]
            if film.business is not None and film.business.pk==n:return film
    return Film(f'business:{n}',None,movie,dict(_business(movie),identity=f'business:{n}',canonical_movie_id=None,django_movie_id=n,mapping_status='BUSINESS_CATALOGUE',detail_url=f'/movie/{n}/',research_detail_url=None))

def catalogue():
    films=projected_research(ResearchMovie.objects.filter(dataset__protocol=M3L,dataset__complete=True).order_by('canonical_movie_id'))
    aliased={f.business.pk for f in films if f.business is not None}
    extra=Movie.objects.exclude(pk__in=aliased).prefetch_related('genres').order_by('pk')
    cards=[f.card for f in films]
    cards.extend(dict(_business(m),identity=f'business:{m.pk}',canonical_movie_id=None,django_movie_id=m.pk,mapping_status='BUSINESS_CATALOGUE',detail_url=f'/movie/{m.pk}/',research_detail_url=None) for m in extra)
    return cards

def search_movies(query='',genre='',year_from=None,year_to=None,sort='score',page=1,page_size=20,scope='all'):
    if not isinstance(query,str) or len(query)>200 or not isinstance(genre,str) or len(genre)>100:raise RecommendationError('INVALID_SEARCH')
    if type(page) is not int or not 1<=page<=100000 or type(page_size) is not int or not 1<=page_size<=50:raise RecommendationError('INVALID_PAGINATION')
    for y in [year_from,year_to]:
        if y is not None and (type(y) is not int or not 1880<=y<=2100):raise RecommendationError('INVALID_YEAR')
    if year_from is not None and year_to is not None and year_from>year_to:raise RecommendationError('INVALID_YEAR_RANGE')
    if sort not in ('score','popular','recent','canonical') or scope not in ('all','research','business'):raise RecommendationError('INVALID_SEARCH_FILTER')
    aliases={'sci-fi':'科幻','science fiction':'科幻','science-fiction':'科幻','animation':'动画','drama':'剧情','comedy':'喜剧','thriller':'惊悚','horror':'恐怖','romance':'爱情','action':'动作','adventure':'冒险','documentary':'纪录'}
    rows=catalogue();q=query.strip().casefold();g=aliases.get(genre.casefold(),genre.casefold())
    rows=[x for x in rows if (scope=='all' or (scope=='research')==(x['canonical_movie_id'] is not None)) and (not q or q in x['movie_title'].casefold() or q in x.get('research_title','').casefold()) and (not g or any(g in aliases.get(y.casefold(),y.casefold()) for y in x['genres'])) and (year_from is None or x['year'] is not None and x['year']>=year_from) and (year_to is None or x['year'] is not None and x['year']<=year_to)]
    if sort=='canonical':rows.sort(key=lambda x:(x['canonical_movie_id'] is None,x['canonical_movie_id'] if x['canonical_movie_id'] is not None else x['django_movie_id']))
    else:rows.sort(key=lambda x:({'score':x['business_rating'] or 0,'popular':x['vote_count'],'recent':x['year'] or 0}[sort],x['vote_count'],x['identity']),reverse=True)
    total=len(rows);pages=max(1,math.ceil(total/page_size));page=min(page,pages)
    return {'items':rows[(page-1)*page_size:page*page_size],'total':total,'page':page,'pages':pages,'page_size':page_size,'scope':scope,'catalogue_protocol':M3L,'mapping_version':MAPPING_VERSION}

def resolve_title(query):
    if not isinstance(query,str) or not query.strip() or len(query)>200:raise RecommendationError('INVALID_MOVIE_QUERY')
    def norm(v):return re.sub(r'\s*\([0-9]{4}\)\s*$','',v).strip().casefold()
    result=search_movies(query=query,page_size=50);rows=result['items'];exact=[x for x in rows if norm(query) in (norm(x['movie_title']),norm(x.get('research_title','')))];matches=exact or rows
    if len(matches)!=1 or not exact and result['total']!=1:return None,matches
    return resolve(matches[0]['identity']),matches
