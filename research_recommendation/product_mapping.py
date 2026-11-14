"""Conservative, namespace-scoped identity corroboration; no fuzzy matching."""
import re
from django.conf import settings
MAPPING_VERSION = getattr(settings,'PAMK_MAPPING_VERSION','ch5-exact-year-v1')
TMDB_SOURCE_SHA256='2e3a0355825632822af803f95c61c7cb0e5592a858cf5447fbc0f4ad5b81e167'
def normalized_title(value):
    return re.sub(r'\s*\([0-9]{4}\)\s*$', '', value).strip().casefold()

def business_identity_verified(bridge,research,movie):
    if bridge.match_method!='EXACT_TMDB_CORROBORATED':
        return str(movie.movielens_id)==research.source_movie_id and normalized_title(movie.title)==normalized_title(research.title)
    evidence=bridge.evidence
    if evidence.get('metadata_source_sha256')!=TMDB_SOURCE_SHA256:return False
    if str(movie.imdb_id)!=evidence.get('tmdb_id'):return False
    if movie.movielens_id and str(movie.movielens_id)!=research.source_movie_id:return False
    titles={normalized_title(t) for t in (evidence.get('tmdb_title'),evidence.get('tmdb_original_title')) if t}
    try:year=str(int(float(research.metadata.get('year') or 0)))
    except (ValueError,TypeError):return False
    if year!=evidence.get('source_year') or not evidence.get('tmdb_release_date','').startswith(year+'-') or normalized_title(research.title) not in titles:return False
    if movie.date and movie.date.year!=int(year):return False
    poster=evidence.get('poster_filename')
    return bool((poster and poster in {str(movie.poster or '').split('/')[-1],str(movie.poster_file or '').split('/')[-1]}) or normalized_title(movie.title) in titles)
