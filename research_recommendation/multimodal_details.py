"""Pinned public metadata projection. Never imported by the scoring worker."""
from functools import lru_cache
from pathlib import Path
import hashlib,json
from django.conf import settings

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_ASSET=(ROOT/'docs/task77/assets/movie_details.json',
    'd1792b0a6125c027d61f3db1b8c0ddd7779a84c0a90039562259cae3110bc87d')

@lru_cache(maxsize=2)
def _load(path,expected,mtime,size):
    if size>12*1024**2:raise ValueError('METADATA_SIZE_LIMIT')
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('METADATA_IDENTITY_MISMATCH')
    value=json.loads(data)
    if value.get('schema')!='task77.movie_details.v1' or not isinstance(value.get('movies'),dict):
        raise ValueError('METADATA_SCHEMA_MISMATCH')
    return value

def details_for(movie):
    missing={'status':'NOT_AVAILABLE','summary':'','actors':[],'directors':[],'kg_relations':[]}
    try:
        path,expected=getattr(settings,'TASK77_DETAIL_ASSET',DEFAULT_ASSET)
        path=Path(path);stat=path.stat()
        value=_load(str(path.resolve()),expected,stat.st_mtime_ns,stat.st_size)
        row=value['movies'].get(str(movie.id))
        if (not row or row.get('django_movie_id')!=movie.id or row.get('title')!=movie.title
                or row.get('source_movie_id')!=movie.source_movie_id):return missing
        return {**row,'status':'VERIFIED_PUBLIC_METADATA','sources':value['sources'],
            'detail_asset_sha256':expected}
    except (OSError,ValueError,TypeError,KeyError,UnicodeError):
        return {**missing,'status':'SOURCE_UNAVAILABLE'}
