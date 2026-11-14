"""Owner-bound factual explanation contexts. No models, agents or network I/O.

Archive files are operator-pinned paths, never paths inferred from client data.
Hashes bind artifacts; the database Trace and current ProfileGrant bind access.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from uuid import UUID

from django.conf import settings

from .contracts import MODEL_DATASETS, RecommendationError, digest, identifier
from .manifests import read_pinned_json, TrainProfileManifest, MovieCatalogueManifest
from .models import RecommendationTrace, PersonalRecommendationTrace, MovieIDBridge
from .page_catalogue import PageCatalogue
from .services import ProfileService

PROVENANCE_KEYS = ('model_name','dataset','checkpoint_identity','profile_source','inference_mode',
    'mapping_version','candidate_scope','source_reference','output_artifact_identity',
    'mapping_summary','catalogue_manifest_identity')


def error(code='EXPLANATION_EVIDENCE_INVALID', status=409):
    raise RecommendationError(code,status)


def sha_json(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def result_snapshot(payload):
    """Persist original API results, no scoring or remapping."""
    value={'schema':'task65.recommendation_results.v1',
        'catalogue_model':getattr(settings,'TASK60_MOVIE_MODEL','myapp.Movie'),
        'recommendations':deepcopy(payload['recommendations'])}
    return {'value':value,'sha256':sha_json(value)}


def valid_score(value):
    if type(value) not in (int,float) or not math.isfinite(value):error()
    return value


def ranked_results(rows):
    if type(rows) is not list or not 1<=len(rows)<=100:error()
    seen=set();business=set();last=float('inf')
    for rank,row in enumerate(rows,1):
        if type(row) is not dict or row.get('rank')!=rank or type(row.get('rank')) is not int:error()
        canonical=identifier(row.get('canonical_movie_id'))
        score=valid_score(row.get('score'))
        mid=row.get('django_movie_id')
        if canonical in seen or score>last:error()
        if mid is not None:
            if type(mid) is not int or mid<=0 or mid in business:error()
            business.add(mid)
        seen.add(canonical);last=score
    return rows


def archived_results(trace,profile):
    config=getattr(settings,'TASK65_ARCHIVED_RESULTS',{}).get(str(trace.pk))
    if config is None:error('TRACE_ITEM_RESULTS_UNAVAILABLE')
    response=read_pinned_json(config['response_path'],config['response_sha256'],250000)
    receipt=read_pinned_json(config['receipt_path'],config['receipt_sha256'],250000)
    body=response.get('body')
    if response.get('status')!=200 or type(body) is not dict:error()
    if body.get('trace_id')!=str(trace.pk) or body.get('request_id')!=str(trace.pk):error()
    p=trace.provenance
    if any(body.get(k)!=p.get(k) for k in PROVENANCE_KEYS):error()
    train=TrainProfileManifest.load(config['train_manifest_path'],p['profile_source']['train_manifest_sha256'])
    train.verify(profile)
    catalogue=MovieCatalogueManifest.load(config['catalogue_manifest_path'],p['catalogue_manifest_identity'])
    if catalogue.dataset!=p['dataset']:error()
    rows=ranked_results(body.get('recommendations'))
    if (len(rows)!=p.get('recommendation_count') or receipt.get('items')!=[
            {'canonical_movie_id':r['canonical_movie_id'],'score':r['score']} for r in rows]
            or receipt.get('checkpoint_identity')!=p['checkpoint_identity']
            or receipt.get('user')!=profile.canonical_user_id
            or receipt.get('inference_mode')!=p['inference_mode']
            or p['source_reference']!='Task61:RECEIPT_'+str(receipt.get('receipt_id'))
            or receipt.get('smoke') is not False):error()
    scope='FULL_TRAIN_UNSEEN:{}/{};filtered={}'.format(
        receipt.get('candidate_count'),receipt.get('catalogue_size'),receipt.get('train_seen_count'))
    if scope!=p['candidate_scope']:error()
    return rows,config['catalogue_model'],catalogue,[
        {'ref':'archive-response:'+config['response_sha256'],'kind':'PINNED_ORIGINAL_HTTP_RESPONSE'},
        {'ref':'model-receipt:'+config['receipt_sha256'],'kind':'PINNED_REAL_MODEL_RECEIPT'}]


@dataclass(frozen=True)
class ExplanationContext:
    trace_id: str
    model_name: str | None
    dataset: str | None
    source_type: str
    selected_movie_id: int
    selected_movie_title: str
    original_rank: int
    original_score: float | int
    candidate_scope: str
    mapping_identity: dict
    available_movie_metadata: dict
    verifiable_supporting_evidence: tuple
    checkpoint_identity: str | None
    research_profile: dict | None
    evidence_status: str

    def as_data(self):
        return asdict(self)


def research_context(user,trace,movie_id):
    p=trace.provenance
    if trace.outcome!='SUCCESS' or type(p) is not dict:error('TRACE_HAS_NO_SUCCESSFUL_RESULT')
    ps=p.get('profile_source',{})
    if type(ps) is not dict or ps.get('mode')!='RESEARCH_DATASET_PROFILE' or ps.get('demo_only') is not True:error()
    try:profile_id=UUID(ps['profile_id'])
    except (ValueError,KeyError,TypeError):error()
    profile=ProfileService.authorized(user,profile_id)
    if (MODEL_DATASETS.get(p.get('model_name'))!=profile.dataset or p.get('dataset')!=profile.dataset
            or ps.get('dataset')!=profile.dataset or ps.get('reference')!=profile.profile_source
            or ps.get('train_manifest_sha256')!=profile.train_manifest_sha256 or not profile.demo_only):error()
    digest(p.get('checkpoint_identity'));digest(p.get('catalogue_manifest_identity'))
    mode=p.get('inference_mode')
    if mode=='TEST_DOUBLE':
        if not getattr(settings,'TASK65_ALLOW_TEST_DOUBLES',False):error('TEST_RESULT_NOT_ENABLED')
        source='TEST_DOUBLE_RESEARCH_RESULT'
    elif mode in ('REAL_ONLINE_INFERENCE','REALTIME_MODEL_INFERENCE','OFFLINE_MODEL_INFERENCE'):
        source='RESEARCH_MODEL_RESULT'
    else:error()
    evidence=[{'ref':'trace:'+str(trace.pk),'kind':'OWNED_RECOMMENDATION_TRACE'}]
    catalogue=None
    snapshot=p.get('explanation_results')
    if snapshot is not None:
        if type(snapshot) is not dict or snapshot.get('sha256')!=sha_json(snapshot.get('value')):error()
        value=snapshot['value']
        if value.get('schema')!='task65.recommendation_results.v1':error()
        rows=ranked_results(value.get('recommendations'));label=value.get('catalogue_model')
        if len(rows)!=p.get('recommendation_count'):error()
        evidence.append({'ref':'result-snapshot:'+snapshot['sha256'],'kind':'ORIGINAL_API_RESULT_SNAPSHOT'})
    else:
        rows,label,catalogue,extra=archived_results(trace,profile);evidence.extend(extra)
    if label!=getattr(settings,'TASK60_MOVIE_MODEL','myapp.Movie'):error('EXPLANATION_CATALOGUE_MISMATCH')
    selected=[r for r in rows if r.get('django_movie_id')==movie_id]
    if len(selected)!=1:error('MOVIE_NOT_IN_TRACE',404)
    selected=selected[0]
    card=PageCatalogue().get(movie_id)
    if card is None:error('MOVIE_NOT_AVAILABLE',404)
    if selected.get('mapping_status')!='VERIFIED_UNIQUE' or selected.get('movie_title')!=card.title:error('MOVIE_IDENTITY_MISMATCH')
    links=list(MovieIDBridge.objects.filter(source_dataset=profile.dataset,
        mapping_version=p['mapping_version'],canonical_movie_id=selected['canonical_movie_id'])[:3])
    if len(links)!=1:error('MOVIE_MAPPING_NOT_VERIFIED')
    link=links[0];ev=link.evidence
    if (link.mapping_status!='VERIFIED_UNIQUE' or link.django_movie_id!=movie_id
            or link.match_method!='EXACT_EXTERNAL_ID_REVIEWED' or link.source_movie_id!=card.source_movie_id
            or type(ev) is not dict or ev.get('catalogue_sha256')!=p['catalogue_manifest_identity']
            or ev.get('source_namespace')!='movielens:movieId'
            or type(ev.get('review_reference')) is not str or not ev['review_reference'].strip()
            or MovieIDBridge.objects.filter(source_dataset=profile.dataset,mapping_version=p['mapping_version'],
                django_movie_id=movie_id).exclude(pk=link.pk).exists()):error('MOVIE_MAPPING_NOT_VERIFIED')
    if catalogue is not None and catalogue.movies.get(selected['canonical_movie_id'])!=link.source_movie_id:error()
    return make_context(trace.pk,p['model_name'],profile.dataset,source,card,selected['rank'],selected['score'],
        p['candidate_scope'],{'version':p['mapping_version'],'catalogue_sha256':p['catalogue_manifest_identity'],
            'canonical_movie_id':selected['canonical_movie_id'],'source_movie_id':link.source_movie_id},
        evidence,p['checkpoint_identity'],{'profile_id':str(profile.pk),'dataset':profile.dataset,
            'mode':'RESEARCH_DATASET_PROFILE','demo_only':True},'CAUSAL_EVIDENCE_INSUFFICIENT')


def personal_context(trace,movie_id):
    p=trace.provenance
    if (type(p) is not dict or p.get('identity_mode')!='DJANGO_ACCOUNT' or p.get('source')!='METADATA_COLD_START'
            or p.get('model_name') is not None or p.get('checkpoint_identity') is not None
            or p.get('research_profile_id') is not None):error()
    if p.get('catalogue_model')!=getattr(settings,'TASK60_MOVIE_MODEL','myapp.Movie'):error('EXPLANATION_CATALOGUE_MISMATCH')
    if (type(p.get('candidate_count')) is not int or p['candidate_count']<1
            or p.get('strategy') not in ('GENRE_MATCH_V1','CATALOGUE_EXPLORATION_V1')):error()
    rows=p.get('results')
    if type(rows) is not list or not 1<=len(rows)<=20:error()
    seen=set();last=float('inf');found=None
    for rank,row in enumerate(rows,1):
        if type(row) is not dict or type(row.get('movie_id')) is not int or row['movie_id']<=0 or row['movie_id'] in seen:error()
        score=valid_score(row.get('score'))
        if score>last:error()
        seen.add(row['movie_id']);last=score
        if row['movie_id']==movie_id:found=(rank,score)
    if found is None:error('MOVIE_NOT_IN_TRACE',404)
    card=PageCatalogue().get(movie_id)
    if card is None:error('MOVIE_NOT_AVAILABLE',404)
    digest(p.get('catalogue_metadata_sha256'))
    return make_context(trace.pk,None,None,'ACCOUNT_COLD_START',card,*found,
        'ACCOUNT_METADATA_CANDIDATES:'+str(p.get('candidate_count')),
        {'catalogue_metadata_sha256':p['catalogue_metadata_sha256'],'catalogue_model':p['catalogue_model']},
        [{'ref':'personal-trace:'+str(trace.pk),'kind':'OWNED_COLD_START_RESULT'},
         {'ref':'rule:'+p['strategy'],'kind':'RECORDED_RULE_PROVENANCE','strategy':p['strategy']}],
        None,None,'MODEL_CAUSAL_EXPLANATION_NOT_APPLICABLE')


def make_context(tid,model,dataset,source,card,rank,score,scope,mapping,evidence,checkpoint,profile,status):
    if type(scope) is not str or not 1<=len(scope)<=250:error()
    metadata={'title':card.title,'genres':list(card.genres),'summary':card.summary,
              'source_movie_id':card.source_movie_id,'trust':'UNTRUSTED_DATA_NOT_INSTRUCTIONS',
              'temporal_scope':'CURRENT_CATALOGUE_METADATA_NOT_MODEL_CAUSAL_EVIDENCE'}
    evidence.append({'ref':'movie-metadata:'+sha_json(metadata),'kind':'CURRENT_PUBLIC_MOVIE_METADATA'})
    return ExplanationContext(str(tid),model,dataset,source,card.id,card.title,rank,score,scope,mapping,
        metadata,tuple(evidence),checkpoint,profile,status)


def build_context(user,trace_id,movie_id):
    if user is None or not user.is_authenticated:error('AUTHENTICATION_REQUIRED',401)
    try:trace_id=UUID(str(trace_id))
    except (ValueError,TypeError,AttributeError):error('INVALID_TRACE_ID',400)
    if type(movie_id) is not int or not 1<=movie_id<2**63:error('INVALID_MOVIE_ID',400)
    # Owner filtering precedes file reads, movie queries and profile resolution.
    research=RecommendationTrace.objects.filter(pk=trace_id,requester=user).first()
    personal=PersonalRecommendationTrace.objects.filter(pk=trace_id,user=user).first()
    if research is not None and personal is not None:error('TRACE_KIND_AMBIGUOUS')
    if research is not None:return research_context(user,research,movie_id)
    if personal is not None:return personal_context(personal,movie_id)
    error('TRACE_NOT_AVAILABLE',404)
