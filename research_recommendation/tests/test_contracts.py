from dataclasses import replace
import ast
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from django.test import SimpleTestCase, override_settings

from research_recommendation.contracts import (
    ML1M, M3L, InferenceBatch, ModelSpec, Prediction, RecommendRequest,
    RecommendationError, SKBHIFAdapter, UKMAPAdapter, identifier,
)
from research_recommendation.manifests import TrainProfileManifest, MovieCatalogueManifest
from research_recommendation.offline import OfflineOutputBackend
from research_recommendation.test_support.fixtures import SPECS, TRAINS, SyntheticBackend


class ContractTests(SimpleTestCase):
    def request(self, **kw):
        v={'model_name':'UKMAP','profile_id':'00000000-0000-0000-0000-000000000001','top_k':10}
        v.update(kw)
        return RecommendRequest.parse(v)

    def test_valid_request(self):
        self.assertEqual(self.request().top_k,10)

    def test_top_k_boundaries(self):
        for k in [1,100]:
            with self.subTest(k=k):self.assertEqual(self.request(top_k=k).top_k,k)

    def test_invalid_top_k(self):
        for k in [0,101,True,1.5,'10',None]:
            with self.subTest(k=k),self.assertRaises(RecommendationError):self.request(top_k=k)

    def test_client_cannot_supply_user_or_dataset(self):
        for key in ['canonical_user_id','dataset','checkpoint','history','backend']:
            with self.subTest(key=key),self.assertRaises(RecommendationError):self.request(**{key:'0'})

    def test_unknown_model(self):
        for model in ['MMAN','UKMAP-v2',False,[]]:
            with self.subTest(model=model),self.assertRaises(RecommendationError):self.request(model_name=model)

    def test_profile_id_is_not_business_user_pk(self):
        for pid in [1,'1',None,{},'../profiles']:
            with self.subTest(pid=pid),self.assertRaises(RecommendationError):self.request(profile_id=pid)

    def test_zero_id_is_legal(self):
        self.assertEqual(identifier(0),'0')
        self.assertEqual(Prediction('0',0.0).canonical_movie_id,'0')

    def test_no_id_coercion(self):
        for v in [True,-1,1.0,'01',' 1','1.0','-1','1'*30]:
            with self.subTest(v=v),self.assertRaises(RecommendationError):identifier(v)

    def test_nonfinite_scores_rejected(self):
        for score in [float('nan'),float('inf'),-float('inf'),True,'1']:
            with self.subTest(score=score),self.assertRaises(RecommendationError):Prediction('0',score)

    def test_model_dataset_is_fixed(self):
        with self.assertRaises(RecommendationError):ModelSpec('UKMAP',ML1M,'a'*64,'v1')

    def test_checkpoint_sha_is_required(self):
        with self.assertRaises(RecommendationError):ModelSpec('UKMAP',M3L,'unverified','v1')

    def test_adapter_names_are_not_interchangeable(self):
        with self.assertRaises(RecommendationError):SKBHIFAdapter(SPECS['UKMAP'])

    def test_missing_checkpoint_backend_fails(self):
        with self.assertRaisesMessage(RecommendationError,'REAL_MODEL_NOT_VERIFIED'):
            UKMAPAdapter(SPECS['UKMAP']).infer(SimpleNamespace(dataset=M3L),10)

    def test_adapter_rejects_cross_dataset_profile(self):
        with self.assertRaisesMessage(RecommendationError,'PROFILE_DATASET_MISMATCH'):
            UKMAPAdapter(SPECS['UKMAP']).infer(SimpleNamespace(dataset=ML1M),10)

    def test_model_receipt_identity_must_match(self):
        profile=SimpleNamespace(dataset=M3L,canonical_user_id='0')
        batch=SyntheticBackend(SPECS['UKMAP']).infer(profile,3)
        for change in [{'checkpoint_identity':'a'*64},{'dataset':ML1M},{'canonical_user_id':'7'},
                       {'model_name':'SKB-HIF'}]:
            backend=SimpleNamespace(infer=lambda p,k,b=replace(batch,**change):b)
            with self.subTest(change=change),self.assertRaisesMessage(RecommendationError,'MODEL_PROVENANCE_MISMATCH'):
                UKMAPAdapter(SPECS['UKMAP'],backend).infer(profile,3)

    def test_rank_duplicates_and_unsorted_output_fail(self):
        p=SimpleNamespace(dataset=M3L,canonical_user_id='0')
        batch=SyntheticBackend(SPECS['UKMAP']).infer(p,3)
        for preds in [(Prediction('0',2),Prediction('0',1)),(Prediction('0',1),Prediction('1',2))]:
            backend=SimpleNamespace(infer=lambda p,k,b=replace(batch,predictions=preds):b)
            with self.subTest(preds=preds),self.assertRaisesMessage(RecommendationError,'INVALID_MODEL_RANKING'):
                UKMAPAdapter(SPECS['UKMAP'],backend).infer(p,3)

    def test_extra_model_results_not_silently_truncated(self):
        p=SimpleNamespace(dataset=M3L,canonical_user_id='0')
        b=SyntheticBackend(SPECS['UKMAP']).infer(p,3)
        with self.assertRaisesMessage(RecommendationError,'INVALID_MODEL_TOP_K'):
            UKMAPAdapter(SPECS['UKMAP'],SimpleNamespace(infer=lambda p,k:b)).infer(p,1)

    def test_same_dataset_different_manifest_is_not_accepted(self):
        p=SimpleNamespace(dataset=M3L,canonical_user_id='0')
        b=replace(SyntheticBackend(SPECS['UKMAP']).infer(p,3),catalogue_manifest_identity='0'*64)
        with self.assertRaisesMessage(RecommendationError,'MODEL_MANIFEST_MISMATCH'):
            UKMAPAdapter(SPECS['UKMAP'],SimpleNamespace(infer=lambda p,k:b)).infer(p,3)

    def test_train_manifest_does_not_treat_range_as_membership(self):
        p=SimpleNamespace(dataset=M3L,canonical_user_id='1',demo_only=True,
                          train_manifest_sha256=TRAINS[M3L].identity,profile_source=TRAINS[M3L].source_reference)
        with self.assertRaisesMessage(RecommendationError,'TRAIN_PROFILE_NOT_VERIFIED'):TRAINS[M3L].verify(p)


class ManifestAndOfflineTests(SimpleTestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory(prefix='task57-synthetic-')
        self.addCleanup(self.tmp.cleanup)
        self.path=Path(self.tmp.name)/'synthetic.json'

    def store(self,value):
        self.path.write_text(json.dumps(value))
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def train(self):
        return {'schema':'task57.train_profiles.v1','dataset':M3L,
                'evidence_kind':'REGISTERED_TRAIN_PROFILE_MEMBERSHIP','train_identity':'a'*64,
                'source_reference':'SYNTHETIC_ONLY','canonical_user_ids':['0','7']}

    def test_pinned_train_manifest_loads(self):
        sha=self.store(self.train())
        self.assertEqual(TrainProfileManifest.load(self.path,sha).users,frozenset({'0','7'}))

    def test_hash_mismatch_rejected(self):
        self.store(self.train())
        with self.assertRaisesMessage(RecommendationError,'MANIFEST_IDENTITY_MISMATCH'):
            TrainProfileManifest.load(self.path,'0'*64)

    def test_duplicate_train_member_rejected(self):
        v=self.train();v['canonical_user_ids']=['0','0'];sha=self.store(v)
        with self.assertRaisesMessage(RecommendationError,'DUPLICATE_TRAIN_PROFILE'):
            TrainProfileManifest.load(self.path,sha)

    def test_non_train_source_rejected(self):
        v=self.train();v['evidence_kind']='UNVERIFIED_USER_LIST';sha=self.store(v)
        with self.assertRaises(RecommendationError):TrainProfileManifest.load(self.path,sha)

    def test_catalogue_requires_external_id_namespace(self):
        v={'schema':'task57.movie_catalogue.v1','dataset':M3L,'source_namespace':'django:pk',
           'movies':[{'canonical_movie_id':'0','source_movie_id':'100'}]}
        sha=self.store(v)
        with self.assertRaises(RecommendationError):MovieCatalogueManifest.load(self.path,sha)

    def test_catalogue_rejects_duplicate_external_ids(self):
        v={'schema':'task57.movie_catalogue.v1','dataset':M3L,'source_namespace':'movielens:movieId',
           'movies':[{'canonical_movie_id':'0','source_movie_id':'100'},
                     {'canonical_movie_id':'1','source_movie_id':'100'}]}
        sha=self.store(v)
        with self.assertRaisesMessage(RecommendationError,'AMBIGUOUS_MOVIE_CATALOGUE'):
            MovieCatalogueManifest.load(self.path,sha)

    def bundle(self):
        return {'schema':'task57.offline_inference.v1','inference_mode':'OFFLINE_MODEL_INFERENCE',
                'model_name':'UKMAP','dataset':M3L,'checkpoint_identity':SPECS['UKMAP'].checkpoint_identity,
                'train_manifest_sha256':TRAINS[M3L].identity,'source_reference':'SYNTHETIC_TEST_BUNDLE',
                'catalogue_manifest_sha256':SPECS['UKMAP'].catalogue_manifest_identity,
                'candidate_scope':'SYNTHETIC_POOL','profiles':{'0':[{'canonical_movie_id':'0','score':3.0},
                                                               {'canonical_movie_id':'1','score':2.0}]}}

    def test_offline_outputs_are_explicitly_labelled(self):
        sha=self.store(self.bundle())
        backend=OfflineOutputBackend(self.path,sha,SPECS['UKMAP'],TRAINS[M3L].identity)
        p=SimpleNamespace(dataset=M3L,canonical_user_id='0',train_manifest_sha256=TRAINS[M3L].identity)
        result=UKMAPAdapter(SPECS['UKMAP'],backend).infer(p,2)
        self.assertEqual(result.inference_mode,'OFFLINE_MODEL_INFERENCE')
        self.assertEqual(result.output_artifact_identity,sha)

    def test_offline_mismatched_checkpoint_rejected(self):
        v=self.bundle();v['checkpoint_identity']='0'*64;sha=self.store(v)
        with self.assertRaises(RecommendationError):OfflineOutputBackend(self.path,sha,SPECS['UKMAP'],TRAINS[M3L].identity)

    def test_offline_insufficient_k_is_not_filled(self):
        sha=self.store(self.bundle());b=OfflineOutputBackend(self.path,sha,SPECS['UKMAP'],TRAINS[M3L].identity)
        p=SimpleNamespace(canonical_user_id='0',train_manifest_sha256=TRAINS[M3L].identity)
        with self.assertRaisesMessage(RecommendationError,'OFFLINE_TOP_K_UNAVAILABLE'):b.infer(p,3)

    def test_offline_unsorted_tail_not_hidden_by_top_k(self):
        v=self.bundle();v['profiles']['0'][1]['score']=4;sha=self.store(v)
        b=OfflineOutputBackend(self.path,sha,SPECS['UKMAP'],TRAINS[M3L].identity)
        p=SimpleNamespace(canonical_user_id='0',train_manifest_sha256=TRAINS[M3L].identity)
        with self.assertRaisesMessage(RecommendationError,'INVALID_OFFLINE_OUTPUT'):b.infer(p,1)

    def test_duplicate_json_keys_are_not_trusted(self):
        self.path.write_text('{"schema":"one","schema":"two"}')
        sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesMessage(RecommendationError,'DUPLICATE_JSON_KEY'):
            TrainProfileManifest.load(self.path,sha)

    def test_configured_file_read_is_bounded(self):
        self.path.write_bytes(b' '*2_000_001)
        with self.assertRaisesMessage(RecommendationError,'MANIFEST_TOO_LARGE'):
            TrainProfileManifest.load(self.path,'0'*64)

    def registry_entry(self):
        train_sha=self.store(self.train())
        catalogue=Path(self.tmp.name)/'synthetic_catalogue.json'
        catalogue.write_text(json.dumps({'schema':'task57.movie_catalogue.v1','dataset':M3L,
             'source_namespace':'movielens:movieId','movies':[{'canonical_movie_id':'0','source_movie_id':'100'}]}))
        return {'model_name':'UKMAP','dataset':M3L,'checkpoint_identity':'b'*64,'mapping_version':'synthetic-only',
                'profile_manifest_path':str(self.path),'profile_manifest_sha256':train_sha,
                'catalogue_manifest_path':str(catalogue),
                'catalogue_manifest_sha256':hashlib.sha256(catalogue.read_bytes()).hexdigest()}

    def test_operator_registry_never_automatically_loads_a_model(self):
        from research_recommendation.registry import configured_service
        with override_settings(TASK57_MODEL_REGISTRY=[self.registry_entry()]):
            app=configured_service()
        self.assertIsNone(app.adapters['UKMAP'].backend)
        self.assertEqual(app.mapping_services['UKMAP'].manifest.movies['0'],'100')
        with self.assertRaisesMessage(RecommendationError,'REAL_MODEL_NOT_VERIFIED'):
            app.adapters['UKMAP'].infer(SimpleNamespace(dataset=M3L),1)

    def test_duplicate_registry_is_not_last_entry_wins(self):
        from research_recommendation.registry import configured_service
        entry=self.registry_entry()
        with override_settings(TASK57_MODEL_REGISTRY=[entry,entry]),self.assertRaisesMessage(
                RecommendationError,'DUPLICATE_MODEL_REGISTRY'):
            configured_service()


class OptInBoundaryTests(SimpleTestCase):
    def test_production_default_is_off_without_importing_settings(self):
        root=Path(__file__).resolve().parents[2]
        tree=ast.parse((root/'movie/settings.py').read_text())
        assignments=[n for n in tree.body if isinstance(n,ast.Assign) and any(
            isinstance(t,ast.Name) and t.id=='TASK57_RECOMMENDATION_MVP_ENABLED' for t in n.targets)]
        self.assertEqual(len(assignments),1)
        self.assertIs(assignments[0].value.value,False)

    def test_production_url_hook_is_conditional(self):
        root=Path(__file__).resolve().parents[2]
        tree=ast.parse((root/'movie/urls.py').read_text())
        containing=[n for n in tree.body if isinstance(n,ast.If) and 'research_recommendation.urls' in ast.unparse(n)]
        self.assertEqual(len(containing),1)
        self.assertIn('TASK57_RECOMMENDATION_MVP_ENABLED',ast.unparse(containing[0].test))
