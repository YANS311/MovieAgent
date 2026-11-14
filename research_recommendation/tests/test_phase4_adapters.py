"""Incremental regression tests for Chapter 5 Model Adapters, Identity Resolution, and Caching.

Covers:
A. Adapter unit tests (UKMAPAdapter, PAMKAdapter, PAMKAVAdapter)
B. Checkpoint loading & config verification
C. Real GPU Worker smoke check against historical Top-5 IDs (not a full replay)
D. Canonical user mapping resolution (4 categories)
E. Canonical movie mapping audit & resolution
F. Unauthorized access rejection (401, 404)
G. New user cold start fallback (no M3L borrowing)
H. Redis cache version isolation
I. Trace explanation tool registration contract (not Agent dispatch)
J. Synthetic Client profile request plus direct synthetic recommendation
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import unittest

from django.conf import settings
from django.contrib.auth import get_user_model
from django.contrib.auth.models import AnonymousUser
from django.test import TestCase, override_settings

from research_recommendation.contracts import (
    M3L,
    ML1M,
    MODEL_DATASETS,
    ModelSpec,
    PAMKAdapter,
    PAMKAVAdapter,
    Prediction,
    RecommendationError,
    SKBHIFAdapter,
    UKMAPAdapter,
)
from research_recommendation.model_cache import (
    digest,
    model_cache_key,
    read_model_cache,
    write_model_cache,
)
from research_recommendation.models import (
    MovieIDBridge,
    ProfileGrant,
    RecommendationTrace,
    ResearchProfile,
    UserPreference,
)
from research_recommendation.page_catalogue import MovieCard
from research_recommendation.services import DjangoMovieCatalogue, MappingService, ProfileService
from research_recommendation.unified_adapter import (
    IdentityResolution,
    UnifiedRecommenderAdapter,
    UserIdentityCategory,
)

User = get_user_model()

CH4_ROOT = Path('/home/daylight/下载/DjangoProject3/chapter4_agy_frontier_2024_2026')
ROOT_M3L = Path('/home/daylight/下载/DjangoProject3_ch4_new/chapter4_m3l_author_aligned')
PYTHON_BIN = '/home/daylight/anaconda3/envs/DjangoProject3/bin/python'

PAMK_CKPT_PATH = CH4_ROOT / 'results/checkpoints/pamk_seed2024_extended/best.pt'
PARENT_CKPT_PATH = ROOT_M3L / '20_canonical_skbhif_final/04_runs/seed2024/best.pt'
UKMAP_CKPT_PATH = ROOT_M3L / '32_corrected_kg_final_candidate/04_m3_paired_evidence/best.pt'

PAMK_SHA = 'd811034f4875c52c0d827a6f46948c700fc18826992b7fec0264bf06c1fb3071'
PARENT_SHA = 'b97feb824f427e6437f728459fbeb443af9b8bfdbba18ac33db8786007c33cd6'
UKMAP_SHA = '73c7d95868005dae1a461d367a5afa0d3550c1a8379a9125764c4334dab2c7d6'


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


@override_settings(ROOT_URLCONF='movie.task57_urls', TASK57_SERVICE_FACTORY='research_recommendation.test_support.fixtures.test_service')
class RecommenderAdapterPhase4Tests(TestCase):
    def setUp(self):
        from research_recommendation.views import service
        svc = service()
        m3l_manifest = svc.profile_manifests.get(M3L)
        if m3l_manifest:
            self.train_sha = m3l_manifest.identity
            self.profile_source = m3l_manifest.source_reference
        else:
            self.train_sha = '7620eea4f28052a0d5ec822acb2676a6262be96220dc4907393324c9a9ca1e54'
            self.profile_source = 'M3L_TRAIN_BENCHMARK'
        self.cat_sha = 'fa2ce14ff4c2b64b3bfcc76fde95882eac74e878d4e9034793b94e5ca03e0266'
        self.user = User.objects.create_user(username='tester_phase4', password='password123')
        self.other_user = User.objects.create_user(username='other_user', password='password123')
        self.profile = ResearchProfile.objects.create(
            dataset=M3L,
            canonical_user_id='0',
            display_name='Canonical M3L User 0',
            profile_source=self.profile_source,
            train_manifest_sha256=self.train_sha,
            demo_only=True,
            is_active=True,
        )
        self.grant = ProfileGrant.objects.create(user=self.user, profile=self.profile, is_active=True)

    # -------------------------------------------------------------
    # Test A: Adapter Unit Tests & Interfaces
    # -------------------------------------------------------------
    def test_adapter_unit_interfaces(self):
        # UKMAP
        spec_ukmap = ModelSpec('UKMAP', M3L, UKMAP_SHA, 'v1', self.train_sha, self.cat_sha)
        adapter_ukmap = UKMAPAdapter(spec_ukmap)
        self.assertEqual(adapter_ukmap.model_name, 'UKMAP')
        self.assertEqual(MODEL_DATASETS['UKMAP'], M3L)

        # PAMK-Net
        spec_pamk = ModelSpec('PAMK-Net', M3L, PAMK_SHA, 'v1', self.train_sha, self.cat_sha)
        adapter_pamk = PAMKAdapter(spec_pamk)
        self.assertEqual(adapter_pamk.model_name, 'PAMK-Net')
        self.assertEqual(MODEL_DATASETS['PAMK-Net'], M3L)

        # PAMK-AV
        spec_av = ModelSpec('PAMK-AV', M3L, PAMK_SHA, 'v1', self.train_sha, self.cat_sha)
        adapter_av = PAMKAVAdapter(spec_av)
        self.assertEqual(adapter_av.model_name, 'PAMK-AV')
        self.assertEqual(MODEL_DATASETS['PAMK-AV'], M3L)
        self.assertEqual(len(adapter_av.feature_manifests), 5)
        self.assertIn('audio', adapter_av.feature_manifests)
        self.assertIn('video', adapter_av.feature_manifests)
        self.assertEqual(adapter_av.feature_manifests['audio'].dimension, 128)
        self.assertEqual(adapter_av.feature_manifests['video'].dimension, 768)

        # PAMK-AV without verified backend fails closed with 503
        with self.assertRaises(RecommendationError) as ctx:
            adapter_av.infer(self.profile, top_k=20)
        self.assertEqual(ctx.exception.status, 503)
        self.assertEqual(ctx.exception.code, 'PAMK_AV_INFERENCE_PENDING_VERIFICATION')

    # -------------------------------------------------------------
    # Test B: Checkpoint Loading & Config Verification
    # -------------------------------------------------------------
    @unittest.skipIf(os.environ.get('DJANGO_SETTINGS_MODULE')=='movie.release_settings', 'Historical d811/UKMAP asset test; release f098 identity is verified by preflight and real HTTP, not this old fixture')
    def test_checkpoint_and_config_verification(self):
        self.assertTrue(PAMK_CKPT_PATH.is_file(), f'PAMK ckpt missing: {PAMK_CKPT_PATH}')
        self.assertTrue(PARENT_CKPT_PATH.is_file(), f'Parent ckpt missing: {PARENT_CKPT_PATH}')
        self.assertTrue(UKMAP_CKPT_PATH.is_file(), f'UKMAP ckpt missing: {UKMAP_CKPT_PATH}')

        self.assertEqual(file_sha256(PAMK_CKPT_PATH), PAMK_SHA)
        self.assertEqual(file_sha256(PARENT_CKPT_PATH), PARENT_SHA)
        self.assertEqual(file_sha256(UKMAP_CKPT_PATH), UKMAP_SHA)

    # -------------------------------------------------------------
    # Test C: Offline Replay Consistency on Identical Inputs
    # -------------------------------------------------------------
    def test_offline_replay_consistency_on_identical_inputs(self):
        if getattr(settings, 'ROOT_URLCONF', None) == 'movie.task57_urls':
            self.skipTest('Skipped in task57 isolated sandbox: subprocess forbidden by task57 Guard')
        # Invoke pamk_worker.py via child process
        worker_script = Path(__file__).resolve().parents[1] / 'pamk_worker.py'
        self.assertTrue(worker_script.is_file())

        proc = subprocess.Popen(
            [PYTHON_BIN, '-B', str(worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            req = json.dumps({'user': '0', 'top_k': 20, 'smoke': False}) + '\n'
            proc.stdin.write(req)
            proc.stdin.flush()
            line = proc.stdout.readline()
            response = json.loads(line)

            self.assertTrue(response.get('ok'), f'Worker error: {response}')
            result = response['result']

            # Invariant: 9,013 candidates scored (9,031 minus 18 Train seen)
            self.assertEqual(result['candidate_count'], 9013)
            self.assertEqual(result['train_seen_count'], 18)
            self.assertEqual(result['catalogue_size'], 9031)

            # Invariant: Top 5 ranking matches Chapter 4 validation
            items = result['items']
            top_ids = [int(x['canonical_movie_id']) for x in items[:5]]
            self.assertEqual(top_ids, [504, 428, 311, 497, 288])

            # Invariant: Scores strictly descending
            scores = [x['score'] for x in items]
            for i in range(len(scores) - 1):
                self.assertGreaterEqual(scores[i], scores[i + 1])

            # Invariant: Telemetry present on each returned item
            for item in items[:5]:
                self.assertIn('telemetry', item)
                t = item['telemetry']
                self.assertIn('pi_text', t)
                self.assertIn('pi_vis', t)
                self.assertIn('pi_kg', t)
                self.assertIn('popularity', t)
                self.assertIn('parent_score', t)
                # Weights sum roughly to 1
                gate_sum = t['pi_text'] + t['pi_vis'] + t['pi_kg']
                self.assertAlmostEqual(gate_sum, 1.0, places=2)
        finally:
            proc.stdin.write(json.dumps({'action': 'close'}) + '\n')
            proc.stdin.flush()
            proc.wait(timeout=5)

    # -------------------------------------------------------------
    # Test D: Canonical User Mapping Resolution (4 Categories)
    # -------------------------------------------------------------
    def test_canonical_user_mapping_resolution(self):
        adapter = UnifiedRecommenderAdapter()

        # 1. Anonymous user -> 401
        with self.assertRaises(RecommendationError) as ctx:
            adapter.resolve_user_identity(AnonymousUser())
        self.assertEqual(ctx.exception.status, 401)

        # 2. A grant authorizes a demo; canonical ID zero proves no human identity.
        res1 = adapter.resolve_user_identity(self.user, str(self.profile.id))
        self.assertEqual(res1.category, UserIdentityCategory.BUSINESS_DEMO_GRANTED)
        self.assertTrue(res1.can_execute_model)
        self.assertEqual(res1.profile.canonical_user_id, '0')

        # 3. Regular user with preferences but NO grant -> BUSINESS_UNMAPPED
        UserPreference.objects.create(user=self.other_user, genres=['Action', 'Sci-Fi'], catalogue_model='myapp.Movie')
        res2 = adapter.resolve_user_identity(self.other_user)
        self.assertEqual(res2.category, UserIdentityCategory.BUSINESS_UNMAPPED)
        self.assertFalse(res2.can_execute_model)
        self.assertIsNone(res2.profile)

        # 4. Pure cold-start new user -> NEW_USER_COLD_START
        cold_user = User.objects.create_user(username='cold_user', password='password123')
        res3 = adapter.resolve_user_identity(cold_user)
        self.assertEqual(res3.category, UserIdentityCategory.NEW_USER_COLD_START)
        self.assertFalse(res3.can_execute_model)
        self.assertIsNone(res3.profile)

    # -------------------------------------------------------------
    # Test E: Canonical Movie Mapping Audit & Resolution
    # -------------------------------------------------------------
    def test_canonical_movie_mapping_resolution(self):
        # Create mock bridge entries
        MovieIDBridge.objects.create(
            source_dataset=M3L,
            source_movie_id='595',
            canonical_movie_id='504',
            django_movie_id=76016,
            match_method='EXACT_EXTERNAL_ID_REVIEWED',
            evidence={'catalogue_sha256': self.cat_sha, 'source_namespace': 'movielens', 'review_reference': 'verified_link'},
            mapping_status='VERIFIED_UNIQUE',
            mapping_version='v1',
        )
        MovieIDBridge.objects.create(
            source_dataset=M3L,
            source_movie_id='9999',
            canonical_movie_id='9999',
            django_movie_id=None,
            match_method='EXACT_EXTERNAL_ID_REVIEWED',
            evidence={'catalogue_sha256': self.cat_sha, 'source_namespace': 'movielens', 'review_reference': 'title_conflict'},
            mapping_status='AMBIGUOUS',
            mapping_version='v1',
        )

        class DummyManifest:
            dataset = M3L
            identity = self.cat_sha
            source_namespace = 'movielens'
            movies = {'504': '595', '9999': '9999', '123': '123'}

        class DummyCatalogue:
            def get_many(self, ids):
                from research_recommendation.services import MovieRecord
                res = {}
                if 76016 in ids:
                    res[76016] = MovieRecord(76016, 'Beauty and the Beast (1991)', '595')
                return res

        service = MappingService(DummyManifest(), 'v1', DummyCatalogue())
        resolved = service.resolve_many(['504', '9999', '123'])

        # 504 is verified unique
        self.assertEqual(resolved['504']['mapping_status'], 'VERIFIED_UNIQUE')
        self.assertEqual(resolved['504']['django_movie_id'], 76016)
        self.assertEqual(resolved['504']['movie_title'], 'Beauty and the Beast (1991)')

        # 9999 is ambiguous
        self.assertEqual(resolved['9999']['mapping_status'], 'AMBIGUOUS')
        self.assertIsNone(resolved['9999']['django_movie_id'])

        # 123 is unmapped
        self.assertEqual(resolved['123']['mapping_status'], 'UNMAPPED')
        self.assertIsNone(resolved['123']['django_movie_id'])

    # -------------------------------------------------------------
    # Test F: Unauthorized Access Rejection Tests
    # -------------------------------------------------------------
    def test_unauthorized_access_rejections(self):
        # 1. User without grant cannot request another user's profile
        with self.assertRaises(RecommendationError) as ctx:
            ProfileService.authorized(self.other_user, str(self.profile.id))
        self.assertEqual(ctx.exception.status, 404)
        self.assertEqual(ctx.exception.code, 'PROFILE_NOT_AVAILABLE')

        # 2. Disabled grant is rejected
        self.grant.is_active = False
        self.grant.save()
        with self.assertRaises(RecommendationError) as ctx:
            ProfileService.authorized(self.user, str(self.profile.id))
        self.assertEqual(ctx.exception.status, 404)

        # 3. Unsupported model version rejected
        adapter = UnifiedRecommenderAdapter()
        with self.assertRaises(RecommendationError) as ctx:
            adapter.recommend(self.user, top_k=20, model_version='INVALID_MODEL', profile_id=str(self.profile.id))
        # Re-enable grant
        self.grant.is_active = True
        self.grant.save()
        with self.assertRaises(RecommendationError) as ctx:
            adapter.recommend(self.user, top_k=20, model_version='INVALID_MODEL', profile_id=str(self.profile.id))
        self.assertEqual(ctx.exception.code, 'UNSUPPORTED_MODEL')

    # -------------------------------------------------------------
    # Test G: New User Cold Start Fallback
    # -------------------------------------------------------------
    def test_new_user_cold_start_fallback(self):
        if getattr(settings, 'ROOT_URLCONF', None) == 'movie.task57_urls':
            self.skipTest('Skipped in task57 isolated sandbox: task60 movie snapshot required')
        new_user = User.objects.create_user(username='newbie_user', password='password123')
        adapter = UnifiedRecommenderAdapter()

        # Execute recommendation for unmapped user
        resp = adapter.recommend(new_user, top_k=10)

        # Invariant: Clearly marked as cold start fallback, NOT model inference
        self.assertEqual(resp['model_name'], 'METADATA_COLD_START_FALLBACK')
        self.assertEqual(resp['checkpoint_identity'], 'none')
        self.assertEqual(resp['inference_mode'], 'RULE_BASED_COLD_START')
        self.assertEqual(resp['profile_source']['mode'], UserIdentityCategory.NEW_USER_COLD_START.value)
        self.assertFalse(resp['profile_source']['demo_only'])

        # Invariant: Never contains M3L canonical user 0 in profile_source
        self.assertNotIn('canonical_user_id', resp['profile_source'])

    # -------------------------------------------------------------
    # Test H: Redis Cache Version & Model Isolation
    # -------------------------------------------------------------
    def test_cache_version_and_model_isolation(self):
        key_pamk, owner_pamk = model_cache_key('PAMK-Net', PAMK_SHA, 'v1', 'user1:0', 20)
        key_ukmap, owner_ukmap = model_cache_key('UKMAP', UKMAP_SHA, 'v1', 'user1:0', 20)
        key_user2, owner_user2 = model_cache_key('PAMK-Net', PAMK_SHA, 'v1', 'user2:0', 20)
        key_v2, owner_v2 = model_cache_key('PAMK-Net', PAMK_SHA, 'v2', 'user1:0', 20)

        # Invariant: Model keys are strictly distinct (no PAMK hitting UKMAP cache)
        self.assertNotEqual(key_pamk, key_ukmap)
        self.assertNotEqual(owner_pamk, owner_ukmap)

        # Invariant: User scopes are strictly distinct (no cross-user sharing)
        self.assertNotEqual(key_pamk, key_user2)
        self.assertNotEqual(owner_pamk, owner_user2)

        # Invariant: Feature versions are strictly distinct (cache invalidation on feature update)
        self.assertNotEqual(key_pamk, key_v2)

        # Write & Read Cache
        with override_settings(TASK84_MODEL_CACHE_ENABLED=True):
            test_payload = {'recommendations': [{'canonical_movie_id': '504', 'score': 3.308}]}
            status = write_model_cache('PAMK-Net', PAMK_SHA, 'v1', 'user1:0', 20, test_payload)
            self.assertIn(status, ('STORED', 'DISABLED'))

            read_back, cache_status = read_model_cache('PAMK-Net', PAMK_SHA, 'v1', 'user1:0', 20)
            if cache_status == 'HIT':
                self.assertTrue(read_back['cache_hit'])
                self.assertEqual(read_back['recommendations'][0]['canonical_movie_id'], '504')

            # Attempting to read with UKMAP fails as MISS
            miss_payload, miss_status = read_model_cache('UKMAP', UKMAP_SHA, 'v1', 'user1:0', 20)
            self.assertIsNone(miss_payload)
            self.assertEqual(miss_status, 'MISS')

    # -------------------------------------------------------------
    # Test I: Agent Recommendation Tool Interface Test
    # -------------------------------------------------------------
    def test_agent_recommendation_tool_interface(self):
        if getattr(settings, 'ROOT_URLCONF', None) == 'movie.task57_urls':
            self.skipTest('Skipped in task57 isolated sandbox: myapp forbidden by task57 Guard')
        from myapp.services.trace_explanation import TraceExplanationTool, register_trace_tool

        tool = TraceExplanationTool()
        self.assertEqual(tool.name, 'explain_trace')

        class DummySpec:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class DummyRegistry:
            def __init__(self):
                self.tools = {}
            def register(self, s):
                self.tools[s.name] = s

        reg = DummyRegistry()
        register_trace_tool(reg, DummySpec)
        spec = reg.tools.get('explain_trace')
        self.assertIsNotNone(spec)
        self.assertIn('trace_id', spec.input_schema)
        self.assertIn('movie_id', spec.input_schema)

        # Create a trace with PAMK-Net provenance
        trace_id = uuid.uuid4()
        RecommendationTrace.objects.create(
            id=trace_id,
            requester=self.user,
            outcome='SUCCESS',
            provenance={
                'model_name': 'PAMK-Net',
                'checkpoint_identity': PAMK_SHA,
                'inference_mode': 'REAL_ONLINE_INFERENCE',
                'user_category': UserIdentityCategory.CANONICAL_M3L_BENCHMARK.value,
            },
        )
        saved_trace = RecommendationTrace.objects.filter(pk=trace_id, requester=self.user).first()
        self.assertIsNotNone(saved_trace)
        self.assertEqual(saved_trace.provenance['model_name'], 'PAMK-Net')

    # -------------------------------------------------------------
    # Test J: Django HTTP Interface Integration Test
    # -------------------------------------------------------------
    def test_django_http_interface_integration(self):
        from django.test import Client
        from research_recommendation.test_support.fixtures import SyntheticBackend, SyntheticCatalogue, CATALOGUES
        from research_recommendation.views import service

        client = Client()
        client.force_login(self.user)

        # GET /api/research/profiles/ returns authorized profile
        # This is a synthetic HTTP contract test. Pin its factory explicitly;
        # a real-manifest settings module must not change its synthetic fixture.
        with override_settings(TASK57_SERVICE_FACTORY='research_recommendation.test_support.fixtures.test_service'):
            resp = client.get('/api/research/profiles/')
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn('profiles', data)
        self.assertEqual(len(data['profiles']), 1)
        self.assertEqual(data['profiles'][0]['canonical_user_id'], '0')

        # Unified adapter recommend call via service
        svc = service()
        mapping_service = svc.mapping_services.get('UKMAP') or MappingService(CATALOGUES[M3L], 'v1', SyntheticCatalogue())
        spec_pamk = ModelSpec('PAMK-Net', M3L, PAMK_SHA, mapping_service.version,
                              self.train_sha, mapping_service.manifest.identity)
        backend = SyntheticBackend(spec_pamk)
        adapter_pamk = PAMKAdapter(spec_pamk, backend)
        m3l_manifest = svc.profile_manifests.get(M3L)
        adapter = UnifiedRecommenderAdapter(
            adapters={'PAMK-Net': adapter_pamk},
            mapping_service=mapping_service,
            profile_manifest=m3l_manifest,
        )
        rec_res = adapter.recommend(self.user, top_k=5, model_version='PAMK-Net', profile_id=str(self.profile.id))
        self.assertEqual(rec_res['model_name'], 'PAMK-Net')
        self.assertEqual(rec_res['dataset'], M3L)
        self.assertIn('recommendations', rec_res)
        self.assertIn('request_id', rec_res)
        self.assertIn('mapping_summary', rec_res)
