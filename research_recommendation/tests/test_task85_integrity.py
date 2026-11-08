"""Synthetic boundary regressions. These are NOT real-model or HTTP evidence."""
from dataclasses import replace
from unittest.mock import patch
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.db import DatabaseError
from django.test import TestCase, override_settings
from research_recommendation.contracts import M3L, ModelSpec, PAMKAdapter, InferenceBatch, Prediction, RecommendationError
from research_recommendation.models import ResearchProfile, ProfileGrant, RecommendationTrace
from research_recommendation.model_cache import model_cache_key
from research_recommendation.unified_adapter import UnifiedRecommenderAdapter, UserIdentityCategory

class Backend:
    def __init__(self,spec):self.spec=spec;self.calls=0
    def infer(self,profile,top_k):
        self.calls+=1
        return InferenceBatch('PAMK-Net',M3L,self.spec.checkpoint_identity,profile.canonical_user_id,
            (Prediction('504',1.0),),'TEST_DOUBLE','FULL_TRAIN_UNSEEN:9013/9031;filtered=18',
            'Task85:SYNTHETIC',None,self.spec.train_manifest_identity,self.spec.catalogue_manifest_identity)

@override_settings(TASK84_MODEL_CACHE_ENABLED=True,CACHES={'default':{'BACKEND':'django.core.cache.backends.locmem.LocMemCache','LOCATION':'task85-regressions'}})
class Task85IntegrityTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user=get_user_model().objects.create_user('task85-demo')
        self.cold=get_user_model().objects.create_user('task85-cold')
        self.profile=ResearchProfile.objects.create(dataset=M3L,canonical_user_id='0',display_name='Demo',profile_source='TRAIN_ONLY_AUDIT',train_manifest_sha256='a'*64)
        self.grant=ProfileGrant.objects.create(user=self.user,profile=self.profile)
        self.spec=ModelSpec('PAMK-Net',M3L,'b'*64,'v1','a'*64,'c'*64)
        self.backend=Backend(self.spec)
        self.facade=UnifiedRecommenderAdapter(adapters={'PAMK-Net':PAMKAdapter(self.spec,self.backend)})
    def rec(self):return self.facade.recommend(self.user,top_k=1,profile_id=str(self.profile.pk))
    def test_grant_is_demo_even_for_canonical_zero(self):
        self.assertEqual(self.facade.resolve_user_identity(self.user,str(self.profile.pk)).category,UserIdentityCategory.BUSINESS_DEMO_GRANTED)
    def test_cold_invalid_k_rejected_before_catalogue(self):
        with self.assertRaises(RecommendationError) as e:self.facade.recommend(self.cold,top_k=-1)
        self.assertEqual(e.exception.code,'INVALID_TOP_K')
    def test_cold_unknown_model_rejected(self):
        with self.assertRaises(RecommendationError) as e:self.facade.recommend(self.cold,model_version='MADE_UP')
        self.assertEqual(e.exception.code,'UNSUPPORTED_MODEL')
    def test_cold_pamk_av_remains_explicitly_unavailable(self):
        with self.assertRaises(RecommendationError) as e:self.facade.recommend(self.cold,model_version='PAMK-AV')
        self.assertEqual(e.exception.code,'PAMK_AV_INFERENCE_PENDING_VERIFICATION')
    def test_cache_hit_and_trace_for_every_request(self):
        first=self.rec();second=self.rec()
        self.assertEqual(second['cache_status'],'HIT');self.assertEqual(self.backend.calls,1)
        self.assertEqual(RecommendationTrace.objects.filter(requester=self.user).count(),2)
        self.assertNotEqual(first['request_id'],second['request_id'])
    def test_train_manifest_change_cannot_reuse_cache(self):
        self.rec();spec=replace(self.spec,train_manifest_identity='d'*64)
        new=Backend(spec);self.facade.adapters['PAMK-Net']=PAMKAdapter(spec,new)
        self.profile.train_manifest_sha256='d'*64;self.profile.save()
        self.rec();self.assertEqual(new.calls,1)
    def test_revoked_grant_cannot_hit_cache(self):
        self.rec();self.grant.is_active=False;self.grant.save()
        with self.assertRaises(RecommendationError) as e:self.rec()
        self.assertEqual(e.exception.status,404)
    def test_feature_version_full_digest(self):
        args=('PAMK-Net','b'*64)
        k1,_=model_cache_key(*args,'1234567890123456A','user',1)
        k2,_=model_cache_key(*args,'1234567890123456B','user',1)
        self.assertNotEqual(k1,k2)
    def test_trace_failure_must_fail_closed(self):
        with patch.object(RecommendationTrace.objects,'create',side_effect=DatabaseError('audit')):
            with self.assertRaises(RecommendationError) as e:self.rec()
        self.assertEqual(e.exception.code,'TRACE_PERSISTENCE_FAILED')
