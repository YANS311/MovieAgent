"""Native Django/MySQL contract tests. Real GPU evidence is the live HTTP suite."""
from unittest.mock import patch
from django.test import TestCase, Client
from django.contrib.auth import get_user_model
from django.core.management import call_command
from io import StringIO
from myapp.models import Movie, UserRating
from research_recommendation.models import ResearchProfile, ProfileGrant, RecommendationTrace, MovieIDBridge, UserPreference
from research_recommendation.product_service import recommend
from research_recommendation.contracts import RecommendationError, M3L
from research_recommendation.product_mapping import normalized_title

class ProductIntegrationTests(TestCase):
    def setUp(self):
        self.user=get_user_model().objects.create_user(username='real_account',password='Safe-password-1',email='a@example.invalid')
        self.other=get_user_model().objects.create_user(username='other',password='Safe-password-2',email='b@example.invalid')
        self.movie=Movie.objects.create(title='Movie A',movielens_id='1',score=4.1)
        Movie.objects.create(title='Movie B',movielens_id='2',score=4.0)
        self.client.force_login(self.user)
    def test_cold_is_not_pamk_and_has_owned_trace(self):
        r=self.client.post('/api/recommendations/',{},content_type='application/json')
        self.assertEqual(r.status_code,200);d=r.json()
        self.assertEqual(d['source'],'METADATA_COLD_START');self.assertIsNone(d['checkpoint_sha256'])
        self.assertIsNone(d['canonical_user_id']);self.assertFalse(ProfileGrant.objects.filter(user=self.user).exists())
        self.assertEqual(RecommendationTrace.objects.get(pk=d['trace_id']).requester,self.user)
    def test_forged_id_and_payload_types(self):
        for data in [{'canonical_user_id':'0'},[],{'top_k':True},{'top_k':0},{'top_k':101},{'model_name':'PAMK-AV'}]:
            self.assertGreaterEqual(self.client.post('/api/recommendations/',data,content_type='application/json').status_code,400)
    def test_anonymous_and_csrf(self):
        c=Client();self.assertEqual(c.post('/api/recommendations/',{},content_type='application/json').status_code,401)
        c=Client(enforce_csrf_checks=True);c.force_login(self.user)
        self.assertEqual(c.post('/api/recommendations/',{},content_type='application/json').status_code,403)
    def test_cross_user_profile(self):
        p=ResearchProfile.objects.create(dataset=M3L,canonical_user_id='0',display_name='demo',profile_source='test-fixture',train_manifest_sha256='a'*64)
        ProfileGrant.objects.create(user=self.other,profile=p)
        r=self.client.post('/api/recommendations/',{'profile_id':str(p.pk)},content_type='application/json')
        self.assertEqual(r.status_code,403)
    def test_revoked_grant_not_implicitly_selected(self):
        p=ResearchProfile.objects.create(dataset=M3L,canonical_user_id='0',display_name='demo',profile_source='fixture',train_manifest_sha256='a'*64)
        ProfileGrant.objects.create(user=self.user,profile=p,is_active=False)
        self.assertEqual(recommend(self.user,{})['source'],'METADATA_COLD_START')
        with self.assertRaisesRegex(RecommendationError,'PROFILE_GRANT_REQUIRED'):recommend(self.user,{'profile_id':str(p.pk)})
    def test_rating_excluded_from_cold(self):
        UserRating.objects.create(user=self.user,movie=self.movie,score=5)
        self.assertNotIn(self.movie.pk,[r['django_movie_id'] for r in recommend(self.user,{})['recommendations']])
    def test_trace_cross_user_and_no_store(self):
        d=recommend(self.user,{})
        self.client.force_login(self.other)
        r=self.client.get('/recommendation-traces/'+d['trace_id']+'/');self.assertEqual(r.status_code,404)
        self.assertIn('no-store',r.headers['Cache-Control'])
    def test_title_normalization_preserves_parenthetical_identity(self):
        self.assertEqual(normalized_title('Toy Story (1995)'),'toy story')
        self.assertNotEqual(normalized_title('Movie (Director cut) (1995)'),normalized_title('Movie (1995)'))
    def test_real_product_urlconf(self):
        from django.urls import resolve
        self.assertEqual(resolve('/api/recommendations/').func.__module__,'research_recommendation.product_views')
        self.assertEqual(resolve('/recommend/').func.__module__,'research_recommendation.product_views')
    def test_register_does_not_create_profile(self):
        c=Client();r=c.post('/register/',{'username':'fresh','email':'fresh@example.invalid','password1':'New-Account-Safe-543!','password2':'New-Account-Safe-543!'})
        self.assertEqual(r.status_code,302)
        u=get_user_model().objects.get(username='fresh');self.assertIsNone(u.user_ID)
        self.assertFalse(ProfileGrant.objects.filter(user=u).exists())
    def test_immutable_research_source_file_not_changed_by_cold(self):
        with patch('research_recommendation.product_service.backend') as backend:
            recommend(self.user,{})
            backend.assert_not_called()
    def test_pages_details_and_interactions(self):
        for path in ['/','/depot/','/recommend/','/preferences/',f'/movie/{self.movie.pk}/']:
            self.assertEqual(self.client.get(path).status_code,200)
        self.assertEqual(self.client.post(f'/movie/{self.movie.pk}/interaction/',{'kind':'rating','score':'4.5'}).status_code,302)
        self.assertEqual(UserRating.objects.get(user=self.user,movie=self.movie).score,4.5)

    def test_trace_failure_returns_explicit_unavailable(self):
        from django.db import DatabaseError
        with patch.object(RecommendationTrace.objects, 'create', side_effect=DatabaseError('forced test trace failure')):
            r=self.client.post('/api/recommendations/',{},content_type='application/json')
        self.assertEqual(r.status_code,503)
        self.assertEqual(r.json()['error'],'TRACE_PERSISTENCE_FAILED')

    def test_import_rollback_preserves_accounts_and_business_data(self):
        from research_recommendation.models import ResearchDataset,ResearchMovie,ResearchUser,ResearchTrainInteraction
        d=ResearchDataset.objects.create(protocol=M3L,manifest={'assets':{'train_context_csr':{'sha256':'a'*64}}},complete=True)
        ResearchMovie.objects.create(dataset=d,canonical_movie_id=0,source_movie_id='1',title='source')
        ResearchUser.objects.create(dataset=d,canonical_user_id=0,train_count=1)
        ResearchTrainInteraction.objects.create(dataset=d,canonical_user_id=0,canonical_movie_id=0)
        p=ResearchProfile.objects.create(dataset=M3L,canonical_user_id='0',display_name='fixture',profile_source='fixture',train_manifest_sha256='a'*64)
        grant=ProfileGrant.objects.create(user=self.user,profile=p)
        call_command('rollback_pamk_import',confirm_train_sha256='a'*64,stdout=StringIO())
        self.assertTrue(ResearchDataset.objects.filter(pk=d.pk).exists())
        call_command('rollback_pamk_import',confirm_train_sha256='a'*64,apply=True,stdout=StringIO())
        self.assertFalse(ResearchDataset.objects.exists());self.assertEqual(ResearchTrainInteraction.objects.count(),0)
        grant.refresh_from_db();self.assertFalse(grant.is_active)
        self.assertTrue(Movie.objects.filter(pk=self.movie.pk).exists());self.assertTrue(get_user_model().objects.filter(pk=self.user.pk).exists())
