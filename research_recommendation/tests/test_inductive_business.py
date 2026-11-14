"""CPU/native-DB boundary tests; real model evidence comes from live HTTP."""
from unittest.mock import patch
from django.test import TestCase,override_settings
from django.contrib.auth import get_user_model
from myapp.models import Movie,UserRating,Collect
from research_recommendation.models import ResearchDataset,ResearchMovie,MovieIDBridge,UserPreference,ProfileGrant
from research_recommendation.product_mapping import MAPPING_VERSION
from research_recommendation.contracts import M3L
from research_recommendation.inductive_service import business_context

@override_settings(PAMK_INDUCTIVE_AVAILABLE=True,PAMK_INDUCTIVE_ENABLED=False)
class InductiveBusinessTests(TestCase):
    def setUp(self):
        self.user=get_user_model().objects.create_user(username='inductive_cpu',email='cpu@example.invalid',password='Safe-password-1')
        self.dataset=ResearchDataset.objects.create(protocol=M3L,complete=True,manifest={'catalogue_sha256':'c'*64,'assets':{'train_context_csr':{'sha256':'b'*64}}})
        self.movie=Movie.objects.create(title='Toy Story',movielens_id='1',score=4)
        ResearchMovie.objects.create(dataset=self.dataset,canonical_movie_id=0,source_movie_id='1',title='Toy Story (1995)')
        MovieIDBridge.objects.create(source_dataset=M3L,source_movie_id='1',canonical_movie_id='0',django_movie_id=self.movie.pk,match_method='EXACT_ID_TITLE_TRAILING_YEAR_ONLY',mapping_status='VERIFIED_UNIQUE',mapping_version=MAPPING_VERSION,evidence={'catalogue_sha256':'c'*64})
    def test_empty_has_no_encoder_input(self):
        self.assertEqual(business_context(self.user,set())[1],[])
    def test_legitimate_canonical_zero_is_input(self):
        UserRating.objects.create(user=self.user,movie=self.movie,score=5)
        d,h,ex,c,_=business_context(self.user,{self.movie.pk})
        self.assertEqual(h,[0]);self.assertEqual(ex,[0]);self.assertEqual(len(c),64)
        self.assertIsNone(self.user.user_ID);self.assertFalse(ProfileGrant.objects.filter(user=self.user).exists())
    def test_repeated_score_updates_no_duplicate_history(self):
        UserRating.objects.create(user=self.user,movie=self.movie,score=5)
        first=business_context(self.user,{self.movie.pk})
        UserRating.objects.update_or_create(user=self.user,movie=self.movie,defaults={'score':4})
        second=business_context(self.user,{self.movie.pk})
        self.assertEqual(second[1],[0]);self.assertNotEqual(first[3],second[3]);self.assertEqual(UserRating.objects.count(),1)
    def test_low_rating_is_not_positive_even_if_collected(self):
        Collect.objects.create(user=self.user,movie=self.movie)
        UserRating.objects.create(user=self.user,movie=self.movie,score=1)
        self.assertEqual(business_context(self.user,{self.movie.pk})[1],[])
    def test_duplicate_positive_sources_deduplicated(self):
        Collect.objects.create(user=self.user,movie=self.movie)
        UserRating.objects.create(user=self.user,movie=self.movie,score=5)
        self.assertEqual(business_context(self.user,{self.movie.pk})[1],[0])
    def test_ambiguous_mapping_never_becomes_encoder_input(self):
        UserRating.objects.create(user=self.user,movie=self.movie,score=5)
        MovieIDBridge.objects.update(mapping_status='AMBIGUOUS')
        d,h,_,_,missing=business_context(self.user,{self.movie.pk})
        self.assertEqual(h,[]);self.assertEqual(missing,1)
    def test_changed_movie_identity_is_rejected(self):
        UserRating.objects.create(user=self.user,movie=self.movie,score=5)
        self.movie.movielens_id='2';self.movie.save()
        self.assertEqual(business_context(self.user,{self.movie.pk})[1],[])
    def test_genre_preference_is_not_fake_movie_history(self):
        UserPreference.objects.create(user=self.user,genres=['动画'],movie_ids=[],catalogue_model='myapp.Movie')
        self.assertEqual(business_context(self.user,set())[1],[])
    def test_flag_off_keeps_cold_default(self):
        from research_recommendation.product_service import recommend
        UserRating.objects.create(user=self.user,movie=self.movie,score=5)
        with patch('research_recommendation.product_service.backend') as b:
            self.assertIsNone(recommend(self.user,{})['model_name']);b.assert_not_called()
    def test_untrusted_request_model_type_is_400(self):
        self.client.force_login(self.user)
        r=self.client.post('/api/recommendations/',{'model_name':[]},content_type='application/json')
        self.assertEqual(r.status_code,400)
