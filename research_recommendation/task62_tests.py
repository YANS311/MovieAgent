"""Real Django/ORM tests; small synthetic film metadata, no model inference."""
from unittest.mock import patch
from django.db import OperationalError
from uuid import uuid4
from django.contrib.auth import get_user_model
from django.test import TestCase, Client, override_settings
from .models import UserPreference, PersonalRecommendationTrace, MovieInteraction, ProfileGrant, ResearchProfile
from .personal import catalogue, rank, catalogue_label
from .task60_support.models import Movie

PASSWORD = 'Local-test-secret-62!'


class PersonalTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(username='ordinary62', email='a@example.test', password=PASSWORD)
        self.other = get_user_model().objects.create_user(username='other62', email='b@example.test', password=PASSWORD)
        Movie.objects.bulk_create([
            Movie(id=10, title='SYNTHETIC Action Seed', movielens_id='s10', genre_labels=['Action', 'Drama']),
            Movie(id=20, title='SYNTHETIC Drama', movielens_id='s20', genre_labels=['Drama']),
            Movie(id=30, title='SYNTHETIC Action', movielens_id='s30', genre_labels=['Action']),
            Movie(id=40, title='SYNTHETIC Comedy', movielens_id='s40', genre_labels=['Comedy']),
        ])

    def login(self, client=None, user=None):
        return (client or self.client).post('/cinema/login/', {'username': (user or self.user).username, 'password': PASSWORD})

    def save_preference(self, **overrides):
        data = {'genres': ['Action'], 'movie_ids': '10'}
        data.update(overrides)
        return self.client.post('/cinema/personal/preferences/', data)

    def generate(self):
        response = self.client.post('/cinema/personal/')
        self.assertEqual(response.status_code, 302)
        return PersonalRecommendationTrace.objects.get(user=self.user), response.url

    def test_registration_login_logout_relogin(self):
        r = self.client.post('/cinema/register/', {'username': 'new62', 'email': 'new@example.test', 'password1': PASSWORD, 'password2': PASSWORD})
        self.assertEqual(r.status_code, 302)
        account = get_user_model().objects.get(username='new62')
        self.assertTrue(account.check_password(PASSWORD))
        self.assertNotEqual(account.password, PASSWORD)
        self.assertNotIn('_auth_user_id', self.client.session)
        self.assertEqual(self.client.post('/cinema/login/', {'username': 'new62', 'password': PASSWORD}).url, '/cinema/personal/')
        self.assertEqual(self.client.get('/cinema/logout/').status_code, 405)
        self.assertEqual(self.client.post('/cinema/logout/').status_code, 302)
        self.assertNotIn('_auth_user_id', self.client.session)
        self.assertEqual(ProfileGrant.objects.count(), 0)
        self.assertEqual(ResearchProfile.objects.count(), 0)

    def test_registration_duplicate_and_password_validation(self):
        for username, p1, p2 in [('ordinary62', PASSWORD, PASSWORD), ('new', 'short', 'short'), ('new', PASSWORD, 'different')]:
            self.assertEqual(self.client.post('/cinema/register/', {'username': username, 'email': 'new@example.test', 'password1': p1, 'password2': p2}).status_code, 200)
        self.assertEqual(get_user_model().objects.count(), 2)

    def test_registration_cannot_set_privilege_or_research_identity(self):
        self.client.post('/cinema/register/', {'username': 'new62', 'email': 'new@example.test', 'password1': PASSWORD, 'password2': PASSWORD,
            'is_staff': 'true', 'is_superuser': 'true', 'user_ID': '0', 'canonical_user_id': '0'})
        account = get_user_model().objects.get(username='new62')
        self.assertFalse(account.is_staff)
        self.assertFalse(account.is_superuser)
        self.assertEqual(ProfileGrant.objects.count(), 0)
        self.assertEqual(ResearchProfile.objects.count(), 0)
        if hasattr(account, 'user_ID'):
            self.assertIsNone(account.user_ID)

    def test_bad_login_and_safe_redirect(self):
        self.assertEqual(self.client.post('/cinema/login/', {'username': self.user.username, 'password': 'wrong'}).status_code, 200)
        self.assertNotIn('_auth_user_id', self.client.session)
        r = self.client.post('/cinema/login/?next=https://evil.example/', {'username': self.user.username, 'password': PASSWORD})
        self.assertEqual(r.url, '/cinema/personal/')

    def test_login_rotates_and_logout_invalidates_session(self):
        session = self.client.session
        session['before'] = 'value'
        session.save()
        old = session.session_key
        self.login()
        self.assertNotEqual(self.client.session.session_key, old)
        session_cookie = self.client.cookies['sessionid'].value
        self.client.post('/cinema/logout/')
        replay = Client()
        replay.cookies['sessionid'] = session_cookie
        self.assertEqual(replay.get('/cinema/personal/').status_code, 302)

    def test_anonymous_private_routes_redirect_without_writes(self):
        for path in ['/cinema/personal/', '/cinema/personal/preferences/', '/cinema/personal/interactions/', f'/cinema/personal/traces/{uuid4()}/']:
            self.assertEqual(self.client.get(path).status_code, 302)
        self.assertEqual(self.client.post('/cinema/personal/').status_code, 302)
        self.assertEqual(self.client.post('/cinema/personal/movies/10/interaction/', {'kind': 'watched'}).status_code, 302)
        self.assertEqual(MovieInteraction.objects.count(), 0)
        self.assertEqual(PersonalRecommendationTrace.objects.count(), 0)

    def test_preference_persists_across_new_login(self):
        self.login()
        self.assertEqual(self.save_preference().status_code, 302)
        self.client.post('/cinema/logout/')
        fresh = Client()
        self.login(fresh)
        r = fresh.get('/cinema/personal/preferences/')
        self.assertEqual(r.context['form'].initial['genres'], ['Action'])
        self.assertEqual(r.context['form'].initial['movie_ids'], '10')
        self.assertEqual(UserPreference.objects.get(user=self.user).catalogue_model, catalogue_label())

    def test_preference_invalid_movie_genre_and_limits(self):
        self.login()
        for data in [{'genres': ['Invented']}, {'movie_ids': '999'}, {'movie_ids': 'not-id'}, {'movie_ids': ','.join(['10']*21)}, {'movie_ids': '9'*80}]:
            self.assertEqual(self.save_preference(**data).status_code, 200)
        self.assertEqual(UserPreference.objects.count(), 0)

    def test_preference_update_empty_and_duplicates(self):
        self.login()
        self.save_preference(movie_ids='10,10', genres=['Action', 'Action'])
        self.assertEqual(UserPreference.objects.get(user=self.user).movie_ids, [10])
        self.assertEqual(UserPreference.objects.get(user=self.user).genres, ['Action'])
        self.save_preference(movie_ids='', genres=[])
        p = UserPreference.objects.get(user=self.user)
        self.assertEqual((p.genres, p.movie_ids), ([], []))

    def test_cross_account_preference_fields_cannot_target_other(self):
        self.login()
        self.save_preference(user_id=self.other.pk, user=self.other.pk)
        self.assertTrue(UserPreference.objects.filter(user=self.user).exists())
        self.assertFalse(UserPreference.objects.filter(user=self.other).exists())

    def test_exact_ranking_trace_and_detail_mapping(self):
        self.login()
        self.save_preference()
        trace, url = self.generate()
        data = trace.provenance
        self.assertEqual(data['identity_mode'], 'DJANGO_ACCOUNT')
        self.assertEqual(data['source'], 'METADATA_COLD_START')
        self.assertEqual(data['strategy'], 'GENRE_MATCH_V1')
        self.assertIsNone(data['model_name'])
        self.assertIsNone(data['checkpoint_identity'])
        self.assertIsNone(data['research_profile_id'])
        self.assertEqual(data['genre_weights'], {'Action': 3, 'Drama': 1})
        self.assertEqual([(r['movie_id'], r['score']) for r in data['results']], [(30, 3), (20, 1), (40, 0)])
        self.assertEqual(data['candidate_count'], 3)
        self.assertEqual(len(data['catalogue_metadata_sha256']), 64)
        self.assertContains(self.client.get(url), '匹配类型：Action')
        for row in data['results']:
            r = self.client.get(f"/cinema/movies/{row['movie_id']}/?personal_trace={trace.id}")
            self.assertEqual(r.status_code, 200)
        self.assertEqual(MovieInteraction.objects.filter(user=self.user, trace=trace, kind='view').count(), 3)

    def test_no_research_or_scoring_service_call(self):
        self.login()
        with patch('research_recommendation.views.service', side_effect=AssertionError('research accessed')):
            self.save_preference()
            self.generate()
        self.assertEqual(ResearchProfile.objects.count(), 0)

    def test_no_preferences_and_empty_catalogue(self):
        self.login()
        self.assertContains(self.client.get('/cinema/personal/'), '尚未设置偏好')
        trace, _ = self.generate()
        self.assertEqual(trace.provenance['strategy'], 'CATALOGUE_EXPLORATION_V1')
        self.assertEqual([r['movie_id'] for r in trace.provenance['results']], [10, 20, 30, 40])
        self.assertTrue(all(r['score'] == 0 for r in trace.provenance['results']))
        Movie.objects.all().delete()
        r = self.client.post('/cinema/personal/', follow=True)
        self.assertContains(r, '暂无可推荐电影')
        self.assertEqual(self.client.get('/cinema/personal/preferences/').status_code, 200)

    def test_all_selected_and_stale_selected_movies(self):
        self.login()
        self.save_preference(genres=[], movie_ids='10,20,30,40')
        trace, url = self.generate()
        self.assertEqual(trace.provenance['candidate_count'], 0)
        self.assertContains(self.client.get(url), '暂无可推荐电影')
        Movie.objects.filter(pk=10).delete()
        self.assertEqual(self.client.post('/cinema/personal/').status_code, 302)

    def test_trace_is_owned_and_immutable_after_preference_update(self):
        self.login()
        self.save_preference()
        trace, url = self.generate()
        before = trace.provenance
        self.save_preference(genres=['Comedy'], movie_ids='')
        trace.refresh_from_db()
        self.assertEqual(trace.provenance, before)
        second = Client()
        self.login(second, self.other)
        self.assertEqual(second.get(url).status_code, 404)
        self.assertEqual(second.get(f'/cinema/movies/30/?personal_trace={trace.id}').status_code, 404)
        self.assertEqual(second.post('/cinema/personal/movies/30/interaction/', {'kind': 'watched', 'personal_trace': str(trace.id)}).status_code, 404)
        self.assertFalse(MovieInteraction.objects.filter(user=self.other).exists())

    def test_trace_invalid_nonmember_and_deleted_film(self):
        self.login()
        self.save_preference()
        trace, url = self.generate()
        self.assertEqual(self.client.get('/cinema/movies/30/?personal_trace=bad').status_code, 400)
        for movie_id, value in [(30, str(uuid4())), (10, str(trace.id))]:
            self.assertEqual(self.client.get(f'/cinema/movies/{movie_id}/?personal_trace={value}').status_code, 404)
        Movie.objects.filter(pk=30).delete()
        self.assertEqual(self.client.get(url).status_code, 200)
        self.assertEqual(self.client.get('/cinema/movies/30/').status_code, 404)

    def test_interactions_persist_and_are_private(self):
        self.login()
        self.client.get('/cinema/movies/10/')
        self.client.post('/cinema/personal/movies/10/interaction/', {'kind': 'watched', 'user_id': self.other.pk})
        self.client.post('/cinema/logout/')
        fresh = Client()
        self.login(fresh)
        r = fresh.get('/cinema/personal/interactions/')
        self.assertContains(r, 'SYNTHETIC Action Seed', count=2)
        self.assertContains(r, '看过')
        self.assertEqual(r.headers['Cache-Control'].find('no-store') >= 0, True)
        second = Client()
        self.login(second, self.other)
        self.assertContains(second.get('/cinema/personal/interactions/'), '暂无交互记录')

    def test_interest_updates_preferences_and_next_ranking(self):
        self.login()
        for _ in range(2):
            self.assertEqual(self.client.post('/cinema/personal/movies/10/interaction/', {'kind': 'interested'}).status_code, 302)
        self.assertEqual(UserPreference.objects.get(user=self.user).movie_ids, [10])
        trace, _ = self.generate()
        self.assertEqual(trace.provenance['genre_weights'], {'Action': 1, 'Drama': 1})
        self.assertEqual([r['movie_id'] for r in trace.provenance['results']], [20, 30, 40])

    def test_interaction_rejects_invalid_method_kind_and_movie(self):
        self.login()
        self.assertEqual(self.client.get('/cinema/personal/movies/10/interaction/').status_code, 405)
        self.assertEqual(self.client.post('/cinema/personal/movies/10/interaction/', {'kind': 'fake'}).status_code, 400)
        self.assertEqual(self.client.post('/cinema/personal/movies/999/interaction/', {'kind': 'watched'}).status_code, 404)
        self.assertEqual(MovieInteraction.objects.count(), 0)

    def test_csrf_on_all_writes(self):
        csrf = Client(enforce_csrf_checks=True)
        self.assertEqual(csrf.post('/cinema/register/', {}).status_code, 403)
        self.assertEqual(csrf.post('/cinema/login/', {}).status_code, 403)
        csrf.force_login(self.user)
        for path in ['/cinema/personal/', '/cinema/personal/preferences/', '/cinema/personal/movies/10/interaction/', '/cinema/logout/']:
            self.assertEqual(csrf.post(path, {}).status_code, 403)
        self.assertEqual(MovieInteraction.objects.count(), 0)

    def test_get_recommendations_has_no_trace_writes(self):
        self.login()
        self.client.get('/cinema/personal/')
        self.client.get('/cinema/personal/preferences/')
        self.assertEqual(PersonalRecommendationTrace.objects.count(), 0)
        self.assertEqual(UserPreference.objects.count(), 0)

    def test_top20_and_determinism(self):
        Movie.objects.bulk_create([Movie(id=100+i, title=f'SYNTHETIC {i}', movielens_id=f'x{i}', genre_labels=['Action']) for i in range(30)])
        pref = UserPreference.objects.create(user=self.user, genres=['Action'], catalogue_model=catalogue_label())
        first, provenance = rank(catalogue(), pref)
        second, again = rank(catalogue(), pref)
        self.assertEqual(first, second)
        self.assertEqual(provenance, again)
        self.assertEqual(len(first), 20)
        self.assertEqual(len({r['movie'].id for r in first}), 20)

    def test_history_bound_and_html_escaping(self):
        self.login()
        Movie.objects.filter(pk=10).update(title='<script>bad62()</script>')
        MovieInteraction.objects.bulk_create([MovieInteraction(user=self.user, movie_id=10, kind='view', catalogue_model=catalogue_label()) for _ in range(55)])
        r = self.client.get('/cinema/personal/interactions/')
        self.assertEqual(len(r.context['entries']), 50)
        self.assertNotContains(r, '<script>bad62()</script>')
        self.assertContains(r, '&lt;script&gt;bad62()&lt;/script&gt;')

    def test_other_catalogue_preference_not_reinterpreted(self):
        pref = UserPreference.objects.create(user=self.user, genres=['Action'], movie_ids=[10], catalogue_model='other.Movie')
        _, provenance = rank(catalogue(), pref)
        self.assertEqual(provenance['strategy'], 'CATALOGUE_EXPLORATION_V1')
        self.assertEqual(provenance['preference']['movie_ids'], [])

    def test_recent_trace_history_persists_and_is_private(self):
        self.login()
        trace, url = self.generate()
        self.client.post('/cinema/logout/')
        fresh = Client()
        self.login(fresh)
        self.assertContains(fresh.get('/cinema/personal/'), url)
        self.login(self.client, self.other)
        self.assertNotContains(self.client.get('/cinema/personal/'), str(trace.pk))

    def test_storage_failure_reports_503_without_fake_result(self):
        self.login()
        with patch('research_recommendation.personal_views.catalogue', side_effect=OperationalError('private detail')):
            self.assertEqual(self.client.get('/cinema/personal/preferences/').status_code, 503)
            self.assertEqual(self.client.post('/cinema/personal/').status_code, 503)
        self.assertEqual(PersonalRecommendationTrace.objects.count(), 0)
