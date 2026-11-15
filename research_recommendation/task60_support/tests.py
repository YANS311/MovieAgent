import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from django.contrib.auth import get_user_model
from django.test import TestCase, Client, override_settings
from django.urls import reverse
from research_recommendation.contracts import M3L, ML1M
from research_recommendation.models import ProfileGrant, MovieIDBridge, ResearchProfile
from research_recommendation.services import ProfileService
from research_recommendation.page_catalogue import safe_poster
from .fixtures import test_service
from .models import Movie


class PageTests(TestCase):
    def setUp(self):
        self.owner = get_user_model().objects.create_user(username='page-owner', password='test-only')
        self.ordinary = get_user_model().objects.create_user(username='page-ordinary', password='test-only')
        self.app = test_service()
        self.profile = ProfileService.register(self.app.profile_manifests[M3L], '0', 'TEST_DOUBLE研究画像')
        self.mlprofile = ProfileService.register(self.app.profile_manifests[ML1M], '7', 'TEST_DOUBLE ML1M')
        ProfileGrant.objects.create(user=self.owner, profile=self.profile)
        ProfileGrant.objects.create(user=self.owner, profile=self.mlprofile)
        Movie.objects.create(id=900, title='TEST_DOUBLE Toy Film', movielens_id='100',
                             genre_labels=['测试类型'], poster='/static/research_recommendation/test-poster.svg', summary='合成测试简介')
        Movie.objects.create(id=901, title='TEST_DOUBLE Another Film', movielens_id='101')
        self.mapping = self.app.mapping_services['UKMAP']
        self.evidence = {'catalogue_sha256': self.mapping.manifest.identity,
                         'source_namespace': 'movielens:movieId', 'review_reference': 'SYNTHETIC_PAGE_TEST_ONLY'}
        self.mapping.record('0', '100', 900, self.evidence)

    def post(self, **kw):
        data = {'model_name': 'UKMAP', 'profile_id': str(self.profile.pk), 'top_k': 3}
        data.update(kw)
        return self.client.post('/api/research/recommendations/', data=json.dumps(data), content_type='application/json')

    def test_home_and_catalogue_are_real_django_pages(self):
        r = self.client.get('/cinema/')
        self.assertContains(r, 'TEST_DOUBLE Toy Film')
        self.assertContains(r, 'TEST_DOUBLE · 页面验收模式')
        self.assertContains(r, 'PENDING_REAL_MODEL_E2E')
        self.assertEqual(self.client.get('/cinema/movies/').status_code, 200)

    def test_detail_fields_and_optional_poster(self):
        r = self.client.get('/cinema/movies/900/')
        for value in ['TEST_DOUBLE Toy Film', '合成测试简介', '测试类型', 'test-poster.svg']:
            self.assertContains(r, value)

    def test_unknown_movie_404(self):
        self.assertEqual(self.client.get('/cinema/movies/999999/').status_code, 404)

    def test_empty_metadata_is_not_fabricated(self):
        r = self.client.get('/cinema/movies/901/')
        for value in ['暂无海报', '类型暂缺', '当前数据源未提供简介']:
            self.assertContains(r, value)

    def test_search_is_bounded_and_escaped(self):
        self.assertContains(self.client.get('/cinema/movies/?q=Toy'), 'TEST_DOUBLE Toy Film')
        self.assertContains(self.client.get('/cinema/movies/?q=not-existing'), '没有找到相关电影')
        self.assertEqual(self.client.get('/cinema/movies/', {'q': 'x'*101}).status_code, 400)
        self.assertEqual(self.client.get('/cinema/movies/', {'q': "' OR 1=1"}).context['page'].paginator.count, 0)

    def test_pagination_handles_bad_page(self):
        for page in ['bad', '-1', '99999']:
            self.assertEqual(self.client.get('/cinema/movies/', {'page': page}).status_code, 200)

    def test_movie_html_is_escaped(self):
        Movie.objects.filter(pk=900).update(title='<script>bad()</script>')
        r = self.client.get('/cinema/movies/900/')
        self.assertContains(r, '&lt;script&gt;bad()&lt;/script&gt;')
        self.assertNotContains(r, '<script>bad()</script>')

    def test_unsafe_poster_schemes_rejected(self):
        for value in ['javascript:bad()', 'data:text/html,evil', '//other/evil', 'https://user:pass@example.org/a']:
            self.assertIsNone(safe_poster(value))
        self.assertEqual(safe_poster('/media/poster.jpg'), '/media/poster.jpg')

    def test_sensitive_movie_poster_hidden(self):
        from types import SimpleNamespace
        from research_recommendation.page_catalogue import PageCatalogue
        card = PageCatalogue.card(SimpleNamespace(pk=1, title='test', genres=None, genre_labels=[],
                                  poster='https://example.org/a', is_sensitive=True))
        self.assertIsNone(card.poster_url)

    def test_recommend_requires_login(self):
        r = self.client.get('/cinema/recommend/')
        self.assertEqual(r.status_code, 302)
        self.assertIn('/cinema/login/', r.url)

    def test_granted_profiles_and_dataset_labels(self):
        self.client.force_login(self.owner)
        r = self.client.get('/cinema/recommend/')
        self.assertContains(r, str(self.profile.pk))
        self.assertContains(r, 'M3L_AUTHOR_ALIGNED_V1')
        self.assertContains(r, 'ML1M_CH3_V1')
        self.assertContains(r, '/api/research/recommendations/')

    def test_ordinary_account_has_no_implicit_profile(self):
        self.client.force_login(self.ordinary)
        self.assertContains(self.client.get('/cinema/recommend/'), '没有获授权的研究画像')
        self.assertEqual(self.post().status_code, 404)
        self.assertEqual(ResearchProfile.objects.count(), 2)

    def test_staff_is_not_an_implicit_grant(self):
        self.ordinary.is_staff = True
        self.ordinary.save()
        self.client.force_login(self.ordinary)
        self.assertContains(self.client.get('/cinema/recommend/'), '没有获授权的研究画像')

    def test_topk_contract_and_unique_mapping(self):
        self.client.force_login(self.owner)
        body = self.post(top_k=1).json()
        self.assertEqual(body['inference_mode'], 'TEST_DOUBLE')
        self.assertEqual(body['recommendations'][0]['rank'], 1)
        self.assertEqual(body['recommendations'][0]['django_movie_id'], 900)
        self.assertEqual(body['recommendations'][0]['mapping_status'], 'VERIFIED_UNIQUE')

    def test_ambiguous_and_unmapped_remain_null(self):
        self.mapping.record('1', '101', 901, self.evidence)
        self.mapping.record('1', '101', 901, self.evidence)
        self.client.force_login(self.owner)
        rows = self.post().json()['recommendations']
        self.assertEqual(rows[1]['mapping_status'], 'AMBIGUOUS')
        self.assertIsNone(rows[1]['django_movie_id'])
        self.assertEqual(rows[2]['mapping_status'], 'UNMAPPED')
        self.assertIsNone(rows[2]['django_movie_id'])

    @override_settings(TASK57_SERVICE_FACTORY='research_recommendation.task60_support.fixtures.unavailable_service')
    def test_model_unavailable_503(self):
        self.client.force_login(self.owner)
        r = self.post()
        self.assertEqual(r.status_code, 503)
        self.assertNotIn('recommendations', r.json())

    @override_settings(TASK57_SERVICE_FACTORY='research_recommendation.task60_support.fixtures.empty_service')
    def test_empty_result_is_not_filled(self):
        self.client.force_login(self.owner)
        self.assertEqual(self.post().json()['recommendations'], [])

    def test_cross_dataset_and_invalid_topk(self):
        self.client.force_login(self.owner)
        self.assertEqual(self.post(profile_id=str(self.mlprofile.pk)).status_code, 409)
        self.assertEqual(self.post(top_k=101).status_code, 400)

    def test_real_login_does_not_create_profile_or_grant(self):
        r = self.client.post('/cinema/login/', {'username': self.ordinary.username, 'password': 'test-only'})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(ProfileGrant.objects.filter(user=self.ordinary).count(), 0)
        self.assertContains(self.client.get('/cinema/recommend/'), '没有获授权的研究画像')

    def test_login_external_redirect_not_allowed(self):
        r = self.client.post('/cinema/login/?next=https://evil.invalid/', {'username': self.owner.username, 'password': 'test-only'})
        self.assertEqual(r.status_code, 302)
        self.assertFalse(r.url.startswith('https://evil'))

    def test_csrf_is_required(self):
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.owner)
        self.assertEqual(client.post('/api/research/recommendations/', data='{}', content_type='application/json').status_code, 403)

    def test_movie_metadata_endpoint_bounded(self):
        r = self.client.get('/cinema/metadata/?ids=900,901')
        self.assertEqual(r.json()['movies']['900']['genres'], ['测试类型'])
        self.assertEqual(self.client.get('/cinema/metadata/?ids=javascript:1').status_code, 400)
        self.assertEqual(self.client.get('/cinema/metadata/', {'ids': ','.join(['1']*101)}).status_code, 400)

    @override_settings(TASK60_MOVIE_MODEL='not_installed.Missing')
    def test_missing_catalogue_returns_503(self):
        self.assertEqual(self.client.get('/cinema/').status_code, 503)

    def test_agent_entry_does_not_run_agent(self):
        self.assertContains(self.client.get('/cinema/assistant/'), '原系统的Agent功能保持原状')

    def test_historical_trace_identity_and_permission(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'history.json'
            path.write_text(json.dumps({'inference_mode': 'REAL_ONLINE_INFERENCE', 'recommendations': [],
                'checkpoint_identity': 'a'*64, 'candidate_scope': 'SYNTHETIC_HISTORY_UNIT_TEST'}))
            identity = hashlib.sha256(path.read_bytes()).hexdigest()
            with override_settings(TASK60_HISTORICAL_TRACE=(path, identity)):
                self.client.force_login(self.ordinary)
                self.assertEqual(self.client.get('/cinema/history/task58/').status_code, 403)
                self.client.force_login(self.owner)
                self.assertContains(self.client.get('/cinema/history/task58/'), '不是本轮实时推荐')
                path.write_text('{}')
                self.assertEqual(self.client.get('/cinema/history/task58/').status_code, 503)

                path.unlink()
                self.assertEqual(self.client.get('/cinema/history/task58/').status_code, 503)
            path.write_text('not-json')
            with override_settings(TASK60_HISTORICAL_TRACE=(path, hashlib.sha256(path.read_bytes()).hexdigest())):
                self.assertEqual(self.client.get('/cinema/history/task58/').status_code, 503)
