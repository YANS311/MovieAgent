"""SQLite memory and TEST_DOUBLE only. No production app/model import chain."""
import json
import os
import sys
import tempfile
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault('PAMK_WORKER_PYTHON', sys.executable)
if 'PAMK_ASSET_CONFIG' not in os.environ:
    _tmp_asset_cfg = Path(tempfile.gettempdir()) / 'movieagent_task60_synthetic_pamk_asset.json'
    _tmp_asset_cfg.write_text(json.dumps({
        'checkpoint': {
            'sha256': 'f098c45c993ff10d365a5dc8b2314ef816a465af867861b691f444a719c81088',
            'path': '/tmp/unused_synthetic_checkpoint.pt',
        }
    }))
    os.environ['PAMK_ASSET_CONFIG'] = str(_tmp_asset_cfg)
SECRET_KEY = 'task60-isolated-only-not-for-deployment'
DEBUG = True
ALLOWED_HOSTS = ['testserver', 'localhost', '127.0.0.1']
INSTALLED_APPS = ['django.contrib.auth', 'django.contrib.contenttypes', 'django.contrib.sessions',
                  'django.contrib.staticfiles', 'research_recommendation',
                  'research_recommendation.task60_support']
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
ROOT_URLCONF = 'movie.task60_urls'
MIDDLEWARE = ['django.contrib.sessions.middleware.SessionMiddleware',
              'django.middleware.csrf.CsrfViewMiddleware',
              'django.contrib.auth.middleware.AuthenticationMiddleware']
TEMPLATES = [{'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'], 'APP_DIRS': True,
    'OPTIONS': {'context_processors': ['django.template.context_processors.request',
                                     'django.contrib.auth.context_processors.auth']}}]
LOGIN_URL = 'research_pages:login'
LOGIN_REDIRECT_URL = 'research_pages:recommend'
CSRF_FAILURE_VIEW = 'research_recommendation.views.csrf_failure'
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
PASSWORD_HASHERS = ['django.contrib.auth.hashers.MD5PasswordHasher']
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
USE_TZ = True
STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
TASK57_SERVICE_FACTORY = 'research_recommendation.task60_support.fixtures.test_service'
TASK60_MOVIE_MODEL = 'task60_snapshot.Movie'
TASK60_CPU_DEMO = True
TASK60_USE_EXISTING_AUTH = False
TASK60_HISTORICAL_TRACE = None
TASK60_HISTORY_PROFILE = ('M3L_AUTHOR_ALIGNED_V1', '0')
DATA_UPLOAD_MAX_MEMORY_SIZE = 8192
