"""Synthetic in-memory SQLite only. Never import movie.settings/.env/myapp."""
SECRET_KEY = 'task57-isolated-synthetic-test-key-not-for-deployment'
DEBUG = False
ALLOWED_HOSTS = ['testserver', 'localhost']
INSTALLED_APPS = ['django.contrib.auth', 'django.contrib.contenttypes',
                  'django.contrib.sessions', 'research_recommendation',
                  'research_recommendation.test_support']
DATABASES = {'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}}
ROOT_URLCONF = 'movie.task57_urls'
MIDDLEWARE = ['django.contrib.sessions.middleware.SessionMiddleware',
              'django.middleware.csrf.CsrfViewMiddleware',
              'django.contrib.auth.middleware.AuthenticationMiddleware']
CSRF_FAILURE_VIEW = 'research_recommendation.views.csrf_failure'
SESSION_ENGINE = 'django.contrib.sessions.backends.db'
PASSWORD_HASHERS = ['django.contrib.auth.hashers.MD5PasswordHasher']
TASK57_SERVICE_FACTORY = 'research_recommendation.test_support.fixtures.test_service'
DATA_UPLOAD_MAX_MEMORY_SIZE = 8192
DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
USE_TZ = True
# No production model, Redis, Neo4j, FAISS, environment or ASGI app is imported.
