"""Local-only final release. Configuration is explicit; production DB denied."""
import os
from pathlib import Path
BASE_DIR=Path(__file__).resolve().parent.parent
SECRET_KEY=os.environ['DJANGO_SECRET_KEY']
DEBUG=False
ALLOWED_HOSTS=['127.0.0.1','localhost','testserver']
PAMK_PRODUCT_ENABLED=True
PAMK_INDUCTIVE_AVAILABLE=True
PAMK_INDUCTIVE_ENABLED=os.environ.get('PAMK_INDUCTIVE_ENABLED','1')=='1'
PAMK_INDUCTIVE_STRATEGY=os.environ.get('PAMK_INDUCTIVE_STRATEGY','mean')
if PAMK_INDUCTIVE_STRATEGY not in ('mean','learned'):raise RuntimeError('INVALID_ENCODING_STRATEGY')
PAMK_MAPPING_VERSION=os.environ.get('PAMK_MAPPING_VERSION','ch5-exact-tmdb-v2')
PAMK_INDUCTIVE_MANIFEST=os.environ['PAMK_INDUCTIVE_MANIFEST']
PAMK_EVIDENCE_DIR=os.environ['PAMK_EVIDENCE_DIR']
TASK57_RECOMMENDATION_MVP_ENABLED=False
INSTALLED_APPS=['django.contrib.auth','django.contrib.contenttypes','django.contrib.sessions','django.contrib.messages','django.contrib.staticfiles','myapp','research_recommendation']
MIDDLEWARE=['research_recommendation.product_middleware.PrivateProductResponseMiddleware','django.middleware.security.SecurityMiddleware','django.contrib.sessions.middleware.SessionMiddleware','django.middleware.common.CommonMiddleware','django.middleware.csrf.CsrfViewMiddleware','django.contrib.auth.middleware.AuthenticationMiddleware','django.contrib.messages.middleware.MessageMiddleware','django.middleware.clickjacking.XFrameOptionsMiddleware']
ROOT_URLCONF='research_recommendation.product_urls'
TEMPLATES=[{'BACKEND':'django.template.backends.django.DjangoTemplates','DIRS':[],'APP_DIRS':True,'OPTIONS':{'context_processors':['django.template.context_processors.request','django.contrib.auth.context_processors.auth','django.contrib.messages.context_processors.messages']}}]
AUTH_USER_MODEL='myapp.UserInfo'
name=os.environ['PAMK_DATABASE_NAME']
if not name.startswith('ch5_pamk_'):raise RuntimeError('ISOLATED_DATABASE_REQUIRED')
DATABASES={'default':{'ENGINE':'django.db.backends.mysql','NAME':name,'USER':os.environ['DB_USER'],'PASSWORD':os.environ['DB_PASSWORD'],'HOST':os.environ.get('DB_HOST','127.0.0.1'),'PORT':os.environ.get('DB_PORT','3306'),'OPTIONS':{'charset':'utf8mb4'}}}
SESSION_ENGINE='django.contrib.sessions.backends.db'
CACHES={'default':{'BACKEND':'django_redis.cache.RedisCache','LOCATION':os.environ.get('PAMK_REDIS_URL','redis://127.0.0.1:6379/15'),'KEY_PREFIX':name,'TIMEOUT':120,'OPTIONS':{'CLIENT_CLASS':'django_redis.client.DefaultClient','SOCKET_CONNECT_TIMEOUT':1,'SOCKET_TIMEOUT':1}}}
AUTH_PASSWORD_VALIDATORS=[{'NAME':'django.contrib.auth.password_validation.MinimumLengthValidator'}]
LANGUAGE_CODE='zh-hans';TIME_ZONE='Asia/Singapore';USE_TZ=True;USE_I18N=True
STATIC_URL='/static/';MEDIA_URL='/media/';MEDIA_ROOT=BASE_DIR/'media'
PAMK_POSTER_ROOT=Path(os.environ.get('PAMK_POSTER_ROOT', str(MEDIA_ROOT/'posters')))
DEFAULT_AUTO_FIELD='django.db.models.BigAutoField';LOGIN_URL='/login/'
REDIS_ANALYTICS_ENABLED=False
LOGGING={'version':1,'disable_existing_loggers':False,'handlers':{'console':{'class':'logging.StreamHandler'}},'loggers':{'django.server':{'handlers':['console'],'level':'INFO','propagate':False},'django.request':{'handlers':['console'],'level':'ERROR','propagate':False}}}

# Local structured tool calling, with explicit model identity and bounded execution.
OLLAMA_BASE_URL=os.environ.get('OLLAMA_BASE_URL','http://127.0.0.1:11434')
AGENT_LLM_MODEL=os.environ.get('AGENT_LLM_MODEL','qwen3:4b-instruct')
AGENT_LLM_DIGEST=os.environ.get('AGENT_LLM_DIGEST','0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0')
AGENT_LLM_TIMEOUT=float(os.environ.get('AGENT_LLM_TIMEOUT','45'))
