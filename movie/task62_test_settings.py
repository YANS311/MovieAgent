"""Isolated CPU tests with real Django auth and synthetic metadata only."""
from .task60_test_settings import *
TASK62_NATIVE_AUTH = True
AUTH_PASSWORD_VALIDATORS = [
    {'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator', 'OPTIONS': {'min_length': 10}},
    {'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator'},
]
