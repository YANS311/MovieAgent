from django.apps import AppConfig


class ResearchRecommendationConfig(AppConfig):
    name = 'research_recommendation'
    default_auto_field = 'django.db.models.BigAutoField'

    def ready(self):
        from django.conf import settings
        if getattr(settings, 'TASK63_CACHE_ENABLED', False):
            from .personal_cache import connect_catalogue_signals
            connect_catalogue_signals()
