"""Isolated URLconf: never imports myapp.urls or its model warm-up graph."""
from django.urls import include, path

urlpatterns = [path('api/research/', include('research_recommendation.urls'))]
