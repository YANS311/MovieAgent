from django.urls import include, path
from django.views.generic.base import RedirectView
urlpatterns = [path('', RedirectView.as_view(pattern_name='research_pages:home', permanent=False)),
               path('cinema/', include('research_recommendation.page_urls')),
               path('api/research/', include('research_recommendation.urls'))]
