"""
URL configuration for movie project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,include
from django.conf import settings

urlpatterns = [
    path('admin/', admin.site.urls),
    path('select2/', include('django_select2.urls')),
    path('', include('research_recommendation.product_urls' if getattr(settings, 'PAMK_PRODUCT_ENABLED', False) else 'myapp.urls'))
]


if getattr(settings, 'TASK57_RECOMMENDATION_MVP_ENABLED', False):
    urlpatterns += [path('api/research/', include('research_recommendation.urls'))]

# Opt-in Task60 pages; existing authentication, Agent and API routes remain.
if getattr(settings, 'TASK60_MOVIE_PAGES_ENABLED', False):
    urlpatterns += [path('cinema/', include('research_recommendation.page_urls'))]
