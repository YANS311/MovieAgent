from django.urls import path
from . import page_views as views
from . import personal_views as personal
from .explanation_views import explain
from .trace_pages import recommendation_record

app_name = 'research_pages'
urlpatterns = [path('', views.home, name='home'),
    path('explain/', explain, name='explain'),
    path('register/', personal.register, name='register'),
    path('personal/', personal.recommendations, name='personal'),
    path('personal/preferences/', personal.preferences, name='preferences'),
    path('personal/traces/<uuid:trace_id>/', personal.trace_detail, name='personal_trace'),
    path('personal/interactions/', personal.interactions, name='interactions'),
    path('personal/movies/<int:movie_id>/interaction/', personal.interaction, name='interaction'),
    path('movies/', views.movies, name='movies'),
    path('movies/<int:movie_id>/', views.detail, name='detail'),
    path('metadata/', views.card_metadata, name='metadata'),
    path('recommend/', views.recommend, name='recommend'),
    path('recommend/records/<uuid:trace_id>/', recommendation_record, name='recommendation_record'),
    path('history/task58/', views.historical_trace, name='history'),
    path('login/', views.PageLogin.as_view(), name='login'),
    path('logout/', views.PageLogout.as_view(), name='logout'),
    path('assistant/', views.assistant, name='assistant')]
