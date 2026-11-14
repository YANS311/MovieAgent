from django.urls import path
from . import views

app_name = 'research_recommendation'
urlpatterns = [
    path('profiles/', views.profiles, name='profiles'),
    path('recommendations/', views.recommend, name='recommendations'),
    path('traces/<uuid:trace_id>/', views.trace_detail, name='trace'),
]
