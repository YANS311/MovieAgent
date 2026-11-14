from django.urls import path
from . import product_views as v
urlpatterns=[path('',v.home,name='home'),path('rankings/',v.rankings,name='rankings'),path('assistant/',v.assistant,name='assistant'),path('depot/',v.browse,name='depot'),
 path('login/',v.account_login,name='login'),path('register/',v.register,name='register'),path('logout/',v.account_logout,name='logout'),
 path('recommend/',v.recommendation_page,name='recommend'),path('api/recommendations/',v.recommendation_api,name='pamk_recommendation'),
 path('movie/<int:pk>/',v.movie_detail,name='movie_detail'),path('movie/<int:pk>/interaction/',v.interaction,name='movie_interaction'),
 path('preferences/',v.preferences,name='preferences'),path('research/movies/<int:canonical_id>/',v.research_detail,name='research_movie'),
 path('recommendation-traces/<uuid:trace_id>/',v.trace,name='recommendation_trace'),path('recommendation-traces/<uuid:trace_id>/view/',v.trace_page,name='trace_page'),path('media/posters/<str:filename>',v.local_poster,name='local_poster')]

# Local release is loopback-only; serve app-local assets without a production static server.
from django.urls import re_path
from django.contrib.staticfiles.views import serve as static_serve
urlpatterns += [re_path(r'^static/(?P<path>.*)$',static_serve,{'insecure':True})]

from . import completion_views as c
urlpatterns += [path('api/catalog/',c.catalogue_api),path('api/catalog/identities/',c.identities_api),path('api/catalog/resolve/',c.resolve_api),path('api/catalog/feedback/',c.feedback_api),path('research/movies/<int:canonical_id>/interaction/',c.canonical_interaction),path('history/',c.history,name='history'),path('api/agent/',c.agent_api,name='agent_api'),path('api/agent/tools/',c.tools_api)]
