from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path

from myapp import views
from myapp import agent_views
from myapp import evaluate_views

urlpatterns = [

    # 后台首页 (Dashboard)
    path('admin_panel/', views.admin_index, name='admin_index'),

    # 电影管理 (CRUD)
    path('admin_panel/movies/', views.admin_movie_list, name='admin_movie_list'),
    path('admin_panel/movie/add/', views.admin_movie_add, name='admin_movie_add'),
    path('admin_panel/movie/<int:pk>/edit/', views.admin_movie_edit, name='admin_movie_edit'),
    path('admin_panel/movie/<int:pk>/delete/', views.admin_movie_delete, name='admin_movie_delete'),

    # 用户管理 (CRUD)
    path('admin_panel/user/add/', views.admin_user_add, name='admin_user_add'),
    path('admin_panel/user/<int:pk>/edit/', views.admin_user_edit, name='admin_user_edit'),
    path('admin_panel/user/<int:pk>/reset_password/', views.admin_user_reset_password, name='admin_user_reset_password'),
    path('admin_panel/users/', views.admin_user_list, name='admin_user_list'),
    # (编辑/重置密码比较复杂，我们先做删除)
    path('admin_panel/user/<int:pk>/delete/', views.admin_user_delete, name='admin_user_delete'),
    path('admin_panel/comments/', views.admin_comments, name='admin_comments'),
    path('admin_panel/comments/delete/<int:rating_id>/', views.admin_comment_delete, name='admin_comment_delete'),

    path('admin_panel/actors/', views.admin_actor_list, name='admin_actor_list'),
    path('admin_panel/actor/add/', views.admin_actor_add, name='admin_actor_add'),
    path('admin_panel/actor/<int:pk>/edit/', views.admin_actor_edit, name='admin_actor_edit'),
    path('admin_panel/actor/<int:pk>/delete/', views.admin_actor_delete, name='admin_actor_delete'),
    path('admin_panel/directors/', views.admin_director_list, name='admin_director_list'),
    path('login/', views.login_user, name='login_user'),
    path('register/', views.register, name='register'),
    path('logout/', views.logout_user, name='logout_user'),
    path('',views.front_index,name="front_index"),

    #path('front_index/', views.front_index, name='front_index'),

    # --- ↓↓↓ 添加新页面的路由 ↓↓↓ ---
    path('rank/', views.rank, name='rank'),
    path('depot/', views.depot, name='depot'),
    path('recommend/', views.recommend, name='recommendations'),
    path('recommend/explain/', views.ajax_explain_rec, name='ajax_explain_rec'),

    path('movie/<int:pk>/', views.movie_detail, name='movie_detail'),
    path('movie/score/', views.score_movie, name='score_movie'),
    path('movie/<int:pk>/collect/add/', views.add_collect, name='add_collect'),
    path('movie/<int:pk>/collect/remove/', views.remove_collect, name='remove_collect'),

    path('admin_panel/kg/', views.admin_kg_view, name='admin_kg_view'),
    path('admin_panel/kg/data/', views.api_kg_data, name='api_kg_data'),

    # ═══════════════════════════════════════════════════
    #  Agent 前端 - 知识图谱可视化 (KAG)
    # ═══════════════════════════════════════════════════
    path('agent/kg/', agent_views.agent_kg_view, name='agent_kg_view'),
    path('agent/kg/query/', agent_views.ajax_agent_kg_query, name='ajax_agent_kg_query'),

    # --- 管理员高级运维 ---
    path('admin_panel/trigger_train/', views.admin_trigger_train, name='admin_trigger_train'),
    path('admin_panel/clear_cache/', views.admin_clear_cache, name='admin_clear_cache'),
    path('admin_panel/api/user_stats/', views.admin_user_stats, name='admin_user_stats'),

    path('search/', views.search_results, name='search_results'),
    path('center/', views.center, name='center'),

    path('ajax_collect/', views.ajax_collect, name='ajax_collect'),
    # 聊天页面和 API 路由（已合并至 Agent 系统）
    path('ajax/explain_rec/', views.ajax_explain_rec, name='ajax_explain_rec'),
    path('ajax/kg_path/', views.ajax_kg_path, name='ajax_kg_path'),
    # /chat/ → Agent 智能推荐（论文级 ReAct 范式）
    path('chat/', agent_views.chat_recommend_view, name='chat_view'),
    path('chat/api/', agent_views.agent_api_view, name='ajax_chat'),
    path('chat/clear/', views.chat_clear_history, name='chat_clear_history'),

    path('search/visual/', views.search_visual, name='search_visual'),

    # ═══════════════════════════════════════════════════
    #  Agent 智能推荐系统 (论文级新增路由)
    # ═══════════════════════════════════════════════════

    # 智能推荐聊天页 (Agent)
    path('agent/chat/', agent_views.chat_recommend_view, name='agent_chat'),

    # Agent API 接口 (ReAct推理 - JSON)
    path('agent/api/', agent_views.agent_api_view, name='agent_api'),

    # Agent SSE 流式接口（打字机效果）
    path('agent/stream/', agent_views.agent_stream_view, name='agent_stream'),

    # 电影推荐解释页
    path('movie/<int:pk>/explain/', agent_views.movie_explain_view, name='movie_explain'),

    # 推荐反馈收集
    path('agent/feedback/', agent_views.recommend_feedback_view, name='recommend_feedback'),

    # Agent推理链展示
    path('agent/trace/', agent_views.agent_trace_view, name='agent_trace_list'),
    path('agent/trace/<int:trace_id>/', agent_views.agent_trace_view, name='agent_trace_detail'),

    # 对话管理 API（清除/新建/编辑/历史/画像总结）
    path('agent/clear_chat/', agent_views.ajax_clear_chat, name='ajax_clear_chat'),
    path('agent/new_chat/', agent_views.ajax_new_chat, name='ajax_new_chat'),
    path('agent/edit_last/', agent_views.ajax_edit_last, name='ajax_edit_last'),
    path('agent/chat_history/', agent_views.ajax_chat_history, name='ajax_chat_history'),
    path('agent/summarize_profile/', agent_views.ajax_summarize_profile, name='ajax_summarize_profile'),

    # 系统健康检查 API（GPU/Ollama/FAISS/Neo4j 状态）
    path('agent/health/', agent_views.ajax_system_health, name='ajax_system_health'),

    # Agent Debug 模式 API
    path('agent/debug/', agent_views.ajax_agent_debug, name='ajax_agent_debug'),

    # ═══════════════════════════════════════════════════
    #  "不喜欢"排除列表（全局生效）
    # ═══════════════════════════════════════════════════
    path('ajax/exclude/add/', views.ajax_exclude_add, name='ajax_exclude_add'),
    path('ajax/exclude/remove/', views.ajax_exclude_remove, name='ajax_exclude_remove'),
    path('ajax/exclude/list/', views.ajax_exclude_list, name='ajax_exclude_list'),

    # ═══════════════════════════════════════════════════
    #  管理员后台 - 新模型 CRUD
    # ═══════════════════════════════════════════════════

    # 用户画像管理 (UserProfile)
    path('admin_panel/profiles/', agent_views.admin_user_profile_list, name='admin_user_profile_list'),
    path('admin_panel/profile/<int:pk>/edit/', agent_views.admin_user_profile_edit, name='admin_user_profile_edit'),
    path('admin_panel/profile/<int:pk>/delete/', agent_views.admin_user_profile_delete, name='admin_user_profile_delete'),

    # Agent推理链管理 (AgentTrace)
    path('admin_panel/traces/', agent_views.admin_agent_trace_list, name='admin_agent_trace_list'),
    path('admin_panel/trace/<int:pk>/', agent_views.admin_agent_trace_detail, name='admin_agent_trace_detail'),
    path('admin_panel/trace/<int:pk>/delete/', agent_views.admin_agent_trace_delete, name='admin_agent_trace_delete'),

    # 用户反馈管理 (UserFeedback)
    path('admin_panel/feedbacks/', agent_views.admin_user_feedback_list, name='admin_user_feedback_list'),
    path('admin_panel/feedback/<int:pk>/delete/', agent_views.admin_user_feedback_delete, name='admin_user_feedback_delete'),

    # ═══════════════════════════════════════════════════
    #  离线评估接口 (论文实验数据)
    # ═══════════════════════════════════════════════════

    # 评估总览页
    path('evaluate/', evaluate_views.evaluate_index, name='evaluate_index'),

    # 各项指标接口
    path('evaluate/auc/', evaluate_views.evaluate_auc, name='evaluate_auc'),
    path('evaluate/ndcg/', evaluate_views.evaluate_ndcg, name='evaluate_ndcg'),
    path('evaluate/hr/', evaluate_views.evaluate_hr, name='evaluate_hr'),
    path('evaluate/mrr/', evaluate_views.evaluate_mrr, name='evaluate_mrr'),

    # 完整评估
    path('evaluate/full/', evaluate_views.evaluate_full, name='evaluate_full'),
]

# 在开发环境下服务媒体文件
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)