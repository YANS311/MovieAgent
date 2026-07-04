# 文件: myapp/admin.py

from django.contrib import admin
from .models import UserInfo, Movie, Genre, Actor, Region, UserRating, Collect, Rec, ChatHistory


# 1. 注册 UserInfo (修复版)
class UserInfoAdmin(admin.ModelAdmin):
    # --- ↓↓↓ 关键修复 ↓↓↓ ---
    # 原来的 user_type -> 改为 is_staff (是否管理员)
    # 原来的 user_status -> 改为 is_active (是否有效)
    # 我们也加上了 occupation (职业), 因为这是你刚加的新字段
    list_display = ('id', 'username', 'email', 'is_staff', 'is_active', 'occupation')

    search_fields = ('username', 'email')

    # 过滤器也同步修改
    list_filter = ('is_staff', 'is_active', 'occupation')
    # --- ↑↑↑ 修复结束 ↑↑↑ ---


admin.site.register(UserInfo, UserInfoAdmin)


# 2. 注册 Movie
class MovieAdmin(admin.ModelAdmin):
    # 确保这里没有引用不存在的字段
    list_display = ('id', 'title', 'score', 'vote_count', 'date')
    search_fields = ('title',)
    list_filter = ('date',)


admin.site.register(Movie, MovieAdmin)


# 3. 注册 UserRating
class UserRatingAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'movie', 'score', 'discussion', 'comment_time')
    search_fields = ('user__username', 'movie__title', 'discussion')
    list_filter = ('score', 'comment_time')


admin.site.register(UserRating, UserRatingAdmin)


# 4. 注册 Collect
class CollectAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'movie', 'collect_time')
    search_fields = ('user__username', 'movie__title')


admin.site.register(Collect, CollectAdmin)


# 5. 注册 Rec (推荐结果)
class RecAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'movie', 'rating')
    search_fields = ('user__username', 'movie__title')


admin.site.register(Rec, RecAdmin)

# 6. 注册其他基础表
admin.site.register(Genre)
admin.site.register(Actor)
admin.site.register(Region)
admin.site.register(ChatHistory)


# ═══════════════════════════════════════════
# 7. 注册论文级新增模型
# ═══════════════════════════════════════════

from .models_upgrade import (
    ConversationHistory, UserProfile, RecommendLog, AgentTrace, UserFeedback
)


@admin.register(ConversationHistory)
class ConversationHistoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'session_id', 'role', 'intent', 'latency_ms', 'timestamp')
    search_fields = ('user__username', 'message', 'session_id')
    list_filter = ('role', 'intent', 'timestamp')


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'total_ratings', 'avg_score', 'total_collections', 'is_cold_start', 'updated_at')
    search_fields = ('user__username',)
    list_filter = ('is_cold_start',)


@admin.register(RecommendLog)
class RecommendLogAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'strategy', 'latency_ms', 'created_at')
    search_fields = ('user__username', 'query_text')
    list_filter = ('strategy', 'created_at')


@admin.register(AgentTrace)
class AgentTraceAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'intent', 'total_latency_ms', 'llm_latency_ms', 'created_at')
    search_fields = ('user__username', 'user_input')
    list_filter = ('intent', 'created_at')


@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'movie', 'feedback_type', 'source', 'created_at')
    search_fields = ('user__username', 'movie__title')
    list_filter = ('feedback_type', 'source', 'created_at')
