"""
论文级模型扩展 - 新增5张表
================================================
1. ConversationHistory  - 结构化对话历史（含Agent追踪）
2. UserProfile          - 用户偏好画像
3. RecommendLog         - 推荐日志（多路召回记录）
4. AgentTrace           - Agent推理链追踪（ReAct范式）
5. UserFeedback         - 推荐反馈收集
================================================
"""

from django.conf import settings
from django.db import models


class ConversationHistory(models.Model):
    """
    结构化对话历史模型
    记录完整的用户-Agent交互过程，支持ReAct范式追踪
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        verbose_name="用户"
    )
    session_id = models.CharField(
        max_length=64, verbose_name="会话ID", db_index=True
    )
    role = models.CharField(
        max_length=16, verbose_name="角色",
        choices=[('user', '用户'), ('agent', 'Agent'), ('system', '系统')]
    )
    message = models.TextField(verbose_name="消息内容")
    
    # 意图分类结果
    intent = models.CharField(
        max_length=64, verbose_name="意图分类", blank=True, default=''
    )
    
    # Agent推理链（JSON格式存储Thought/Action/Observation）
    agent_trace = models.JSONField(
        verbose_name="Agent推理链", null=True, blank=True,
        help_text="存储 ReAct 范式的 Thought→Action→Observation→Final Answer"
    )
    
    # 推荐的电影ID列表
    recommended_movie_ids = models.JSONField(
        verbose_name="推荐电影列表", null=True, blank=True
    )
    
    # 耗时统计（毫秒）
    latency_ms = models.IntegerField(verbose_name="响应耗时(ms)", default=0)
    
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name="时间戳", db_index=True)

    class Meta:
        verbose_name = "结构化对话历史"
        verbose_name_plural = verbose_name
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', 'session_id']),
        ]

    def __str__(self):
        return f"[{self.session_id}] {self.role}: {self.message[:50]}"


class UserProfile(models.Model):
    """
    用户偏好画像模型
    存储从用户行为中提取的偏好特征向量与元信息
    """
    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='profile',
        verbose_name="用户"
    )
    
    # 偏好类型权重（JSON: {"科幻": 0.85, "剧情": 0.72, ...}）
    genre_preferences = models.JSONField(
        verbose_name="类型偏好", default=dict, blank=True
    )
    
    # 偏好演员权重
    actor_preferences = models.JSONField(
        verbose_name="演员偏好", default=dict, blank=True
    )
    
    # 偏好导演权重
    director_preferences = models.JSONField(
        verbose_name="导演偏好", default=dict, blank=True
    )
    
    # 用户行为统计
    total_ratings = models.IntegerField(verbose_name="总评分数", default=0)
    avg_score = models.FloatField(verbose_name="平均评分", default=0.0)
    total_collections = models.IntegerField(verbose_name="总收藏数", default=0)
    
    # 画像文本描述（由LLM生成）
    profile_text = models.TextField(
        verbose_name="画像描述", blank=True, default=''
    )
    
    # 冷启动标记
    is_cold_start = models.BooleanField(verbose_name="是否冷启动", default=True)
    
    # 最后更新时间
    updated_at = models.DateTimeField(auto_now=True, verbose_name="最后更新")

    class Meta:
        verbose_name = "用户画像"
        verbose_name_plural = verbose_name

    def __str__(self):
        return f"画像: {self.user.username} (冷启动={self.is_cold_start})"


class RecommendLog(models.Model):
    """
    推荐日志模型
    记录每次推荐的完整过程，支持离线评估与论文实验
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        verbose_name="用户"
    )
    
    # 推荐来源/策略
    STRATEGY_CHOICES = [
        ('hybrid', '混合召回'),
        ('collaborative', '协同过滤'),
        ('content', '内容推荐'),
        ('model', '模型推理'),
        ('agent', 'Agent推荐'),
        ('hot', '热门兜底'),
        ('visual', '视觉搜索'),
    ]
    strategy = models.CharField(
        max_length=32, verbose_name="推荐策略", choices=STRATEGY_CHOICES, default='hybrid'
    )
    
    # 各路召回结果（JSON格式）
    # {"vector": [id1, id2, ...], "content": [id3, id4, ...], "model": [id5, ...]}
    recall_results = models.JSONField(
        verbose_name="召回结果", null=True, blank=True
    )
    
    # RRF融合后的排序结果
    ranked_results = models.JSONField(
        verbose_name="精排结果", null=True, blank=True
    )
    
    # 最终推荐列表（Top-K）
    final_results = models.JSONField(
        verbose_name="最终推荐列表", null=True, blank=True
    )
    
    # 推荐理由（JSON: {movie_id: "reason text", ...}）
    explanations = models.JSONField(
        verbose_name="推荐理由", null=True, blank=True
    )
    
    # 用户查询文本
    query_text = models.TextField(verbose_name="查询文本", blank=True, default='')
    
    # 耗时统计
    latency_ms = models.IntegerField(verbose_name="总耗时(ms)", default=0)
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间", db_index=True)

    class Meta:
        verbose_name = "推荐日志"
        verbose_name_plural = verbose_name
        ordering = ['-created_at']

    def __str__(self):
        return f"推荐日志[{self.strategy}] - {self.user.username} @ {self.created_at}"


class AgentTrace(models.Model):
    """
    Agent推理链追踪模型
    完整记录 ReAct 范式的 Thought → Action → Observation → Final Answer
    用于论文展示与系统可解释性分析
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        verbose_name="用户"
    )
    
    # 关联的对话（可选）
    conversation = models.ForeignKey(
        ConversationHistory,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        verbose_name="关联对话"
    )
    
    # 原始用户输入
    user_input = models.TextField(verbose_name="用户输入")
    
    # 意图分类
    intent = models.CharField(max_length=64, verbose_name="意图分类")
    
    # ── ReAct 范式四阶段 ──
    
    # Thought: Agent的思考过程
    thought = models.TextField(
        verbose_name="Thought(思考)", blank=True, default='',
        help_text="Agent对用户需求的理解与推理规划"
    )
    
    # Action: Agent执行的动作
    actions = models.JSONField(
        verbose_name="Action(动作列表)", default=list, blank=True,
        help_text='[{"tool": "search_vector", "input": "科幻 高评分", "output": "..."}]'
    )
    
    # Observation: 动作执行后的观察结果
    observations = models.JSONField(
        verbose_name="Observation(观察)", default=list, blank=True,
        help_text="每个Action对应的执行结果"
    )
    
    # Final Answer: 最终答案
    final_answer = models.TextField(
        verbose_name="Final Answer(最终答案)", blank=True, default=''
    )
    
    # 推荐的电影列表
    recommended_movies = models.JSONField(
        verbose_name="推荐电影列表", null=True, blank=True
    )
    
    # 推荐理由
    explanations = models.JSONField(
        verbose_name="推荐理由", null=True, blank=True
    )
    
    # 性能指标
    total_latency_ms = models.IntegerField(verbose_name="总耗时(ms)", default=0)
    llm_latency_ms = models.IntegerField(verbose_name="LLM耗时(ms)", default=0)
    tool_latency_ms = models.IntegerField(verbose_name="工具耗时(ms)", default=0)
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "Agent推理链"
        verbose_name_plural = verbose_name
        ordering = ['-created_at']

    def __str__(self):
        return f"Trace[{self.intent}] {self.user.username}: {self.user_input[:30]}"
    
    def to_react_display(self):
        """
        生成用于论文展示的ReAct链格式化文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"[Thought]")
        lines.append(self.thought)
        lines.append("")
        
        for i, action in enumerate(self.actions):
            lines.append(f"[Action {i+1}]")
            lines.append(f"  工具: {action.get('tool', 'N/A')}")
            lines.append(f"  输入: {action.get('input', 'N/A')}")
            lines.append("")
        
        for i, obs in enumerate(self.observations):
            lines.append(f"[Observation {i+1}]")
            lines.append(f"  {str(obs)[:200]}")
            lines.append("")
        
        lines.append("[Final Answer]")
        lines.append(self.final_answer[:500])
        lines.append("=" * 60)
        
        return "\n".join(lines)


class UserFeedback(models.Model):
    """
    用户反馈模型
    收集用户对推荐结果的满意度反馈，用于离线评估
    """
    FEEDBACK_CHOICES = [
        ('like', '喜欢'),
        ('dislike', '不喜欢'),
        ('click', '点击'),
        ('skip', '跳过'),
        ('collect', '收藏'),
        ('share', '分享'),
    ]
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        verbose_name="用户"
    )
    
    movie = models.ForeignKey(
        'Movie',
        on_delete=models.CASCADE,
        verbose_name="电影"
    )
    
    # 关联的推荐日志
    recommend_log = models.ForeignKey(
        RecommendLog,
        on_delete=models.SET_NULL,
        null=True, blank=True,
        verbose_name="关联推荐日志"
    )
    
    feedback_type = models.CharField(
        max_length=16, verbose_name="反馈类型", choices=FEEDBACK_CHOICES
    )
    
    # 反馈来源（推荐页/详情页/聊天/搜索）
    source = models.CharField(
        max_length=32, verbose_name="反馈来源", blank=True, default=''
    )
    
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="反馈时间")

    class Meta:
        verbose_name = "用户反馈"
        verbose_name_plural = verbose_name
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['user', 'feedback_type']),
            models.Index(fields=['movie', 'feedback_type']),
        ]

    def __str__(self):
        return f"{self.user.username} {self.get_feedback_type_display()} 《{self.movie.title}》"