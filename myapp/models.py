from django import forms
from django.conf import settings
from django.contrib.auth.base_user import BaseUserManager, AbstractBaseUser
from django.contrib.auth.models import AbstractUser, PermissionsMixin
from django.db import models
from django.db.models import fields
from django.forms import widgets


# --- 1. 先定义 "标签" 模型 ---

class Genre(models.Model):
    name = models.CharField(max_length=100, unique=True, verbose_name="类型")
    def __str__(self):
        return self.name

class Region(models.Model):
    name = models.CharField(max_length=100, unique=True, verbose_name="地区")
    def __str__(self):
        return self.name

class Actor(models.Model):
    name = models.CharField(max_length=255, unique=True, verbose_name="演员")
    def __str__(self):
        return self.name

# --- 2. 然后在 Movie 模型中 "引用" 它们 ---

class Movie(models.Model):
    """
    电影主模型 (最终融合版)
    """

    # --- ↓↓↓ 1. 关键改动：区分 ID ---
    movielens_id = models.CharField(
        verbose_name="MovieLens ID",
        max_length=32,
        null=True, blank=True,  # 允许豆瓣电影没有 ML ID
        db_index=True, unique=True  # ML ID 必须是唯一的
    )
    imdb_id = models.CharField(
        verbose_name="IMDB ID",
        max_length=32,
        null=True, blank=True,  # 允许 ML 电影没有 豆瓣 ID
        db_index=True, unique=True  # 豆瓣 ID 必须是唯一的
    )
    # --- ↑↑↑ 改动结束 ↑↑↑ ---

    title = models.CharField(
        verbose_name="电影标题",
        max_length=255,
        db_index=True
    )
    score = models.DecimalField(
        verbose_name="评分", max_digits=3, decimal_places=1,
        null=True, blank=True
    )
    date = models.DateField(verbose_name="发布日期", null=True, blank=True)
    poster = models.URLField(verbose_name="海报链接", max_length=500, null=True, blank=True)
    # 🔥 1. 新增：本地海报字段
    poster_file = models.ImageField(
        verbose_name="本地海报",
        upload_to='posters/',  # 图片会存在 media/posters/ 下
        null=True,
        blank=True
    )
    # 🔥 2. 新增：视觉描述 (Caption) - 用于 RAG 聊天
    # 允许为空，旧数据默认填 NULL，绝对安全
    poster_caption = models.TextField(
        verbose_name="[AI] 视觉描述",
        null=True,
        blank=True
    )

    # 🔥 3. 新增：视觉向量 (Embedding) - 用于未来多模态推荐
    # JSONField 自动存取 List，非常方便
    poster_embedding_json = models.JSONField(
        verbose_name="[AI] 视觉向量",
        null=True,
        blank=True
    )

    # 🔥 4. 状态标记
    has_mm_features = models.BooleanField(
        default=False,
        verbose_name="已提取多模态特征"
    )
    summary = models.TextField(verbose_name="简介", null=True, blank=True)
    vote_count = models.IntegerField(verbose_name="评分人数", default=0)

    # 是否包含敏感内容
    is_sensitive = models.BooleanField(default=False, verbose_name="敏感内容",null=True, blank=True)
    # 具体是哪类敏感（如：血腥、恐怖、暴露）
    sensitive_type = models.CharField(max_length=50, blank=True, verbose_name="敏感类型", null=True)



    # --- (M2M 关系, Meta, __str__ ... 保持不变) ---
    actors = models.ManyToManyField(Actor, verbose_name="演员列表", blank=True)
    # 🔥 新增：导演字段 (复用 Actor 表，通过 related_name 区分)
    directors = models.ManyToManyField(Actor, verbose_name="导演", blank=True, related_name='directed_movies')
    regions = models.ManyToManyField(Region, verbose_name="地区", blank=True)
    genres = models.ManyToManyField(Genre, verbose_name="类型", blank=True)

    class Meta:
        verbose_name = "电影"
        verbose_name_plural = verbose_name
        ordering = ['-score', '-vote_count']

    def __str__(self):
        return self.title



class Collect(models.Model):
    # 使用外键关联用户 (User)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        verbose_name="用户"
    )
    # 使用外键关联电影 (Movie)
    movie = models.ForeignKey(
        'Movie',
        on_delete=models.CASCADE,
        verbose_name="电影"
    )
    # 收藏时间 (用于排序)
    collect_time = models.DateTimeField(auto_now_add=True, verbose_name="收藏时间")


    class Meta:
        verbose_name = "收藏"
        verbose_name_plural = verbose_name
        # 联合唯一索引：防止重复收藏同一部电影
        unique_together = ('user', 'movie')

    def __str__(self):
        return f"{self.user.username} 收藏了 {self.movie.title}"


class UserRating(models.Model):
    # 关联用户
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        verbose_name="用户"
    )
    # 关联电影
    movie = models.ForeignKey(
        'Movie',
        on_delete=models.CASCADE,
        verbose_name="电影"
    )
    # 评分 (可以为空, 比如只评论不打分)
    score = models.FloatField(verbose_name="评分", null=True, blank=True)
    # 评论内容 (可以为空, 比如只打分不评论)
    discussion = models.TextField(verbose_name="评论内容", null=True, blank=True)
    # 评论时间 (用于排序)
    comment_time = models.DateTimeField(auto_now=True, verbose_name="评论时间") # auto_now=True 表示每次更新都会刷新时间

    class Meta:
        verbose_name = "用户评分/评论"
        verbose_name_plural = verbose_name
        # 联合唯一: 一个用户对一部电影只能有一条记录
        unique_together = ('user', 'movie')

    def __str__(self):
        return f"{self.user.username} - {self.movie.title} ({self.score})"



class UserManager(BaseUserManager):
    def _create_user(self,username, password,email, **kwargs):
        if not username:
            raise ValueError('The given username must be set')
        if not password:
            raise ValueError('The given password must be set')
        if not email:
            raise ValueError('The given email must be set')
        user=self.model(username=username,email=email, **kwargs)
        user.set_password(password)
        user.save()
        return user

    def create_user(self,username, password,email, **kwargs):
        kwargs.setdefault('is_superuser', False)
        return self._create_user(username, password,email, **kwargs)

    def create_superuser(self,username, password,email, **kwargs):
        kwargs.setdefault('is_superuser', True)
        kwargs.setdefault('is_staff', True)
        return self._create_user(username, password,email, **kwargs)


class UserInfo(AbstractBaseUser, PermissionsMixin):
    """用户表"""
    is_staff = models.BooleanField(default=False)

    # --- ↓↓↓ 在这里添加新字段 ↓↓↓ ---
    navbar_color = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        verbose_name="导航栏颜色",
        default=None  # 默认不设置
    )

    user_ID = models.CharField(
        verbose_name="用户ID",
        max_length=32,
        null=True, blank=True,  # <-- 必须允许为空
        unique=True  # <-- 保持唯一
    )
    # --- 你必须自己定义的字段 ---
    username = models.CharField(verbose_name="用户名", max_length=150, unique=True, null=False)
    email = models.EmailField(verbose_name="邮箱", null=False, unique=False)
    registration = models.DateTimeField(
        verbose_name="注册时间",
        auto_now_add=True,  # <-- 必须有这个, 它会自动填入时间
        null=True, blank=True  # (最好也加上, 确保迁移安全)
    )
    # --- 你的自定义字段 (这些 OK) ---

    nickname = models.CharField(verbose_name="昵称", max_length=255, null=True)
    sex_choices = (
        (1, "男"),
        (2, "女"),
    )
    sex = models.IntegerField(choices=sex_choices, null=True, verbose_name="性别", default=1)
    age = models.IntegerField(verbose_name="年龄", null=True, blank=True)

    # MovieLens 的职业编码 (0-20)
    OCCUPATION_CHOICES = (
        (0, "其他/未指定"), (1, "学术/教育"), (2, "艺术/娱乐"), (3, "行政/管理"),
        (4, "大专/技术"), (5, "计算机/程序员"), (6, "医生/医疗"), (7, "技工/修理"),
        (8, "农民"), (9, "家庭主妇"), (10, "中小学生"), (11, "律师"),
        (12, "大学生"), (13, "军人"), (14, "退休"), (15, "销售/市场"),
        (16, "科学家"), (17, "服务业"), (18, "技工/蓝领"), (19, "失业"), (20, "作家")
    )

    # --- ↓↓↓ 新增字段 ↓↓↓ ---
    occupation = models.IntegerField(
        verbose_name="职业",
        choices=OCCUPATION_CHOICES,
        null=True, blank=True
    )
    zip_code = models.CharField(verbose_name="邮编", max_length=10, null=True, blank=True)

    # --- 为 Admin 和 Auth 系统必须添加的字段 ---

    # is_active 决定用户是否可以登录
    is_active = models.BooleanField(default=True)

    # is_staff 决定用户是否可以登录 admin 后台
    is_staff = models.BooleanField(default=False)

    # --- AbstractBaseUser 已经提供了 password 和 last_login ---
    # --- PermissionsMixin 已经提供了 is_superuser, groups, user_permissions ---

    # --- 关键设置 ---

    # 告诉 Django 这个模型的管理器是 UserManager
    objects = UserManager()

    # 告诉 Django 登录时用 'username' 字段
    USERNAME_FIELD = 'username'

    # 告诉 Django 创建超级用户时, 'email' 是必须的
    REQUIRED_FIELDS = ['email']

    # 告诉 Django 哪个字段是 email 字段
    EMAIL_FIELD = 'email'

    def __str__(self):
        return self.username



class Rec(models.Model):
    """
    推荐结果表 (存储离线计算的结果)
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, # 链接到 UserInfo
        on_delete=models.CASCADE,
        verbose_name="推荐用户"
    )
    movie = models.ForeignKey(
        Movie,                  # 链接到 Movie
        on_delete=models.CASCADE,
        verbose_name="推荐影片"
    )
    rating = models.FloatField( # 遵循截图的 FloatField
        verbose_name="推荐度 (预测评分)",
        null=True,
        blank=True
    )

    class Meta:
        verbose_name = "电影推荐"
        verbose_name_plural = verbose_name
        # 确保一个用户不会被重复推荐同一部电影
        constraints = [
            models.UniqueConstraint(fields=['user', 'movie'], name='unique_user_recommendation')
        ]

class ChatHistory(models.Model):
    """
    存储 RAG 助手的聊天记录
    """
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, # 链接到 UserInfo
        on_delete=models.CASCADE,
        verbose_name="用户"
    )
    #
    # 'Human' (用户) 或 'AI' (助手)
    #
    role = models.CharField(max_length=10)
    message = models.TextField(verbose_name="消息内容")
    timestamp = models.DateTimeField(auto_now_add=True, verbose_name="时间戳")

    class Meta:
        verbose_name = "聊天记录"
        verbose_name_plural = verbose_name
        ordering = ['timestamp'] # 保证按时间排序