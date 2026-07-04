# ── CUDA 环境变量（防御性设置，避免 CUDA 初始化失败导致进程崩溃）──
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # 已禁用: 此设置在某些环境下会导致CUDA初始化失败


# ── 安全的 CUDA 检查缓存（避免重复触发 CUDA 初始化错误）──
_CUDA_CACHE = {"checked": False, "available": False}
def _safe_cuda_available():
    """安全检查 CUDA 可用性，首次检查后缓存结果，避免重复触发初始化错误"""
    if not _CUDA_CACHE['checked']:
        try:
            import torch
            _CUDA_CACHE['available'] = torch.cuda.is_available()
        except Exception as _e:
            logger.warning(f"CUDA 初始化失败，降级为 CPU: {_e}")
            _CUDA_CACHE['available'] = False
        _CUDA_CACHE['checked'] = True
    return _CUDA_CACHE['available']

import hashlib
import json
import pickle
import re
import threading
import asyncio  # 🔥 新增：异步支持

import markdown  # 🔥 新增依赖: pip install markdown
import bleach
import time
import traceback
from collections import Counter
import jieba
import jieba.posseg as pseg # 用于词性标注
import jieba.analyse
from concurrent.futures import ThreadPoolExecutor # 🔥 引入线程池
import numpy as np
import requests
import torch
# 抑制 CUDA 初始化警告（环境变量冲突导致）
import warnings
warnings.filterwarnings("ignore", message=".*CUDA initialization.*")
from deep_translator import GoogleTranslator
from deepctr_torch.inputs import VarLenSparseFeat, SparseFeat, DenseFeat
from deepctr_torch.models import DeepFM, DIN
from django.conf import settings
from urllib.parse import quote
from django.contrib.auth.forms import SetPasswordForm, UserCreationForm
from django.core.cache import cache
from django.core.management import call_command
from django.shortcuts import render, redirect, get_object_or_404
from django.http import HttpResponseRedirect, JsonResponse, HttpResponseBadRequest
from asgiref.sync import sync_to_async  # 🔥 新增：Django ORM 异步包装
import random
# --- ↓↓↓ 1. 从 "datetime" 导入 timedelta ↓↓↓ ---
from datetime import datetime, timedelta, date

from django.urls import reverse
# --- ↓↓↓ 2. 从 "django.utils" 导入 timezone ↓↓↓ ---
from django.utils import timezone
from django.contrib.auth.decorators import user_passes_test
from django.utils.http import urlencode
from django.utils.safestring import mark_safe
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from django_select2.forms import Select2TagWidget
# 我们不再导入 "fields" 或 "widgets"
# 我们只导入 "forms"
from django import forms
# 我们只导入 "models"
from django.db import models, transaction
from django.db.models import Count
try:
    from keras.src.utils import pad_sequences
except ImportError:
    pad_sequences = None
try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None
try:
    from torch import device
except ImportError:
    device = None

from myapp import rag_agent
# 2. 导入你的模型 (这是安全的)
from myapp.models import Movie, UserInfo, Collect, Genre, Actor, Rec, UserRating, Region, ChatHistory
import logging
logger = logging.getLogger('movie_agent')

# ── 工程化工具模块导入 ──────────────────────────────────────────
# 【论文可写点】：统一的 Agent Trace、Tool Registry、Response 规范、Agent Service 层
from myapp.utils.agent_trace import AgentTracer, trace_log_simple
from myapp.utils.tool_registry import register_tool, dispatch_tool, list_registered_tools
from myapp.utils.response_helper import api_success, api_error, api_chat_response, api_explain_response
from myapp.services.agent_service import (
    safe_ollama_call, safe_neo4j_query,
    get_cached_explain, set_cached_explain,
)

# 3. 导入所有需要的 "视图" 模块
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout, update_session_auth_hash
from django.contrib.auth.decorators import login_required
from django.db.models import Q, Count, Max, Avg
from django.views.decorators.cache import never_cache

from myapp.rag_agent import chat_with_agent
from myapp.skb_model import SKB_FMLP_Online
from myapp.mman_model import MMAN  # 🔥 新增：MMAN 多模态注意力网络
# 5. 导入你的分页工具
from myapp.utils.pagination import Pagination

from django.contrib.auth.decorators import user_passes_test

from myapp.utils.vector_rag import query_vector_rag
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 🔥 LangChain 高级机制集成
from myapp.langchain_memory_enhancer import (
    ConversationSummaryBufferMemory,
    ContextualCompressionRetriever,
    ChatBranchRouter,
    initialize_chat_memory,
    compress_retrieval_results,
    route_user_intent,
    build_memory_enhanced_prompt,
    format_memory_for_display,
)


def admin_required(function):
    """
    一个自定义的装饰器，检查用户是否已登录 且 是管理员 (is_staff)
    如果不是，则重定向到登录页面。
    """
    return user_passes_test(lambda u: u.is_authenticated and u.is_staff, login_url='login_user')(function)


class ColorPreferenceForm(forms.ModelForm):
    class Meta:
        model = UserInfo
        fields = ['navbar_color'] # 假设你的用户模型里有这个字段
        widgets = {
            'navbar_color': forms.TextInput(attrs={'type': 'color', 'style': 'height: 40px; width: 100px;'})
        }
        labels = {
            'navbar_color': '导航栏主题色'
        }

class LoginForm(forms.Form):
    username = forms.CharField(
        label="用户名",
        required=True,
        min_length=3,
        max_length=18,
        error_messages={
            "required": "用户名不能为空",
            "min_length": "用户名最少3位",
            "max_length": "用户名最多18位",
        }
    )
    password = forms.CharField(
        label="密码",  # <-- 2. 添加中文标签
        widget=forms.PasswordInput,
        required=True,
        min_length=6,
        max_length=18,
        error_messages={
            "required": "密码不能为空",
            "min_length": "密码最少6位",
            "max_length": "密码最多18位",
        }
    )


class RegisterForm(forms.Form):
    username = forms.CharField(
        label="用户名",
        required=True,
        min_length=3,
        max_length=18,
        error_messages={
            "required": "用户名不能为空",
            "min_length": "用户名最少3位",
            "max_length": "用户名最多18位",
        }
    )
    email = forms.EmailField(
        required=True,
        error_messages={
            "required": "邮箱不能为空",
            "invalid": "邮箱格式不正确",
        }
    )
    password = forms.CharField(
        label="密码",  # <-- 2. 添加中文标签
        widget=forms.PasswordInput,
        required=True,
        min_length=6,
        max_length=18,
        error_messages={
            "required": "密码不能为空",
            "min_length": "密码最少6位",
            "max_length": "密码最多18位",
        }
    )
    # --- 🔥 新增画像字段 (冷启动关键) ---
    sex = forms.ChoiceField(
        label="性别",
        choices=UserInfo.sex_choices,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    age = forms.IntegerField(
        label="年龄",
        required=False,
        min_value=1, max_value=120,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': '选填，用于推荐'})
    )
    occupation = forms.ChoiceField(
        label="职业",
        choices=UserInfo.OCCUPATION_CHOICES,
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )

    def clean_password2(self):
        password = self.cleaned_data.get('password')
        password2 = self.cleaned_data.get('password2')
        if password and password2 and password != password2:
            raise forms.ValidationError("两次输入的密码不一致")
        return password2



class UserRatingForm(forms.ModelForm):
    class Meta:
        model = UserRating
        fields = ['score', 'discussion']
        widgets = {
            'score': forms.NumberInput(attrs={
                'class': 'form-control',
                'min': 0, 'max': 10, 'step': 0.5,
                'placeholder': '打分 (0-10)'
            }),
            'discussion': forms.Textarea(attrs={
                'class': 'form-control',
                'rows': 3,
                'placeholder': '写下你的想法...'
            }),
        }
        labels = {
            'score': '评分',
            'discussion': '评论'
        }


class MovieModelForm(forms.ModelForm):
    """
    电影管理表单 (V3 - 文本输入版)
    管理员可以直接输入 "中国, 美国" 或 "周星驰, 吴孟达"，系统自动关联。
    """
    # 1. 覆盖默认的 M2M 选择框，改为 CharField (文本框)
    #    required=False 允许不填
    actors = forms.CharField(
        label="演员列表",
        required=False,
        widget=forms.TextInput(
            attrs={'class': 'form-control', 'placeholder': '用逗号分隔，例如: 汤姆·汉克斯, 莱昂纳多'}),
        help_text="多个演员请用中文或英文逗号分隔"
    )

    # 🔥 新增：导演输入框
    directors = forms.CharField(
        label="导演",
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '例如: 克里斯托弗·诺兰'}),
        help_text="多个导演请用逗号分隔"
    )

    regions = forms.CharField(
        label="地区",
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '例如: 中国大陆, 美国'}),
        help_text="多个地区请用逗号分隔"
    )

    genres = forms.CharField(
        label="类型",
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '例如: 剧情, 科幻'}),
        help_text="多个类型请用逗号分隔"
    )

    class Meta:
        model = Movie
        # 我们仍然包含所有字段，但上面定义的三个会覆盖默认行为
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 给其他字段添加 bootstrap 样式
        for field_name, field in self.fields.items():
            if field_name not in ['actors', 'regions', 'genres']:  # 这三个我们上面已经加了
                if not isinstance(field.widget, forms.CheckboxInput):
                    field.widget.attrs['class'] = 'form-control'



    def clean_genres(self):
        # --- ↓↓↓ 关键修复：
        genre_names_or_ids = self.data.getlist('genres')
        # --- ↑↑↑

        final_genres = []
        for value in genre_names_or_ids:
            if value.isdigit():
                try:
                    genre = Genre.objects.get(pk=value)
                    final_genres.append(genre)
                except Genre.DoesNotExist:
                    pass
            else:
                genre, created = Genre.objects.get_or_create(name=value.strip())
                final_genres.append(genre)
        return final_genres

    def clean_regions(self):
        # --- ↓↓↓ 关键修复：
        region_names_or_ids = self.data.getlist('regions')
        # --- ↑↑↑

        final_regions = []
        for value in region_names_or_ids:
            if value.isdigit():  # <-- 'value' 现在是字符串, .isdigit() 可以工作了
                try:
                    region = Region.objects.get(pk=value)
                    final_regions.append(region)
                except Region.DoesNotExist:
                    pass
            else:
                region, created = Region.objects.get_or_create(name=value.strip())
                final_regions.append(region)
        return final_regions

    # 🔥 新增：清洗逻辑 (复用 Actor 模型)
    def clean_directors(self):
        names = self.data.get('directors', '').replace('，', ',').split(',')
        objs = []
        for name in names:
            name = name.strip()
            if name:
                # 导演也存在 Actor 表里
                obj, _ = Actor.objects.get_or_create(name=name)
                objs.append(obj)
        return objs

class ActorModelForm(forms.ModelForm):
    """
    演员管理表单
    """
    class Meta:
        model = Actor
        fields = ['name']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '输入演员姓名'})
        }
        labels = {
            'name': '演员姓名'
        }

def _process_m2m_text(movie_obj, text_data, model_class, field_name):
    """
    辅助函数：处理逗号分隔的字符串，自动创建/关联 M2M 对象
    :param movie_obj: 电影对象
    :param text_data: 输入的字符串 (如 "中国, 美国")
    :param model_class: 关联的模型类 (如 Region)
    :param field_name: Movie模型中的字段名 (如 "regions")
    """
    if not text_data:
        getattr(movie_obj, field_name).clear()  # 如果清空了输入，就清空关联
        return

    # 1. 分割字符串 (支持中文逗号和英文逗号)
    names = text_data.replace('，', ',').split(',')

    # 2. 准备对象列表
    objs = []
    for name in names:
        name = name.strip()
        if name:
            # 核心逻辑：有则获取，无则创建
            obj, created = model_class.objects.get_or_create(name=name)
            objs.append(obj)

    # 3. 设置关联 (set 会自动处理增删)
    getattr(movie_obj, field_name).set(objs)

class UserProfileForm(forms.ModelForm):
    """
    用户个人资料编辑表单 (用于个人中心)
    """
    class Meta:
        model = UserInfo
        # 允许用户修改的字段
        fields = ['username', 'email', 'sex', 'age', 'occupation']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control', 'readonly': 'readonly'}), # 用户名通常不允许随意改，设为只读
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
            'sex': forms.Select(attrs={'class': 'form-control'}),
            'age': forms.NumberInput(attrs={'class': 'form-control'}),
            'occupation': forms.Select(attrs={'class': 'form-control'}),
        }
        labels = {
            'username': '用户名 (不可修改)',
            'email': '邮箱',
            'sex': '性别',
            'age': '年龄',
            'occupation': '职业'
        }


class AdminUserCreationForm(UserCreationForm):
    """管理员创建用户表单"""
    class Meta(UserCreationForm.Meta):
        model = UserInfo
        # 🔥 新增 sex, age, occupation
        fields = ('username', 'email', 'user_ID', 'is_staff', 'sex', 'age', 'occupation')
        labels = {
            'username': '用户名 (不可修改)',
            'email': '邮箱',
            'is_staff': '管理员权限',
            'sex': '性别',
            'age': '年龄',
            'occupation': '职业'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            if not isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs['class'] = 'form-control'

class AdminUserChangeForm(forms.ModelForm):
    """管理员编辑用户表单"""
    class Meta:
        model = UserInfo
        # 🔥 新增 sex, age, occupation
        fields = ('username', 'email', 'user_ID', 'is_active', 'is_staff', 'sex', 'age', 'occupation')
        labels = {
            'username': '用户名 (不可修改)',
            'email': '邮箱',
            'is_staff': '管理员权限',
            'sex': '性别',
            'age': '年龄',
            'occupation': '职业'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field in self.fields.values():
            if not isinstance(field.widget, forms.CheckboxInput):
                field.widget.attrs['class'] = 'form-control'


class AdminUserResetPasswordForm(SetPasswordForm):
    """
    管理员 "重置密码" 表单 (U)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 自动添加 Bootstrap 样式
        for field_name, field in self.fields.items():
            field.widget.attrs['class'] = 'form-control'
            field.widget.attrs['placeholder'] = f'输入{field.label}'


# ── 排行榜辅助函数（消除 front_index / rank 重复逻辑）──────────
def _get_hot_and_high_movies(minimum_votes=1000, limit=10):
    """
    获取热门榜和高分榜电影（按 title 去重）。
    返回 (hot_queryset, high_queryset)
    先取 top-N title 列表，再取每个 title 的最佳影片，共 2 次查询。
    """
    # 热门榜：先取 vote_count 最高的 N 个 title，再取每个 title 的最佳影片
    hot_titles = list(
        Movie.objects.values('title')
        .annotate(max_vc=Max('vote_count'))
        .order_by('-max_vc')
        .values_list('title', flat=True)[:limit]
    )
    # 一次查询取所有候选，按 title 分组取第一条（vote_count 最大）
    hot_all = Movie.objects.filter(title__in=hot_titles).order_by('-vote_count')
    seen_titles = set()
    hot_pks = []
    for m in hot_all:
        if m.title not in seen_titles:
            seen_titles.add(m.title)
            hot_pks.append(m.pk)
            if len(hot_pks) >= limit:
                break
    query_hot = Movie.objects.filter(pk__in=hot_pks).order_by('-vote_count')

    # 高分榜：先取 score 最高的 N 个 title（需满足最低投票数），再取每个 title 的最佳影片
    high_titles = list(
        Movie.objects.filter(vote_count__gt=minimum_votes)
        .values('title')
        .annotate(max_sc=Max('score'))
        .order_by('-max_sc')
        .values_list('title', flat=True)[:limit]
    )
    high_all = Movie.objects.filter(
        title__in=high_titles, vote_count__gt=minimum_votes
    ).order_by('-score')
    seen_titles2 = set()
    high_pks = []
    for m in high_all:
        if m.title not in seen_titles2:
            seen_titles2.add(m.title)
            high_pks.append(m.pk)
            if len(high_pks) >= limit:
                break
    query_high = Movie.objects.filter(pk__in=high_pks).order_by('-score')

    return query_hot, query_high


# Create your views here.
def register(request):
    if request.method == 'GET':
        form = RegisterForm()
        return render(request, 'register.html', {'form': form})

    form = RegisterForm(data=request.POST)
    if form.is_valid():
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        email = form.cleaned_data['email']

        # 🔥 获取画像数据
        sex = form.cleaned_data.get('sex') or 1
        age = form.cleaned_data.get('age')
        occupation = form.cleaned_data.get('occupation')

        if UserInfo.objects.filter(username=username).exists():
            messages.error(request, "用户名已存在")
            return render(request, 'register.html', {'form': form})

        if UserInfo.objects.filter(email=email).exists():
            messages.error(request, "邮箱已注册")
            return render(request, 'register.html', {'form': form})

        User_ID = datetime.now().strftime('%Y%m%d%H%M%S') + str(random.randint(1000, 9999))

        # 🔥 创建用户时传入画像数据
        # 注意：UserInfo.objects.create_user 内部使用 **kwargs 接收额外参数
        UserInfo.objects.create_user(
            username=username,
            password=password,
            email=email,
            user_ID=User_ID,
            sex=sex,
            age=age,
            occupation=occupation
        )

        messages.success(request, "注册成功，请登录")
        return redirect('login_user')
    else:
        return render(request, 'register.html', {'form': form})

def logout_user(request):
    logout(request)
    return redirect('login_user')


def login_user(request):
    if request.method == 'GET':
        form = LoginForm()  # <-- 修复 Bug 4: 传入空表单
        return render(request, 'login.html', {'form': form})

    # --- POST ---
    form = LoginForm(data=request.POST)

    if form.is_valid():
        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        user = authenticate(request, username=username, password=password)  # 建议传入 request

        if user:
            login(request, user)
            request.session.set_expiry(None)
            return redirect('front_index')
        else:
            messages.error(request, "用户名或密码错误")
            return HttpResponseRedirect('/login/')

    else:
        # 表单验证失败 (例如，字段为空或太短)
        # 你的这个逻辑是正确的
        return render(request, 'login.html', {'form': form})


def front_index(request):
    """首页 (Homepage)"""
    query_hot, query_high = _get_hot_and_high_movies()

    hot_list = list(query_hot)
    hot_list.reverse()
    high_list = list(query_high)
    high_list.reverse()

    context = {
        "query_hot_reversed": hot_list,
        "query_high_reversed": high_list
    }
    return render(request, 'front_index.html', context)


def rank(request):
    """电影排行榜"""
    total_movie_count = Movie.objects.count()
    query_hot, query_high = _get_hot_and_high_movies()

    context = {
        "query_hot": query_hot,
        "query_high": query_high,
        "total_movie_count": total_movie_count
    }
    return render(request, 'rank.html', context)


def depot(request):
    # --- 1. 获取筛选参数 ---
    selected_genre_id = request.GET.get('genre')
    selected_region_id = request.GET.get('region')
    selected_year = request.GET.get('year')

    # --- 2. 获取用于填充下拉框的数据 ---
    all_genres = Genre.objects.all().order_by('name')
    all_regions = Region.objects.all().order_by('name')
    year_list = [
        '2026','2025','2024', '2023', '2022', '2021', '2020', '2019', '2018', '2017',
        '2016', '2015', '2014', '2013', '2012', '2011', '2010', '2009','其他'
    ]

    # --- 3. 筛选电影 ---
    # 🔥 优化：使用 prefetch_related 预加载 genres 和 directors，提升加载速度
    all_movies_queryset = Movie.objects.prefetch_related('genres', 'directors').all()

    if selected_genre_id:
        all_movies_queryset = all_movies_queryset.filter(genres__id=selected_genre_id)

    if selected_region_id:
        all_movies_queryset = all_movies_queryset.filter(regions__id=selected_region_id)

    if selected_year:
        if selected_year == '其他':
            all_movies_queryset = all_movies_queryset.filter(date__year__lt=2010)
        else:
            all_movies_queryset = all_movies_queryset.filter(date__year=selected_year)

    # 确定排序顺序
    all_movies_queryset = all_movies_queryset.distinct().order_by('-score')

    # --- 4. 实例化分页对象 ---
    page_object = Pagination(request, all_movies_queryset, page_size=10)

    # --- 5. 准备 context ---
    context = {
        'movies': page_object.page_queryset,
        'page_string': page_object.html(),
        'genres': all_genres,
        'regions': all_regions,
        'years': year_list,
        'selected_genre_id': selected_genre_id,
        'selected_region_id': selected_region_id,
        'selected_year': selected_year,
    }
    return render(request, 'depot.html', context)


# --- 新增：构建用户时序画像 ---
def get_time_aware_profile(user, limit=50):
    """
    获取基于时间感知的用户画像 (Time-Aware User Profile)
    策略：获取不低于 6.0 分(非极度反感)的近期交互，结合收藏记录。
    """
    # 1. 获取最近的及格评分 (最近 50 条，放宽至 6.0 分)
    recent_ratings = list(UserRating.objects.filter(
        user=user, score__gte=6.0  # 🔥 将 8.0 改为 6.0
    ).order_by('-comment_time').values_list('movie_id', flat=True)[:limit])

    # 2. 获取最近的收藏 (收藏行为代表绝对正向意图)
    recent_collects = list(Collect.objects.filter(
        user=user
    ).order_by('-collect_time').values_list('movie_id', flat=True)[:limit])

    # 3. 合并并去重，保持时间上的近似优先级
    merged_ids = list(dict.fromkeys(recent_ratings + recent_collects))

    return merged_ids[:limit]


def update_recommendations(user, top_k=15):
    """带 MMR 的实时推断"""
    load_model_assets()
    if MODEL_CACHE['model'] is None: return 0

    model = MODEL_CACHE['model']
    meta = MODEL_CACHE['meta']

    lbe_user = meta['lbe_user']
    lbe_movie = meta['lbe_movie']
    feature_store = meta['feature_store']
    SEQ_LEN = meta['SEQ_LEN']
    DIM = meta['UNIFIED_EMBED_DIM']

    user_history_raw = list(
        UserRating.objects.filter(user=user).order_by('comment_time').values_list('movie_id', flat=True))
    hist_enc = [lbe_movie.transform([str(m)])[0] + 1 for m in user_history_raw if str(m) in lbe_movie.classes_]

    if len(hist_enc) == 0:
        hist_padded = np.zeros(SEQ_LEN, dtype=np.int32)
    else:
        hist_padded = np.pad(hist_enc[-SEQ_LEN:], (0, max(0, SEQ_LEN - len(hist_enc))), 'constant') if len(
            hist_enc) < SEQ_LEN else hist_enc[-SEQ_LEN:]

    all_raw_mids = feature_store['raw_movie_ids']
    all_enc_mids = feature_store['enc_movie_ids']

    user_history_set = set(str(x) for x in user_history_raw)
    valid_mask = np.array([str(m) not in user_history_set for m in all_raw_mids])

    cand_raw_mids = all_raw_mids[valid_mask]
    cand_enc_mids = all_enc_mids[valid_mask]
    N_cand = len(cand_enc_mids)
    if N_cand == 0: return 0

    u_str = str(user.id)
    u_idx = lbe_user.transform([u_str])[0] + 1 if u_str in lbe_user.classes_ else 0

    infer_input = {
        'user_id': np.full(N_cand, u_idx, dtype=np.int32),
        'movie_id': cand_enc_mids,
        'hist_movie_id': np.tile(hist_padded, (N_cand, 1)),
        'sl': np.full(N_cand, min(len(hist_enc), SEQ_LEN), dtype=np.int32)
    }

    # 🔥 关键修复：把三大 KG 矩阵喂给模型
    if 'genres_matrix' in feature_store:
        infer_input['genres'] = feature_store['genres_matrix'][valid_mask]
        infer_input['actors'] = feature_store['actors_matrix'][valid_mask]
    if 'directors_matrix' in feature_store:
        infer_input['directors'] = feature_store['directors_matrix'][valid_mask]

    rag_b = feature_store['rag_matrix'][valid_mask]
    for i in range(DIM):
        infer_input[f'rag_{i}'] = rag_b[:, i]

    # 人口统计学特征（如果模型支持）
    if getattr(model, 'use_demographic', False):
        occupation = getattr(user, 'occupation', 0) or 0
        sex = getattr(user, 'sex', 1) or 1
        age = getattr(user, 'age', 25) or 25
        age_max = 56  # MovieLens 最大年龄
        infer_input['occupation'] = np.full(N_cand, occupation, dtype=np.int32)
        infer_input['sex'] = np.full(N_cand, sex, dtype=np.int32)
        infer_input['age_norm'] = np.full(N_cand, age / age_max, dtype=np.float32)

    with torch.no_grad():
        preds = model.predict(infer_input, batch_size=2048).flatten()

    top_100_idx = preds.argsort()[::-1][:100]
    cand_100_preds = preds[top_100_idx]
    cand_100_raw = cand_raw_mids[top_100_idx]
    cand_100_vecs = rag_b[top_100_idx]

    final_rec_indices = []
    ALPHA = 0.7

    for _ in range(min(top_k, len(cand_100_raw))):
        if not final_rec_indices:
            best_idx = 0
        else:
            best_score = -float('inf')
            best_idx = -1
            for i in range(len(cand_100_raw)):
                if i in final_rec_indices: continue
                sims = [np.dot(cand_100_vecs[i], cand_100_vecs[j]) for j in final_rec_indices]
                max_sim = max(sims) if sims else 0
                mmr_score = ALPHA * cand_100_preds[i] - (1 - ALPHA) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
        final_rec_indices.append(best_idx)

    Rec.objects.filter(user=user).delete()
    rec_objs = [
        Rec(user=user, movie_id=cand_100_raw[idx], rating=float(cand_100_preds[idx]))
        for idx in final_rec_indices
    ]
    Rec.objects.bulk_create(rec_objs)
    return len(rec_objs)


# ==========================================
# 混合检索召回 (Hybrid Retrieval Recall)
# ==========================================

def build_user_profile_text(user, top_n_genres=5, top_n_actors=5):
    """
    将用户的历史行为、偏好特征、人口属性等融合成一段自然语言「用户画像文本」，
    作为向量语义召回的 Query。

    返回: (profile_text: str, liked_ids: list)
    """
    liked_ids = get_time_aware_profile(user, limit=100)

    # --- 1. 统计偏好 Genre ---
    genre_counter = Counter(
        Movie.genres.through.objects.filter(movie_id__in=liked_ids)
        .values_list('genre__name', flat=True)
    )
    top_genres = [g for g, _ in genre_counter.most_common(top_n_genres)]

    # --- 2. 统计偏好 Actor ---
    actor_counter = Counter(
        Actor.objects.filter(movie__id__in=liked_ids)
        .values_list('name', flat=True)
    )
    top_actors = [a for a, _ in actor_counter.most_common(top_n_actors)]

    # --- 3. 拼接用户画像文本 ---
    parts = []
    if top_genres:
        parts.append(f"喜欢的电影类型：{'、'.join(top_genres)}")
    if top_actors:
        parts.append(f"喜爱的演员：{'、'.join(top_actors)}")

    # 加入人口属性 (冷启动补充信号)
    occ_dict = dict(UserInfo.OCCUPATION_CHOICES)
    if getattr(user, 'occupation', None) is not None:
        parts.append(f"职业：{occ_dict.get(user.occupation, '其他')}")
    if getattr(user, 'age', None):
        parts.append(f"年龄：{user.age}岁")
    if getattr(user, 'sex', None):
        parts.append("性别：男" if user.sex == 1 else "性别：女")

    # 取近期高分电影标题增强语义
    recent_high = list(
        Movie.objects.filter(
            id__in=liked_ids,
            userrating__user=user,
            userrating__score__gte=8.0
        ).values_list('title', flat=True)[:5]
    )
    if recent_high:
        parts.append(f"近期高分电影：{'、'.join(recent_high)}")

    if not parts:
        profile_text = "优质电影 高评分 经典"
    else:
        profile_text = "；".join(parts)

    return profile_text, liked_ids


def vector_recall(profile_text, excluded_ids, k=60):
    """
    路径 A：向量语义召回
    用用户画像文本在 FAISS 向量库里做相似度检索，返回 [(movie_id, score), ...]
    """
    results = []
    vectorstore = RAG_RESOURCES.get("vectorstore")
    if not vectorstore:
        return results
    try:
        docs_scores = vectorstore.similarity_search_with_score(profile_text, k=k + len(excluded_ids))
        seen = set(excluded_ids)
        for doc, dist_score in docs_scores:
            mid = (
                doc.metadata.get('id')
                or doc.metadata.get('movie_id')
                or doc.metadata.get('mid')
            )
            if not mid:
                continue
            mid = int(mid)
            if mid in seen:
                continue
            seen.add(mid)
            # FAISS L2 距离转相似度（归一化向量时 dist ≈ 2*(1-cos_sim)，越小越好）
            sim = max(0.0, 1.0 - float(dist_score) / 2.0)
            results.append((mid, sim))
            if len(results) >= k:
                break
    except Exception as e:
        logger.warning(f"[HybridRecall] vector_recall 异常: {e}")
    return results  # [(movie_id, similarity_score)]


def content_recall(user, excluded_ids, k=60):
    """
    路径 B：内容特征召回（Genre + Actor 共现匹配）
    返回 [(movie_id, match_score), ...]
    """
    pref_genres = list(
        Movie.genres.through.objects
        .filter(movie_id__in=excluded_ids)
        .values_list('genre_id', flat=True)
    )
    pref_actors = list(
        Actor.objects.filter(movie__id__in=excluded_ids)
        .annotate(c=Count('id'))
        .order_by('-c')
        .values_list('id', flat=True)[:50]
    )
    if not pref_genres and not pref_actors:
        # 冷启动：返回热门
        hot = list(
            Movie.objects.exclude(id__in=excluded_ids)
            .order_by('-vote_count', '-score')
            .values_list('id', 'vote_count')[:k]
        )
        max_vc = hot[0][1] if hot else 1
        return [(mid, vc / max_vc) for mid, vc in hot]

    candidates = (
        Movie.objects.filter(vote_count__gt=50)
        .exclude(id__in=excluded_ids)
        .annotate(
            match_score=(
                Count('genres', filter=Q(genres__in=pref_genres), distinct=True) * 2
                + Count('actors', filter=Q(actors__in=pref_actors), distinct=True) * 3
            )
        )
        .filter(match_score__gt=0)
        .order_by('-match_score')
        .values_list('id', 'match_score')[:k]
    )
    items = list(candidates)
    if not items:
        return []
    max_s = items[0][1] or 1
    return [(mid, score / max_s) for mid, score in items]


def model_recall(user, k=60):
    """
    路径 C：模型推理召回（从 Rec 表读取 SKB-FMLP 预测分）
    返回 [(movie_id, pred_score), ...]
    """
    recs = (
        Rec.objects.filter(user=user)
        .order_by('-rating')
        .values_list('movie_id', 'rating')[:k]
    )
    items = list(recs)
    if not items:
        return []
    max_r = items[0][1] or 1.0
    return [(mid, (r or 0) / max_r) for mid, r in items]


def rrf_merge(ranked_lists, k_rrf=60, weights=None):
    """
    Reciprocal Rank Fusion (RRF) 多路召回融合
    ranked_lists: List[List[(movie_id, score)]]，每路已按 score 降序
    weights: 每路权重，默认等权
    返回 [(movie_id, rrf_score), ...] 已按 rrf_score 降序
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    score_map = {}
    for ranked_list, w in zip(ranked_lists, weights):
        for rank_idx, (mid, _) in enumerate(ranked_list):
            score_map[mid] = score_map.get(mid, 0.0) + w / (k_rrf + rank_idx + 1)

    merged = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return merged  # [(movie_id, rrf_score), ...]


def hybrid_recall_recommend(user, top_k=30, force_refresh=False):
    """
    混合检索主函数：
    1. 构建用户画像文本
    2. 三路并行召回 (Vector / Content / Model)
    3. RRF 融合 & 取 top_k
    4. 返回 (Movie QuerySet, profile_text, source_stats)
    """
    cache_key = f"hybrid_rec_{user.id}"
    if not force_refresh:
        cached = cache.get(cache_key)
        if cached:
            return cached

    profile_text, liked_ids = build_user_profile_text(user)
    liked_set = set(liked_ids)

    # 确保 RAG 资源已加载
    load_rag_resources()

    # --- 三路召回 ---
    vec_results = vector_recall(profile_text, liked_ids, k=80)
    cb_results = content_recall(user, liked_ids, k=80)
    model_results = model_recall(user, k=80)

    source_stats = {
        'vector': len(vec_results),
        'content': len(cb_results),
        'model': len(model_results),
        'profile_text': profile_text,
    }

    # --- RRF 融合 (vector:1.2, content:1.0, model:1.5) ---
    merged = rrf_merge(
        [vec_results, cb_results, model_results],
        k_rrf=60,
        weights=[1.2, 1.0, 1.5]
    )

    # 过滤已看过的，取 top_k
    final_ids_scores = [(mid, s) for mid, s in merged if mid not in liked_set][:top_k]
    if not final_ids_scores:
        # 绝对兜底：热门
        fallback = list(
            Movie.objects.exclude(id__in=liked_ids)
            .order_by('-vote_count', '-score')
            .values_list('id', flat=True)[:top_k]
        )
        final_ids_scores = [(mid, 0.0) for mid in fallback]

    id_order = {mid: idx for idx, (mid, _) in enumerate(final_ids_scores)}
    id_score = {mid: s for mid, s in final_ids_scores}
    final_ids = [mid for mid, _ in final_ids_scores]

    movies_qs = list(
        Movie.objects.filter(id__in=final_ids)
        .prefetch_related('genres', 'actors', 'directors')
    )
    # 按 RRF 分排序，并附上 rrf_score 属性
    movies_qs.sort(key=lambda m: id_order.get(m.id, 9999))
    for m in movies_qs:
        m.rrf_score = round(id_score.get(m.id, 0.0), 4)

    result = (movies_qs, profile_text, source_stats)
    cache.set(cache_key, result, 1800)  # 缓存 30 分钟
    return result


# 视图函数：个性化推荐页入口
# ==========================================
@login_required
@never_cache
def recommend(request):
    """
    个性化推荐 (V_Optimize - 数据库下推与缓存提速版)
    逻辑：同时准备 KAG 模型推理 和 Content/Hot 数据，互不干扰，由前端 Tab 切换。
    """
    user = request.user

    # =================================================
    # Track 1: 混合智能推荐 (SKB-FMLP 在线推理)
    # =================================================
    force_refresh = request.GET.get('refresh') == 'true'
    deep_rec_exists = Rec.objects.filter(user=user).exists()

    if force_refresh or not deep_rec_exists:
        try:
            # 🚀 实时调用 KAG 模型预测！
            update_recommendations(user, top_k=40)
        except Exception as e:
            logger.error(f"模型推理失败: {e}")

    # 🔥 优化一：加上 prefetch_related，防止前端渲染卡片时触发 N+1 查询风暴
    # 🔥 排除"不喜欢"的电影
    excluded_ids = set(get_excluded_movie_ids(user))

    deep_rec_query = Rec.objects.filter(user=user).exclude(
        movie_id__in=excluded_ids
    ).select_related('movie').prefetch_related(
        'movie__genres', 'movie__directors', 'movie__actors'
    ).order_by('-rating')

    deep_rec_count = deep_rec_query.count()

    # 判断展示哪个酷炫的前端 Badge
    rec_source_main = 'skb_kag' if deep_rec_count > 0 else 'none'

    deep_paginator = Pagination(
        request,
        deep_rec_query,
        page_size=8,
        page_param="page_deep",
        exclude_params=['page_cb']
    )

    # =================================================
    # Track 2: Content-Based / Hot 在线推荐 (缓存+SQL提速)
    # =================================================
    MIN_LIKED_MOVIES = 5
    liked_ids = get_time_aware_profile(user, limit=50)
    total_liked_count = len(liked_ids)

    # 🔥 优化二：引入缓存。如果用户只是在翻页，直接从缓存拿兜底数据，避免重算
    cache_key = f"cb_recs_{user.id}"
    cached_cb_data = cache.get(cache_key) if not force_refresh else None

    if cached_cb_data:
        rec_source = cached_cb_data['source']
        final_rec_list = cached_cb_data['list']
    else:
        rec_source = "unknown"
        final_rec_list = []

        # --- 策略 A: Content-Based (兴趣匹配) ---
        if total_liked_count >= MIN_LIKED_MOVIES:
            rec_source = "content"
            # 转化为 list 供后续 ORM 查询使用
            pref_genres = list(
                Movie.genres.through.objects.filter(movie_id__in=liked_ids).values_list('genre_id', flat=True))
            pref_actors = list(
                Actor.objects.filter(movie__id__in=liked_ids).annotate(c=Count('id')).order_by('-c').values_list('id',
                                                                                                                flat=True)[
                    :50])

            # 🔥 优化三：利用 Django ORM 的 Count + Q对象，将 for 循环下推到数据库 C 语言层执行
            candidates = Movie.objects.filter(vote_count__gt=50).exclude(
                id__in=list(liked_ids) + list(excluded_ids)
            ).annotate(
                match_score=(
                        Count('genres', filter=Q(genres__in=pref_genres), distinct=True) * 2 +
                        Count('actors', filter=Q(actors__in=pref_actors), distinct=True) * 3
                )
            ).filter(match_score__gt=0).order_by('-match_score')[:50]

            # 打平为 list 并预加载关联，存入内存
            final_rec_list = list(candidates.prefetch_related('genres', 'actors'))

        # --- 策略 B: Hot (热门榜单兜底) ---
        if rec_source == "unknown" or not final_rec_list:
            rec_source = "hot"
            final_rec_list = list(
                Movie.objects.exclude(id__in=excluded_ids).prefetch_related(
                    'genres', 'actors').order_by('-vote_count', '-score')[:50])

        # 写入缓存，保留 1 小时
        cache.set(cache_key, {'source': rec_source, 'list': final_rec_list}, 3600)

    cb_paginator = Pagination(
        request,
        final_rec_list,
        page_size=8,
        page_param="page_cb",
        exclude_params=['page_deep']
    )

    # =================================================
    # 3. 视图控制 (Tab 激活逻辑)
    # =================================================
    active_tab = "deep"  # 默认

    if request.GET.get('page_cb'):
        active_tab = "cb"
    elif request.GET.get('page_deep'):
        active_tab = "deep"
    elif deep_rec_count == 0:
        active_tab = "cb"

    context = {
        'deep_rec_list': deep_paginator.page_queryset,
        'deep_page_string': deep_paginator.html(),
        'deep_count': deep_rec_count,

        'content_rec_list': cb_paginator.page_queryset,
        'content_page_string': cb_paginator.html(),
        'content_count': cb_paginator.total_count,


        'rec_source': rec_source,
        'rec_source_main': rec_source_main,
        'total_liked_count': total_liked_count,
        'MIN_LIKED_MOVIES': MIN_LIKED_MOVIES,
        'active_tab': active_tab,
    }
    return render(request, 'recommendations.html', context)


@login_required
def score_movie(request):
    """
    处理用户在电影详情页提交的打分与评论
    """
    if request.method == 'POST':
        movie_id = request.POST.get('movie_id')
        score_val = request.POST.get('score')
        discussion_val = request.POST.get('discussion', '').strip()

        if movie_id and score_val:
            try:
                score_val = float(score_val)
                # 限制分数在 0 到 10 之间
                if score_val < 0 or score_val > 10:
                    messages.error(request, "评分必须在 0 到 10 之间。")
                else:
                    # update_or_create：如果该用户之前评过这部电影，就更新分数和评论；没评过就新建
                    UserRating.objects.update_or_create(
                        user=request.user,
                        movie_id=movie_id,
                        defaults={
                            'score': score_val,
                            'discussion': discussion_val if discussion_val else None
                        }
                    )
                    messages.success(request, "评价提交成功！系统将根据您的最新反馈调整推荐。")
            except ValueError:
                messages.error(request, "无效的评分格式。")
        else:
            messages.error(request, "提交失败，请填写完整评分。")

    # 操作完成后，重定向回刚才的那部电影的详情页
    return redirect('movie_detail', pk=movie_id)

@never_cache  # (这个我们之前加过，防止缓存)
def movie_detail(request, pk):
    # 🔥 优化：使用 prefetch_related 预加载所有 M2M 字段 (含 directors)
    # 这样在模板里循环时，不会产生几十条 SQL 查询
    movie = get_object_or_404(
        Movie.objects.prefetch_related('actors', 'directors', 'genres', 'regions', 'userrating_set__user'),
        pk=pk
    )

    is_collected = False
    my_rating = None
    rating_form = UserRatingForm()

    if request.user.is_authenticated:
        is_collected = Collect.objects.filter(user=request.user, movie=movie).exists()
        try:
            my_rating = UserRating.objects.get(user=request.user, movie=movie)
        except UserRating.DoesNotExist:
            my_rating = None

    if request.method == 'POST' and request.user.is_authenticated:
        rating_form = UserRatingForm(data=request.POST)
        if rating_form.is_valid():
            UserRating.objects.update_or_create(
                user=request.user,
                movie=movie,
                defaults={
                    'score': rating_form.cleaned_data['score'],
                    'discussion': rating_form.cleaned_data['discussion']
                }
            )
            messages.success(request, "评论发表成功！")
            return redirect('movie_detail', pk=pk)

    if my_rating and request.method == 'GET':
        rating_form = UserRatingForm(initial={
            'score': my_rating.score,
            'discussion': my_rating.discussion
        })

    # 取最近 20 条评论（已通过 prefetch_related 预加载，无额外查询）
    comments = sorted(movie.userrating_set.all(), key=lambda r: r.comment_time or r.id, reverse=True)[:20]

    context = {
        'movie': movie,
        'is_collected': is_collected,
        'my_rating': my_rating,
        'rating_form': rating_form,
        'comments': comments,
    }
    return render(request, 'movie_detail.html', context)


@login_required  # 2. 确保只有登录用户才能调用
def add_collect(request, pk):
    # 获取电影, 如果不存在则 404
    movie = get_object_or_404(Movie, pk=pk)

    # 3. 创建收藏记录
    #    get_or_create 会自动检查是否已存在, 存在则不操作, 不存在则创建
    Collect.objects.get_or_create(
        user=request.user,  # 当前登录的用户
        movie=movie  # 当前电影
    )

    # 4. 操作完成后, 重定向回电影详情页
    return redirect('movie_detail', pk=pk)


@login_required  # 2. 确保只有登录用户才能调用
def remove_collect(request, pk):
    movie = get_object_or_404(Movie, pk=pk)

    # 3. 查找并删除收藏记录
    Collect.objects.filter(
        user=request.user,  # 当前登录的用户
        movie=movie  # 当前电影
    ).delete()

    # 4. 操作完成后, 重定向回电影详情页
    return redirect('movie_detail', pk=pk)


# --- ↓↓↓ 新增 "收藏" 视图 ↓↓↓ ---
@login_required
@require_POST
def ajax_collect(request):
    movie_id = request.POST.get('movie_id')
    if not movie_id:
        return JsonResponse({'status': 'error', 'msg': '参数错误'})

    try:
        movie = Movie.objects.get(pk=movie_id)
        # 使用 get_or_create 防止重复写入报错
        collect_obj, created = Collect.objects.get_or_create(user=request.user, movie=movie)

        if not created:
            # 如果已经存在 (created=False)，说明是"取消收藏"操作
            collect_obj.delete()
            return JsonResponse({'status': 'success', 'action': 'uncollected', 'msg': '已取消收藏'})
        else:
            # 如果是新建 (created=True)，说明是"收藏"操作
            return JsonResponse({'status': 'success', 'action': 'collected', 'msg': '收藏成功'})

    except Movie.DoesNotExist:
        return JsonResponse({'status': 'error', 'msg': '电影不存在'})
    except Exception as e:
        return JsonResponse({'status': 'error', 'msg': str(e)})


# 🧱 全局资源容器 (Global Resource Containers)
# ==============================================================================
VISUAL_RESOURCES = {
    "index": None, "ids": None, "model": None, "processor": None,
    "movie_embeddings": None, "movie_objects": None  # 🔥 新增这两项
}

MODEL_CACHE = {
    'model': None,
    'meta': None,
    'rag_map': None
}

RAG_RESOURCES = {
    "embeddings": None, "vectorstore": None, "retriever": None
}

# 🔥 新增：推荐解释模块的全局资源
EXPLAIN_RESOURCES = {
    "model": None  # 用于存放 SentenceTransformer
}
# ==============================================================================
# 🔥 核心加载逻辑 (带 Safetensors & 显存保护)
# ==============================================================================




def load_model_assets():
    """将模型和特征库加载到 Django 全局内存中 (动态维度版)"""
    if MODEL_CACHE['model'] is not None:
        return

    try:
        logger.info("正在初始化在线 KAG 推荐引擎...")

        # 🔥🔥🔥 救命代码：强制限制 PyTorch 只允许使用 20% 的显存 🔥🔥🔥
        import torch
        _cuda_ok = False
        try:
            _cuda_ok = _safe_cuda_available()
            if _cuda_ok:
                try:
                    torch.cuda.set_per_process_memory_fraction(0.2, 0)
                    try:
                        torch.cuda.empty_cache()
                    except Exception: pass
                except Exception:
                    pass  # CUDA操作失败静默降级
        except Exception as _e:
            logger.warning(f"CUDA 初始化失败，将使用 CPU: {_e}")
            _cuda_ok = False
        artifacts_dir = os.path.join(settings.BASE_DIR, 'ml_artifacts')
        meta_path = os.path.join(artifacts_dir, 'online_features_meta.pkl')
        model_path = os.path.join(artifacts_dir, 'skb_fmlp_online.pt')

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        device = 'cuda' if _safe_cuda_available() else 'cpu'

        # 🔥 1. 先把 state_dict 加载到内存里，不急着赋给模型
        state_dict = torch.load(model_path, map_location=device, weights_only=True)

        # 🔥 2. 从真实权重中动态读取你的本地图谱字典大小
        vocab_genre = state_dict['embedding_dict.genres.weight'].shape[0]
        vocab_actor = state_dict['embedding_dict.actors.weight'].shape[0]
        vocab_director = state_dict['embedding_dict.directors.weight'].shape[0]

        vocab_user = len(meta['lbe_user'].classes_) + 1
        vocab_movie = len(meta['lbe_movie'].classes_) + 1
        DIM = meta['UNIFIED_EMBED_DIM']
        SEQ = meta['SEQ_LEN']

        user_col = SparseFeat('user_id', vocab_user, DIM, embedding_name='user_id')
        movie_col = SparseFeat('movie_id', vocab_movie, DIM, embedding_name='movie_id')

        # 🔥 3. 使用真实动态维度构建网络
        genre_col = VarLenSparseFeat(SparseFeat('genres', vocab_genre, DIM), maxlen=5, combiner='mean')
        actor_col = VarLenSparseFeat(SparseFeat('actors', vocab_actor, DIM), maxlen=5, combiner='mean')
        director_col = VarLenSparseFeat(SparseFeat('directors', vocab_director, DIM), maxlen=3, combiner='mean')

        rag_cols = [DenseFeat(f'rag_{i}', 1) for i in range(DIM)]
        seq_col = VarLenSparseFeat(SparseFeat('hist_movie_id', vocab_movie, DIM, embedding_name='movie_id'), maxlen=SEQ,
                                    length_name='sl', combiner='mean')

        linear_cols = [movie_col] + rag_cols
        dnn_cols = [user_col, movie_col, genre_col, actor_col, director_col, seq_col] + rag_cols

        # 🔥 根据 meta 中的 model_type 自动选择模型架构
        model_type = meta.get('model_type', 'skb_fmlp')
        TEXT_DIM = meta.get('TEXT_DIM', 64)
        VISUAL_DIM = meta.get('VISUAL_DIM', 64)
        DROPOUT = meta.get('FIXED_DROPOUT', 0.1)

        if model_type == 'mman':
            # 检查权重是否包含人口统计学特征
            has_demographic = any(
                'demo_encoder' in k or 'occupation' in k or 'sex' in k
                for k in state_dict.keys()
            )
            # 如果有权重包含人口统计学特征，补充特征列
            if has_demographic:
                occupation_col = SparseFeat('occupation', 21, 8)
                sex_col = SparseFeat('sex', 3, 4)
                age_col = DenseFeat('age_norm', 1)
                dnn_cols = dnn_cols + [occupation_col, sex_col, age_col]
            model = MMAN(
                linear_cols, dnn_cols,
                history_feature_list=['movie_id'],
                text_dim=TEXT_DIM,
                visual_dim=VISUAL_DIM,
                hidden_dim=256,
                num_heads=4,
                dropout=DROPOUT,
                use_demographic=has_demographic,
                device=device
            )
            logger.info(f"加载 MMAN 多模态注意力模型 (TEXT={TEXT_DIM}, VISUAL={VISUAL_DIM}, demographic={has_demographic})")
        else:
            model = SKB_FMLP_Online(
                linear_feature_columns=linear_cols,
                dnn_feature_columns=dnn_cols,
                history_feature_list=['movie_id'],
                device=device
            )
            logger.info("加载 SKB-FMLP 模型")

        # 加载刚才读好的权重并开启 Eval 模式
        model.load_state_dict(state_dict)
        model.eval()

        MODEL_CACHE['model'] = model
        MODEL_CACHE['meta'] = meta
        logger.info(f"在线 KAG 引擎启动成功 (模型={model_type}, 图谱节点数: 类型{vocab_genre}, 演员{vocab_actor}, 导演{vocab_director})")
        # 清理显存垃圾
        if _safe_cuda_available():
            try: torch.cuda.empty_cache()
            except Exception: pass
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        import traceback
        traceback.print_exc()

def load_rag_resources():
    """
    [3/3] 预热 RAG 向量检索
    ✅ 特性: Safetensors 安全加载 | 显存保护
    """
    global RAG_RESOURCES
    if RAG_RESOURCES["embeddings"] is not None: return

    logger.info("[预热] 正在加载 RAG 向量模型 (BGE)...")
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_community.vectorstores import FAISS

        device = "cuda" if _safe_cuda_available() else "cpu"
        # 优先使用本地模型，避免网络问题导致加载超时
        local_bge_path = os.path.join(settings.BASE_DIR, "local_models", "bge-small-zh-v1.5")
        # HuggingFace 缓存结构：snapshots/<hash>/config.json
        snapshot_dir = os.path.join(local_bge_path, "snapshots")
        # 优先使用 snapshots 子目录（HuggingFace cache 格式，包含完整 SentenceTransformer config）
        if os.path.isdir(snapshot_dir):
            # 尝试 snapshots 子目录（HuggingFace cache 格式）
            snap_hashes = os.listdir(snapshot_dir)
            snap_config = None
            for h in snap_hashes:
                candidate = os.path.join(snapshot_dir, h)
                if os.path.exists(os.path.join(candidate, "config.json")):
                    snap_config = candidate
                    break
            if snap_config:
                embedding_model_name = snap_config
                logger.info(f"使用本地 BGE 模型 (snapshot): {snap_config}")
            else:
                embedding_model_name = "BAAI/bge-small-zh-v1.5"
                logger.warning("本地 BGE snapshot 无 config.json，回退到远程加载")
        else:
            embedding_model_name = "BAAI/bge-small-zh-v1.5"
            logger.warning("本地 BGE 模型不存在，回退到 HuggingFace 远程加载")

        # 🔥 重点: 在 model_kwargs 中传递 use_safetensors
        # SentenceTransformers 后端支持此参数
        model_kwargs = {'device': device}
        encode_kwargs = {'normalize_embeddings': True}

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            # 预热一次推理
            embeddings.embed_query("warmup")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("显存不足，RAG 降级为 CPU 模式...")
                model_kwargs['device'] = 'cpu'
                embeddings = HuggingFaceEmbeddings(
                    model_name=embedding_model_name,
                    model_kwargs=model_kwargs,
                    encode_kwargs=encode_kwargs
                )
            else:
                raise e

        # 加载索引
        index_path = os.path.join(settings.BASE_DIR, 'faiss_movie_index')
        if os.path.exists(index_path):
            RAG_RESOURCES["vectorstore"] = FAISS.load_local(
                index_path, embeddings, allow_dangerous_deserialization=True
            )
            logger.info(f"RAG 引擎就绪 (知识条目: {RAG_RESOURCES['vectorstore'].index.ntotal})")
            RAG_RESOURCES["embeddings"] = embeddings
        else:
            logger.warning("缺少 RAG 索引文件，跳过。")

    except Exception as e:
        logger.error(f"RAG 加载失败: {e}")


def load_explain_resources():
    """
    [4/4] 预热推荐解释模块 (SentenceTransformer)
    ✅ 特性: 强制 CPU 运行 (省显存), 内存常驻
    """
    global EXPLAIN_RESOURCES
    if EXPLAIN_RESOURCES["model"] is not None:
        return

    logger.info("[预热] 正在加载推荐解释模型 (Genre Similarity)...")
    try:
        from sentence_transformers import SentenceTransformer

        # 指定一个小巧的模型，用于计算文本相似度
        model_name = "all-MiniLM-L6-v2"

        # 强制使用 CPU，因为这个模型在 CPU 上推理也只需要 10ms
        # 这样可以防止挤占显存导致 OOM
        model = SentenceTransformer(model_name, device='cpu')

        EXPLAIN_RESOURCES["model"] = model
        logger.info("解释模型就绪 (CPU Mode)")

    except Exception as e:
        logger.error(f"解释模型加载失败: {e}")

def keep_alive_ollama():
    try:
        ollama_url = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434') + '/api/generate'
        requests.post(ollama_url,
                        json={"model": "qwen3:4b-instruct", "keep_alive": "60m"},
                        timeout=1)
    except:
        pass


def load_visual_resources():
    """
    视觉大模型加载器 (CLIP / SentenceTransformer 架构)
    用于文本到图像 (Text-to-Image) 的多模态检索预热
    """
    if VISUAL_RESOURCES.get("model") is not None:
        return  # 已经加载过了，直接跳过

    logger.info("正在加载视觉多模态大模型 (CLIP-ViT-B-32)...")
    try:
        # 指定计算设备，优先使用 GPU 提速
        device = 'cuda' if _safe_cuda_available() else 'cpu'

        # 加载 CLIP 文本/图像双模态编码器
        # 注意：这里必须与你之前生成 poster_embedding_json 时用的模型型号一致！
        model = SentenceTransformer('clip-ViT-B-32', device=device)

        # 挂载到全局字典
        VISUAL_RESOURCES["model"] = model
        logger.info(f"视觉模型加载成功 (Device: {device})")

    except Exception as e:
        logger.error(f"视觉模型加载失败: {e}")
        # 如果由于网络问题下不到模型，可以不阻断整个系统的启动
        pass
    # 🔥 新增：预先加载所有电影的视觉向量到内存矩阵中
    if VISUAL_RESOURCES.get("movie_embeddings") is None:
        logger.info("正在缓存电影视觉向量矩阵...")
        candidates = list(
            Movie.objects.filter(poster_embedding_json__isnull=False).exclude(poster_embedding_json=''))
        emb_list = []
        valid_movies = []
        for m in candidates:
            try:
                emb_data = json.loads(m.poster_embedding_json) if isinstance(m.poster_embedding_json,
                                                                            str) else m.poster_embedding_json
                emb_list.append(np.array(emb_data))
                valid_movies.append(m)
            except:
                pass

        VISUAL_RESOURCES["movie_embeddings"] = np.array(emb_list) if emb_list else None
        VISUAL_RESOURCES["movie_objects"] = valid_movies
        logger.info(f"成功缓存 {len(valid_movies)} 部电影的视觉向量")

def warmup_all_systems():
    """
    🚀 全局预热入口 (Entry Point)
    """
    logger.info("=" * 50)
    logger.info("[System Warmup] 系统全链路预热开始...")
    logger.info("安全模式: Safetensors Enabled")
    t_start = time.time()

    # 1. NLP 分词预热
    try:
        list(jieba.cut("系统启动"))

    except:
        pass

    # 2. 三大引擎预热
    load_visual_resources()  # CLIP
    load_model_assets()  # 混合

    load_rag_resources()  # BGE/FAISS
    # . 新增：推荐解释
    load_explain_resources()
    keep_alive_ollama()
    if _safe_cuda_available():
        try: torch.cuda.empty_cache()
        except Exception: pass
    logger.info(f"预热完成 总耗时: {time.time() - t_start:.2f}s")
    logger.info("=" * 50)



# 🌍 鲁棒的翻译函数 (新增)
# ==============================================================================
def robust_translate_to_english(text):
    """
    智能翻译函数 (双保险策略)
    1. 优先尝试 Google 翻译 (速度快，但依赖网络)
    2. 失败则调用本地 Qwen 模型 (速度尚可，完全离线，结果精准)
    3. 再失败则返回原词
    """
    if not text: return ""

    # 纯 ASCII 字符不需要翻译 (已经是英文)
    if all(ord(char) < 128 for char in text):
        return text

    # --- Plan A: 在线翻译 (Google) ---
    try:
        from deep_translator import GoogleTranslator
        # 设置超时，防止断网时卡太久
        # 注意: deep_translator 底层用 requests，这里并未直接暴露 timeout 参数，
        # 但通常它会比较快抛出异常。如果为了更稳，可以用 requests 封一层。
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        if translated:
            logger.info(f"[Google] 翻译成功: {text} -> {translated}")
            return translated
    except Exception as e:
        logger.warning(f"[Google] 翻译失败 (断网/超时): {e}")

    # --- Plan B: 本地 LLM 翻译 (Qwen) ---
    try:
        logger.info(f"[Qwen] 启动本地翻译兜底: {text}")
        from langchain_ollama import ChatOllama

        # 实例化一个专门用于翻译的 client (temperature=0 保证准确)
        llm = ChatOllama(
            model="qwen3:4b-instruct",
            temperature=0,

        )

        # 极简 Prompt，强制只输出结果
        prompt = (
            f"Translate the following search keyword from Chinese to English.\n"
            f"Keyword: {text}\n"
            f"Requirements: Output ONLY the English translation. Do not explain. Do not add punctuation."
        )

        response = llm.invoke(prompt)
        translated = response.content.strip()

        # 简单的清洗，防止 LLM 话痨
        translated = translated.replace('"', '').replace("'", "").replace(".", "")

        logger.info(f"[Qwen] 翻译成功: {translated}")
        return translated

    except Exception as e:
        logger.error(f"[Qwen] 本地翻译也挂了: {e}")

    # --- Plan C: 原样返回 ---
    return text


def contains_chinese(text):
    """检测字符串是否包含中文字符"""
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def robust_translate_to_chinese(text):
    """
    健壮的中文翻译函数：
    1. 自动检测语言，如果是中文则直接返回
    2. 尝试使用 GoogleTranslator 在线翻译
    3. 如果在线翻译失败，调用本地 Ollama (Qwen2.5) 进行离线翻译
    4. 兜底返回原字符串
    """
    if not text or not str(text).strip():
        return ""

    # 1. 检查是否已经是中文（避免重复调用）
    if contains_chinese(text):
        return text

    # 2. 尝试在线翻译 (Google)
    try:
        # 设置超时时间，防止网络卡顿导致整个请求挂起
        translated = GoogleTranslator(source='auto', target='zh-CN').translate(text)
        if translated and contains_chinese(translated):
            logger.info(f"[Online Translate] '{text}' -> '{translated}'")
            return translated
    except Exception as e:
        logger.warning(f"[Online Translate Error] 尝试离线方案: {e}")

    # 3. 尝试离线翻译 (本地 LLM)
    try:

        # 设置较低的 temperature 以保证翻译的准确性
        llm = ChatOllama(model="qwen3:4b-instruct", temperature=0.1)

        prompt = f"你是一个专业的翻译官。请将以下内容翻译成简洁自然的中文，不要输出任何解释，只返回翻译结果：\n\n{text}"

        response = llm.invoke(prompt)
        translated_llm = response.content.strip()

        if translated_llm and contains_chinese(translated_llm):
            logger.info(f"[Offline LLM Translate] '{text}' -> '{translated_llm}'")
            return translated_llm
    except Exception as e:
        logger.error(f"[Offline Translate Error] 全部翻译方案失效: {e}")

    # 4. 最终兜底：返回原始文本
    return text


# ==============================================================================
# 👁️ 视觉搜索视图 (更新版)
# ==============================================================================
# 如果没安装，需确保环境里有 sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    # 加载 CLIP 文本编码器 — 强制CPU避免CUDA初始化冲突
    CLIP_MODEL = SentenceTransformer('clip-ViT-B-32', device='cpu')
except Exception:
    CLIP_MODEL = None


def search_visual(request):
    """
    视觉搜索 V100：彻底解决超时卡顿 + 全局资源对齐 (矩阵加速版)
    """
    t_start = time.time()
    raw_query = request.GET.get('q', '').strip()
    DEFAULT_POSTER = "/static/img/no_poster.png"

    if not raw_query:
        return render(request, 'search_visual.html', {'movies': [], 'visual_query': ''})

    logger.info(f"[Visual Search] 原始输入: {raw_query}")

    # =================================================
    # 1. 翻译模块 (CLIP视觉搜索专用：带上下文的精准翻译)
    # =================================================
    search_query_en = raw_query
    import re
    # 只有当包含中文时，才去触发翻译
    if re.search(r'[\u4e00-\u9fa5]', raw_query):
        try:
            t_trans = time.time()
            # 🔥 CLIP视觉翻译优化：添加电影海报视觉上下文，避免缩写歧义
            # "人工智能" → "AI" 太短会匹配到 "爱" 相关海报
            # 策略：用 "artificial intelligence movie poster" 而非 "AI"
            clip_hint = f"{raw_query} movie film poster visual"
            search_query_en = GoogleTranslator(source='auto', target='en').translate(clip_hint)
            # 如果翻译结果太短（<5字符），回退到带完整上下文的版本
            if search_query_en and len(search_query_en.strip()) < 5:
                search_query_en = GoogleTranslator(source='auto', target='en').translate(raw_query)
            logger.info(f"[翻译成功] 耗时: {time.time() - t_trans:.2f}s -> '{search_query_en}'")
        except Exception as e:
            logger.warning(f"[翻译超时/失败] 直接使用原词。报错: {e}")

    # =================================================
    # 2. 核心：调用全局 CLIP 模型进行多模态比对 (矩阵极速版)
    # =================================================
    use_vector_search = False
    sorted_movies = []

    # 确保只筛选数据库中【真的存有视觉向量】的电影
    candidates = Movie.objects.filter(poster_embedding_json__isnull=False).exclude(poster_embedding_json='')

    # 从 warmup 中读取全局预热的资源
    clip_model = VISUAL_RESOURCES.get("model")
    all_embs = VISUAL_RESOURCES.get("movie_embeddings")
    all_movies = VISUAL_RESOURCES.get("movie_objects")

    if clip_model is not None and all_embs is not None:
        try:
            t_clip = time.time()

            # 1. 把文本变成高维向量
            text_emb = clip_model.encode([search_query_en])[0]

            # 2. 利用 Numpy 的矩阵乘法一次性计算所有相似度，代替 for 循环！
            text_norm = text_emb / np.linalg.norm(text_emb)
            embs_norm = all_embs / np.linalg.norm(all_embs, axis=1, keepdims=True)
            similarities = np.dot(embs_norm, text_norm)

            # 3. 按相似度从高到低排序，直接取前 60 名的索引
            top_indices = np.argsort(similarities)[::-1][:60]

            # 4. 根据索引从全局电影对象列表中取出电影
            sorted_movies = [all_movies[i] for i in top_indices]
            use_vector_search = True

            logger.info(f"[CLIP 极速命中] 成功计算 {len(all_movies)} 部电影的向量相似度 耗时: {time.time() - t_clip:.4f}s")

        except Exception as e:
            logger.warning(f"[CLIP 异常] 向量计算出错: {e}")
            import traceback
            traceback.print_exc()

    # =================================================
    # 3. 文本降级兜底 (Fallback)
    # =================================================
    if not use_vector_search:
        if clip_model is None:
            logger.warning("[降级原因] 全局 VISUAL_RESOURCES['model'] 未加载。")
        elif all_embs is None:
            logger.warning("[降级原因] 全局 VISUAL_RESOURCES['movie_embeddings'] 矩阵为空！")

        sorted_movies = Movie.objects.filter(poster_file__isnull=False).filter(
            Q(poster_caption__icontains=search_query_en) |
            Q(title__icontains=raw_query)
        ).order_by('-vote_count')

    # =================================================
    # 4. 分页与数据封装
    # =================================================
    page_object = Pagination(request, sorted_movies, page_size=12)

    results = []
    for movie in page_object.page_queryset:
        poster_url = movie.poster_file.url if movie.poster_file else DEFAULT_POSTER
        # 让前端卡片显示是 向量命中 还是 文本命中
        prefix = "🎯[视觉匹配] " if use_vector_search else "📖[文本兜底] "

        results.append({
            'id': movie.id,
            'title': movie.title,
            'score': float(movie.score) if movie.score else 0.0,
            'poster_url': poster_url,
            'caption': prefix + (movie.poster_caption[:40] + "..." if movie.poster_caption else "已提取视觉特征")
        })

    logger.info(f"[视觉搜索总耗时]: {time.time() - t_start:.2f}s")

    return render(request, 'search_visual.html', {
        'visual_query': raw_query,
        'movies': results,
        'page_string': page_object.html()
    })
# 尝试引入图谱工具
try:
    from myapp.utils.graph_rag import graph, query_graph_rag
except ImportError:
    graph = None






# --- 辅助函数：将文本中的 "《电影》(ID:123)" 替换为 HTML 链接 ---
def inject_movie_links(text):
    """
    V82 增强版：同时匹配书名号和 ID，确保跳转成功
    """
    # 匹配 《电影名》(ID:123) 或 《电影名》
    pattern = re.compile(r'《([^《》]+?)》(?:\(ID:(\d+)\))?')

    def replace_link(match):
        title = match.group(1).strip()
        movie_id = match.group(2)  # 提取 ID 组

        movie = None
        if movie_id:
            # 1. 优先根据 ID 匹配 (最准)
            movie = Movie.objects.filter(pk=movie_id).first()

        if not movie:
            # 2. 如果没 ID 或 ID 匹配失败，根据标题精确匹配
            movie = Movie.objects.filter(title__iexact=title).first()

        if movie:
            url = f"/movie/{movie.pk}/"
            return f'<a href="{url}" target="_blank" class="chat-movie-link">《{title}》</a>'
        return f'《{title}》'

    return pattern.sub(replace_link, text)


def classify_intent_fast(text):
    """
    基于规则的极速意图分类
    """
    words = list(jieba.cut(text))

    # 规则 1: 画像查询
    self_keywords = {'我', '喜欢', '口味', '偏好', '分析', '总结', '画像', '报告'}
    if '我' in words and len(set(words) & self_keywords) >= 1:
        return "QUERY_SELF"

    # 规则 2: 电影查询
    movie_keywords = {'推荐', '介绍', '电影', '片子', '片', '看看', '找', '什么'}
    if len(set(words) & movie_keywords) >= 1:
        return "QUERY_MOVIE"

    return "UNKNOWN"

def classify_intent(user_input):
    """
    极速意图分类器 (规则优先 + LLM兜底)
    """
    # 0. 先跑极速规则

    text = user_input.lower()
    fast_intent = classify_intent_fast(text)
    if fast_intent != "UNKNOWN":
        logger.info(f"Jieba 命中意图: {fast_intent}")
        return fast_intent
    # 1. 规则匹配 (0秒耗时)
    # QUERY_SELF 关键词
    self_keywords = ['我喜欢', '我的口味', '我的偏好', '分析我', '总结我', '画像', '看过']
    if any(k in text for k in self_keywords):
        return "QUERY_SELF"

    # QUERY_MOVIE 关键词
    movie_keywords = ['推荐', '几部', '电影', '片', '导演', '演员', '讲什么', '是谁', '评分', '好看吗']
    if any(k in text for k in movie_keywords):
        return "QUERY_MOVIE"

    # CHAT 关键词
    chat_keywords = ['你好', '谢谢', '再见', 'hi', 'hello']
    if any(k in text for k in chat_keywords):
        return "CHAT"

    # 2. 如果规则没命中，再用小模型兜底 (只在模糊情况下调用)
    try:
        # 为了极速，这里建议用 3B 模型
        llm = ChatOllama(model="qwen3:4b-instruct", temperature=0)
        prompt = f"分类用户意图(QUERY_SELF, QUERY_MOVIE, CHAT)。输入:'{user_input}'。只输出标签。"
        intent = llm.invoke(prompt).content.strip()
        if "SELF" in intent: return "QUERY_SELF"
        if "MOVIE" in intent: return "QUERY_MOVIE"
        return "CHAT"
    except:
        return "QUERY_MOVIE"  # 默认兜底

# --- 新增辅助函数：获取用户画像摘要 ---
# --- 1. 辅助函数：构建用户深度画像 (Context Injection) ---
def get_user_deep_profile(user):
    """
    获取用户的观影偏好画像，用于注入 Prompt
    """
    # 查最近喜欢的高分电影 (评分>=8)
    liked_movies = UserRating.objects.filter(user=user, score__gte=8.0).order_by('-comment_time')

    if not liked_movies.exists():
        return "【用户画像】：新用户，暂无历史偏好，请推荐大众口碑极佳的电影。"

    # 统计偏好类型
    top_genres = Genre.objects.filter(movie__in=liked_movies.values('movie')) \
        .annotate(c=Count('id')).order_by('-c')[:3]
    fav_genres = [g.name for g in top_genres]

    # 获取最近看过的 3 部 (带ID，方便 AI 引用)
    recent_watches = [f"《{r.movie.title}》(ID:{r.movie.id})" for r in liked_movies[:3]]

    return f"""
    【用户画像】
    - 核心口味：偏好 {', '.join(fav_genres)} 类型的电影。
    - 最近高分记录：{', '.join(recent_watches)}。
    - 推荐策略：请基于用户的口味偏好进行关联推荐。
    """


def get_user_interaction_summary_enhanced(user):
    """
    获取用户画像统计数据 (V2: 防幻觉数据版)
    """
    # 1. 基础筛选：高分(>=8) 或 收藏
    high_score_ratings = UserRating.objects.filter(user=user, score__gte=8.0)
    collections = Collect.objects.filter(user=user)

    # 获取涉及的所有电影 ID (去重)
    movie_ids = set(high_score_ratings.values_list('movie_id', flat=True)) | \
                set(collections.values_list('movie_id', flat=True))

    liked_movies = Movie.objects.filter(id__in=movie_ids)
    count = liked_movies.count()

    if count < 3:
        return None, "数据不足"

    # 2. 统计 Top 3 类型 (客观数据)
    top_genres = Genre.objects.filter(movie__in=liked_movies) \
        .annotate(c=Count('id')).order_by('-c')[:3]
    genre_str = "、".join([g.name for g in top_genres])

    # 3. 统计 Top 3 演员 (客观数据)
    top_actors = Actor.objects.filter(movie__in=liked_movies) \
        .annotate(c=Count('id')).order_by('-c')[:3]
    actor_str = "、".join([a.name for a in top_actors])

    # 4. 计算平均口味分 (客观数据)
    avg_score = UserRating.objects.filter(user=user).aggregate(a=Avg('score'))['a'] or 0

    # 5. 最近喜欢的 5 部 (带ID，作为证据)
    # 格式： 《电影名》(类型)
    recent_objs = liked_movies.order_by('-id')[:5]
    recent_list = []
    for m in recent_objs:
        g_name = m.genres.first().name if m.genres.exists() else '剧情'
        recent_list.append(f"《{m.title}》({g_name})")
    recent_str = "，".join(recent_list)

    # 6. 组装给 LLM 的“绝对事实”
    summary_text = f"""
    【用户观影事实数据】
    - 阅片总量：{count} 部
    - 平均打分：{avg_score:.1f} 分 (严厉度参考：>8.5分属宽容，<7.0分属挑剔)
    - 核心偏好类型：{genre_str}
    - 核心偏好演员：{actor_str}
    - 最近喜欢：{recent_str}
    """
    return summary_text, "has_data"


def sanitize_user_input(text):
    """
    [安全核心] 用户输入清洗与防注入 V4 - 超强防护版本（全覆盖+Unicode对抗）

    改进：从 V3 升级为超强防护版本。
    - 一旦检测到任何注入攻击，立即返回特定常数标记，阻断后续处理。
    - 新增：Unicode 规范化、空白字符注入对抗、DAN/越狱高级检测
    """
    if not text:
        return ""

    # 0.5. 【预防Unicode规范化绕过】Unicode NFKC 标准化
    # 攻击者可能利用 Unicode 相似字符（如：ｉｇｎｏｒｅ、ＩＧＮ∘RE）来绕过
    import unicodedata
    text = unicodedata.normalize('NFKC', text)

    # 0.6. 【激进防御】早期删除所有不可见Unicode字符（零宽、BOM、方向控制等）
    # 这一步在 bleach 之前执行，确保没有任何不可见字符残留
    # ⚠️ 重点：用空格替换不可见字符，避免删除后单词连接导致正则失配
    text = re.sub(r'[\u200b-\u200d\ufeff\u200e\u200f\u202a-\u202e\u061c]', ' ', text)
    # 额外：删除其他可能的不可见字符
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', ' ', text)  # 控制字符

    # 1. 基础清洗 (HTML标签 & 空白)
    # 使用 bleach 彻底去除潜在的 XSS 攻击标签
    cleaned = bleach.clean(text, tags=[], strip=True).strip()

    # 1.5. 【白字符防御】删除异常空白字符（零宽字符、不可见字符）
    # 攻击者可能插入零宽字符来迷惑正则匹配
    # （注意：第一遍已在 0.6 阶段删除了所有不可见字符，此处作为二重防御）
    cleaned = re.sub(r'[\u200b-\u200d\ufeff\u200e\u200f\u202a-\u202e\u061c]', '', cleaned)

    # 2. 【物理防御】长度截断
    # 大多数复杂的 Jailbreak (越狱) 提示词都需要很长的铺��� (如 DAN 模式)
    # 对于电影推荐场景，300个字通常足够了，超过直接截断
    if len(cleaned) > 300:
        logger.warning(f"检测到超长输入 ({len(cleaned)}字)，已截断。")
        cleaned = cleaned[:300]

    # 3. 【干扰防御】特殊符号清洗
    # 攻击者常利用 ```, ###, <system> 等符号来欺骗 LLM 认为是系统指令
    sensitive_chars = ['{', '}', '[', ']', '```', '###', 'User:', 'System:', 'Assistant:',
                       '<system>', '</system>', '<command>', '<instruction>']
    for char in sensitive_chars:
        cleaned = cleaned.replace(char, ' ')

    # 4. 【Circuit Breaker 防御】正则对抗 (Prompt Injection Patterns)
    # 🔥 改进：不再使用 re.sub 替换，而是用 re.search 检测。
    # 一旦匹配到任何攻击正则，立即触发熔断，返回危险标记常数。
    # 覆盖：指令覆盖、角色扮演、套取设定、重复指令、DAN/越狱、权限提升
    injection_patterns = [
        # --- 英文攻击（第一层） ---
        r'ignore\s+(all\s+)?(previous|prior|above)\s+instruct',  # Ignore previous instructions
        r'disregard\s+(all\s+)?(rules|guidelines|instruction)',  # Disregard rules
        r'you\s+are\s+now',  # You are now (Roleplay)
        r'act\s+as\s+a',  # Act as a...
        r'repeat\s+the\s+above',  # Repeat the above (Leak context)
        r'system\s+prompt',  # System prompt
        r'developer\s+mode',  # Developer mode
        
        # --- 英文攻击（第二层：高级/DAN/越狱） ---
        r'do\s+anything\s+now',  # DAN 模式
        r'dan\s+mode',
        r'[^a-z]dan[^a-z]',  # 隔离 DAN 关键字
        r'jailbreak',  # 越狱
        r'bypass\s+(.*?)(filter|rule|restrict)',
        r'unlimi.*prompt',
        r'gpt[4-9]',  # 冒充更高级模型
        r'assume\s+(role|persona|the\s+role)',  # 假设角色（增强型：覆盖"assume the role"）
        r'pretend\s+to\s+be',
        r'roleplay\s+as',
        r'speak\s+as\s+if',
        
        # --- 英文攻击（第三层：信息泄露） ---
        r'reveal\s+(.*?)(secret|password|api|key)',
        r'show\s+me\s+(your|the).*(system|internal|secret|api)',  # 增强型：覆盖"Show me your API"
        r'what\s+is\s+your.*prompt',
        r'give\s+me.*instruction',
        r'extract.*initial',
        r'display.*api.*key',  # 直接覆盖"display API keys"变体
        
        # --- 中文攻击（第一层） ---
        r'忽略.*(之前|所有|原有|上述).*(指令|规则|限制)',
        r'无视.*(之前|所有).*(规则|设置|指令)',
        r'忘记.*(你|自己).*是谁',
        r'你的.*(设定|prompt|提示词|系统指令|初始化)',
        r'重复.*(上面|之前|上文).*的内容',
        r'现在.*(开始|是).*角色',
        r'扮演.*(猫娘|上帝|黑客|医生|律师)',
        r'输出.*(初始化|开头|最开始).*指令',
        r'把.*(上文|上面).*翻译',
        
        # --- 中文攻击（第二层：高级/DAN/越狱） ---
        r'do\s*anything\s*now|DAN模式|越狱',
        r'突破.*(限制|规则)',
        r'打破.*(限制|规则)',
        r'绕过.*(过滤|规则|限制)',
        r'假设.*你是',
        r'我要你.*角色',
        r'我现在要你',
        r'从现在开始.*(你|您)就是',
        r'扮演.*(管理员|admin)',  # 新增：覆盖管理员角色扮演
        
        # --- 中文攻击（第三层：信息泄露） ---
        r'告诉我.*(秘密|密码|系统提示|api|密钥)',  # 增强型：加入api和密钥
        r'揭露.*(系统|初始|设定|密码|api)',
        r'你的.*初始化.*指令',
        r'泄露.*(你的|系统|密钥|api)',
        r'暴露.*(内部|设定|密码|api)',
        r'展示.*(源代码|原始指令|密钥|api)',
    ]

    # 🔥 执行 Circuit Breaker 检测（物理熔断）
    for p in injection_patterns:
        # 使用 re.search 检测，而非 re.sub 替换
        if re.search(p, cleaned, flags=re.IGNORECASE):
            # 立即触发熔断，返回危险标记常数
            logger.warning(f"[Prompt Injection 拦截 V4] 检测到高危注入模式: {p}")
            logger.warning(f"原始恶意输入: {text[:100]}")
            return "MALICIOUS_INJECTION_DETECTED"

    # 5. 【额外防御】统计异常特殊字符占比
    # 如果特殊符号超过 40%，可能是混淆攻击
    special_char_count = sum(1 for c in cleaned if not c.isalnum() and c not in ' \t\n，。！？：；""''（）')
    if len(cleaned) > 10 and special_char_count / len(cleaned) > 0.4:
        logger.warning(f"[Prompt Injection 拦截] 检测到异常字符占比过高 ({special_char_count}/{len(cleaned)})")
        return "MALICIOUS_INJECTION_DETECTED"

    # 6. 二次检查：如果清洗后变成空了 (说明全是攻击符号)，返回兜底
    if not cleaned.strip():
        return ""

    return cleaned.strip()


def clean_markdown_marks(text):
    if not text: return ""

    original_text = text  # 备份原文本



    # 原有逻辑
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 去粗体
    text = re.sub(r'#+\s', '', text, flags=re.MULTILINE)  # 去标题
    text = text.replace("```html", "").replace("```", "")  # 去代码块
    text = re.sub(r'\n{3,}', '\n\n', text)  # 去多余换行

    cleaned_text = text.strip()



    return cleaned_text


def extract_query_constraints(text):
    """
    🔥 从用户自然语言中提取结构化查询条件（年份范围、类型等）。
    
    例如：
        "推荐1995年以来的科幻片"  → {'year_from': 1995, 'genre': '科幻'}
        "2020年以后的动作电影"    → {'year_from': 2020, 'genre': '动作'}
        "最近好看的电影"          → {}
        "90年代的恐怖片"          → {'year_from': 1990, 'year_to': 1999, 'genre': '恐怖'}
    
    Returns:
        dict: {'year_from': int|None, 'year_to': int|None, 'genre': str|None, 'region': str|None}
    """
    constraints = {'year_from': None, 'year_to': None, 'genre': None, 'region': None}

    # --- 1. 年份提取（支持多种表达） ---
    # 模式 A: "1995年以来/以后/之后/后的" / "1995年以后上映的"
    year_after_patterns = [
        r'(\d{4})\s*(?:年)?(?:以来|以后|之后|过后|后的|后上映|年后)',
        r'(?:从|自)\s*(\d{4})\s*(?:年)?(?:开始|起)',
    ]
    for pat in year_after_patterns:
        m = re.search(pat, text)
        if m:
            constraints['year_from'] = int(m.group(1))
            break

    # 模式 B: "2010年之前/以前的"
    year_before_patterns = [
        r'(\d{4})\s*(?:年)?(?:之前|以前|前的)',
    ]
    if not constraints['year_from']:
        for pat in year_before_patterns:
            m = re.search(pat, text)
            if m:
                constraints['year_to'] = int(m.group(1))
                break

    # 模式 C: "90年代" → 1990-1999 / "2000年代" → 2000-2009
    decade_patterns = [
        r'(\d{2})\s*年代',  # "90年代", "00年代"
    ]
    for pat in decade_patterns:
        m = re.search(pat, text)
        if m:
            decade_str = m.group(1)
            if int(decade_str) >= 30:  # "90年代" → 1990
                constraints['year_from'] = 1900 + int(decade_str)
                constraints['year_to'] = 1900 + int(decade_str) + 9
            else:  # "00年代", "10年代", "20年代" → 2000, 2010, 2020
                constraints['year_from'] = 2000 + int(decade_str)
                constraints['year_to'] = 2000 + int(decade_str) + 9
            break

    # 模式 D: "2020年的电影" → 精确年份
    if not constraints['year_from'] and not constraints['year_to']:
        exact_year = re.search(r'(\d{4})\s*年', text)
        if exact_year:
            yr = int(exact_year.group(1))
            if 1900 <= yr <= 2030:
                constraints['year_from'] = yr
                constraints['year_to'] = yr

    # --- 2. 电影类型提取 ---
    genre_map = {
        '科幻': '科幻', '喜剧': '喜剧', '动作': '动作', '爱情': '爱情',
        '恐怖': '恐怖', '悬疑': '悬疑', '动画': '动画', '剧情': '剧情',
        '战争': '战争', '犯罪': '犯罪', '奇幻': '奇幻', '冒险': '冒险',
        '纪录片': '纪录片', '历史': '历史', '音乐': '音乐', '家庭': '家庭',
        '西部': '西部', '武侠': '武侠', '传记': '传记', '惊悚': '惊悚',
        '灾难': '灾难', '短片': '短片', '古装': '古装', '仙侠': '仙侠',
    }
    for genre_keyword in genre_map:
        if genre_keyword in text:
            constraints['genre'] = genre_map[genre_keyword]
            break

    # --- 3. 地区提取 ---
    region_keywords = ['中国', '美国', '日本', '韩国', '英国', '法国', '德国', '印度', '泰国', '香港', '台湾', '好莱坞']
    for rk in region_keywords:
        if rk in text:
            constraints['region'] = rk
            break

    return constraints


def detect_negative_intent(text):
    neg_patterns = [r'不要(.*?)', r'不看(.*?)', r'讨厌(.*?)', r'不喜欢(.*?)', r'除了(.*?)']
    neg_keywords = []
    for pattern in neg_patterns:
        match = re.search(pattern, text)
        if match:
            target = match.group(1).strip()[:4]
            if target: neg_keywords.append(target)
    return bool(neg_keywords), neg_keywords


def check_domain_entities(text):
    logic_keywords = ['和', '与', '对比', '区别', '一样', '类似', '像', '不要', '讨厌', '除了']
    if any(k in text for k in logic_keywords) and len(text) > 2: return True

    words = pseg.cut(text)
    potential_entities = [
        p.word for p in words
        if (p.flag.startswith('n') or p.flag in ['i', 'l']) and len(p.word) > 1
    ]
    if not potential_entities: return False

    if Movie.objects.filter(title__in=potential_entities).exists(): return True
    if Actor.objects.filter(name__in=potential_entities).exists(): return True
    if Genre.objects.filter(name__in=potential_entities).exists(): return True
    return False


# --- 1. 升级意图分类器 (加入视觉意图) ---
def classify_intent_advanced(text, history_msgs=None):
    """
    [意图分类器 V108] 纯规则与正则驱动版
    彻底移除小模型，零显存占用，解决 OOM 风险。
    """
    import re
    text_clean = text.lower().strip()

    # --- 0. 优先处理追问 (Follow-up) 逻辑 ---
    # 扩大追问关键词范围
    follow_up_kws = ['再', '还', '换', '继续', '更多', '其他', '类似', '别的', '再来']
    is_follow_up = any(k in text_clean for k in follow_up_kws) and len(text_clean) < 15

    if is_follow_up and history_msgs:
        last_ai_msg = history_msgs[0].message if history_msgs else ""
        # 只要上一轮 AI 提到了关键 HTML 标记或视觉词，就判定为追问海报
        if any(k in last_ai_msg for k in ["visual-card", "海报库", "图库", "画面"]):
            return "QUERY_VISUAL"
        return "QUERY_MOVIE"

    # --- 1. 强特征规则匹配 (正则表达式版) ---

    # A. 视觉意图：包含视觉词 + 动作词，或者明确提到“海报/海拔/封面”
    visual_pattern = r'(海报|海拔|封面|图|照片|长什么样|画面|视觉|色调|风格|样子|看下)'
    action_pattern = r'(找|看|搜|有|显|来|推|给)'
    if re.search(visual_pattern, text_clean):
        if re.search(action_pattern, text_clean) or len(text_clean) < 6:
            return "QUERY_VISUAL"

    # B. 用户画像：分析、口味、报告等
    # B-1. 🔥 画像驱动推荐（必须在 B-2 前检测，防止被"画像"关键词直接吞入 QUERY_SELF）
    #      覆盖："根据我的画像推荐" / "按我的偏好推几部" / "看我的口味来点片"
    _profile_kw = r'(画像|偏好|口味|喜好|品位|历史记录|看过的|我喜欢)'
    _rec_action = r'(推荐|推几部|给我推|给我找|帮我找|推点|来几部|来点|找几部)'
    if re.search(_profile_kw, text_clean) and re.search(_rec_action, text_clean):
        return "QUERY_PROFILE_REC"

    # B-2. 纯画像分析（无推荐动词 → 只做侧写/分析）
    self_pattern = r'(分析我|口味|偏好|画像|我看过|我的|总结我|报告)'
    if re.search(self_pattern, text_clean):
        return "QUERY_SELF"

    # C. 榜单与最新：排行、高分
    rank_pattern = r'(热门|榜单|高分|前十|排名|排行|最火)'
    if re.search(rank_pattern, text_clean) and any(k in text_clean for k in ['电影', '片', '剧']):
        return "QUERY_RANK"

    # 🔥 优化：纯新片查询 vs 复合查询精准区分
    # 只有当用户"单纯"询问新片时才路由到 QUERY_NEW
    # 一旦包含质量、类型、年份等复合条件，应路由到 QUERY_MOVIE（走 RAG 检索）
    new_pattern = r'(最新|新出|上映|刚出|最近)'
    quality_pattern = r'(好看|精彩|经典|高分|推荐|评分|口碑|值得|不错|好看|佳作|神作)'
    year_pattern = r'(\d{4})\s*(?:年|以来|以后|之前|之前上映|后上映|年以后|年之前|至今|到今)'
    type_pattern = r'(科幻|喜剧|动作|爱情|恐怖|悬疑|动画|剧情|战争|犯罪|奇幻|冒险|纪录片|历史|音乐|家庭|西部|武侠|传记|惊悚|灾难|短片)'

    if re.search(new_pattern, text_clean) and any(k in text_clean for k in ['电影', '片', '剧']):
        has_quality = re.search(quality_pattern, text_clean)
        has_year = re.search(year_pattern, text_clean)
        has_type = re.search(type_pattern, text_clean)
        has_any_genre_kw = any(k in text_clean for k in ['类型', '题材', '片'])

        # 如果包含质量词、年份条件、类型条件 → 这是复合查询，走 QUERY_MOVIE
        if has_quality or has_year or has_type:
            return "QUERY_MOVIE"
        return "QUERY_NEW"

    # --- 2. 领域实体与推荐语感拦截 ---
    # 只要调用 check_domain_entities 命中数据库，基本就是求片
    if check_domain_entities(text_clean):
        if any(k in text_clean for k in ['对比', '区别', 'vs', '不同']):
            return "QUERY_COMPARISON"
        return "QUERY_MOVIE"

    # 额外的推荐语感补拦截：比如“有没有适合晚上看的”
    movie_vibe_kws = ['看点', '推荐', '电影', '片子', '介绍', '讲什么', '好看吗', '有哪些', '找点']
    if any(k in text_clean for k in movie_vibe_kws):
        return "QUERY_MOVIE"

    # --- 3. 最终兜底 ---
    # 如果包含比较长的输入且不符合上述规则，为了保证体验，默认进 QUERY_MOVIE 让 RAG 尝试检索
    if len(text_clean) > 8:
        return "QUERY_MOVIE"

    return "CHAT"


# --- 2. 新增：视觉搜索匹配逻辑 ---
def search_visual_match(user_input, limit=3):
    """
    [视觉搜索核心引擎 V105]
    功能：自动纠错、多词模糊检索、生成前端卡片组件
    """
    # 1. 基础鲁棒性处理：纠正输入法常见的误打 (海拔 -> 海报)
    corrected_input = user_input.replace("海拔", "海报").replace("海报", "")

    # 2. 提取核心视觉词 (调用我们修复过的 extract_visual_keywords)
    keywords_str = extract_visual_keywords(corrected_input)

    # 防空处理：如果提取完只剩废话，直接引导
    if not keywords_str or len(keywords_str) < 2:
        return None, "请提供更具体的画面描述，例如：'赛博朋克风格的霓虹灯' 或 '深海中的巨兽'。"

    logger.info(f"[Visual Search] 处理后的检索词: {keywords_str}")

    # 3. 构造多维度模糊搜索逻辑 (ORM Q 对象)
    # 策略：只要电影标题或 AI 生成的海报描述 (poster_caption) 中包含任何一个关键词，即进入候选
    search_terms = keywords_str.split(' ')
    query_obj = Q()
    for term in search_terms:
        if len(term) >= 1:
            query_obj |= Q(poster_caption__icontains=term)
            query_obj |= Q(title__icontains=term)

    # 4. 执行数据库查询：必须有海报文件，且按人气 (vote_count) 排序以保证质量
    candidates = Movie.objects.filter(query_obj).filter(
        poster_file__isnull=False
    ).exclude(poster_file='').order_by('-vote_count')[:limit]

    # 5. 兜底策略：如果多词联合搜索无结果，降级为单词匹配 (针对“科技风格的海报”)
    if not candidates.exists() and len(search_terms) > 1:
        # 尝试只搜索最核心的第一个词
        primary_term = search_terms[0]
        candidates = Movie.objects.filter(
            Q(poster_caption__icontains=primary_term) |
            Q(title__icontains=primary_term)
        ).filter(poster_file__isnull=False)[:limit]

    # 6. 最终结果判定
    if not candidates.exists():
        # 返回 None 让 ajax_chat 触发 QUERY_VISUAL_RETRY 逻辑
        return None, f"抱歉，馆藏图库中暂未找到符合 '{keywords_str}' 视觉特征的海报。"

    # 7. 构建响应式 HTML 卡片组件 (直接注入聊天气泡)
    # 使用 Flex 布局实现横向滚动效果
    html_cards = (
        '<div class="visual-search-container" style="display: flex; gap: 15px; '
        'overflow-x: auto; padding: 15px 5px; scrollbar-width: thin; -webkit-overflow-scrolling: touch;">'
    )

    for m in candidates:
        # 截断描述，防止卡片高度失控
        cap_short = m.poster_caption[:45] + "..." if m.poster_caption else "视觉特征已收录"
        poster_url = m.poster_file.url
        detail_url = reverse('movie_detail', args=[m.id])

        card = f"""
        <div class="visual-card" style="min-width: 140px; max-width: 140px; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 10px rgba(0,0,0,0.15); border: 1px solid #eee;">
            <a href="{detail_url}" target="_blank" style="text-decoration: none; color: inherit;">
                <div style="width: 100%; height: 190px; overflow: hidden; background: #f0f0f0;">
                    <img src="{poster_url}" style="width: 100%; height: 100%; object-fit: cover; transition: all 0.3s;"
                        onmouseover="this.style.opacity='0.8'" onmouseout="this.style.opacity='1'">
                </div>
                <div style="padding: 10px; background: white;">
                    <div style="font-size: 13px; font-weight: bold; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; color: #2c3e50;">{m.title}</div>
                    <div style="font-size: 10px; color: #7f8c8d; margin-top: 5px; line-height: 1.4; height: 2.8em; overflow: hidden;">{cap_short}</div>
                </div>
            </a>
        </div>
        """
        html_cards += card

    html_cards += '</div>'

    # 提示语与卡片拼接
    msg = f"为您在馆藏中找到 {candidates.count()} 张符合【{keywords_str}】风格的海报："
    return html_cards, msg


def get_chat_history_context(user, limit=10):
    return ChatHistory.objects.filter(user=user, role='user').order_by('-timestamp')[:limit]


def get_recommended_movies_from_history(user, limit=5):
    """
    提取最近 AI 回复中的电影名，用于去重
    """
    recent_ai = ChatHistory.objects.filter(user=user, role='ai').order_by('-timestamp')[:limit]
    titles = []
    for msg in recent_ai:
        # 提取 《...》
        found = re.findall(r'《(.*?)》', msg.message)
        titles.extend(found)
    return list(set(titles))


def extract_visual_keywords(text):
    """
    NLP 预处理：从用户句子中提取核心视觉关键词
    修复：解决 pair object 无法解包导致的 TypeError
    """
    stop_words = {
        '推荐', '找', '搜', '看', '有没有', '帮我', '想', '要', '喜欢',
        '电影', '片子', '海报', '图', '风格', '色调'
    }

    # pseg.cut 返回的是 pair 对象的生成器
    words_pairs = pseg.cut(text)

    keywords = []
    for p in words_pairs:
        # 使用 p.word 和 p.flag 访问
        w, flag = p.word, p.flag

        # 保留名词、形容词、英文、成语等核心词
        if (flag.startswith('n') or flag.startswith('a') or flag == 'eng' or flag == 'i'):
            if w not in stop_words and len(w) > 1:
                keywords.append(w)
        elif re.match(r'[a-zA-Z]+', w):
            keywords.append(w)

    if not keywords:
        return text

    return " ".join(keywords)


# 🔥 关键 1：使用 threading.Lock 替代 asyncio.Lock，确保在 Django WSGI 多线程模式下
# 跨请求的 LLM 并发保护是可靠的。asyncio.Lock 在 Python 3.9 + WSGI 下
# 因每次请求创建独立事件循环，会导致 "Future attached to a different loop" 异常。
LLM_CHAT_LOCK = threading.Lock()


def _build_visual_prompt(user_input, request, is_thinking_mode):
    """
    构建视觉搜索分支的 Prompt

    Returns:
        tuple: (visual_response, final_prompt, temperature)
            - visual_response: 直接返回给前端的HTML内容，无需LLM处理时使用
            - final_prompt: 需要LLM处理的提示词
            - temperature: LLM温度参数
    """
    visual_html, text_response = search_visual_match(user_input)

    if not visual_html:
        logger.info(f"[视觉语义检索] 关键词未命中，正在深度挖掘海报意象: {user_input}")

        # 利用 RAG 函数搜海报描述
        expanded_v_query = f"{user_input} 视觉元素 构图 色调 风格"
        semantic_v_context = query_vector_rag(expanded_v_query, k=15)
        found_ids = re.findall(r'\(ID:(\d+)\)', semantic_v_context)

        if found_ids:
            # ✅ 注意：这里不能用 @sync_to_async，因为本函数本身是同步的
            # _build_visual_prompt 通过 sync_to_async(builder)() 从 async 上下文调用
            # 所以内部直接调用 ORM 即可，不需要再包一层异步
            rec_movies = list(Movie.objects.filter(id__in=found_ids[:6]))
            if rec_movies:
                visual_html = '<div class="visual-search-container" style="display: flex; gap: 12px; overflow-x: auto; padding: 10px 0;">'
                for rm in rec_movies:
                    p_url = rm.poster_file.url if rm.poster_file else "/static/img/no_poster.png"
                    d_url = reverse('movie_detail', args=[rm.id])
                    visual_html += f"""
                        <div class="visual-card" style="min-width: 120px; max-width: 120px; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border: 1px solid #eee;">
                            <a href="{d_url}" target="_blank" style="text-decoration: none;">
                                <img src="{p_url}" style="width: 100%; height: 160px; object-fit: cover;">
                                <div style="padding: 5px; font-size: 11px; font-weight: bold; color: #333; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{rm.title}</div>
                            </a>
                        </div>"""
                visual_html += '</div>'
                text_response = f"根据海报视觉意象，为您找到了几部极具「{user_input}」感官冲击力的作品："

    if visual_html:
        # 直接返回HTML，无需LLM处理
        return f"{text_response}<br>{visual_html}", None, None
    else:
        # 需要LLM引导用户
        final_prompt = f"""
        用户想在馆藏海报库中搜索关于"{user_input}"的视觉风格图，但系统数据库未命中。
        请以"智能观影助手"身份，用极其简短、专业的口吻：
        1. 礼貌说明目前图库中暂未收录符合该具体描述的海报。
        2. 引导用户更换更明确的视觉关联词（如：赛博朋克、冷色调、极简、废土等）。
        严禁进行普通闲聊或随意推荐电影。
        """
        return None, final_prompt, 0.3



# ── 职业集：用于元数据冲突检测 ───────────────────────────────────────────────
# 凡属于"成年人才会从事"的职业代码
_ADULT_OCC_CODES = {1, 2, 3, 4, 5, 6, 7, 11, 13, 15, 16, 17, 18, 20}
# 其中属于高专业度（需要大量受教育年限）的职业
_PROFESSIONAL_OCC_CODES = {1, 2, 3, 5, 6, 11, 15, 16, 20}


def _detect_profile_anomaly(user):
    """
    [多维特征一致性校验器]

    检测用户静态元数据（年龄、职业）之间的特征冲突，
    为 _build_self_profile_prompt 提供决策依据，
    防止模型因脏数据发生"推荐逻辑坍缩（Reasoning Collapse）"。

    返回:
        (anomaly_level, conflict_desc)
        - anomaly_level: 0=无冲突, 1=轻度, 2=严重
        - conflict_desc: 供 LLM 阅读的冲突描述（中文）
    """
    age = user.age
    occ = user.occupation
    if age is None and occ is None:
        return 0, ""

    occ_name = dict(UserInfo.OCCUPATION_CHOICES).get(occ, "未指定") if occ is not None else "未指定"

    try:
        age_int = int(age) if age is not None else None
    except (ValueError, TypeError):
        return 0, ""

    # ── 规则 1：年龄无效值（0 或超过 120）──────────────────────────────
    if age_int is not None and (age_int == 0 or age_int > 120):
        return 2, f"年龄={age_int}岁 为无效/误填值（元数据噪声）"

    # ── 规则 2：年龄 < 6 岁 + 成年职业 → 严重冲突 ───────────────────────
    if age_int is not None and age_int < 6 and occ in _ADULT_OCC_CODES:
        return 2, f"年龄={age_int}岁 与职业「{occ_name}」存在明显矛盾（元数据噪声，疑似误填）"

    # ── 规则 3：年龄 < 14 岁 + 高专业度职业 → 轻度冲突 ─────────────────
    if age_int is not None and age_int < 14 and occ in _PROFESSIONAL_OCC_CODES:
        return 1, f"年龄={age_int}岁 与职业「{occ_name}」存在轻度不一致，可能为未成年用户误选职业"

    return 0, ""


def _build_self_profile_prompt(user, interaction_summary):
    """
    构建影迷画像分析分支的 Prompt
    V2: 含元数据冲突 CoT 防御 + Few-shot 异常处理约束

    新增能力：
    - 调用 _detect_profile_anomaly 进行多维特征一致性校验
    - 检测到严重冲突时，注入 Few-shot 示例，强制 LLM 礼貌告知用户
        并基于交互记录（权重100%）而非异常静态元数据来生成侧写
    - 轻度冲突时降低静态年龄的参考权重
    """
    sex_map = dict(UserInfo.sex_choices)
    occ_map = dict(UserInfo.OCCUPATION_CHOICES)
    u_profile = (
        f"性别：{sex_map.get(user.sex, '保密')} | "
        f"职业：{occ_map.get(user.occupation, '保密')} | "
        f"年龄：{user.age or '保密'}"
    )
    safe_summary = interaction_summary[:500] if interaction_summary else "暂无交互记录"

    # ── 元数据一致性校验 ──────────────────────────────────────────────────
    anomaly_level, conflict_desc = _detect_profile_anomaly(user)

    # ── 根据冲突级别构建不同约束区块 ───────────────────────────────────────
    anomaly_constraint = ""

    if anomaly_level == 2:
        # 严重冲突：完全忽略年龄，注入 Few-shot 冲突处理范例
        anomaly_constraint = f"""
⚠️ 【系统检测：元数据严重冲突 (Metadata Conflict Detected)】
检测结果：{conflict_desc}

冲突处理协议（必须严格执行）：
1. 年龄数据权重 = 0%，严禁从年龄推断用户性格、兴趣或标签。
2. 职业数据 + 交互行为记录 权重 = 100%，侧写完全基于这两项。
3. 必须在侧写结果中礼貌告知用户其年龄数据疑似有误，说明侧写依据。
4. 严禁输出"儿童向"、"亲子"、"小朋友"、"玩具"等年龄衍生标签。

【Few-shot 冲突处理示例 (必须参照此格式)】：

    ▸ 输入特征：年龄=1，职业=科学家，近期高分影片：《星际穿越》《人工智能》《她》
    ▸ 内部 CoT 推理：
        - 检测到年龄(1岁)与职业(科学家)的严重矛盾 → 元数据噪声。
        - 决策：完全丢弃年龄信息，以职业(科学家)+交互行为(科幻/AI主题高评)为基础构建侧写。
        - 应礼貌提醒用户检查档案，再给出基于真实行为的侧写。
    ▸ 正确输出格式：
        您的档案年龄数据似乎出现了一点小状况（显示为1岁？），不影响我对您观影品位的分析！
        您是一位【星际思索者】。思维严谨理性，对科技与人性的交叉地带有深度关注。勋章：奇点见证者。
    ▸ 严禁输出（反例）：
        您是一位【童趣探险家】，热爱玩具与儿童冒险故事。（← 这是从错误年龄推断出的无效侧写）
"""

    elif anomaly_level == 1:
        # 轻度冲突：降低年龄权重，提示 LLM
        anomaly_constraint = f"""
ℹ️ 【系统提示：元数据轻度不一致】
{conflict_desc}
处理建议：以职业背景与交互行为记录为主要依据（权重 80%），年龄仅作次要参考（权重 20%）。
若两者判断结果矛盾，优先采用行为数据的结论。
"""

    # ── 最终 Prompt 组装 ───────────────────────────────────────────────────
    final_prompt = f"""你是一位精准的影迷侧写师，任务是基于用户档案和观影行为生成个性化侧写。
要求：直接输出结果，严禁 Markdown，总字数控制在 100 字以内。
{anomaly_constraint}
【正常侧写示例（无冲突时的参考格式）】：
您是一位【观影漫游者】。性格沉稳，偏爱赛博朋克深处的哲学思考。勋章：银翼侦探。

用户档案：{u_profile}
近期交互行为：{safe_summary}

请按以下格式输出（必须包含两个标签）：
[你的思考过程]（在此进行多维特征一致性检验：年龄与职业是否一致？行为与职业是否一致？）
[你的侧写结果]（最终对用户的侧写，若有数据冲突需礼貌告知）
"""
    return None, final_prompt, 0.2



def _build_profile_rec_prompt(user, interaction_summary):
    """
    [混合检索召回分支] QUERY_PROFILE_REC 专用 Prompt 构建器
    ──────────────────────────────────────────────────────
    流程：
        1. build_user_profile_text()  →  自然语言用户画像
        2. hybrid_recall_recommend()  →  三路召回 + RRF 融合
        3. get_kg_subgraph()          →  Neo4j 知识图谱关联路径
        4. 把召回结果 + 图谱上下文注入 Prompt
    """
    # --- 1. 调用混合检索 ---
    try:
        movies_list, profile_text, stats = hybrid_recall_recommend(user, top_k=8)
    except Exception as e:
        logger.warning(f"[ProfileRec] hybrid_recall 失败，降级: {e}")
        movies_list, profile_text, stats = [], "", {}

    # --- 2. 序列化召回结果 ---
    if movies_list:
        movie_lines = []
        for m in movies_list[:8]:
            genres_str = "、".join(g.name for g in m.genres.all()[:3])
            summary_short = (m.summary or "")[:60].replace("\n", " ")
            score_str = str(m.score) if m.score else "暂无"
            movie_lines.append(
                f"《{m.title}》(ID:{m.id}) | 评分:{score_str} | 类型:{genres_str} | {summary_short}..."
            )
        movie_context = "\n".join(movie_lines)
    else:
        movie_context = "（暂无召回结果，请先评分或收藏几部电影）"

    # --- 3. 查询 Neo4j 知识图谱：基于用户最偏好的类型做图谱拓扑扩展 ---
    kg_context = ""
    try:
        # 从 profile_text 中提取第一个偏好类型关键词作为图谱查询词
        kg_topic = ""
        if profile_text:
            # 尝试提取"喜欢的电影类型：XXX" 中的第一个类型名
            import re as _re
            m_genre = _re.search(r'喜欢的电影类型[：:]\s*([^；;]+)', profile_text)
            if m_genre:
                kg_topic = m_genre.group(1).split('、')[0].strip()
        if not kg_topic and movies_list:
            # 降级：取第一个召回电影的第一个类型
            first_genres = list(movies_list[0].genres.all()[:1])
            kg_topic = first_genres[0].name if first_genres else ""

        if kg_topic:
            user_history_mids = list(
                UserRating.objects.filter(user=user, score__gte=7.5)
                .values_list('movie_id', flat=True)[:20]
            )
            kg_context = get_kg_subgraph(kg_topic, user_history_mids=user_history_mids, max_triples=8)
    except Exception as e:
        logger.warning(f"[ProfileRec] KG subgraph 查询失败: {e}")

    # --- 4. 召回来源说明 ---
    source_note = (
        f"[混合检索] 向量召回{stats.get('vector',0)}条 · "
        f"内容召回{stats.get('content',0)}条 · "
        f"模型召回{stats.get('model',0)}条 → RRF融合"
    ) if stats else ""

    # --- 5. 组装 Prompt ---
    safe_summary = (interaction_summary or "暂无记录")[:300]

    kg_section = ""
    if kg_context:
        kg_section = f"""
【知识图谱关联路径 (Neo4j)】
{kg_context}
⚡ 若路径中存在合适候选，推荐时请引用图谱连线关系（如同导演、同类型）作为理由依据。
"""

    final_prompt = f"""你是一位热情的智能观影助手，正在根据用户的个性化画像向用户推荐电影。

【用户画像】
{profile_text}

【用户近期行为摘要】
{safe_summary}

【系统已通过混合检索（Vector语义+内容特征+模型预测→RRF融合）召回以下候选影片】
{movie_context}

{source_note}
{kg_section}
【你的任务】
请从上方候选影片中挑选3~5部最适合该用户的电影，用活泼自然的语气进行个性化点评。
若知识图谱路径中有同导演或同类型关联，请在对应电影的推荐理由中自然引用。
要求：
- 每部电影单独一段，格式为：《电影名》(ID:xxx) + 推荐理由（30字以内）
- 结合用户的偏好类型和观影历史来解释推荐原因
- 严禁编造不在候选列表中的电影
- 不使用 Markdown，纯文本输出，总字数不超过350字
"""
    return None, final_prompt, 0.4


def _build_rank_prompt(user_input):
    """构建排行榜分支的 Prompt"""
    # 根据用户输入选择查询方式
    if '高分' in user_input:
        movies = Movie.objects.filter(vote_count__gt=1000).order_by('-score')[:5]
    else:
        movies = Movie.objects.order_by('-vote_count')[:5]

    # 构建电影列表字符串
    movie_list = "\n".join([f"- 《{m.title}》 (ID:{m.id}) 评分:{m.score}" for m in movies])

    # 构建最终提示词
    final_prompt = f"实时榜单数据如下：\n{movie_list}\n请以智能观影助手身份进行极简点评，引导用户点击观看。"
    return None, final_prompt, 0.3


def _build_new_movies_prompt(user_input=None):
    """构建最新电影分支的 Prompt"""
    movies = Movie.objects.filter(date__isnull=False).order_by('-date')[:5]
    movie_list = "\n".join([f"- 《{m.title}》 (ID:{m.id}) 上映日:{m.date}" for m in movies])
    final_prompt = f"最新入库的电影如下：\n{movie_list}\n请以智能观影助手身份热情安利这些新片。"
    return None, final_prompt, 0.3


def get_kg_subgraph(topic_keyword, user_history_mids=None, max_triples=12):
    """
    [KAG 核心桥接层] 从 Neo4j 提取与主题相关的图谱子图，
    并序列化为 LLM 可直接推理的三元组字符串。

    查询路径（优先级顺序）：
    A. 主题词 → 类型/标题/摘要 → 电影 → 导演（主题拓扑子图）
    B. 主题词 → 演员名 → 电影（演员关联补充）
    C. 用户历史高分电影 → 同导演 → 未看新片（个性化关联）
    """
    if not topic_keyword:
        return ""

    _key_hash = hashlib.md5(topic_keyword.encode("utf-8")).hexdigest()
    hist_hash = hashlib.md5(str(user_history_mids).encode()).hexdigest()[:8]
    cache_key = f"kg_subgraph_v3_{_key_hash}_{hist_hash}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    # 复用模块级 neo_graph 单例，避免每次请求重新建连
    _g = neo_graph
    if _g is None:
        logger.warning("[KAG] neo_graph 未初始化，图谱上下文降级为空")
        return ""

    triples = []
    seen_mids = set()

    try:
        # --- 路径 A: 主题词 → 类型/标题/摘要 匹配 → 电影 → 导演 ---
        cypher_topic = """
        MATCH (m:Movie)<-[:DIRECTED_BY]-(d:Person)
        WHERE m.title CONTAINS $topic
            OR m.summary CONTAINS $topic
        OPTIONAL MATCH (m)-[:BELONGS_TO]->(g:Genre)
        WITH d, m, collect(g.name)[..2] AS genres
        RETURN d.name AS director, m.title AS title, m.mid AS mid, genres
        LIMIT 6
        """
        rows = _g.run(cypher_topic, topic=topic_keyword[:15]).data()
        for r in rows:
            mid = r.get('mid')
            if mid in seen_mids:
                continue
            seen_mids.add(mid)
            genre_str = '/'.join(r.get('genres') or [])
            triples.append(
                f"《{r['title']}》(ID:{mid})--[导演]-->{r['director']}"
                + (f"  |  《{r['title']}》--[类型]-->{genre_str}" if genre_str else "")
            )

        # --- 路径 A2: 类型名精确匹配 → 电影（补充类型维度）---
        if len(triples) < 4:
            cypher_genre = """
            MATCH (m:Movie)-[:BELONGS_TO]->(g:Genre)
            WHERE g.name CONTAINS $topic
            OPTIONAL MATCH (m)<-[:DIRECTED_BY]-(d:Person)
            RETURN m.title AS title, m.mid AS mid,
                    g.name AS genre, d.name AS director
            ORDER BY m.score DESC
            LIMIT 4
            """
            genre_rows = _g.run(cypher_genre, topic=topic_keyword[:10]).data()
            for r in genre_rows:
                mid = r.get('mid')
                if mid in seen_mids:
                    continue
                seen_mids.add(mid)
                triples.append(
                    f"《{r['title']}》(ID:{mid})--[类型]-->{r['genre']}"
                    + (f"  |  导演:{r['director']}" if r.get('director') else "")
                )

        # --- 路径 B: 演员名匹配（演员关联补充）---
        if len(triples) < 6:
            cypher_actor = """
            MATCH (a:Person)-[:ACTED_IN]->(m:Movie)
            WHERE a.name CONTAINS $topic
            OPTIONAL MATCH (m)<-[:DIRECTED_BY]-(d:Person)
            RETURN a.name AS actor, m.title AS title,
                    m.mid AS mid, d.name AS director
            ORDER BY m.score DESC
            LIMIT 4
            """
            actor_rows = _g.run(cypher_actor, topic=topic_keyword[:10]).data()
            for r in actor_rows:
                mid = r.get('mid')
                if mid in seen_mids:
                    continue
                seen_mids.add(mid)
                triples.append(
                    f"《{r['title']}》(ID:{mid})--[主演]-->{r['actor']}"
                    + (f"  |  导演:{r['director']}" if r.get('director') else "")
                )

        # --- 路径 C: 用户历史高分电影 → 同导演 → 未看新片（个性化关联）---
        if user_history_mids:
            cypher_hist = """
            MATCH (h:Movie)<-[:DIRECTED_BY]-(d:Person)-[:DIRECTED_BY]->(m:Movie)
            WHERE h.mid IN $hist_mids
                AND NOT m.mid IN $hist_mids
                AND h <> m
            RETURN h.title AS hist_title, d.name AS director,
                    m.title AS rec_title, m.mid AS rec_mid
            LIMIT 5
            """
            hist_rows = _g.run(cypher_hist, hist_mids=user_history_mids).data()
            for r in hist_rows:
                rec_mid = r.get('rec_mid')
                if rec_mid in seen_mids:
                    continue
                seen_mids.add(rec_mid)
                triples.append(
                    f"《{r['hist_title']}》--[同导演:{r['director']}]"
                    f"-->《{r['rec_title']}》(ID:{rec_mid})"
                )

        result = "\n".join(triples[:max_triples]) if triples else ""
        cache.set(cache_key, result, 60 * 30)  # 缓存 30 分钟
        return result

    except Exception as e:
        logger.error(f"[KAG] 图谱子图查询异常: {e}")
        return ""


def _build_user_prior_context(user, interaction_summary):
    """
    [KAG Expert Prior Generator V1]
    将用户观影事实统计 + 职业画像，转化为 LLM 可直接消费的
    「认知对齐s」，注入推荐 Prompt。

    核心思路：不同职业背景的用户，对同一部电影的关注维度截然不同。
    工程师关注系统逻辑，教师关注人文深度，艺术家关注视觉语言。
    通过 Expert Prior，让 4B 小模型也能输出「职业穿透力」的推荐理由。
    """
    if not interaction_summary:
        return ""

    # ── 职业 → 认知先验映射表 ──────────────────────────────────────────────
    # key: occupation int (同 UserInfo.OCCUPATION_CHOICES)
    OCCUPATION_EXPERT_PRIOR = {
        5: {  # 计算机/程序员
            'label': '计算机/程序员',
            'thought': '用户具有强系统性思维和底层逻辑偏好，应回避空洞情感渲染，寻找具有严密内部逻辑与系统设计感的作品。',
            'req': '推荐硬核一点的科幻片',
            'movie': '《升级》',
            'reason': '该片展示了人工智能对人体硬件接管的精密逻辑，底层控制流程与您的系统性思维高度共振，每一帧都有工程美学。',
        },
        4: {  # 大专/技术
            'label': '技术/大专',
            'thought': '用户具备技术实操背景，偏好逻辑连贯、有现实技术根基的内容，排斥纯视觉轰炸。',
            'req': '推荐逻辑感强的影片',
            'movie': '《模仿游戏》',
            'reason': '图灵破译密码的故事本质是一个系统工程问题，代码与战争双线叙事节奏极具张力，技术背景让您能看懂它的精髓。',
        },
        16: {  # 科学家
            'label': '科研/科学家',
            'thought': '用户习惯批判性思维分析信息，偏好叙事严谨、有科学内核的作品，排斥伪科学和逻辑漏洞。',
            'req': '推荐科幻但科学严谨的影片',
            'movie': '《火星救援》',
            'reason': '每一个救援方案都是真实可执行的科学推演，马克·瓦特尼"科学出奇迹"的思维模式与您的研究范式高度吻合。',
        },
        1: {  # 学术/教育
            'label': '学术/教育',
            'thought': '用户有叙事结构解析力和人文关怀，偏好思辨深度的剧本，历史、教育类主题有天然吸引力。',
            'req': '推荐有人文深度的影片',
            'movie': '《死亡诗社》',
            'reason': '凯廷老师用桌上的一首诗颠覆了整个课堂权力结构，对有教育者视角的您来说，不仅是电影，更是一面镜子。',
        },
        6: {  # 医生/医疗
            'label': '医疗/医生',
            'thought': '用户有高度的生死伦理敏感度和对人体/心理机制的专业认知，宜推荐具备医学真实感或生命伦理深度的作品。',
            'req': '推荐关于人性与生死的影片',
            'movie': '《心灵捕手》',
            'reason': '影片通过心理治疗的真实博弈，触及了自我价值认同的核心议题，治疗逻辑的专业性和人文深度对您来说有双重共鸣。',
        },
        2: {  # 艺术/娱乐
            'label': '艺术/娱乐',
            'thought': '用户对视觉语言和美学构成高度敏感，应优先考虑画面叙事强、导演风格鲜明的作品。',
            'req': '推荐视觉震撼的影片',
            'movie': '《银翼杀手2049》',
            'reason': '维伦纽瓦用荒原与霓虹的极致张力重新定义了赛博朋克美学，每一个镜头构图本身都是一件视觉装置艺术作品。',
        },
        20: {  # 作家
            'label': '作家',
            'thought': '用户对叙事结构和文字张力有专业级敏感度，倾向非线性叙事、元叙事或高度文学性的电影文本。',
            'req': '推荐叙事结构独特的影片',
            'movie': '《改编剧本》',
            'reason': '考夫曼将"写剧本"本身作为故事主体，元叙事与现实叙事的交织折叠，对写作者来说是一次叙事观的彻底解构。',
        },
        11: {  # 律师
            'label': '法律/律师',
            'thought': '用户具有逻辑辩证和证据推理的职业训练，偏好有叙事反转、道德博弈和庭审张力的作品。',
            'req': '推荐逻辑推理和道德博弈的影片',
            'movie': '《十二怒汉》',
            'reason': '一个陪审团室里，12人的辩证与说服构成了最精彩的庭审剧场，证据链的逐一拆解对您来说是职业级享受。',
        },
        15: {  # 销售/市场
            'label': '商业/市场',
            'thought': '用户有竞争博弈与策略思维，对权力结构、市场博弈和人性议题有天然敏锐度，宜推荐决策张力强的作品。',
            'req': '推荐关于商战和策略的影片',
            'movie': '《社交网络》',
            'reason': '芬奇用硅谷的背刺和合同拆解展示了商业博弈的人性底层，创业执念与背叛代价的双重叙事对您的商业直觉是一次精准共鸣。',
        },
        3: {  # 行政/管理
            'label': '管理/行政',
            'thought': '用户具有全局视野，偏好有宏大格局、权谋博弈和领导力主题的作品。',
            'req': '推荐有格局感的历史或政治影片',
            'movie': '《至暗时刻》',
            'reason': '丘吉尔在危机中的决策过程揭示了领导力的真正内核——极限压力下的信念坚守，对管理者有极强的启示性。',
        },
        12: {  # 大学生
            'label': '大学生',
            'thought': '用户处于世界观构建期，对多元文化和人生选择主题有强烈共鸣，宜推荐具有成长叙事或思辨价值的作品。',
            'req': '推荐关于成长和选择的影片',
            'movie': '《心灵捕手》',
            'reason': '天才在制度框架与内心自由之间的撕裂，恰好照见每个大学生面对未来的困惑，那句"这不是你的错"会在某个夜里击中你。',
        },
        10: {  # 中小学生
            'label': '学生',
            'thought': '用户年龄较小，宜推荐正能量、有冒险精神或成长主题的家庭友好影片，避免暴力和成人向内容。',
            'req': '推荐冒险和成长的影片',
            'movie': '《哈利·波特与魔法石》',
            'reason': '霍格沃茨的世界里，每一个普通孩子都可能发现自己的不凡，冒险与友情的核心叙事跨越所有年龄层。',
        },
        13: {  # 军人
            'label': '军人',
            'thought': '用户有纪律意识和牺牲精神，偏好荣誉、战争、集体主义或极限生存类主题的作品。',
            'req': '推荐战争或极限意志的影片',
            'movie': '《血战钢锯岭》',
            'reason': '戴斯蒙德在不携带武器的情况下救下75人，信念与意志的极限张力，对有军旅经历的您来说有最深层的精神共振。',
        },
    }

    # ── 1. 获取职业代码 ──────────────────────────────────────────────────
    occ_code = getattr(user, 'occupation', None)
    expert_prior = OCCUPATION_EXPERT_PRIOR.get(occ_code) if occ_code is not None else None

    # ── 2. 用户画像事实区块 ──────────────────────────────────────────────
    profile_block = f"""
【用户认知画像】（客观事实，严格参考，不得无中生有）：
{interaction_summary.strip()}
⚠️ 画像约束：推荐理由必须能解释「为何此片符合该用户的核心偏好类型和近期喜好」，禁止输出与用户偏好完全无关的通用理由。
"""

    # ── 3. 职业专家先验 + Few-shot 示例 ────────────────────────────────
    if not expert_prior:
        return profile_block  # 无职业匹配时只注入事实画像

    fewshot_block = f"""
【职业认知先验 (Expert Prior)】：
该用户职业背景为「{expert_prior['label']}」，推荐时请激活以下认知对齐策略：
→ 隐式思维模式：{expert_prior['thought']}

【Few-shot 对齐示例】（仿照此格式，输出具有职业穿透力的推荐理由）：
    用户请求："{expert_prior['req']}"
    职业隐式逻辑：{expert_prior['thought']}
    对齐输出：{expert_prior['movie']}：{expert_prior['reason']}

⚡ 核心对齐要求：在本次推荐理由中，必须将电影内容与用户的「{expert_prior['label']}」职业认知框架显式关联，
使每条理由都具备"只有这类用户才能感受到的共鸣点"，而非泛泛而谈的通用推荐语。
严禁出现"适合所有人"、"故事精彩"等无效泛化描述。
"""

    return profile_block + fewshot_block


def _build_movie_recommendation_prompt(user, search_query, is_thinking_mode, is_follow_up,
                                        interaction_summary=None):
    """构建核心推荐分支的 Prompt（KAG V201 — 向量语义 + 图谱拓扑 + 结构化条件过滤）"""
    # 1. 语义脱水
    stop_words = ['再', '还', '继续', '换', '类似', '别的', '其他', '再来', '推荐', '几部', '关于', '电影',
                    '类似的', '的说']
    clean_topic = search_query
    for word in stop_words:
        clean_topic = clean_topic.replace(word, "")
    clean_topic = clean_topic.strip() or "电影"

    # 🔥 1.5. 提取结构化查询条件（年份范围、类型、地区）
    constraints = extract_query_constraints(search_query)
    logger.info(f"[约束提取] 输入: '{search_query}' → 约束: {constraints}")

    # 2. 强化去重
    ignore_titles = get_recommended_movies_from_history(user, limit=10)

    # 3. 获取用户历史高分电影 mid，供 KG 关联路径查询使用
    user_history_mids = []
    try:
        user_history_mids = list(
            UserRating.objects.filter(user=user, score__gte=7.5)
            .values_list('movie__id', flat=True)[:20]
        )
    except Exception:
        pass

    # 4. 双轨检索：向量语义（What） + 知识图谱拓扑（Why）
    t_rag_start = time.time()
    vector_context = query_vector_rag(search_query, k=15) or ""
    kg_context = get_kg_subgraph(clean_topic, user_history_mids=user_history_mids)
    t_rag_duration = time.time() - t_rag_start

    # 🔥 4.2 结构化条件过滤：用数据库精确筛选，将结果注入到向量语义资料之前
    db_filtered_movies = []
    constraint_filter_desc = ""
    if constraints['year_from'] or constraints['year_to'] or constraints['genre'] or constraints['region']:
        try:
            qs = Movie.objects.all()
            filter_parts = []

            if constraints['genre']:
                qs = qs.filter(genres__name__icontains=constraints['genre'])
                filter_parts.append(f"类型:{constraints['genre']}")

            if constraints['region']:
                qs = qs.filter(
                    Q(regions__name__icontains=constraints['region']) |
                    Q(title__icontains=constraints['region'])
                )
                filter_parts.append(f"地区:{constraints['region']}")

            if constraints['year_from']:
                qs = qs.filter(date__year__gte=constraints['year_from'])
                filter_parts.append(f"≥{constraints['year_from']}年")

            if constraints['year_to']:
                qs = qs.filter(date__year__lte=constraints['year_to'])
                filter_parts.append(f"≤{constraints['year_to']}年")

            db_filtered_movies = list(
                qs.order_by('-score', '-vote_count')
                .prefetch_related('genres', 'directors', 'actors')
                .distinct()[:10]
            )
            constraint_filter_desc = " + ".join(filter_parts) if filter_parts else ""
            logger.info(f"[条件筛选] 命中 {len(db_filtered_movies)} 部: {constraint_filter_desc}")
        except Exception as e:
            logger.warning(f"[条件筛选] 数据库查询异常: {e}")

    # 4.5 用户认知画像 + 职业专家先验 (Expert Prior Few-shot)
    user_prior_section = _build_user_prior_context(user, interaction_summary)
    
    # ★ 未成年人内容保护：注入安全约束到 LLM Prompt
    content_safety_prompt = ""
    try:
        from myapp.utils.content_safety import get_content_safety_prompt
        content_safety_prompt = get_content_safety_prompt(user)
    except Exception:
        pass

    # 5. 动态视觉锚点
    visual_anchors = "机械、芯片、实验室、冷色调、电子元件、金属" if "AI" in clean_topic.upper() or "人工" in clean_topic else \
        "宇航服、星空、飞船、虚空、精密仪器" if "太空" in clean_topic else "相关风格元素"

    # 🔥 4.3 将条件筛选结果序列化为 LLM 可消费的文本
    constraint_section = ""
    if db_filtered_movies:
        constraint_lines = []
        for m in db_filtered_movies:
            genres_str = "、".join(g.name for g in m.genres.all()[:3])
            directors_str = "、".join(d.name for d in m.directors.all()[:2])
            summary_short = (m.summary or "")[:50].replace("\n", " ")
            score_str = str(m.score) if m.score else "暂无"
            date_str = str(m.date) if m.date else "未知"
            constraint_lines.append(
                f"《{m.title}》(ID:{m.id}) | 评分:{score_str} | 上映:{date_str} | "
                f"类型:{genres_str} | 导演:{directors_str} | {summary_short}..."
            )
        constraint_section = f"""
【条件筛选结果】（系统根据用户"{constraint_filter_desc}"条件从数据库精确匹配）：
{chr(10).join(constraint_lines)}

⚡ 优先从以上筛选结果中推荐。如果结果不足3部，可结合【向量语义资料】补充。
"""
        logger.info(f"[条件注入] 已注入 {len(constraint_lines)} 部条件筛选电影到 Prompt")

    # 6. 构建知识图谱上下文区块（有则注入，无则静默跳过）
    kg_section = ""
    if kg_context:
        kg_section = f"""
【知识图谱推理路径】（格式：实体--[关系]-->实体，来自 Neo4j 拓扑遍历）：
{kg_context}

⚡ 图谱节点推理权重（优先级从高到低，必须遵守）：
① 导演关联 Director-Link【最高权重 ★★★★★】
    导演是创作风格的源头与视觉语言的锚点。若路径存在"同导演"连边，
    该导演执导的未推荐影片必须列为首选候选，并在理由中明确点出导演名。
② 历史对齐 History-Base【高权重 ★★★★】
    用户历史高分影片（ID节点）是所有推理路径的个性化起点。
    必须优先从用户已交互节点出发寻找相似路径，将因果关系落到用户真实审美上。
③ 类型桥接 Genre-Bridge【中权重 ★★★】
    用于辅助解释跨题材的兴趣迁移。当导演路径不存在时，以类型共鸣作为次选依据。
④ 演员关联 Actor-Link【低权重 ★★】
    演员节点流量大但噪声多，仅当为绝对核心主演时才引用，一般演员不值得单独成为理由。
⑤ 地区关联 Region-Link【极低权重 ★】
    仅作最后的辅助筛选手段，不应单独作为推荐理由。

🔒 表达约束：
- 推荐理由必须自然融入上述最高优先级的图谱连线信息。
- 示例格式："该片由您喜爱的【XX导演】执导，延续了其在【XX类型】领域的深邃创作风格。"
- 严禁使用"根据图谱"、"知识图谱显示"、"系统检测到"等内部词语，要自然叙述。
"""

    # 7. 构建完整 Prompt
    kag_instruction = f"""
💡 【KAG 推理引擎 V201 — 极端事实约束版】：
- **核心主题**：用户正在寻找关于"{clean_topic}"的电影。请忽略搜索词中的"再来、类似"等交互指令词。
- **排除红线**：绝对禁止推荐列表中的电影：[{'、'.join(ignore_titles) or '无'}]。
- **视觉共鸣**：若简介不匹配，必须查看 [海报视觉描述]。只要包含（{visual_anchors}），即视为符合"{clean_topic}"主题，并标注"【视觉风格关联】"。
- **图谱优先**：若【知识图谱推理路径】中存在合适候选，优先推荐并在理由中明确引用图谱连边关系。
- **关键约束**：推荐的电影信息（导演、类型、演员、简介）必须全部来源于提供的资料。任何凭空编造都会被判定为系统故障！
"""

    fact_lock = f"""
🛑 【致命红线：事实一致性 — 零容忍策略】：
1. 你的职责是"解释器"而非"创造者"。**严禁引用、提及任何不在【向量语义资料】或【知识图谱推理路径】中出现的电影、导演或演员！**
2. 如果你要推荐一部电影，该电影的导演信息必须来源于提供的资料。资料里没有的，绝对不许你自己补充！
3. 如果资料不足以支持3部推荐，宁可只推荐2部，也绝不凭空捏造。
4. 理由必须直接追溯到【向量语义资料】或【知识图谱推理路径】。任何不能溯源的信息都视为幻觉，绝对禁止！
5. 严禁以下行为：
   ❌ 自己补充演员、导演、年份信息
   ❌ 编造电影简介或剧情细节
   ❌ 声称"根据资料推断"或"合理推测"，然后编造内容
   ❌ 输出"资料未查到"等内部排除逻辑词
"""

    # 🔥 Few-shot 通用推荐示例（不依赖职业，适用于所有用户）
    general_fewshot = """
【通用推荐示例（严格仿照此风格输出）】：

▸ 示例1（同导演关联）：
  《信条》(ID:789)：与您喜爱的《星际穿越》同为诺兰执导，延续了其独特的非线性叙事与冷色调视觉美学，时间维度的探索再次令人叹为观止。

▸ 示例2（同类型关联）：
  《银翼杀手2049》(ID:321)：本片与您心目中的经典《黑客帝国》同属赛博朋克题材，探讨了人工智能与人类意识的边界，视觉美学登峰造极。

▸ 示例3（情感基调共鸣）：
  《绿皮书》(ID:654)：基于您对《肖申克的救赎》的高分评价，本片在温暖治愈的情感基调上与您的品味高度契合，幽默与感动并存。

▸ 示例4（演员关联）：
  《盗梦空间》(ID:456)：莱昂纳多在《荒野猎人》中展现了极高造诣，这部科幻巨制中他的演技同样令人惊艳，角色层次极为丰富。
"""

    final_prompt = f"""你是一位专业的影库专家，能够融合语义检索与知识图谱拓扑进行深度推荐。

⚠️ 【严格事实验证】：在输出每一条推荐之前，你必须完成以下验证步骤：
   1. 确认电影名称是否出现在【向量语义资料】或【知识图谱推理路径】中
   2. 确认导演信息是否完全来自提供的资料，不可自行补充
   3. 确认推荐理由能否直接溯源到资料中的某个明确信息点
   如果任何信息无法通过上述验证，立即放弃推荐该电影！

{user_prior_section}
{constraint_section}
{kg_section}
【向量语义资料】（来自影库文本检索）：
{vector_context[:1200]}

{kag_instruction}
{fact_lock}
{content_safety_prompt}

📋 任务：根据【核心主题】推荐最多 3 部不在排除名单内的电影，必须带 ID。
   - 如果资料不足以安全推荐 3 部，请只推荐 1-2 部
   - 每部推荐都必须在【资料验证】中通过

📤 输出格式：
1. 《电影名》(ID:xxx)：推荐理由
   （若有画像支撑，须体现用户职业/偏好共鸣；若有图谱路径支撑，须引用关系链）
2. 《电影名》(ID:xxx)：推荐理由
... （最多 3 部）

⛔ 警告：如果你输出了任何无法溯源到提供资料中的信息，这将被判定为系统失败！
"""

    # 8. 动态温度 + 严格区分 Thinking 模式
    if is_thinking_mode:
        final_prompt += """
【重要】请先使用 <think> 和 </think> 标签包裹你的分析推理过程（重点：分析图谱路径是否与用户意图匹配，然后结合语义资料做最终决策）。思考结束后，在标签外部输出最终的推荐结果。
"""
        temperature = 0.6
    else:
        # 🔥 非 Thinking 模式：明确禁止模型输出任何推理过程或 <think> 标签，防止跑死
        final_prompt += "\n请直接输出推荐结果，不要输出任何推理过程或 <think> 标签，也不要解释你是如何得出结论的。\n"
        temperature = 0.5 if is_follow_up else 0.2

    return None, final_prompt, temperature


def _build_chat_prompt(user_input):
    """构建闲聊分支的 Prompt"""
    final_prompt = f"你是智能观影助手。请用一句话幽默地回应用户：'{user_input}'，引导他询问影片推荐。"
    return None, final_prompt, 0.6


def _extract_think_content(response_text):
    """
    提取思考链内容并打印到控制台
    用于调试和监控AI的推理过程
    """
    # 🔥 修复：明确匹配 <think> 标签内部的内容
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, response_text, re.DOTALL)

    if think_match:
        think_content = think_match.group(1).strip()
        logger.info(f"[AI 深度思考过程]: {think_content[:100]}...")
        # 移除思考链标签及内容，只返回正式回复
        return re.sub(think_pattern, '', response_text, flags=re.DOTALL).strip()
    else:
        return response_text.strip()

# =================================================
# 2. 推荐解释接口 (ajax_explain_rec)
# =================================================



# --- 领域停用词 (保持 V26 的清洗力度) ---

# 🔥 V34 黑名单：加入"导演"、"执导"等词，防止 jieba 提取它们作为关键词
STOP_WORDS = {
    '电影', '影片', '故事', '导演', '编剧', '主演', '饰演', '角色', '演员', '执导', '监制',
    '上映', '作品', '片子', '观众', '影史', '经典', '神作', '佳作', '高分',
    '制作', '拍摄', '镜头', '画面', '视觉', '特效', '音效', '配乐', '剧本',
    '表现', '呈现', '还原', '打造', '构建', '设定', '背景', '评价', '评分',
    '讲述', '关于', '一个', '开始', '最终', '发生', '出现', '面对', '充满',
    '成功', '著名', '精彩', '深刻', '复杂', '独特', '完美', '不仅', '而且',
    '这部', '系列', '风格', '感觉', '觉得', '喜欢', '好看', '就是',
    '找到', '寻找', '发现', '试图', '成为', '作为', '利用', '以及', '或者',
    '妻子', '丈夫', '朋友', '秘密', '父亲', '母亲', '孩子', '男人', '女人',
    '时间', '地点', '事件', '虽然', '但是', '之中', '之后', '一切', '所有',
    '问题', '关系', '生活', '世界', '人生', '命运', '选择', '决定', '能够',
    '需要', '经历', '展开', '描绘', '展现', '聚焦', '探讨', '揭示',
    '核心', '情节', '主题', '元素', '精神', '价值', '意义', '旅程', '过程',
    '方式', '手法', '态度', '环境', '社会', '时代', '历史', '文化'
}

def extract_core_keywords(text, top_k=5):
    """
    V38: 仅提取普通名词 (n) 和其他专名 (nz, ns)，
    🔥 坚决剔除人名 (nr)，防止"安迪"、"杰克"这种角色名成为关联点。
    """
    if not text: return set()
    try:
        # 🔥 allowPOS 修改：移除 'nr' (人名)
        keywords = jieba.analyse.textrank(
            text, topK=20, withWeight=False,
            allowPOS=('n', 'ns', 'nz') # 只留: 名词, 地名, 其他专名
        )
        clean = set()
        for kw in keywords:
            if len(kw) >= 2 and kw not in STOP_WORDS:
                clean.add(kw)
        return clean
    except:
        return set()


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def calculate_director_similarity(list1, list2):
    """计算导演的 Jaccard 相似度 (交并比)"""
    set1, set2 = set(list1), set(list2)
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def calculate_genre_similarity(list1, list2):
    """计算电影类型的 Jaccard 相似度 (交并比)"""
    set1, set2 = set(list1), set(list2)
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


# 建议在外部定义专门给“解释引擎”用的锁，避免与聊天引擎互相死锁
EXPLAIN_LOCK = threading.Lock()


@login_required
def ajax_explain_rec(request):
    """
    [推荐解释 V200 - 极速 KAG 终极版]
    解决：冷启动卡死、图谱失效、Token 乱跑问题
    
    【论文可写点】：推荐解释缓存机制 — 相同用户对同一部电影的解释结果在 TTL 期间
                  直接从缓存返回，避免重复 LLM 推理，缓存命中时延迟 <10ms。
    """
    movie_id = request.GET.get('movie_id')
    if not movie_id:
        return JsonResponse({'status': 'error', 'content': '未找到目标电影'})

    # 🔥 [工程优化] 推荐解释缓存：避免同一用户对同一电影重复调用 LLM
    # 【性能收益】：缓存命中时响应时间从 2-5s 降低到 <10ms
    force_refresh = request.GET.get('refresh') == 'true'
    if not force_refresh:
        cached_result = get_cached_explain(request.user.id, int(movie_id))
        if cached_result:
            logger.info(f"[ExplainCache] 缓存命中 user={request.user.id} movie={movie_id}")
            return JsonResponse(cached_result)

    # 🔥 导演美学签名库（数据库驱动版：种子库 + 自动补全）
    from myapp.utils.director_styles import get_director_styles
    DIRECTOR_STYLES = get_director_styles()

    # 【论文可写点】：使用 AgentTracer 记录整个推荐解释的推理链路
    start_time = time.time()
    tracer = AgentTracer(user_id=request.user.id, action="explain_rec", tool="kag_explain")
    tracer.start_time = time.time()  # 手动启动计时（避免 with 块内逻辑过长）
    if EXPLAIN_RESOURCES["model"] is None:
        load_explain_resources()

    try:
        # 1. 深度准备：目标电影画像
        target_movie = Movie.objects.prefetch_related('genres', 'directors').get(id=movie_id)
        target_genres = [g.name for g in target_movie.genres.all()]
        target_directors = [d.name for d in target_movie.directors.all()]
        target_summary = target_movie.summary[:150] if target_movie.summary else "暂无简介"

        # 2. 获取用户历史（优先 8 分以上，无则取最近评分，再无则为空）
        history_qs = UserRating.objects.filter(user=request.user).select_related('movie').prefetch_related(
            'movie__genres', 'movie__directors').order_by('-score', '-comment_time')[:30]
        history_list = list(history_qs)

        reason_type = "系统推荐"
        director_strength = 0.0
        t_llm_duration = 0.0
        processed_content = ""

        # ==========================================
        # 分支 A：绝对冷启动（无任何观影历史） -> 极速放行
        # ==========================================
        if len(history_list) == 0:
            prompt = f"你是一位电影推荐官。向新用户推荐电影《{target_movie.title}》。简介：{target_summary}。请用50字简单介绍影片核心亮点，直接输出推荐语，不要任何前缀。"
            reason_type = "冷启动热推"

            t_llm_start = time.time()
            try:
                with EXPLAIN_LOCK:
                    llm = ChatOllama(model="qwen3:4b-instruct", temperature=0.3, num_ctx=1024, num_predict=150)
                    response = llm.invoke(prompt)
                    processed_content = response.content.strip()
                    # 提取 <think> 标签（qwen3 模型可能输出思考过程）
                    processed_content = re.sub(r'<think>.*?</think>', '', processed_content, flags=re.DOTALL).strip()
                    t_llm_duration = time.time() - t_llm_start
                    logger.debug(f"冷启动 LLM 调用成功，耗时: {t_llm_duration:.2f}s，输出长度: {len(processed_content)}")
            except Exception as llm_error:
                t_llm_duration = time.time() - t_llm_start
                logger.error(f"冷启动 LLM 调用失败: {llm_error}")
                traceback.print_exc()
                # LLM 失败时，返回兜底文本
                processed_content = f"推荐您观看《{target_movie.title}》这部精彩影片。"

        # ==========================================
        # 分支 B：有观影历史 -> 触发 KAG 图谱关联
        # ==========================================
        else:
            best_anchor = None
            max_reason_score = -1.0
            best_h_dirs = []
            best_h_genres = []

            # 寻找最优锚点（导演绝对优先）
            for h in history_list:
                h_movie = h.movie
                if h_movie.id == target_movie.id: continue

                v_score = calculate_visual_sim(target_movie.poster_embedding_json, h_movie.poster_embedding_json)
                h_dirs = [d.name for d in h_movie.directors.all()]
                d_score = calculate_director_similarity(target_directors, h_dirs)
                h_genres = [g.name for g in h_movie.genres.all()]
                g_score = calculate_genre_similarity(target_genres, h_genres)

                # 🔥 导演霸权逻辑：有相同导演直接 +5.0 基础分
                if d_score > 0:
                    combined_score = 5.0 + v_score * 0.1 + g_score * 0.2
                else:
                    combined_score = v_score * 0.2 + g_score * 0.3

                if combined_score > max_reason_score:
                    max_reason_score = combined_score
                    best_anchor = h_movie
                    best_h_dirs = h_dirs
                    best_h_genres = h_genres

            logic_chain = ""

            # 🔥 如果没找到锚点，从历史中随机选一部
            if not best_anchor and history_list:
                best_anchor = history_list[0].movie
                best_h_dirs = [d.name for d in best_anchor.directors.all()]
                best_h_genres = [g.name for g in best_anchor.genres.all()]
                logger.debug(f"使用兜底锚点: {best_anchor.title}")

            # Neo4j 图谱深度查询（增强版：多路径发现 + 子图推理）
            kg_paths = []  # 用于前端可视化的结构化图谱路径
            kg_sub_context = ""  # 子图推理上下文

            if best_anchor and neo_graph is not None:
                try:
                    src_mid, dst_mid = best_anchor.id, target_movie.id

                    # ── 路径1: 直接关系查询（导演/演员/类型）──
                    dir_res = neo_graph.run(
                        "MATCH (src:Movie {mid: $src})<-[:DIRECTED_BY]-(d:Person)-[:DIRECTED_BY]->(dst:Movie {mid: $dst}) RETURN d.name AS name LIMIT 1",
                        src=src_mid, dst=dst_mid).data()
                    act_res = neo_graph.run(
                        "MATCH (src:Movie {mid: $src})<-[:ACTED_IN]-(a:Person)-[:ACTED_IN]->(dst:Movie {mid: $dst}) RETURN a.name AS name LIMIT 1",
                        src=src_mid, dst=dst_mid).data()
                    gen_res = neo_graph.run(
                        "MATCH (src:Movie {mid: $src})-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(dst:Movie {mid: $dst}) RETURN g.name AS name LIMIT 1",
                        src=src_mid, dst=dst_mid).data()

                    if dir_res:
                        d_name = dir_res[0]['name']
                        reason_type = f"同源导演【{d_name}】"
                        director_strength = 0.95
                        kg_paths.append({"from": best_anchor.title, "to": target_movie.title, "relation": "同导演", "node": d_name})

                        # 美学签名注入
                        aesthetic_signature = DIRECTOR_STYLES.get(d_name, "")
                        if aesthetic_signature:
                            logic_chain = f"《{best_anchor.title}》--[同导演:{d_name}]-->《{target_movie.title}》。导演{d_name}的艺术风格：{aesthetic_signature}。请在推荐语中融合这些美学特征，具象化两部影片的视听风格。"
                        else:
                            logic_chain = f"《{best_anchor.title}》--[同导演:{d_name}]-->《{target_movie.title}》。请重点强调两部影片都出自导演{d_name}之手，在创作风格和视听语言上一脉相承。"

                    elif act_res:
                        a_name = act_res[0]['name']
                        reason_type = f"同主演【{a_name}】"
                        director_strength = 0.65
                        kg_paths.append({"from": best_anchor.title, "to": target_movie.title, "relation": "同主演", "node": a_name})
                        logic_chain = f"《{best_anchor.title}》--[同主演:{a_name}]-->《{target_movie.title}》。请指出演员{a_name}在这两部影片中都贡献了出色的表演。"

                    elif gen_res:
                        g_name = gen_res[0]['name']
                        reason_type = f"题材共鸣【{g_name}】"
                        director_strength = 0.40
                        kg_paths.append({"from": best_anchor.title, "to": target_movie.title, "relation": "同类型", "node": g_name})
                        logic_chain = f"《{best_anchor.title}》--[同类型:{g_name}]-->《{target_movie.title}》。指出这两部影片在类型元素和剧情氛围上相似。"
                    else:
                        reason_type = "综合美学相似"
                        director_strength = 0.30
                        logic_chain = f"《{best_anchor.title}》和《{target_movie.title}》在视觉氛围和情感表达上有较强的共鸣。"

                    # ── 路径2: 子图推理 — 导演其他高分作品（扩展图谱深度）──
                    if dir_res:
                        d_name = dir_res[0]['name']
                        dir_filmography = neo_graph.run("""
                            MATCH (d:Person {name: $name})-[:DIRECTED_BY]->(m:Movie)
                            WHERE m.mid <> $src AND m.mid <> $dst
                            RETURN m.mid AS mid, m.title AS title, m.score AS score
                            ORDER BY m.score DESC LIMIT 3
                        """, name=d_name, src=src_mid, dst=dst_mid).data()
                        if dir_filmography:
                            titles = [f"《{r['title']}》(ID:{r['mid']})" for r in dir_filmography if r.get('title')]
                            if titles:
                                kg_sub_context += f"\n导演{d_name}的其他代表作：{'、'.join(titles)}。"
                                for r in dir_filmography:
                                    if r.get('title') and r.get('mid'):
                                        kg_paths.append({"from": d_name, "to": r['title'], "relation": "执导", "node": r['title']})

                    # ── 路径3: 子图推理 — 演员合作网络 ──
                    if act_res:
                        a_name = act_res[0]['name']
                        act_filmography = neo_graph.run("""
                            MATCH (a:Person {name: $name})-[:ACTED_IN]->(m:Movie)
                            WHERE m.mid <> $src AND m.mid <> $dst
                            RETURN m.mid AS mid, m.title AS title, m.score AS score
                            ORDER BY m.score DESC LIMIT 3
                        """, name=a_name, src=src_mid, dst=dst_mid).data()
                        if act_filmography:
                            titles = [f"《{r['title']}》(ID:{r['mid']})" for r in act_filmography if r.get('title')]
                            if titles:
                                kg_sub_context += f"\n{a_name}的其他代表作：{'、'.join(titles)}。"
                                for r in act_filmography:
                                    if r.get('title') and r.get('mid'):
                                        kg_paths.append({"from": a_name, "to": r['title'], "relation": "出演", "node": r['title']})

                    # ── 路径4: 子图推理 — 类型邻域高分电影 ──
                    if gen_res:
                        g_name = gen_res[0]['name']
                        genre_neighbors = neo_graph.run("""
                            MATCH (g:Genre {name: $name})<-[:BELONGS_TO]-(m:Movie)
                            WHERE m.mid <> $src AND m.mid <> $dst
                            RETURN m.mid AS mid, m.title AS title, m.score AS score
                            ORDER BY m.score DESC LIMIT 3
                        """, name=g_name, src=src_mid, dst=dst_mid).data()
                        if genre_neighbors:
                            titles = [f"《{r['title']}》(ID:{r['mid']})" for r in genre_neighbors if r.get('title')]
                            if titles:
                                kg_sub_context += f"\n{g_name}类型高分代表：{'、'.join(titles)}。"

                    # ── 路径5: 跨类型多路径发现（Sub-graph Reasoning）──
                    cross_type_res = neo_graph.run("""
                        MATCH (src:Movie {mid: $src})-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(mid:Movie)<-[:DIRECTED_BY]-(d:Person)-[:DIRECTED_BY]->(dst:Movie {mid: $dst})
                        WHERE mid.mid <> $src AND mid.mid <> $dst
                        RETURN d.name AS director, g.name AS genre, mid.title AS bridge_movie, mid.mid AS bridge_mid
                        LIMIT 2
                    """, src=src_mid, dst=dst_mid).data()
                    if cross_type_res:
                        for r in cross_type_res:
                            bridge = r.get('bridge_movie', '')
                            d_name = r.get('director', '')
                            g_name = r.get('genre', '')
                            bridge_mid = r.get('bridge_mid', '')
                            if bridge and d_name:
                                kg_paths.append({"from": best_anchor.title, "to": bridge, "relation": f"同类型({g_name})", "node": bridge})
                                kg_paths.append({"from": bridge, "to": target_movie.title, "relation": f"导演:{d_name}", "node": d_name})
                                if not kg_sub_context:
                                    kg_sub_context = f"通过{g_name}类型桥梁《{bridge}》(ID:{bridge_mid})和导演{d_name}建立关联。"

                except Exception as e:
                    logger.error(f"[KAG Error]: {e}")
                    traceback.print_exc()
                    reason_type = "ORM兜底关联"

            # 🔥 如果仍然没有锚点，降级到冷启动 Prompt
            if not best_anchor:
                logger.warning("无法找到合适的锚点，降级到冷启动模式")
                prompt = f"你是一位电影推荐官。向用户推荐电影《{target_movie.title}》。简介：{target_summary}。请用50字简单介绍影片核心亮点，直接输出推荐语，不要任何前缀。"
                reason_type = "降级冷启动"
            else:
                # KAG 极简推理 Prompt（增强版：融合子图上下文）
                sub_context_section = ""
                if kg_sub_context:
                    sub_context_section = f"\n【图谱关联线索】：{kg_sub_context}\n请在推荐语中自然融入这些关联信息，让用户感受到推荐的专业深度。"

                prompt = f"""你是专业电影评论家。用40-50字向用户推荐《{target_movie.title}》。
            【推荐理由来源】：{reason_type}
            【关联上下文】：{logic_chain}{sub_context_section}
            【电影信息】：评分{target_movie.score}，类型{', '.join(target_genres)}，导演{', '.join(target_directors)}。简介：{target_summary[:120]}。

            【写作要求】：
            1. 必须包含电影片名《{target_movie.title}》。
            2. 从电影本身的亮点出发（剧情、导演风格、演员表现、视觉美学等），不要泛泛而谈"品味契合"。
            3. 禁止使用"相似、图谱、关联、由于、高度契合"等机械词汇。
            4. 每部电影的推荐语要有独特角度，不能套用统一模板。

            直接输出推荐语，无前缀："""

            t_llm_start = time.time()
            try:
                with EXPLAIN_LOCK:
                    llm = ChatOllama(model="qwen3:4b-instruct", temperature=0.3, num_ctx=1024, num_predict=150)
                    response = llm.invoke(prompt)
                    processed_content = response.content.strip()
                    # 提取 <think> 标签（qwen3 模型可能输出思考过程）
                    processed_content = re.sub(r'<think>.*?</think>', '', processed_content, flags=re.DOTALL).strip()
                    t_llm_duration = time.time() - t_llm_start
                    logger.debug(f"LLM 调用成功，耗时: {t_llm_duration:.2f}s，输出长度: {len(processed_content)}")
            except Exception as llm_error:
                t_llm_duration = time.time() - t_llm_start
                logger.error(f"LLM 调用失败: {llm_error}")
                traceback.print_exc()
                # LLM 失败时，返回兜底文本
                processed_content = f"推荐您观看《{target_movie.title}》这部精彩影片。"

        # ==========================================
        # 3. 输出安全过滤 + 幻觉检测 + 结果后处理
        # ==========================================
        # 3a. 安全过滤：拦截不安全内容（成人影片标题等）
        processed_content = _sanitize_llm_output(processed_content)
        
        # 3a-bis. 安全过滤兜底：如果内容被截断或为空，从 KG 路径数据重建推荐理由
        # 知识图谱已经包含了关联关系，即使电影卡片加载失败，理由仍应展示
        if len(processed_content.strip()) < 20 and best_anchor:
            fallback_reason = ""
            if logic_chain:
                # 从逻辑链中提取推荐理由（去除内部标记）
                chain_text = logic_chain.split('。导演')[0] if '。导演' in logic_chain else logic_chain.split('。请')[0] if '。请' in logic_chain else logic_chain
                chain_text = chain_text.replace('--[', '→').replace(']-->', '→')
                fallback_reason = f"与您喜爱的《{best_anchor.title}》存在关联：{chain_text}。"
            if kg_sub_context:
                fallback_reason += f" {kg_sub_context.strip()}"
            if not fallback_reason:
                fallback_reason = f"《{target_movie.title}》与您之前喜爱的《{best_anchor.title}》在风格和题材上有较强共鸣，值得一看。"
            processed_content = fallback_reason
            logger.warning("[Safety Fallback] 安全过滤后内容过短，已从 KG 路径重建推荐理由")
        
        # 3b. 电影链接注入
        found_titles = set(re.findall(r'《(.*?)》', processed_content))
        for title in found_titles:
            linked_movie = Movie.objects.filter(title__icontains=title).first()
            if linked_movie:
                url = reverse('movie_detail', args=[linked_movie.id])
                link_html = f"<a href='{url}' class='text-primary fw-bold' style='text-decoration:none;'>《{title}》</a>"
                processed_content = processed_content.replace(f"《{title}》", link_html)
            else:
                processed_content = processed_content.replace(f"《{title}》", f"<b>《{title}》</b>")

        total_time = time.time() - start_time

        # 渲染 Footer
        r_str = "极强 ★★★★★" if director_strength > 0.8 else "强 ★★★★" if director_strength > 0.6 else "中 ★★★" if director_strength > 0.3 else "弱 ★★"
        footer = f"<hr><div class='d-flex justify-content-between align-items-center' style='font-size: 0.75rem; color: #999;'><span>🔍 归因: {reason_type} | 关联强度: {r_str}</span><span>⏱️ 耗时: {total_time:.2f}s (LLM:{t_llm_duration:.2f}s)</span></div>"

        # 构建增强版响应（含图谱路径数据，供前端可视化渲染）
        response_data = {
            'status': 'success',
            'content': processed_content + footer,
            'reason_type': reason_type,
            'director_strength': director_strength,
        }

        # 注入图谱路径数据（供前端 ECharts 图谱可视化使用）
        if kg_paths:
            response_data['kg_paths'] = kg_paths
            response_data['kg_path_count'] = len(kg_paths)

        # 注入子图推理上下文（供前端展示推理深度）
        if kg_sub_context:
            response_data['kg_sub_context'] = kg_sub_context

        # ★ XAI：归因雷达数据（供前端 ECharts 雷达图渲染）
        try:
            from myapp.utils.xai_explainer import build_attribution_radar
            radar_data = build_attribution_radar(request.user, int(movie_id))
            response_data['attribution_radar'] = radar_data
            response_data['confidence_score'] = radar_data.get('confidence_score', 0)
        except Exception as radar_err:
            logger.debug(f"[XAI] 归因雷达构建跳过: {radar_err}")

        # 🔥 [工程优化] 缓存解释结果，避免同一用户对同一电影重复调用 LLM
        # 【论文可写点】：推荐解释缓存策略，TTL=10分钟
        set_cached_explain(request.user.id, int(movie_id), response_data, ttl=600)

        # 【论文可写点】：AgentTracer 记录推理完成，输出结构化 trace
        trace_log_simple(
            user_id=request.user.id, action="explain_rec",
            latency_ms=round((time.time() - start_time) * 1000, 1),
            tool="kag_explain", extra=f"reason={reason_type} strength={director_strength}"
        )

        return JsonResponse(response_data)

    except Exception as e:
        traceback.print_exc()
        return JsonResponse({'status': 'error', 'content': '推荐官正在查阅胶片，请稍后。'})


def extract_concepts_jieba(movie_title, summary):
    """
    智能概念提取 V111 - 彻底移除 LLM，实现零显存消耗
    使用 TF-IDF 算法提取电影核心概念
    """
    import jieba.analyse

    # 1. 缓存检查（🔥 同样使用 md5 避免中文标题直接进入 cache key）
    _title_hash = hashlib.md5(movie_title.encode("utf-8")).hexdigest()
    cache_key = f"movie_concepts_jieba_{_title_hash}"
    cached_data = cache.get(cache_key)
    if cached_data: return cached_data

    # 2. 文本清洗与预处理
    # 将标题重复两次以增加权重，合并简介
    text = f"{movie_title} {movie_title} {summary}"

    try:
        # 3. 使用 TF-IDF 算法提取权重最高的 3 个关键词
        # topK=3: 提取3个词
        # allowPOS: 仅提取名词(n)、专有名词(nr, nz)、科技词(nt)
        concepts = jieba.analyse.extract_tags(
            text,
            topK=3,
            allowPOS=('n', 'nr', 'nz', 'nt', 'nw', 'v')  # 允许名词和动词
        )

        # 4. 兜底逻辑：如果提取失败，返回标题中的分词
        if not concepts:
            concepts = list(jieba.cut(movie_title))[:2]

        # 5. 写入缓存 (1天)
        cache.set(cache_key, concepts, 60 * 60 * 24)
        return concepts

    except Exception as e:
        logger.error(f"Jieba Extraction Error: {e}")
        return [movie_title[:4]]  # 极端情况返回标题前四个


def calculate_semantic_sim(text1, text2):
    """
    计算文本语义相似度 (导演/类型/剧情)
    利用预热好的 SentenceTransformer
    """
    model = EXPLAIN_RESOURCES["model"]
    if not model or not text1 or not text2:
        # 兜底：如果模型没加载完，用简单的 Jaccard 集合相似度
        s1, s2 = set(text1.split()), set(text2.split())
        union = len(s1 | s2)
        return len(s1 & s2) / union if union > 0 else 0.0

    try:
        # 编码 & 计算余弦相似度
        embeddings = model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    except:
        return 0.0

def calculate_visual_sim(vec1, vec2):
    """
    计算视觉向量相似度 (海报风格)
    基于 NumPy 纯数学计算，极速
    """
    if not vec1 or not vec2: return 0.0
    try:
        v1, v2 = np.array(vec1), np.array(vec2)
        if np.all(v1 == 0) or np.all(v2 == 0): return 0.0
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    except:
        return 0.0


# =============================================================
# 输出内容安全过滤器 (Output Content Safety Filter)
# =============================================================
# 拦截 LLM 可能生成的不安全内容（成人影片、暴力内容等）

# 不安全关键词黑名单（精确匹配，避免误杀正常推荐文本）
# ★ 修复：所有模式使用更严格的上下文匹配，防止"人性的爱情"等正常文本被误杀
_UNSAFE_PATTERNS = [
    # 日文成人影片标题模式（需要冒号分隔等明确上下文）
    r'^[^a-zA-Z\u4e00-\u9fff]*妻[：:]',      # 行首"妻:"，前面无正常文字
    r'[：:][^a-zA-Z\u4e00-\u9fff]*泣',        # 冒号后紧跟"泣"
    r'(?:色情|成人).*性[爱愛]',                 # "色情"+"性爱"组合才触发
    r'性[爱愛](?:影片|视频|内容|电影)',          # "性爱影片/视频/内容"才触发
    r'皮肤.*哭',
    r'(?:学生|年轻)\s*妻',                      # "学生妻"/"年轻妻"，带空格分隔
    r'人妻[：:]',                               # "人妻:"需要冒号
    r'^[^a-zA-Z\u4e00-\u9fff]*人妻$',          # 独占一行的"人妻"
    r'[处處]女[：:]',                           # "处女:"需要冒号
    r'黑[色肤膚].*哭',
    r'山手夫人',
    r'夫人[：:]',                               # "夫人:"需要冒号
    # 通用不安全词
    r'porn',
    r'xxx',
    r'adult\s*film',
    r'色情',
    r'成人影片',
]

_UNSAFE_RE = re.compile('|'.join(_UNSAFE_PATTERNS), re.IGNORECASE)


def _sanitize_llm_output(text, context_titles=None):
    """
    对 LLM 输出进行安全过滤 + 幻觉检测。
    
    Args:
        text: LLM 原始输出
        context_titles: 资料中出现的电影名集合（用于幻觉检测）
    
    Returns:
        str: 过滤后的安全文本
    """
    if not text:
        return text
    
    # 1. 内容安全过滤：删除包含不安全关键词的整个句子
    lines = text.split('\n')
    safe_lines = []
    for line in lines:
        if _UNSAFE_RE.search(line):
            logger.warning(f"[Safety] 拦截不安全内容: {line[:60]}...")
            continue
        safe_lines.append(line)
    text = '\n'.join(safe_lines)
    
    # 2. 幻觉检测：如果指定了上下文标题，删除引用了不存在电影的句子
    if context_titles:
        lines = text.split('\n')
        verified_lines = []
        for line in lines:
            mentioned = re.findall(r'《([^》]+)》', line)
            if mentioned:
                # 检查提到的电影是否在上下文中
                all_in_context = True
                for title in mentioned:
                    found = False
                    for ctx_title in context_titles:
                        if title.lower() in ctx_title.lower() or ctx_title.lower() in title.lower():
                            found = True
                            break
                    if not found:
                        all_in_context = False
                        break
                
                if not all_in_context and len(line) > 20:
                    # 该句子引用了不在资料中的电影 → 替换为安全兜底
                    logger.warning(f"[Hallucination] 拦截幻觉内容: {line[:60]}...")
                    continue
            
            verified_lines.append(line)
        text = '\n'.join(verified_lines)
    
    # 3. 最终检查：如果过滤后为空，返回兜底文本
    if not text.strip():
        return "推荐观看这部电影，值得一看。"
    
    return text





# 知识图谱图谱数据接口 (确保这个函数存在)
def ajax_kg_path(request):
    """
    知识图谱接口 (KAG 多模态关联版)
    利用 Neo4j 提取: 电影 -> [导演/演员/类型] -> 历史行为 的完整图谱路径
    """
    movie_id = request.GET.get('movie_id')
    if not movie_id: return JsonResponse({"nodes": [], "links": []})

    # 复用模块级 neo_graph 单例，避免每次请求重新建立连接
    graph = neo_graph
    if graph is None:
        logger.warning("Neo4j 未连接，无法查询图谱")
        return JsonResponse({"nodes": [], "links": []})

    try:
        target_mid = int(movie_id)
        user = request.user

        # 2. 预加载映射字典 (用于生成精准跳转链接)
        genre_map = {g.name: g.id for g in Genre.objects.all()}
        region_map = {r.name: r.id for r in Region.objects.all()}

        # 3. 获取用户历史高分电影 ID (用于"我看过的"关联)
        history_mids = []
        if user.is_authenticated:
            history_mids = list(UserRating.objects.filter(user=user, score__gte=7.5)
                                .order_by('?')
                                .values_list('movie_id', flat=True)[:15])

        # 4. Cypher 查询 — 🔥 优先级策略：导演 > 演员 > 类型 > 地区
        #    在 Cypher 层打上 type 标签并排序，确保导演最优先
        cypher = """
        MATCH (target:Movie {mid: $mid})

        // A. 找导演（取前 5 个，优先级最高）
        OPTIONAL MATCH (target)<-[:DIRECTED_BY]-(d:Person)
        WITH target, collect({node: d, type: 'director'}) AS directors

        // B. 找演员（取前 3 个，次优先级）
        OPTIONAL MATCH (target)<-[:ACTED_IN]-(a:Person)
        WITH target, directors, collect({node: a, type: 'actor'})[..3] AS actors

        // C. 找类型（取前 3 个）
        OPTIONAL MATCH (target)-[:BELONGS_TO]->(g:Genre)
        WITH target, directors, actors, collect({node: g, type: 'genre'})[..3] AS genres

        // D. 找地区（取前 2 个）
        OPTIONAL MATCH (target)-[:RELEASED_IN]->(r:Region)
        WITH target, directors, actors, genres, collect({node: r, type: 'region'})[..2] AS regions

        // E. 合并所有属性，🔥 导演优先列表（Directors first）
        WITH target,
            directors[..5] +  // 🔥 导演最多取 5 个，突显重要性
            actors +
            genres +
            regions AS attrs
        UNWIND attrs AS item

        // F. 解包为独立变量
        WITH target, item.node AS attr, item.type AS attr_type

        // G. 找历史关联（我看过的电影中，有没有和这个属性节点相连的）
        OPTIONAL MATCH (attr)--(h:Movie)
        WHERE h.mid IN $hist_mids AND h.mid <> $mid

        RETURN target, attr, attr_type, h AS history
        """

        result = graph.run(cypher, mid=target_mid, hist_mids=history_mids).data()

        # 5. 结果构建 — 🔥 优先处理导演关联
        nodes_dict = {}
        links_set = set()
        director_links = []  # 🔥 单独存储导演关联，最后优先添加
        other_links = []     # 其他关联
        director_names = set()  # 🔥 用于去重：记录所有导演名称

        def process_node(node_obj, category, is_center=False):
            if not node_obj: return None
            nid = str(node_obj.identity)

            # ── 权重尺寸矩阵（与 LLM KAG 推理优先级对齐）─────────────────────────
            # 5:导演(风格锚点,最重要)  0:当前电影(核心中心)  4:我看过的(个性化基准)
            # 2:类型(语义桥梁)      1:演员(流量节点)   3:地区(弱关联噪声)
            # 🔥 导演权重提升到最高：从60→65
            size_map = {0: 70, 5: 65, 4: 50, 2: 45, 1: 25, 3: 15}
            name = node_obj.get('name', 'Unknown')

            # --- URL 生成 ---
            url = None
            if category in [0, 4]:
                mid = node_obj.get('mid')
                if mid: url = f"/movie/{mid}/"
            elif category == 2:
                gid = genre_map.get(name)
                if gid: url = f"/depot/?genre={gid}"
            elif category == 3:
                rid = region_map.get(name)
                if rid: url = f"/depot/?region={rid}"

            if nid not in nodes_dict:
                nodes_dict[nid] = {
                    "id": nid,
                    "name": name,
                    "category": category,
                    "symbolSize": size_map.get(category, 25),
                    "draggable": True,
                    "url": url,
                    "is_center": is_center,  # 🔥 前端中心节点保护标志
                }
            return nid

        added_history_nids = set()
        MAX_HISTORY = 4

        for row in result:
            target  = row['target']
            attr    = row['attr']
            history = row['history']

            if not attr: continue

            # ── 优先使用 Cypher 标记的 attr_type（最可靠），标签解析作兜底 ──────
            attr_type = row.get('attr_type', '')
            acat      = 3
            link_name = "关联"
            attr_name = attr.get('name', 'Unknown')

            if attr_type == 'director':
                acat = 5; link_name = "导演"
                director_names.add(attr_name)  # 🔥 记录导演名称，用于去重
            elif attr_type == 'actor':
                acat = 1; link_name = "主演"
                # 🔥 如果该演员也是导演，则跳过（防止重复）
                if attr_name in director_names:
                    continue
            elif attr_type == 'genre':
                acat = 2; link_name = "类型"
            elif attr_type == 'region':
                acat = 3; link_name = "地区"
            else:
                # 兜底：按 Neo4j 节点 Label 严格优先级判断
                # Director 优先于 Actor，防止"导演兼演员"被降级
                node_labels = set(attr.labels)
                if 'Director' in node_labels:
                    acat = 5; link_name = "导演"
                    director_names.add(attr_name)
                elif 'Actor' in node_labels:
                    acat = 1; link_name = "主演"
                    # 🔥 如果该演员也是导演，则跳过
                    if attr_name in director_names:
                        continue
                elif 'Genre' in node_labels:
                    acat = 2; link_name = "类型"
                elif 'Region' in node_labels:
                    acat = 3; link_name = "地区"
                # 'Person' only（无子标签）→ 保持默认 acat=3，后续可扩展

            tid = process_node(target, 0, is_center=True)   # 中心电影始终 is_center=True
            aid = process_node(attr, acat)

            # 🔥 导演关联优先存储
            if acat == 5:  # 导演关联
                if acat in [1, 5]:
                    director_links.append((aid, tid, link_name))
                else:
                    director_links.append((tid, aid, link_name))
            else:
                if acat in [1, 5]:
                    other_links.append((aid, tid, link_name))
                else:
                    other_links.append((tid, aid, link_name))

            if history:
                hid_raw = str(history.identity)
                if hid_raw in added_history_nids or len(added_history_nids) < MAX_HISTORY:
                    hid = process_node(history, 4)
                    added_history_nids.add(hid)
                    other_links.append((hid, aid, "关联"))

        # ── 🔥 导演链接优先加入 links_set ──────────────────────────────────────
        for link in director_links:
            links_set.add(link)
        for link in other_links:
            links_set.add(link)

        # ── 中心节点保护：无论何种情况都强制保持最大尺寸 ────────────────────────
        for node in nodes_dict.values():
            if node.get('is_center'):
                node['symbolSize'] = 70

        # 6. 将 set 转为 ECharts list，🔥 导演链接首先排序并增强视觉
        links_list = []
        # 🔥 第一层：导演链接优先，加强视觉属性
        for src, tgt, val in links_set:
            if val == "导演":
                links_list.append({
                    "source": src, 
                    "target": tgt, 
                    "value": val,
                    "lineStyle": {"width": 3, "color": "#9b59b6", "opacity": 0.95},  # 粗线、紫色、高对比
                    "emphasis": {"lineStyle": {"width": 6}}  # hover时变更粗
                })
        # 第二层：其他链接
        for src, tgt, val in links_set:
            if val != "导演":
                links_list.append({"source": src, "target": tgt, "value": val})

        # 7. 图例与配色（category 索引与 size_map 键对应）
        categories = [
            {"name": "当前电影", "itemStyle": {"color": "#d9534f"}},   # 0: 红
            {"name": "演员",     "itemStyle": {"color": "#f0ad4e"}},   # 1: 橙
            {"name": "类型",     "itemStyle": {"color": "#5cb85c"}},   # 2: 绿
            {"name": "地区",     "itemStyle": {"color": "#5bc0de"}},   # 3: 蓝
            {"name": "我看过的", "itemStyle": {"color": "#999999"}},   # 4: 灰
            {"name": "导演",     "itemStyle": {"color": "#9b59b6"}},   # 5: 紫
        ]

        return JsonResponse({
            "nodes":      list(nodes_dict.values()),
            "links":      links_list,
            "categories": categories,
        })

    except Exception as e:
        logger.error(f"KG Error: {e}")
        import traceback
        traceback.print_exc()
        return JsonResponse({"nodes": [], "links": []})

# 🔥 专门为图谱优化的宽松版关联判断函数
def get_connection_point_for_kg(target_movie, h_movie):
    """
    图谱专用的宽松版关联判断
    🔥 现已支持：演员 | 导演 | 类型 三维关联
    """
    # 1. 演员重合 (最强)
    t_actors = set(target_movie.actors.values_list('name', flat=True))
    h_actors = set(h_movie.actors.values_list('name', flat=True))
    common_actors = t_actors & h_actors
    if common_actors:
        actor = list(common_actors)[0]
        return "actor", f"同主演: {actor}", actor, True

    # 2. 导演重合 (非常强) - 新增导演关联
    t_directors = list(target_movie.directors.values_list('name', flat=True))
    h_directors = list(h_movie.directors.values_list('name', flat=True))
    common_directors = set(t_directors) & set(h_directors)
    if common_directors:
        director = list(common_directors)[0]
        return "director", f"同导演: {director}", director, True
    
    # 3. 导演风格相似 (宽松阈值 > 0.3)
    # 即使不是同一导演，如果导演风格相似度高，也可以作为关联理由
    d_sim_score = calculate_director_similarity(t_directors, h_directors) if t_directors and h_directors else 0.0
    if d_sim_score > 0.3:
        return "director_style", f"导演风格相似", "导演风格相似", True

    # 4. 类型相似 (宽松阈值 > 0.45)
    # 因为图谱是为了展示"路径"，所以只要沾边就可以连，不必像推荐语那么严谨
    t_genres = list(target_movie.genres.values_list('name', flat=True))
    h_genres = list(h_movie.genres.values_list('name', flat=True))

    sim_score = calculate_genre_similarity(t_genres, h_genres)  # 复用已有的计算函数

    if sim_score > 0.45:
        # 找共同点
        common = set(t_genres) & set(h_genres)
        kw = list(common)[0] if common else "风格相似"
        return "genre", f"同类型: {kw}", kw, True

    return "none", "", "", False




# --- 辅助函数 (放在 views.py 内部或同级) ---
def _create_node_dict(raw_node, is_center=False):
    """根据 Label 确定类别和样式"""
    lbls = raw_node['labels']
    cat = 0  # Movie
    size = 40 if is_center else 30

    if 'Person' in lbls:
        cat = 1; size = 20
    elif 'Genre' in lbls:
        cat = 2; size = 20

    return {
        'id': raw_node['id'],
        'name': raw_node['name'],
        'category': cat,
        'symbolSize': size,
        'draggable': True
    }


def _format_nodes(raw_nodes_list):
    """格式化路径查询返回的节点列表"""
    nodes = []
    seen = set()
    for n in raw_nodes_list:
        if n['id'] not in seen:
            nodes.append(_create_node_dict(n, is_center=('Movie' in n['labels'])))
            seen.add(n['id'])
    return nodes



def search_results(request):
    # 1. 从 URL (GET请求) 中获取 'q' 参数的值
    query = request.GET.get('q')

    context = {
        'query': query,  # 把用户搜的词传回模板
        'movies': None  # 默认结果为空
    }

    if query:
        # 优化：先搜标题（走 db_index），结果足够则跳过 M2M JOIN
        title_qs = Movie.objects.filter(title__icontains=query)
        title_count = title_qs.count()

        if title_count >= 10:
            movies = title_qs.prefetch_related('genres', 'actors', 'directors')
        else:
            # 标题结果不足，补充 M2M 关联搜索
            m2m_pks = set(title_qs.values_list('pk', flat=True))
            m2m_pks.update(
                Movie.objects.filter(actors__name__icontains=query).values_list('pk', flat=True)[:20]
            )
            m2m_pks.update(
                Movie.objects.filter(genres__name__icontains=query).values_list('pk', flat=True)[:20]
            )
            m2m_pks.update(
                Movie.objects.filter(directors__name__icontains=query).values_list('pk', flat=True)[:20]
            )
            m2m_pks.update(
                Movie.objects.filter(regions__name__icontains=query).values_list('pk', flat=True)[:20]
            )
            movies = Movie.objects.filter(pk__in=m2m_pks).prefetch_related(
                'genres', 'actors', 'directors'
            )

        context['movies'] = movies
    else:
        # 3.1 如果用户没输入内容，可以选择显示所有电影，或者保持为空
        context['movies'] = Movie.objects.prefetch_related('genres', 'actors', 'directors').all()
        query = None

    # 4. 渲染一个新的模板
    return render(request, 'search_results.html', context)


@login_required
def center(request):
    """
    个人中心视图 (V4 - 双表单版)
    功能：展示收藏、评论、设置(颜色+资料)，并处理分页和表单提交
    """

    # --- 1. 智能 Tab 激活逻辑 ---
    active_tab = "collections"
    if request.GET.get('page_rate'):
        active_tab = "ratings"
    elif request.GET.get('tab') == 'settings':
        active_tab = "settings"
    elif request.GET.get('tab') == 'excluded':
        active_tab = "excluded"

    # --- 2. 处理表单提交 (POST) ---
    if request.method == 'POST':
        # 判断用户提交的是哪个表单
        if 'submit_color' in request.POST:
            # 🔵 处理颜色设置
            color_form = ColorPreferenceForm(request.POST, instance=request.user)
            profile_form = UserProfileForm(instance=request.user)  # 另一个表单保持原样

            if color_form.is_valid():
                color_form.save()
                messages.success(request, "🎨 个性化设置已保存！")
                return redirect(f"{request.path}?tab=settings")

        elif 'submit_profile' in request.POST:
            # 🟢 处理资料修改
            profile_form = UserProfileForm(request.POST, instance=request.user)
            color_form = ColorPreferenceForm(instance=request.user)  # 另一个表单保持原样

            if profile_form.is_valid():
                profile_form.save()
                messages.success(request, "📝 个人资料已更新！")
                # 更新 Session 中的用户信息 (防止页面显示旧数据)
                update_session_auth_hash(request, request.user)
                return redirect(f"{request.path}?tab=settings")
        else:
            # 未知提交，重置
            color_form = ColorPreferenceForm(instance=request.user)
            profile_form = UserProfileForm(instance=request.user)

    else:
        # GET 请求：初始化两个表单
        color_form = ColorPreferenceForm(instance=request.user)
        profile_form = UserProfileForm(instance=request.user)

    # --- 3. 处理 "我的收藏" 数据 (分页) ---
    collections_query = Collect.objects.filter(user=request.user).select_related('movie').order_by('-collect_time')
    collections_count = collections_query.count()
    coll_paginator = Pagination(request, collections_query, page_size=12, page_param="page_coll")

    # --- 4. 处理 "我的评论/评分" 数据 (分页) ---
    user_ratings_query = UserRating.objects.filter(user=request.user).select_related('movie').order_by('-comment_time')
    user_ratings_count = user_ratings_query.count()
    rate_paginator = Pagination(request, user_ratings_query, page_size=5, page_param="page_rate")

    # --- 5. 构造上下文 ---
    context = {
        'collections_list': coll_paginator.page_queryset,
        'collections_page_string': coll_paginator.html(),
        'collections_count': collections_count,

        'ratings_list': rate_paginator.page_queryset,
        'ratings_page_string': rate_paginator.html(),
        'ratings_count': user_ratings_count,

        'color_form': color_form,
        'profile_form': profile_form,  # 🔥 传入新表单

        'active_tab': active_tab,
    }

    return render(request, 'front_center.html', context)


# --- 1. 后台首页 ---
@admin_required  # <-- 使用我们刚创建的"安全"装饰器
def admin_index(request):
    # (这里可以添加一些统计数据)
    movie_count = Movie.objects.count()
    user_count = UserInfo.objects.count()
    rating_count = UserRating.objects.count()
    context = {
        'movie_count': movie_count,
        'user_count': user_count,
        'rating_count': rating_count,
    }
    return render(request, 'admin_index.html', context)


# --- 2. 电影管理 (CRUD) ---

@admin_required
def admin_movie_list(request):
    # (我们复用 "影片库" 的搜索逻辑)
    query = request.GET.get('q')
    if query:
        movies_queryset = Movie.objects.filter(title__icontains=query).order_by('-id')
    else:
        movies_queryset = Movie.objects.all().order_by('-id')

    # (复用分页)
    page_object = Pagination(request, movies_queryset, page_size=10)
    context = {
        'movies': page_object.page_queryset,
        'page_string': page_object.html(),
        'query': query,  # 传回搜索词
    }
    return render(request, 'admin_movie_list.html', context)


@admin_required
def admin_movie_add(request):
    """
    后台：添加电影 (文本输入版)
    """
    if request.method == 'POST':
        form = MovieModelForm(request.POST)
        if form.is_valid():
            # 1. 先保存电影基本信息 (commit=True 因为我们需要 ID 来建立 M2M)
            movie = form.save()

            # 2. 手动处理三个文本框的数据
            _process_m2m_text(movie, form.cleaned_data['actors'], Actor, 'actors')
            _process_m2m_text(movie, form.cleaned_data['regions'], Region, 'regions')
            _process_m2m_text(movie, form.cleaned_data['genres'], Genre, 'genres')
            _process_m2m_text(movie, form.cleaned_data['directors'], Actor, 'directors')

            messages.success(request, f"电影《{movie.title}》添加成功！")
            return redirect('admin_movie_list')
    else:
        form = MovieModelForm()

    return render(request, 'admin_panel/admin_movie_form.html', {
        'form': form,
        'title': '添加新电影'
    })


@admin_required
def admin_movie_edit(request, pk):
    """
    后台：编辑电影 (文本输入版)
    """
    movie_obj = get_object_or_404(Movie, pk=pk)

    if request.method == 'POST':
        form = MovieModelForm(request.POST, instance=movie_obj)
        if form.is_valid():
            # 1. 保存基本信息
            movie = form.save()

            # 2. 处理 M2M 文本数据
            _process_m2m_text(movie, form.cleaned_data['actors'], Actor, 'actors')
            _process_m2m_text(movie, form.cleaned_data['regions'], Region, 'regions')
            _process_m2m_text(movie, form.cleaned_data['genres'], Genre, 'genres')
            _process_m2m_text(movie, form.cleaned_data['directors'], Actor, 'directors')

            messages.success(request, f"电影《{movie.title}》更新成功！")
            return redirect('admin_movie_list')
    else:
        # --- 关键：GET 请求时，准备初始数据 (Initial Data) ---
        # 把 M2M 对象列表 变成 字符串: "中国, 美国"
        initial_data = {
            'actors': ", ".join([a.name for a in movie_obj.actors.all()]),
            'regions': ", ".join([r.name for r in movie_obj.regions.all()]),
            'genres': ", ".join([g.name for g in movie_obj.genres.all()]),
            # 🔥 新增：回显导演
            'directors': ", ".join([d.name for d in movie_obj.directors.all()]),
        }

        # 将 instance 和 initial 同时传给 Form
        form = MovieModelForm(instance=movie_obj, initial=initial_data)

    return render(request, 'admin_panel/admin_movie_form.html', {
        'form': form,
        'title': f'编辑: {movie_obj.title}'
    })


@admin_required
def admin_movie_delete(request, pk):
    movie_obj = get_object_or_404(Movie, pk=pk)

    if request.method == 'POST':
        # (POST 请求) 确认删除
        movie_obj.delete()
        messages.success(request, "电影删除成功。")
        return redirect('admin_movie_list')

    # (GET 请求) 显示确认页面
    return render(request, 'admin_confirm_delete.html',
                    {'target_name': movie_obj.title, 'cancel_url': 'admin_movie_list'})


# --- 3. 用户管理 (Read/Delete) ---

@admin_required
def admin_user_list(request):
    users_queryset = UserInfo.objects.all().order_by('id')
    page_object = Pagination(request, users_queryset, page_size=10)
    context = {
        'users': page_object.page_queryset,
        'page_string': page_object.html(),
    }
    return render(request, 'admin_user_list.html', context)





# --- ↓↓↓ 在这里添加 3 个新视图 ↓↓↓ ---

@admin_required
def admin_user_add(request):
    """
    C - 创建新用户 (管理员)
    """
    if request.method == 'POST':
        form = AdminUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()  # UserCreationForm 会自动哈希密码
            messages.success(request, f"用户 {user.username} 创建成功！")
            return redirect('admin_user_list')
    else:
        form = AdminUserCreationForm()

    return render(request, 'admin_user_form.html', {'form': form, 'title': '添加新用户'})


@admin_required
def admin_user_edit(request, pk):
    """
    U - 编辑用户信息 (管理员)
    """
    user_obj = get_object_or_404(UserInfo, pk=pk)

    if request.method == 'POST':
        form = AdminUserChangeForm(request.POST, instance=user_obj)
        if form.is_valid():
            form.save()
            messages.success(request, f"用户 {user_obj.username} 信息更新成功！")
            return redirect('admin_user_list')
    else:
        form = AdminUserChangeForm(instance=user_obj)

    return render(request, 'admin_user_form.html',
                    {'form': form, 'title': f'编辑用户: {user_obj.username}'})


@admin_required
def admin_user_reset_password(request, pk):
    """
    U - 重置用户密码 (管理员)
    """
    user_obj = get_object_or_404(UserInfo, pk=pk)

    if request.method == 'POST':
        # SetPasswordForm 需要 user 对象作为第一个参数
        form = AdminUserResetPasswordForm(user_obj, request.POST)
        if form.is_valid():
            form.save()  # SetPasswordForm 会自动哈希并保存
            messages.success(request, f"用户 {user_obj.username} 的密码重置成功！")
            return redirect('admin_user_list')
    else:
        form = AdminUserResetPasswordForm(user_obj)

    return render(request, 'admin_user_reset_password.html', {'form': form, 'target_user': user_obj})



@admin_required
def admin_user_delete(request, pk):
    user_obj = get_object_or_404(UserInfo, pk=pk)

    # (防止管理员删除自己)
    if user_obj == request.user:
        messages.error(request, "不能删除自己！")
        return redirect('admin_user_list')

    if request.method == 'POST':
        user_obj.delete()
        messages.success(request, f"用户 {user_obj.username} 删除成功。")
        return redirect('admin_user_list')

    return render(request, 'admin_panel/admin_confirm_delete.html', {
        'target_name': f"用户: {user_obj.username}",
        'cancel_url': 'admin_user_list'
    })


@login_required
@never_cache
def admin_comments(request):
    """
    后台：评论/评分管理列表
    """
    # 1. 安全检查：只有管理员能进
    if not request.user.is_staff:
        return redirect('front_index')

    # 2. 查询所有评论 (按时间倒序)
    # (使用 select_related 优化查询，防止 N+1 问题)
    queryset = UserRating.objects.all().select_related('user', 'movie').order_by('-comment_time')

    # 3. 搜索功能 (可选，支持搜内容或用户名)
    q = request.GET.get('q')
    if q:
        queryset = queryset.filter(
            Q(discussion__icontains=q) |
            Q(user__username__icontains=q) |
            Q(movie__title__icontains=q)
        )

    # 4. 分页 (使用你的 V3 分页器)
    paginator = Pagination(request, queryset, page_size=20, page_param="page")

    context = {
        'ratings': paginator.page_queryset,
        'page_string': paginator.html(),
        'total_count': queryset.count(),
    }

    return render(request, 'admin_comments.html', context)


@login_required
def admin_comment_delete(request, rating_id):
    """
    后台：删除评论
    """
    if not request.user.is_staff:
        return redirect('front_index')

    # 获取并删除
    UserRating.objects.filter(id=rating_id).delete()

    # 提示并跳回列表
    messages.success(request, "评论/评分已删除。")
    return redirect('admin_comments')


@admin_required
def admin_actor_list(request):
    """
    后台：演员列表
    """
    # 1. 获取搜索关键词
    query = request.GET.get('q')
    if query:
        # 按名字搜索
        actors_queryset = Actor.objects.filter(name__icontains=query).order_by('-id')
    else:
        actors_queryset = Actor.objects.all().order_by('-id')

    # 2. 分页 (每页 20 个)
    page_object = Pagination(request, actors_queryset, page_size=20)

    context = {
        'actors': page_object.page_queryset,
        'page_string': page_object.html(),
        'query': query,
    }
    return render(request, 'admin_panel/admin_actor_list.html', context)


@admin_required
def admin_director_list(request):
    """
    后台：导演列表（有导演作品的演员）
    """
    query = request.GET.get('q')
    if query:
        directors_qs = Actor.objects.filter(
            directed_movies__isnull=False, name__icontains=query
        ).distinct().order_by('-id')
    else:
        directors_qs = Actor.objects.filter(
            directed_movies__isnull=False
        ).distinct().order_by('-id')

    page_object = Pagination(request, directors_qs, page_size=20)

    context = {
        'directors': page_object.page_queryset,
        'page_string': page_object.html(),
        'query': query,
    }
    return render(request, 'admin_panel/admin_director_list.html', context)


@admin_required
def admin_actor_add(request):
    """
    后台：添加演员
    """
    if request.method == 'POST':
        form = ActorModelForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, "演员添加成功！")
            return redirect('admin_actor_list')
    else:
        form = ActorModelForm()

    return render(request, 'admin_panel/admin_actor_form.html', {
        'form': form,
        'title': '添加新演员'
    })


@admin_required
def admin_actor_edit(request, pk):
    """
    后台：编辑演员
    """
    actor_obj = get_object_or_404(Actor, pk=pk)

    if request.method == 'POST':
        form = ActorModelForm(request.POST, instance=actor_obj)
        if form.is_valid():
            form.save()
            messages.success(request, f"演员 {actor_obj.name} 更新成功！")
            return redirect('admin_actor_list')
    else:
        form = ActorModelForm(instance=actor_obj)

    return render(request, 'admin_panel/admin_actor_form.html', {
        'form': form,
        'title': f'编辑: {actor_obj.name}'
    })


@admin_required
def admin_actor_delete(request, pk):
    """
    后台：删除演员
    """
    actor_obj = get_object_or_404(Actor, pk=pk)

    if request.method == 'POST':
        actor_obj.delete()
        messages.success(request, "演员已删除。")
        return redirect('admin_actor_list')

    # 复用通用的删除确认页面
    return render(request, 'admin_panel/admin_confirm_delete.html', {
        'target_name': f"演员: {actor_obj.name}",
        'cancel_url': 'admin_actor_list'
    })


# ========================================================
#  管理员高级功能区
# ========================================================

def _run_training_thread():
    """后台线程运行训练，防止阻塞主进程"""
    logger.info("后台训练任务已启动...")
    try:
        # 调用我们之前写的 train_hybrid命令
        call_command('train_hybrid')
        logger.info("后台训练任务完成")
    except Exception as e:
        logger.error(f"训练失败: {e}")


@admin_required
def admin_trigger_train(request):
    """
    [接口] 触发模型重训练
    """
    if request.method == 'POST':
        # 开启一个新线程去跑训练，让 HTTP 请求立刻返回
        t = threading.Thread(target=_run_training_thread)
        t.setDaemon(True)
        t.start()

        return JsonResponse({
            'status': 'success',
            'msg': '🚀 训练任务已在后台启动！请关注控制台日志，约1-5分钟后生效。'
        })
    return JsonResponse({'status': 'error', 'msg': '不支持的请求方式'})


@admin_required
def admin_clear_cache(request):
    """
    [接口] 清除系统缓存 (Redis/LocalMem)
    """
    if request.method == 'POST':
        try:
            cache.clear()
            # 同时也清除模型缓存，强制下次推理时重新加载新权重
            global MODEL_CACHE
            MODEL_CACHE = {
                'model': None,
                'feature_columns': None,
                'encoders': {},
            }
            return JsonResponse({'status': 'success', 'msg': '🧹 系统缓存 & 模型权重缓存已清除！'})
        except Exception as e:
            return JsonResponse({'status': 'error', 'msg': str(e)})
    return JsonResponse({'status': 'error', 'msg': '不支持的请求方式'})


@admin_required
def admin_user_stats(request):
    """
    [接口] 获取用户画像统计数据 (用于前端画图)
    """
    # 1. 性别分布
    sex_dist = list(UserInfo.objects.values('sex').annotate(count=Count('sex')))
    # 转换 1->男, 2->女
    for item in sex_dist:
        item['name'] = dict(UserInfo.sex_choices).get(item['sex'], '未知')

    # 2. 职业分布 (取前10大职业)
    occ_dist = list(UserInfo.objects.values('occupation').annotate(count=Count('occupation')).order_by('-count')[:10])
    for item in occ_dist:
        item['name'] = dict(UserInfo.OCCUPATION_CHOICES).get(item['occupation'], '未填写')

    return JsonResponse({
        'sex_data': sex_dist,
        'occ_data': occ_dist
    })


@login_required
@never_cache
def chat_view(request):
    """
    渲染聊天页面，并加载最近的历史记录
    """
    # 1. 加载最近 24 小时的聊天记录 (时间可以改长一点，比如24小时，方便演示)
    time_threshold = timezone.now() - timedelta(hours=24)

    history_qs = ChatHistory.objects.filter(
        user=request.user,
        timestamp__gte=time_threshold
    ).order_by('timestamp')

    # 2. 转换为 JSON 格式供前端 JS 使用
    history_list = []
    for msg in history_qs:
        history_list.append({
            'role': msg.role,  # 'user' or 'ai'
            'message': msg.message
        })

    context = {
        'chat_history_json': json.dumps(history_list)
    }
    return render(request, 'chat.html', context)


# ==========================================
# 2. 聊天 API 接口 (POST) - ajax_chat 异步重构版本
# ==========================================
# ⚠️  注意：此函数绝对不能加 @require_POST / @csrf_exempt 装饰器！
# 原因：两者都会生成同步 wrapper，导致 Django 的 iscoroutinefunction() 检测失败，
#      Django 会以同步方式调用本视图，拿到 coroutine 而非 HttpResponse，抛出 ValueError。
# 解决方案：在函数体内手动处理 POST 校验，csrf_exempt 用属性方式设置（见函数结尾）。
async def ajax_chat(request):
    """
    智能观影助手 V108 - 异步重构版
    使用 asyncio + sync_to_async 提升高并发性能

    架构：
    - 参数接收 -> 意图识别 -> 路由调用辅助函数 -> LLM生成 -> 思考链提取 -> 响应前端
    """
    # 手动替代 @require_POST（避免同步装饰器破坏 async 特征）
    if request.method != 'POST':
        return JsonResponse({'error': 'Method Not Allowed'}, status=405)

    t_start = time.time()

    # ==========================================
    # 1. 参数接收与预处理
    # ==========================================
    raw_input = request.POST.get('msg', '').strip()
    is_thinking_mode = request.POST.get('is_thinking') == 'true'

    if not raw_input:
        return JsonResponse({'response': '您好！我是您的"智能观影助手"，请问今天想看点什么类型的电影？'})

    user_input = sanitize_user_input(raw_input)
    
    # 🔴 【关键调试】打印 sanitize_user_input 的返回值
    logger.debug(f"[Sanitize返回] user_input = {repr(user_input)}")
    logger.debug(f"[类型检查] type(user_input) = {type(user_input)}, len = {len(user_input) if isinstance(user_input, str) else 'N/A'}")

    # 🛡️ 【前置拦截】Circuit Breaker 熔断检查
    # 如果检测到高危 Prompt Injection，直接拦截，不进行任何后续处理
    if user_input == "MALICIOUS_INJECTION_DETECTED":
        logger.warning("[系统拦截] Prompt Injection 熔断触发，拒绝处理恶意输入")
        return JsonResponse({
            'response': '🛡️ [系统拦截] 滴滴滴！检测到试图跨界或篡改设定的高危指令。我是一名忠诚的电影推荐助理，拒绝执行非本职工作哦。🎬'
        })
    else:
        logger.debug("[Sanitize未拦截] 拦截条件检查失败")

    logger.info(f"用户输入: {user_input} | Thinking模式: {is_thinking_mode}")

    # ==========================================
    # 2. 获取对话历史与初始化记忆管理 (异步)
    # ==========================================
    @sync_to_async
    def get_all_chat_history():
        # 获取完整的聊天历史（用于记忆初始化）
        all_msgs = list(ChatHistory.objects.filter(
            user=request.user
        ).order_by('timestamp'))
        
        # 转换为记忆格式
        history_list = []
        for msg in all_msgs:
            history_list.append({
                'role': msg.role,
                'content': msg.message
            })
        
        # 最近的对话（用于上下文）
        recent_ai_msgs = [msg for msg in all_msgs if msg.role == 'ai'][-2:]
        recent_user_msgs = [msg for msg in all_msgs if msg.role == 'user'][-2:]
        
        return history_list, recent_ai_msgs, recent_user_msgs
    
    history_list, recent_ai_msgs, recent_user_msgs = await get_all_chat_history()
    
    # 🔥 初始化 LangChain 记忆管理器
    # 根据上下文窗口大小自适应 Token 预算
    memory_token_limit = 1200 if is_thinking_mode else 600
    chat_memory = await sync_to_async(initialize_chat_memory)(
        history_list, max_token_limit=memory_token_limit
    )
    logger.info(f"[Memory] 初始化完成 | 历史记录: {len(history_list)} 条 | Token 预算: {memory_token_limit}")
    logger.info(f"[Memory 状态] {format_memory_for_display(chat_memory)}")

    # ==========================================
    # 3. 意图识别 + 智能路由
    # ==========================================
    # ✅ 修复 Bug 2: classify_intent_advanced → check_domain_entities → ORM
    # 必须通过 sync_to_async 在线程池中执行，否则抛出 SynchronousOnlyOperation
    intent = await sync_to_async(classify_intent_advanced)(user_input, history_msgs=recent_ai_msgs)
    _, neg_keywords = detect_negative_intent(user_input)

    # 处理追问逻辑
    search_query = user_input
    trigger_keywords = ['再', '还', '继续', '换', '类似', '别的', '其他', '再来']
    is_follow_up = any(k in user_input for k in trigger_keywords)

    if intent in ["QUERY_MOVIE", "QUERY_COMPARISON"] and is_follow_up and recent_user_msgs:
        last_user_msg = recent_user_msgs[0].message
        search_query = f"{last_user_msg} {user_input}"

    # 拦截域外请求
    if intent == "OUT_OF_DOMAIN":
        rejection_responses = [
            "抱歉，作为专属电影推荐助理，我的脑子里装满了胶片和故事，写代码或解数学题可不是我的强项哦。聊点电影相关的吧？🍿",
            "这个超出了我的专业领域啦！您可以问我一些关于好电影的问题，比如'最近有什么高分科幻片？'🎬",
            "我只是一名专注电影世界的 AI，这些无关影视的话题我们就不探讨啦。要不要我帮您挑一部适合今晚看的佳作？🎥"
        ]
        return JsonResponse({'response': random.choice(rejection_responses)})

    if intent == "CHAT" and await sync_to_async(check_domain_entities)(user_input):
        intent = "QUERY_MOVIE"

    # 🔥 LangChain LCEL 智能路由（替代硬编码的 if intent == ...）
    user_context = {
        'interaction_summary': None,  # 暂时为 None，稍后在获取用户画像时更新
        'is_thinking_mode': is_thinking_mode,
        'search_query': search_query,
        'is_follow_up': is_follow_up,
    }
    routing_result = await sync_to_async(route_user_intent)(
        intent, user_input, user_context
    )
    logger.info(f"[路由决策] {routing_result['description']} | 分支: {routing_result['branch']}")

    # ==========================================
    # 4. 模型配置
    # ==========================================
    if is_thinking_mode:
        target_model = "qwen3-vl:4b"
        num_ctx = 4096
        logger.info(f"[算力爆发] 切换至高级模型: {target_model}")
    else:
        target_model = "qwen3:4b-instruct"
        num_ctx = 2048
        logger.info(f"[极速模式] 使用基础模型: {target_model}")

    # ==========================================
    # 5. 获取用户画像
    # ==========================================
    # ✅ 修复 Bug 3: get_user_interaction_summary_enhanced → ORM，必须在线程池执行
    interaction_summary, profile_status = await sync_to_async(
        get_user_interaction_summary_enhanced
    )(request.user)
    
    # 🔥 更新路由上下文（加入用户画像信息）
    user_context['interaction_summary'] = interaction_summary

    # ==========================================
    # 6. 路由调用辅助函数生成 Prompt
    # ==========================================
    prompt_builders = {
        "QUERY_VISUAL": lambda: _build_visual_prompt(user_input, request, is_thinking_mode),
        "QUERY_VISUAL_RETRY": lambda: _build_visual_prompt(user_input, request, is_thinking_mode),
        "QUERY_PROFILE_REC": lambda: _build_profile_rec_prompt(request.user, interaction_summary),
        "QUERY_SELF": lambda: _build_self_profile_prompt(request.user, interaction_summary),
        "QUERY_RANK": lambda: _build_rank_prompt(user_input),
        "QUERY_NEW": lambda: _build_new_movies_prompt(),
        "QUERY_MOVIE": lambda: _build_movie_recommendation_prompt(
            request.user, search_query, is_thinking_mode, is_follow_up,
            interaction_summary=interaction_summary,   # 🔥 注入用户画像 + Expert Prior
        ),
        "QUERY_COMPARISON": lambda: _build_movie_recommendation_prompt(
            request.user, search_query, is_thinking_mode, is_follow_up,
            interaction_summary=interaction_summary,   # 🔥 注入用户画像 + Expert Prior
        ),
        "CHAT": lambda: _build_chat_prompt(user_input),
    }

    builder = prompt_builders.get(intent)
    if builder:
        # ✅ 修复 Bug 4a: 所有 builder 函数内部可能包含 ORM 调用（_build_rank_prompt、
        # _build_new_movies_prompt、_build_movie_recommendation_prompt 等），
        # 必须用 sync_to_async 包装后在线程池中执行
        visual_response, final_prompt, temperature = await sync_to_async(builder)()
    else:
        visual_response, final_prompt, temperature = None, None, 0.3

    # 如果有直接返回的视觉结果，直接返回
    if visual_response:
        return JsonResponse({'response': visual_response})

    # ✅ 修复 Bug 4b: final_prompt 为 None 时（意图未知等边界情况）提前兜底，避免 LLM 崩溃
    if not final_prompt:
        return JsonResponse({'response': "抱歉，我暂时无法理解您的意图，请换个方式问我吧～"})

    # ==========================================
    # 6.5. 记忆增强 Prompt 构造 (LangChain 高级机制 #1)
    # ==========================================
    # 🔥 使用 ConversationSummaryBufferMemory 增强 Prompt
    # 仅对非视觉搜索的意图进行记忆增强
    if intent not in ["QUERY_VISUAL", "QUERY_VISUAL_RETRY"]:
        # 初始 RAG 上下文为空（后续如有 RAG 检索会补充）
        rag_context = ""
        
        # 用记忆增强的 Prompt 替换原始 Prompt
        final_prompt = build_memory_enhanced_prompt(
            user_input=user_input,
            memory=chat_memory,
            rag_context=rag_context,
            system_role="电影推荐助手"
        )
        logger.info(f"[Prompt 增强] 注入记忆摘要和对话历史 | Prompt 长度: {len(final_prompt)}")
    else:
        logger.info("[Prompt 增强] 跳过（视觉搜索模式）")

    # ==========================================
    # 6.6. RAG 检索优化 (LangChain 高级机制 #2)
    # ==========================================
    # 🔥 使用 ContextualCompressionRetriever 对 RAG 结果进行语义压缩
    # （如果实现了 RAG 召回，可在此处调用 compress_retrieval_results）
    rag_compression_threshold = 0.75
    logger.info(f"[Compression] 准备就绪，相似度阈值: {rag_compression_threshold}")

    # ==========================================
    # 6.7. KAG 路由的导演模糊匹配（使用 DIRECTOR_STYLES）
    # ==========================================
    # 🔥 导演美学签名库（数据库驱动版：种子库 + 自动补全）
    from myapp.utils.director_styles import get_director_styles
    DIRECTOR_STYLES = get_director_styles()

    # 变量约定（用于后续 Prompt 注入与首段强制关键词要求）
    matched_director = None
    kag_evidence = None  # ① 命中后：只存放“导演美学签名”（不含事实核查）
    kag_type = None

    user_input_lower = user_input.lower()

    # 1) 强制命中规则：只要提到“诺兰”或“Nolan”，必须强行命中 Christopher Nolan
    nolan_force_pattern = r"(?:诺兰|nolan|christopher\s*nolan)"
    if re.search(nolan_force_pattern, user_input_lower, flags=re.IGNORECASE):
        matched_director = "Christopher Nolan"
        kag_evidence = DIRECTOR_STYLES.get(matched_director, "")
        kag_type = "导演美学"
        logger.info("[KAG 路由] 命中关键字(诺兰/Nolan) → 强制映射到 Christopher Nolan")
    else:
        # 2) 路由算法升级：遍历 DIRECTOR_STYLES 的所有键；使用正则或 keyword in 做软匹配
        #    要点：不要做“精确 in（整段导演名完全等于）”式匹配，改为 token 级软命中。
        for director_name in DIRECTOR_STYLES.keys():
            tokens = [t.strip().lower() for t in director_name.split() if t.strip()]
            tokens = [t for t in tokens if len(t) >= 3]
            if not tokens:
                continue

            hit = False
            for kw in tokens:
                # regex：单词边界命中（更稳）；同时保留简单 in 作为兜底
                if re.search(rf"\b{re.escape(kw)}\b", user_input_lower, flags=re.IGNORECASE):
                    hit = True
                    break
                if kw in user_input_lower:
                    hit = True
                    break

            if hit:
                matched_director = director_name
                kag_evidence = DIRECTOR_STYLES.get(matched_director, "")
                kag_type = "导演美学"
                logger.info(f"[KAG 模糊匹配] 命中导演 token → {matched_director}")
                break

    # 3) 知识强制注入：命中后将美学签名写入 kag_evidence
    kag_knowledge_content = None
    fact_check_line = "【事实核查】：如果提及克里斯托弗·诺兰，请确保承认《追随》(Following) 是他的导演处女作，不要产生误导。"

    if matched_director and kag_evidence:
        # 仅用于“Prompt 注入块”的完整内容；kag_evidence 本身仍保持为美学签名（用于首段关键词要求）
        kag_knowledge_content = kag_evidence

        # 4) 纠偏 Prompt：事实核查行必须进入发送给 LLM 的 Prompt，并且不允许被截断吞掉
        if matched_director == "Christopher Nolan":
            kag_knowledge_content = f"{kag_evidence}\n{fact_check_line}"

    # ==========================================
    # 6.8. KAG 检索结果的动态注入与 Prompt 增强
    # ==========================================
    if kag_knowledge_content:
        # 截断仅作用于“美学签名正文”，不作用于事实核查行
        max_kag_sig_len = 400 if num_ctx <= 2048 else 800

        if intent not in ["QUERY_VISUAL", "QUERY_VISUAL_RETRY"]:
            if matched_director == "Christopher Nolan":
                sig_part = (kag_evidence or "")[:max_kag_sig_len]
                # 关键：把事实核查“单独一行”写进最终 Prompt
                kag_injection = (
                    "\n\n【导演知识库】\n"
                    f"您提到的是 {matched_director}。\n"
                    f"其美学签名：{sig_part}\n"
                    f"{fact_check_line}"
                )
            else:
                sig_part = (kag_evidence or "")[:max_kag_sig_len]
                kag_injection = (
                    "\n\n【导演知识库】\n"
                    f"您提到的是 {matched_director}，其美学特征：{sig_part}"
                )

            final_prompt = final_prompt + kag_injection

        logger.info(f"[KAG强制注入] 已注入导演 {matched_director} 到 Prompt")
    else:
        logger.info("[KAG路由] 未匹配任何导演信息")

    # ==========================================
    # 6.9. 🔥 Prompt 纠偏：强制融入导演美学特征词汇
    # ==========================================
    # 如果命中导演 KAG 路由，添加指导：要求 LLM 在回复第一段（首段/首句到第一个换行前）自然融入美学特征词汇
    if matched_director and kag_evidence:
        aesthetic_keywords = [kw.strip() for kw in re.split(r"[、,，/]+", kag_evidence) if kw.strip()]
        # 取前 4 个避免太长，但仍保持“字典关键词”在首段出现
        aesthetic_keywords = aesthetic_keywords[:4]
        keywords_str = "、".join(aesthetic_keywords)

        kag_guidance = (
            "\n\n【重要指导】：请把下面这些“导演美学签名关键词”自然融入你的回复【第一段】的第一句内（换行前必须出现至少2个关键词）："
            f"{keywords_str}。请不要做无意义堆砌，保证语义自然。"
        )
        _captured_prompt = final_prompt + kag_guidance
        logger.info(f"[Prompt 纠偏] 已添加导演美学融入指导 | 关键词: {keywords_str}")
    else:
        _captured_prompt = final_prompt

    # ==========================================
    # 7. LLM 生成
    # ==========================================
    try:
        # 🔥 修复：将 LLM 调用封装进 sync_to_async 的线程池中执行。
        # 原因：asyncio.Lock 在 Python 3.9 WSGI 模式下会跨事件循环导致异常，
        #      改用 threading.Lock 配合线程池调用，彻底规避该问题。
        # 同时为 4B instruct 模型增加 num_predict=800，防止幻觉时无限生成导致超时。
        _captured_model = target_model
        _captured_temp = temperature
        _captured_ctx = num_ctx

        def _blocking_llm_call():
            with LLM_CHAT_LOCK:
                import asyncio as _asyncio
                _llm = ChatOllama(
                    model=_captured_model,
                    temperature=_captured_temp,
                    top_p=0.8,
                    num_ctx=_captured_ctx,
                    num_predict=800,  # 🔥 限制最大生成长度，防止模型幻觉时跑死
                )
                # 在独立线程中创建新事件循环运行异步 LLM 调用
                _loop = _asyncio.new_event_loop()
                try:
                    return _loop.run_until_complete(_llm.ainvoke(_captured_prompt))
                finally:
                    _loop.close()

        t_gen_start = time.time()
        response = await sync_to_async(_blocking_llm_call)()
        raw_content = response.content.strip()
        t_llm_duration = time.time() - t_gen_start

        # ==========================================
        # 8. 思考链提取与后处理
        # ==========================================
        content = _extract_think_content(raw_content)
        content = clean_markdown_marks(content)

        if not content:
            content = "抱歉，我的思绪刚刚稍微飘远了一点。能换个方式再问我一次吗？"

        # 电影链接注入
        # ✅ 修复 Bug 4c: inject_movie_links → Movie.objects.filter (ORM)，必须在线程池执行
        if 'inject_movie_links' in globals():
            content = await sync_to_async(inject_movie_links)(content)
            found_ids = re.findall(r'\(ID:(\d+)\)', content)
            if found_ids and intent == "QUERY_MOVIE":
                @sync_to_async
                def get_rec_movies():
                    return list(Movie.objects.filter(id__in=found_ids[:3]))

                rec_movies = await get_rec_movies()
                if rec_movies:
                    card_html = '<div class="visual-search-container" style="display: flex; gap: 12px; overflow-x: auto; padding: 10px 0;">'
                    for rm in rec_movies:
                        p_url = rm.poster_file.url if rm.poster_file else "/static/img/no_poster.png"
                        d_url = reverse('movie_detail', args=[rm.id])
                        card_html += f"""
                                <div class="visual-card" style="min-width: 120px; max-width: 120px; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); border: 1px solid #eee;">
                                    <a href="{d_url}" target="_blank" style="text-decoration: none;">
                                        <img src="{p_url}" style="width: 100%; height: 160px; object-fit: cover;">
                                        <div style="padding: 5px; font-size: 11px; font-weight: bold; color: #333; text-align: center; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{rm.title}</div>
                                    </a>
                                </div>"""
                    card_html += '</div>'
                    content += f"<br><b>🎬 馆藏直达：</b>{card_html}"

        # 异步保存历史 + 更新 LangChain 记忆
        @sync_to_async
        def save_chat_history():
            ChatHistory.objects.create(user=request.user, role='user', message=user_input)
            ChatHistory.objects.create(user=request.user, role='ai', message=content)

        await save_chat_history()
        
        # 🔥 更新 LangChain 记忆管理器（用于下一轮对话）
        chat_memory.add_message('user', user_input)
        chat_memory.add_message('ai', content)
        memory_stats = format_memory_for_display(chat_memory)
        logger.info(f"[Memory 更新] 消息已添加到记忆 | {memory_stats}")

        t_total = time.time() - t_start
        logger.info(f"[Agent] 总: {t_total:.2f}s | 意图: {intent} | 模式: {'Thinking' if is_thinking_mode else 'Fast(4B)'} | LLM: {t_llm_duration:.2f}s")

        # 显存保护
        if is_thinking_mode and _safe_cuda_available():
            try:
                torch.cuda.empty_cache()
            except Exception: pass
            logger.info("[显存管理] 深度思考结束，已主动回收显存。")

        return JsonResponse({'response': content})

    except Exception as e:
        logger.error(f"Chat Error: {e}")
        traceback.print_exc()

        if is_thinking_mode and _safe_cuda_available():
            try:
                torch.cuda.empty_cache()
            except Exception: pass

        return JsonResponse({'response': "智能观影助手的大脑正在飞速运转（可能算力超载），请稍后刷新重试。"})


# ✅ 用属性方式设置 CSRF 豁免，等价于 @csrf_exempt，但不包裹任何 wrapper，
# CSRF 中间件检测的是 view_func.csrf_exempt 属性，不要求必须是装饰器形式。
ajax_chat.csrf_exempt = True


# ==========================================
# 3. 清除历史
# ==========================================
@login_required
def chat_clear_history(request):
    """
    清空当前用户的聊天记录
    """
    ChatHistory.objects.filter(user=request.user).delete()
    return redirect('chat_view')

from django.http import JsonResponse
from py2neo import Graph

# 连接 Neo4j
try:
    from django.conf import settings as _s
    neo_graph = Graph(
        getattr(_s, 'NEO4J_URI', 'bolt://localhost:7687'),
        auth=(getattr(_s, 'NEO4J_USER', 'neo4j'), getattr(_s, 'NEO4J_PASSWORD', ''))
    )
except:
    logger.warning("无法连接到 Neo4j 数据库")
    neo_graph = None



# ═══════════════════════════════════════════════════
#  "不喜欢"排除列表（全局生效）
#  使用 UserFeedback(feedback_type='dislike') 存储
# ═══════════════════════════════════════════════════

def get_excluded_movie_ids(user):
    """
    获取用户标记为"不喜欢"的电影ID列表（全局辅助函数）
    供 recommend()、Agent推荐 等多个视图复用
    
    ★ 未成年人内容保护：自动追加不适宜内容的排除列表
    """
    from myapp.models_upgrade import UserFeedback
    if not user.is_authenticated:
        return []
    
    # 基础排除：用户手动标记的"不喜欢"
    excluded = set(
        UserFeedback.objects.filter(user=user, feedback_type='dislike')
        .values_list('movie_id', flat=True)
    )
    
    # 未成年人内容保护：自动追加不适宜电影
    try:
        from myapp.utils.content_safety import get_minor_excluded_ids
        minor_excluded = get_minor_excluded_ids(user)
        if minor_excluded:
            excluded.update(minor_excluded)
    except Exception:
        pass
    
    return list(excluded)


@csrf_exempt
@login_required
def ajax_exclude_add(request):
    """
    将电影加入"不喜欢"排除列表
    POST 参数: movie_id
    返回: {"status": "ok", "movie_id": N, "title": "..."}
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'msg': 'Method Not Allowed'}, status=405)

    movie_id = request.POST.get('movie_id')
    if not movie_id:
        return JsonResponse({'status': 'error', 'msg': '缺少 movie_id'})

    from myapp.models_upgrade import UserFeedback
    try:
        movie = Movie.objects.get(pk=movie_id)
    except Movie.DoesNotExist:
        return JsonResponse({'status': 'error', 'msg': '电影不存在'})

    # 先移除已有的 dislike 记录（如有），再创建（防重复）
    UserFeedback.objects.filter(
        user=request.user, movie=movie, feedback_type='dislike'
    ).delete()
    UserFeedback.objects.create(
        user=request.user, movie=movie,
        feedback_type='dislike', source=request.POST.get('source', 'recommend_page')
    )

    # 清除推荐缓存，确保下次推荐时排除生效
    cache_key = f"hybrid_rec_{request.user.id}"
    cache.delete(cache_key)
    cache_key2 = f"cb_recs_{request.user.id}"
    cache.delete(cache_key2)

    # ★ 查找候补电影：从 Rec 表中排除所有已显示+已不喜欢的电影，取下一个
    excluded_ids = set(get_excluded_movie_ids(request.user))
    displayed_ids = request.POST.get('displayed_ids', '')
    if displayed_ids:
        try:
            for did in displayed_ids.split(','):
                did = did.strip()
                if did.isdigit():
                    excluded_ids.add(int(did))
        except Exception:
            pass

    backup_movie = None
    backup_rec = (
        Rec.objects.filter(user=request.user)
        .exclude(movie_id__in=excluded_ids)
        .select_related('movie')
        .order_by('-rating')
        .first()
    )
    if backup_rec and backup_rec.movie:
        m = backup_rec.movie
        poster_url = ''
        if hasattr(m, 'poster_file') and m.poster_file:
            poster_url = m.poster_file.url
        elif m.poster:
            poster_url = m.poster
        backup_movie = {
            'id': m.id,
            'title': m.title,
            'score': float(m.score) if m.score else 0,
            'poster': poster_url,
            'rating': round(float(backup_rec.rating), 2) if backup_rec.rating else 0,
        }

    response_data = {
        'status': 'ok',
        'movie_id': int(movie_id),
        'title': movie.title,
        'msg': f'已将《{movie.title}》加入排除列表',
    }
    if backup_movie:
        response_data['backup_movie'] = backup_movie
    return JsonResponse(response_data)


@csrf_exempt
@login_required
def ajax_exclude_remove(request):
    """
    将电影从"不喜欢"排除列表中移除（恢复推荐）
    POST 参数: movie_id
    返回: {"status": "ok", "movie_id": N, "title": "..."}
    """
    if request.method != 'POST':
        return JsonResponse({'status': 'error', 'msg': 'Method Not Allowed'}, status=405)

    movie_id = request.POST.get('movie_id')
    if not movie_id:
        return JsonResponse({'status': 'error', 'msg': '缺少 movie_id'})

    from myapp.models_upgrade import UserFeedback
    try:
        movie = Movie.objects.get(pk=movie_id)
    except Movie.DoesNotExist:
        return JsonResponse({'status': 'error', 'msg': '电影不存在'})

    deleted, _ = UserFeedback.objects.filter(
        user=request.user, movie=movie, feedback_type='dislike'
    ).delete()

    # 清除推荐缓存
    cache_key = f"hybrid_rec_{request.user.id}"
    cache.delete(cache_key)
    cache_key2 = f"cb_recs_{request.user.id}"
    cache.delete(cache_key2)

    return JsonResponse({
        'status': 'ok',
        'movie_id': int(movie_id),
        'title': movie.title,
        'msg': f'已将《{movie.title}》从排除列表中移除'
    })


@login_required
def ajax_exclude_list(request):
    """
    获取用户的排除列表（用于个人中心展示）
    返回: {"status": "ok", "excluded_movies": [...]}
    """
    from myapp.models_upgrade import UserFeedback
    excluded = UserFeedback.objects.filter(
        user=request.user, feedback_type='dislike'
    ).select_related('movie').order_by('-created_at')

    movies = []
    for fb in excluded:
        m = fb.movie
        poster_url = ''
        if m.poster_file:
            poster_url = m.poster_file.url
        elif m.poster:
            poster_url = m.poster
        movies.append({
            'id': m.id,
            'title': m.title,
            'poster': poster_url,
            'reason': fb.source or '',
            'created_at': fb.created_at.strftime('%Y-%m-%d %H:%M') if fb.created_at else '',
        })

    return JsonResponse({'status': 'ok', 'excluded_movies': movies})


@login_required
def admin_kg_view(request):
    """
    知识图谱可视化全屏展示页
    """
    return render(request, 'admin_panel/kg_view.html')


def api_kg_data(request):
    """
    知识图谱概览数据 API（ECharts 图格式）
    从 Neo4j 查询 Top 300 条关系，返回 {nodes, links, categories}
    供 admin_kg_view.html 及 agent 前端复用
    """
    _g = neo_graph
    if _g is None:
        return JsonResponse({"nodes": [], "links": [], "categories": []})

    try:
        cypher = """
        MATCH (m:Movie)<-[:DIRECTED_BY]-(d:Person)
        WITH m, d LIMIT 150
        OPTIONAL MATCH (m)-[:BELONGS_TO]->(g:Genre)
        WITH m, d, g
        RETURN m.mid AS mid, m.title AS mtitle,
               d.name AS dname, g.name AS gname
        LIMIT 300
        """
        rows = _g.run(cypher).data()

        nodes_dict = {}
        links_set = set()
        cat_map = {"movie": 0, "director": 1, "genre": 2}

        for r in rows:
            mid = r.get('mid')
            if mid is None:
                continue
            # Movie node
            m_id = f"m_{mid}"
            if m_id not in nodes_dict:
                nodes_dict[m_id] = {
                    "id": m_id,
                    "name": r.get('mtitle', f'Movie#{mid}'),
                    "category": 0,
                    "symbolSize": 30,
                }
            # Director node
            dname = r.get('dname')
            if dname:
                d_id = f"d_{dname}"
                if d_id not in nodes_dict:
                    nodes_dict[d_id] = {
                        "id": d_id,
                        "name": dname,
                        "category": 1,
                        "symbolSize": 20,
                    }
                links_set.add((d_id, m_id, "导演"))
            # Genre node
            gname = r.get('gname')
            if gname:
                g_id = f"g_{gname}"
                if g_id not in nodes_dict:
                    nodes_dict[g_id] = {
                        "id": g_id,
                        "name": gname,
                        "category": 2,
                        "symbolSize": 15,
                    }
                links_set.add((m_id, g_id, "类型"))

        links_list = [{"source": s, "target": t, "value": v} for s, t, v in links_set]
        categories = [
            {"name": "电影", "itemStyle": {"color": "#d9534f"}},
            {"name": "导演", "itemStyle": {"color": "#9b59b6"}},
            {"name": "类型", "itemStyle": {"color": "#5cb85c"}},
        ]

        return JsonResponse({
            "nodes": list(nodes_dict.values()),
            "links": links_list,
            "categories": categories,
        })

    except Exception as e:
        logger.error(f"[api_kg_data] Error: {e}")
        return JsonResponse({"nodes": [], "links": [], "categories": []})

