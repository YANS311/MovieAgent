#!/usr/bin/env python
"""
测试 ajax_explain_rec 的修复是否有效
"""
import os
import sys
import django

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie.settings')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

django.setup()

from django.test import RequestFactory
from django.contrib.auth import get_user_model
from myapp.views import ajax_explain_rec
from myapp.models import Movie
import json

# 获取用户模型
User = get_user_model()

# 创建一个测试请求
factory = RequestFactory()

# 获取或创建一个测试用户
user, _ = User.objects.get_or_create(username='testuser')

# 获取第一部电影（如果存在）
movie = Movie.objects.first()

if movie:
    print(f"✅ 测试电影: {movie.title} (ID: {movie.id})")

    # 创建请求
    request = factory.get('/ajax/explain_rec/', {'movie_id': movie.id, 'source': 'test'})
    request.user = user

    # 调用视图
    print("\n⏳ 正在调用 ajax_explain_rec...")
    try:
        response = ajax_explain_rec(request)

        # 解析 JSON 响应
        data = json.loads(response.content)

        print(f"\n✅ 响应状态: {response.status_code}")
        print(f"📊 响应数据:")
        print(f"   - status: {data.get('status')}")
        print(f"   - content 长度: {len(data.get('content', ''))}")
        print(f"\n📝 推荐内容:\n{data.get('content', '无内容')}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
else:
    print("❌ 数据库中没有电影数据")

