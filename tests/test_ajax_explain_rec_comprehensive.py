#!/usr/bin/env python
"""
完整测试 ajax_explain_rec 的所有场景
"""
import os
import sys
import django

# 设置 Django 环境
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie.settings')
sys.path.insert(0, '/home/daylight/下载/DjangoProject3/DjangoProject3')

django.setup()

from django.test import RequestFactory
from django.contrib.auth import get_user_model
from myapp.views import ajax_explain_rec
from myapp.models import Movie, UserRating
import json

# 获取用户模型
User = get_user_model()

# 创建一个测试请求
factory = RequestFactory()

# 创建或获取测试用户
user, created = User.objects.get_or_create(username='testuser_kag_test')
if created:
    print(f"✅ 创建新用户: {user.username}")
else:
    print(f"✅ 使用现有用户: {user.username}")

# 获取一些电影数据
movies = list(Movie.objects.all()[:5])

if len(movies) < 2:
    print("❌ 数据库中的电影数据不足")
    sys.exit(1)

print(f"\n📊 测试数据：")
print(f"   - 总电影数：{len(movies)}")

# 清除用户的历史评分（如果有的话）
UserRating.objects.filter(user=user).delete()
print(f"   - 已清除用户的历史评分")

# ==========================================
# 场景 1：冷启动（无观影历史）
# ==========================================
print("\n" + "="*60)
print("🧪 场景 1：冷启动（无观影历史）")
print("="*60)

target_movie = movies[0]
request = factory.get('/ajax/explain_rec/', {'movie_id': target_movie.id, 'source': 'test'})
request.user = user

try:
    response = ajax_explain_rec(request)
    data = json.loads(response.content)

    print(f"\n✅ 测试电影: {target_movie.title}")
    print(f"✅ 响应状态: {response.status_code}")
    print(f"📝 推荐内容（前 100 字）: {data.get('content', '')[:100]}...")

    if 'error' in data.get('status', ''):
        print(f"❌ 错误: {data.get('content')}")
    else:
        print(f"✅ 场景 1 通过")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# ==========================================
# 场景 2：有观影历史（KAG 分支）
# ==========================================
print("\n" + "="*60)
print("🧪 场景 2：有观影历史（KAG 分支）")
print("="*60)

# 创建一些历史评分
history_movie = movies[1]
rating = UserRating.objects.create(
    user=user,
    movie=history_movie,
    score=9.0,
    discussion="很好的电影"
)
print(f"\n✅ 为用户添加历史评分:")
print(f"   - 电影: {history_movie.title}")
print(f"   - 评分: 9.0")

# 选择不同的目标电影
target_movie2 = movies[2]
request2 = factory.get('/ajax/explain_rec/', {'movie_id': target_movie2.id, 'source': 'test'})
request2.user = user

try:
    response2 = ajax_explain_rec(request2)
    data2 = json.loads(response2.content)

    print(f"\n✅ 测试电影: {target_movie2.title}")
    print(f"✅ 响应状态: {response2.status_code}")
    print(f"📝 推荐内容（前 150 字）: {data2.get('content', '')[:150]}...")

    if 'error' in data2.get('status', ''):
        print(f"❌ 错误: {data2.get('content')}")
    else:
        print(f"✅ 场景 2 通过")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback
    traceback.print_exc()

# ==========================================
# 场景 3：不存在的电影 ID
# ==========================================
print("\n" + "="*60)
print("🧪 场景 3：不存在的电影 ID")
print("="*60)

request3 = factory.get('/ajax/explain_rec/', {'movie_id': 999999, 'source': 'test'})
request3.user = user

try:
    response3 = ajax_explain_rec(request3)
    data3 = json.loads(response3.content)

    print(f"✅ 响应状态: {response3.status_code}")
    print(f"📝 响应内容: {data3.get('content')}")

    if data3.get('status') == 'error':
        print(f"✅ 场景 3 通过（正确地返回错误）")
    else:
        print(f"⚠️  意外的响应状态")

except Exception as e:
    print(f"❌ 错误: {e}")

print("\n" + "="*60)
print("🎉 全部测试完成")
print("="*60)

