import os
import datetime
from django.utils import timezone
from django.core.management.base import BaseCommand
from tqdm import tqdm

from myapp.models import Movie, Genre, UserRating, UserInfo
from django.db import transaction


class Command(BaseCommand):
    help = '修正版：导入 ML-1M 数据 (保留时间戳 + 适配 Movie.score)'

    def handle(self, *args, **options):
        # 0. 路径检查
        ML_ROOT = 'ml-1m'
        if not os.path.exists(ML_ROOT):
            self.stderr.write(f"❌ 找不到目录: {ML_ROOT}")
            return

        # --- 1. 导入电影 (Movies) ---
        self.stdout.write("1. 正在导入电影...")
        movie_file_path = os.path.join(ML_ROOT, 'movies.dat')
        movie_map = {}  # ML_ID -> Movie_Obj

        # 预加载现有 Genre 避免重复查询
        all_genres = {g.name: g for g in Genre.objects.all()}

        movies_to_create = []
        # 注意：这里我们使用 bulk_create 可能比较麻烦因为涉及 M2M
        # 为了稳妥，还是逐个处理或分批，这里保持原逻辑但修正字段

        with open(movie_file_path, 'r', encoding='latin-1') as f:
            for line in f:
                if not line.strip(): continue
                parts = line.strip().split('::')
                ml_id = parts[0]
                title = parts[1].split('(')[0].strip()
                genres = parts[2].split('|')

                # 使用 get_or_create 防止重复
                movie, _ = Movie.objects.get_or_create(
                    movielens_id=ml_id,
                    defaults={'title': title}
                )
                movie_map[ml_id] = movie

                # 关联 Genre
                g_list = []
                for g_name in genres:
                    if g_name not in all_genres:
                        g_obj = Genre.objects.create(name=g_name)
                        all_genres[g_name] = g_obj
                    g_list.append(all_genres[g_name])
                movie.genres.set(g_list)

        self.stdout.write(f"✅ 电影导入完成。")

        # --- 2. 导入用户 (Users) ---
        self.stdout.write("2. 正在导入用户...")
        user_file_path = os.path.join(ML_ROOT, 'users.dat')
        users_map = {}  # ML_ID -> User_Obj

        # 为了防止主键冲突，先清空测试用户 (保留管理员)
        UserInfo.objects.filter(username__startswith='ml_user_').delete()

        # --- 2. 导入用户 (优化版) ---
        self.stdout.write(f"正在从 {user_file_path} 导入用户...")

        users_to_create = []
        gender_map = {'M': 1, 'F': 2}

        # 先读取所有数据到列表
        with open(user_file_path, 'r', encoding='latin-1') as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Hashing Passwords"):
            if not line.strip(): continue
            parts = line.strip().split('::')
            u_id = parts[0]

            # 创建内存对象，不保存
            user = UserInfo(
                username=f'ml_user_{u_id}',
                email=f'ml_{u_id}@example.com',
                user_ID=u_id,  # 适配 models.py
                sex=gender_map.get(parts[1], 1),  # 适配 models.py
                age=int(parts[2]),
                occupation=int(parts[3]),  # 适配 models.py
                zip_code=parts[4]
            )
            # 🔥 关键：在内存中先设置好密码（这一步依然耗时，但在 bulk_create 前做更快）
            user.set_password(os.getenv("DEFAULT_USER_PASSWORD", "changeme"))
            users_to_create.append(user)

        # 批量创建用户
        UserInfo.objects.bulk_create(users_to_create, ignore_conflicts=True)

        # 重新查询建立映射 (因为 bulk_create 不返回 ID)
        all_ml_users = UserInfo.objects.filter(username__startswith='ml_user_')
        for u in all_ml_users:
            # 提取 ML ID (username 是 ml_user_123)
            ml_id = u.username.split('_')[-1]
            users_map[ml_id] = u

        self.stdout.write(f"✅ 用户导入完成 ({len(users_map)} 人)。")

        # --- 3. 导入评分 (Ratings) [关键修正] ---
        self.stdout.write("3. 正在导入评分 (带时间戳)...")
        rating_file_path = os.path.join(ML_ROOT, 'ratings.dat')

        # 清空旧评分
        UserRating.objects.all().delete()

        ratings_batch = []
        BATCH_SIZE = 10000
        count = 0

        with open(rating_file_path, 'r', encoding='latin-1') as f:
            for line in f:
                parts = line.strip().split('::')
                u_id = parts[0]
                m_id = parts[1]
                rating_val = float(parts[2]) * 2  # 5分制 -> 10分制
                timestamp = int(parts[3])  # 🔥 读取第4列时间戳

                # 转换时间
                dt = datetime.datetime.fromtimestamp(timestamp, tz=timezone.utc)

                user = users_map.get(u_id)
                movie = movie_map.get(m_id)

                if user and movie:
                    ratings_batch.append(UserRating(
                        user=user,
                        movie=movie,
                        score=rating_val,
                        comment_time=dt,  # 🔥 显式设置时间
                        discussion="Fetched from MovieLens"
                    ))

                if len(ratings_batch) >= BATCH_SIZE:
                    UserRating.objects.bulk_create(ratings_batch)
                    count += len(ratings_batch)
                    ratings_batch = []
                    self.stdout.write(f"   已导入 {count} 条...")

        if ratings_batch:
            UserRating.objects.bulk_create(ratings_batch)

        self.stdout.write(self.style.SUCCESS(f"✅ 全部完成！真实序列数据已恢复。"))