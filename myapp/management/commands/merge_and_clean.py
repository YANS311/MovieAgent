import os
import csv
import re
from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Count, Q
from myapp.models import Movie, UserRating, Rec, Collect


class Command(BaseCommand):
    help = '数据清洗：合并中英文重复电影，优先保留中文信息和ML-1M的评分数据'

    def handle(self, *args, **options):
        links_file = 'links.csv'  # 确保这个文件在项目根目录
        if not os.path.exists(links_file):
            self.stderr.write("❌ 找不到 links.csv 文件，无法进行 ID 对齐！")
            return

        self.stdout.write("⚠️  开始执行数据合并与清洗...")

        # 1. 加载 ID 映射表 (MovieLens ID -> TMDB ID)
        ml_to_tmdb = {}
        with open(links_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # 跳过表头
            for row in reader:
                if len(row) >= 3:
                    # row[0]: ML_ID, row[1]: IMDB_ID, row[2]: TMDB_ID
                    ml_to_tmdb[row[0]] = row[2]

        merged_count = 0
        deleted_count = 0

        # 开启事务，确保数据安全
        with transaction.atomic():
            # 遍历所有有 MovieLens ID 的电影 (这是我们的核心资产，有评分)
            ml_movies = Movie.objects.filter(movielens_id__isnull=False)

            for ml_movie in ml_movies:
                tmdb_id = ml_to_tmdb.get(ml_movie.movielens_id)

                if not tmdb_id:
                    continue

                # 寻找“双胞胎”：数据库里是否有另一个电影，它的 tmdb_id 或 imdb_id 匹配，但没有 ML_ID
                # (这种通常是 import_new_movies 导入的纯 TMDB 数据)
                duplicates = Movie.objects.filter(
                    Q(imdb_id=tmdb_id) | Q(summary__contains=tmdb_id)  # 有时 ID 会混在其他字段，这里主要靠 imdb_id 匹配
                ).exclude(id=ml_movie.id)

                # 如果没找到 ID 匹配，尝试用中文名模糊匹配（慎用，这里只做精确 ID 匹配）
                # 现在的策略：如果 ml_movie 是英文，且我们找到了一个匹配 ID 的中文电影

                # 查找是否有对应的“中文版替身”
                # 假设我们之前导入过 TMDB 数据，它们可能存了 imdb_id 或者就是单纯的重复
                # 这里我们简化逻辑：如果我们发现当前电影是英文，我们去 TMDB 重新“洗”一遍它的标题

                pass

                # --- 上面的逻辑太复杂且依赖 ID 完整性，我们换一种更暴力的“去英文保中文”策略 ---

        self.stdout.write("🚀 策略调整：执行【英文清理与中文保留】...")

        # 1. 找出所有“看起来像英文”的电影 (包含字母，不包含中文)
        # 这里的正则匹配：不包含中文字符
        english_movies = []
        all_movies = Movie.objects.all()

        for m in all_movies:
            if not self.has_chinese(m.title):
                english_movies.append(m)

        self.stdout.write(f"发现 {len(english_movies)} 部非中文标题电影。正在尝试修复...")

        for eng_movie in english_movies:
            # 尝试找到对应的中文电影 (通过 ML_ID 关联的 TMDB ID，或者相同的 IMDB ID)
            # 如果这是一个 ML-1M 电影 (有评分)，我们不仅不能删，还得想办法把它变成中文

            # 方案 A: 它是 ML-1M 电影 (重要数据)
            if eng_movie.movielens_id:
                # 检查库里有没有同一个 ID 的中文版 (可能是重复导入导致的)
                # 或者是否有同一个 IMDB_ID 的中文版
                siblings = Movie.objects.filter(imdb_id=eng_movie.imdb_id).exclude(id=eng_movie.id)

                chinese_sibling = None
                for sib in siblings:
                    if self.has_chinese(sib.title):
                        chinese_sibling = sib
                        break

                if chinese_sibling:
                    self.stdout.write(f"  🔄 合并: {eng_movie.title} <- {chinese_sibling.title}")
                    # 把中文版的信息吸收到英文版 (因为英文版有 UserRating)
                    eng_movie.title = chinese_sibling.title
                    eng_movie.summary = chinese_sibling.summary or eng_movie.summary
                    eng_movie.poster = chinese_sibling.poster or eng_movie.poster
                    if not eng_movie.poster_file and chinese_sibling.poster_file:
                        eng_movie.poster_file = chinese_sibling.poster_file

                    eng_movie.save()

                    # 删掉那个没有评分的中文替身
                    chinese_sibling.delete()
                    merged_count += 1

                else:
                    # 如果没有替身，说明它就是单纯没翻译
                    # 可以在这里调用翻译接口，或者标记为待处理
                    pass

            # 方案 B: 它不是 ML-1M 电影 (没有评分，且是英文)
            # 这种通常是垃圾数据，直接删
            else:
                # 双重确认：真的没有评分吗？
                rating_count = UserRating.objects.filter(movie=eng_movie).count()
                if rating_count == 0:
                    self.stdout.write(f"  🗑️ 删除无用英文数据: {eng_movie.title}")
                    eng_movie.delete()
                    deleted_count += 1

        self.stdout.write(self.style.SUCCESS(f"✅ 清洗完成！合并了 {merged_count} 部，删除了 {deleted_count} 部废弃数据。"))

    def has_chinese(self, text):
        """判断是否包含中文字符"""
        if not text: return False
        return bool(re.search(r'[\u4e00-\u9fff]', text))