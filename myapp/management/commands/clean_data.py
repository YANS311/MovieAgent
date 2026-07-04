# 文件: myapp/management/commands/clean_data.py (V2 - 修复唯一约束冲突)

import os
import csv
from django.core.management.base import BaseCommand
from myapp.models import Movie, UserRating, Collect, Rec
from django.db import transaction
from django.db.models import Count, Q


class Command(BaseCommand):
    help = '数据清洗：合并重复电影，清除无效数据 (修复 IntegrityError)'

    def handle(self, *args, **options):
        self.stdout.write("--- 开始数据清洗 ---")

        # --- 1. 加载 links.csv (保持不变) ---
        links_file = 'links.csv'
        tmdb_to_ml_map = {}
        if os.path.exists(links_file):
            with open(links_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    if len(row) >= 3 and row[0] and row[2]:
                        tmdb_to_ml_map[row[2]] = row[0]
        self.stdout.write(f"加载了 {len(tmdb_to_ml_map)} 条 TMDB->ML 映射关系。")

        # --- 2. 执行 "跨库合并" (保持不变) ---
        self.stdout.write("正在检查跨库重复 (TMDB vs MovieLens)...")
        new_movies = Movie.objects.filter(movielens_id__isnull=True, imdb_id__isnull=False)

        merged_count = 0
        with transaction.atomic():
            for new_movie in new_movies:
                tmdb_id = new_movie.imdb_id
                target_ml_id = tmdb_to_ml_map.get(tmdb_id)

                if target_ml_id:
                    try:
                        old_movie = Movie.objects.get(movielens_id=target_ml_id)
                        self.merge_movies(new_movie, old_movie)
                        merged_count += 1
                        if merged_count % 50 == 0:
                            self.stdout.write(f"已合并: {new_movie.title} -> {old_movie.title}")

                    except Movie.DoesNotExist:
                        # (这里有个小坑：如果 target_ml_id 已经被占用了怎么办？)
                        # 为了安全，我们先检查一下
                        if not Movie.objects.filter(movielens_id=target_ml_id).exists():
                            new_movie.movielens_id = target_ml_id
                            new_movie.save()

        self.stdout.write(self.style.SUCCESS(f"跨库合并完成：共合并了 {merged_count} 部电影。"))

        # --- 3. 执行 "标题去重" (保持不变) ---
        self.stdout.write("正在检查标题重复...")
        duplicates = Movie.objects.values('title').annotate(count=Count('id')).filter(count__gt=1)

        title_dedup_count = 0
        with transaction.atomic():
            for dup in duplicates:
                title = dup['title']
                candidates = Movie.objects.filter(title=title).order_by(
                    '-movielens_id', '-vote_count', '-id'
                )

                if candidates.count() < 2: continue

                winner = candidates[0]
                losers = candidates[1:]

                for loser in losers:
                    self.merge_movies(loser, winner)
                    title_dedup_count += 1

        self.stdout.write(self.style.SUCCESS(f"标题去重完成：共清理了 {title_dedup_count} 部重复电影。"))

        # --- 4. 执行 "垃圾清理" (保持不变) ---
        self.stdout.write("正在清理无效数据 (无简介且无海报)...")
        garbage = Movie.objects.filter(
            movielens_id__isnull=True
        ).filter(
            Q(summary__isnull=True) | Q(summary='') | Q(poster__isnull=True) | Q(poster='')
        )
        deleted_count, _ = garbage.delete()
        self.stdout.write(self.style.SUCCESS(f"垃圾清理完成：删除了 {deleted_count} 部低质量电影。"))

        self.stdout.write(self.style.SUCCESS(f"--- 所有清洗工作完成，当前电影总数: {Movie.objects.count()} ---"))

    def merge_movies(self, source, target):
        """
        (核心修复) 将 source 电影的数据和关联项合并到 target，然后删除 source
        """
        # 1. 如果 target 缺数据，用 source 补全 (内存操作)
        if not target.summary and source.summary:
            target.summary = source.summary
        if not target.poster and source.poster:
            target.poster = source.poster
        if not target.date and source.date:
            target.date = source.date

        # 2. 迁移关联数据 (UserRating, Collect, Rec)
        # (使用 ignore_conflicts=True 类似的逻辑，或者先删除冲突的)

        # 处理评分冲突：如果 target 也有同一个用户的评分，保留 target 的，删除 source 的
        source_ratings = UserRating.objects.filter(movie=source)
        for rating in source_ratings:
            if not UserRating.objects.filter(user=rating.user, movie=target).exists():
                rating.movie = target
                rating.save()
            else:
                rating.delete()  # 冲突了，丢弃 source 的评分

        # 处理收藏冲突
        source_collects = Collect.objects.filter(movie=source)
        for collect in source_collects:
            if not Collect.objects.filter(user=collect.user, movie=target).exists():
                collect.movie = target
                collect.save()
            else:
                collect.delete()

        # 处理推荐冲突
        source_recs = Rec.objects.filter(movie=source)
        for rec in source_recs:
            if not Rec.objects.filter(user=rec.user, movie=target).exists():
                rec.movie = target
                rec.save()
            else:
                rec.delete()

        # 迁移 M2M
        target.actors.add(*source.actors.all())
        target.genres.add(*source.genres.all())
        target.regions.add(*source.regions.all())
        target.directors.add(*source.directors.all())


        # 3. 处理唯一字段 (imdb_id)
        # 如果我们需要把 source.douban_id 赋给 target...
        if not target.imdb_id and source.imdb_id:
            # A. 先把想要的值存下来
            new_imdb_id = source.imdb_id

            # B. 关键：先把 source 的 douban_id 设为 None 并保存！
            #    这样就释放了 "Unique" 约束的占用
            source.imdb_id = None
            source.save()

            # C. 现在 target 可以安全地使用这个 ID 了
            target.imdb_id = new_imdb_id

        # 同理处理 movielens_id (虽然逻辑上不太可能走到这步，但为了健壮性)
        if not target.movielens_id and source.movielens_id:
            new_ml_id = source.movielens_id
            source.movielens_id = None
            source.save()
            target.movielens_id = new_ml_id


        target.save()

        # 4. 删除源电影
        source.delete()