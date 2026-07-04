import os, time, requests, dotenv
from django.core.management.base import BaseCommand
from myapp.models import Movie, Actor, Genre, Region
from django.db.models import Q


class Command(BaseCommand):
    help = '使用标题搜索补全缺失的 900+ 部电影数据'

    def handle(self, *args, **options):
        dotenv.load_dotenv()
        api_key = os.getenv("TMDB_API_KEY")

        # 1. 筛选出 summary 为空或者 score 为空的“空壳电影”
        missing_movies = Movie.objects.filter(
            Q(summary__isnull=True) | Q(summary="") | Q(score__isnull=True)
        )
        total = missing_movies.count()
        self.stdout.write(f"待补救电影数: {total}")

        search_url = "https://api.themoviedb.org/3/search/movie"
        poster_base = "https://image.tmdb.org/t/p/w500"

        for i, movie in enumerate(missing_movies):
            try:
                # 去掉括号里的年份，搜索更准。如 "Toy Story (1995)" -> "Toy Story"
                clean_name = movie.title.split('(')[0].strip()

                # 2. 调用搜索接口
                params = {'api_key': api_key, 'query': clean_name, 'language': 'zh-CN'}
                r = requests.get(search_url, params=params, timeout=5)
                results = r.json().get('results')

                if results:
                    data = results[0]  # 取最匹配的一个

                    # 3. 补全核心字段
                    movie.summary = data.get('overview')
                    movie.score = data.get('vote_average')
                    movie.date = data.get('release_date') or None
                    if data.get('poster_path'):
                        movie.poster = poster_base + data.get('poster_path')

                    movie.save()
                    self.stdout.write(f"[{i + 1}/{total}] 成功救回: {movie.title}")

                time.sleep(0.1)  # 补漏不需要太慢
            except Exception:
                continue

        self.stdout.write(self.style.SUCCESS("--- 补漏完成 ---"))