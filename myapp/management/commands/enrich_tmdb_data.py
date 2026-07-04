# 文件: myapp/management/commands/enrich_tmdb_data.py (V5 - 地区增强版)

import os
import csv
import time
import requests
import dotenv
from django.core.management.base import BaseCommand
from myapp.models import Movie, Actor, Region, Genre
from django.db import transaction
from django.db.models import Q, Count

# --- 1. 代理设置 ---
proxies_dict = {
    'http': 'http://127.0.0.1:7897',
    'https': 'http://127.0.0.1:7897',
}

# --- 2. (增强版) 地区名称规范化字典 ---
REGION_NORMALIZE_MAP = {
    # 亚洲
    "China": "中国大陆", "Hong Kong": "中国香港", "Taiwan": "中国台湾",
    "Japan": "日本", "South Korea": "韩国", "India": "印度", "Thailand": "泰国",
    "Vietnam": "越南", "Singapore": "新加坡", "Philippines": "菲律宾",
    "Indonesia": "印度尼西亚", "Malaysia": "马来西亚", "Iran": "伊朗",
    "Israel": "以色列", "Turkey": "土耳其",

    # 北美
    "United States of America": "美国", "Canada": "加拿大", "Mexico": "墨西哥",

    # 欧洲
    "United Kingdom": "英国", "France": "法国", "Germany": "德国",
    "Italy": "意大利", "Spain": "西班牙", "Russia": "俄罗斯", "Soviet Union": "苏联",
    "Sweden": "瑞典", "Netherlands": "荷兰", "Belgium": "比利时",
    "Denmark": "丹麦", "Norway": "挪威", "Finland": "芬兰",
    "Poland": "波兰", "Czech Republic": "捷克", "Hungary": "匈牙利",
    "Austria": "奥地利", "Switzerland": "瑞士", "Ireland": "爱尔兰",
    "Greece": "希腊", "Portugal": "葡萄牙", "Ukraine": "乌克兰",

    # 大洋洲
    "Australia": "澳大利亚", "New Zealand": "新西兰",

    # 南美
    "Brazil": "巴西", "Argentina": "阿根廷", "Chile": "智利", "Colombia": "哥伦比亚",

    # 非洲
    "South Africa": "南非", "Egypt": "埃及"
}


class Command(BaseCommand):
    help = '使用 TMDB API 智能补全电影数据 (含增强的地区映射)'

    def handle(self, *args, **options):
        dotenv.load_dotenv()
        api_key = os.getenv("TMDB_API_KEY")
        if not api_key:
            self.stderr.write("错误: TMDB_API_KEY 未设置。")
            return

        links_file = 'links.csv'
        if not os.path.exists(links_file):
            self.stderr.write(f"错误: {links_file} 未找到。")
            return

        self.stdout.write(f"--- 智能数据补全开始  ---")

        # 加载 links.csv 映射
        movielens_to_tmdb_map = {}
        with open(links_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if len(row) >= 3 and row[0] and row[2]:
                    movielens_to_tmdb_map[row[0]] = row[2]

        # 筛选需要更新的电影
        # (逻辑：ML电影 或 有ID但没演员的新电影)
        movies_to_update = Movie.objects.annotate(
            actor_count=Count('actors')
        ).filter(
            Q(movielens_id__isnull=False) |
            Q(imdb_id__isnull=False, actor_count=0)
        )

        total_count = movies_to_update.count()
        self.stdout.write(f"找到 {total_count} 部电影待处理。")

        updated_count = 0
        poster_base_url = "https://image.tmdb.org/t/p/w500"

        for i, movie_obj in enumerate(movies_to_update.iterator()):
            tmdb_id = None

            if movie_obj.movielens_id:
                tmdb_id = movielens_to_tmdb_map.get(movie_obj.movielens_id)
            else:
                tmdb_id = movie_obj.imdb_id  # (这就是我们在 import_new_movies 里存的)

            if not tmdb_id: continue

            # 请求 API
            api_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&language=zh-CN&append_to_response=credits"

            try:
                r = requests.get(api_url, timeout=10, proxies=proxies_dict)
                if r.status_code == 404: continue

                data = r.json()

                # 更新基础信息
                if data.get('overview'): movie_obj.summary = data.get('overview')
                if data.get('poster_path'): movie_obj.poster = poster_base_url + data.get('poster_path')
                if data.get('release_date'): movie_obj.date = data.get('release_date')
                if data.get('vote_average'): movie_obj.score = data.get('vote_average')
                if data.get('vote_count'): movie_obj.vote_count = data.get('vote_count')

                movie_obj.save()

                # --- 关键：更新地区 (使用增强字典) ---
                countries = data.get('production_countries', [])
                if countries:
                    region_objs = []
                    for c in countries:
                        raw_name = c.get('name')
                        # 使用增强字典进行映射
                        zh_name = REGION_NORMALIZE_MAP.get(raw_name, raw_name)
                        region, _ = Region.objects.get_or_create(name=zh_name)
                        region_objs.append(region)
                    movie_obj.regions.set(region_objs)

                # 更新演员
                credits = data.get('credits', {})
                cast = credits.get('cast', [])
                if cast:
                    actor_objs = []
                    for person in cast[:8]:  # 取前8
                        name = person.get('name')
                        if name:
                            actor, _ = Actor.objects.get_or_create(name=name)
                            actor_objs.append(actor)
                    movie_obj.actors.set(actor_objs)
                # 更新导演
                # 🔥 2. (新增) 更新导演 (Director)
                crew = credits.get('crew', [])
                if crew:
                    director_objs = []
                    # 过滤出所有 job 为 Director 的人 (有时一部电影有多个导演)
                    directors_list = [member for member in crew if member.get('job') == 'Director']
                    for d in directors_list:
                        d_name = d.get('name')
                        if d_name:
                            # 导演在模型中也是 Actor 类的实例
                            director, _ = Actor.objects.get_or_create(name=d_name)
                            director_objs.append(director)
                    # 写入 ManyToMany 字段
                    movie_obj.directors.set(director_objs)

                # 更新类型
                genres = data.get('genres', [])
                if genres:
                    genre_objs = []
                    for g in genres:
                        g_name = g.get('name')
                        if g_name:
                            genre, _ = Genre.objects.get_or_create(name=g_name)
                            genre_objs.append(genre)
                    movie_obj.genres.set(genre_objs)

                updated_count += 1
                if updated_count % 50 == 0:
                    self.stdout.write(f"进度: {updated_count}/{total_count} ... 《{movie_obj.title}》")

            except Exception:
                pass

            time.sleep(0.2)

        self.stdout.write(self.style.SUCCESS(f"--- 完成 ---"))
        self.stdout.write(self.style.SUCCESS(f"共丰富了 {updated_count} 部电影。"))
