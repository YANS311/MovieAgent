# 文件: myapp/management/commands/import_new_movies.py (V3 - 终极修正版)

import os
import time
import requests
import dotenv
from django.core.management.base import BaseCommand
from myapp.models import Movie, Genre, Actor, Region
from django.db import transaction

# --- 1. 代理设置 (防止连接失败) ---
_http_proxy = os.environ.get('HTTP_PROXY', '')
_https_proxy = os.environ.get('HTTPS_PROXY', '')
proxies_dict = {}
if _http_proxy:
    proxies_dict['http'] = _http_proxy
if _https_proxy:
    proxies_dict['https'] = _https_proxy

# --- 2. 地区规范化字典 ---
REGION_NORMALIZE_MAP = {
    "United States of America": "美国", "United Kingdom": "英国", "China": "中国大陆",
    "Hong Kong": "中国香港", "Japan": "日本", "South Korea": "韩国",
    "France": "法国", "Germany": "德国", "Italy": "意大利", "Spain": "西班牙",
    "India": "印度", "Canada": "加拿大", "Australia": "澳大利亚", "Russia": "俄罗斯",
    "Taiwan": "中国台湾"
}


class Command(BaseCommand):
    help = '使用 TMDB "discover" API 爬取 (1949-2025) 电影，并自动关联演员/地区'

    def handle(self, *args, **options):

        # --- 3. 加载配置 ---
        dotenv.load_dotenv()
        api_key = os.getenv("TMDB_API_KEY")
        if not api_key:
            self.stderr.write(self.style.ERROR("错误: TMDB_API_KEY 未在 .env 文件中设置。"))
            return

        # --- 4. 设定目标年份和数量 ---
        # (可以根据需要调整范围，比如只爬最近几年的)
        years_to_fetch = [i for i in range(1949, 2027)]


        movies_per_year = 80  # 每年爬取的电影数量上限
        poster_base_url = "https://image.tmdb.org/t/p/w500"

        created_count = 0
        updated_count = 0

        self.stdout.write(f"--- 开始爬取 {len(years_to_fetch) * movies_per_year} 部电影 ---")

        for year in years_to_fetch:
            self.stdout.write(f"--- 正在处理 {year} 年的电影 ---")

            # TMDB 每页 20 条
            max_page = (movies_per_year // 20) + 1

            for page in range(1, max_page):

                # --- 5. 调用 Discover API (含 credits) ---
                # 注意：discover 接口不支持 append_to_response，所以这里只拿基础信息
                # 如果需要演员，我们在循环里再单独请求一次详情（为了效率，这里先只存基础，或者看下面的策略）

                # *修正策略*: Discover 接口拿不到演员。为了数据完整，我们先 Discover 拿到 ID，
                # 然后再请求一次 Movie Details 接口拿演员。虽然慢一点，但数据是一步到位的。

                discover_url = (
                    f"https://api.themoviedb.org/3/discover/movie?"
                    f"api_key={api_key}"
                    f"&language=zh-CN"
                    f"&primary_release_year={year}"
                    f"&sort_by=popularity.desc"
                    f"&page={page}"
                )

                try:
                    r = requests.get(discover_url, timeout=10, proxies=proxies_dict)
                    r.raise_for_status()
                    data = r.json()

                    if not data.get('results'):
                        break

                        # --- 6. 遍历结果 ---
                    for movie_dict in data.get('results', []):
                        tmdb_id = movie_dict.get('id')
                        if not tmdb_id: continue

                        # 为了获取演员和地区，我们需要再请求一次详情接口
                        # (如果你觉得太慢，可以注释掉这一段，只用 discover 的数据，但那样就没有演员了)
                        detail_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?api_key={api_key}&language=zh-CN&append_to_response=credits"

                        try:
                            r_detail = requests.get(detail_url, timeout=10, proxies=proxies_dict)
                            if r_detail.status_code == 200:
                                movie_dict = r_detail.json()  # 用详情数据覆盖
                        except:
                            pass  # 如果详情请求失败，就用 discover 的基础数据兜底

                        # --- 提取数据 ---
                        tmdb_id_str = str(tmdb_id)
                        title = movie_dict.get('title')
                        summary = movie_dict.get('overview')
                        poster_path = movie_dict.get('poster_path')

                        if not title or not summary:
                            continue

                        with transaction.atomic():
                            # --- 7. (核心修正) 存入 imdb_id ---
                            # 逻辑：如果 movielens_id 也没有，这就纯粹是 TMDB 新片
                            movie_obj, created = Movie.objects.get_or_create(
                                imdb_id=tmdb_id_str,  # <--- 这里存 TMDB ID
                                defaults={
                                    'title': title,
                                    'movielens_id': None,  # 明确标记无 ML ID
                                }
                            )

                            # --- 8. 更新/填充数据 ---
                            movie_obj.title = title
                            movie_obj.summary = summary
                            if poster_path:
                                movie_obj.poster = poster_base_url + poster_path
                            movie_obj.date = movie_dict.get('release_date') or None
                            movie_obj.score = movie_dict.get('vote_average')
                            movie_obj.vote_count = movie_dict.get('vote_count')

                            movie_obj.save()

                            # --- 9. 保存 M2M (演员/地区/类型) ---

                            # 地区
                            countries = movie_dict.get('production_countries', [])
                            if countries:
                                region_objs = []
                                for c in countries:
                                    zh_name = REGION_NORMALIZE_MAP.get(c.get('name'), c.get('name'))
                                    region, _ = Region.objects.get_or_create(name=zh_name)
                                    region_objs.append(region)
                                movie_obj.regions.set(region_objs)

                            # 演员
                            credits = movie_dict.get('credits', {})
                            cast = credits.get('cast', [])
                            if cast:
                                actor_objs = []
                                for person in cast[:8]:  # 取前8
                                    name = person.get('name')
                                    if name:
                                        actor, _ = Actor.objects.get_or_create(name=name)
                                        actor_objs.append(actor)
                                movie_obj.actors.set(actor_objs)

                            # --- [新增] 处理导演 (Director) ---
                            # 导演在 'crew' 列表中，且 job 为 'Director'
                            crew = credits.get('crew', [])
                            if crew:
                                director_objs = []
                                for member in crew:
                                    if member.get('job') == 'Director':
                                        d_name = member.get('name')
                                        if d_name:
                                            # 复用 Actor 表来存储导演 (或者你新建的 Director 表)
                                            d_obj, _ = Actor.objects.get_or_create(name=d_name)
                                            director_objs.append(d_obj)

                                # 将提取到的导演列表关联到电影
                                # 注意：前提是你的 Movie 模型里已经有了 directors 字段
                                movie_obj.directors.set(director_objs)

                            # 类型
                            genres = movie_dict.get('genres', [])
                            if genres:
                                genre_objs = []
                                for g in genres:
                                    if g.get('name'):
                                        genre, _ = Genre.objects.get_or_create(name=g.get('name'))
                                        genre_objs.append(genre)
                                movie_obj.genres.set(genre_objs)

                            if created:
                                created_count += 1
                                if created_count % 10 == 0:
                                    self.stdout.write(f"已新增: {title}")
                            else:
                                updated_count += 1

                        # 速率限制 (因为加了详情请求，要慢一点)
                        time.sleep(0.1)

                except requests.exceptions.RequestException as e:
                    self.stderr.write(f"爬取错误: {e}")
                    time.sleep(2)  # 出错休息久一点

        self.stdout.write(self.style.SUCCESS(f"--- 爬取完成 ---"))
        self.stdout.write(self.style.SUCCESS(f"新增: {created_count}, 更新: {updated_count}"))