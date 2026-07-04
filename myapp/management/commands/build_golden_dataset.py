"""
build_golden_dataset.py — 从数据库读取真实电影数据，构建黄金数据集
================================================
Django Management Command，基于 MovieLens-1M 数据库中的真实电影、
导演、演员、类型数据，自动构建论文 5.9 节所需的黄金数据集。

使用方式：
  # 生成完整数据集（默认300条）
  python manage.py build_golden_dataset

  # 指定输出路径和数量
  python manage.py build_golden_dataset --output=my_dataset.json --count=100

  # 快速测试（仅生成30条）
  python manage.py build_golden_dataset --count=30

输出格式：JSON 文件，每条查询包含：
  - id: 查询编号
  - difficulty: single_hop / multi_hop / implicit_semantic
  - query: 用户查询文本
  - ground_truth_movies: [电影标题列表]
  - ground_truth_ids: [电影ID列表]
  - reference_entities: [相关实体列表]
  - expected_tool_chain: [预期工具链]
================================================
"""

import json
import random
from django.core.management.base import BaseCommand
from myapp.models import Movie, Genre, Actor, Region


class Command(BaseCommand):
    help = '从数据库读取真实电影数据，构建黄金数据集'

    def add_arguments(self, parser):
        parser.add_argument(
            '--output',
            type=str,
            default='myapp/management/commands/golden_dataset_agent_eval.json',
            help='输出文件路径'
        )
        parser.add_argument(
            '--count',
            type=int,
            default=300,
            help='总查询数量（默认300，每种难度100条）'
        )
        parser.add_argument(
            '--seed',
            type=int,
            default=42,
            help='随机种子（保证可复现）'
        )

    def handle(self, *args, **options):
        random.seed(options['seed'])
        total = options['count']
        per_difficulty = total // 3
        output_path = options['output']

        self.stdout.write(self.style.SUCCESS(
            f"\n{'='*60}\n"
            f"  黄金数据集构建器\n"
            f"  目标: {total} 条查询（每种难度 {per_difficulty} 条）\n"
            f"  输出: {output_path}\n"
            f"{'='*60}\n"
        ))

        # ── 步骤 1: 加载数据库中的真实数据 ──
        self.stdout.write("  [1/5] 加载电影数据...")

        # 高热度电影（vote_count > 500 或 score >= 8.0）
        hot_movies = list(
            Movie.objects.filter(
                vote_count__gte=300,
                score__gte=7.0,
            ).prefetch_related('genres', 'actors', 'directors', 'regions')
            .order_by('-vote_count')[:500]
        )
        self.stdout.write(f"    高热度电影: {len(hot_movies)} 部")

        # 高分电影
        high_score_movies = list(
            Movie.objects.filter(
                score__gte=8.5,
            ).prefetch_related('genres', 'actors', 'directors', 'regions')
            .order_by('-score')[:200]
        )
        self.stdout.write(f"    高分电影: {len(high_score_movies)} 部")

        # 加载所有类型
        all_genres = list(Genre.objects.all())
        genre_names = [g.name for g in all_genres]
        self.stdout.write(f"    类型数量: {len(genre_names)}")

        # 加载有多个作品的导演
        directors_with_movies = self._get_directors_with_movies(min_movies=3)
        self.stdout.write(f"    多作品导演: {len(directors_with_movies)} 位")

        # 加载有多个作品的演员
        actors_with_movies = self._get_actors_with_movies(min_movies=3)
        self.stdout.write(f"    多作品演员: {len(actors_with_movies)} 位")

        # ── 步骤 2: 构建单跳查询 ──
        self.stdout.write("\n  [2/5] 构建单跳查询...")
        single_hop = self._build_single_hop_queries(
            hot_movies, high_score_movies, directors_with_movies,
            actors_with_movies, genre_names, per_difficulty
        )
        self.stdout.write(f"    生成: {len(single_hop)} 条")

        # ── 步骤 3: 构建多跳推理查询 ──
        self.stdout.write("\n  [3/5] 构建多跳推理查询...")
        multi_hop = self._build_multi_hop_queries(
            hot_movies, directors_with_movies, actors_with_movies,
            genre_names, per_difficulty
        )
        self.stdout.write(f"    生成: {len(multi_hop)} 条")

        # ── 步骤 4: 构建隐式语义查询 ──
        self.stdout.write("\n  [4/5] 构建隐式语义查询...")
        implicit = self._build_implicit_queries(
            hot_movies, high_score_movies, genre_names, per_difficulty
        )
        self.stdout.write(f"    生成: {len(implicit)} 条")

        # ── 步骤 5: 组装并保存 ──
        all_queries = single_hop + multi_hop + implicit
        multi_turn = self._build_multi_turn_sessions(genre_names)

        dataset = {
            "metadata": {
                "description": "MovieAgent 端到端评估黄金数据集（基于真实数据库）",
                "dataset": "MovieLens-1M",
                "total_movies_in_db": Movie.objects.count(),
                "total_queries": len(all_queries),
                "difficulty_levels": ["single_hop", "multi_hop", "implicit_semantic"],
                "random_seed": options['seed'],
                "version": "2.0"
            },
            "queries": all_queries,
            "multi_turn_sessions": multi_turn
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

        self.stdout.write(self.style.SUCCESS(
            f"\n  ✓ 已保存 {len(all_queries)} 条查询至 {output_path}\n"
            f"    单跳: {len(single_hop)} | 多跳: {len(multi_hop)} | 隐式: {len(implicit)}\n"
            f"    多轮会话: {len(multi_turn)} 组\n"
        ))

    # ============================================================
    # 单跳查询构建
    # ============================================================

    def _build_single_hop_queries(self, hot_movies, high_score_movies,
                                   directors_with_movies, actors_with_movies,
                                   genre_names, count):
        """构建单跳查询（不涉及多实体关系推理）"""
        queries = []
        query_id = 0

        # 类型 1: 按导演查询
        for director, movies in directors_with_movies[:min(20, count // 5)]:
            if len(queries) >= count:
                break
            query_id += 1
            movie_titles = [m.title for m in movies[:5]]
            movie_ids = [m.id for m in movies[:5]]
            queries.append({
                "id": f"SH-{query_id:03d}",
                "difficulty": "single_hop",
                "query": f"{director.name}导演的电影有哪些",
                "ground_truth_movies": movie_titles,
                "ground_truth_ids": movie_ids,
                "reference_entities": [director.name],
                "expected_tool_chain": ["kg_query", "maan_rerank", "rerank"]
            })

        # 类型 2: 按演员查询
        for actor, movies in actors_with_movies[:min(15, count // 6)]:
            if len(queries) >= count:
                break
            query_id += 1
            movie_titles = [m.title for m in movies[:5]]
            movie_ids = [m.id for m in movies[:5]]
            queries.append({
                "id": f"SH-{query_id:03d}",
                "difficulty": "single_hop",
                "query": f"推荐几部{actor.name}主演的电影",
                "ground_truth_movies": movie_titles,
                "ground_truth_ids": movie_ids,
                "reference_entities": [actor.name],
                "expected_tool_chain": ["recall_hybrid", "maan_rerank", "rerank"]
            })

        # 类型 3: 按类型查询
        popular_genres = self._get_popular_genres(hot_movies, genre_names)
        for genre in popular_genres[:min(15, count // 6)]:
            if len(queries) >= count:
                break
            query_id += 1
            genre_movies = [m for m in hot_movies if genre in [g.name for g in m.genres.all()]]
            random.shuffle(genre_movies)
            selected = genre_movies[:5]
            if not selected:
                continue
            queries.append({
                "id": f"SH-{query_id:03d}",
                "difficulty": "single_hop",
                "query": f"推荐几部好看的{genre}电影",
                "ground_truth_movies": [m.title for m in selected],
                "ground_truth_ids": [m.id for m in selected],
                "reference_entities": [genre],
                "expected_tool_chain": ["recall_hybrid", "maan_rerank", "rerank"]
            })

        # 类型 4: 按评分查询
        if len(queries) < count:
            query_id += 1
            top_movies = sorted(high_score_movies, key=lambda m: float(m.score or 0), reverse=True)[:5]
            if top_movies:
                queries.append({
                    "id": f"SH-{query_id:03d}",
                    "difficulty": "single_hop",
                    "query": "推荐评分最高的几部电影",
                    "ground_truth_movies": [m.title for m in top_movies],
                    "ground_truth_ids": [m.id for m in top_movies],
                    "reference_entities": ["高分"],
                    "expected_tool_chain": ["recall_hybrid", "maan_rerank", "rerank"]
                })

        # 类型 5: 按地区查询
        regions = ['美国', '日本', '韩国', '中国', '英国', '法国']
        for region_name in regions:
            if len(queries) >= count:
                break
            region_movies = [m for m in hot_movies
                           if region_name in [r.name for r in m.regions.all()]]
            if len(region_movies) < 3:
                continue
            random.shuffle(region_movies)
            selected = region_movies[:5]
            query_id += 1
            queries.append({
                "id": f"SH-{query_id:03d}",
                "difficulty": "single_hop",
                "query": f"推荐几部{region_name}电影",
                "ground_truth_movies": [m.title for m in selected],
                "ground_truth_ids": [m.id for m in selected],
                "reference_entities": [region_name],
                "expected_tool_chain": ["recall_hybrid", "maan_rerank", "rerank"]
            })

        random.shuffle(queries)
        return queries[:count]

    # ============================================================
    # 多跳推理查询构建
    # ============================================================

    def _build_multi_hop_queries(self, hot_movies, directors_with_movies,
                                  actors_with_movies, genre_names, count):
        """构建多跳推理查询（需要多实体关系遍历）"""
        queries = []
        query_id = 0

        # 类型 1: 同导演 + 类型过滤
        for director, movies in directors_with_movies[:min(20, count // 4)]:
            if len(queries) >= count:
                break
            # 获取该导演的主要类型
            all_genres_of_director = []
            for m in movies:
                all_genres_of_director.extend([g.name for g in m.genres.all()])
            if not all_genres_of_director:
                continue
            main_genre = max(set(all_genres_of_director), key=all_genres_of_director.count)

            # 选取一部锚点电影
            anchor = movies[0]
            # 同导演同类型的其他电影
            same_dir_genre = [m for m in movies[1:]
                            if main_genre in [g.name for g in m.genres.all()]]
            if not same_dir_genre:
                same_dir_genre = movies[1:4]

            query_id += 1
            queries.append({
                "id": f"MH-{query_id:03d}",
                "difficulty": "multi_hop",
                "query": f"推荐几部和《{anchor.title}》同导演的{main_genre}电影",
                "ground_truth_movies": [m.title for m in same_dir_genre[:5]],
                "ground_truth_ids": [m.id for m in same_dir_genre[:5]],
                "reference_entities": [anchor.title, director.name, main_genre],
                "expected_tool_chain": ["kg_query", "recall_hybrid", "maan_rerank", "rerank"]
            })

        # 类型 2: 同演员 + 类型过滤
        for actor, movies in actors_with_movies[:min(15, count // 5)]:
            if len(queries) >= count:
                break
            if len(movies) < 2:
                continue

            anchor = movies[0]
            other_movies = movies[1:]
            anchor_genres = set(g.name for g in anchor.genres.all())

            # 同演员且类型相关的电影
            related = [m for m in other_movies
                      if anchor_genres & set(g.name for g in m.genres.all())]
            if not related:
                related = other_movies[:3]

            # 选一个相关类型
            common_genre = list(anchor_genres)[0] if anchor_genres else "剧情"

            query_id += 1
            queries.append({
                "id": f"MH-{query_id:03d}",
                "difficulty": "multi_hop",
                "query": f"推荐和《{anchor.title}》同主演的{common_genre}片",
                "ground_truth_movies": [m.title for m in related[:5]],
                "ground_truth_ids": [m.id for m in related[:5]],
                "reference_entities": [anchor.title, actor.name, common_genre],
                "expected_tool_chain": ["kg_query", "recall_hybrid", "maan_rerank", "rerank"]
            })

        # 类型 3: 同类型 + 高分过滤
        popular_genres = self._get_popular_genres(hot_movies, genre_names)
        for genre in popular_genres[:min(10, count // 8)]:
            if len(queries) >= count:
                break
            genre_movies = sorted(
                [m for m in hot_movies if genre in [g.name for g in m.genres.all()]],
                key=lambda m: float(m.score or 0),
                reverse=True
            )
            if len(genre_movies) < 3:
                continue

            anchor = genre_movies[0]
            high_score_related = [m for m in genre_movies[1:]
                                 if float(m.score or 0) >= 8.0]

            if not high_score_related:
                high_score_related = genre_movies[1:4]

            query_id += 1
            queries.append({
                "id": f"MH-{query_id:03d}",
                "difficulty": "multi_hop",
                "query": f"推荐几部和《{anchor.title}》同类型的高分{genre}片",
                "ground_truth_movies": [m.title for m in high_score_related[:5]],
                "ground_truth_ids": [m.id for m in high_score_related[:5]],
                "reference_entities": [anchor.title, genre, "评分>8"],
                "expected_tool_chain": ["kg_query", "recall_hybrid", "maan_rerank", "rerank"]
            })

        # 类型 4: 双演员合作查询（从图谱中找共同出演电影）
        actor_pairs = self._find_actor_pairs(hot_movies, min_pairs=1)
        for actor1_name, actor2_name, movies in actor_pairs[:min(10, count // 8)]:
            if len(queries) >= count:
                break
            query_id += 1
            queries.append({
                "id": f"MH-{query_id:03d}",
                "difficulty": "multi_hop",
                "query": f"找一部{actor1_name}和{actor2_name}合作的电影",
                "ground_truth_movies": [m.title for m in movies[:3]],
                "ground_truth_ids": [m.id for m in movies[:3]],
                "reference_entities": [actor1_name, actor2_name],
                "expected_tool_chain": ["kg_query", "maan_rerank", "rerank", "explain"]
            })

        random.shuffle(queries)
        return queries[:count]

    # ============================================================
    # 隐式语义查询构建
    # ============================================================

    def _build_implicit_queries(self, hot_movies, high_score_movies,
                                genre_names, count):
        """构建隐式语义查询（依赖情感/氛围/风格等隐式语义）"""
        queries = []
        query_id = 0

        # 隐式查询模板（手工设计，映射到类型）
        implicit_templates = [
            {
                "query": "推荐几部让人看完觉得压抑，探讨人性的冷色调电影",
                "target_genres": ["科幻", "剧情", "悬疑", "惊悚"],
                "keywords": ["人性", "压抑"],
                "reference_entities": ["压抑", "人性", "冷色调"]
            },
            {
                "query": "想看那种烧脑但结局反转特别精彩的悬疑片",
                "target_genres": ["悬疑", "惊悚", "犯罪"],
                "keywords": ["烧脑", "反转"],
                "reference_entities": ["烧脑", "反转", "悬疑"]
            },
            {
                "query": "推荐几部温暖治愈、看完心情会变好的电影",
                "target_genres": ["喜剧", "爱情", "动画"],
                "keywords": ["温暖", "治愈"],
                "reference_entities": ["温暖", "治愈", "轻松"]
            },
            {
                "query": "有没有那种节奏很快、全程紧张刺激的动作片",
                "target_genres": ["动作", "冒险", "犯罪"],
                "keywords": ["紧张", "刺激"],
                "reference_entities": ["节奏快", "紧张刺激", "动作"]
            },
            {
                "query": "想看那种看完会思考人生的文艺片",
                "target_genres": ["剧情", "爱情", "科幻"],
                "keywords": ["思考人生", "文艺"],
                "reference_entities": ["思考人生", "文艺"]
            },
            {
                "query": "推荐几部画面唯美、视觉效果惊艳的电影",
                "target_genres": ["奇幻", "动画", "科幻", "冒险"],
                "keywords": ["唯美", "视觉"],
                "reference_entities": ["画面唯美", "视觉效果"]
            },
            {
                "query": "推荐那种看完会哭的感人至深的电影",
                "target_genres": ["剧情", "爱情", "战争"],
                "keywords": ["感人", "催泪"],
                "reference_entities": ["感人", "催泪", "情感"]
            },
            {
                "query": "有没有那种探讨科技与人类关系的科幻片",
                "target_genres": ["科幻", "剧情"],
                "keywords": ["科技", "人类"],
                "reference_entities": ["科技", "人类关系", "科幻"]
            },
            {
                "query": "推荐几部适合深夜一个人静静看的电影",
                "target_genres": ["剧情", "爱情", "文艺"],
                "keywords": ["安静", "深夜"],
                "reference_entities": ["深夜", "安静", "文艺"]
            },
            {
                "query": "想看那种充满想象力、天马行空的奇幻片",
                "target_genres": ["奇幻", "动画", "冒险"],
                "keywords": ["想象力", "奇幻"],
                "reference_entities": ["想象力", "奇幻", "天马行空"]
            },
            {
                "query": "推荐几部节奏紧凑的犯罪悬疑片",
                "target_genres": ["犯罪", "悬疑", "惊悚"],
                "keywords": ["紧凑", "犯罪"],
                "reference_entities": ["节奏紧凑", "犯罪", "悬疑"]
            },
            {
                "query": "有没有那种看完会沉默很久的深度剧情片",
                "target_genres": ["剧情", "战争", "历史"],
                "keywords": ["深度", "沉默"],
                "reference_entities": ["深度", "剧情"]
            },
            {
                "query": "推荐几部让人肾上腺素飙升的动作大片",
                "target_genres": ["动作", "冒险", "科幻"],
                "keywords": ["肾上腺素", "刺激"],
                "reference_entities": ["肾上腺素", "动作大片"]
            },
            {
                "query": "想看那种浪漫唯美的爱情电影",
                "target_genres": ["爱情", "剧情"],
                "keywords": ["浪漫", "唯美"],
                "reference_entities": ["浪漫", "唯美", "爱情"]
            },
            {
                "query": "推荐那种看完让人心情愉悦的喜剧片",
                "target_genres": ["喜剧"],
                "keywords": ["愉悦", "搞笑"],
                "reference_entities": ["愉悦", "喜剧"]
            },
            {
                "query": "有没有那种探讨生死哲学的深刻电影",
                "target_genres": ["剧情", "奇幻", "科幻"],
                "keywords": ["生死", "哲学"],
                "reference_entities": ["生死", "哲学", "深刻"]
            },
            {
                "query": "推荐几部紧张到不敢呼吸的惊悚片",
                "target_genres": ["惊悚", "悬疑", "恐怖"],
                "keywords": ["紧张", "惊悚"],
                "reference_entities": ["紧张", "惊悚"]
            },
            {
                "query": "想看那种史诗级宏大的战争片",
                "target_genres": ["战争", "历史", "剧情"],
                "keywords": ["史诗", "宏大"],
                "reference_entities": ["史诗", "宏大", "战争"]
            },
            {
                "query": "推荐那种充满童趣的动画电影",
                "target_genres": ["动画", "奇幻", "冒险"],
                "keywords": ["童趣", "动画"],
                "reference_entities": ["童趣", "动画"]
            },
            {
                "query": "有没有那种探讨社会现实的纪实风格电影",
                "target_genres": ["剧情", "犯罪", "历史"],
                "keywords": ["社会", "现实"],
                "reference_entities": ["社会现实", "纪实"]
            },
        ]

        for template in implicit_templates:
            if len(queries) >= count:
                break

            # 根据目标类型从数据库中筛选匹配电影
            target_genres = template["target_genres"]
            matched_movies = []

            for movie in hot_movies:
                movie_genres = [g.name for g in movie.genres.all()]
                # 至少匹配一个目标类型
                if any(tg in movie_genres for tg in target_genres):
                    # 优先选择高分电影
                    if float(movie.score or 0) >= 7.5:
                        matched_movies.append(movie)

            if len(matched_movies) < 3:
                # 放宽条件
                for movie in hot_movies:
                    movie_genres = [g.name for g in movie.genres.all()]
                    if any(tg in movie_genres for tg in target_genres):
                        if movie not in matched_movies:
                            matched_movies.append(movie)

            if not matched_movies:
                continue

            # 按评分排序，取Top5
            matched_movies.sort(key=lambda m: float(m.score or 0), reverse=True)
            selected = matched_movies[:5]

            query_id += 1
            queries.append({
                "id": f"IS-{query_id:03d}",
                "difficulty": "implicit_semantic",
                "query": template["query"],
                "ground_truth_movies": [m.title for m in selected],
                "ground_truth_ids": [m.id for m in selected],
                "reference_entities": template["reference_entities"],
                "expected_tool_chain": ["recall_hybrid", "maan_rerank", "rerank"]
            })

        # 补充：如果隐式查询不够，生成变体
        extra_genres = ["科幻", "动作", "喜剧", "爱情", "恐怖", "动画", "犯罪"]
        emotion_words = [
            ("紧张刺激", "动作"), ("温馨感人", "爱情"), ("悬疑烧脑", "悬疑"),
            ("轻松愉快", "喜剧"), ("震撼壮观", "科幻"), ("恐怖惊悚", "恐怖"),
            ("催泪感人", "剧情"), ("热血沸腾", "动作"), ("治愈暖心", "动画"),
        ]

        while len(queries) < count and emotion_words:
            emotion, genre = emotion_words.pop(0)
            genre_movies = [m for m in hot_movies
                          if genre in [g.name for g in m.genres.all()]
                          and float(m.score or 0) >= 7.5]
            if len(genre_movies) < 3:
                continue
            genre_movies.sort(key=lambda m: float(m.score or 0), reverse=True)
            selected = genre_movies[:5]
            query_id += 1
            queries.append({
                "id": f"IS-{query_id:03d}",
                "difficulty": "implicit_semantic",
                "query": f"推荐几部{emotion}的{genre}电影",
                "ground_truth_movies": [m.title for m in selected],
                "ground_truth_ids": [m.id for m in selected],
                "reference_entities": [emotion, genre],
                "expected_tool_chain": ["recall_hybrid", "maan_rerank", "rerank"]
            })

        random.shuffle(queries)
        return queries[:count]

    # ============================================================
    # 多轮对话会话构建
    # ============================================================

    def _build_multi_turn_sessions(self, genre_names):
        """构建多轮对话测试会话"""
        sessions = []

        # 会话1: 类型→年份→评分
        genre1 = random.choice(["科幻", "动作", "悬疑", "喜剧"])
        sessions.append({
            "session_id": "MT-001",
            "turns": [
                {"role": "user", "content": f"想看{genre1}片",
                 "expected_slots": {"genre": genre1}},
                {"role": "user", "content": "不要太老的",
                 "expected_slots": {"genre": genre1, "year_min": 2015}},
                {"role": "user", "content": "最好评分高一点",
                 "expected_slots": {"genre": genre1, "year_min": 2015, "score_min": 8.0}}
            ]
        })

        # 会话2: 类型→排除→年份
        genre2 = random.choice(["恐怖", "战争", "犯罪", "剧情"])
        sessions.append({
            "session_id": "MT-002",
            "turns": [
                {"role": "user", "content": f"推荐{genre2}推理片",
                 "expected_slots": {"genre": genre2}},
                {"role": "user", "content": "不要国产的",
                 "expected_slots": {"genre": genre2, "exclude_country": "中国"}},
                {"role": "user", "content": "最好是近五年的",
                 "expected_slots": {"genre": genre2, "exclude_country": "中国", "year_min": 2021}}
            ]
        })

        # 会话3: 模糊→具体→个性化
        sessions.append({
            "session_id": "MT-003",
            "turns": [
                {"role": "user", "content": "推荐一部好看的电影",
                 "expected_slots": {}},
                {"role": "user", "content": "想要轻松一点的喜剧",
                 "expected_slots": {"genre": "喜剧", "keyword": "轻松"}},
                {"role": "user", "content": "最好是周星驰的",
                 "expected_slots": {"genre": "喜剧", "keyword": "轻松", "actor": "周星驰"}}
            ]
        })

        return sessions

    # ============================================================
    # 辅助方法
    # ============================================================

    def _get_directors_with_movies(self, min_movies=3):
        """获取有多部作品的导演及其电影列表"""
        directors = {}
        movies = Movie.objects.filter(
            vote_count__gte=100
        ).prefetch_related('directors', 'genres').order_by('-vote_count')[:300]

        for movie in movies:
            for director in movie.directors.all():
                if director.name not in directors:
                    directors[director.name] = {'director': director, 'movies': []}
                directors[director.name]['movies'].append(movie)

        result = []
        for name, data in directors.items():
            if len(data['movies']) >= min_movies:
                result.append((data['director'], data['movies']))

        # 按作品数量排序
        result.sort(key=lambda x: len(x[1]), reverse=True)
        return result

    def _get_actors_with_movies(self, min_movies=3):
        """获取有多部作品的演员及其电影列表"""
        actors = {}
        movies = Movie.objects.filter(
            vote_count__gte=100
        ).prefetch_related('actors', 'genres').order_by('-vote_count')[:300]

        for movie in movies:
            for actor in movie.actors.all()[:5]:  # 取前5位主演
                if actor.name not in actors:
                    actors[actor.name] = {'actor': actor, 'movies': []}
                actors[actor.name]['movies'].append(movie)

        result = []
        for name, data in actors.items():
            if len(data['movies']) >= min_movies:
                result.append((data['actor'], data['movies']))

        result.sort(key=lambda x: len(x[1]), reverse=True)
        return result

    def _get_popular_genres(self, movies, all_genre_names):
        """统计电影列表中最常见的类型"""
        genre_count = {}
        for movie in movies:
            for g in movie.genres.all():
                genre_count[g.name] = genre_count.get(g.name, 0) + 1

        # 排序并过滤掉过于宽泛的类型
        skip = {'Drama', 'Unknown', 'Documentary', 'Film-Noir'}
        sorted_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)
        return [g for g, c in sorted_genres if g not in skip]

    def _find_actor_pairs(self, movies, min_pairs=1):
        """从电影列表中找出共同出演的演员对"""
        pairs = {}  # (actor1, actor2) -> [movies]

        for movie in movies:
            actor_list = list(movie.actors.all()[:5])
            for i in range(len(actor_list)):
                for j in range(i + 1, len(actor_list)):
                    key = tuple(sorted([actor_list[i].name, actor_list[j].name]))
                    if key not in pairs:
                        pairs[key] = []
                    pairs[key].append(movie)

        result = []
        for (a1, a2), movie_list in pairs.items():
            if len(movie_list) >= min_pairs:
                result.append((a1, a2, movie_list))

        result.sort(key=lambda x: len(x[2]), reverse=True)
        return result