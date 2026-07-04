"""
重建黄金数据集V3 - 修复隐式语义查询的GT
核心改进：使用数据库中实际存在的电影，按类型正确匹配GT
"""
import json, random
from django.core.management.base import BaseCommand
from myapp.models import Movie, Genre, Actor


class Command(BaseCommand):
    help = '重建黄金数据集V3（修复隐式语义GT）'

    def handle(self, *args, **options):
        random.seed(42)
        
        # 获取类型到电影的映射
        genre_movies = {}
        for genre in Genre.objects.all():
            movies = list(Movie.objects.filter(genres=genre).order_by('-score', '-vote_count')[:20])
            genre_movies[genre.name] = [{'id': m.id, 'title': m.title, 'score': float(m.score or 0)} for m in movies]
        
        # 获取导演到电影的映射
        director_movies = {}
        for m in Movie.objects.prefetch_related('directors').all()[:3000]:
            for d in m.directors.all():
                if d.name not in director_movies:
                    director_movies[d.name] = []
                if len(director_movies[d.name]) < 10:
                    director_movies[d.name].append({'id': m.id, 'title': m.title})
        
        # 获取演员到电影的映射
        actor_movies = {}
        for a in Actor.objects.all()[:500]:
            movies = list(Movie.objects.filter(actors=a).order_by('-score')[:10])
            if movies:
                actor_movies[a.name] = [{'id': m.id, 'title': m.title} for m in movies]
        
        queries = []
        qid = 0
        
        # ── 单跳查询：导演查询 ──
        top_directors = [
            ('克里斯托弗·诺兰', '克里斯托弗·诺兰'),
            ('史蒂文·斯皮尔伯格', '史蒂文·斯皮尔伯格'),
            ('昆汀·塔伦蒂诺', '昆汀·塔伦蒂诺'),
            ('大卫·芬奇', '大卫·芬奇'),
            ('詹姆斯·卡梅隆', '詹姆斯·卡梅隆'),
            ('雷德利·斯科特', '雷德利·斯科特'),
        ]
        for dname, ref in top_directors:
            if dname in director_movies:
                dms = director_movies[dname][:5]
                qid += 1
                queries.append({
                    "id": f"SH-{qid:03d}",
                    "difficulty": "single_hop",
                    "query": f"{dname}导演的电影有哪些",
                    "ground_truth_movies": [m['title'] for m in dms],
                    "ground_truth_ids": [m['id'] for m in dms],
                    "reference_entities": [ref],
                    "expected_tool_chain": ["kg_query", "maan_rerank", "rerank"],
                })
        
        # ── 单跳查询：类型查询 ──
        genre_queries = [
            ('科幻', '推荐几部好看的科幻电影'),
            ('动作', '推荐几部好看的动作电影'),
            ('喜剧', '推荐几部好看的喜剧电影'),
            ('恐怖', '推荐几部好看的恐怖电影'),
            ('动画', '推荐几部好看的动画电影'),
            ('犯罪', '推荐几部好看的犯罪电影'),
            ('悬疑', '推荐几部好看的悬疑电影'),
            ('爱情', '推荐几部好看的爱情电影'),
        ]
        for genre_name, query_text in genre_queries:
            if genre_name in genre_movies and genre_movies[genre_name]:
                gms = genre_movies[genre_name][:5]
                qid += 1
                queries.append({
                    "id": f"SH-{qid:03d}",
                    "difficulty": "single_hop",
                    "query": query_text,
                    "ground_truth_movies": [m['title'] for m in gms],
                    "ground_truth_ids": [m['id'] for m in gms],
                    "reference_entities": [genre_name],
                    "expected_tool_chain": ["recall_hybrid", "maan_rerank", "rerank"],
                })
        
        # ── 多跳查询：同导演 ──
        multi_hop_directors = [
            ('星际穿越', '克里斯托弗·诺兰', '剧情'),
            ('Fight Club', '大卫·芬奇', '剧情'),
            ('Pulp Fiction', '昆汀·塔伦蒂诺', '剧情'),
            ('阿凡达', '詹姆斯·卡梅隆', '动作'),
            ('指环王1：护戒使者', '彼得·杰克逊', '奇幻'),
        ]
        for anchor, dname, genre in multi_hop_directors:
            if dname in director_movies:
                dms = [m for m in director_movies[dname] if m['title'] != anchor][:5]
                if dms:
                    qid += 1
                    queries.append({
                        "id": f"MH-{qid:03d}",
                        "difficulty": "multi_hop",
                        "query": f"推荐几部和《{anchor}》同导演的{genre}电影",
                        "ground_truth_movies": [m['title'] for m in dms],
                        "ground_truth_ids": [m['id'] for m in dms],
                        "reference_entities": [anchor, dname, genre],
                        "expected_tool_chain": ["kg_query", "recall_hybrid", "maan_rerank", "rerank"],
                    })
        
        # ── 多跳查询：合作演员 ──
        collab_pairs = [
            ('斯嘉丽·约翰逊', '马克·鲁法洛'),
            ('汤姆·汉克斯', '蒂姆·艾伦'),
            ('克里斯蒂安·贝尔', '迈克尔·凯恩'),
        ]
        for a1, a2 in collab_pairs:
            if a1 in actor_movies and a2 in actor_movies:
                s1 = {m['id'] for m in actor_movies[a1]}
                s2 = {m['id'] for m in actor_movies[a2]}
                common = list(s1 & s2)
                if common:
                    common_movies = [m for m in actor_movies[a1] if m['id'] in common][:3]
                    qid += 1
                    queries.append({
                        "id": f"MH-{qid:03d}",
                        "difficulty": "multi_hop",
                        "query": f"找一部{a1}和{a2}合作的电影",
                        "ground_truth_movies": [m['title'] for m in common_movies],
                        "ground_truth_ids": [m['id'] for m in common_movies],
                        "reference_entities": [a1, a2],
                        "expected_tool_chain": ["kg_query", "maan_rerank", "rerank"],
                    })
        
        # ── 隐式语义查询（关键修复：GT与查询类型匹配）──
        implicit_queries = [
            ("推荐几部温馨感人的爱情电影", "爱情", ["Heartwarming", "温馨"]),
            ("推荐几部震撼壮观的科幻电影", "科幻", ["史诗", "壮观"]),
            ("推荐那种充满童趣的动画电影", "动画", ["童趣"]),
            ("推荐几部紧张刺激的动作电影", "动作", ["紧张", "刺激"]),
            ("推荐几部催泪感人的剧情电影", "剧情", ["催泪", "感人"]),
            ("推荐几部温暖治愈、看完心情会变好的电影", "喜剧", ["治愈", "温暖"]),
            ("推荐几部紧张到不敢呼吸的惊悚片", "惊悚", ["紧张", "惊悚"]),
            ("推荐几部悬疑烧脑的悬疑电影", "悬疑", ["烧脑", "悬疑"]),
            ("推荐几部治愈暖心的动画电影", "动画", ["治愈", "暖心"]),
            ("推荐几部热血沸腾的动作电影", "动作", ["热血", "动作"]),
            ("推荐几部轻松愉快的喜剧电影", "喜剧", ["轻松", "愉快"]),
            ("想看那种浪漫唯美的爱情电影", "爱情", ["浪漫", "唯美"]),
            ("推荐几部好看的冒险电影", "冒险", ["冒险"]),
            ("推荐几部好看的家庭电影", "家庭", ["家庭"]),
            ("推荐几部好看的奇幻电影", "奇幻", ["奇幻"]),
            ("推荐几部好看的恐怖电影", "恐怖", ["恐怖"]),
        ]
        for query_text, genre_name, keywords in implicit_queries:
            if genre_name in genre_movies and genre_movies[genre_name]:
                gms = genre_movies[genre_name][:5]
                qid += 1
                queries.append({
                    "id": f"IS-{qid:03d}",
                    "difficulty": "implicit_semantic",
                    "query": query_text,
                    "ground_truth_movies": [m['title'] for m in gms],
                    "ground_truth_ids": [m['id'] for m in gms],
                    "reference_entities": keywords,
                    "expected_tool_chain": ["recall_hybrid", "maan_rerank", "rerank"],
                })
        
        # 多轮对话测试集
        multi_turn_sessions = [
            {
                "session_id": "MT-001",
                "turns": [
                    {"role": "user", "content": "想看动作片", "expected_slots": {"genre": "动作"}},
                    {"role": "user", "content": "不要太老的", "expected_slots": {"genre": "动作", "year_min": 2015}},
                    {"role": "user", "content": "最好评分高一点", "expected_slots": {"genre": "动作", "year_min": 2015, "score_min": 8.0}},
                ]
            },
            {
                "session_id": "MT-002",
                "turns": [
                    {"role": "user", "content": "推荐犯罪推理片", "expected_slots": {"genre": "犯罪"}},
                    {"role": "user", "content": "不要国产的", "expected_slots": {"genre": "犯罪", "exclude_country": "中国"}},
                    {"role": "user", "content": "最好是近五年的", "expected_slots": {"genre": "犯罪", "exclude_country": "中国", "year_min": 2021}},
                ]
            },
            {
                "session_id": "MT-003",
                "turns": [
                    {"role": "user", "content": "推荐一部好看的电影", "expected_slots": {}},
                    {"role": "user", "content": "想要轻松一点的喜剧", "expected_slots": {"genre": "喜剧", "keyword": "轻松"}},
                    {"role": "user", "content": "最好是周星驰的", "expected_slots": {"genre": "喜剧", "keyword": "轻松", "actor": "周星驰"}},
                ]
            },
        ]
        
        dataset = {
            "metadata": {
                "description": "MovieAgent 端到端评估黄金数据集 V3（修复隐式语义GT）",
                "dataset": "MovieLens-1M",
                "total_movies_in_db": Movie.objects.count(),
                "total_queries": len(queries),
                "difficulty_levels": ["single_hop", "multi_hop", "implicit_semantic"],
                "random_seed": 42,
                "version": "3.0"
            },
            "queries": queries,
            "multi_turn_sessions": multi_turn_sessions,
        }
        
        out_path = "myapp/management/commands/golden_dataset_agent_eval.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        self.stdout.write(f"✅ 黄金数据集V3已生成: {len(queries)} 条查询")
        
        # 统计
        for diff in ['single_hop', 'multi_hop', 'implicit_semantic']:
            count = len([q for q in queries if q['difficulty'] == diff])
            self.stdout.write(f"  {diff}: {count} 条")