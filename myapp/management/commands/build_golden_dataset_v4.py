"""
重建黄金数据集V4 - 扩展到100+条，统一三类查询比例
目标：single_hop ~40, multi_hop ~30, implicit_semantic ~40
"""
import json, random
from django.core.management.base import BaseCommand
from myapp.models import Movie, Genre, Actor


class Command(BaseCommand):
    help = '重建黄金数据集V4（100+条，三类均衡）'

    def handle(self, *args, **options):
        random.seed(42)
        
        # 获取类型到电影的映射
        genre_movies = {}
        for genre in Genre.objects.all():
            movies = list(Movie.objects.filter(genres=genre).order_by('-score', '-vote_count')[:25])
            genre_movies[genre.name] = [{'id': m.id, 'title': m.title, 'score': float(m.score or 0)} for m in movies]
        
        # 获取导演到电影的映射
        director_movies = {}
        for m in Movie.objects.prefetch_related('directors').all()[:5000]:
            for d in m.directors.all():
                if d.name not in director_movies:
                    director_movies[d.name] = []
                if len(director_movies[d.name]) < 12:
                    director_movies[d.name].append({'id': m.id, 'title': m.title, 'score': float(m.score or 0)})
        
        # 获取演员到电影的映射
        actor_movies = {}
        for a in Actor.objects.all()[:800]:
            movies = list(Movie.objects.filter(actors=a).order_by('-score')[:12])
            if movies:
                actor_movies[a.name] = [{'id': m.id, 'title': m.title} for m in movies]
        
        queries = []
        qid = 0
        
        # ═══════════════════════════════════════
        # 单跳查询 (~40条)
        # ═══════════════════════════════════════
        
        # 导演查询 (10条)
        all_directors = [
            '克里斯托弗·诺兰', '史蒂文·斯皮尔伯格', '昆汀·塔伦蒂诺',
            '大卫·芬奇', '詹姆斯·卡梅隆', '雷德利·斯科特',
            '马丁·斯科塞斯', '宫崎骏', '王家卫', '周星驰',
            '迈克尔·贝', '彼得·杰克逊', '盖·里奇', '丹尼斯·维伦纽瓦',
        ]
        for dname in all_directors:
            if dname in director_movies and len(director_movies[dname]) >= 3:
                dms = director_movies[dname][:5]
                qid += 1
                queries.append({
                    "id": f"SH-{qid:03d}",
                    "difficulty": "single_hop",
                    "query": f"{dname}导演的电影有哪些",
                    "ground_truth_movies": [m['title'] for m in dms],
                    "ground_truth_ids": [m['id'] for m in dms],
                    "reference_entities": [dname],
                    "expected_tool_chain": ["kg_query", "maan_rerank", "rerank"],
                })
        
        # 类型查询 (12条)
        genre_query_templates = [
            ('科幻', '推荐几部好看的科幻电影'),
            ('动作', '推荐几部好看的动作电影'),
            ('喜剧', '推荐几部好看的喜剧电影'),
            ('恐怖', '推荐几部好看的恐怖电影'),
            ('动画', '推荐几部好看的动画电影'),
            ('犯罪', '推荐几部好看的犯罪电影'),
            ('悬疑', '推荐几部好看的悬疑电影'),
            ('爱情', '推荐几部好看的爱情电影'),
            ('战争', '推荐几部好看的战争电影'),
            ('奇幻', '推荐几部好看的奇幻电影'),
            ('冒险', '推荐几部好看的冒险电影'),
            ('惊悚', '推荐几部好看的惊悚电影'),
        ]
        for genre_name, query_text in genre_query_templates:
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
        
        # 演员查询 (8条)
        actor_queries = [
            ('莱昂纳多·迪卡普里奥', '莱昂纳多主演的电影'),
            ('汤姆·汉克斯', '汤姆·汉克斯主演的电影'),
            ('斯嘉丽·约翰逊', '斯嘉丽·约翰逊主演的电影'),
            ('周星驰', '周星驰主演的电影'),
            ('布拉德·皮特', '布拉德·皮特主演的电影'),
            ('摩根·弗里曼', '摩根·弗里曼主演的电影'),
            ('约翰尼·德普', '约翰尼·德普主演的电影'),
            ('休·杰克曼', '休·杰克曼主演的电影'),
        ]
        for aname, qtext in actor_queries:
            if aname in actor_movies:
                ams = actor_movies[aname][:5]
                qid += 1
                queries.append({
                    "id": f"SH-{qid:03d}",
                    "difficulty": "single_hop",
                    "query": qtext,
                    "ground_truth_movies": [m['title'] for m in ams],
                    "ground_truth_ids": [m['id'] for m in ams],
                    "reference_entities": [aname],
                    "expected_tool_chain": ["kg_query", "maan_rerank", "rerank"],
                })
        
        # 高分/热门查询 (5条)
        for genre in ['科幻', '喜剧', '动作', '剧情', '动画']:
            if genre in genre_movies:
                high_score = sorted(genre_movies[genre], key=lambda x: x['score'], reverse=True)[:5]
                qid += 1
                queries.append({
                    "id": f"SH-{qid:03d}",
                    "difficulty": "single_hop",
                    "query": f"推荐几部高分{genre}电影",
                    "ground_truth_movies": [m['title'] for m in high_score],
                    "ground_truth_ids": [m['id'] for m in high_score],
                    "reference_entities": [genre, '高分'],
                    "expected_tool_chain": ["recall_hybrid", "maan_rerank", "rerank"],
                })
        
        # ═══════════════════════════════════════
        # 多跳查询 (~30条)
        # ═══════════════════════════════════════
        
        # 同导演多跳 (10条)
        multi_hop_directors = [
            ('星际穿越', '克里斯托弗·诺兰', '剧情'),
            ('Fight Club', '大卫·芬奇', '剧情'),
            ('Pulp Fiction', '昆汀·塔伦蒂诺', '剧情'),
            ('阿凡达', '詹姆斯·卡梅隆', '动作'),
            ('指环王1：护戒使者', '彼得·杰克逊', '奇幻'),
            ('盗梦空间', '克里斯托弗·诺兰', '科幻'),
            ('泰坦尼克号', '詹姆斯·卡梅隆', '爱情'),
            ('沉默的羔羊', '乔纳森·戴米', '惊悚'),
            ('低俗小说', '昆汀·塔伦蒂诺', '犯罪'),
            ('辛德勒的名单', '史蒂文·斯皮尔伯格', '剧情'),
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
        
        # 合作演员多跳 (8条)
        collab_pairs = [
            ('斯嘉丽·约翰逊', '马克·鲁法洛'),
            ('汤姆·汉克斯', '蒂姆·艾伦'),
            ('克里斯蒂安·贝尔', '迈克尔·凯恩'),
            ('莱昂纳多·迪卡普里奥', '汤姆·汉克斯'),
            ('布拉德·皮特', '摩根·弗里曼'),
            ('约翰尼·德普', '海伦娜·伯翰·卡特'),
            ('休·杰克曼', '帕特里克·斯图尔特'),
            ('小罗伯特·唐尼', '格温妮斯·帕特洛'),
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
        
        # 同类型+高分多跳 (8条)
        type_high_score = [
            ('星际穿越', '科幻', 8.0),
            ('肖申克的救赎', '剧情', 9.0),
            ('盗梦空间', '科幻', 8.5),
            ('蝙蝠侠：黑暗骑士', '动作', 8.5),
            ('阿甘正传', '剧情', 8.5),
            ('飞屋环游记', '动画', 8.5),
            ('教父', '犯罪', 9.0),
            ('沉默的羔羊', '惊悚', 8.5),
        ]
        for anchor, genre, min_score in type_high_score:
            if genre in genre_movies:
                # 从同类型高分电影中排除锚点
                same_genre = [m for m in genre_movies[genre] if m['title'] != anchor and m['score'] >= min_score][:5]
                if len(same_genre) >= 2:
                    qid += 1
                    queries.append({
                        "id": f"MH-{qid:03d}",
                        "difficulty": "multi_hop",
                        "query": f"推荐几部和《{anchor}》类似的高分{genre}电影",
                        "ground_truth_movies": [m['title'] for m in same_genre],
                        "ground_truth_ids": [m['id'] for m in same_genre],
                        "reference_entities": [anchor, genre, f'评分>={min_score}'],
                        "expected_tool_chain": ["kg_query", "recall_hybrid", "maan_rerank", "rerank"],
                    })
        
        # ═══════════════════════════════════════
        # 隐式语义查询 (~40条)
        # ═══════════════════════════════════════
        
        implicit_queries = [
            # 情感基调类
            ("推荐几部温馨感人的爱情电影", "爱情", ["温馨", "感人"]),
            ("推荐几部催泪感人的剧情电影", "剧情", ["催泪", "感人"]),
            ("推荐几部温暖治愈、看完心情会变好的电影", "喜剧", ["治愈", "温暖"]),
            ("想看那种浪漫唯美的爱情电影", "爱情", ["浪漫", "唯美"]),
            ("推荐几部轻松愉快的喜剧电影", "喜剧", ["轻松", "愉快"]),
            ("推荐几部让人热血沸腾的电影", "动作", ["热血", "沸腾"]),
            ("想看一部看完会觉得人生美好的电影", "剧情", ["人生", "美好"]),
            ("推荐几部让人心情愉悦的动画电影", "动画", ["愉悦", "心情"]),
            # 紧张/刺激类
            ("推荐几部震撼壮观的科幻电影", "科幻", ["震撼", "壮观"]),
            ("推荐几部紧张刺激的动作电影", "动作", ["紧张", "刺激"]),
            ("推荐几部紧张到不敢呼吸的惊悚片", "惊悚", ["紧张", "惊悚"]),
            ("推荐几部悬疑烧脑的悬疑电影", "悬疑", ["烧脑", "悬疑"]),
            ("推荐几部充满悬念的犯罪电影", "犯罪", ["悬念", "犯罪"]),
            ("推荐几部让人屏息凝神的谍战电影", "悬疑", ["屏息", "谍战"]),
            # 视觉/风格类
            ("推荐那种充满童趣的动画电影", "动画", ["童趣"]),
            ("推荐几部视觉效果震撼的科幻片", "科幻", ["视觉", "震撼"]),
            ("推荐几部画面唯美的爱情电影", "爱情", ["画面", "唯美"]),
            ("推荐几部很有文艺气息的电影", "剧情", ["文艺", "气息"]),
            ("推荐几部风格独特的犯罪片", "犯罪", ["风格", "独特"]),
            # 主题/思考类
            ("推荐几部探讨人性的深度电影", "剧情", ["人性", "深度"]),
            ("推荐几部关于时间旅行的电影", "科幻", ["时间旅行"]),
            ("推荐几部让人反思社会的电影", "剧情", ["反思", "社会"]),
            ("推荐几部关于人工智能的电影", "科幻", ["人工智能"]),
            ("推荐几部讲述友情的感人电影", "剧情", ["友情", "感人"]),
            ("推荐几部关于家庭温情的电影", "家庭", ["家庭", "温情"]),
            # 混合修饰类
            ("推荐几部治愈暖心的动画电影", "动画", ["治愈", "暖心"]),
            ("推荐几部热血沸腾的动作电影", "动作", ["热血", "动作"]),
            ("推荐几部好看的冒险电影", "冒险", ["冒险"]),
            ("推荐几部好看的家庭电影", "家庭", ["家庭"]),
            ("推荐几部好看的奇幻电影", "奇幻", ["奇幻"]),
            ("推荐几部好看的恐怖电影", "恐怖", ["恐怖"]),
            ("推荐几部节奏紧凑的悬疑电影", "悬疑", ["节奏", "紧凑"]),
            ("推荐几部经典的西部片", "西部", ["经典", "西部"]),
            ("推荐几部好看的音乐电影", "音乐", ["音乐"]),
            ("推荐几部关于太空探索的电影", "科幻", ["太空", "探索"]),
            ("推荐几部有深度的传记电影", "传记", ["深度", "传记"]),
            ("推荐几部好看的体育励志电影", "剧情", ["体育", "励志"]),
            ("推荐几部经典黑色电影", "犯罪", ["经典", "黑色"]),
            ("推荐几部好看的灾难电影", "剧情", ["灾难"]),
            ("推荐几部轻松搞笑的喜剧片", "喜剧", ["轻松", "搞笑"]),
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
        
        # ═══════════════════════════════════════
        # 多轮对话测试集 (10组)
        # ═══════════════════════════════════════
        multi_turn_sessions = [
            {
                "session_id": f"MT-{i:03d}",
                "turns": [
                    {"role": "user", "content": q1, "expected_slots": s1},
                    {"role": "user", "content": q2, "expected_slots": s2},
                    {"role": "user", "content": q3, "expected_slots": s3},
                ]
            }
            for i, (q1, s1, q2, s2, q3, s3) in enumerate([
                ("想看动作片", {"genre": "动作"}, "不要太老的", {"genre": "动作", "year_min": 2015}, "最好评分高一点", {"genre": "动作", "year_min": 2015, "score_min": 8.0}),
                ("推荐犯罪推理片", {"genre": "犯罪"}, "不要国产的", {"genre": "犯罪", "exclude_country": "中国"}, "最好是近五年的", {"genre": "犯罪", "exclude_country": "中国", "year_min": 2021}),
                ("推荐一部好看的电影", {}, "想要轻松一点的喜剧", {"genre": "喜剧", "keyword": "轻松"}, "最好是周星驰的", {"genre": "喜剧", "keyword": "轻松", "actor": "周星驰"}),
                ("想看科幻片", {"genre": "科幻"}, "要有太空题材的", {"genre": "科幻", "keyword": "太空"}, "评分8分以上", {"genre": "科幻", "keyword": "太空", "score_min": 8.0}),
                ("推荐动画电影", {"genre": "动画"}, "宫崎骏的", {"genre": "动画", "director": "宫崎骏"}, "治愈一点的", {"genre": "动画", "director": "宫崎骏", "keyword": "治愈"}),
                ("想看悬疑片", {"genre": "悬疑"}, "烧脑的那种", {"genre": "悬疑", "keyword": "烧脑"}, "最好是近十年的", {"genre": "悬疑", "keyword": "烧脑", "year_min": 2015}),
                ("推荐爱情电影", {"genre": "爱情"}, "不要太虐的", {"genre": "爱情", "exclude_keyword": "虐"}, "轻松浪漫的", {"genre": "爱情", "keyword": "浪漫"}),
                ("想看战争片", {"genre": "战争"}, "二战题材", {"genre": "战争", "keyword": "二战"}, "评分高的", {"genre": "战争", "keyword": "二战", "score_min": 8.0}),
                ("推荐好看的电影", {}, "动作片", {"genre": "动作"}, "要有追车场面的", {"genre": "动作", "keyword": "追车"}),
                ("想找类似盗梦空间的", {"anchor": "盗梦空间"}, "诺兰导演的", {"anchor": "盗梦空间", "director": "诺兰"}, "科幻类的", {"anchor": "盗梦空间", "director": "诺兰", "genre": "科幻"}),
            ], 1)
        ]
        
        dataset = {
            "metadata": {
                "description": "MovieAgent 端到端评估黄金数据集 V4（100+条，三类均衡）",
                "dataset": "MovieLens-1M",
                "total_movies_in_db": Movie.objects.count(),
                "total_queries": len(queries),
                "difficulty_levels": ["single_hop", "multi_hop", "implicit_semantic"],
                "random_seed": 42,
                "version": "4.0"
            },
            "queries": queries,
            "multi_turn_sessions": multi_turn_sessions,
        }
        
        out_path = "myapp/management/commands/golden_dataset_agent_eval.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        self.stdout.write(f"✅ 黄金数据集V4已生成: {len(queries)} 条查询")
        
        for diff in ['single_hop', 'multi_hop', 'implicit_semantic']:
            count = len([q for q in queries if q['difficulty'] == diff])
            self.stdout.write(f"  {diff}: {count} 条")
        
        self.stdout.write(f"  多轮对话: {len(multi_turn_sessions)} 组")