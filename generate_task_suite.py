#!/usr/bin/env python
"""生成 Agent 任务集（120 个任务）"""
import json, os, sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'movie.settings')

import django
django.setup()
from myapp.models import Movie

# 获取热门电影用于任务设计
movies = list(Movie.objects.prefetch_related('genres', 'directors').order_by('-vote_count')[:100])
MOVIE_DB = []
for m in movies:
    g = [g.name for g in m.genres.all()[:3]]
    d = [d.name for d in m.directors.all()[:1]]
    MOVIE_DB.append({
        'id': m.id, 'title': m.title, 'score': float(m.score) if m.score else 0,
        'genres': g, 'directors': d, 'year': m.date.year if m.date else None
    })

def find_movies(genre=None, director=None, min_score=None, min_year=None, max_year=None, limit=5):
    """从数据库中查找符合条件的电影"""
    results = MOVIE_DB
    if genre:
        results = [m for m in results if any(genre in g for g in m['genres'])]
    if director:
        results = [m for m in results if any(director in d for d in m['directors'])]
    if min_score:
        results = [m for m in results if m['score'] >= min_score]
    if min_year:
        results = [m for m in results if m['year'] and m['year'] >= min_year]
    if max_year:
        results = [m for m in results if m['year'] and m['year'] <= max_year]
    return results[:limit]

tasks = []
task_id = 1

# ============================================================
# Simple 任务 (40 个)
# ============================================================

# 1-10: 类型查询
simple_genre_queries = [
    ("推荐科幻片", "科幻", ["search_vector", "maan_rerank", "rerank"]),
    ("想看喜剧电影", "喜剧", ["search_vector", "maan_rerank", "rerank"]),
    ("推荐动作片", "动作", ["search_vector", "maan_rerank", "rerank"]),
    ("有什么好看的悬疑片", "悬疑", ["search_vector", "maan_rerank", "rerank"]),
    ("推荐爱情电影", "爱情", ["search_vector", "maan_rerank", "rerank"]),
    ("想看动画电影", "动画", ["search_vector", "maan_rerank", "rerank"]),
    ("推荐犯罪片", "犯罪", ["search_vector", "maan_rerank", "rerank"]),
    ("有什么战争片推荐", "战争", ["search_vector", "maan_rerank", "rerank"]),
    ("推荐恐怖片", "恐怖", ["search_vector", "maan_rerank", "rerank"]),
    ("想看奇幻电影", "奇幻", ["search_vector", "maan_rerank", "rerank"]),
]

for query, genre, tool_chain in simple_genre_queries:
    gt = find_movies(genre=genre, limit=5)
    tasks.append({
        'task_id': f'SIM-{task_id:03d}',
        'category': 'simple',
        'input': query,
        'expected_tool_chain': tool_chain,
        'expected_intent': 'QUERY_MOVIE',
        'success_criteria': {'type': 'genre_match', 'genre': genre},
        'ground_truth_movies': [m['title'] for m in gt],
        'ground_truth_ids': [m['id'] for m in gt],
    })
    task_id += 1

# 11-20: 导演查询
simple_director_queries = [
    ("诺兰导演的电影有哪些", "克里斯托弗·诺兰"),
    ("周星驰的电影推荐", "周星驰"),
    ("斯皮尔伯格导演的作品", "史蒂文·斯皮尔伯格"),
    ("昆汀·塔伦蒂诺的电影", "昆汀·塔伦蒂诺"),
    ("大卫·芬奇导演的电影", "大卫·芬奇"),
    ("詹姆斯·卡梅隆的作品", "詹姆斯·卡梅隆"),
    ("彼得·杰克逊导演的电影", "彼得·杰克逊"),
    ("马丁·斯科塞斯的电影", "马丁·斯科塞斯"),
    ("宫崎骏的动画电影", "宫崎骏"),
    ("克里斯托弗·诺兰的科幻片", "克里斯托弗·诺兰"),
]

for query, director in simple_director_queries:
    gt = find_movies(director=director, limit=5)
    tasks.append({
        'task_id': f'SIM-{task_id:03d}',
        'category': 'simple',
        'input': query,
        'expected_tool_chain': ['search_vector', 'maan_rerank', 'rerank'],
        'expected_intent': 'QUERY_MOVIE',
        'success_criteria': {'type': 'director_match', 'director': director},
        'ground_truth_movies': [m['title'] for m in gt],
        'ground_truth_ids': [m['id'] for m in gt],
    })
    task_id += 1

# 21-30: 评分查询
simple_score_queries = [
    ("评分8分以上的科幻片", "科幻", 8.0),
    ("高分动作电影推荐", "动作", 8.0),
    ("有没有评分9分以上的电影", None, 9.0),
    ("推荐高分悬疑片", "悬疑", 8.0),
    ("评分8.5以上的电影", None, 8.5),
    ("高分喜剧推荐", "喜剧", 8.0),
    ("评分最高的犯罪片", "犯罪", 8.0),
    ("推荐高分动画电影", "动画", 8.0),
    ("评分8分以上的爱情片", "爱情", 8.0),
    ("有什么高分战争片", "战争", 8.0),
]

for query, genre, min_score in simple_score_queries:
    gt = find_movies(genre=genre, min_score=min_score, limit=5)
    tasks.append({
        'task_id': f'SIM-{task_id:03d}',
        'category': 'simple',
        'input': query,
        'expected_tool_chain': ['search_vector', 'maan_rerank', 'rerank'],
        'expected_intent': 'QUERY_MOVIE',
        'success_criteria': {'type': 'score_match', 'min_score': min_score, 'genre': genre},
        'ground_truth_movies': [m['title'] for m in gt],
        'ground_truth_ids': [m['id'] for m in gt],
    })
    task_id += 1

# 31-40: 年份查询
simple_year_queries = [
    ("近五年的高分科幻片", "科幻", 2021, None),
    ("2020年以后的动作片", "动作", 2020, None),
    ("最近三年的喜剧电影", "喜剧", 2023, None),
    ("2010年以后的悬疑片", "悬疑", 2010, None),
    ("近十年的高分犯罪片", "犯罪", 2016, None),
    ("2020年以后的动画电影", "动画", 2020, None),
    ("最近五年的爱情片", "爱情", 2021, None),
    ("2015年以后的科幻片", "科幻", 2015, None),
    ("近八年的高分剧情片", "剧情", 2018, None),
    ("2020年以后的奇幻电影", "奇幻", 2020, None),
]

for query, genre, min_year, max_year in simple_year_queries:
    gt = find_movies(genre=genre, min_year=min_year, max_year=max_year, limit=5)
    tasks.append({
        'task_id': f'SIM-{task_id:03d}',
        'category': 'simple',
        'input': query,
        'expected_tool_chain': ['search_vector', 'maan_rerank', 'rerank'],
        'expected_intent': 'QUERY_MOVIE',
        'success_criteria': {'type': 'year_match', 'min_year': min_year, 'genre': genre},
        'ground_truth_movies': [m['title'] for m in gt],
        'ground_truth_ids': [m['id'] for m in gt],
    })
    task_id += 1

# ============================================================
# Complex 任务 (30 个)
# ============================================================

# 1-10: 锚点电影查询
complex_anchor_queries = [
    ("类似《星际穿越》的科幻片", "星际穿越", "科幻"),
    ("和《盗梦空间》差不多的烧脑电影", "盗梦空间", "科幻"),
    ("推荐像《阿凡达》那样的奇幻大片", "阿凡达", "奇幻"),
    ("类似《肖申克的救赎》的高分剧情片", "肖申克的救赎", "剧情"),
    ("和《蝙蝠侠：黑暗骑士》风格相近的犯罪片", "蝙蝠侠：黑暗骑士", "犯罪"),
    ("推荐像《复仇者联盟》那样的超级英雄片", "复仇者联盟", "动作"),
    ("类似《搏击俱乐部》的惊悚片", "搏击俱乐部", "惊悚"),
    ("和《指环王》风格类似的奇幻史诗", "指环王", "奇幻"),
    ("推荐像《死侍》那样的喜剧动作片", "死侍", "喜剧"),
    ("类似《银翼杀手》的科幻经典", "银翼杀手", "科幻"),
]

for query, anchor, genre in complex_anchor_queries:
    gt = find_movies(genre=genre, limit=5)
    tasks.append({
        'task_id': f'COM-{task_id:03d}',
        'category': 'complex',
        'input': query,
        'expected_tool_chain': ['kg_query', 'search_vector', 'maan_rerank', 'rerank'],
        'expected_intent': 'QUERY_MOVIE',
        'success_criteria': {'type': 'anchor_match', 'anchor': anchor, 'genre': genre},
        'ground_truth_movies': [m['title'] for m in gt],
        'ground_truth_ids': [m['id'] for m in gt],
    })
    task_id += 1

# 11-20: 多约束查询
complex_multi_constraint = [
    ("近五年的高分科幻片", "科幻", 2021, None, 8.0),
    ("诺兰导演的科幻电影", "科幻", None, "克里斯托弗·诺兰", None),
    ("2020年以后的高分动作片", "动作", 2020, None, 8.0),
    ("昆汀导演的犯罪片", "犯罪", None, "昆汀·塔伦蒂诺", None),
    ("近八年的高分悬疑片", "悬疑", 2018, None, 8.0),
    ("斯皮尔伯格的战争片", "战争", None, "史蒂文·斯皮尔伯格", None),
    ("2015年以后的高分动画", "动画", 2015, None, 8.0),
    ("大卫·芬奇的惊悚片", "惊悚", None, "大卫·芬奇", None),
    ("近五年的高分喜剧", "喜剧", 2021, None, 8.0),
    ("詹姆斯·卡梅隆的科幻片", "科幻", None, "詹姆斯·卡梅隆", None),
]

for item in complex_multi_constraint:
    query, genre, min_year, director, min_score = item
    gt = find_movies(genre=genre, min_year=min_year, director=director, min_score=min_score, limit=5)
    criteria = {'type': 'multi_constraint', 'genre': genre}
    if min_year:
        criteria['min_year'] = min_year
    if director:
        criteria['director'] = director
    if min_score:
        criteria['min_score'] = min_score
    tool_chain = ['search_vector', 'maan_rerank', 'rerank']
    if director:
        tool_chain = ['kg_query', 'search_vector', 'maan_rerank', 'rerank']
    tasks.append({
        'task_id': f'COM-{task_id:03d}',
        'category': 'complex',
        'input': query,
        'expected_tool_chain': tool_chain,
        'expected_intent': 'QUERY_MOVIE',
        'success_criteria': criteria,
        'ground_truth_movies': [m['title'] for m in gt],
        'ground_truth_ids': [m['id'] for m in gt],
    })
    task_id += 1

# 21-30: 情感+约束查询
complex_emotion_queries = [
    ("轻松治愈的动画电影", "动画", "轻松治愈"),
    ("烧脑的科幻悬疑片", "科幻", "烧脑"),
    ("温馨感人的爱情片", "爱情", "温馨感人"),
    ("刺激的动作冒险片", "动作", "刺激"),
    ("压抑深刻的剧情片", "剧情", "压抑深刻"),
    ("搞笑轻松的喜剧片", "喜剧", "搞笑轻松"),
    ("黑暗风格的犯罪片", "犯罪", "黑暗"),
    ("温暖治愈的家庭片", "家庭", "温暖治愈"),
    ("紧张刺激的惊悚片", "惊悚", "紧张刺激"),
    ("浪漫甜蜜的爱情片", "爱情", "浪漫甜蜜"),
]

for query, genre, emotion in complex_emotion_queries:
    gt = find_movies(genre=genre, limit=5)
    tasks.append({
        'task_id': f'COM-{task_id:03d}',
        'category': 'complex',
        'input': query,
        'expected_tool_chain': ['search_vector', 'maan_rerank', 'rerank'],
        'expected_intent': 'QUERY_MOVIE',
        'success_criteria': {'type': 'emotion_match', 'genre': genre, 'emotion': emotion},
        'ground_truth_movies': [m['title'] for m in gt],
        'ground_truth_ids': [m['id'] for m in gt],
    })
    task_id += 1

# ============================================================
# Vague 任务 (30 个)
# ============================================================

vague_queries = [
    ("推荐电影", "模糊查询，无任何约束"),
    ("好看的", "极简输入，无类型/情感信息"),
    ("不知道看什么", "无明确需求"),
    ("随便推荐", "无明确需求"),
    ("有什么好看的", "泛泛询问"),
    ("推荐一下", "极简输入"),
    ("想看电影", "无类型约束"),
    ("无聊想看电影", "情感化输入，无具体需求"),
    ("最近有什么好看的", "时间模糊，无类型"),
    ("帮我选一部电影", "无具体偏好"),
    ("经典", "极简输入，意图不明"),
    ("想哭", "情感化输入，需追问类型"),
    ("想笑", "情感化输入，需追问类型"),
    ("烧脑", "极简情感输入，需追问类型"),
    ("治愈", "极简情感输入，需追问类型"),
    ("热血", "极简情感输入，需追问类型"),
    ("放松一下", "情感化输入，无具体需求"),
    ("打发时间", "目的性输入，无偏好"),
    ("睡前看的", "场景化输入，需追问类型"),
    ("和朋友一起看", "场景化输入，需追问类型"),
    ("适合一个人看的", "场景化输入，需追问类型"),
    ("不想动脑子", "情感化输入，需追问类型"),
    ("来点刺激的", "情感化输入，需追问类型"),
    ("想看点有深度的", "情感化输入，需追问类型"),
    ("轻松一点的", "情感化输入，需追问类型"),
    ("不要恐怖的", "排除性输入，需追问正面偏好"),
    ("不要太长的", "约束性输入，需追问类型"),
    ("最近很火的", "热度查询，无类型"),
    ("大家都在看什么", "从众查询，无具体偏好"),
    ("有什么新片", "新片查询，无类型"),
]

for query, reason in vague_queries:
    tasks.append({
        'task_id': f'VAG-{task_id:03d}',
        'category': 'vague',
        'input': query,
        'expected_tool_chain': [],
        'expected_intent': 'QUERY_MOVIE',
        'success_criteria': {'type': 'clarification_or_vague', 'reason': reason},
        'ground_truth_movies': [],
        'ground_truth_ids': [],
    })
    task_id += 1

# ============================================================
# Multi-turn 任务 (20 个)
# ============================================================

multiturn_sessions = [
    {
        'turns': ["想看动作片", "不要太老的", "评分高一点"],
        'description': '类型收敛+时间+评分',
    },
    {
        'turns': ["推荐犯罪推理片", "不要国产的", "最好是近五年的"],
        'description': '类型收敛+地区+时间',
    },
    {
        'turns': ["推荐一部好看的电影", "想要轻松一点的喜剧", "最好是周星驰的"],
        'description': '泛→类型→导演',
    },
    {
        'turns': ["想看科幻片", "要有太空题材的", "评分8分以上"],
        'description': '类型→题材→评分',
    },
    {
        'turns': ["推荐动画电影", "宫崎骏的", "治愈一点的"],
        'description': '类型→导演→情感',
    },
    {
        'turns': ["想看悬疑片", "烧脑的那种", "最好是近十年的"],
        'description': '类型→情感→时间',
    },
    {
        'turns': ["推荐爱情电影", "不要太虐的", "轻松浪漫的"],
        'description': '类型→排除→情感',
    },
    {
        'turns': ["想看战争片", "二战题材", "评分高的"],
        'description': '类型→题材→评分',
    },
    {
        'turns': ["推荐好看的电影", "动作片", "要有追车场面的"],
        'description': '泛→类型→场景',
    },
    {
        'turns': ["想找类似盗梦空间的", "诺兰导演的", "科幻类的"],
        'description': '锚点→导演→类型',
    },
    {
        'turns': ["推荐高分电影", "剧情片", "不要太压抑的"],
        'description': '评分→类型→情感',
    },
    {
        'turns': ["想看喜剧", "最近几年的", "评分8分以上"],
        'description': '类型→时间→评分',
    },
    {
        'turns': ["推荐电影", "要有刘德华的", "动作片"],
        'description': '泛→演员→类型',
    },
    {
        'turns': ["想看动画", "日本的", "宫崎骏或者新海诚的"],
        'description': '类型→地区→导演',
    },
    {
        'turns': ["推荐悬疑片", "不要国产的", "近五年的高分"],
        'description': '类型→地区→时间+评分',
    },
    {
        'turns': ["想看科幻大片", "要有视觉冲击的", "评分7.5以上"],
        'description': '类型→视觉→评分',
    },
    {
        'turns': ["推荐犯罪片", "韩国的", "近十年的"],
        'description': '类型→地区→时间',
    },
    {
        'turns': ["想看纪录片", "自然题材", "BBC的"],
        'description': '类型→题材→制作方',
    },
    {
        'turns': ["推荐恐怖片", "不要血腥的", "心理恐怖那种"],
        'description': '类型→排除→子类型',
    },
    {
        'turns': ["想看爱情片", "经典老片", "评分8分以上"],
        'description': '类型→时间→评分',
    },
]

for i, session in enumerate(multiturn_sessions):
    tasks.append({
        'task_id': f'MT-{i+1:03d}',
        'category': 'multiturn',
        'input': session['turns'],
        'expected_tool_chain': ['search_vector', 'maan_rerank', 'rerank'],
        'expected_intent': 'QUERY_MOVIE',
        'success_criteria': {'type': 'multiturn_convergence', 'description': session['description']},
        'ground_truth_movies': [],
        'ground_truth_ids': [],
    })

# 保存任务集
output = {
    'metadata': {
        'total_tasks': len(tasks),
        'categories': {
            'simple': len([t for t in tasks if t['category'] == 'simple']),
            'complex': len([t for t in tasks if t['category'] == 'complex']),
            'vague': len([t for t in tasks if t['category'] == 'vague']),
            'multiturn': len([t for t in tasks if t['category'] == 'multiturn']),
        },
    },
    'tasks': tasks,
}

filepath = '/home/daylight/下载/DjangoProject3/DjangoProject3/myapp/management/commands/golden_dataset_agent_eval.json'
with open(filepath, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f'Generated {len(tasks)} tasks')
print(f'  Simple: {output["metadata"]["categories"]["simple"]}')
print(f'  Complex: {output["metadata"]["categories"]["complex"]}')
print(f'  Vague: {output["metadata"]["categories"]["vague"]}')
print(f'  Multi-turn: {output["metadata"]["categories"]["multiturn"]}')
print(f'Saved to {filepath}')
