"""
导演美学签名库 (Director Aesthetic Signatures)
================================================
从数据库动态生成导演的美学特征描述，为推荐解释提供"人情味"先验知识。

设计策略：
  1. 精英种子库：对知名导演保留手写的高质量美学签名（学术精炼度高）
  2. 数据库自动补全：对种子库外的导演，从其作品的 Genre/Topic/Mood 分布自动生成
  3. 缓存机制：首次生成后缓存至内存，避免重复查询

使用方式：
    from myapp.utils.director_styles import get_director_signature, get_director_styles
    sig = get_director_signature("Christopher Nolan")  # 返回美学签名字符串
    styles = get_director_styles()  # 返回完整 {导演名: 签名} 字典
================================================
"""

import re
from collections import Counter
from functools import lru_cache


# =============================================================
# 精英种子库（手写高质量签名）
# =============================================================
# 这些签名经过学术提炼，融合了导演的视觉风格、叙事手法、情感基调等维度
SEED_SIGNATURES = {
    "Christopher Nolan": "非线性叙事、冷色调视觉、实景拍摄、时间维度探索",
    "Steven Spielberg": "暖色调逆光、人文关怀、宏大视角、童话感",
    "Wes Anderson": "极致对称构图、高饱和色调、平面化视觉、冷幽默荒诞感",
    "Wong Kar-wai": "蓝调/琥珀色滤镜、抽帧视觉、极致孤独感、暧昧氛围",
    "Hayao Miyazaki": "清新手绘风格、飞行梦境、人与自然、治愈感",
    "Quentin Tarantino": "非线性章节结构、暴力美学、复古配乐、话痨式对白",
    "David Fincher": "暗色调悬疑、精密构图、心理惊悚、冷峻质感",
    "Martin Scorsese": "长镜头运动、摇滚配乐、暴力与救赎、城市底层叙事",
    "Stanley Kubrick": "对称构图、冷峻疏离、哲学思辨、视觉符号系统",
    "Ridley Scott": "史诗质感、暗色调光效、宏大世界观、异形美学",
    "James Cameron": "技术创新驱动、水/光元素、宏大动作场面、浪漫主义",
    "Denis Villeneuve": "沙漠美学、缓慢节奏、极简构图、存在主义主题",
    "赵婷": "自然光纪实、广袤风景、边缘人物、诗意现实主义",
    "奉俊昊": "阶层空间隐喻、黑色幽默、类型杂糅、社会批判",
    "是枝裕和": "家庭日常、固定机位、自然光、含蓄情感表达",
    "陈凯歌": "宏大历史叙事、戏曲美学、色彩象征、文化反思",
    "张艺谋": "浓烈色彩、对称构图、民俗元素、视觉冲击力",
    "姜文": "黑色幽默、历史解构、隐喻蒙太奇、浪漫英雄主义",
    "冯小刚": "都市喜剧、平民视角、温情与讽刺、贺岁档美学",
}


# =============================================================
# 类型 → 美学关键词映射（用于自动生成签名）
# =============================================================
# 从电影 Genre/Topic 自动映射到美学风格描述词
GENRE_AESTHETIC_MAP = {
    # 类型标签 → 美学描述词
    "Sci-Fi": "科幻视觉",
    "科幻": "科幻视觉",
    "Fantasy": "奇幻想象",
    "奇幻": "奇幻想象",
    "Animation": "动画风格",
    "动画": "动画风格",
    "Horror": "暗黑氛围",
    "恐怖": "暗黑氛围",
    "Thriller": "悬疑张力",
    "惊悚": "悬疑张力",
    "Mystery": "悬疑推理",
    "悬疑": "悬疑推理",
    "Crime": "犯罪叙事",
    "犯罪": "犯罪叙事",
    "War": "战争史诗",
    "战争": "战争史诗",
    "Action": "动作场面",
    "动作": "动作场面",
    "Adventure": "冒险探索",
    "冒险": "冒险探索",
    "Romance": "浪漫情感",
    "爱情": "浪漫情感",
    "Drama": "情感深度",
    "剧情": "情感深度",
    "Comedy": "喜剧节奏",
    "喜剧": "喜剧节奏",
    "Musical": "音乐韵律",
    "音乐": "音乐韵律",
    "Documentary": "纪实质感",
    "纪录片": "纪实质感",
    "Western": "西部旷野",
    "History": "历史厚重",
    "历史": "历史厚重",
    "Family": "家庭温情",
    "家庭": "家庭温情",
}

# Mood → 情感基调描述词
MOOD_AESTHETIC_MAP = {
    "Thought-provoking": "发人深省",
    "Intense": "紧张刺激",
    "Light-hearted": "轻松愉悦",
    "Heartwarming": "温馨治愈",
    "Exciting": "惊险刺激",
    "Imaginative": "充满想象",
    "Gripping": "扣人心弦",
    "Emotional": "情感深沉",
    "Reflective": "沉思回味",
    "General": "均衡叙事",
}


def _query_director_stats(director_name):
    """
    从数据库查询导演的作品类型分布和情绪基调分布。
    
    Returns:
        dict: {
            'genre_counter': Counter,   # 类型标签计数
            'mood_counter': Counter,    # 情绪基调计数
            'movie_count': int,         # 作品数量
            'avg_score': float,         # 平均评分
        }
    """
    try:
        from myapp.models import Movie, Actor
        
        # 找到导演对象
        director = Actor.objects.filter(name=director_name).first()
        if not director:
            return None
        
        # 查询该导演执导的所有电影
        movies = Movie.objects.filter(
            directors=director
        ).prefetch_related('genres').exclude(score__isnull=True)
        
        if not movies.exists():
            return None
        
        genre_counter = Counter()
        mood_counter = Counter()
        scores = []
        
        for movie in movies:
            # 统计类型分布
            for genre in movie.genres.all():
                genre_counter[genre.name] += 1
            
            # 推断情绪基调（简化版，复用 build_kg 的规则）
            mood = _infer_mood(movie)
            if mood:
                mood_counter[mood] += 1
            
            if movie.score:
                scores.append(movie.score)
        
        return {
            'genre_counter': genre_counter,
            'mood_counter': mood_counter,
            'movie_count': movies.count(),
            'avg_score': sum(scores) / len(scores) if scores else 0,
        }
    except Exception as e:
        print(f"[DirectorStyles] 查询导演 {director_name} 数据失败: {e}")
        return None


def _infer_mood(movie):
    """
    基于电影类型和评分推断情绪基调（简化版规则引擎）。
    """
    genre_names = set(g.name for g in movie.genres.all())
    
    if genre_names & {'Sci-Fi', 'Mystery', 'Thriller', '科幻', '悬疑', '惊悚'}:
        return 'Thought-provoking'
    if genre_names & {'Horror', '恐怖'}:
        return 'Intense'
    if genre_names & {'Comedy', '喜剧'}:
        return 'Light-hearted'
    if genre_names & {'Romance', '爱情'}:
        return 'Heartwarming'
    if genre_names & {'Action', 'Adventure', '动作', '冒险'}:
        return 'Exciting'
    if genre_names & {'Animation', 'Fantasy', '动画', '奇幻'}:
        return 'Imaginative'
    if genre_names & {'Crime', '犯罪'}:
        return 'Gripping'
    if genre_names & {'War', '战争'}:
        return 'Emotional'
    if genre_names & {'Drama', '剧情'}:
        return 'Reflective'
    
    # 默认
    if movie.score and movie.score >= 8.0:
        return 'Thought-provoking'
    return 'General'


def _generate_signature_from_stats(stats):
    """
    从导演作品统计信息自动生成美学签名。
    
    策略：
    1. 取 Top-3 类型 → 映射为美学描述词
    2. 取 Top-2 情绪基调 → 映射为情感描述词
    3. 如果作品数量≥10，添加"高产"标签
    4. 如果平均评分≥8.0，添加"口碑佳作"标签
    """
    if not stats:
        return ""
    
    parts = []
    
    # 类型美学特征（取 Top-3）
    for genre, count in stats['genre_counter'].most_common(3):
        aesthetic = GENRE_AESTHETIC_MAP.get(genre, "")
        if aesthetic and aesthetic not in parts:
            parts.append(aesthetic)
    
    # 情绪基调（取 Top-2）
    for mood, count in stats['mood_counter'].most_common(2):
        aesthetic = MOOD_AESTHETIC_MAP.get(mood, "")
        if aesthetic and aesthetic not in parts:
            parts.append(aesthetic)
    
    # 产量与口碑标签
    if stats['movie_count'] >= 10:
        parts.append("高产稳定")
    if stats['avg_score'] >= 8.0:
        parts.append("口碑保证")
    
    return "、".join(parts[:5]) if parts else ""


# =============================================================
# 公开 API
# =============================================================

# 内存缓存（避免重复查询数据库）
_cached_styles = None


def get_director_signature(director_name):
    """
    获取单个导演的美学签名。
    
    优先级：种子库 > 数据库自动生成 > 空字符串
    
    Args:
        director_name: 导演姓名（英文/中文）
    
    Returns:
        str: 美学签名，如"非线性叙事、冷色调视觉、实景拍摄、时间维度探索"
    """
    # 1. 精英种子库命中
    if director_name in SEED_SIGNATURES:
        return SEED_SIGNATURES[director_name]
    
    # 2. 数据库自动补全
    stats = _query_director_stats(director_name)
    if stats and stats['movie_count'] >= 2:  # 至少2部作品才生成签名
        return _generate_signature_from_stats(stats)
    
    return ""


def get_director_styles():
    """
    获取完整的导演美学签名库。
    
    合并策略：种子库 + 数据库中所有有≥3部作品的导演的自动签名。
    结果缓存至内存，下次调用直接返回。
    
    Returns:
        dict: {导演名: 美学签名字符串}
    """
    global _cached_styles
    if _cached_styles is not None:
        return _cached_styles
    
    styles = dict(SEED_SIGNATURES)  # 以种子库为基础
    
    try:
        from myapp.models import Movie, Actor
        from django.db.models import Count
        
        # 查询所有执导过≥3部电影的导演
        directors = Actor.objects.annotate(
            movie_count=Count('directed_movies')
        ).filter(movie_count__gte=3).order_by('-movie_count')
        
        for director in directors:
            if director.name in styles:
                continue  # 种子库已有，跳过
            
            signature = get_director_signature(director.name)
            if signature:
                styles[director.name] = signature
        
        print(f"[DirectorStyles] 美学签名库加载完成: 种子{len(SEED_SIGNATURES)}人 + "
              f"自动补全{len(styles) - len(SEED_SIGNATURES)}人 = 共{len(styles)}人")
        
    except Exception as e:
        print(f"[DirectorStyles] 数据库查询失败，仅使用种子库: {e}")
    
    _cached_styles = styles
    return _cached_styles


def clear_cache():
    """清除缓存（用于数据更新后刷新）"""
    global _cached_styles
    _cached_styles = None