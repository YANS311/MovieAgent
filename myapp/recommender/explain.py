"""
推荐可解释性模块 (Explainability)
================================================
为推荐结果生成自然语言推荐理由，实现：
  1. 基于知识图谱的归因解释
  2. 基于用户画像的个性化解释
  3. 基于内容特征的匹配解释
  4. 多模态视觉关联解释

论文核心公式（§5.7.3 最优锚点选择算子）：
  score(h) = { 5.0 + S_g × 0.2 + S_v × 0.1   if S_d > 0
             { S_g × 0.3 + S_v × 0.2            otherwise

其中：
  S_d = 1.0 if 同导演 else 0
  S_a = Jaccard(A_target, A_anchor)  # 主演集合 Jaccard 系数
  S_g = |G_target ∩ G_anchor| / |G_target ∪ G_anchor|  # 类型重合度
  S_v = cosine_sim(CLIP(target), CLIP(anchor))  # 视觉向量余弦相似度

论文展示核心："推荐《降临》，因为你偏好硬科幻与高概念叙事。"
================================================
"""

import re
import numpy as np
import hashlib
from collections import Counter
from django.core.cache import cache
from django.db.models import Count, Q

from myapp.models import Movie, UserRating, Genre, Actor


# =============================================================
# 视觉向量工具函数 (§5.7.3 S_v 计算)
# =============================================================

_visual_cache = {}  # 内存缓存，避免重复加载 FAISS 索引

def _load_visual_index():
    """加载 FAISS 视觉索引和 ID 映射（单例缓存）"""
    if 'index' in _visual_cache and 'id_map' in _visual_cache:
        return _visual_cache['index'], _visual_cache['id_map']
    
    try:
        import faiss
        import pickle
        import os
        
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        index_path = os.path.join(base_dir, 'faiss_visual_index.bin')
        id_path = os.path.join(base_dir, 'visual_ids.pkl')
        
        if os.path.exists(index_path) and os.path.exists(id_path):
            index = faiss.read_index(index_path)
            with open(id_path, 'rb') as f:
                id_map = pickle.load(f)  # {str(movie_id): int(faiss_idx)}
            _visual_cache['index'] = index
            _visual_cache['id_map'] = id_map
            return index, id_map
    except Exception as e:
        print(f"[Explain] 视觉索引加载失败: {e}")
    
    return None, None


def _get_visual_embedding(movie_id):
    """
    从 FAISS 索引中获取电影的 CLIP 视觉向量。
    
    Args:
        movie_id: 电影ID
    
    Returns:
        np.ndarray (128维) 或 None
    """
    index, id_map = _load_visual_index()
    if index is None or id_map is None:
        return None
    
    faiss_idx = id_map.get(str(movie_id))
    if faiss_idx is not None:
        try:
            return index.reconstruct(int(faiss_idx))
        except Exception:
            pass
    return None


def _cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度 (§5.7.3 S_v)。
    
    S_v = cos(θ) = (v1 · v2) / (||v1|| × ||v2||)
    """
    if vec1 is None or vec2 is None:
        return 0.0
    dot = np.dot(vec1, vec2)
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    return float(dot / norm) if norm > 0 else 0.0


# =============================================================
# 归因引擎：分析推荐原因
# =============================================================
def analyze_recommend_reason(user, movie_id, anchor_movie=None):
    """
    分析推荐某部电影给某个用户的原因。
    
    Args:
        user: 用户对象
        movie_id: 推荐的电影ID
        anchor_movie: 锚点电影（用户看过的、与推荐电影关联的电影）
    
    Returns:
        Dict: {
            'reason_type': str,      # 推荐归因类型
            'strength': float,       # 归因强度 0-1
            'reason_text': str,      # 自然语言推荐理由
            'evidence': dict,        # 证据链
        }
    """
    target = Movie.objects.filter(id=movie_id).prefetch_related(
        'genres', 'actors', 'directors'
    ).first()
    
    if not target:
        return {'reason_type': 'unknown', 'strength': 0, 'reason_text': '未找到电影信息', 'evidence': {}}
    
    # §5.7.2(1) 锚点候选集构建：从用户 Long-term Profile 中，
    # 选取评分 ≥ 7.5 的最近 20 部影片构建动态锚点池 H
    user_high_ratings = UserRating.objects.filter(
        user=user, score__gte=7.5
    ).select_related('movie').prefetch_related(
        'movie__genres', 'movie__directors', 'movie__actors'
    ).order_by('-comment_time')[:20]
    
    history_list = list(user_high_ratings)
    
    # 如果没有历史，返回冷启动推荐
    if not history_list:
        return _cold_start_explain(target)
    
    # §5.7.3 最优锚点选择算子
    if not anchor_movie:
        anchor_movie, reason_type, strength = _find_best_anchor(target, history_list)
    else:
        reason_type, strength = _classify_relation(target, anchor_movie)
    
    # 生成推荐理由
    evidence = {
        'target_movie': target.title,
        'anchor_movie': anchor_movie.title if anchor_movie else None,
        'target_genres': [g.name for g in target.genres.all()],
        'target_directors': [d.name for d in target.directors.all()],
    }
    
    if anchor_movie:
        reason_text = _generate_natural_reason(target, anchor_movie, reason_type, strength)
        evidence['common_elements'] = _find_common_elements(target, anchor_movie)
    else:
        reason_text = f"推荐《{target.title}》(ID:{target.id})——评分{target.score}分的优质影片，值得一观。"
    
    return {
        'reason_type': reason_type,
        'strength': strength,
        'reason_text': reason_text,
        'evidence': evidence,
    }


def _cold_start_explain(movie):
    """冷启动推荐解释"""
    genres = [g.name for g in movie.genres.all()[:3]]
    genre_str = "、".join(genres) if genres else "精彩"
    directors = [d.name for d in movie.directors.all()[:1]]
    
    if directors:
        text = f"推荐《{movie.title}》(ID:{movie.id})——由{directors[0]}执导的{genre_str}佳作，评分{movie.score}分，不容错过。"
    else:
        text = f"推荐《{movie.title}》(ID:{movie.id})——评分{movie.score}分的{genre_str}佳作，广受好评。"
    
    return {
        'reason_type': '冷启动热推',
        'strength': 0.3,
        'reason_text': text,
        'evidence': {'cold_start': True},
    }


# =============================================================
# §5.7.3 最优锚点选择算子（核心评分函数）
# =============================================================
def _find_best_anchor(target, history_list):
    """
    基于论文 §5.7.3 的分段计分函数寻找最优锚点 h*。
    
    评分公式（Eq.5-1）：
        score(h) = 5.0 + S_g × 0.2 + S_v × 0.1    if S_d > 0
        score(h) = S_g × 0.3 + S_v × 0.2            otherwise
    
    其中：
        S_d: 导演关联强度（同导演=1.0，否则=0）
        S_a: 演员关联强度（Jaccard 系数）——用于归因分类
        S_g: 类型关联强度（Genre 标签 Jaccard 重合度）
        S_v: 视觉特征相似度（CLIP 向量余弦相似度）
    """
    best_anchor = None
    best_type = '综合美学相似'
    best_score = -1.0
    
    # 预计算目标电影特征集合
    target_genres = set(g.name for g in target.genres.all())
    target_directors = set(d.name for d in target.directors.all())
    target_actors = set(a.name for a in target.actors.all())
    
    # 预加载目标电影视觉向量（§5.7.3 S_v）
    target_visual = _get_visual_embedding(target.id)
    
    for h in history_list:
        h_movie = h.movie
        if h_movie.id == target.id:
            continue
        
        h_genres = set(g.name for g in h_movie.genres.all())
        h_directors = set(d.name for d in h_movie.directors.all())
        h_actors = set(a.name for a in h_movie.actors.all())
        
        # ── §5.7.2(2) 多维关联强度计算 ──
        
        # S_d: 导演关联强度（同导演=1.0，否则=0）
        common_dirs = target_directors & h_directors
        S_d = 1.0 if common_dirs else 0.0
        
        # S_a: 演员关联强度（Jaccard 系数）
        # Jaccard(A, B) = |A ∩ B| / |A ∪ B|
        actor_union = target_actors | h_actors
        actor_inter = target_actors & h_actors
        S_a = len(actor_inter) / len(actor_union) if actor_union else 0.0
        
        # S_g: 类型关联强度（Genre Jaccard 重合度）
        genre_union = target_genres | h_genres
        genre_inter = target_genres & h_genres
        S_g = len(genre_inter) / len(genre_union) if genre_union else 0.0
        
        # S_v: 视觉特征相似度（CLIP 向量余弦相似度）
        h_visual = _get_visual_embedding(h_movie.id)
        S_v = _cosine_similarity(target_visual, h_visual) if (target_visual is not None and h_visual is not None) else 0.0
        
        # ── §5.7.3 分段计分函数 (Eq.5-1) ──
        if S_d > 0:
            # 同导演：基础分 5.0 + 类型加权 + 视觉加权
            score = 5.0 + S_g * 0.2 + S_v * 0.1
        else:
            # 不同导演：纯类型+视觉加权
            score = S_g * 0.3 + S_v * 0.2
        
        if score > best_score:
            best_score = score
            best_anchor = h_movie
            
            # 确定归因类型（按优先级：导演 > 演员 > 类型 > 综合）
            if S_d > 0:
                director_name = list(common_dirs)[0]
                best_type = f"同导演【{director_name}】"
            elif S_a > 0.3:  # 演员 Jaccard > 0.3 视为显著关联
                actor_name = list(actor_inter)[0]
                best_type = f"同主演【{actor_name}】"
            elif S_g > 0.3:  # 类型 Jaccard > 0.3 视为显著关联
                genre_name = list(genre_inter)[0]
                best_type = f"题材共鸣【{genre_name}】"
            else:
                best_type = '综合美学相似'
    
    # 兜底：如果没找到锚点，用最近一部高分电影
    if not best_anchor and history_list:
        best_anchor = history_list[0].movie
        best_score = 0.2
        best_type = '观影习惯关联'
    
    return best_anchor, best_type, best_score


def _classify_relation(target, anchor):
    """
    分类两部电影的关系类型（外部指定锚点时使用）。
    """
    target_directors = set(d.name for d in target.directors.all())
    anchor_directors = set(d.name for d in anchor.directors.all())
    
    if target_directors & anchor_directors:
        return '同导演', 0.95
    
    target_actors = set(a.name for a in target.actors.all())
    anchor_actors = set(a.name for a in anchor.actors.all())
    if target_actors & anchor_actors:
        return '同主演', 0.7
    
    target_genres = set(g.name for g in target.genres.all())
    anchor_genres = set(g.name for g in anchor.genres.all())
    if target_genres & anchor_genres:
        return '同类型', 0.5
    
    return '综合美学相似', 0.3


def _find_common_elements(target, anchor):
    """找出两部电影的共同元素"""
    common = {}
    
    t_dirs = set(d.name for d in target.directors.all())
    a_dirs = set(d.name for d in anchor.directors.all())
    if t_dirs & a_dirs:
        common['directors'] = list(t_dirs & a_dirs)
    
    t_acts = set(a.name for a in target.actors.all())
    a_acts = set(a.name for a in anchor.actors.all())
    if t_acts & a_acts:
        common['actors'] = list(t_acts & a_acts)
    
    t_gens = set(g.name for g in target.genres.all())
    a_gens = set(g.name for g in anchor.genres.all())
    if t_gens & a_gens:
        common['genres'] = list(t_gens & a_gens)
    
    return common


# =============================================================
# §5.7.4 自然语言模板生成 (NLG)
# =============================================================
# 表 5-3 典型归因场景下的解释生成示例
# ┌─────────┬────────────────────┬────────────────────────────────────────────────────────────────────────────┐
# │归因类型  │归因逻辑描述         │典型输出示例                                                                │
# ├─────────┼────────────────────┼────────────────────────────────────────────────────────────────────────────┤
# │同导演    │强调导演的个人风格印记 │"与您喜爱的《星际穿越》同为诺兰执导，延续了其独特的非线性叙事风格。"               │
# │同主演    │突出演员的演技魅力    │"莱昂纳多在《荒野猎人》中展现了极高造诣，在本片中同样有惊艳表现。"                  │
# │同类型    │链接相近的审美主题    │"本片与您心目中的经典《黑客帝国》同属赛博朋克题材，探讨了意识的边界。"               │
# │综合相似   │结合视觉与情感基调的融合│"基于您对《绿皮书》的评价，本片在温暖治愈的情感基调上与您的品味高度契合。"            │
# └─────────┴────────────────────┴────────────────────────────────────────────────────────────────────────────┘

# 动态槽位定义（Dynamic Slot Filling）
# {anchor_ref}   - 锚点电影引用（含ID，前端可渲染为可点击链接）
# {target_ref}   - 目标电影引用
# {director}     - 共享导演姓名
# {actor}        - 共享演员姓名
# {genre}        - 共享类型标签
# {style_hint}   - 从电影简介中动态提取的风格关键词

def _generate_natural_reason(target, anchor, reason_type, strength):
    """
    §5.7.4 自然语言模板生成 (NLG)。
    
    设计原则：
    1. 事实准确性：每个槽位填充均来自知识图谱的真实关系
    2. 修辞美感：使用情感化措辞（"您喜爱的"、"心目中的经典"）建立共鸣
    3. 交互性：电影引用附带 ID 标记，前端可渲染为可点击链接
    4. 人情味：融入情感锚定词，避免冷冰冰的事实陈述
    
    模板场景覆盖（表 5-3）：
    - 同导演：强调导演的个人风格印记
    - 同主演：突出演员的演技魅力
    - 同类型：链接相近的审美主题
    - 综合相似：结合视觉与情感基调的融合
    """
    target_title = target.title
    anchor_title = anchor.title
    # §5.7.4 来源电影带上 ID，前端正则可将其转为可点击链接
    anchor_ref = f"《{anchor_title}》(ID:{anchor.id})"
    target_ref = f"《{target_title}》(ID:{target.id})"
    target_genres = [g.name for g in target.genres.all()[:2]]
    genre_str = "、".join(target_genres) if target_genres else "电影"
    
    # ── 场景1: 同导演 ──
    # 表 5-3 归因逻辑：强调导演的个人风格印记
    # 典型输出："与您喜爱的《星际穿越》同为诺兰执导，延续了其独特的非线性叙事风格。"
    if '同导演' in reason_type:
        director = reason_type.split('【')[1].rstrip('】') if '【' in reason_type else '同一位导演'
        
        # 动态槽位：从电影简介中提取风格关键词（表 5-3 设计原则）
        style_hint = ""
        if target.summary:
            style_keywords = [
                '非线性叙事', '蒙太奇', '视觉美学', '长镜头', '意识流',
                '风格', '叙事', '视觉', '美学', '镜头', '氛围', '节奏',
                '悬疑', '烧脑', '反转', '隐喻', '象征'
            ]
            for kw in style_keywords:
                if kw in target.summary:
                    idx = target.summary.index(kw)
                    snippet = target.summary[max(0, idx-8):idx+15].replace('\n', ' ').strip()
                    if snippet and len(snippet) > 3:
                        style_hint = f"，延续了其独特的{snippet}"
                        break
        
        if style_hint:
            return f"与您喜爱的{anchor_ref}同为{director}执导{style_hint}。"
        else:
            return f"与您喜爱的{anchor_ref}同为{director}执导，延续了其独特的创作风格与视听语言。"
    
    # ── 场景2: 同主演 ──
    # 表 5-3 归因逻辑：突出演员的演技魅力
    # 典型输出："莱昂纳多在《荒野猎人》中展现了极高造诣，在本片中同样有惊艳表现。"
    elif '同主演' in reason_type:
        actor = reason_type.split('【')[1].rstrip('】') if '【' in reason_type else '同一位演员'
        
        # 动态槽位：从简介中提取表演相关描述
        performance_hint = ""
        if target.summary:
            perf_keywords = ['演技', '表演', '角色', '塑造', '演绎', '诠释']
            for kw in perf_keywords:
                if kw in target.summary:
                    idx = target.summary.index(kw)
                    snippet = target.summary[max(0, idx-5):idx+12].replace('\n', ' ').strip()
                    if snippet and len(snippet) > 3:
                        performance_hint = f"，{snippet}"
                        break
        
        if performance_hint:
            return f"{actor}在{anchor_ref}中展现了极高造诣，在本片{target_ref}中同样有惊艳表现{performance_hint}。"
        else:
            return f"{actor}在{anchor_ref}中展现了极高造诣，在本片{target_ref}中同样有惊艳表现，其精湛演技为这部{genre_str}作品增色不少。"
    
    # ── 场景3: 同类型 ──
    # 表 5-3 归因逻辑：链接相近的审美主题
    # 典型输出："本片与您心目中的经典《黑客帝国》同属赛博朋克题材，探讨了意识的边界。"
    elif '题材共鸣' in reason_type or '同类型' in reason_type:
        genre = reason_type.split('【')[1].rstrip('】') if '【' in reason_type else genre_str
        
        # 动态槽位：从简介中提取主题描述
        theme_hint = ""
        if target.summary:
            theme_keywords = ['探讨', '反思', '审视', '追问', '揭示', '展现', '描绘', '讲述']
            for kw in theme_keywords:
                if kw in target.summary:
                    idx = target.summary.index(kw)
                    snippet = target.summary[idx:idx+20].replace('\n', ' ').strip()
                    if snippet and len(snippet) > 5:
                        # 截取到第一个句号
                        period_idx = snippet.find('。')
                        if period_idx > 0:
                            snippet = snippet[:period_idx]
                        theme_hint = f"，{snippet}。"
                        break
        
        if theme_hint:
            return f"本片与您心目中的经典{anchor_ref}同属{genre}题材{theme_hint}"
        else:
            return f"本片与您心目中的经典{anchor_ref}同属{genre}题材，在叙事主题与审美意趣上一脉相承。"
    
    # ── 场景4: 综合相似 ──
    # 表 5-3 归因逻辑：结合视觉与情感基调的融合
    # 典型输出："基于您对《绿皮书》的评价，本片在温暖治愈的情感基调上与您的品味高度契合。"
    else:
        # 动态槽位：从简介中提取情感/氛围描述
        mood_hint = ""
        if target.summary:
            mood_keywords = [
                '温暖', '治愈', '感动', '幽默', '轻松', '深刻', '震撼',
                '温馨', '浪漫', '压抑', '紧张', '悬疑', '热血', '燃'
            ]
            for kw in mood_keywords:
                if kw in target.summary:
                    mood_hint = f"在{kw}的情感基调上"
                    break
        
        if mood_hint:
            return f"基于您对{anchor_ref}的高分评价，本片{mood_hint}与您的品味高度契合。"
        else:
            return (
                f"基于您对{anchor_ref}的高分评价，"
                f"本片{target_ref}在视觉风格和情感基调上"
                f"与您的观影品味高度契合，同属{genre_str}佳作。"
            )


# =============================================================
# 批量解释生成
# =============================================================
def batch_explain(user, movie_ids, max_items=5):
    """
    为一批推荐电影生成解释。
    
    Args:
        user: 用户对象
        movie_ids: 推荐的电影ID列表
        max_items: 最多解释几部
    
    Returns:
        Dict[int, str] - {movie_id: 推荐理由文本}
    """
    explanations = {}
    for mid in movie_ids[:max_items]:
        result = analyze_recommend_reason(user, mid)
        explanations[mid] = verify_explanation(result['reason_text'], mid)
    return explanations


# =============================================================
# KG 三元组校验层 —— 防幻觉后处理
# =============================================================

def verify_explanation(reason_text, movie_id):
    """
    防幻觉后处理：校验推荐理由中的核心事实性断言。
    
    校验流程：
    1. 从推荐理由中提取人名（导演/演员）和类型标签
    2. 逐一在数据库中验证是否与目标电影关联
    3. 将未验证的断言替换为安全的泛化表述
    
    Args:
        reason_text: 原始推荐理由
        movie_id: 目标电影ID
    
    Returns:
        str: 经过校验的推荐理由
    """
    target = Movie.objects.filter(id=movie_id).prefetch_related(
        'directors', 'actors', 'genres'
    ).first()
    
    if not target:
        return reason_text
    
    # 获取目标电影的真实属性
    real_directors = set(d.name for d in target.directors.all())
    real_actors = set(a.name for a in target.actors.all())
    real_genres = set(g.name for g in target.genres.all())
    
    # 校验规则1：如果理由中提到了"同为XX执导"，验证XX确实是目标电影的导演
    director_pattern = re.search(r'同为([\u4e00-\u9fff·]{2,20})执导', reason_text)
    if director_pattern:
        claimed_director = director_pattern.group(1)
        if claimed_director not in real_directors:
            # 幻觉：声称的导演不是目标电影的真实导演
            safe_director = list(real_directors)[0] if real_directors else "优秀导演"
            reason_text = reason_text.replace(
                f'同为{claimed_director}执导',
                f'由{safe_director}执导'
            )
    
    # 校验规则2：如果理由中提到了演员名，验证其确实出演了目标电影
    actor_pattern = re.search(r'([\u4e00-\u9fff·]{2,10})在.*?中展现了', reason_text)
    if actor_pattern:
        claimed_actor = actor_pattern.group(1)
        if claimed_actor not in real_actors and claimed_actor not in real_directors:
            # 幻觉：声称的演员既非演员也非导演
            genre_str = "、".join(list(real_genres)[:2]) if real_genres else "电影"
            reason_text = (
                f"推荐《{target.title}》(ID:{target.id})——"
                f"评分{target.score}分的{genre_str}佳作，"
                f"与您喜爱的影片在风格气质上高度契合。"
            )
    
    # 校验规则3：确保引用的电影名确实是目标电影
    if target.title not in reason_text and f"《{target.title}》" not in reason_text:
        reason_text = reason_text.rstrip('。') + f"。推荐《{target.title}》(ID:{target.id})。"
    
    return reason_text
