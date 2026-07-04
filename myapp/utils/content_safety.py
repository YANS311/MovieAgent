"""
未成年人内容保护模块 (Content Safety for Minors)
================================================
根据用户年龄自动过滤不适宜的电影内容。

核心功能：
  1. 检测用户是否为未成年人（age < 18）
  2. 自动排除标记为 is_sensitive=True 的电影
  3. 基于 sensitive_type 分级过滤
  4. 基于类型标签过滤（恐怖、暴力等）
  5. 生成给 LLM 的内容安全约束 Prompt

使用方式：
    from myapp.utils.content_safety import (
        is_minor_user,
        get_minor_excluded_ids,
        get_content_safety_prompt,
    )
================================================
"""

import logging
from typing import Set

logger = logging.getLogger('movie_agent')


# =============================================================
# 年龄分级配置
# =============================================================

# 敏感类型 → 最低允许年龄映射
SENSITIVE_TYPE_AGE_MAP = {
    '血腥': 18,
    '暴力': 18,
    '恐怖': 16,
    '暴露': 18,
    '成人内容': 18,
    '性暗示': 18,
    '毒品': 18,
    '赌博': 18,
    '黑暗': 14,
    '惊悚': 14,
}

# 不适宜未成年人的类型标签 → 最低允许年龄
MINOR_UNSAFE_GENRES = {
    '恐怖': 16,
    '惊悚': 14,
    'Horror': 16,
    'Thriller': 14,
}


def is_minor_user(user) -> bool:
    """判断用户是否为未成年人 (age < 18)"""
    age = getattr(user, 'age', None)
    if age is None:
        return False
    try:
        return int(age) < 18
    except (ValueError, TypeError):
        return False


def get_minor_excluded_ids(user) -> Set[int]:
    """
    获取未成年人应排除的电影 ID 集合。
    排除规则：
    1. is_sensitive=True 且 sensitive_type 对应年龄 > 用户年龄
    2. 包含恐怖/惊悚类型标签且年龄不满足要求
    """
    if not is_minor_user(user):
        return set()

    try:
        age = int(user.age)
    except (ValueError, TypeError):
        return set()

    from myapp.models import Movie

    excluded_ids = set()

    # 规则 1: 敏感电影
    for movie_id, s_type in Movie.objects.filter(
        is_sensitive=True
    ).values_list('id', 'sensitive_type'):
        if s_type:
            min_age = SENSITIVE_TYPE_AGE_MAP.get(s_type, 18)
            if age < min_age:
                excluded_ids.add(movie_id)
        else:
            excluded_ids.add(movie_id)

    # 规则 2: 类型标签过滤
    for genre_name, min_age in MINOR_UNSAFE_GENRES.items():
        if age < min_age:
            ids = set(
                Movie.objects.filter(genres__name__icontains=genre_name)
                .values_list('id', flat=True)
            )
            excluded_ids.update(ids)

    logger.info(f"[ContentSafety] 用户 {user.username} (age={age}): 排除 {len(excluded_ids)} 部不适宜电影")
    return excluded_ids


def get_content_safety_prompt(user) -> str:
    """
    为 LLM 生成内容安全约束 Prompt。
    未成年用户时注入约束，确保不推荐不适宜内容。
    """
    if not is_minor_user(user):
        return ""

    try:
        age = int(user.age)
    except (ValueError, TypeError):
        return ""

    return f"""

⚠️ 【内容安全约束 — 未成年用户保护】
该用户年龄为 {age} 岁（未成年），请严格遵守：
1. 严禁推荐包含血腥、暴力、恐怖、成人内容、性暗示的电影
2. 严禁推荐 R 级、NC-17 级或类似分级的影片
3. 推荐理由中不得描述暴力、恐怖或成人场景
4. 优先推荐正能量影片（冒险、友情、成长、科幻探索主题）
5. 如果用户主动要求看恐怖或暴力内容，礼貌拒绝并推荐替代方案

【替代推荐策略】：
- 恐怖片 → 推荐悬疑推理片（如《盗梦空间》）
- 暴力动作片 → 推荐冒险片（如《夺宝奇兵》）
- 成人剧情 → 推荐成长励志片（如《摔跤吧！爸爸》）
"""