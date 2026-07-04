"""
用户偏好记忆模块 (Memory Module)
================================================
实现多轮对话中的用户偏好追踪与记忆管理。

核心功能：
  1. 短期记忆：当前会话的槽位状态（genre/year/score/director...）
  2. 长期记忆：持久化到数据库的用户画像
  3. 槽位累加：多轮对话逐步细化偏好
  4. 追问检测：识别用户对上一轮结果的补充/修正

记忆结构：
  SlotState = {
      'genre': str,         # 类型偏好
      'year_min': int,      # 最早年份
      'score_min': float,   # 最低评分
      'director': str,      # 导演偏好
      'actor': str,         # 演员偏好
      'exclude_genre': str, # 排除类型
      'country': str,       # 地区
      'keyword': str,       # 关键词
  }
================================================
"""

import re
import json
import time
import logging
from collections import defaultdict
from django.core.cache import cache

logger = logging.getLogger('movie_agent')


# =============================================================
# 槽位定义
# =============================================================

SLOT_PATTERNS = {
    'genre': [
        (r'(科幻|科幻片|sci-?fi)', '科幻'),
        (r'(悬疑|悬疑片|推理|mystery)', '悬疑'),
        (r'(恐怖|恐怖片|horror)', '恐怖'),
        (r'(喜剧|喜剧片|搞笑|comedy)', '喜剧'),
        (r'(动作|动作片|action)', '动作'),
        (r'(爱情|爱情片|romance)', '爱情'),
        (r'(剧情|剧情片|drama)', '剧情'),
        (r'(动画|动画片|anime|animation)', '动画'),
        (r'(战争|战争片|war)', '战争'),
        (r'(犯罪|犯罪片|crime)', '犯罪'),
        (r'(奇幻|奇幻片|fantasy)', '奇幻'),
        (r'(纪录片|documentary)', '纪录片'),
        (r'(冒险|adventure)', '冒险'),
        (r'(传记|biography)', '传记'),
        (r'(家庭|family)', '家庭'),
        (r'(历史|history)', '历史'),
    ],
    'score_min': [
        (r'(评分|分数)[^\d]*(\d+(?:\.\d+)?)', None),  # 动态提取
        (r'(\d+(?:\.\d+)?)分以上', None),
        (r'高分', '8.0'),
        (r'(经典|好评)', '7.5'),
    ],
    'year_min': [
        # 通用"近N年"模式（动态计算年份）
        (r'近\s*(\d{1,2})\s*年', None),  # 动态提取：近20年、近5年等
        (r'最近\s*(\d{1,2})\s*年', None),  # 最近20年、最近5年等
        # 硬编码快捷模式
        (r'(近五年|最近五年|近5年|最近5年)', '2021'),
        (r'(近三年|最近三年|近3年|最近3年)', '2023'),
        (r'(近两年|最近两年|近2年|最近2年)', '2024'),
        (r'(近十年|近10年)', '2016'),
        (r'(2[0-9]{3})年?[以之]后', None),  # 动态提取
        (r'(2[0-9]{3})年?[以之]上', None),
        (r'(新|最近|最新)', '2020'),
        (r'(不要太老)', '2015'),
    ],
    'director': [
        (r'(导演[是为]?|导演[:：]?\s*)([\u4e00-\u9fff]{2,4})', None),  # 动态提取
        (r'([\u4e00-\u9fff]{2,4})导演', None),
        # 知名导演直接匹配
        (r'(诺兰|克里斯托弗·诺兰|nolan)', 'Christopher Nolan'),
        (r'(宫崎骏|miyazaki)', '宫崎骏'),
        (r'(斯皮尔伯格|spielberg)', 'Steven Spielberg'),
        (r'(昆汀|塔伦蒂诺|tarantino)', 'Quentin Tarantino'),
        (r'(大卫·芬奇|芬奇|david fincher)', 'David Fincher'),
    ],
    'actor': [
        (r'(主演[是为]?|主演[:：]?\s*)([\u4e00-\u9fff]{2,4})', None),
    ],
    'exclude_genre': [
        (r'不要(恐怖|恐怖片)', '恐怖'),
        (r'不要太(恐怖|吓人)', '恐怖'),
        (r'不想看(恐怖)', '恐怖'),
        (r'不要(动画|动画片)', '动画'),
    ],
    'keyword': [
        (r'(烧脑|悬疑推理)', '烧脑'),
        (r'(感人|催泪|温暖)', '温情'),
        (r'(硬核|硬科幻)', '硬科幻'),
        (r'(爽片|爽剧)', '爽片'),
        (r'(治愈|温馨)', '治愈'),
        (r'(深度|有深度)', '深度'),
    ],
}


# =============================================================
# 记忆管理器
# =============================================================

class MemoryManager:
    """
    多轮对话记忆管理器
    
    支持：
      - 短期槽位记忆（会话级，Redis缓存）
      - 长期画像记忆（数据库持久化）
      - 追问检测（识别补全/修正意图）
    """
    
    # 槽位有效期（秒）
    SLOT_TTL = 3600  # 1小时
    
    def __init__(self, user=None, session_id=None):
        """
        Args:
            user: Django User 对象
            session_id: 会话ID（用于匿名用户或缓存键）
        """
        self.user = user
        self.session_id = session_id or 'default'
        self._cache_key = f"agent_memory_{self.session_id}"
    
    def get_slots(self):
        """
        获取当前会话的槽位状态。
        
        Returns:
            Dict: 当前槽位字典
        """
        cached = cache.get(self._cache_key)
        if cached and isinstance(cached, dict):
            return cached
        return {}
    
    def update_slots(self, user_input):
        """
        根据用户输入提取并更新槽位。
        
        Args:
            user_input: 用户输入文本
            
        Returns:
            Dict: 更新后的槽位字典
        """
        slots = self.get_slots()
        newly_extracted = self.extract_slots(user_input)
        
        # 合并槽位（新值覆盖旧值，空值不覆盖）
        for key, value in newly_extracted.items():
            if value is not None and value != '':
                slots[key] = value
        
        # 检测否定指令（清除某槽位）
        if re.search(r'(不要|不要了|去掉|取消)', user_input):
            for key in ['genre', 'director', 'actor']:
                if key in newly_extracted and newly_extracted[key] is None:
                    slots.pop(key, None)
        
        # 保存到缓存
        cache.set(self._cache_key, slots, self.SLOT_TTL)
        return slots
    
    def clear_slots(self):
        """清除当前会话的所有槽位"""
        cache.delete(self._cache_key)
    
    def extract_slots(self, text):
        """
        从文本中提取槽位值。

        Args:
            text: 输入文本

        Returns:
            Dict: 提取到的槽位值（可能不完整）
        """
        from datetime import datetime
        text_lower = text.lower().strip()
        extracted = {}

        for slot_name, patterns in SLOT_PATTERNS.items():
            for pattern, fixed_value in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    if fixed_value is not None:
                        extracted[slot_name] = fixed_value
                    else:
                        # 动态提取值
                        groups = match.groups()
                        if groups:
                            # 特殊处理：year_min 的 "近N年" 模式
                            if slot_name == 'year_min' and len(groups) == 1 and groups[0] and groups[0].isdigit():
                                n_years = int(groups[0])
                                extracted[slot_name] = str(datetime.now().year - n_years)
                            else:
                                extracted[slot_name] = groups[-1] if groups else None
                    break  # 每个槽位只取第一个匹配

        # 数值类型转换
        if 'score_min' in extracted:
            try:
                extracted['score_min'] = float(extracted['score_min'])
            except (ValueError, TypeError):
                extracted.pop('score_min', None)

        if 'year_min' in extracted:
            try:
                extracted['year_min'] = int(extracted['year_min'])
            except (ValueError, TypeError):
                extracted.pop('year_min', None)

        return extracted
    
    # 🔥 增强版追问检测正则（覆盖更多口语表达）
    FOLLOWUP_PATTERN = re.compile(
        r'(再来|再给|还要|继续|换一批|换几个|别的|其他|不一样的|'
        r'更多|来点|来几部|多推荐|换个|类似的|相似的|'
        r'不要了|排除|去掉|太[老新短长]|评分[更再]高|'
        r'有没有.*类似的|还有吗|还有没有)', re.IGNORECASE
    )
    
    def is_followup(self, text):
        """
        检测是否为追问/补全指令。
        
        Returns:
            bool: 是否为追问
        """
        text_lower = text.lower().strip()
        slots = self.get_slots()
        
        # 🔥 只要有追问关键词就判定为追问（不要求槽位存在）
        # 因为 IntentClassifier 已经提前匹配，这里作为二次确认
        if self.FOLLOWUP_PATTERN.search(text_lower):
            return True
        
        # 兜底：有槽位 + 短文本 + 修正类关键词
        followup_patterns = [
            r'(不要|排除|去掉)',
            r'(太[老新短长])',
            r'(评分[更再]高)',
            r'(最好|希望|喜欢)',
        ]
        if slots and len(text_lower) < 20:
            for p in followup_patterns:
                if re.search(p, text_lower):
                    return True
        
        return False
    
    def slots_to_query_text(self, slots=None):
        """
        将槽位转换为自然语言查询文本（用于向量召回）。
        
        Args:
            slots: 槽位字典，默认使用当前槽位
            
        Returns:
            str: 自然语言查询文本
        """
        if slots is None:
            slots = self.get_slots()
        
        parts = []
        if slots.get('genre'):
            parts.append(f"{slots['genre']}类型电影")
        if slots.get('keyword'):
            parts.append(slots['keyword'])
        if slots.get('director'):
            parts.append(f"{slots['director']}导演")
        if slots.get('actor'):
            parts.append(f"{slots['actor']}主演")
        if slots.get('score_min'):
            parts.append(f"评分{slots['score_min']}以上")
        if slots.get('year_min'):
            parts.append(f"{slots['year_min']}年以后")
        if slots.get('exclude_genre'):
            parts.append(f"排除{slots['exclude_genre']}")
        
        return " ".join(parts) if parts else "优质电影推荐"
    
    def get_memory_summary(self, slots=None):
        """
        获取当前记忆的可视化摘要（论文展示用）。
        
        Returns:
            str: 可读的记忆摘要
        """
        if slots is None:
            slots = self.get_slots()
        
        if not slots:
            return "暂无用户偏好记忆"
        
        lines = ["[Memory State]"]
        for k, v in slots.items():
            if v is not None:
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
    
    def persist_to_profile(self):
        """
        将当前会话槽位持久化到 UserProfile 表（长期记忆）。
        """
        if not self.user:
            return
        
        try:
            from myapp.models_upgrade import UserProfile
            slots = self.get_slots()
            
            profile, created = UserProfile.objects.get_or_create(user=self.user)
            
            # 合并类型偏好到 genre_preferences (dict格式: {"科幻": 0.85, ...})
            genre_prefs = profile.genre_preferences or {}
            new_genre = slots.get('genre', '')
            if new_genre:
                genre_prefs[new_genre] = genre_prefs.get(new_genre, 0) + 1.0
                profile.genre_preferences = genre_prefs
            
            # 合并导演偏好
            director_prefs = profile.director_preferences or {}
            new_director = slots.get('director', '')
            if new_director:
                director_prefs[new_director] = director_prefs.get(new_director, 0) + 1.0
                profile.director_preferences = director_prefs
            
            # 更新画像文本（包含槽位摘要）
            slots_text = ", ".join(f"{k}={v}" for k, v in slots.items() if v)
            if slots_text:
                profile_text = profile.profile_text or ''
                if slots_text not in profile_text:
                    profile.profile_text = f"{profile_text}\n最近偏好: {slots_text}".strip()
            
            # 标记非冷启动
            if slots:
                profile.is_cold_start = False
            
            profile.save()
        except Exception as e:
            logger.error(f"[Memory] 持久化失败: {e}")
    
    # ===========================================================
    # 会话总结与画像补充（类似 Gemini/ChatGPT 记忆机制）
    # ===========================================================
    
    def summarize_and_update_profile(self, chat_history=None):
        """
        根据当前会话的对话历史，总结用户偏好并补充到长期画像。
        
        设计思路（类似 Gemini/ChatGPT 的记忆机制）：
          1. 获取当前会话的最近对话记录
          2. 从对话中提取偏好信号（规则+模式匹配）
          3. 与已有画像合并（去重+权重更新）
          4. 持久化到 UserProfile
        
        Args:
            chat_history: 对话历史列表 [{'role': 'user'/'ai', 'message': str}, ...]
        
        Returns:
            dict: 总结出的新偏好信息
        """
        if not self.user:
            return {}
        
        # 获取对话历史（从参数或数据库）
        if chat_history is None:
            chat_history = self._get_recent_chat_history()
        
        if not chat_history:
            return {}
        
        # 从对话历史中提取偏好信号
        extracted_preferences = self._extract_preferences_from_history(chat_history)
        
        # 合并到已有画像
        updated = self._merge_to_profile(extracted_preferences)
        
        # 同时将当前槽位也持久化
        self.persist_to_profile()
        
        return updated
    
    def _get_recent_chat_history(self, limit=20):
        """获取当前用户最近的对话历史"""
        try:
            from myapp.models import ChatHistory
            from django.utils import timezone
            from datetime import timedelta
            
            time_threshold = timezone.now() - timedelta(hours=24)
            history = ChatHistory.objects.filter(
                user=self.user,
                timestamp__gte=time_threshold
            ).order_by('timestamp')[:limit]
            
            return [{'role': h.role, 'message': h.message} for h in history]
        except Exception as e:
            logger.error(f"[Memory] 获取对话历史失败: {e}")
            return []
    
    def _extract_preferences_from_history(self, chat_history):
        """
        从对话历史中提取用户偏好信号。
        
        提取策略（规则驱动，零LLM依赖）：
          1. 类型偏好：统计用户提到的类型频率
          2. 排除偏好：提取"不要/不想看"等否定表达
          3. 导演/演员偏好：提取提到的人名
          4. 情感偏好：提取氛围关键词（轻松/压抑/烧脑等）
          5. 评分偏好：提取"高分/评分X以上"等
          6. 年份偏好：提取时间约束
          7. 锚点电影：统计用户提到的参考电影
        """
        preferences = {
            'genres': [],          # 喜欢的类型
            'exclude_genres': [],  # 不喜欢的类型
            'directors': [],       # 喜欢的导演
            'actors': [],          # 喜欢的演员
            'keywords': [],        # 偏好关键词
            'anchor_movies': [],   # 提及的参考电影
            'score_min': None,     # 最低评分要求
            'sentiment': [],       # 情感偏好（轻松/压抑等）
        }
        
        user_messages = [msg['message'] for msg in chat_history if msg['role'] == 'user']
        
        for msg in user_messages:
            msg_lower = msg.lower()
            
            # 1. 类型偏好提取
            genre_map = {
                '科幻': ['科幻', 'sci-fi'], '悬疑': ['悬疑', '推理'],
                '喜剧': ['喜剧', '搞笑'], '动作': ['动作'],
                '爱情': ['爱情', '浪漫'], '恐怖': ['恐怖', '惊悚'],
                '动画': ['动画'], '剧情': ['剧情'],
                '战争': ['战争'], '犯罪': ['犯罪'],
                '奇幻': ['奇幻', '魔幻'], '冒险': ['冒险'],
            }
            for genre, keywords in genre_map.items():
                # 检查是否是否定语境
                is_negative = bool(re.search(rf'不要.*?{keywords[0]}|不想.*?{keywords[0]}|排除.*?{keywords[0]}', msg_lower))
                if is_negative:
                    if genre not in preferences['exclude_genres']:
                        preferences['exclude_genres'].append(genre)
                else:
                    for kw in keywords:
                        if kw in msg_lower:
                            if genre not in preferences['genres']:
                                preferences['genres'].append(genre)
                            break
            
            # 2. 导演偏好提取
            director_patterns = [
                (r'诺兰|克里斯托弗·诺兰|nolan', 'Christopher Nolan'),
                (r'宫崎骏', '宫崎骏'), (r'昆汀', 'Quentin Tarantino'),
                (r'大卫·芬奇|david fincher', 'David Fincher'),
                (r'斯皮尔伯格', 'Steven Spielberg'),
                (r'王家卫', '王家卫'), (r'周星驰', '周星驰'),
            ]
            for pattern, name in director_patterns:
                if re.search(pattern, msg_lower) and name not in preferences['directors']:
                    preferences['directors'].append(name)
            
            # 3. 情感偏好提取
            sentiment_map = {
                '轻松': ['轻松', '愉快', '欢快', '温馨'],
                '治愈': ['治愈', '温暖', '暖心'],
                '烧脑': ['烧脑', '硬核', '深度'],
                '热血': ['热血', '燃', '刺激'],
                '压抑': ['压抑', '沉重', '悲伤'],
            }
            for sentiment, keywords in sentiment_map.items():
                # 排除否定语境
                is_negative = bool(re.search(rf'不要.*?{sentiment}|不想.*?{sentiment}', msg_lower))
                if not is_negative:
                    for kw in keywords:
                        if kw in msg_lower:
                            if sentiment not in preferences['sentiment']:
                                preferences['sentiment'].append(sentiment)
                            break
            
            # 4. 评分偏好提取
            score_match = re.search(r'(\d+(?:\.\d+)?)\s*分以上', msg_lower)
            if score_match:
                score = float(score_match.group(1))
                if preferences['score_min'] is None or score > preferences['score_min']:
                    preferences['score_min'] = score
            if '高分' in msg_lower or '经典' in msg_lower:
                if preferences['score_min'] is None:
                    preferences['score_min'] = 8.0
            
            # 5. 锚点电影提取
            anchor_match = re.findall(r'《([^》]+)》', msg)
            for movie_name in anchor_match:
                if movie_name not in preferences['anchor_movies']:
                    preferences['anchor_movies'].append(movie_name)
        
        return preferences
    
    def _merge_to_profile(self, preferences):
        """
        将提取的偏好合并到用户长期画像。
        
        合并策略（兼容 UserProfile 模型字段）：
          - genre_preferences: dict {"科幻": 1.0, "悬疑": 2.0, ...}
          - director_preferences: dict {"Christopher Nolan": 1.0, ...}
          - actor_preferences: dict {"演员名": 1.0, ...}
          - profile_text: 文本摘要
        """
        try:
            from myapp.models_upgrade import UserProfile
            
            profile, created = UserProfile.objects.get_or_create(user=self.user)
            
            # ── 合并类型偏好 ──
            genre_prefs = profile.genre_preferences or {}
            for genre in preferences.get('genres', []):
                if genre and genre not in preferences.get('exclude_genres', []):
                    genre_prefs[genre] = genre_prefs.get(genre, 0) + 1.0
            # 排除类型降低权重而非删除
            for genre in preferences.get('exclude_genres', []):
                if genre in genre_prefs:
                    genre_prefs[genre] = max(0, genre_prefs[genre] - 0.5)
            profile.genre_preferences = genre_prefs
            
            # ── 合并导演偏好 ──
            director_prefs = profile.director_preferences or {}
            for director in preferences.get('directors', []):
                if director:
                    director_prefs[director] = director_prefs.get(director, 0) + 1.0
            profile.director_preferences = director_prefs
            
            # ── 合并演员偏好 ──
            actor_prefs = profile.actor_preferences or {}
            for actor in preferences.get('actors', []):
                if actor:
                    actor_prefs[actor] = actor_prefs.get(actor, 0) + 1.0
            profile.actor_preferences = actor_prefs
            
            # ── 更新画像文本（含锚点电影、情感偏好）──
            summary_parts = []
            if preferences.get('anchor_movies'):
                summary_parts.append(f"参考电影: {'、'.join(preferences['anchor_movies'])}")
            if preferences.get('sentiment'):
                summary_parts.append(f"情感偏好: {'、'.join(preferences['sentiment'])}")
            if preferences.get('score_min'):
                summary_parts.append(f"最低评分: {preferences['score_min']}")
            
            if summary_parts:
                new_summary = "; ".join(summary_parts)
                existing_text = profile.profile_text or ''
                if new_summary not in existing_text:
                    profile.profile_text = f"{existing_text}\n{new_summary}".strip()
            
            # 标记非冷启动
            profile.is_cold_start = False
            profile.save()
            
            # 准备返回值
            all_genres = sorted(genre_prefs.keys(), key=lambda g: genre_prefs[g], reverse=True)
            all_directors = sorted(director_prefs.keys(), key=lambda d: director_prefs[d], reverse=True)
            
            return {
                'genres': all_genres,
                'directors': all_directors,
                'anchor_movies': preferences.get('anchor_movies', []),
                'sentiment': preferences.get('sentiment', []),
                'exclude_genres': preferences.get('exclude_genres', []),
                'score_min': preferences.get('score_min'),
            }
        except Exception as e:
            logger.error(f"[Memory] 画像合并失败: {e}")
            return {}
    
    def get_profile_summary(self):
        """
        获取用户长期画像的可读摘要（兼容 UserProfile 模型字段）。
        
        Returns:
            str: 画像摘要文本
        """
        if not self.user:
            return "匿名用户，暂无画像"
        
        try:
            from myapp.models_upgrade import UserProfile
            profile = UserProfile.objects.filter(user=self.user).first()
            if not profile:
                return "新用户，暂无画像"
            
            parts = []
            
            # 类型偏好 (dict: {"科幻": 2.0, "悬疑": 1.0, ...})
            genre_prefs = profile.genre_preferences or {}
            if genre_prefs:
                sorted_genres = sorted(genre_prefs.items(), key=lambda x: x[1], reverse=True)
                genre_str = "、".join(f"{g}({int(v)})" for g, v in sorted_genres[:5])
                parts.append(f"偏好类型: {genre_str}")
            
            # 导演偏好 (dict)
            director_prefs = profile.director_preferences or {}
            if director_prefs:
                sorted_directors = sorted(director_prefs.items(), key=lambda x: x[1], reverse=True)
                dir_str = "、".join(f"{d}({int(v)})" for d, v in sorted_directors[:3])
                parts.append(f"偏好导演: {dir_str}")
            
            # 演员偏好 (dict)
            actor_prefs = profile.actor_preferences or {}
            if actor_prefs:
                sorted_actors = sorted(actor_prefs.items(), key=lambda x: x[1], reverse=True)
                actor_str = "、".join(f"{a}({int(v)})" for a, v in sorted_actors[:3])
                parts.append(f"偏好演员: {actor_str}")
            
            # 画像文本（含锚点电影、情感偏好等）
            if profile.profile_text:
                parts.append(f"画像笔记: {profile.profile_text[:200]}")
            
            # 冷启动状态
            if profile.is_cold_start:
                parts.append("状态: 冷启动（画像数据不足）")
            else:
                parts.append("状态: 已建立画像")
            
            # 统计信息
            if profile.total_ratings > 0:
                parts.append(f"观影统计: {profile.total_ratings}部, 均分{profile.avg_score:.1f}")
            
            return "\n".join(parts) if parts else "画像数据较少"
        except Exception as e:
            return f"画像读取失败: {e}"
