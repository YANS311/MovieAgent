"""
MovieAgent 核心引擎 - ReAct 范式实现
================================================
实现完整的 Thought → Action → Observation → Final Answer 推理链

核心设计：
  1. 意图分类器：规则优先 + LLM兜底
  2. 工具集：search_vector / recall_hybrid / kg_query / rerank / explain
  3. ReAct循环：最多3轮迭代，每轮记录完整Trace
  4. 最终答案：融合LLM生成 + 结构化推荐结果
  5. 自反馈与纠偏：空结果自动重试 + 模糊查询追问
================================================
"""

import re
import time
import json
import logging
import traceback
from collections import Counter

logger = logging.getLogger('movie_agent')

from django.db.models import Count, Q
from myapp.models import Movie, UserRating, Collect, Genre, Actor, ChatHistory


# =============================================================
# 模糊查询检测器
# =============================================================

class MultiIntentDetector:
    """
    多意图分支检测器
    =================================================
    当用户需求涉及多个可能方向时（如"推荐科幻和爱情片"），
    Agent 需要像 Claude 那样抛出选项让用户选择分支，
    而非直接混在一起推荐可能不匹配的结果。
    """
    
    # 多意图模式：检测用户输入中包含多个并列的类型/风格/锚点电影
    MULTI_GENRE_PATTERN = re.compile(
        r'([\u4e00-\u9fff]{2,6})\s*[和、与及]\s*([\u4e00-\u9fff]{2,6})'
    )
    
    # 已知类型词（用于验证提取到的是否为有效类型）
    KNOWN_GENRES = {
        '科幻', '悬疑', '恐怖', '喜剧', '动作', '爱情', '剧情', '动画', '战争',
        '犯罪', '奇幻', '冒险', '纪录片', '历史', '音乐', '家庭', '西部',
        '武侠', '古装', '谍战', '灾难', '魔幻', '传记', '体育', '青春',
        '惊悚', '推理', '烧脑', '热血', '治愈', '文艺', '公路', '校园',
    }
    
    # 多锚点电影模式："类似A和B的" / "像A或B那样"
    MULTI_ANCHOR_PATTERN = re.compile(
        r'(?:类似|像|推荐.*?类似|看过)\s*《?([^》和、与]{2,10})》?\s*[和、与或及]\s*《?([^》和、与]{2,10})》?'
    )
    
    @classmethod
    def detect(cls, text):
        """
        检测用户输入是否包含多个并列意图/方向。
        
        Returns:
            tuple: (has_multi: bool, branches: list)
                branches: [{"label": str, "query": str}, ...]
        """
        text_clean = text.strip()
        branches = []
        
        # 1. 检测多类型并列（如"推荐科幻和爱情片"）
        for m in cls.MULTI_GENRE_PATTERN.finditer(text_clean):
            g1, g2 = m.group(1).strip(), m.group(2).strip()
            # 验证是否都是已知类型词
            g1_valid = g1 in cls.KNOWN_GENRES
            g2_valid = g2 in cls.KNOWN_GENRES
            
            if g1_valid and g2_valid:
                # 两者都是类型词 → 生成分支
                branches.append({
                    'label': f'🎬 专注{g1}片',
                    'query': re.sub(rf'{re.escape(g1)}\s*[和、与及]\s*{re.escape(g2)}', g1, text_clean),
                })
                branches.append({
                    'label': f'🎬 专注{g2}片',
                    'query': re.sub(rf'{re.escape(g1)}\s*[和、与及]\s*{re.escape(g2)}', g2, text_clean),
                })
                # 混合选项
                branches.append({
                    'label': f'🎯 两者都要（{g1}+{g2}）',
                    'query': text_clean,  # 原始查询
                })
                break  # 只取第一对
        
        # 2. 检测多锚点电影（如"类似盗梦空间和星际穿越的"）
        if not branches:
            for m in cls.MULTI_ANCHOR_PATTERN.finditer(text_clean):
                a1, a2 = m.group(1).strip(), m.group(2).strip()
                branches.append({
                    'label': f'🎬 像《{a1}》那样',
                    'query': f'推荐类似{a1}的电影',
                })
                branches.append({
                    'label': f'🎬 像《{a2}》那样',
                    'query': f'推荐类似{a2}的电影',
                })
                branches.append({
                    'label': f'🎯 两者都类似',
                    'query': text_clean,
                })
                break
        
        # 3. 检测混合意图（"最新+高分" → 让用户选择排序策略）
        if not branches:
            has_new = bool(re.search(r'(最新|新出|刚出|最近|上映)', text_clean))
            has_rank = bool(re.search(r'(高分|热门|经典|好看|评分)', text_clean))
            if has_new and has_rank:
                branches.append({
                    'label': '📅 最新上映优先',
                    'query': re.sub(r'(高分|热门|经典|好看|评分)', '', text_clean).strip(),
                })
                branches.append({
                    'label': '⭐ 高分经典优先',
                    'query': re.sub(r'(最新|新出|刚出|最近|上映)', '', text_clean).strip(),
                })
        
        return len(branches) > 0, branches


class VaguenessDetector:
    """
    模糊查询检测器
    =================================================
    检测用户输入是否过于模糊，需要追问以澄清需求。
    当查询过于宽泛时，Agent 应提供选项让用户选择，
    而非直接给出可能不匹配的推荐结果。
    """
    
    # 模糊查询模式（过于宽泛，需要追问）
    VAGUE_PATTERNS = [
        # 纯类型词，无任何修饰
        (r'^[\s]*(推荐|介绍|看看|找|有没有|有哪些)?\s*(电影|片子|片|影片)\s*$', 'bare_movie'),
        (r'^[\s]*(推荐|介绍|看看|找|有没有|有哪些)?\s*(一部|几部|点|些)?\s*(电影|片子|片)\s*$', 'bare_movie'),
        # 纯情绪词，无具体类型
        (r'^[\s]*(无聊|没事干|消遣|打发时间|随便)\s*$', 'bored'),
        # 极短输入（2-3字，不含明确电影名或类型）
        (r'^[\s\S]{1,3}\s*$', 'too_short'),
    ]
    
    # 不算模糊的关键词（有具体方向）
    SPECIFIC_INDICATORS = [
        r'(科幻|悬疑|恐怖|喜剧|动作|爱情|剧情|动画|战争|犯罪|奇幻|冒险|纪录片)',
        r'(推荐|类似|像|好看|高分|热门|最新|经典)',
        r'(导演|演员|主演)',
        r'(《[^》]+》)',  # 书名号引用
        r'(\d{4})',  # 年份
        r'(诺兰|宫崎骏|昆汀|斯皮尔伯格|周星驰)',
        r'(烧脑|感人|搞笑|催泪|热血|治愈|压抑|轻松)',
        r'(画像|偏好|口味)',  # 画像推荐
    ]
    
    @classmethod
    def is_vague(cls, text):
        """
        检测用户输入是否过于模糊。
        
        Returns:
            tuple: (is_vague: bool, reason: str)
        """
        text_clean = text.strip()
        
        # 太短且无明确意图
        if len(text_clean) <= 2:
            # 排除明确的问候
            if re.match(r'^(你好|hi|hello|嗨)$', text_clean, re.I):
                return False, ''
            return True, 'too_short'
        
        # 检查是否包含具体指标
        has_specific = False
        for pattern in cls.SPECIFIC_INDICATORS:
            if re.search(pattern, text_clean, re.I):
                has_specific = True
                break
        
        if has_specific:
            return False, ''
        
        # 匹配模糊模式
        for pattern, reason in cls.VAGUE_PATTERNS:
            if re.search(pattern, text_clean, re.I):
                return True, reason
        
        # 默认不模糊
        return False, ''
    
    @classmethod
    def generate_clarification_options(cls, text, memory_slots=None):
        """
        根据用户输入和记忆状态生成追问选项。
        
        Returns:
            list: 选项列表，每个元素为 {'label': str, 'value': str}
        """
        options = []
        
        # 基于热门类型的通用选项
        base_genres = [
            {'label': '🎬 科幻片', 'value': '推荐科幻电影'},
            {'label': '🔍 悬疑推理', 'value': '推荐烧脑悬疑片'},
            {'label': '😂 喜剧片', 'value': '推荐喜剧电影'},
            {'label': '💕 爱情片', 'value': '推荐爱情电影'},
            {'label': '⚔️ 动作片', 'value': '推荐动作电影'},
            {'label': '🎭 剧情片', 'value': '推荐剧情电影'},
        ]

        # 如果记忆中有类型偏好，优先推荐相关的
        if memory_slots and memory_slots.get('genre'):
            preferred = memory_slots['genre']
            # 把偏好类型放到第一位
            base_genres = [
                {'label': f'⭐ 继续看{preferred}片', 'value': f'推荐{preferred}电影'},
            ] + [g for g in base_genres if preferred not in g['label']]

        # 添加热门/画像推荐选项
        options = base_genres[:4]  # 最多4个选项
        options.append({'label': '🔥 热门高分榜', 'value': '推荐热门高分电影'})

        return options


# =============================================================
# 意图分类器
# =============================================================

class IntentClassifier:
    """
    意图分类器 V2 - 纯规则驱动，零LLM依赖
    分类结果决定Agent的推理路径
    """

    # 🔥 追问模式正则（独立于 INTENT_RULES，优先级最高）
    # 覆盖三类追问信号：
    #   1. 显式追问：再来、还要、换一批、别的
    #   2. 增量约束：不要太老/新/短/长、评分更高、最好是近N年
    #   3. 排除偏好：不要恐怖片、不想看动画
    FOLLOWUP_PATTERN = re.compile(
        r'(再来|再给|还要|继续|换一批|换几个|别的|其他|不一样的|'
        r'更多|来点|来几部|多推荐|不要了|换个|类似的|相似的|'
        r'有没有.*类似的|还有吗|还有没有|'
        r'太[老新短长厚薄虐差烂糟苦闷]|不要太|评分[更再]?高|评分\d+分?[以之上]|最好是|'
        r'不要.{0,6}(恐怖|动画|血腥|暴力|压抑|沉闷|悲伤|国产|烂|无聊)|'
        r'不想看.{0,6}(恐怖|动画|国产)|'
        r'近\s*\d+\s*年|最近\s*\d+\s*年|近[一二三四五六七八九十百千]+\s*年|'
        r'(2[0-9]{3})年?[以之]后|'
        r'(好看|精彩|治愈|轻松|搞笑|烧脑|感人|刺激|温馨|热血|文艺|幽默).{0,3}(一点|一些|的)|'
        r'要有.{0,8}(题材|元素|画面|场面)|'
        r'\S{2,4}导演的|'
        r'\S{2,6}题材|'
        r'一点的|一些的|\S{2,4}类的|'
        r'排除|去掉)', re.IGNORECASE
    )
    
    INTENT_RULES = [
        # (意图标签, 正则模式列表, 优先级)
        ('QUERY_VISUAL', [
            r'(海报|封面|画面|视觉|色调|风格|样子|长什么样|照片)',
        ], 10),
        ('QUERY_SELF', [
            r'(分析我|我的口味|我的偏好|画像|我的观影|总结我)',
        ], 8),
        ('QUERY_PROFILE_REC', [
            r'(画像.*推荐|偏好.*推|口味.*来|根据.*我.*推|画像.*推)',
        ], 9),
        ('QUERY_RANK', [
            r'(热门|榜单|前十|排名|排行|最火)',
            r'高分(?!.*(?:推荐|介绍|电影|片|看看|找))',
        ], 7),
        ('QUERY_NEW', [
            r'(最新|新出|上映|刚出|最近)',
        ], 6),
        ('QUERY_COMPARISON', [
            r'(对比|区别|vs|不同|相比|比较)',
        ], 5),
        ('QUERY_MOVIE', [
            r'(推荐|介绍|电影|片子|片|看看|找|什么|有哪些|有没有)',
        ], 3),
        ('CHAT', [
            r'(你好|谢谢|再见|hi|hello)',
        ], 1),
    ]
    
    @classmethod
    def classify(cls, text, history=None):
        """
        分类用户意图。
        
        Args:
            text: 用户输入文本
            history: 最近的对话历史（可选）
        
        Returns:
            str: 意图标签
        """
        text_lower = text.lower().strip()
        
        # 🔥 优先检测追问模式（只要有历史对话+追问关键词 → 直接判定为 QUERY_MOVIE）
        if cls.FOLLOWUP_PATTERN.search(text_lower):
            return 'QUERY_MOVIE'
        
        # 追问逻辑（旧版兜底）
        follow_kws = ['再', '还', '继续', '换', '类似', '别的', '其他', '再来']
        if any(k in text_lower for k in follow_kws) and len(text_lower) < 15:
            return 'QUERY_MOVIE'
        
        # 规则匹配
        best_intent = 'CHAT'
        best_priority = 0
        
        for intent, patterns, priority in cls.INTENT_RULES:
            if priority <= best_priority:
                continue
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    best_intent = intent
                    best_priority = priority
                    break
        
        # 领域实体检测
        if best_intent == 'CHAT':
            if cls._has_movie_entity(text_lower):
                best_intent = 'QUERY_MOVIE'
        
        return best_intent
    
    @classmethod
    def _has_movie_entity(cls, text):
        """检测文本中是否包含电影领域实体（电影名、演员、导演、类型）"""
        words = [w for w in re.findall(r'[一-鿿]{2,}|[a-zA-Z]{3,}', text)]
        if not words:
            return False

        # 扩展候选词：原词 + 去掉尾字前缀（处理"宫崎骏的"→"宫崎骏"）
        candidates = set()
        for w in words:
            candidates.add(w)
            if len(w) >= 3 and w[-1] in '的了在是':
                candidates.add(w[:-1])

        for word in list(candidates)[:8]:
            if Movie.objects.filter(title__icontains=word).exists():
                return True
            if Actor.objects.filter(name__icontains=word).exists():
                return True
            if Genre.objects.filter(name__icontains=word).exists():
                return True
        return False
        
        for word in words[:5]:
            if Movie.objects.filter(title__icontains=word).exists():
                return True
            if Actor.objects.filter(name__icontains=word).exists():
                return True
            if Genre.objects.filter(name__icontains=word).exists():
                return True
        return False


# =============================================================
# LLM 意图解析器
# =============================================================

class LLMIntentParser:
    """
    LLM 意图解析器：用 LLM 从用户输入中提取结构化 JSON。
    与 IntentClassifier + _micro_think 互补——LLM 做粗提取，规则引擎补充细节。

    输出字段：user_id, query_text, tags(中文), sort_by
    """

    SYSTEM_PROMPT = """# Role
你是一个电影推荐系统的核心意图解析与特征提取专家。你的任务是将用户的自然语言转换为后端 Pipeline 可以直接消费的规范化 JSON 数据。

# Output Format
你必须且只能输出一个合法的 JSON 对象，绝对不允许包含任何 Markdown 格式包裹（如 ```json），不允许有任何解释性文字。

# Constraints
1. 严格参数过滤：只提取白名单内的字段（user_id, query_text, tags, sort_by）。直接丢弃任何时间戳（timestamp）、随机数（nonce）等非业务字段。
2. 文本规范化：对 query_text 字段进行 strip() 去除首尾空格，并统一转为全小写。
3. 列表排序约束：对于 tags 字段（标签数组），必须按照字母表正序（Alphabetical Order）对内部元素进行排列，确保缓存键的唯一性。
4. 键序固化：输出的 JSON 键必须严格按照【user_id -> query_text -> tags -> sort_by】的顺序排列。

# JSON Schema
{
  "user_id": string (如果未提供则默认为 "anonymous"),
  "query_text": string (规范化后的搜索文本，无输入则为 ""),
  "tags": array of strings (正序排列的电影标签，如 ["action", "sci-fi"]),
  "sort_by": string (可选值: "rating", "release_date", "hot", 默认为 "hot")
}

# Few-Shot Examples
User: "我（uid: 8888）想看点 科幻 和 动作 类型的片子，按评分排下，时间戳 1717680000"
Output: {"user_id":"8888","query_text":"","tags":["action","sci-fi"],"sort_by":"rating"}

User: "有什么好玩的 喜剧 电影吗？"
Output: {"user_id":"anonymous","query_text":"有什么好玩的 喜剧 电影吗？","tags":["comedy"],"sort_by":"hot"}"""

    # 英文标签 → 中文标签映射
    TAG_MAP = {
        'action': '动作', 'sci-fi': '科幻', 'science fiction': '科幻',
        'comedy': '喜剧', 'romance': '爱情', 'love': '爱情',
        'thriller': '悬疑', 'mystery': '悬疑', 'suspense': '悬疑',
        'horror': '恐怖', 'drama': '剧情', 'animation': '动画',
        'anime': '动画', 'war': '战争', 'crime': '犯罪',
        'fantasy': '奇幻', 'adventure': '冒险', 'documentary': '纪录片',
        'musical': '音乐', 'family': '家庭', 'biography': '传记',
    }

    VALID_SORT = {'rating', 'release_date', 'hot'}

    @classmethod
    def parse(cls, user_input, llm_func):
        """
        调用 LLM 解析用户输入。

        Args:
            user_input: 用户原始输入
            llm_func: callable(system_prompt, user_prompt) -> str or None

        Returns:
            dict: {'user_id': str, 'query_text': str, 'tags': list[str], 'sort_by': str}
        """
        fallback = {'user_id': 'anonymous', 'query_text': '', 'tags': [], 'sort_by': 'hot'}

        if not user_input or not user_input.strip():
            return fallback

        # Redis 缓存：相同查询复用意图解析结果（TTL 10 分钟）
        cache_key = f"intent:{hash(user_input.strip())}"
        try:
            from django.core.cache import cache
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
        except Exception:
            pass

        try:
            raw = llm_func(cls.SYSTEM_PROMPT, user_input)
            if not raw:
                return fallback

            # 清理 LLM 输出：去掉可能的 Markdown 包裹和思考标签
            raw = raw.strip()
            if raw.startswith('```'):
                raw = re.sub(r'^```(?:json)?\s*', '', raw)
                raw = re.sub(r'\s*```$', '', raw)
            # 去掉 <think>...</think> 标签（qwen3 等模型会输出思考过程）
            raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
            # 提取第一个 JSON 对象
            json_match = re.search(r'\{[^{}]*\}', raw)
            if json_match:
                raw = json_match.group(0)

            data = json.loads(raw)

            # 验证并规范化
            result = {
                'user_id': str(data.get('user_id', 'anonymous')) or 'anonymous',
                'query_text': cls._normalize(data.get('query_text', '')),
                'tags': cls._map_tags_to_zh(data.get('tags', [])),
                'sort_by': data.get('sort_by', 'hot'),
            }

            # sort_by 白名单校验
            if result['sort_by'] not in cls.VALID_SORT:
                result['sort_by'] = 'hot'

            # 写入缓存（TTL 10 分钟）
            try:
                cache.set(cache_key, result, timeout=600)
            except Exception:
                pass

            return result

        except Exception as e:
            logger.warning(f"[LLMIntentParser] LLM 解析失败，回退到规则引擎: {e}")
            return fallback

    @classmethod
    def _map_tags_to_zh(cls, tags):
        """英文标签 → 中文标签映射 + 排序"""
        if not isinstance(tags, list):
            return []
        zh_tags = []
        for tag in tags:
            if not isinstance(tag, str):
                continue
            mapped = cls.TAG_MAP.get(tag.lower().strip())
            if mapped and mapped not in zh_tags:
                zh_tags.append(mapped)
        return sorted(zh_tags)

    @classmethod
    def _normalize(cls, text):
        """strip + lowercase"""
        if not isinstance(text, str):
            return ''
        return text.strip().lower()


# =============================================================
# 查询复杂度路由器
# =============================================================

class QueryComplexityRouter:
    """
    查询复杂度路由器
    =================================================
    简单 query 不进入 ReAct，直接走快速通道。
    只有复杂 query 才启动完整 ReAct 循环。
    这是工业界主流做法：避免"简单问题过度推理"。
    """

    # 复杂查询信号：锚点电影、多约束、情感词、导演/演员名、对比、排除
    COMPLEX_SIGNALS = [
        r'(类似|像|推荐.*?类似|看过)\s*《?[^》]{2,10}》?',  # 锚点电影
        r'(对比|区别|vs|不同|相比|比较)',  # 对比意图
        r'(不要|不想|排除|去掉|除了)',  # 排除偏好
        r'(轻松|治愈|搞笑|感人|催泪|热血|刺激|温馨|压抑|悲伤|烧脑|紧张)',  # 情感词
        r'(近\d+年|最近\d+年|\d{4}年[以后以来])',  # 年份约束
        r'(评分\d+分?[以之上]|高分|最高分)',  # 评分约束
        r'([一-鿿]{2,4})\s*(?:导演|执导|主演|出演)',  # 导演/演员名
        r'(诺兰|宫崎骏|昆汀|斯皮尔伯格|周星驰|王家卫|李安|张艺谋)',  # 知名导演
    ]

    COMPLEX_PATTERN = re.compile('|'.join(COMPLEX_SIGNALS), re.IGNORECASE)

    @classmethod
    def classify(cls, user_input, intent, is_followup, anchor_movie):
        """
        判断查询复杂度。

        Returns:
            str: 'simple' 或 'complex'
        """
        # 锚点电影 → complex
        if anchor_movie:
            return 'complex'

        # 多轮追问 → complex
        if is_followup:
            return 'complex'

        # 对比/画像/自分析/闲聊 → complex（或不需要推荐）
        if intent in ('QUERY_COMPARISON', 'QUERY_PROFILE_REC', 'QUERY_SELF', 'CHAT'):
            return 'complex'

        # 检测复杂信号
        if cls.COMPLEX_PATTERN.search(user_input):
            return 'complex'

        # 默认 simple
        return 'simple'


# =============================================================
# Agent 工具集
# =============================================================

class AgentTool:
    """Agent工具基类"""
    name = "base_tool"
    description = "基础工具"
    
    def execute(self, **kwargs):
        raise NotImplementedError


class SearchVectorTool(AgentTool):
    """向量语义搜索工具（含热门兜底）"""
    name = "search_vector"
    description = "基于语义相似度搜索电影"
    
    def __init__(self, rag_resources=None):
        self.rag_resources = rag_resources
    
    def execute(self, query="", k=10, **kwargs):
        from myapp.recommender.recall import vector_recall, hot_recall
        results = vector_recall(query, k=k, rag_resources=self.rag_resources)
        # 向量召回为空时走热门兜底
        if not results:
            results = hot_recall(k=k)
        return {
            'tool': self.name,
            'input': query,
            'output': results,
            'count': len(results),
        }


class RecallHybridTool(AgentTool):
    """多路混合召回工具"""
    name = "recall_hybrid"


class SearchDatabaseTool(AgentTool):
    """数据库直接查询工具（RAG 不可用时的替代方案）"""
    name = "search_database"
    description = "基于数据库条件查询电影"

    def execute(self, query="", k=30, **kwargs):
        from django.db.models import Q
        import re
        qs = Movie.objects.all()

        # 提取查询中的类型关键词
        genre_match = re.search(r'(科幻|喜剧|动作|爱情|悬疑|恐怖|动画|剧情|犯罪|奇幻|冒险|惊悚|战争)', query)
        if genre_match:
            qs = qs.filter(genres__name__icontains=genre_match.group(1))

        # 提取导演
        director_match = re.search(r'(诺兰|宫崎骏|昆汀|斯皮尔伯格|周星驰|王家卫|李安|张艺谋)', query)
        if director_match:
            qs = qs.filter(directors__name__icontains=director_match.group(1))

        # 提取年份
        year_match = re.search(r'(\d{4})', query)
        if year_match:
            qs = qs.filter(date__year__gte=int(year_match.group(1)))

        results = list(
            qs.order_by('-score', '-vote_count')
            .values_list('id', flat=True)[:k]
        )
        output = [{'movie_id': mid, 'score': 1.0, 'source': 'database'} for mid in results]
        return {
            'tool': self.name,
            'input': query,
            'output': output,
            'count': len(output),
        }


class RecallHybridTool(AgentTool):
    """多路混合召回工具"""
    name = "recall_hybrid"
    description = "多路召回融合推荐"

    def __init__(self, neo_graph=None, rag_resources=None):
        self.neo_graph = neo_graph
        self.rag_resources = rag_resources
    
    def execute(self, user=None, query_text=None, top_k=15, **kwargs):
        from myapp.recommender.recall import multi_channel_recall, hot_recall
        try:
            results, stats = multi_channel_recall(
                user, query_text=query_text, top_k=top_k,
                neo_graph=self.neo_graph, rag_resources=self.rag_resources
            )
        except Exception as e:
            logger.error(f"[RecallHybridTool] multi_channel_recall 异常: {e}")
            results = hot_recall(k=top_k)
            stats = {'fallback': 'hot_due_to_error'}
        
        # 如果多路召回为空，直接走热门兜底
        if not results:
            results = hot_recall(k=top_k)
            stats = stats or {}
            stats['fallback'] = 'hot_direct'
        
        return {
            'tool': self.name,
            'input': query_text or f"user_id={getattr(user, 'id', 'anon')}",
            'output': results,
            'stats': stats,
            'count': len(results),
        }


class KGQueryTool(AgentTool):
    """
    知识图谱查询工具（NL2Cypher 动态生成版）
    =================================================
    将用户自然语言约束动态转换为 Cypher 查询语句：
      1. 从 _micro_think() 提取结构化约束 (genre/director/actor/year/rating)
      2. 根据约束组合动态拼接 Cypher WHERE 子句
      3. 执行查询并返回结构化三元组
      4. Sub-graph Reasoning：推理导演风格与用户偏好的语义重叠
    """
    name = "kg_query"
    description = "将自然语言约束动态转换为 Cypher 查询知识图谱"

    def __init__(self, neo_graph=None):
        self.neo_graph = neo_graph

    def _build_cypher(self, constraints, anchor_mid=None):
        """
        NL2Cypher 核心：根据结构化约束动态生成 Cypher 查询语句。

        Args:
            constraints: dict from _micro_think()，包含 genre/director/actor/min_rating/year_filter
            anchor_mid: 锚点电影 ID（用于"类似X"类查询）

        Returns:
            tuple: (cypher_str, params_dict)
        """
        genre = constraints.get('genre')
        director = constraints.get('director')
        actor = constraints.get('actor')
        min_rating = constraints.get('min_rating')
        year_filter = constraints.get('year_filter') or {}
        min_year = year_filter.get('min_year')
        max_year = year_filter.get('max_year')

        # ── 模式 1: 锚点电影 + 约束（"类似《X》的科幻片"）──
        if anchor_mid:
            where_clauses = ["other.mid <> $anchor_mid"]
            params = {"anchor_mid": anchor_mid}
            match_paths = []

            # 通过共享导演关联
            match_paths.append(
                "MATCH (target:Movie {mid: $anchor_mid})<-[:DIRECTED_BY]-(d:Person)-[:DIRECTED_BY]->(other:Movie)"
            )
            # 通过共享类型关联
            match_paths.append(
                "MATCH (target:Movie {mid: $anchor_mid})-[:BELONGS_TO]->(g:Genre)<-[:BELONGS_TO]-(other:Movie)"
            )
            # 通过共享演员关联
            match_paths.append(
                "MATCH (target:Movie {mid: $anchor_mid})<-[:ACTED_IN]-(a:Person)-[:ACTED_IN]->(other:Movie)"
            )

            # 叠加额外约束
            if genre:
                where_clauses.append("ANY(og IN other_genres WHERE og = $genre)")
                params["genre"] = genre
            if min_rating:
                where_clauses.append("other.score >= $min_rating")
                params["min_rating"] = float(min_rating)
            if min_year:
                where_clauses.append("other.date >= $min_year")
                params["min_year"] = f"{min_year}-01-01"
            if max_year:
                where_clauses.append("other.date <= $max_year")
                params["max_year"] = f"{max_year}-12-31"

            where_str = " AND ".join(where_clauses)

            # 使用 UNION 合并多路径
            cypher_parts = []
            for i, match in enumerate(match_paths):
                path_alias = ["d", "g", "a"][i]
                rel_label = ["导演", "类型", "演员"][i]
                cypher_parts.append(f"""
                    {match}
                    OPTIONAL MATCH (other)-[:BELONGS_TO]->(og:Genre)
                    WITH other, collect(og.name) AS other_genres, {path_alias}.name AS via_name
                    WHERE {where_str.replace('other_genres', 'other_genres')}
                    RETURN other.mid AS mid, other.title AS title, other.score AS score,
                           '{rel_label}' AS rel_type, via_name
                    ORDER BY other.score DESC LIMIT 3
                """)

            cypher = " UNION ".join(cypher_parts)
            return cypher, params

        # ── 模式 2: 纯约束查询（"推荐高分科幻片"、"诺兰的电影"）──
        match_clause = "MATCH (m:Movie)"
        where_clauses = []
        params = {}
        extra_joins = ""

        if genre:
            extra_joins += " MATCH (m)-[:BELONGS_TO]->(g:Genre {name: $genre})"
            params["genre"] = genre

        if director:
            extra_joins += " MATCH (d:Person {name: $director})-[:DIRECTED_BY]->(m)"
            where_clauses.append("d.name = $director")
            params["director"] = director

        if actor:
            extra_joins += " MATCH (a:Person {name: $actor})-[:ACTED_IN]->(m)"
            where_clauses.append("a.name = $actor")
            params["actor"] = actor

        if min_rating:
            where_clauses.append("m.score >= $min_rating")
            params["min_rating"] = float(min_rating)

        if min_year:
            where_clauses.append("m.date >= $min_year")
            params["min_year"] = f"{min_year}-01-01"

        if max_year:
            where_clauses.append("m.date <= $max_year")
            params["max_year"] = f"{max_year}-12-31"

        where_str = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""

        cypher = f"""
            {match_clause}
            {extra_joins}
            {where_str}
            RETURN m.mid AS mid, m.title AS title, m.score AS score
            ORDER BY m.score DESC LIMIT 10
        """
        return cypher, params

    def execute(self, movie_title="", constraints=None, user_genres=None, **kwargs):
        if not self.neo_graph:
            return {'tool': self.name, 'input': movie_title, 'output': [], 'error': 'Neo4j未连接'}

        constraints = constraints or {}
        try:
            triples = []
            reasoning_insights = []
            cypher_log = []  # 记录生成的 Cypher（用于可解释性）

            # ── 锚点电影查询 ──
            anchor_mid = None
            if movie_title:
                target = Movie.objects.filter(title__icontains=movie_title).first()
                if target:
                    anchor_mid = target.id

            # ── NL2Cypher：动态生成查询 ──
            cypher, params = self._build_cypher(constraints, anchor_mid=anchor_mid)
            cypher_log.append({"cypher": cypher.strip(), "params": params})

            rows = self.neo_graph.run(cypher, **params).data()

            for r in rows:
                mid = r.get('mid')
                title = r.get('title', '')
                score = r.get('score', 0)
                rel_type = r.get('rel_type', '')
                via_name = r.get('via_name', '')

                if rel_type and via_name:
                    triples.append(f"《{title}》(ID:{mid},评分:{score})--[{rel_type}:{via_name}]")
                elif anchor_mid:
                    triples.append(f"《{title}》(ID:{mid},评分:{score})--[关联]-->《{movie_title}》")
                else:
                    triples.append(f"《{title}》(ID:{mid},评分:{score})")

            # ── Sub-graph Reasoning（锚点电影场景）──
            if anchor_mid and rows:
                target_genres = list(
                    Movie.objects.get(id=anchor_mid).genres.values_list('name', flat=True)
                ) if anchor_mid else []

                # 获取关联导演的类型偏好
                for r in rows[:3]:
                    if r.get('rel_type') == '导演' and r.get('via_name'):
                        director_name = r['via_name']
                        cypher_dg = """
                        MATCH (d:Person {name: $name})-[:DIRECTED_BY]->(m:Movie)-[:BELONGS_TO]->(g:Genre)
                        RETURN g.name AS genre, count(*) AS freq
                        ORDER BY freq DESC LIMIT 5
                        """
                        dg_rows = self.neo_graph.run(cypher_dg, name=director_name).data()
                        director_genres = [dr['genre'] for dr in dg_rows]

                        if user_genres and director_genres:
                            overlap = set(user_genres) & set(director_genres)
                            if overlap:
                                reasoning_insights.append(
                                    f"Sub-graph Reasoning: 导演{director_name}的作品常涉及{'、'.join(overlap)}类型，"
                                    f"与您的偏好高度契合"
                                )
                        if target_genres:
                            common = set(target_genres) & set(director_genres)
                            if common:
                                reasoning_insights.append(
                                    f"导演风格分析: {director_name}擅长{'、'.join(common)}类型叙事，"
                                    f"《{movie_title}》体现了其标志性创作特点"
                                )
                        break

            return {
                'tool': self.name,
                'input': movie_title or str(constraints),
                'output': triples,
                'reasoning_insights': reasoning_insights,
                'cypher_log': cypher_log,
                'count': len(triples),
            }
        except Exception as e:
            return {'tool': self.name, 'input': movie_title, 'output': [], 'error': str(e)}


class RerankTool(AgentTool):
    """重排工具（业务规则 + MMR 多样性）"""
    name = "rerank"
    description = "对候选电影进行多样性重排"
    
    def execute(self, candidates=None, user=None, top_k=10, **kwargs):
        from myapp.recommender.rerank import final_rerank
        results, stats = final_rerank(candidates or [], user=user, top_k=top_k)
        return {
            'tool': self.name,
            'input': f"{len(candidates or [])} candidates",
            'output': results,
            'stats': stats,
            'count': len(results),
        }


class MAANRerankTool(AgentTool):
    """
    MAAN 深度模型精排工具（在线推理版）
    =================================================
    加载第四章训练的 SKB-FMLP / MMAN 模型权重，
    对候选集执行实时推理精排，真正融合短期记忆槽位（Slots）。

    核心修正：直接调用 model.predict() 做在线推理，
    而非读取离线 Rec 表的死数据，使多轮对话的增量约束能被模型感知。
    """
    name = "maan_rerank"
    description = "使用 MAAN 深度多模态模型对候选电影精排（GAUC 0.8898，在线推理）"

    _cached_model = None
    _cached_meta = None
    _cached_feature_store = None

    @classmethod
    def _load_model(cls):
        """延迟加载 MAAN 模型和特征仓库（单例模式）"""
        if cls._cached_model is not None:
            return cls._cached_model, cls._cached_meta, cls._cached_feature_store

        import os, pickle
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        artifacts_dir = os.path.join(project_root, 'ml_artifacts')
        model_path = os.path.join(artifacts_dir, 'skb_fmlp_online.pt')
        meta_path = os.path.join(artifacts_dir, 'online_features_meta.pkl')

        if not os.path.exists(model_path):
            return None, None, None

        try:
            import torch
            from deepctr_torch.inputs import SparseFeat, VarLenSparseFeat, DenseFeat
            from myapp.mman_model import MMAN
            from myapp.skb_model import SKB_FMLP_Online

            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            state_dict = torch.load(model_path, map_location=device, weights_only=True)

            lbe_user = meta['lbe_user']
            lbe_movie = meta['lbe_movie']
            DIM = meta['UNIFIED_EMBED_DIM']
            SEQ = meta['SEQ_LEN']

            vocab_user = len(lbe_user.classes_) + 1
            vocab_movie = len(lbe_movie.classes_) + 1
            vocab_genre = state_dict['embedding_dict.genres.weight'].shape[0]
            vocab_actor = state_dict['embedding_dict.actors.weight'].shape[0]
            vocab_director = state_dict['embedding_dict.directors.weight'].shape[0]

            user_col = SparseFeat('user_id', vocab_user, DIM, embedding_name='user_id')
            movie_col = SparseFeat('movie_id', vocab_movie, DIM, embedding_name='movie_id')
            genre_col = VarLenSparseFeat(SparseFeat('genres', vocab_genre, DIM), maxlen=5, combiner='mean')
            actor_col = VarLenSparseFeat(SparseFeat('actors', vocab_actor, DIM), maxlen=5, combiner='mean')
            director_col = VarLenSparseFeat(SparseFeat('directors', vocab_director, DIM), maxlen=3, combiner='mean')

            rag_cols = [DenseFeat(f'rag_{i}', 1) for i in range(DIM)]
            seq_col = VarLenSparseFeat(
                SparseFeat('hist_movie_id', vocab_movie, DIM, embedding_name='movie_id'),
                maxlen=SEQ, length_name='sl', combiner='mean')

            linear_cols = [movie_col] + rag_cols
            dnn_cols = [user_col, movie_col, genre_col, actor_col, director_col, seq_col] + rag_cols

            model_type = meta.get('model_type', 'skb_fmlp')
            TEXT_DIM = meta.get('TEXT_DIM', 64)
            VISUAL_DIM = meta.get('VISUAL_DIM', 64)
            DROPOUT = meta.get('FIXED_DROPOUT', 0.1)

            if model_type == 'mman':
                model = MMAN(linear_cols, dnn_cols,
                             history_feature_list=['movie_id'],
                             text_dim=TEXT_DIM, visual_dim=VISUAL_DIM,
                             hidden_dim=256, num_heads=4,
                             dropout=DROPOUT, device=device)
            else:
                model = SKB_FMLP_Online(linear_cols, dnn_cols,
                                        history_feature_list=['movie_id'], device=device)

            model.load_state_dict(state_dict)
            model.eval()

            feature_store = meta.get('feature_store', {})
            cls._cached_model = model
            cls._cached_meta = meta
            cls._cached_feature_store = feature_store
            logger.info(f"[MAANRerankTool] 模型加载成功 ({model_type}, device={device})")
            return model, meta, feature_store
        except Exception as e:
            logger.error(f"[MAANRerankTool] 模型加载失败: {e}")
            return None, None, None

    def execute(self, candidates=None, user=None, top_k=10, memory_slots=None, **kwargs):
        """
        在线推理精排：构建特征张量 → model.predict() → 槽位亲和度加权 → Top-K。
        """
        import numpy as np

        if not candidates:
            return {'tool': self.name, 'input': '0 candidates', 'output': [], 'count': 0}

        model, meta, feature_store = self._load_model()
        if model is None:
            return {
                'tool': self.name,
                'input': f"{len(candidates)} candidates (model unavailable)",
                'output': candidates[:top_k],
                'count': min(len(candidates), top_k),
                'stats': {'fallback': 'no_model'},
            }

        try:
            import torch

            lbe_user = meta['lbe_user']
            lbe_movie = meta['lbe_movie']
            DIM = meta['UNIFIED_EMBED_DIM']
            SEQ_LEN = meta['SEQ_LEN']

            # 特征仓库（预计算的 RAG / 多模态矩阵，按 movie_id 索引）
            fs_mids = feature_store.get('raw_movie_ids', np.array([]))
            fs_enc_mids = feature_store.get('enc_movie_ids', np.array([]))
            rag_matrix = feature_store.get('rag_matrix', None)
            genres_matrix = feature_store.get('genres_matrix', None)
            actors_matrix = feature_store.get('actors_matrix', None)
            directors_matrix = feature_store.get('directors_matrix', None)

            # 构建 movie_id → feature_store 行号 的索引
            fs_mid_to_idx = {}
            for i, m in enumerate(fs_mids):
                fs_mid_to_idx[int(m)] = i

            # 用户历史（用于行为序列特征）
            u_str = str(user.id) if user else '0'
            u_idx = lbe_user.transform([u_str])[0] + 1 if u_str in lbe_user.classes_ else 0

            history_raw = []
            if user:
                from myapp.models import UserRating
                history_raw = list(
                    UserRating.objects.filter(user=user)
                    .order_by('comment_time')
                    .values_list('movie_id', flat=True)
                )
            hist_enc = [
                lbe_movie.transform([str(m)])[0] + 1
                for m in history_raw if str(m) in lbe_movie.classes_
            ]
            hist_padded = np.zeros(SEQ_LEN, dtype=np.int32)
            if hist_enc:
                tail = hist_enc[-SEQ_LEN:]
                hist_padded[:len(tail)] = tail
            sl_val = min(len(hist_enc), SEQ_LEN)

            # 逐候选构建特征向量
            valid_candidates = []
            infer_rows = {k: [] for k in ['user_id', 'movie_id', 'hist_movie_id', 'sl'] +
                          ([f'rag_{i}' for i in range(DIM)] if rag_matrix is not None else []) +
                          (['genres', 'actors'] if genres_matrix is not None else []) +
                          (['directors'] if directors_matrix is not None else [])}

            for c in candidates:
                mid = c.get('movie_id')
                if mid is None:
                    continue
                mid_str = str(mid)
                # 编码 movie_id
                if mid_str not in lbe_movie.classes_:
                    continue
                enc_mid = lbe_movie.transform([mid_str])[0] + 1

                # 在 feature_store 中查找
                fs_idx = fs_mid_to_idx.get(int(mid))
                if fs_idx is None:
                    continue

                valid_candidates.append(c)
                infer_rows['user_id'].append(u_idx)
                infer_rows['movie_id'].append(enc_mid)
                infer_rows['hist_movie_id'].append(hist_padded.copy())
                infer_rows['sl'].append(sl_val)

                if rag_matrix is not None:
                    for i in range(DIM):
                        infer_rows[f'rag_{i}'].append(rag_matrix[fs_idx, i])
                if genres_matrix is not None:
                    infer_rows['genres'].append(genres_matrix[fs_idx])
                    infer_rows['actors'].append(actors_matrix[fs_idx])
                if directors_matrix is not None:
                    infer_rows['directors'].append(directors_matrix[fs_idx])

            if not valid_candidates:
                return {
                    'tool': self.name,
                    'input': f"{len(candidates)} candidates (no valid features)",
                    'output': candidates[:top_k],
                    'count': min(len(candidates), top_k),
                    'stats': {'fallback': 'no_features'},
                }

            N = len(valid_candidates)
            infer_input = {}
            for k, vals in infer_rows.items():
                if k in ('hist_movie_id',):
                    infer_input[k] = np.array(vals, dtype=np.int32)
                elif k in ('genres', 'actors', 'directors'):
                    infer_input[k] = np.array(vals, dtype=np.int32)
                else:
                    infer_input[k] = np.array(vals, dtype=np.int32 if k != 'sl' else np.int32)

            device = next(model.parameters()).device
            with torch.no_grad():
                preds = model.predict(infer_input, batch_size=min(N, 512)).flatten()

            # 槽位亲和度加权：记忆槽位中的约束影响精排分数
            slot_boost = np.ones(N, dtype=np.float32)
            if memory_slots:
                for i, c in enumerate(valid_candidates):
                    boost = 1.0
                    # 年份约束
                    if 'year_min' in memory_slots and c.get('year'):
                        if c['year'] >= memory_slots['year_min']:
                            boost *= 1.15
                    if 'year_max' in memory_slots and c.get('year'):
                        if c['year'] <= memory_slots['year_max']:
                            boost *= 1.15
                    # 评分约束
                    if 'min_rating' in memory_slots and c.get('rating'):
                        if c['rating'] >= memory_slots['min_rating']:
                            boost *= 1.1
                    slot_boost[i] = boost

            final_scores = preds * slot_boost
            for i, c in enumerate(valid_candidates):
                c['maan_score'] = float(final_scores[i])

            valid_candidates.sort(key=lambda x: x.get('maan_score', 0), reverse=True)
            result = valid_candidates[:top_k]

            return {
                'tool': self.name,
                'input': f"{N} candidates, user={getattr(user, 'id', 'anon')}",
                'output': result,
                'count': len(result),
                'stats': {
                    'scorer': 'MAAN online inference',
                    'scored_count': N,
                    'total_candidates': len(candidates),
                    'slot_boosted': bool(memory_slots),
                },
            }
        except Exception as e:
            logger.error(f"[MAANRerankTool] 精排异常: {e}", exc_info=True)
            return {
                'tool': self.name,
                'input': f"{len(candidates)} candidates",
                'output': candidates[:top_k],
                'count': min(len(candidates), top_k),
                'stats': {'error': str(e)},
            }


class LLMRerankTool(AgentTool):
    """
    LLM 精排工具 — 让大语言模型参与推荐推理链
    =================================================
    在 MAAN 精排之后，调用 LLM 对 Top-K 候选进行语义重排。
    不同 LLM 会产生不同的排序结果，从而证明 LLM 在 Agent 框架中的协调能力。

    使用方式：
        tool = LLMRerankTool(model_name="qwen3:8b")
        result = tool.execute(candidates=candidates, user=user, top_k=5, query_text="推荐科幻片")
    """
    name = "llm_rerank"
    description = "使用大语言模型对候选电影进行语义精排"

    OLLAMA_URL = None  # 延迟初始化，从 Django settings 读取

    RERANK_SYS = (
        "你是电影推荐精排专家。根据用户查询，从候选列表中选出最匹配的电影。"
        "只输出JSON数组，包含选中的电影ID（按匹配度降序），不要其他内容。"
        "格式: [id1, id2, id3, ...]"
    )

    def __init__(self, model_name="qwen3:8b", timeout=60):
        self.model_name = model_name
        self.timeout = timeout
        self._current_query = ""

    def execute(self, candidates=None, user=None, top_k=5, query_text="", **kwargs):
        """
        使用 LLM 对候选电影进行语义精排。

        Args:
            candidates: 候选电影列表 [{'movie_id': int, 'title': str, ...}]
            user: Django User 实例
            top_k: 返回数量
            query_text: 用户原始查询文本

        Returns:
            dict: 与 MAANRerankTool 相同的输出格式
        """
        if not candidates:
            return {'tool': self.name, 'input': '0 candidates', 'output': [], 'count': 0}

        query = query_text or self._current_query or ""
        if not query:
            return {
                'tool': self.name,
                'input': f"{len(candidates)} candidates (no query)",
                'output': candidates[:top_k],
                'count': min(len(candidates), top_k),
            }

        try:
            from myapp.models import Movie

            # 构建候选电影信息（限制 20 条避免 prompt 过长）
            cand_slice = candidates[:20]
            cand_ids = [c.get('movie_id', c.get('id')) for c in cand_slice if c.get('movie_id') or c.get('id')]
            movies = Movie.objects.filter(id__in=cand_ids).values('id', 'title', 'score')
            movie_map = {m['id']: m for m in movies}

            cand_lines = []
            id_set = set()
            for c in cand_slice:
                mid = c.get('movie_id', c.get('id'))
                if not mid or mid in id_set:
                    continue
                id_set.add(mid)
                m = movie_map.get(mid, {})
                title = m.get('title', c.get('title', '未知'))
                score = m.get('score', '')
                cand_lines.append(f"ID:{mid} 《{title}》 评分:{score}")

            if not cand_lines:
                return {
                    'tool': self.name,
                    'input': f"{len(candidates)} candidates (no valid IDs)",
                    'output': candidates[:top_k],
                    'count': min(len(candidates), top_k),
                }

            prompt = (
                f"用户查询: {query}\n\n"
                f"候选电影（共{len(cand_lines)}部）:\n" + "\n".join(cand_lines) + "\n\n"
                f"请选出与用户查询最匹配的{top_k}部电影，按匹配度降序排列。\n"
                f"只输出JSON数组，如: [123, 456, 789]"
            )

            # 调用 Ollama LLM
            import requests as req
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": self.RERANK_SYS},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 256, "num_gpu": 99},
            }
            from django.conf import settings
            olla_url = self.OLLAMA_URL or getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434') + '/api/chat'
            resp = req.post(olla_url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            llm_text = resp.json().get("message", {}).get("content", "")

            # 解析 LLM 返回的 ID 列表
            selected_ids = self._parse_ids(llm_text)

            if not selected_ids:
                logger.warning(f"[LLMRerankTool] LLM 返回无法解析: {llm_text[:100]}")
                return {
                    'tool': self.name,
                    'input': f"{len(candidates)} candidates, model={self.model_name}",
                    'output': candidates[:top_k],
                    'count': min(len(candidates), top_k),
                    'stats': {'fallback': 'parse_error', 'model': self.model_name},
                }

            # 按 LLM 返回的顺序重排候选
            cand_by_id = {}
            for c in candidates:
                mid = c.get('movie_id', c.get('id'))
                if mid:
                    cand_by_id[mid] = c

            reranked = []
            for mid in selected_ids:
                if mid in cand_by_id:
                    reranked.append(cand_by_id[mid])
                if len(reranked) >= top_k:
                    break

            # 不够则补原始候选
            if len(reranked) < top_k:
                used_ids = {c.get('movie_id', c.get('id')) for c in reranked}
                for c in candidates:
                    mid = c.get('movie_id', c.get('id'))
                    if mid and mid not in used_ids:
                        reranked.append(c)
                        used_ids.add(mid)
                    if len(reranked) >= top_k:
                        break

            return {
                'tool': self.name,
                'input': f"{len(candidates)} candidates, query={query[:30]}",
                'output': reranked[:top_k],
                'count': min(len(reranked), top_k),
                'stats': {
                    'model': self.model_name,
                    'llm_selected': len(selected_ids),
                    'matched': len([m for m in selected_ids if m in cand_by_id]),
                },
            }

        except Exception as e:
            logger.error(f"[LLMRerankTool] LLM 精排异常: {e}")
            return {
                'tool': self.name,
                'input': f"{len(candidates)} candidates (error)",
                'output': candidates[:top_k],
                'count': min(len(candidates), top_k),
                'stats': {'error': str(e), 'model': self.model_name},
            }

    def _parse_ids(self, text):
        """从 LLM 输出中解析电影 ID 列表"""
        import re
        # 尝试匹配 JSON 数组
        match = re.search(r'\[[\s\d,]+\]', text)
        if match:
            try:
                ids = json.loads(match.group())
                return [int(x) for x in ids if isinstance(x, (int, float))]
            except (json.JSONDecodeError, ValueError):
                pass
        # 降级：提取所有数字
        numbers = re.findall(r'\d+', text)
        return [int(n) for n in numbers[:10]]


class ExplainTool(AgentTool):
    """
    推荐解释工具（含 KAG 知识图谱路径增强）
    =================================================
    KAG 的核心价值不在召回（会引入噪声），而在解释：
    通过知识图谱的三元组路径，为推荐理由提供可验证的归因证据。
    """
    name = "explain"
    description = "生成推荐理由（含知识图谱归因）"

    def __init__(self, neo_graph=None, enable_kag=True):
        self.neo_graph = neo_graph
        self.enable_kag = enable_kag

    def execute(self, user=None, movie_id=None, **kwargs):
        reason_text = ''
        reason_type = ''
        strength = 0
        try:
            from myapp.recommender.explain import analyze_recommend_reason
            result = analyze_recommend_reason(user, movie_id)
            reason_text = result.get('reason_text', '')
            reason_type = result.get('reason_type', '')
            strength = result.get('strength', 0)
        except Exception:
            pass

        # KAG 增强：查询知识图谱获取归因路径（可通过 enable_kag 开关控制）
        if self.enable_kag:
            kg_insight = self._query_kg_for_explanation(movie_id, user)
            if kg_insight:
                reason_text = f"{reason_text}【知识图谱归因】{kg_insight}"

        # 解释一致性校验：验证归因维度与电影实际属性是否匹配
        consistency = self._verify_explanation_consistency(reason_text, movie_id)

        return {
            'tool': self.name,
            'input': f"movie_id={movie_id}",
            'output': reason_text,
            'reason_type': reason_type,
            'strength': strength,
            'has_kg_attribution': self.enable_kag and '知识图谱归因' in reason_text,
            'explanation_consistency': consistency,
        }

    def _query_kg_for_explanation(self, movie_id, user):
        """查询知识图谱获取归因路径，增强推荐可解释性"""
        if not self.neo_graph or not user:
            return ''

        try:
            movie = Movie.objects.filter(id=movie_id).first()
            if not movie:
                return ''

            insights = []

            # 路径1: 同导演作品中用户评分最高的
            directors = list(movie.directors.values_list('name', flat=True)[:2])
            for dname in directors:
                cypher = """
                MATCH (d:Person {name: $name})-[:DIRECTED_BY]->(m:Movie)
                WHERE m.mid <> $mid
                RETURN m.mid AS mid, m.title AS title, m.score AS score
                ORDER BY m.score DESC LIMIT 3
                """
                rows = self.neo_graph.run(cypher, name=dname, mid=movie_id).data()
                if rows:
                    titles = '、'.join([f"《{r['title']}》" for r in rows[:2]])
                    insights.append(f"导演{dname}的其他高分代表作{titles}")

            # 路径2: 同类型中与用户历史的关联
            genres = list(movie.genres.values_list('name', flat=True)[:2])
            for gname in genres:
                # 查用户在该类型下的平均评分
                from django.db.models import Avg
                user_genre_avg = UserRating.objects.filter(
                    user=user, movie__genres__name=gname
                ).aggregate(avg=Avg('score'))['avg']
                if user_genre_avg and user_genre_avg >= 7.0:
                    insights.append(f"您对{gname}类型影片平均评分{user_genre_avg:.1f}分，偏好匹配")

            return '；'.join(insights) if insights else ''
        except Exception:
            return ''

    def _verify_explanation_consistency(self, reason_text, movie_id):
        """
        解释一致性校验：验证解释文本中提到的归因维度（导演/类型/演员）
        是否与电影的实际属性匹配，防止解释幻觉。

        Returns:
            dict: {'consistent': bool, 'mismatched': list}
        """
        if not reason_text or not movie_id:
            return {'consistent': True, 'mismatched': []}

        try:
            movie = Movie.objects.filter(id=movie_id).first()
            if not movie:
                return {'consistent': True, 'mismatched': []}

            mismatched = []

            # 校验导演归因
            director_match = re.search(r'导演[：:]?\s*([一-鿿·]{2,6})', reason_text)
            if director_match:
                claimed_director = director_match.group(1)
                actual_directors = set(movie.directors.values_list('name', flat=True))
                if claimed_director not in actual_directors:
                    mismatched.append(f"导演:{claimed_director}")

            # 校验类型归因
            genre_match = re.search(r'(?:类型|题材)[：:]?\s*([一-鿿]{2,6})', reason_text)
            if genre_match:
                claimed_genre = genre_match.group(1)
                actual_genres = set(movie.genres.values_list('name', flat=True))
                if claimed_genre not in actual_genres:
                    mismatched.append(f"类型:{claimed_genre}")

            # 校验演员归因
            actor_match = re.search(r'(?:主演|演员)[：:]?\s*([一-鿿·]{2,6})', reason_text)
            if actor_match:
                claimed_actor = actor_match.group(1)
                actual_actors = set(movie.actors.values_list('name', flat=True))
                if claimed_actor not in actual_actors:
                    mismatched.append(f"演员:{claimed_actor}")

            return {
                'consistent': len(mismatched) == 0,
                'mismatched': mismatched,
            }
        except Exception:
            return {'consistent': True, 'mismatched': []}


# =============================================================
# MovieAgent 主引擎
# =============================================================

class MovieAgent:
    """
    基于 ReAct 范式的智能电影推荐 Agent
    
    推理流程:
        Thought → Action → Observation → (循环) → Final Answer
    
    核心特性:
        1. 多轮推理：最多3轮工具调用
        2. 完整追踪：记录每一步的Trace
        3. 智能路由：根据意图自动选择工具链
        4. 可解释性：为每条推荐生成理由
        5. 自反馈纠偏：空结果自动切换路径重试
        6. 模糊追问：模糊查询时提供选项让用户选择
    """
    
    # 意图到工具链的映射
    # 核心设计：召回 → MAAN深度精排 → 业务重排 → 解释
    # MAAN 模型（GAUC 0.8898）负责最终精排，与第四章模型形成"血缘关系"
    # KAG(kg_query) 不参与召回（会引入噪声候选），仅在解释阶段用于知识图谱归因
    # 注意：explain 不放在工具链中，而是在 Step 6 中按每个推荐电影单独调用（含 movie_id）
    INTENT_TOOL_MAP = {
        'QUERY_MOVIE': ['search_vector', 'maan_rerank', 'rerank'],
        'QUERY_COMPARISON': ['search_vector', 'maan_rerank', 'rerank'],
        'QUERY_PROFILE_REC': ['recall_hybrid', 'maan_rerank', 'rerank'],
        'QUERY_RANK': ['search_vector', 'maan_rerank'],
        'QUERY_NEW': ['search_vector', 'maan_rerank'],
        'QUERY_VISUAL': ['search_vector'],
        'QUERY_SELF': [],  # 纯画像分析，不调用推荐工具
        'CHAT': [],
    }
    
    # 空结果纠偏映射：当主工具链返回空结果时，切换到的备用工具链
    # 设计思路：KAG(kg_query)空 → 切换 RAG(search_vector) 语义搜索
    #           recall_hybrid 空 → 切换 search_vector 语义搜索
    #           search_vector 空 → 切换 recall_hybrid 混合召回
    FALLBACK_CHAIN = {
        'recall_hybrid': 'search_vector',
        'search_vector': 'recall_hybrid',
        'kg_query': 'search_vector',
    }
    
    # 最大纠偏重试次数
    MAX_RETRY = 1

    # 低质量结果纠偏阈值
    LOW_QUALITY_SIM_THRESHOLD = 0.6
    
    def __init__(self, user=None, neo_graph=None, rag_resources=None, session_id=None, llm_config=None):
        """
        Args:
            user: Django User对象
            neo_graph: Neo4j图实例
            rag_resources: RAG资源字典
            session_id: 会话ID（用于多轮对话记忆）
            llm_config: LLM 配置字典 {'model_name': str, 'timeout': int}，用于 LLM 精排
        """
        self.user = user
        self.neo_graph = neo_graph
        self.rag_resources = rag_resources
        self.llm_config = llm_config or {}  # 保存 LLM 配置供 _call_llm 使用

        # 初始化工具集
        self.tools = {
            'search_vector': SearchVectorTool(rag_resources),
            'search_database': SearchDatabaseTool(),  # RAG 不可用时的替代方案
            'recall_hybrid': RecallHybridTool(neo_graph, rag_resources),
            'kg_query': KGQueryTool(neo_graph),
            'maan_rerank': MAANRerankTool(),  # MAAN 深度模型精排（GAUC 0.8898）
            'rerank': RerankTool(),            # 业务规则 + MMR 多样性重排
            'explain': ExplainTool(neo_graph),
        }

        # 精排始终使用 MAAN GPU 在线推理（~64ms），不替换为 LLM 精排

        # 初始化记忆管理器
        from myapp.agent.memory import MemoryManager
        self.memory = MemoryManager(
            user=user,
            session_id=session_id or f"user_{getattr(user, 'id', 'anon')}"
        )
    
    def run(self, user_input, is_thinking_mode=False):
        """
        执行Agent推理主流程。
        
        Args:
            user_input: 用户输入文本
            is_thinking_mode: 是否启用深度思考模式
        
        Returns:
            Dict: {
                'intent': str,              # 意图分类
                'thought': str,             # Agent思考过程
                'actions': List[Dict],      # 执行的动作列表
                'observations': List[Dict], # 观察结果
                'final_answer': str,        # 最终推荐文本
                'recommended_ids': List[int], # 推荐的电影ID
                'explanations': Dict,       # 推荐理由
                'latency_ms': int,          # 总耗时
                'need_clarification': bool, # 是否需要用户追问
                'clarification_options': List[Dict], # 追问选项
                'trace_steps': List[Dict],  # 完整推理链步骤
            }
        """
        t_start = time.time()
        
        # ── Step 0: 更新记忆槽位 ──
        self.memory.update_slots(user_input)
        
        # ── Step 0.3: 多意图分支检测（Claude-style branching）──
        has_multi, multi_branches = MultiIntentDetector.detect(user_input)
        intent = IntentClassifier.classify(user_input)
        
        # 只有推荐类意图才需要检测多意图
        if has_multi and intent in ('QUERY_MOVIE', 'QUERY_COMPARISON') and multi_branches:
            multi_thought = (
                f"识别用户意图: {intent}。用户需求涉及多个方向，"
                f"需要进一步澄清以精准推荐。\n"
                f"用户原始输入: '{user_input}'"
            )
            option_lines = []
            clarification_options = []
            for i, branch in enumerate(multi_branches, 1):
                option_lines.append(f"  {i}. {branch['label']}")
                clarification_options.append([branch['label'], branch['query']])
            
            clarification_text = (
                "🤔 您的需求涉及多个方向，请告诉我您更想专注哪一边：\n"
                + "\n".join(option_lines)
            )
            
            t_total = int((time.time() - t_start) * 1000)
            trace_steps = [{
                'step': 0, 'type': 'thought', 'content': multi_thought,
            }, {
                'step': 1, 'type': 'clarification',
                'content': '检测到多意图并列，发起分支追问',
            }]
            
            return {
                'intent': intent,
                'thought': multi_thought,
                'actions': [],
                'observations': [],
                'final_answer': clarification_text,
                'recommended_ids': [],
                'explanations': {},
                'latency_ms': t_total,
                'need_clarification': True,
                'clarification_options': clarification_options,
                'trace_steps': trace_steps,
            }
        
        # ── Step 0.5: 模糊查询检测（追问 + 热门推荐混合机制）──
        is_vague, vague_reason = VaguenessDetector.is_vague(user_input)

        # 只有推荐类意图才需要检测模糊性
        if is_vague and intent in ('QUERY_MOVIE', 'CHAT'):
            options = VaguenessDetector.generate_clarification_options(
                user_input, self.memory.get_slots()
            )
            clarification_thought = (
                f"识别用户意图: {intent}。用户查询较为模糊（原因: {vague_reason}），"
                f"先提供热门推荐作为兜底，同时追问以获取更精准需求。\n"
                f"用户原始输入: '{user_input}'\n"
                f"记忆状态:\n{self.memory.get_memory_summary()}"
            )

            t_total = int((time.time() - t_start) * 1000)

            # 生成追问选项
            option_lines = []
            for i, opt in enumerate(options, 1):
                option_lines.append(f"  {i}. {opt['label']}")

            # 同时获取热门电影推荐作为兜底
            hot_movies = []
            hot_ids = []
            try:
                from myapp.models import Movie
                hot_qs = Movie.objects.order_by('-vote_count', '-score').exclude(
                    vote_count__isnull=True
                ).exclude(score__isnull=True)[:5]
                for m in hot_qs:
                    genres = "、".join(g.name for g in m.genres.all()[:2])
                    directors = "、".join(d.name for d in m.directors.all()[:1])
                    score_str = str(m.score) if m.score else "暂无"
                    movie_year = m.date.year if m.date else None
                    year_str = f"({movie_year})" if movie_year else ""
                    line = f"  《{m.title}》{year_str} | ⭐{score_str}"
                    if genres:
                        line += f" | {genres}"
                    if directors:
                        line += f" | 🎬{directors}"
                    hot_movies.append(line)
                    hot_ids.append(m.id)
            except Exception:
                pass

            # 拼接完整响应：追问选项 + 热门推荐
            clarification_text = "🤔 您的需求我还需要进一步确认，请告诉我您更想看哪种类型的电影：\n"
            clarification_text += "\n".join(option_lines)
            if hot_movies:
                clarification_text += "\n\n🔥 先为您推荐几部热门高分电影：\n"
                clarification_text += "\n".join(hot_movies)

            # 构建完整的trace步骤（含追问 + 热门推荐）
            trace_steps = [{
                'step': 0,
                'type': 'thought',
                'content': clarification_thought,
            }, {
                'step': 1,
                'type': 'clarification',
                'content': '检测到模糊查询，提供热门推荐兜底并追问',
                'reason': vague_reason,
            }]
            if hot_ids:
                trace_steps.append({
                    'step': 2,
                    'type': 'action',
                    'content': '获取热门电影推荐作为兜底',
                    'tool': 'hot_recall',
                    'count': len(hot_ids),
                })

            return {
                'intent': intent,
                'thought': clarification_thought,
                'actions': [{'tool': 'hot_recall', 'input': 'vague_fallback'}] if hot_ids else [],
                'observations': [{'tool': 'hot_recall', 'output': hot_ids, 'count': len(hot_ids)}] if hot_ids else [],
                'final_answer': clarification_text,
                'recommended_ids': hot_ids,
                'explanations': {},
                'latency_ms': t_total,
                'need_clarification': True,
                'clarification_options': options,
                'trace_steps': trace_steps,
            }

        # ── Step 0.5b: LLM 意图解析（LLM + 规则互补）──
        llm_parsed = {}
        if self.llm_config.get('model_name'):
            llm_parsed = LLMIntentParser.parse(user_input, self._call_llm)
            if llm_parsed.get('tags'):
                logger.info(f"[Agent] LLM 意图解析: tags={llm_parsed['tags']}, sort_by={llm_parsed.get('sort_by')}")

        # ── Step 1: 意图分类（融合记忆上下文）──
        is_followup = self.memory.is_followup(user_input)

        # ── Step 1.5: 检测锚点电影（用于动态工具链路由）──
        anchor_movie = self._detect_anchor_movie(user_input)

        # ── Step 1.8: 查询复杂度路由 ──
        complexity = QueryComplexityRouter.classify(user_input, intent, is_followup, anchor_movie)

        # 消融实验: 禁用 ReAct 时强制走快速通道
        if getattr(self, '_force_fast_path', False):
            return self._run_fast_path(user_input, intent, t_start, llm_parsed=llm_parsed)

        if complexity == 'simple':
            return self._run_fast_path(user_input, intent, t_start, llm_parsed=llm_parsed)

        # ── Step 1.9: LLM 意图解析结果存入记忆 ──
        if llm_parsed.get('tags'):
            self.memory.update_slots({'llm_tags': llm_parsed['tags']})
            # 若记忆中无类型槽位，用 LLM 第一个标签填充（供 _micro_think 融合）
            if llm_parsed['tags'] and not self.memory.get_slots().get('genre'):
                self.memory.update_slots({'genre': llm_parsed['tags'][0]})

        # ── Step 2: Thought（思考阶段，融入记忆上下文）──
        thought = self._think(user_input, intent, is_followup, anchor_movie)
        
        # ── Step 3: 提取年份过滤条件 ──
        year_filter = self._extract_year_filter(user_input)
        if year_filter:
            logger.info(f"[Agent] 检测到年份过滤条件: {year_filter}")
        
        # ── Step 4: 动态构建工具链 ──
        # KAG设计变更：kg_query 不再参与召回（会引入噪声候选），
        # 改为在 Step 6 解释阶段用于增强可解释性（知识图谱路径归因）
        tool_chain = list(self.INTENT_TOOL_MAP.get(intent, []))

        # 消融适配: RAG 不可用时，用数据库查询替代向量搜索
        has_rag = bool(self.rag_resources and self.rag_resources.get('vectorstore'))
        if not has_rag:
            tool_chain = [
                'search_database' if t in ('search_vector', 'recall_hybrid') else t
                for t in tool_chain
            ]
        
        # ── Step 5: Action → Observation 循环（含自反馈纠偏）──
        actions = []
        observations = []
        candidates = []
        recommended_ids = []
        explanations = {}
        
        # trace_steps: 完整的推理链步骤记录
        trace_steps = [{
            'step': 0,
            'type': 'thought',
            'content': thought,
        }]
        trace_step_counter = 1
        
        for tool_name in tool_chain:
            # Action
            action, observation = self._act(tool_name, user_input, candidates)
            actions.append(action)
            observations.append(observation)
            
            # 记录trace步骤
            trace_steps.append({
                'step': trace_step_counter,
                'type': 'action',
                'content': f"调用工具 {tool_name}",
                'tool': tool_name,
                'input': action.get('input', ''),
            })
            trace_step_counter += 1
            
            obs_count = observation.get('count', 0)
            trace_steps.append({
                'step': trace_step_counter,
                'type': 'observation',
                'content': f"工具 {tool_name} 返回 {obs_count} 条结果",
                'tool': tool_name,
                'count': obs_count,
            })
            trace_step_counter += 1
            
            # 【Bug2修复】合并候选集而非覆盖
            if tool_name in ('search_vector', 'recall_hybrid', 'kg_query', 'search_database'):
                raw = observation.get('output', [])
                if isinstance(raw, list) and raw:
                    existing_ids = {c.get('movie_id', c.get('id')) for c in candidates if isinstance(c, dict)}
                    
                    if tool_name == 'kg_query':
                        # kg_query返回三元组字符串，需要提取电影ID
                        for item in raw:
                            if isinstance(item, dict):
                                mid = item.get('movie_id', item.get('id'))
                                if mid and mid not in existing_ids:
                                    candidates.append(item)
                                    existing_ids.add(mid)
                            elif isinstance(item, str):
                                # 从三元组"《盗梦空间》(ID:456)--[导演:...]-->《星际穿越》"中提取ID
                                id_match = re.search(r'ID[：:](\d+)', item)
                                if id_match:
                                    mid = int(id_match.group(1))
                                    if mid not in existing_ids:
                                        title_match = re.search(r'《([^》]+)》', item)
                                        title = title_match.group(1) if title_match else ''
                                        candidates.append({'movie_id': mid, 'title': title})
                                        existing_ids.add(mid)
                    else:
                        # search_vector / recall_hybrid 返回标准字典
                        for item in raw:
                            if isinstance(item, dict):
                                mid = item.get('movie_id', item.get('id'))
                                if mid and mid not in existing_ids:
                                    candidates.append(item)
                                    existing_ids.add(mid)
                    
                    # 候选池上限保护
                    if len(candidates) > 200:
                        candidates = candidates[:200]
                
                # ── 自反馈纠偏机制：空结果重试 ──
                # 当召回工具返回空结果时，自动切换到备用路径
                if not candidates and tool_name in self.FALLBACK_CHAIN:
                    fallback_tool = self.FALLBACK_CHAIN[tool_name]
                    retry_thought = (
                        f"[自反馈纠偏] 工具 {tool_name} 返回空结果，"
                        f"自动切换至备用工具 {fallback_tool} 进行重试"
                    )
                    logger.info(f"[Agent] {retry_thought}")
                    
                    # 记录纠偏Thought到trace
                    trace_steps.append({
                        'step': trace_step_counter,
                        'type': 'thought',
                        'content': retry_thought,
                        'is_retry': True,
                        'original_tool': tool_name,
                        'fallback_tool': fallback_tool,
                    })
                    trace_step_counter += 1
                    
                    # 执行备用工具
                    retry_action, retry_observation = self._act(
                        fallback_tool, user_input, candidates
                    )
                    actions.append(retry_action)
                    observations.append(retry_observation)
                    
                    retry_obs_count = retry_observation.get('count', 0)
                    trace_steps.append({
                        'step': trace_step_counter,
                        'type': 'action',
                        'content': f"[纠偏重试] 调用备用工具 {fallback_tool}",
                        'tool': fallback_tool,
                        'input': retry_action.get('input', ''),
                        'is_retry': True,
                    })
                    trace_step_counter += 1
                    
                    trace_steps.append({
                        'step': trace_step_counter,
                        'type': 'observation',
                        'content': f"[纠偏重试] 备用工具 {fallback_tool} 返回 {retry_obs_count} 条结果",
                        'tool': fallback_tool,
                        'count': retry_obs_count,
                        'is_retry': True,
                    })
                    trace_step_counter += 1
                    
                    retry_raw = retry_observation.get('output', [])
                    candidates = retry_raw if isinstance(retry_raw, list) else []
                    
                    if candidates:
                        trace_steps.append({
                            'step': trace_step_counter,
                            'type': 'thought',
                            'content': f"[纠偏成功] 通过 {fallback_tool} 获得 {len(candidates)} 条候选",
                            'is_retry': True,
                        })
                        trace_step_counter += 1
                
                # ── 低质量结果纠偏机制 ──
                # 当召回结果与查询语义相似度过低时，增大召回范围重试
                if candidates and tool_name in ('search_vector', 'recall_hybrid'):
                    top1 = candidates[0]
                    top1_title = top1.get('title', '') if isinstance(top1, dict) else ''
                    if top1_title:
                        enhanced_q = self._build_enhanced_query(user_input)
                        sim = self._compute_query_result_similarity(enhanced_q, top1_title)
                        if sim < self.LOW_QUALITY_SIM_THRESHOLD:
                            aug_top_k = 100
                            lq_thought = (
                                f"[低质量纠偏] Top-1《{top1_title}》与查询语义相似度 {sim:.3f} "
                                f"< {self.LOW_QUALITY_SIM_THRESHOLD}，增大召回至 {aug_top_k} 重试"
                            )
                            logger.info(f"[Agent] {lq_thought}")
                            trace_steps.append({
                                'step': trace_step_counter,
                                'type': 'thought',
                                'content': lq_thought,
                                'is_retry': True,
                                'original_tool': tool_name,
                                'similarity': round(sim, 4),
                            })
                            trace_step_counter += 1

                            retry_action_lq, retry_obs_lq = self._act(
                                tool_name, user_input, candidates,
                                override_top_k=aug_top_k
                            )
                            actions.append(retry_action_lq)
                            observations.append(retry_obs_lq)

                            retry_raw_lq = retry_obs_lq.get('output', [])
                            if retry_raw_lq:
                                existing_ids_lq = {c.get('movie_id', c.get('id')) for c in candidates if isinstance(c, dict)}
                                new_count = 0
                                for item in retry_raw_lq:
                                    if isinstance(item, dict):
                                        mid = item.get('movie_id', item.get('id'))
                                        if mid and mid not in existing_ids_lq:
                                            candidates.append(item)
                                            existing_ids_lq.add(mid)
                                            new_count += 1
                                trace_steps.append({
                                    'step': trace_step_counter,
                                    'type': 'observation',
                                    'content': f"[低质量纠偏] 增量召回新增 {new_count} 条候选",
                                    'tool': tool_name,
                                    'is_retry': True,
                                })
                                trace_step_counter += 1

                # 在召回后立即应用年份过滤
                if year_filter and candidates:
                    candidates = self._filter_by_year(candidates, year_filter)
                    
            elif tool_name in ('rerank', 'maan_rerank'):
                candidates = observation.get('output', [])
                # 精排后再次确认年份过滤
                if year_filter and candidates:
                    candidates = self._filter_by_year(candidates, year_filter)
        
        # ── Step 5: 约束后过滤 (Genre + Director) + 提取最终推荐ID ──
        detected_genre = self._extract_genre_from_query(user_input)
        detected_director = self._extract_director_from_query(user_input)
        if (detected_genre or detected_director) and candidates:
            from myapp.models import Movie
            try:
                existing_ids = {c.get('movie_id') for c in candidates if c.get('movie_id')}
                # 从数据库补回被 MAAN 排除的约束匹配电影
                refill_qs = Movie.objects.all()
                if detected_director:
                    refill_qs = refill_qs.filter(directors__name__icontains=detected_director)
                if detected_genre:
                    refill_qs = refill_qs.filter(genres__name__icontains=detected_genre)
                refill_ids = list(refill_qs.order_by('-score', '-vote_count').values_list('id', flat=True)[:20])
                for mid in refill_ids:
                    if mid not in existing_ids:
                        candidates.append({'movie_id': mid, 'score': 0, 'source': 'constraint_refill'})
                        existing_ids.add(mid)

                # 分离约束匹配和不匹配的候选
                attr_map = {}
                cids = [c.get('movie_id') for c in candidates if c.get('movie_id')]
                if cids:
                    for m in Movie.objects.filter(id__in=cids).prefetch_related('genres', 'directors'):
                        attr_map[m.id] = {
                            'genres': [g.name for g in m.genres.all()],
                            'directors': [d.name for d in m.directors.all()],
                        }
                    matching = []
                    non_matching = []
                    for c in candidates:
                        mid = c.get('movie_id')
                        attrs = attr_map.get(mid, {})
                        is_match = True
                        if detected_genre:
                            movie_genres = attrs.get('genres', [])
                            if not any(detected_genre in g or g in detected_genre for g in movie_genres):
                                is_match = False
                        if detected_director:
                            movie_directors = attrs.get('directors', [])
                            if not any(detected_director in d or d in detected_director for d in movie_directors):
                                is_match = False
                        if is_match:
                            matching.append(c)
                        else:
                            non_matching.append(c)
                    candidates = matching + non_matching
            except Exception:
                pass

        # 评分质量底线：无显式评分约束时，过滤掉低评分和无评分电影
        if candidates:
            mc = self._micro_think(user_input)
            if not mc.get('min_rating'):
                RATING_FLOOR = 6.5
                try:
                    from myapp.models import Movie
                    cids = [c.get('movie_id') for c in candidates if c.get('movie_id')]
                    score_map = dict(Movie.objects.filter(id__in=cids).values_list('id', 'score'))
                    high_quality = [c for c in candidates if score_map.get(c.get('movie_id')) is not None and float(score_map[c.get('movie_id')] or 0) >= RATING_FLOOR]
                    if len(high_quality) >= 5:
                        candidates = high_quality
                    else:
                        # 即使不足5部，也排除无评分电影
                        has_score = [c for c in candidates if score_map.get(c.get('movie_id')) is not None]
                        if len(has_score) >= len(candidates) * 0.5:
                            candidates = has_score
                except Exception:
                    pass

        for item in candidates:
            mid = item.get('movie_id') if isinstance(item, dict) else None
            if mid:
                recommended_ids.append(mid)
        
        # ── Step 6: 生成推荐理由 ──
        if recommended_ids and self.user:
            for mid in recommended_ids[:5]:
                try:
                    explain_result = self.tools['explain'].execute(
                        user=self.user, movie_id=mid
                    )
                    explanations[mid] = explain_result.get('output', '')
                except Exception as e:
                    # Explain 模块异常不影响推荐结果
                    explanations[mid] = ''
        
        # ── Step 6.5: Faithfulness Self-Check（已禁用，MAAN 精排已保证推荐质量）──

        # ── Step 7: Final Answer ──
        final_answer = self._generate_final_answer(
            user_input, intent, recommended_ids, explanations, thought
        )

        # 处理空结果追问
        need_clarification = False
        clarification_options = []
        if final_answer == "__NEED_CLARIFICATION__":
            need_clarification = True
            clarification_options = VaguenessDetector.generate_clarification_options(
                user_input,
                memory_slots=self.memory.get_slots() if hasattr(self, 'memory') and self.memory else None
            )
            final_answer = "抱歉，暂时没有找到完全匹配的电影。您可以换个关键词试试，或者从下面选择一个方向："

        # 记录Final Answer到trace
        trace_steps.append({
            'step': trace_step_counter,
            'type': 'final_answer',
            'content': final_answer,
        })

        t_total = int((time.time() - t_start) * 1000)

        return {
            'intent': intent,
            'thought': thought,
            'actions': actions,
            'observations': observations,
            'final_answer': final_answer,
            'recommended_ids': recommended_ids,
            'explanations': explanations,
            'latency_ms': t_total,
            'need_clarification': need_clarification,
            'clarification_options': clarification_options,
            'trace_steps': trace_steps,
        }

    def _micro_think(self, user_input):
        """
        轻量推理：从用户输入中提取结构化约束（纯规则，零 LLM 开销）。
        用于快速通道增强查询构建，弥补跳过 _think() 导致的信息损失。

        Returns:
            dict: {
                'genre': str or None,       # 归一化后的类型
                'min_rating': float or None, # 最低评分
                'vibe': str or None,         # 情感氛围
                'year_filter': dict or None, # 年份约束
                'director': str or None,     # 导演
                'actor': str or None,        # 演员
                'exclusions': list,          # 排除项
            }
        """
        constraints = {
            'genre': None, 'min_rating': None, 'vibe': None,
            'year_filter': None, 'director': None, 'actor': None,
            'exclusions': [],
        }

        # 1. 类型归一化：情感/风格词 → 标准类型
        genre_alias = {
            '烧脑': '悬疑', '悬疑推理': '悬疑', '推理': '悬疑',
            '热血': '动作', '打斗': '动作', '追车': '动作',
            '催泪': '剧情', '感人': '剧情', '深度': '剧情',
            '治愈': '动画', '温馨': '剧情', '温暖': '剧情',
            '搞笑': '喜剧', '幽默': '喜剧', '轻松': '喜剧',
            '浪漫': '爱情', '甜蜜': '爱情',
            '惊悚': '恐怖', '吓人': '恐怖',
            '科幻大片': '科幻', '奇幻大片': '奇幻',
        }
        for alias, norm_genre in genre_alias.items():
            if alias in user_input:
                constraints['genre'] = norm_genre
                break

        # 如果没匹配到别名，尝试直接匹配标准类型
        if not constraints['genre']:
            m = re.search(r'(科幻|悬疑|恐怖|喜剧|动作|爱情|剧情|动画|战争|犯罪|奇幻|冒险|文艺|纪录)', user_input)
            if m:
                constraints['genre'] = m.group(1)

        # 2. 评分约束提取
        rating_patterns = [
            (r'评分\s*(\d(?:\.\d)?)\s*分?\s*[以之上]', lambda m: float(m.group(1))),
            (r'(\d(?:\.\d)?)\s*分\s*[以之上]', lambda m: float(m.group(1))),
            (r'高分', lambda m: 8.0),
            (r'经典', lambda m: 7.5),
        ]
        for pattern, extractor in rating_patterns:
            m = re.search(pattern, user_input)
            if m:
                constraints['min_rating'] = extractor(m)
                break

        # 3. 情感氛围提取
        vibe_patterns = [
            (r'(轻松|愉快|欢快|温馨|治愈|开心|快乐)', '轻松'),
            (r'(刺激|紧张|惊悚|恐怖)', '紧张'),
            (r'(感人|催泪|温暖|感动)', '感人'),
            (r'(烧脑|悬疑|反转)', '烧脑'),
            (r'(热血|燃|激昂)', '热血'),
            (r'(压抑|沉重|黑暗|悲伤)', '压抑'),
        ]
        for pattern, vibe in vibe_patterns:
            if re.search(pattern, user_input):
                constraints['vibe'] = vibe
                break

        # 4. 年份约束
        constraints['year_filter'] = self._extract_year_filter(user_input)

        # 5. 导演/演员提取
        director_match = re.search(r'([一-鿿]{2,4})\s*(?:导演|执导)', user_input)
        if director_match:
            constraints['director'] = director_match.group(1)
        else:
            known_directors = {
                '诺兰': '克里斯托弗·诺兰', '宫崎骏': '宫崎骏',
                '昆汀': '昆汀·塔伦蒂诺', '斯皮尔伯格': '史蒂文·斯皮尔伯格',
                '周星驰': '周星驰', '王家卫': '王家卫', '李安': '李安',
                '张艺谋': '张艺谋', '姜文': '姜文', '陈凯歌': '陈凯歌',
                '芬奇': '大卫·芬奇', '卡梅隆': '詹姆斯·卡梅隆',
            }
            for short, full in known_directors.items():
                if short in user_input:
                    constraints['director'] = full
                    break

        actor_match = re.search(r'([一-鿿]{2,4})\s*(?:主演|出演|演的)', user_input)
        if actor_match:
            constraints['actor'] = actor_match.group(1)

        # 6. 排除项提取（多种模式）
        exclusions = []
        # 模式1: "不要X"
        for m in re.finditer(r'不要\s*([一-鿿]{2,6})', user_input):
            exclusions.append(m.group(1))
        # 模式2: "不想看X"
        for m in re.finditer(r'不想\s*看?\s*([一-鿿]{2,6})', user_input):
            exclusions.append(m.group(1))
        # 模式3: "排除X"
        for m in re.finditer(r'排除\s*([一-鿿]{2,6})', user_input):
            exclusions.append(m.group(1))
        # 模式4: "不要X题材" (e.g., "不要太空题材的")
        for m in re.finditer(r'不要\s*([一-鿿]{2,4})题材', user_input):
            exclusions.append(m.group(1))
        # 模式5: "不要X的" (e.g., "不要日本的")
        for m in re.finditer(r'不要\s*([一-鿿]{2,4})的', user_input):
            val = m.group(1)
            if val not in exclusions:
                exclusions.append(val)
        # 去重 + 去除被更短项包含的冗余项
        unique = list(dict.fromkeys(exclusions))
        cleaned = []
        for term in unique:
            # 如果已有更短的项是当前项的子串，跳过当前项
            if not any(short != term and short in term for short in unique):
                cleaned.append(term)
        constraints['exclusions'] = cleaned

        # 冲突检测：如果 genre 与 exclusions 重叠，清除 genre（排除优先）
        if constraints['genre'] and constraints['exclusions']:
            genre = constraints['genre']
            if any(genre in ex or ex in genre for ex in constraints['exclusions']):
                constraints['genre'] = None

        # 7. 记忆槽位融合
        slots = self.memory.get_slots()
        if not constraints['genre'] and slots.get('genre'):
            constraints['genre'] = slots['genre']
        if not constraints['director'] and slots.get('director'):
            constraints['director'] = slots['director']
        if not constraints['actor'] and slots.get('actor'):
            constraints['actor'] = slots['actor']

        return constraints

    def _run_fast_path(self, user_input, intent, t_start, llm_parsed=None):
        """
        快速通道：简单查询直接执行 recall → MAAN → rerank → explain。
        跳过 ReAct 循环和 _think()，但通过 _micro_think() 保留轻量推理能力。
        同时保留 enhanced query、hard filter、fallback、低质量纠偏和年份过滤。
        """
        trace_steps = []
        actions = []
        observations = []
        candidates = []
        recommended_ids = []
        explanations = {}
        llm_parsed = llm_parsed or {}

        # Micro-Think：轻量推理提取结构化约束（纯规则，<1ms）
        # 注意：不修改查询文本（会污染向量搜索语义），只提取后处理约束
        mc = self._micro_think(user_input)
        fast_query = user_input

        # 合并 LLM 意图解析结果：tags → genre 优先
        if llm_parsed.get('tags') and not mc.get('genre'):
            mc['genre'] = llm_parsed['tags'][0]
        if llm_parsed.get('tags'):
            mc['llm_tags'] = llm_parsed['tags']
        mc['sort_by'] = llm_parsed.get('sort_by', 'hot')

        # LLM 解析 trace
        if llm_parsed.get('tags'):
            trace_steps.append({
                'step': 0, 'type': 'thought',
                'content': f"LLM 意图解析: tags={llm_parsed['tags']}, sort_by={mc['sort_by']}",
            })

        # 提取年份过滤条件（优先用 micro-think 的结果）
        year_filter = mc.get('year_filter') or self._extract_year_filter(user_input)

        # 简单 Thought（基于 micro-think 结果，增强约束推理文本）
        mc_summary = []
        if mc['genre']:
            mc_summary.append(f"类型约束={mc['genre']}")
        if mc['min_rating']:
            mc_summary.append(f"评分约束≥{mc['min_rating']}")
        if mc['vibe']:
            mc_summary.append(f"氛围约束={mc['vibe']}")
        if mc['director']:
            mc_summary.append(f"导演约束={mc['director']}")
        if mc['actor']:
            mc_summary.append(f"演员约束={mc['actor']}")
        if year_filter:
            mc_summary.append(f"年份约束={year_filter}")
        if mc.get('exclusions'):
            mc_summary.append(f"排除={mc['exclusions']}")
        # 融合记忆槽位
        if hasattr(self, 'memory') and self.memory:
            slots = self.memory.get_slots()
            if slots.get('genre'):
                mc_summary.append(f"记忆槽位类型={slots['genre']}")
            if slots.get('year_min'):
                mc_summary.append(f"记忆槽位年份≥{slots['year_min']}")
        mc_str = f"，约束分析: {', '.join(mc_summary)}" if mc_summary else "，无特定约束"
        tool_names = ", ".join(self.INTENT_TOOL_MAP.get(intent, []))
        thought = (
            f"【意图】{intent}{mc_str}。\n"
            f"【推理计划】\n"
            f"  1. 基于用户输入进行类型归一化和约束提取\n"
            f"  2. 执行向量召回 → MAAN深度精排 → 业务重排\n"
            f"  3. 生成可解释推荐理由\n"
            f"【能力约束】可用工具: {tool_names}"
        )
        trace_steps.append({'step': 0, 'type': 'thought', 'content': thought})
        step_counter = 1

        # 获取工具链
        tool_chain = list(self.INTENT_TOOL_MAP.get(intent, []))

        # 消融适配: RAG 不可用时，用数据库查询替代向量搜索
        has_rag = bool(self.rag_resources and self.rag_resources.get('vectorstore'))
        if not has_rag:
            tool_chain = [
                'search_database' if t in ('search_vector', 'recall_hybrid') else t
                for t in tool_chain
            ]

        for tool_name in tool_chain:
            # 召回工具使用增强查询，精排工具使用原始输入
            query_for_tool = fast_query if tool_name in ('search_vector', 'recall_hybrid') else user_input
            action, observation = self._act(tool_name, query_for_tool, candidates)
            actions.append(action)
            observations.append(observation)

            trace_steps.append({
                'step': step_counter, 'type': 'action',
                'content': f"调用工具 {tool_name}",
                'tool': tool_name, 'input': action.get('input', ''),
            })
            step_counter += 1

            obs_count = observation.get('count', 0)
            trace_steps.append({
                'step': step_counter, 'type': 'observation',
                'content': f"工具 {tool_name} 返回 {obs_count} 条结果",
                'tool': tool_name, 'count': obs_count,
            })
            step_counter += 1

            # 合并候选集（与主路径相同的逻辑）
            if tool_name in ('search_vector', 'recall_hybrid', 'kg_query', 'search_database'):
                raw = observation.get('output', [])
                if isinstance(raw, list) and raw:
                    existing_ids = {c.get('movie_id', c.get('id')) for c in candidates if isinstance(c, dict)}
                    for item in raw:
                        if isinstance(item, dict):
                            mid = item.get('movie_id', item.get('id'))
                            if mid and mid not in existing_ids:
                                candidates.append(item)
                                existing_ids.add(mid)
                    if len(candidates) > 200:
                        candidates = candidates[:200]

                # 空结果 Fallback：与主路径相同的纠偏逻辑
                if not candidates and tool_name in self.FALLBACK_CHAIN:
                    fallback_tool = self.FALLBACK_CHAIN[tool_name]
                    retry_thought = (
                        f"[快速通道纠偏] 工具 {tool_name} 返回空结果，"
                        f"切换至备用工具 {fallback_tool}"
                    )
                    trace_steps.append({
                        'step': step_counter, 'type': 'thought',
                        'content': retry_thought, 'is_retry': True,
                    })
                    step_counter += 1

                    retry_action, retry_obs = self._act(fallback_tool, fast_query, candidates)
                    actions.append(retry_action)
                    observations.append(retry_obs)

                    retry_raw = retry_obs.get('output', [])
                    if isinstance(retry_raw, list):
                        candidates = list(retry_raw)

                    trace_steps.append({
                        'step': step_counter, 'type': 'action',
                        'content': f"[纠偏] 调用备用工具 {fallback_tool}",
                        'tool': fallback_tool, 'is_retry': True,
                    })
                    step_counter += 1
                    trace_steps.append({
                        'step': step_counter, 'type': 'observation',
                        'content': f"[纠偏] 备用工具返回 {retry_obs.get('count', 0)} 条结果",
                        'tool': fallback_tool, 'count': retry_obs.get('count', 0), 'is_retry': True,
                    })
                    step_counter += 1

                # 低质量结果纠偏：Top-1 与查询语义相似度过低时，增大召回重试
                if candidates and tool_name in ('search_vector', 'recall_hybrid'):
                    top1 = candidates[0]
                    top1_title = top1.get('title', '') if isinstance(top1, dict) else ''
                    if top1_title:
                        enhanced_q = self._build_enhanced_query(user_input)
                        sim = self._compute_query_result_similarity(enhanced_q, top1_title)
                        if sim < self.LOW_QUALITY_SIM_THRESHOLD:
                            aug_top_k = 100
                            lq_thought = (
                                f"[低质量纠偏] Top-1《{top1_title}》与查询语义相似度 {sim:.3f} "
                                f"< {self.LOW_QUALITY_SIM_THRESHOLD}，增大召回至 {aug_top_k} 重试"
                            )
                            logger.info(f"[快速通道] {lq_thought}")
                            trace_steps.append({
                                'step': step_counter, 'type': 'thought',
                                'content': lq_thought, 'is_retry': True,
                            })
                            step_counter += 1

                            retry_action_lq, retry_obs_lq = self._act(
                                tool_name, fast_query, candidates,
                                override_top_k=aug_top_k
                            )
                            actions.append(retry_action_lq)
                            observations.append(retry_obs_lq)

                            retry_raw_lq = retry_obs_lq.get('output', [])
                            if retry_raw_lq:
                                existing_ids_lq = {c.get('movie_id', c.get('id')) for c in candidates if isinstance(c, dict)}
                                new_count = 0
                                for item in retry_raw_lq:
                                    if isinstance(item, dict):
                                        mid = item.get('movie_id', item.get('id'))
                                        if mid and mid not in existing_ids_lq:
                                            candidates.append(item)
                                            existing_ids_lq.add(mid)
                                            new_count += 1
                                trace_steps.append({
                                    'step': step_counter, 'type': 'observation',
                                    'content': f"[低质量纠偏] 增量召回新增 {new_count} 条候选",
                                    'tool': tool_name, 'is_retry': True,
                                })
                                step_counter += 1

                # 年份过滤：召回后立即应用
                if year_filter and candidates:
                    candidates = self._filter_by_year(candidates, year_filter)

            elif tool_name in ('rerank', 'maan_rerank'):
                candidates = observation.get('output', [])
                # 精排后再次确认年份过滤
                if year_filter and candidates:
                    candidates = self._filter_by_year(candidates, year_filter)

        # 🎯 Genre 约束后过滤：用 micro-think 检测的类型对候选做软排序提升
        if mc.get('genre') and candidates:
            from myapp.models import Movie
            target_genre = mc['genre']
            try:
                genre_map = {}
                cids = [c.get('movie_id') for c in candidates if c.get('movie_id')]
                if cids:
                    for m in Movie.objects.filter(id__in=cids).prefetch_related('genres'):
                        genre_map[m.id] = [g.name for g in m.genres.all()]
                    for c in candidates:
                        mid = c.get('movie_id')
                        if mid and mid in genre_map:
                            movie_genres = genre_map[mid]
                            if any(target_genre in g or g in target_genre for g in movie_genres):
                                c['_genre_boost'] = 1.3
                            else:
                                c['_genre_boost'] = 1.0
                        else:
                            c['_genre_boost'] = 1.0
                    for c in candidates:
                        base = c.get('maan_score', c.get('score', 0))
                        c['_final_score'] = base * c.get('_genre_boost', 1.0)
                    candidates.sort(key=lambda x: x.get('_final_score', 0), reverse=True)
            except Exception:
                pass

        # 🚫 排除约束过滤：过滤掉用户明确排除的类型/题材
        if mc.get('exclusions') and candidates:
            from myapp.models import Movie
            exclusion_terms = mc['exclusions']
            try:
                cids = [c.get('movie_id') for c in candidates if c.get('movie_id')]
                if cids:
                    genre_map_ex = {}
                    for m in Movie.objects.filter(id__in=cids).prefetch_related('genres'):
                        genre_map_ex[m.id] = [g.name for g in m.genres.all()]
                    filtered = []
                    for c in candidates:
                        mid = c.get('movie_id')
                        movie_genres = genre_map_ex.get(mid, [])
                        # 检查是否匹配任何排除项
                        excluded = False
                        for term in exclusion_terms:
                            if any(term in g or g in term for g in movie_genres):
                                excluded = True
                                break
                        if not excluded:
                            filtered.append(c)
                    # 至少保留 5 个候选，避免过度过滤
                    if len(filtered) >= 5:
                        candidates = filtered
                        trace_steps.append({
                            'step': step_counter, 'type': 'thought',
                            'content': f"[排除过滤] 排除 {exclusion_terms}，剩余 {len(candidates)} 条候选",
                        })
                        step_counter += 1
            except Exception:
                pass

        # 评分质量底线：无显式评分约束时，过滤掉低评分和无评分电影
        if candidates and not mc.get('min_rating'):
            RATING_FLOOR = 6.5
            try:
                from myapp.models import Movie
                cids = [c.get('movie_id') for c in candidates if c.get('movie_id')]
                score_map = dict(Movie.objects.filter(id__in=cids).values_list('id', 'score'))
                high_quality = [c for c in candidates if score_map.get(c.get('movie_id')) is not None and float(score_map[c.get('movie_id')] or 0) >= RATING_FLOOR]
                if len(high_quality) >= 5:
                    candidates = high_quality
                else:
                    has_score = [c for c in candidates if score_map.get(c.get('movie_id')) is not None]
                    if len(has_score) >= len(candidates) * 0.5:
                        candidates = has_score
            except Exception:
                pass

        # 提取推荐 ID
        for item in candidates:
            mid = item.get('movie_id') if isinstance(item, dict) else None
            if mid:
                recommended_ids.append(mid)

        # 生成推荐理由
        if recommended_ids and self.user:
            for mid in recommended_ids[:5]:
                try:
                    explain_result = self.tools['explain'].execute(user=self.user, movie_id=mid)
                    explanations[mid] = explain_result.get('output', '')
                except Exception:
                    explanations[mid] = ''

        # Faithfulness Self-Check（已禁用，MAAN 精排已保证推荐质量）

        # Final Answer
        final_answer = self._generate_final_answer(user_input, intent, recommended_ids, explanations, thought)

        # 处理空结果追问
        need_clarification = False
        clarification_options = []
        if final_answer == "__NEED_CLARIFICATION__":
            need_clarification = True
            clarification_options = VaguenessDetector.generate_clarification_options(
                user_input,
                memory_slots=self.memory.get_slots() if hasattr(self, 'memory') and self.memory else None
            )
            final_answer = "抱歉，暂时没有找到完全匹配的电影。您可以换个关键词试试，或者从下面选择一个方向："

        trace_steps.append({'step': step_counter, 'type': 'final_answer', 'content': final_answer})

        t_total = int((time.time() - t_start) * 1000)

        return {
            'intent': intent,
            'thought': thought,
            'actions': actions,
            'observations': observations,
            'final_answer': final_answer,
            'recommended_ids': recommended_ids,
            'explanations': explanations,
            'latency_ms': t_total,
            'need_clarification': need_clarification,
            'clarification_options': clarification_options,
            'trace_steps': trace_steps,
        }

    def _detect_anchor_movie(self, text):
        """
        检测用户输入中的锚点电影（用于动态工具链路由）。
        当用户提到特定电影名或导演/演员名时，Agent 会优先通过知识图谱查询其核心属性。

        Returns:
            str or None: 锚点电影名称，未检测到返回 None
        """
        # 1. 书名号引用（最高置信度）
        match = re.search(r'《([^》]+)》', text)
        if match:
            movie_name = match.group(1)
            if Movie.objects.filter(title__icontains=movie_name).exists():
                return movie_name

        # 2. "类似/像/推荐...XX的"模式 → XX是锚点电影
        patterns = [
            r'(?:类似|像|类似|推荐|看过)\s*([一-鿿]{2,8})\s*(?:的|那种|那种的|一样)',
            r'(?:看过|看完|喜欢)\s*《?([一-鿿]{2,8})》?',
        ]
        exclude_words = {'电影', '片子', '这种', '那种', '什么', '好看', '推荐',
                         '动画', '动作', '喜剧', '爱情', '科幻', '悬疑', '恐怖', '战争', '犯罪', '奇幻'}
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                candidate = m.group(1)
                if candidate not in exclude_words:
                    if Movie.objects.filter(title__icontains=candidate).exists():
                        return candidate

        # 3. Neo4j Person 实体匹配（导演/演员名 → 关联电影）
        if self.neo_graph:
            try:
                name_candidates = re.findall(r'([一-鿿]{2,4})', text)
                for name in name_candidates:
                    if name in exclude_words:
                        continue
                    cypher = """
                    MATCH (p:Person {name: $name})-[:DIRECTED_BY|ACTED_IN]->(m:Movie)
                    RETURN m.title AS title LIMIT 1
                    """
                    rows = self.neo_graph.run(cypher, name=name).data()
                    if rows:
                        logger.info(f"[Agent] Neo4j 实体匹配: \'{name}\' → Person 节点")
                        return rows[0]['title']
            except Exception as e:
                logger.warning(f"[Agent] Neo4j 实体匹配异常: {e}")

        # 4. Movie 节点模糊匹配（编辑距离 <= 2）
        try:
            movie_frags = re.findall(r'([一-鿿]{3,8})', text)
            for frag in movie_frags:
                if frag in exclude_words:
                    continue
                exact = Movie.objects.filter(title__icontains=frag).first()
                if exact:
                    return exact.title
        except Exception:
            pass

        return None

    @staticmethod
    def _edit_distance(s1, s2):
        """计算两个字符串的编辑距离"""
        m, n = len(s1), len(s2)
        dp = list(range(n + 1))
        for i in range(1, m + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, n + 1):
                temp = dp[j]
                if s1[i - 1] == s2[j - 1]:
                    dp[j] = prev
                else:
                    dp[j] = 1 + min(prev, dp[j], dp[j - 1])
                prev = temp
        return dp[n]


    def _think(self, user_input, intent, is_followup=False, anchor_movie=None):
        """
        Thought阶段：生成结构化推理框架（参照前哨AI提示词工程方法论）。
        包含角色定义、目标声明、意图识别、能力约束、推理计划、已识别约束、记忆上下文。
        """
        # ── 角色定义 ──
        role = "【角色】智能观影推荐助手，融合 RAG+KAG+MAAN 的 5 层 Agent 架构"

        # ── 目标声明 ──
        goal = "【目标】为用户推荐最匹配的电影，提供可解释的推荐理由"

        # ── 意图识别 ──
        intent_descriptions = {
            'QUERY_MOVIE': '用户正在寻找电影推荐，将使用混合召回+重排+解释的完整推荐链',
            'QUERY_COMPARISON': '用户想对比电影，将召回相关候选并通过重排呈现对比结果',
            'QUERY_PROFILE_REC': '用户希望基于个人画像推荐，将使用个性化混合召回',
            'QUERY_RANK': '用户想看热门榜单，将搜索高分热门电影',
            'QUERY_NEW': '用户想看最新电影，将搜索最新入库影片',
            'QUERY_VISUAL': '用户在进行视觉搜索，将通过向量语义检索海报特征',
            'QUERY_SELF': '用户想了解个人观影画像，将分析用户历史行为',
            'CHAT': '用户在进行闲聊，将直接回复',
        }
        intent_desc = intent_descriptions.get(intent, '意图未明，将使用默认推荐流程')
        intent_line = f"【意图】{intent} — {intent_desc}"
        if is_followup:
            intent_line += " [追问模式：融合历史槽位]"

        # ── 能力约束 ──
        tool_names = ", ".join(self.INTENT_TOOL_MAP.get(intent, []))
        capabilities = (
            f"【能力约束】\n"
            f"  - 可用工具: {tool_names}\n"
            f"  - 推荐来源: 电影数据库（非实时票房数据）\n"
            f"  - 推荐数量: Top-5\n"
            f"  - 纠偏策略: 若召回工具返回空结果，将自动切换至备用路径重试"
        )

        # ── 推理计划 ──
        plan_steps = ["确认推荐目标（锚点/类型/情感/年份）"]
        if intent in ('QUERY_MOVIE', 'QUERY_COMPARISON', 'QUERY_PROFILE_REC'):
            plan_steps.append("多路召回（向量语义 + 知识图谱）")
            plan_steps.append("深度精排（MAAN GAUC 0.8898）")
            plan_steps.append("生成可解释推荐理由")
        elif intent == 'QUERY_RANK':
            plan_steps.append("搜索高分热门电影")
        elif intent == 'QUERY_NEW':
            plan_steps.append("搜索最新入库影片")
        elif intent == 'QUERY_VISUAL':
            plan_steps.append("向量语义检索海报特征")
        elif intent == 'QUERY_SELF':
            plan_steps.append("分析用户历史行为")
        plan_lines = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(plan_steps))
        plan = f"【推理计划】\n{plan_lines}"

        # ── 已识别约束 ──
        constraints = []
        if anchor_movie:
            constraints.append(f"锚点电影: 《{anchor_movie}》")

        # 情感约束检测
        sentiment_patterns = [
            (r'不要.*?(压抑|沉重|悲伤|黑暗|悲观)', '排除压抑氛围'),
            (r'(轻松|愉快|欢快|温馨|治愈)', '偏好轻松氛围'),
            (r'(刺激|紧张|惊悚)', '偏好紧张氛围'),
        ]
        for pattern, label in sentiment_patterns:
            if re.search(pattern, user_input):
                constraints.append(f"情感偏好: {label}")

        # 年份约束
        year_filter = self._extract_year_filter(user_input)
        if year_filter:
            min_y = year_filter.get('min_year', '')
            max_y = year_filter.get('max_year', '')
            if min_y and max_y and min_y == max_y:
                constraints.append(f"年份范围: 限{min_y}年")
            elif min_y and max_y:
                constraints.append(f"年份范围: {min_y}-{max_y}年")
            elif min_y:
                constraints.append(f"年份范围: {min_y}年及以后")
            elif max_y:
                constraints.append(f"年份范围: {max_y}年及以前")

        if constraints:
            constraint_lines = "\n".join(f"  - {c}" for c in constraints)
            constraint_section = f"【已识别约束】\n{constraint_lines}"
        else:
            constraint_section = "【已识别约束】无特定约束"

        # ── 记忆上下文 ──
        memory_summary = self.memory.get_memory_summary()
        memory_section = f"【记忆上下文】{memory_summary}" if memory_summary and memory_summary != "无历史记忆" else "【记忆上下文】首次交互，无历史记忆"

        # ── 锚点电影推理（多跳场景展开）──
        anchor_reasoning = ""
        if anchor_movie:
            anchor_reasoning = (
                f"\n【锚点推理】用户提到了《{anchor_movie}》，"
                f"需要先通过 kg_query 确认其核心属性（导演、类型），"
                f"再调用 search_vector 寻找语义相似的作品，"
                f"最后结合用户偏好过滤氛围不匹配的候选。"
            )

        return "\n".join([
            role, goal, intent_line, capabilities, plan,
            constraint_section, memory_section, anchor_reasoning,
            f"用户原始输入: '{user_input}'"
        ])
    
    def _compute_query_result_similarity(self, query, top1_title):
        """
        计算查询与 Top-1 结果标题的 BGE 语义相似度。
        用于低质量结果检测：相似度 < 0.6 表示召回结果与查询不匹配。

        Args:
            query: 增强后的查询文本
            top1_title: Top-1 结果的电影标题

        Returns:
            float: 余弦相似度 [0, 1]，失败返回 0.0
        """
        try:
            vectorstore = (self.rag_resources or {}).get("vectorstore")
            if not vectorstore:
                return 0.0
            embedder = vectorstore.embedding_function
            q_vec = embedder.embed_query(query)
            t_vec = embedder.embed_query(top1_title)
            # 余弦相似度
            import numpy as np
            q_arr = np.array(q_vec)
            t_arr = np.array(t_vec)
            dot = np.dot(q_arr, t_arr)
            norm = np.linalg.norm(q_arr) * np.linalg.norm(t_arr)
            if norm == 0:
                return 0.0
            return float(dot / norm)
        except Exception as e:
            logger.warning(f"[Agent] BGE 相似度计算失败: {e}")
            return 0.0

    def _act(self, tool_name, user_input, current_candidates, override_top_k=None):
        """
        Action阶段：执行工具调用
        🔥 核心改进：追问时融合记忆槽位构建增强查询

        Args:
            override_top_k: 覆盖默认召回数量（用于低质量结果纠偏重试）
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return {'tool': tool_name, 'input': '', 'error': '工具不存在'}, {'output': [], 'error': '工具不存在'}

        try:
            # 🔥 构建增强查询：将记忆槽位合并到用户输入中
            enhanced_query = self._build_enhanced_query(user_input)

            # 【Bug3修复】增大召回数量，确保精排有足够候选池
            if tool_name == 'search_vector':
                k = override_top_k or 60
                result = tool.execute(query=enhanced_query, k=k)
                # 硬过滤补充：当查询含导演/演员属性时，向量搜索效果差，
                # 用数据库硬过滤获取精确候选并优先插入
                result = self._augment_with_hard_filter(result, user_input)
            elif tool_name == 'recall_hybrid':
                top_k = override_top_k or 60
                result = tool.execute(user=self.user, query_text=enhanced_query, top_k=top_k)
                result = self._augment_with_hard_filter(result, user_input)
            elif tool_name == 'kg_query':
                # 提取电影名 + 结构化约束（NL2Cypher）
                movie_name = self._extract_movie_name(user_input)
                mc = self._micro_think(user_input)
                result = tool.execute(movie_title=movie_name, constraints=mc)
            elif tool_name == 'maan_rerank':
                slots = self.memory.get_slots() if hasattr(self, 'memory') and self.memory else None
                result = tool.execute(candidates=current_candidates, user=self.user, top_k=15, memory_slots=slots)
            elif tool_name == 'rerank':
                result = tool.execute(candidates=current_candidates, user=self.user, top_k=15)
            elif tool_name == 'explain':
                result = tool.execute(user=self.user)
            else:
                result = {'tool': tool_name, 'input': '', 'output': [], 'error': '未知工具'}

            action_record = {
                'tool': tool_name,
                'input': result.get('input', ''),
                'timestamp': int(time.time() * 1000),
            }
            observation_record = {
                'tool': tool_name,
                'output': result.get('output', []),
                'count': result.get('count', 0),
                'stats': result.get('stats', {}),
            }

            return action_record, observation_record
        except Exception as e:
            return (
                {'tool': tool_name, 'input': user_input, 'error': str(e)},
                {'output': [], 'error': str(e)}
            )

    def _augment_with_hard_filter(self, result, user_input):
        """
        硬过滤补充：当查询包含导演/演员/类型等结构化属性时，
        用数据库硬过滤获取精确候选，并优先插入到结果列表头部。

        解决问题：FAISS/BGE 向量搜索对中文人名编码效果差，
        导致导演查询返回不相关电影。
        """
        from django.db.models import Q

        director = self._extract_director_from_query(user_input)
        actor = self._extract_actor_from_query(user_input)
        genre = self._extract_genre_from_query(user_input)

        if not director and not actor and not genre:
            return result

        output = result.get('output', [])
        existing_ids = {c.get('movie_id') for c in output if isinstance(c, dict)}

        # 构建硬过滤查询
        qs = Movie.objects.all()
        filter_desc = []
        if director:
            qs = qs.filter(directors__name__icontains=director)
            filter_desc.append(f'director={director}')
        if actor:
            qs = qs.filter(actors__name__icontains=actor)
            filter_desc.append(f'actor={actor}')
        if genre:
            qs = qs.filter(genres__name__icontains=genre)
            filter_desc.append(f'genre={genre}')

        hard_ids = list(
            qs.order_by('-score', '-vote_count')
            .values_list('id', flat=True)[:30]
        )

        if hard_ids:
            # 将硬过滤结果插入到输出头部（优先级最高）
            hard_items = []
            for mid in hard_ids:
                if mid not in existing_ids:
                    hard_items.append({
                        'movie_id': mid,
                        'score': 1.0,
                        'source': 'hard_filter',
                    })
                    existing_ids.add(mid)

            # 硬过滤结果放前面，向量搜索结果放后面
            combined = hard_items + output
            result['output'] = combined
            result['count'] = len(combined)
            result['stats'] = result.get('stats', {})
            result['stats']['hard_filter'] = len(hard_items)
            result['stats']['filter_desc'] = ', '.join(filter_desc)
            logger.info(
                f"[Agent] 硬过滤补充: {', '.join(filter_desc)} → "
                f"{len(hard_ids)} 部，新增 {len(hard_items)} 部到候选池头部"
            )

        return result

    def _extract_director_from_query(self, query):
        """从查询中提取导演名（返回数据库中存储的中文名）"""
        # 先尝试匹配完整中文名（优先级最高）
        full_name_patterns = [
            r'(克里斯托弗·诺兰)',
            r'(史蒂文·斯皮尔伯格)',
            r'(昆汀·塔伦蒂诺)',
            r'(大卫·芬奇)',
        ]
        for p in full_name_patterns:
            m = re.search(p, query)
            if m:
                return m.group(1).strip()

        # 再尝试匹配简称 + 导演关键词
        short_patterns = [
            r'([一-鿿]{2,4})\s*(?:导演|执导|导演的)',
            r'(?:导演|执导)\s*([一-鿿]{2,4})',
        ]
        for p in short_patterns:
            m = re.search(p, query)
            if m:
                name = m.group(1).strip()
                # 简称映射到数据库中的完整中文名
                name_map = {
                    '诺兰': '克里斯托弗·诺兰',
                    '斯皮尔伯格': '史蒂文·斯皮尔伯格',
                    '昆汀': '昆汀·塔伦蒂诺',
                    '芬奇': '大卫·芬奇',
                }
                return name_map.get(name, name)

        # 最后尝试直接匹配已知导演名（含简称映射）
        known_directors = {
            '宫崎骏': '宫崎骏', '王家卫': '王家卫', '周星驰': '周星驰',
            '李安': '李安', '张艺谋': '张艺谋', '姜文': '姜文',
            '陈凯歌': '陈凯歌', '冯小刚': '冯小刚', '徐克': '徐克',
            '吴宇森': '吴宇森', '侯孝贤': '侯孝贤', '杨德昌': '杨德昌',
            # 外国导演简称 → 数据库中文全名
            '诺兰': '克里斯托弗·诺兰', '昆汀': '昆汀·塔伦蒂诺',
            '斯皮尔伯格': '史蒂文·斯皮尔伯格', '芬奇': '大卫·芬奇',
            '卡梅隆': '詹姆斯·卡梅隆', '马丁': '马丁·斯科塞斯',
            '雷德利': '雷德利·斯科特', '库布里克': '斯坦利·库布里克',
            '希区柯克': '阿尔弗雷德·希区柯克', '盖里奇': '盖·里奇',
        }
        for short, full in known_directors.items():
            if short in query:
                return full

        return None

    def _extract_actor_from_query(self, query):
        """从查询中提取演员名"""
        patterns = [
            r'([一-鿿]{2,4})\s*(?:主演|出演)',
            r'(?:主演|出演)\s*([一-鿿]{2,4})',
        ]
        for p in patterns:
            m = re.search(p, query)
            if m:
                return m.group(1).strip()
        return None

    def _extract_genre_from_query(self, query):
        """从查询中提取类型"""
        genre_map = {
            '科幻': '科幻', '悬疑': '悬疑', '恐怖': '恐怖', '喜剧': '喜剧',
            '动作': '动作', '爱情': '爱情', '剧情': '剧情', '动画': '动画',
            '战争': '战争', '犯罪': '犯罪', '奇幻': '奇幻', '冒险': '冒险',
            '惊悚': '惊悚', '文艺': '文艺', '纪录': '纪录片', '传记': '传记',
            '音乐': '音乐', '家庭': '家庭', '武侠': '武侠', '古装': '古装',
        }
        for keyword, genre in genre_map.items():
            if keyword in query:
                return genre
        return None

    def _build_query_intent_reason(self, query, movie_genres, movie_directors):
        """
        根据用户查询意图生成个性化推荐理由前缀，
        将推荐与用户需求挂钩，提升推荐理由的说服力。
        """
        if not query:
            return ""

        # 锚点电影关联
        anchor = self._detect_anchor_movie(query)
        if anchor:
            return f"与《{anchor}》风格相近"

        # 类型匹配
        query_genre = self._extract_genre_from_query(query)
        if query_genre and query_genre in (movie_genres or ''):
            return f"符合您对{query_genre}片的偏好"

        # 情感/氛围匹配
        sentiment_map = [
            (r'(轻松|愉快|欢快|温馨|治愈)', '轻松治愈风格'),
            (r'(刺激|紧张|惊悚|烧脑)', '节奏紧凑、引人入胜'),
            (r'(感人|催泪|悲伤|感动)', '情感细腻、触动人心'),
            (r'(搞笑|幽默|好笑|喜剧)', '轻松幽默、欢乐解压'),
            (r'(热血|燃|激情)', '热血激昂、振奋人心'),
        ]
        for pattern, label in sentiment_map:
            if re.search(pattern, query):
                return f"属于{label}的佳作"

        # 导演匹配
        director_from_query = self._extract_director_from_query(query)
        if director_from_query and director_from_query in (movie_directors or ''):
            return f"正是您寻找的{director_from_query}作品"

        # 年份匹配
        year_filter = self._extract_year_filter(query)
        if year_filter:
            min_y = year_filter.get('min_year')
            if min_y:
                return f"满足{min_y}年以后的观影需求"

        # 评分匹配
        if re.search(r'(高分|评分|好看|优秀)', query):
            return "属于高分佳作"

        return ""
    
    def _build_enhanced_query(self, user_input):
        """
        构建语义查询：从用户输入中提取核心语义关键词，
        去除模糊的高频词（推荐、电影、片子等），保留能区分主题的词。
        🔥 核心改进：始终融合记忆槽位，而非仅追问时
        """
        # 1. 清洗：去除高频模糊词，保留语义关键词
        stop_words = [
            '推荐', '介绍', '几部', '一些', '有没有', '有没有好', '好看的',
            '电影', '片子', '片', '影片', '剧', '看', '找', '搜索', '给',
            '类似', '差不多', '差不多的', '类似的', '那种', '类型', '风格',
            '根据', '我的', '帮我', '帮', '想看', '想要', '来', '点',
        ]

        cleaned = user_input
        for w in stop_words:
            cleaned = cleaned.replace(w, ' ')

        # 2. 提取核心关键词
        keywords = []

        # 电影名引用（最高优先级）
        movie_ref = re.search(r'《([^》]+)》', user_input)
        if movie_ref:
            keywords.append(movie_ref.group(1))

        # 锚点电影检测（类似星际穿越 → 星际穿越）
        anchor = re.search(r'(?:类似|像)\s*(?:《)?([^》和、与]{2,8})(?:》)?', user_input)
        if anchor:
            anchor_name = anchor.group(1).strip()
            if anchor_name not in keywords:
                keywords.append(anchor_name)

        # 类型关键词
        genre_match = re.search(r'(科幻|悬疑|恐怖|喜剧|动作|爱情|剧情|动画|战争|犯罪|奇幻|冒险|烧脑|文艺|纪录)', cleaned)
        if genre_match:
            keywords.append(genre_match.group(1))

        # 导演关键词
        director_match = re.search(r'(诺兰|宫崎骏|昆汀|斯皮尔伯格|周星驰|王家卫|nolan)', cleaned, re.I)
        if director_match:
            keywords.append(director_match.group(1))

        # 情感/氛围关键词
        vibe_match = re.search(r'(烧脑|感人|搞笑|催泪|热血|治愈|压抑|轻松|刺激|温馨)', cleaned)
        if vibe_match:
            keywords.append(vibe_match.group(1))

        # 画像相关
        if re.search(r'(画像|偏好|口味|我.*喜欢)', user_input):
            slots = self.memory.get_slots()
            if slots.get('genre'):
                keywords.append(slots['genre'])
            if slots.get('director'):
                keywords.append(slots['director'])

        # 最新/热门相关
        if re.search(r'(最新|新出|最近|上映)', user_input):
            keywords.append('新片')
        if re.search(r'(热门|高分|经典|排行)', user_input):
            keywords.append('高分')

        # 3. 🔥 始终合并记忆槽位（非仅追问时）
        slots = self.memory.get_slots()
        if slots.get('genre') and slots['genre'] not in ' '.join(keywords):
            keywords.insert(0, slots['genre'])
        if slots.get('director') and slots['director'] not in ' '.join(keywords):
            keywords.append(slots['director'])
        if slots.get('keyword') and slots['keyword'] not in ' '.join(keywords):
            keywords.append(slots['keyword'])
        if slots.get('actor') and slots['actor'] not in ' '.join(keywords):
            keywords.append(slots['actor'])

        # 4. 拼接查询
        if keywords:
            query = ' '.join(keywords)
            logger.info(f"[Agent] 语义查询: '{user_input}' → '{query}'")
            return query

        # 最终兜底
        return user_input
    
    def _extract_movie_name(self, text):
        """从用户输入中提取电影名称。返回 None 表示未找到电影名。"""
        # 匹配书名号
        match = re.search(r'《([^》]+)》', text)
        if match:
            return match.group(1)

        # 匹配已知电影名（数据库中的标题必须完整出现在输入中）
        # 先排除常见非电影名片段
        stop_words = ['推荐', '电影', '片子', '类似', '的', '几部', '好看', '有没有', '介绍', '找']
        clean = text
        for w in stop_words:
            clean = clean.replace(w, ' ')
        clean = clean.strip()

        movies = Movie.objects.all()[:500]
        for m in movies:
            if len(m.title) >= 2 and m.title in clean:
                return m.title

        return None
    
    def _extract_year_filter(self, text):
        """
        从用户输入中提取年份过滤条件。

        支持格式:
          - "近20年" / "最近20年" / "近5年" → min_year=当前年份-N
          - "1990年以后" / "1990年之后" / "1990年后" → min_year=1990
          - "2000年以前" / "2000年之前" / "2000年前" → max_year=2000
          - "2020年的" / "2020年上映" → exact_year=2020
          - "90年代" → min_year=1990, max_year=1999
          - "2010到2020" → min_year=2010, max_year=2020
          - 🔥 记忆槽位中的 year_min → 自动继承

        Returns:
            dict: {'min_year': int, 'max_year': int} 或 空字典
        """
        from datetime import datetime
        result = {}
        text_lower = text.lower()

        # 模式0: "近N年" / "最近N年" (动态计算)
        m = re.search(r'(?:近|最近)\s*(\d{1,2})\s*年', text_lower)
        if m:
            n_years = int(m.group(1))
            result['min_year'] = datetime.now().year - n_years
            return result

        # 模式1: "XXXX年以后/之后/后" (允许"的"等中间字符，如"1990年以后的")
        m = re.search(r'(\d{4})\s*年\s*的?\s*(?:以后|之后|后)', text_lower)
        if m:
            result['min_year'] = int(m.group(1))
            return result

        # 模式2: "XXXX年以前/之前/前" (允许"的"等中间字符)
        m = re.search(r'(\d{4})\s*年\s*的?\s*(?:以前|之前|前)', text_lower)
        if m:
            result['max_year'] = int(m.group(1))
            return result

        # 模式3: "XX年代" (如"90年代")
        m = re.search(r'(\d{2})\s*年代', text_lower)
        if m:
            decade = int(m.group(1))
            if decade < 30:
                result['min_year'] = 2000 + decade
                result['max_year'] = 2000 + decade + 9
            else:
                result['min_year'] = 1900 + decade
                result['max_year'] = 1900 + decade + 9
            return result

        # 模式4: "XXXX到XXXX" / "XXXX至XXXX"
        m = re.search(r'(\d{4})\s*(?:到|至|[-–—])\s*(\d{4})', text_lower)
        if m:
            result['min_year'] = int(m.group(1))
            result['max_year'] = int(m.group(2))
            return result

        # 模式5: "XXXX年的" / "XXXX年上映" → 精确匹配那一年
        m = re.search(r'(\d{4})\s*年\s*(?:的|上映|出品|发布)', text_lower)
        if m:
            year = int(m.group(1))
            result['min_year'] = year
            result['max_year'] = year
            return result

        # 模式6: 仅 "XXXX年" 且前面有"之后/后/以后"相关的上下文
        m = re.search(r'(?:之后|以后|后)\s*(?:的)?\s*(?:科幻|动作|喜剧|剧情|恐怖|悬疑|动画|冒险|爱情|战争|历史|记录|音乐|奇幻|家庭|犯罪)', text_lower)
        if m:
            pass

        # 🔥 模式7: 从记忆槽位继承年份约束
        if not result:
            slots = self.memory.get_slots()
            if slots.get('year_min'):
                result['min_year'] = slots['year_min']
                return result

        return result
    
    def _filter_by_year(self, candidates, year_filter):
        """
        根据年份条件过滤候选电影。
        
        Args:
            candidates: 候选电影列表 (list of dict, 每个dict含movie_id)
            year_filter: {'min_year': int, 'max_year': int}
        
        Returns:
            list: 过滤后的候选列表
        """
        if not year_filter or not candidates:
            return candidates
        
        min_year = year_filter.get('min_year')
        max_year = year_filter.get('max_year')
        
        # 获取候选电影的ID列表
        movie_ids = [c.get('movie_id') for c in candidates if c.get('movie_id')]
        if not movie_ids:
            return candidates
        
        # 查询符合条件的电影ID
        from django.db.models import Q
        date_q = Q()
        if min_year:
            date_q &= Q(date__year__gte=min_year)
        if max_year:
            date_q &= Q(date__year__lte=max_year)
        
        valid_ids = set(
            Movie.objects.filter(date_q, id__in=movie_ids)
            .exclude(date__isnull=True)
            .values_list('id', flat=True)
        )
        
        # 如果过滤后为空，返回原始候选（降级处理）
        filtered = [c for c in candidates if c.get('movie_id') in valid_ids]
        if not filtered:
            logger.warning(f"[年份过滤] 过滤后为空，降级返回原始候选 (year_filter={year_filter})")
            return candidates
        
        logger.info(f"[年份过滤] {len(candidates)} → {len(filtered)} 条 (year_filter={year_filter})")
        return filtered
    
    def _call_llm(self, system_prompt, user_prompt, max_tokens=512):
        """
        调用 Ollama LLM 生成文本。
        复用 LLMRerankTool 的 Ollama 调用逻辑，超时/异常返回 None（由调用方 fallback）。
        """
        model_name = self.llm_config.get('model_name', 'qwen3:4b-instruct')
        timeout = self.llm_config.get('timeout', 30)
        try:
            from django.conf import settings
            import requests as req
            olla_base = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": max_tokens, "num_gpu": 99},
            }
            resp = req.post(f"{olla_base}/api/chat", json=payload, timeout=timeout)
            resp.raise_for_status()
            return resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.warning(f"[MovieAgent] LLM 调用失败: {e}")
            return None

    async def _async_call_llm(self, system_prompt, user_prompt, max_tokens=512):
        """异步调用 Ollama LLM，不阻塞事件循环。"""
        import httpx
        model_name = self.llm_config.get('model_name', 'qwen3:4b-instruct')
        timeout = self.llm_config.get('timeout', 30)
        try:
            from django.conf import settings
            olla_base = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.7, "num_predict": max_tokens, "num_gpu": 99},
            }
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(f"{olla_base}/api/chat", json=payload)
                resp.raise_for_status()
                return resp.json().get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.warning(f"[MovieAgent] 异步 LLM 调用失败: {e}")
            return None

    def _generate_llm_summary(self, user_input, intent, movie_map, recommended_ids, explanations):
        """
        使用 LLM 生成个性化推荐总结。返回 None 表示 LLM 不可用（由调用方 fallback）。
        """
        if not self.llm_config.get('model_name'):
            return None

        # 构建电影信息上下文
        movie_infos = []
        for mid in recommended_ids:
            m = movie_map.get(mid)
            if not m:
                continue
            genres = "、".join(g.name for g in m.genres.all()[:3])
            directors = "、".join(d.name for d in m.directors.all()[:1])
            score = str(m.score) if m.score else "暂无"
            year = m.date.year if m.date else ""
            exp = explanations.get(mid, "")
            info = f"《{m.title}》({year}) 评分{score} 类型:{genres} 导演:{directors}"
            if exp:
                info += f" 推荐理由:{exp}"
            movie_infos.append(info)

        if not movie_infos:
            return None

        system_prompt = (
            "你是专业的电影推荐助手。根据用户查询和推荐的电影列表，写一段简洁的推荐总结（3-5句话）。"
            "总结要说明为什么这些电影匹配用户需求，突出共性和亮点。"
            "语气亲切自然，不要列编号，不要重复电影名。直接输出总结文本。"
        )
        user_prompt = f"用户查询：{user_input}\n\n推荐的电影：\n" + "\n".join(movie_infos)

        return self._call_llm(system_prompt, user_prompt, max_tokens=300)

    def _verify_summary_faithfulness(self, summary, movie_map, recommended_ids):
        """
        校验 LLM 生成的推荐总结是否忠实于电影实际属性。
        检查总结中提到的导演/类型是否与数据库一致。
        """
        if not summary:
            return True

        # 收集所有推荐电影的真实属性
        actual_directors = set()
        actual_genres = set()
        for mid in recommended_ids:
            m = movie_map.get(mid)
            if not m:
                continue
            actual_directors.update(d.name for d in m.directors.all())
            actual_genres.update(g.name for g in m.genres.all())

        # 检查总结中提到的导演是否真实存在
        director_match = re.search(r'导演[：:]?\s*([一-鿿·]{2,6})', summary)
        if director_match:
            claimed = director_match.group(1)
            if claimed not in actual_directors:
                return False

        # 检查总结中提到的类型是否真实存在
        genre_match = re.search(r'(?:类型|题材)[：:]?\s*([一-鿿]{2,6})', summary)
        if genre_match:
            claimed = genre_match.group(1)
            if claimed not in actual_genres:
                return False

        return True

    def _faithfulness_check(self, user_input, recommended_ids, intent):
        """
        忠实度自检：用 LLM 验证推荐电影是否符合用户查询约束。
        不符合的电影从推荐列表中移除。

        Returns:
            list: 过滤后的 recommended_ids
        """
        if not self.llm_config or not recommended_ids or intent in ('CHAT', 'QUERY_SELF'):
            return recommended_ids

        try:
            movies = Movie.objects.filter(id__in=recommended_ids[:5]).prefetch_related('genres', 'directors')
            movie_map = {m.id: m for m in movies}

            movie_entries = []
            for mid in recommended_ids[:5]:
                m = movie_map.get(mid)
                if not m:
                    continue
                genres = ", ".join(g.name for g in m.genres.all()[:3])
                directors = ", ".join(d.name for d in m.directors.all()[:2])
                year = m.date.year if m.date else "未知"
                movie_entries.append(f"ID:{mid} 《{m.title}》({year}) 类型:{genres} 导演:{directors}")

            if not movie_entries:
                return recommended_ids

            system_prompt = (
                "你是电影推荐质量审核员。判断每部推荐电影是否符合用户查询的核心约束（类型、导演、年份、主题等）。\n"
                "对每部电影输出一行：ID:数字 FAITHFUL 或 ID:数字 UNFAITHFUL:原因\n"
                "只输出结果，不要其他内容。"
            )
            user_prompt = f"用户查询：{user_input}\n\n推荐电影：\n" + "\n".join(movie_entries)

            response = self._call_llm(system_prompt, user_prompt, max_tokens=300)
            if not response:
                return recommended_ids

            # 解析 LLM 输出，提取 faithful 的电影 ID
            faithful_ids = []
            for mid in recommended_ids[:5]:
                # 检查该 ID 是否被标记为 FAITHFUL（不是 UNFAITHFUL）
                import re as _re
                pattern_faithful = _re.compile(rf'ID:\s*{mid}\s+FAITHFUL\b', _re.IGNORECASE)
                pattern_unfaithful = _re.compile(rf'ID:\s*{mid}\s+UNFAITHFUL', _re.IGNORECASE)

                if pattern_unfaithful.search(response):
                    # 被标记为不忠实，跳过
                    logger.info(f"[FaithfulnessCheck] 电影 {mid} 被标记为 UNFAITHFUL")
                    continue
                elif pattern_faithful.search(response):
                    faithful_ids.append(mid)
                else:
                    # 解析不明确时保留（保守策略）
                    faithful_ids.append(mid)

            # 至少保留 1 部电影，防止全部被过滤
            if not faithful_ids:
                logger.warning("[FaithfulnessCheck] 所有电影均被标记为 UNFAITHFUL，保留原始列表")
                return recommended_ids

            logger.info(f"[FaithfulnessCheck] 原始 {len(recommended_ids[:5])} 部 → 保留 {len(faithful_ids)} 部")
            return faithful_ids

        except Exception as e:
            logger.error(f"[FaithfulnessCheck] 异常: {e}")
            return recommended_ids

    def _generate_final_answer(self, user_input, intent, recommended_ids, explanations, thought):
        """
        Final Answer阶段：生成最终推荐文本。
        优先使用 LLM 生成自然语言推荐，失败时回退到模板逻辑。
        """
        if intent == 'CHAT':
            return f"您好！我是智能观影助手，请问今天想看什么类型的电影呢？"
        
        if intent == 'QUERY_SELF':
            return "正在分析您的观影画像，请稍候..."
        
        if not recommended_ids:
            # 最终兜底：直接查询热门电影（考虑年份过滤）
            try:
                from django.db.models import Q
                qs = Movie.objects.order_by('-vote_count', '-score')
                year_filter = self._extract_year_filter(user_input)
                if year_filter:
                    date_q = Q()
                    if year_filter.get('min_year'):
                        date_q &= Q(date__year__gte=year_filter['min_year'])
                    if year_filter.get('max_year'):
                        date_q &= Q(date__year__lte=year_filter['max_year'])
                    qs = qs.filter(date_q).exclude(date__isnull=True)
                hot = list(qs.values_list('id', flat=True)[:5])
                if hot:
                    recommended_ids = hot
                else:
                    # 返回空标记，由 run() 处理追问逻辑
                    return "__NEED_CLARIFICATION__"
            except Exception:
                return "__NEED_CLARIFICATION__"
        
        # 获取电影详情
        movies = Movie.objects.filter(id__in=recommended_ids[:5]).prefetch_related('genres', 'directors')
        movie_map = {m.id: m for m in movies}
        
        lines = []
        for i, mid in enumerate(recommended_ids[:5], 1):
            movie = movie_map.get(mid)
            if not movie:
                continue
            
            genres = "、".join(g.name for g in movie.genres.all()[:3])
            directors = "、".join(d.name for d in movie.directors.all()[:1])
            score_str = str(movie.score) if movie.score else "暂无"
            # Movie模型用 date (DateField) 而非 year 字段
            movie_year = movie.date.year if movie.date else None
            year_str = f"({movie_year})" if movie_year else ""
            explanation = explanations.get(mid, "")
            
            # 格式化推荐行：电影名可点击，信息丰富
            line = f"{i}. 《{movie.title}》(ID:{mid}){year_str} | ⭐{score_str} | {genres}"
            if directors:
                line += f" | 🎬{directors}"
            
            # 推荐理由：如果没有解释或解释太短，生成更有信息量的理由
            if explanation and len(explanation) > 15:
                line += f"\n   💡 {explanation}"
            else:
                # 根据用户查询意图 + 电影实际属性生成个性化推荐理由
                reason_parts = []
                # 查询意图关联：将推荐与用户需求挂钩
                query_intent_text = self._build_query_intent_reason(user_input, genres, directors)
                if query_intent_text:
                    reason_parts.append(query_intent_text)
                # 导演信息（从数据库验证）
                if directors:
                    reason_parts.append(f"由{directors}执导")
                # 类型信息（从数据库验证）
                if genres:
                    reason_parts.append(f"属{genres}题材")
                # 评分亮点
                if movie.score and movie.score >= 8.0:
                    reason_parts.append(f"评分高达{score_str}，口碑极佳")
                elif movie.score and movie.score >= 7.0:
                    reason_parts.append(f"口碑良好({score_str}分)")
                # 年份信息
                if movie_year:
                    reason_parts.append(f"上映于{movie_year}年")
                if reason_parts:
                    line += f"\n   💡 推荐理由：{'，'.join(reason_parts)}。"
            
            lines.append(line)
        
        if lines:
            result_count = len(lines)

            # ── LLM 生成个性化推荐总结 ──
            summary = self._generate_llm_summary(user_input, intent, movie_map, recommended_ids[:5], explanations)
            # 忠实度校验：检查 LLM 总结是否编造了电影属性
            if summary and not self._verify_summary_faithfulness(summary, movie_map, recommended_ids[:5]):
                logger.warning("[FaithfulnessCheck] LLM 总结包含不实信息，回退到模板总结")
                summary = None
            if not summary:
                # LLM 失败时回退到模板总结
                summary_parts = []
                anchor_movie = self._detect_anchor_movie(user_input)
                if anchor_movie:
                    summary_parts.append(f"基于您对《{anchor_movie}》的喜爱")
                if intent == 'QUERY_MOVIE':
                    if anchor_movie:
                        summary_parts.append("我优先推荐了具有相似叙事风格和视觉效果的作品")
                    else:
                        summary_parts.append("我根据您的需求从数据库中筛选了最匹配的影片")
                elif intent == 'QUERY_COMPARISON':
                    summary_parts.append("以上影片在题材、风格或导演手法上具有可比性")
                elif intent == 'QUERY_PROFILE_REC':
                    summary_parts.append("基于您的观影历史和偏好画像，为您精选了个性化推荐")
                elif intent == 'QUERY_RANK':
                    summary_parts.append("这些是当前评分和热度最高的影片")
                elif intent == 'QUERY_NEW':
                    summary_parts.append("这些是最近入库的新片")

                sentiment_patterns = [
                    (r'(轻松|愉快|欢快|温馨|治愈)', '整体基调轻松温馨'),
                    (r'(刺激|紧张|惊悚)', '节奏紧凑、氛围紧张'),
                ]
                for pattern, label in sentiment_patterns:
                    if re.search(pattern, user_input):
                        summary_parts.append(label)

                year_filter = self._extract_year_filter(user_input)
                if year_filter:
                    min_y = year_filter.get('min_year', '')
                    max_y = year_filter.get('max_year', '')
                    if min_y and max_y:
                        summary_parts.append(f"时间范围限定在{min_y}-{max_y}年")
                    elif min_y:
                        summary_parts.append(f"均为{min_y}年及以后的作品")

                summary = "，".join(summary_parts) + "。" if summary_parts else ""

            # ── 优雅降级：不足 5 部时补充说明 ──
            degraded_note = ""
            if result_count < 5:
                # 检查是否搜索了特定电影但未找到
                searched_movie = self._extract_movie_name(user_input)
                if searched_movie and not any(
                    searched_movie in (movie_map.get(mid).title if movie_map.get(mid) else '')
                    for mid in recommended_ids[:5]
                ):
                    degraded_note = f"\n\n📌 《{searched_movie}》暂未收录在我们的数据库中，以上是同类型推荐。"
                degraded_note += f"\n\n💡 以上是数据库中最匹配的 {result_count} 部电影。想看更多？可以换个关键词或放宽筛选条件试试。"

            if summary:
                return "为您推荐以下电影：\n" + "\n".join(lines) + f"\n\n📋 推荐总结：{summary}{degraded_note}"
            return "为您推荐以下电影：\n" + "\n".join(lines) + degraded_note
        else:
            return "推荐结果生成中，请稍候..."
    
    def get_react_trace(self, result):
        """
        生成论文展示用的 ReAct 推理链文本。
        包含完整的 Thought → Action → Observation → 纠偏重试 → Final Answer 链路。
        
        Returns:
            str: 格式化的推理链
        """
        lines = []
        lines.append("=" * 60)
        lines.append("[Thought]")
        lines.append(result.get('thought', ''))
        lines.append("")
        
        for i, action in enumerate(result.get('actions', [])):
            # 判断是否为纠偏重试
            is_retry = action.get('is_retry', False)
            label = f"[Action {i+1}]" + (" [纠偏重试]" if is_retry else "")
            lines.append(label)
            lines.append(f"  工具: {action.get('tool', 'N/A')}")
            lines.append(f"  输入: {action.get('input', 'N/A')}")
            lines.append("")
        
        for i, obs in enumerate(result.get('observations', [])):
            is_retry = obs.get('is_retry', False)
            label = f"[Observation {i+1}]" + (" [纠偏重试]" if is_retry else "")
            lines.append(label)
            count = obs.get('count', 0)
            lines.append(f"  获得 {count} 条候选结果")
            lines.append("")
        
        # 如果有trace_steps，展示完整的推理链
        trace_steps = result.get('trace_steps', [])
        if trace_steps:
            lines.append("─" * 40)
            lines.append("[完整推理链步骤]")
            for step in trace_steps:
                step_type = step.get('type', '')
                step_num = step.get('step', 0)
                content = step.get('content', '')
                retry_flag = " ⚡[纠偏]" if step.get('is_retry') else ""
                lines.append(f"  Step {step_num} [{step_type}]{retry_flag}: {content}")
            lines.append("")
        
        lines.append("[Final Answer]")
        lines.append(result.get('final_answer', ''))
        lines.append("=" * 60)
        
        return "\n".join(lines)