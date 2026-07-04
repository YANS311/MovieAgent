"""
意图路由策略模式重构
================================================
将 views.py 中臃肿的 Prompt 构建逻辑（God Object）
拆分为独立的策略类，遵循 Strategy Pattern + Factory Pattern。

升级动机（方案四）：
  - 原实现：prompt_builders 字典 + 庞大私有函数 → views.py 成为神仙文件
  - 新实现：每个 Intent 独立为一个 PromptBuilder 类，统一接口

使用方式：
    from myapp.utils.intent_strategies import get_strategy
    strategy = get_strategy("QUERY_MOVIE")
    result = strategy.build(user, user_input, context)
    # result = {"visual_response": None, "final_prompt": "...", "temperature": 0.3}
================================================
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

logger = logging.getLogger('movie_agent')


# =============================================================
# 基类定义
# =============================================================

class PromptStrategy(ABC):
    """
    Prompt 构建策略基类。
    所有意图策略必须继承此类并实现 build() 方法。
    """
    
    intent_name: str = "UNKNOWN"
    
    @abstractmethod
    def build(self, user, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        构建该意图对应的 Prompt。
        
        Args:
            user: Django User 对象
            user_input: 用户原始输入
            context: 上下文信息字典，包含：
                - interaction_summary: 用户画像
                - is_thinking_mode: 是否深度思考
                - search_query: 搜索关键词
                - is_follow_up: 是否追问
                - history: 对话历史
        
        Returns:
            dict: {
                "visual_response": str or None,  # 直接返回的视觉HTML
                "final_prompt": str or None,     # 发送给 LLM 的 Prompt
                "temperature": float             # LLM 温度参数
            }
        """
        pass
    
    def _get_user_history_mids(self, user, limit=20):
        """获取用户历史高分电影 ID 列表"""
        try:
            from myapp.models import UserRating
            return list(
                UserRating.objects.filter(user=user, score__gte=7.5)
                .values_list('movie__id', flat=True)[:limit]
            )
        except Exception:
            return []


# =============================================================
# 具体策略实现
# =============================================================

class QueryMovieStrategy(PromptStrategy):
    """
    电影推荐策略（核心策略）
    融合向量语义 + 知识图谱拓扑双轨检索
    """
    intent_name = "QUERY_MOVIE"
    
    def build(self, user, user_input, context):
        search_query = context.get('search_query', user_input)
        is_thinking_mode = context.get('is_thinking_mode', False)
        is_follow_up = context.get('is_follow_up', False)
        interaction_summary = context.get('interaction_summary')
        
        # 语义脱水
        stop_words = ['再', '还', '继续', '换', '类似', '别的', '其他', '再来', '推荐', '几部', '关于', '电影', '类似的', '的说']
        clean_topic = search_query
        for word in stop_words:
            clean_topic = clean_topic.replace(word, "")
        clean_topic = clean_topic.strip() or "电影"
        
        # 强化去重
        from myapp.utils.llm_output_parser import extract_movie_ids_from_text
        from myapp.views import get_recommended_movies_from_history, query_vector_rag, get_kg_subgraph, _build_user_prior_context
        
        ignore_titles = get_recommended_movies_from_history(user, limit=10)
        user_history_mids = self._get_user_history_mids(user)
        
        # 双轨检索
        vector_context = query_vector_rag(search_query, k=15) or ""
        kg_context = get_kg_subgraph(clean_topic, user_history_mids=user_history_mids)
        
        # 用户认知画像 + 职业专家先验
        user_prior_section = _build_user_prior_context(user, interaction_summary)
        
        # 动态视觉锚点
        visual_anchors = "机械、芯片、实验室、冷色调、电子元件、金属" if "AI" in clean_topic.upper() or "人工" in clean_topic else \
            "宇航服、星空、飞船、虚空、精密仪器" if "太空" in clean_topic else "相关风格元素"
        
        # 知识图谱区块
        kg_section = ""
        if kg_context:
            kg_section = f"""
【知识图谱推理路径】（格式：实体--[关系]-->实体，来自 Neo4j 拓扑遍历）：
{kg_context}

⚡ 图谱节点推理权重（优先级从高到低）：
① 导演关联 Director-Link【最高权重 ★★★★★】
② 历史对齐 History-Base【高权重 ★★★★】
③ 类型桥接 Genre-Bridge【中权重 ★★★】
④ 演员关联 Actor-Link【低权重 ★★】
⑤ 地区关联 Region-Link【极低权重 ★】

🔒 表达约束：推荐理由必须自然融入图谱连线信息，严禁使用内部词语。
"""
        
        # 构建完整 Prompt
        kag_instruction = f"""
💡 【KAG 推理引擎 V201】：
- **核心主题**：用户正在寻找关于"{clean_topic}"的电影。
- **排除红线**：绝对禁止推荐：[{'、'.join(ignore_titles) or '无'}]。
- **视觉共鸣**：若简介不匹配，查看 [海报视觉描述]。包含（{visual_anchors}）即视为符合。
- **图谱优先**：若图谱路径中存在合适候选，优先推荐。
- **关键约束**：推荐信息必须全部来源于提供的资料。
"""
        
        fact_lock = """
🛑 【事实一致性 — 零容忍】：
1. 严禁引用不在资料中出现的电影、导演或演员！
2. 导演信息必须来源于提供的资料。
3. 资料不足时宁可少推荐，绝不凭空捏造。
4. 理由必须直接追溯到提供的资料。
"""
        
        # Few-shot 通用示例
        general_fewshot = """
【通用推荐示例（严格仿照此风格输出）】：

▸ 示例1（同导演关联）：
  《信条》(ID:789)：与您喜爱的《星际穿越》同为诺兰执导，延续了其独特的非线性叙事与冷色调视觉美学。

▸ 示例2（同类型关联）：
  《银翼杀手2049》(ID:321)：本片与您心目中的经典《黑客帝国》同属赛博朋克题材，视觉美学登峰造极。

▸ 示例3（情感基调共鸣）：
  《绿皮书》(ID:654)：基于您对《肖申克的救赎》的高分评价，本片在温暖治愈的情感基调上与您高度契合。
"""
        
        final_prompt = f"""你是一位专业的影库专家，能够融合语义检索与知识图谱拓扑进行深度推荐。

⚠️ 【严格事实验证】：在输出推荐前，你必须验证电影名称、导演信息是否出现在提供的资料中。

{user_prior_section}
{kg_section}
【向量语义资料】：
{vector_context[:1200]}

{kag_instruction}
{fact_lock}
{general_fewshot}

📋 任务：推荐最多 3 部不在排除名单内的电影，必须带 ID。

📤 输出格式：
1. 《电影名》(ID:xxx)：推荐理由
2. 《电影名》(ID:xxx)：推荐理由
... （最多 3 部）
"""
        
        if is_thinking_mode:
            final_prompt += "\n【重要】请先使用 <think> 和 </think> 标签包裹分析推理过程。思考结束后输出推荐。\n"
            temperature = 0.6
        else:
            final_prompt += "\n请直接输出推荐结果，不要输出推理过程。\n"
            temperature = 0.5 if is_follow_up else 0.2
        
        return {"visual_response": None, "final_prompt": final_prompt, "temperature": temperature}


class QueryRankStrategy(PromptStrategy):
    """排行榜策略"""
    intent_name = "QUERY_RANK"
    
    def build(self, user, user_input, context):
        from myapp.models import Movie
        if '高分' in user_input:
            movies = Movie.objects.filter(vote_count__gt=1000).order_by('-score')[:5]
        else:
            movies = Movie.objects.order_by('-vote_count')[:5]
        
        movie_list = "\n".join([f"- 《{m.title}》 (ID:{m.id}) 评分:{m.score}" for m in movies])
        final_prompt = f"实时榜单数据如下：\n{movie_list}\n请以智能观影助手身份进行极简点评，引导用户点击观看。"
        return {"visual_response": None, "final_prompt": final_prompt, "temperature": 0.3}


class QueryNewStrategy(PromptStrategy):
    """最新电影策略"""
    intent_name = "QUERY_NEW"
    
    def build(self, user, user_input, context):
        from myapp.models import Movie
        movies = Movie.objects.filter(date__isnull=False).order_by('-date')[:5]
        movie_list = "\n".join([f"- 《{m.title}》 (ID:{m.id}) 上映日:{m.date}" for m in movies])
        final_prompt = f"最新入库的电影如下：\n{movie_list}\n请以智能观影助手身份热情安利这些新片。"
        return {"visual_response": None, "final_prompt": final_prompt, "temperature": 0.3}


class ChatStrategy(PromptStrategy):
    """闲聊策略"""
    intent_name = "CHAT"
    
    def build(self, user, user_input, context):
        final_prompt = f"你是智能观影助手。请用一句话幽默地回应用户：'{user_input}'，引导他询问影片推荐。"
        return {"visual_response": None, "final_prompt": final_prompt, "temperature": 0.6}


class QueryComparisonStrategy(QueryMovieStrategy):
    """对比策略（继承推荐策略）"""
    intent_name = "QUERY_COMPARISON"


# =============================================================
# 策略工厂
# =============================================================

_STRATEGY_REGISTRY = {
    "QUERY_MOVIE": QueryMovieStrategy(),
    "QUERY_COMPARISON": QueryComparisonStrategy(),
    "QUERY_RANK": QueryRankStrategy(),
    "QUERY_NEW": QueryNewStrategy(),
    "CHAT": ChatStrategy(),
}


def get_strategy(intent: str) -> Optional[PromptStrategy]:
    """
    根据意图标签获取对应的策略实例。
    
    Args:
        intent: 意图标签，如 "QUERY_MOVIE"
    
    Returns:
        PromptStrategy 实例或 None
    """
    strategy = _STRATEGY_REGISTRY.get(intent)
    if strategy is None:
        logger.warning(f"[StrategyFactory] 未找到意图 '{intent}' 对应的策略")
    return strategy


def register_strategy(intent: str, strategy: PromptStrategy):
    """
    动态注册新策略（用于插件化扩展）。
    """
    _STRATEGY_REGISTRY[intent] = strategy
    logger.info(f"[StrategyFactory] 已注册策略: {intent} → {strategy.__class__.__name__}")