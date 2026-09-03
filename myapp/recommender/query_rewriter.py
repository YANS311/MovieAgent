"""
Query Rewrite 模块 (Query Rewriter Module)
================================================
通过 LLM 语义重写与启发式规则兜底，对用户输入的简单或模糊查询进行扩充、实体补全及同义词拓展，
提升下游倒排 (BM25) 与向量 (FAISS) 检索的召回率。

核心功能：
  1. LLM 语义扩充 (Query Expansion)
  2. 领域实体同义词补全 (Domain Synonyms & Entities)
  3. 异常与超时自动规则降级 (Rule-based Fallback)
================================================
"""

import re
import logging
from django.core.cache import cache

logger = logging.getLogger('movie_agent')

# 领域常见映射与扩充字典 (用于规则兜底)
GENRE_EXPANSION = {
    '科幻': '科幻 太空 未来 时空 穿越 人工智能 宇宙 机器人',
    '悬疑': '悬疑 推理 烧脑 谋杀 惊悚 谜题 解密 剧情',
    '恐怖': '恐怖 惊悚 鬼怪 灵异 吓人 血腥 诡异',
    '喜剧': '喜剧 搞笑 幽默 爆笑 轻松 荒诞 搞笑片',
    '动作': '动作 打斗 枪战 功夫 武侠 漫威 刺激 冒险',
    '爱情': '爱情 浪漫 甜 催泪 情感 情侣 动人 剧情',
    '动画': '动画 动漫 二次元 宫崎骏 迪士尼 奇幻 治愈',
    '战争': '战争 史诗 军旅 战斗 历史 枪战 英雄',
    '犯罪': '犯罪 黑帮 警匪 缉毒 刑侦 谋杀 法律',
}

DIRECTOR_ALIASES = {
    '诺兰': '克里斯托弗·诺兰 Christopher Nolan',
    '宫崎骏': '宫崎骏 Miyazaki',
    '斯皮尔伯格': '史蒂文·斯皮尔伯格 Steven Spielberg',
    '昆汀': '昆汀·塔伦蒂诺 Quentin Tarantino',
    '芬奇': '大卫·芬奇 David Fincher',
    '卡梅隆': '詹姆斯·卡梅隆 James Cameron',
}


class QueryRewriter:
    """
    检索前置 Query 重写与拓展器
    """

    def __init__(self, model_name="qwen3:4b-instruct", use_llm=True):
        self.model_name = model_name
        self.use_llm = use_llm
        self._llm_instance = None

    def _get_llm(self):
        """延迟加载 Ollama 模型"""
        if not self.use_llm:
            return None
        if self._llm_instance is None:
            try:
                from langchain_ollama import ChatOllama
                self._llm_instance = ChatOllama(
                    model=self.model_name,
                    temperature=0.3,
                    request_timeout=3.0,  # 快速响应，防止拖慢检索
                )
            except Exception as e:
                logger.warning(f"[QueryRewriter] LLM 初始化失败，将使用规则模式: {e}")
                self.use_llm = False
        return self._llm_instance

    def rewrite(self, query: str) -> str:
        """
        对查询进行拓展，返回适合高召回检索的拓展查询字符串。
        """
        if not query or not query.strip():
            return ""

        query_clean = query.strip()

        # 1. 尝试缓存命中
        cache_key = f"query_rewrite_{hash(query_clean)}"
        cached_result = cache.get(cache_key)
        if cached_result:
            return cached_result

        rewritten = ""

        # 2. 尝试 LLM 重写
        llm = self._get_llm()
        if llm:
            try:
                prompt = (
                    "你是一个电影检索系统前置查询拓展器。请将用户的自然语言查询转换为适合文本与向量检索的关键词组合。\n"
                    "要求：输出包含原词、同义词、相关电影类型、风格词、核心主题等关键词，用空格分隔，不要回答其他解释性内容。\n"
                    f"用户输入: {query_clean}\n"
                    "拓展关键词:"
                )
                res = llm.invoke(prompt)
                res_text = res.content.strip() if hasattr(res, 'content') else str(res).strip()
                # 过滤可能的思考前缀
                if "</think>" in res_text:
                    res_text = res_text.split("</think>")[-1].strip()
                if res_text and len(res_text) > len(query_clean):
                    rewritten = res_text
            except Exception as e:
                logger.debug(f"[QueryRewriter] LLM 改写超时或异常，切至规则降级: {e}")

        # 3. 规则兜底 / 规则补充
        if not rewritten:
            rewritten = self._rule_based_rewrite(query_clean)

        # 4. 写入缓存 (1小时)
        cache.set(cache_key, rewritten, 3600)
        return rewritten

    def _rule_based_rewrite(self, query: str) -> str:
        """规则兜底拓展逻辑"""
        terms = [query]

        # 类型同义词拓展
        for genre_key, expanded in GENRE_EXPANSION.items():
            if genre_key in query:
                terms.append(expanded)

        # 导演别名/英文名拓展
        for dir_key, alias in DIRECTOR_ALIASES.items():
            if dir_key in query:
                terms.append(alias)

        # 相似/风格词匹配
        if "类似" in query or "像" in query or "风格" in query:
            target = re.sub(r'(推荐|有没有|类似|像|那样的|风格|的电影|片)', '', query).strip()
            if target:
                terms.append(f"{target} 类似 同类型 推荐 经典 风格")

        if "高分" in query or "经典" in query:
            terms.append("高分 经典 必看 口碑 榜单 佳作")

        return " ".join(dict.fromkeys(" ".join(terms).split()))


_query_rewriter_instance = None


def get_query_rewriter():
    """获取单例 QueryRewriter"""
    global _query_rewriter_instance
    if _query_rewriter_instance is None:
        _query_rewriter_instance = QueryRewriter()
    return _query_rewriter_instance
