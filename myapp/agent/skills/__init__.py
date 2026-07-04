"""
MovieAgent Skill 抽象层
=================================================
轻量级 Skill 接口，为后续 Multi-Agent 和 MCP Adapter 做准备。

使用方式:
    from myapp.agent.skills import SkillRegistry, VectorSearchSkill

    registry = SkillRegistry()
    registry.register(VectorSearchSkill(rag_resources=...))

    # 按名称调用
    skill = registry.get("search_vector")
    result = skill.run({"query": "科幻电影"})

    # 自动选择
    skill = registry.select({"intent": "QUERY_MOVIE"})
=================================================
"""

from .base import BaseSkill
from .registry import SkillRegistry
from .vector_search_skill import VectorSearchSkill
from .graph_reasoning_skill import GraphReasoningSkill
from .hybrid_recall_skill import HybridRecallSkill
from .neural_rerank_skill import NeuralRerankSkill
from .explanation_skill import ExplanationSkill

__all__ = [
    'BaseSkill',
    'SkillRegistry',
    'VectorSearchSkill',
    'GraphReasoningSkill',
    'HybridRecallSkill',
    'NeuralRerankSkill',
    'ExplanationSkill',
]
