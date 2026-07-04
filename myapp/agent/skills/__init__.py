"""
MovieAgent Skill 抽象层（v2）
=================================================
轻量级 Skill 接口，支持:
  - SkillContext 统一上下文
  - Skill Metadata（priority, latency, cost, tags）
  - SkillRouter 路由
  - SkillMetrics 指标

使用方式:
    from myapp.agent.skills import SkillRegistry, VectorSearchSkill

    registry = SkillRegistry()
    registry.register(VectorSearchSkill(rag_resources=...))

    # 按名称调用
    skill = registry.get("search_vector")
    result = skill.run(context)

    # 按标签选择
    skills = registry.select_by_tags(["retrieval"])
=================================================
"""

from .base import BaseSkill
from .registry import SkillRegistry
from .vector_search_skill import VectorSearchSkill
from .graph_reasoning_skill import GraphReasoningSkill
from .hybrid_recall_skill import HybridRecallSkill
from .neural_rerank_skill import NeuralRerankSkill
from .explanation_skill import ExplanationSkill
from .knowledge_retrieval_skill import KnowledgeRetrievalSkill

__all__ = [
    'BaseSkill',
    'SkillRegistry',
    'VectorSearchSkill',
    'GraphReasoningSkill',
    'HybridRecallSkill',
    'NeuralRerankSkill',
    'ExplanationSkill',
    'KnowledgeRetrievalSkill',
]
