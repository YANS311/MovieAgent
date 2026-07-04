"""
KnowledgeRetrievalSkill — 能力导向的统一知识检索 Skill（v2）
=================================================
设计思路:
  Skill 表示 Capability（能力），而不是 Tool（动作）。
  本 Skill 封装 SQL → Neo4j → Vector → Merge 的完整检索管线，
  返回统一的候选结果。

  与 VectorSearchSkill / GraphReasoningSkill 的区别:
    - 后者是单工具封装（Tool-oriented）
    - 本 Skill 是能力封装（Capability-oriented）

  当前为适配层，内部调用已有 Skill。
  后续可替换为真正的融合检索逻辑。
=================================================
"""

import time
import logging
from .base import BaseSkill

logger = logging.getLogger('movie_agent')


class KnowledgeRetrievalSkill(BaseSkill):
    """统一知识检索能力（SQL + Neo4j + Vector + Merge）。"""

    name = "knowledge_retrieval"
    description = "统一知识检索：SQL 条件查询 + Neo4j 图谱推理 + FAISS 向量语义 + 结果融合"
    version = "1.0.0"
    priority = 80
    latency_level = "medium"
    cost_level = "medium"
    tags = ["retrieval", "unified", "capability"]
    examples = [
        {"input": {"query": "推荐诺兰的科幻片"}, "output": "融合后的候选列表"},
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "user": {"description": "用户对象"},
            "top_k": {"type": "integer", "default": 15},
        },
        "required": ["query"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "sources": {"type": "array"},
            "count": {"type": "integer"},
        },
    }

    def __init__(self, registry=None, neo_graph=None, rag_resources=None):
        self.registry = registry
        self.neo_graph = neo_graph
        self.rag_resources = rag_resources

    def can_handle(self, context) -> bool:
        if hasattr(context, 'intent'):
            intent = context.intent
        else:
            intent = context.get('intent', '')
        return intent in ('QUERY_MOVIE', 'QUERY_COMPARISON', 'QUERY_PROFILE_REC', 'RECOMMEND')

    def run(self, context) -> dict:
        t0 = time.time()

        if hasattr(context, 'user_input'):
            query = context.user_input
            top_k = context.metadata.get('top_k', 15) if hasattr(context, 'metadata') else 15
        else:
            query = context.get('query', '')
            top_k = context.get('top_k', 15)

        all_candidates = []
        sources = []

        # 1. SQL 条件查询
        sql_results = self._sql_retrieve(query)
        if sql_results:
            all_candidates.extend(sql_results)
            sources.append('sql')

        # 2. Neo4j 图谱推理
        kg_results = self._kg_retrieve(query, context)
        if kg_results:
            all_candidates.extend(kg_results)
            sources.append('neo4j')

        # 3. FAISS 向量语义
        vector_results = self._vector_retrieve(query)
        if vector_results:
            all_candidates.extend(vector_results)
            sources.append('vector')

        # 4. 去重融合（按 movie_id 去重，保留最高 score）
        merged = self._merge_results(all_candidates, top_k)

        # 5. 兜底
        if not merged:
            merged = self._hot_fallback(top_k)
            sources.append('hot_fallback')

        elapsed = time.time() - t0

        return self._success(
            data=merged,
            sources=sources,
            count=len(merged),
            elapsed=f"{elapsed:.3f}s",
        )

    def _sql_retrieve(self, query: str) -> list:
        """SQL 条件查询。"""
        try:
            from myapp.recommender.recall import content_recall
            return content_recall(query, k=30) or []
        except Exception as e:
            logger.debug(f"[KnowledgeRetrieval] SQL retrieve failed: {e}")
            return []

    def _kg_retrieve(self, query: str, context) -> list:
        """Neo4j 图谱查询。"""
        try:
            if hasattr(context, 'constraints'):
                constraints = context.constraints
            else:
                constraints = context.get('constraints', {})

            from myapp.agent.movie_agent import KGQueryTool
            tool = KGQueryTool(neo_graph=self.neo_graph)
            result = tool.execute(query=query, constraints=constraints)
            return result.get('output', []) or []
        except Exception as e:
            logger.debug(f"[KnowledgeRetrieval] KG retrieve failed: {e}")
            return []

    def _vector_retrieve(self, query: str) -> list:
        """FAISS 向量检索。"""
        try:
            from myapp.recommender.recall import vector_recall
            return vector_recall(query, k=30, rag_resources=self.rag_resources) or []
        except Exception as e:
            logger.debug(f"[KnowledgeRetrieval] Vector retrieve failed: {e}")
            return []

    def _merge_results(self, candidates: list, top_k: int) -> list:
        """按 movie_id 去重，保留最高 score，截取 top_k。"""
        seen = {}
        for item in candidates:
            mid = item.get('movie_id') or item.get('mid')
            if not mid:
                continue
            score = item.get('score', 0)
            if mid not in seen or score > seen[mid].get('score', 0):
                seen[mid] = item
        merged = sorted(seen.values(), key=lambda x: x.get('score', 0), reverse=True)
        return merged[:top_k]

    def _hot_fallback(self, top_k: int) -> list:
        """热门兜底。"""
        try:
            from myapp.recommender.recall import hot_recall
            return hot_recall(k=top_k) or []
        except Exception:
            return []

    def fallback(self, context, error: Exception) -> dict:
        top_k = context.metadata.get('top_k', 15) if hasattr(context, 'metadata') else context.get('top_k', 15)
        results = self._hot_fallback(top_k)
        return {
            'skill': self.name,
            'success': True,
            'data': results,
            'meta': {'fallback': True, 'source': 'hot_fallback', 'error': str(error)},
        }
