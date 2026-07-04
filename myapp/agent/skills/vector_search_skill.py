"""
VectorSearchSkill — 向量语义搜索 Skill（v2）
=================================================
封装 FAISS 向量召回 + 热门兜底逻辑。
对应原有 AgentTool: SearchVectorTool
=================================================
"""

from .base import BaseSkill


class VectorSearchSkill(BaseSkill):
    """基于 FAISS 向量语义相似度的电影搜索。"""

    name = "search_vector"
    description = "基于语义相似度搜索电影（FAISS + BGE 向量召回，热门兜底）"
    version = "2.0.0"
    priority = 90
    latency_level = "low"
    cost_level = "low"
    tags = ["retrieval", "semantic", "vector"]
    examples = [
        {"input": {"query": "推荐科幻电影"}, "output": "候选电影列表"},
        {"input": {"query": "类似盗梦空间的"}, "output": "语义相似电影"},
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索文本"},
            "k": {"type": "integer", "description": "召回数量", "default": 10},
        },
        "required": ["query"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "count": {"type": "integer"},
            "source": {"type": "string"},
        },
    }

    def __init__(self, rag_resources=None):
        self.rag_resources = rag_resources

    def can_handle(self, context) -> bool:
        if hasattr(context, 'intent'):
            intent = context.intent
        else:
            intent = context.get('intent', '')
        return intent in ('QUERY_MOVIE', 'QUERY_COMPARISON', 'QUERY_PROFILE_REC')

    def run(self, context) -> dict:
        import time
        t0 = time.time()

        if hasattr(context, 'user_input'):
            query = context.user_input
            k = context.metadata.get('k', 10) if hasattr(context, 'metadata') else 10
        else:
            query = context.get('query', '')
            k = context.get('k', 10)

        from myapp.recommender.recall import vector_recall, hot_recall

        results = vector_recall(query, k=k, rag_resources=self.rag_resources)
        source = 'vector'

        if not results:
            results = hot_recall(k=k)
            source = 'hot_fallback'

        elapsed = time.time() - t0

        return self._success(
            data=results,
            count=len(results),
            source=source,
            elapsed=f"{elapsed:.3f}s",
        )

    def fallback(self, context, error: Exception) -> dict:
        try:
            from myapp.recommender.recall import hot_recall
            k = context.metadata.get('k', 10) if hasattr(context, 'metadata') else context.get('k', 10)
            results = hot_recall(k=k)
            return {
                'skill': self.name,
                'success': True,
                'data': results,
                'meta': {'fallback': True, 'source': 'hot_fallback', 'error': str(error)},
            }
        except Exception:
            return super().fallback(context, error)
