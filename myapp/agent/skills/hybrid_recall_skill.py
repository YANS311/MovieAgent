"""
HybridRecallSkill — 多路混合召回 Skill（v2）
=================================================
封装五路并行召回（向量+内容+模型+图谱+热门）。
对应原有 AgentTool: RecallHybridTool
=================================================
"""

from .base import BaseSkill


class HybridRecallSkill(BaseSkill):
    """多路并行召回融合。"""

    name = "recall_hybrid"
    description = "五路并行召回（向量语义+内容特征+深度模型+知识图谱+热门兜底）"
    version = "2.0.0"
    priority = 95
    latency_level = "medium"
    cost_level = "medium"
    tags = ["retrieval", "hybrid", "multi-channel"]
    examples = [
        {"input": {"query": "推荐好看的电影"}, "output": "多路融合候选"},
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "user": {"description": "用户对象"},
            "query_text": {"type": "string"},
            "top_k": {"type": "integer", "default": 15},
        },
    }

    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "stats": {"type": "object"},
            "count": {"type": "integer"},
        },
    }

    def __init__(self, neo_graph=None, rag_resources=None):
        self.neo_graph = neo_graph
        self.rag_resources = rag_resources

    def can_handle(self, context) -> bool:
        if hasattr(context, 'intent'):
            intent = context.intent
        else:
            intent = context.get('intent', '')
        return intent in ('QUERY_PROFILE_REC', 'QUERY_MOVIE', 'RECOMMEND')

    def run(self, context) -> dict:
        import time
        t0 = time.time()

        if hasattr(context, 'user'):
            user = context.user
            query_text = context.user_input or ''
            top_k = context.metadata.get('top_k', 15) if hasattr(context, 'metadata') else 15
        else:
            user = context.get('user')
            query_text = context.get('query_text') or context.get('query', '')
            top_k = context.get('top_k', 15)

        from myapp.recommender.recall import multi_channel_recall, hot_recall

        try:
            results, stats = multi_channel_recall(
                user, query_text=query_text, top_k=top_k,
                neo_graph=self.neo_graph, rag_resources=self.rag_resources,
            )
        except Exception as e:
            results = hot_recall(k=top_k)
            stats = {'fallback': 'hot_due_to_error', 'error': str(e)}

        elapsed = time.time() - t0

        return self._success(
            data=results,
            stats=stats,
            count=len(results),
            elapsed=f"{elapsed:.3f}s",
        )

    def fallback(self, context, error: Exception) -> dict:
        try:
            from myapp.recommender.recall import hot_recall
            top_k = context.metadata.get('top_k', 15) if hasattr(context, 'metadata') else context.get('top_k', 15)
            results = hot_recall(k=top_k)
            return {
                'skill': self.name,
                'success': True,
                'data': results,
                'meta': {'fallback': True, 'source': 'hot_fallback', 'error': str(error)},
            }
        except Exception:
            return super().fallback(context, error)
