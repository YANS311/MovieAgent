"""
ExplanationSkill — 推荐理由生成 Skill（v2）
=================================================
封装推荐可解释性逻辑（知识图谱归因 + 用户画像匹配）。
对应原有 AgentTool: ExplainTool
=================================================
"""

from .base import BaseSkill


class ExplanationSkill(BaseSkill):
    """生成个性化推荐理由（含知识图谱归因）。"""

    name = "explain"
    description = "生成推荐理由（含知识图谱归因 + 用户画像匹配）"
    version = "2.0.0"
    priority = 60
    latency_level = "medium"
    cost_level = "low"
    tags = ["explanation", "xai", "knowledge"]
    examples = [
        {"input": {"movie_id": 1}, "output": "推荐理由文本"},
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "user": {"description": "用户对象"},
            "movie_id": {"type": "integer"},
            "enable_kag": {"type": "boolean", "default": True},
        },
        "required": ["user", "movie_id"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "reason_text": {"type": "string"},
            "reason_type": {"type": "string"},
            "strength": {"type": "number"},
        },
    }

    def __init__(self, neo_graph=None, enable_kag=True):
        self.neo_graph = neo_graph
        self.enable_kag = enable_kag

    def can_handle(self, context) -> bool:
        if hasattr(context, 'user') and hasattr(context, 'metadata'):
            return context.user is not None and context.metadata.get('current_movie_id') is not None
        return context.get('user') is not None and context.get('movie_id') is not None

    def run(self, context) -> dict:
        import time
        t0 = time.time()

        if hasattr(context, 'user'):
            user = context.user
            movie_id = context.metadata.get('current_movie_id')
        else:
            user = context.get('user')
            movie_id = context.get('movie_id')

        from myapp.agent.movie_agent import ExplainTool
        tool = ExplainTool(neo_graph=self.neo_graph, enable_kag=self.enable_kag)
        result = tool.execute(user=user, movie_id=movie_id)

        elapsed = time.time() - t0

        return self._success(
            data={
                'reason_text': result.get('reason_text', ''),
                'reason_type': result.get('reason_type', ''),
                'strength': result.get('strength', 0),
                'kg_path': result.get('kg_path', []),
            },
            elapsed=f"{elapsed:.3f}s",
        )

    def fallback(self, context, error: Exception) -> dict:
        if hasattr(context, 'metadata'):
            movie_id = context.metadata.get('current_movie_id')
        else:
            movie_id = context.get('movie_id')
        try:
            from myapp.models import Movie
            movie = Movie.objects.get(mid=movie_id)
            reason = f"《{movie.title}》是一部值得一看的电影。"
        except Exception:
            reason = "这是一部值得推荐的电影。"

        return {
            'skill': self.name,
            'success': True,
            'data': {
                'reason_text': reason,
                'reason_type': 'generic',
                'strength': 0.5,
                'kg_path': [],
            },
            'meta': {'fallback': True, 'error': str(error)},
        }
