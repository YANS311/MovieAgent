"""
ExplanationSkill — 推荐理由生成 Skill
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

    input_schema = {
        "type": "object",
        "properties": {
            "user": {"description": "用户对象"},
            "movie_id": {"type": "integer", "description": "目标电影ID"},
            "enable_kag": {
                "type": "boolean",
                "description": "是否启用知识图谱增强",
                "default": True,
            },
        },
        "required": ["user", "movie_id"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "reason_text": {"type": "string", "description": "推荐理由文本"},
            "reason_type": {"type": "string", "description": "理由类型"},
            "strength": {"type": "number", "description": "推荐强度"},
            "kg_path": {"type": "array", "description": "知识图谱归因路径"},
        },
    }

    def __init__(self, neo_graph=None, enable_kag=True):
        self.neo_graph = neo_graph
        self.enable_kag = enable_kag

    def can_handle(self, context: dict) -> bool:
        # 有 user 和 movie_id 就能生成解释
        return context.get('user') is not None and context.get('movie_id') is not None

    def run(self, context: dict) -> dict:
        import time
        t0 = time.time()

        user = context['user']
        movie_id = context['movie_id']

        # 复用原有 ExplainTool 的逻辑
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

    def fallback(self, context: dict, error: Exception) -> dict:
        """降级：返回通用推荐理由。"""
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
