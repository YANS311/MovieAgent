"""
GraphReasoningSkill — 知识图谱推理 Skill
=================================================
封装 Neo4j 图谱查询 + NL2Cypher 逻辑。
对应原有 AgentTool: KGQueryTool
=================================================
"""

from .base import BaseSkill


class GraphReasoningSkill(BaseSkill):
    """知识图谱推理查询（Neo4j + NL2Cypher）。"""

    name = "kg_query"
    description = "将自然语言约束动态转换为 Cypher 查询知识图谱"

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "自然语言查询"},
            "constraints": {
                "type": "object",
                "description": "结构化约束（genre/director/actor/year/rating）",
            },
            "anchor_mid": {"type": "integer", "description": "锚点电影ID（可选）"},
        },
        "required": ["query"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "triples": {"type": "array", "description": "图谱三元组"},
            "cypher": {"type": "string", "description": "生成的 Cypher 查询"},
            "count": {"type": "integer"},
        },
    }

    def __init__(self, neo_graph=None):
        self.neo_graph = neo_graph

    def can_handle(self, context: dict) -> bool:
        intent = context.get('intent', '')
        return intent in ('QUERY_KG', 'QUERY_MOVIE', 'QUERY_COMPARISON')

    def run(self, context: dict) -> dict:
        import time
        t0 = time.time()

        query = context.get('query', '')
        constraints = context.get('constraints', {})
        anchor_mid = context.get('anchor_mid')

        # 复用原有 KGQueryTool 的逻辑
        from myapp.agent.movie_agent import KGQueryTool
        tool = KGQueryTool(neo_graph=self.neo_graph)

        result = tool.execute(
            query=query,
            constraints=constraints,
            anchor_mid=anchor_mid,
        )

        elapsed = time.time() - t0

        return self._success(
            data=result.get('output', []),
            triples=result.get('triples', []),
            cypher=result.get('cypher', ''),
            count=result.get('count', 0),
            elapsed=f"{elapsed:.3f}s",
        )
