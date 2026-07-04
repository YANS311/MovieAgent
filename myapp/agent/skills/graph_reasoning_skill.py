"""
GraphReasoningSkill — 知识图谱推理 Skill（v2）
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
    version = "2.0.0"
    priority = 70
    latency_level = "medium"
    cost_level = "low"
    tags = ["retrieval", "graph", "knowledge"]
    examples = [
        {"input": {"query": "诺兰导演的科幻片"}, "output": "图谱三元组"},
    ]

    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "自然语言查询"},
            "constraints": {"type": "object"},
            "anchor_mid": {"type": "integer"},
        },
        "required": ["query"],
    }

    output_schema = {
        "type": "object",
        "properties": {
            "triples": {"type": "array"},
            "cypher": {"type": "string"},
            "count": {"type": "integer"},
        },
    }

    def __init__(self, neo_graph=None):
        self.neo_graph = neo_graph

    def can_handle(self, context) -> bool:
        if hasattr(context, 'intent'):
            intent = context.intent
        else:
            intent = context.get('intent', '')
        return intent in ('QUERY_KG', 'QUERY_MOVIE', 'QUERY_COMPARISON')

    def run(self, context) -> dict:
        import time
        t0 = time.time()

        if hasattr(context, 'user_input'):
            query = context.user_input
            constraints = context.constraints
        else:
            query = context.get('query', '')
            constraints = context.get('constraints', {})

        from myapp.agent.movie_agent import KGQueryTool
        tool = KGQueryTool(neo_graph=self.neo_graph)
        result = tool.execute(query=query, constraints=constraints)

        elapsed = time.time() - t0

        return self._success(
            data=result.get('output', []),
            triples=result.get('triples', []),
            cypher=result.get('cypher', ''),
            count=result.get('count', 0),
            elapsed=f"{elapsed:.3f}s",
        )
