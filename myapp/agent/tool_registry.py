"""
Tool Registry — Agent 工具注册中心
================================================
统一管理 MovieAgent 所有工具的元信息，
包括名称、描述、输入/输出 Schema、推理用途等。
支持：
  - 工具注册与发现
  - 工具 Schema 查询
  - 动态工具链构建
  - 工具能力匹配
================================================
"""

from collections import OrderedDict
from typing import Dict, List, Optional, Any


class ToolSpec:
    """
    工具元信息规格
    每个工具必须声明以下字段，用于 ReAct 规划器的动态选择
    """
    __slots__ = [
        'name', 'description', 'input_schema', 'output_schema',
        'reasoning_purpose', 'category', 'dependencies',
        'fallback_tool', 'priority', 'avg_latency_ms',
    ]

    def __init__(
        self,
        name: str,
        description: str,
        input_schema: dict,
        output_schema: dict,
        reasoning_purpose: str,
        category: str = 'general',
        dependencies: list = None,
        fallback_tool: str = '',
        priority: int = 5,
        avg_latency_ms: float = 0.0,
    ):
        self.name = name
        self.description = description
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.reasoning_purpose = reasoning_purpose
        self.category = category  # 'recall' | 'rerank' | 'reasoning' | 'explanation' | 'general'
        self.dependencies = dependencies or []
        self.fallback_tool = fallback_tool
        self.priority = priority  # 1-10, 数值越大优先级越高
        self.avg_latency_ms = avg_latency_ms

    def to_dict(self) -> OrderedDict:
        d = OrderedDict()
        d['name'] = self.name
        d['description'] = self.description
        d['category'] = self.category
        d['input_schema'] = self.input_schema
        d['output_schema'] = self.output_schema
        d['reasoning_purpose'] = self.reasoning_purpose
        d['dependencies'] = self.dependencies
        d['fallback_tool'] = self.fallback_tool
        d['priority'] = self.priority
        d['avg_latency_ms'] = self.avg_latency_ms
        return d

    def __repr__(self):
        return f"ToolSpec(name='{self.name}', category='{self.category}')"


class ToolRegistry:
    """
    工具注册中心 — 管理所有可用工具的元信息
    
    核心职责：
      1. 注册工具（ToolSpec）
      2. 按名称查询工具
      3. 按推理目的匹配工具
      4. 生成工具描述 Prompt（供 LLM 使用）
      5. 验证工具链的依赖完整性
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec):
        """注册一个工具"""
        self._tools[spec.name] = spec

    def get(self, name: str) -> Optional[ToolSpec]:
        """按名称获取工具"""
        return self._tools.get(name)

    def list_all(self) -> List[ToolSpec]:
        """列出所有已注册工具"""
        return list(self._tools.values())

    def list_by_category(self, category: str) -> List[ToolSpec]:
        """按分类列出工具"""
        return [t for t in self._tools.values() if t.category == category]

    def match_by_purpose(self, purpose_keywords: List[str]) -> List[ToolSpec]:
        """
        按推理目的关键词匹配工具
        
        Args:
            purpose_keywords: 目的关键词列表，如 ['语义搜索', '模糊查询']
        
        Returns:
            匹配的工具列表（按优先级降序）
        """
        matched = []
        for spec in self._tools.values():
            score = 0
            for kw in purpose_keywords:
                if kw in spec.reasoning_purpose or kw in spec.description:
                    score += 1
            if score > 0:
                matched.append((score, spec))
        matched.sort(key=lambda x: (-x[0], -x[1].priority))
        return [spec for _, spec in matched]

    def generate_tool_prompt(self) -> str:
        """
        生成工具描述 Prompt，供 LLM 在 ReAct 推理中使用
        """
        lines = ["# Available Tools\n"]
        for spec in self._tools.values():
            lines.append(f"## {spec.name}")
            lines.append(f"  Description: {spec.description}")
            lines.append(f"  Purpose: {spec.reasoning_purpose}")
            lines.append(f"  Input: {spec.input_schema}")
            lines.append(f"  Output: {spec.output_schema}")
            if spec.fallback_tool:
                lines.append(f"  Fallback: {spec.fallback_tool}")
            lines.append("")
        return "\n".join(lines)

    def validate_tool_chain(self, tool_names: List[str]) -> dict:
        """
        验证工具链的依赖完整性
        
        Returns:
            {'valid': bool, 'missing_deps': list, 'warnings': list}
        """
        missing = []
        warnings = []
        available = set(tool_names)

        for name in tool_names:
            spec = self._tools.get(name)
            if not spec:
                missing.append(f"工具 '{name}' 未注册")
                continue
            for dep in spec.dependencies:
                if dep not in available:
                    warnings.append(f"工具 '{name}' 依赖 '{dep}'，但该工具不在工具链中")

        return {
            'valid': len(missing) == 0,
            'missing_deps': missing,
            'warnings': warnings,
        }

    def build_fallback_chain(self, tool_names: List[str]) -> Dict[str, str]:
        """
        根据工具链构建纠偏映射
        """
        chain = {}
        for name in tool_names:
            spec = self._tools.get(name)
            if spec and spec.fallback_tool:
                chain[name] = spec.fallback_tool
        return chain


# =============================================================
# 预注册 MovieAgent 全部工具
# =============================================================

def build_movieagent_registry() -> ToolRegistry:
    """
    构建 MovieAgent 的完整工具注册表
    
    每个工具包含：
      - name: 工具名称
      - description: 功能描述
      - input_schema: 输入参数 JSON Schema
      - output_schema: 输出格式 JSON Schema
      - reasoning_purpose: 推理用途说明（供 ReAct 规划器参考）
      - category: 工具分类
      - fallback_tool: 失败时的备用工具
      - priority: 优先级
    """
    registry = ToolRegistry()

    # ── 1. search_vector: RAG 向量语义搜索 ──
    registry.register(ToolSpec(
        name='search_vector',
        description='基于 FAISS 向量索引的语义相似度搜索。将用户查询通过 BGE-small-zh 编码为 512 维向量，在 FAISS 索引中检索语义最相似的电影文档。适用于模糊查询、剧情描述检索、自然语言需求匹配等场景。',
        input_schema={
            'query': 'str — 用户查询文本（自动提取语义关键词）',
            'k': 'int — 返回结果数量，默认 60',
        },
        output_schema={
            'tool': 'str — 工具名称',
            'output': 'list[dict] — 候选电影列表，每项含 movie_id, title, score',
            'count': 'int — 结果数量',
        },
        reasoning_purpose='语义相似度匹配：当用户使用自然语言描述观影需求（如"想看那种看完会思考人生的电影"），RAG 通过 BGE 嵌入模型将查询映射到语义空间，检索文本最相似的电影。擅长模糊语义、长尾电影召回、自然语言检索。',
        category='recall',
        fallback_tool='recall_hybrid',
        priority=8,
        avg_latency_ms=15.0,
    ))

    # ── 2. recall_hybrid: 多路混合召回 ──
    registry.register(ToolSpec(
        name='recall_hybrid',
        description='多路召回融合工具。集成四路召回信号（向量语义 + 内容特征 + 模型推理 + 知识图谱），通过 RRF 算法融合排序。适用于个性化推荐、画像驱动推荐等场景。',
        input_schema={
            'user': 'User — Django 用户对象',
            'query_text': 'str — 查询文本',
            'top_k': 'int — 返回结果数量，默认 60',
        },
        output_schema={
            'tool': 'str — 工具名称',
            'output': 'list[dict] — 融合后的候选电影列表',
            'stats': 'dict — 各路召回统计',
            'count': 'int — 结果数量',
        },
        reasoning_purpose='个性化多路召回：综合用户历史行为、语义匹配、模型预测和图谱关联四路信号，通过 RRF 融合生成个性化候选集。适用于需要深度个性化的推荐场景。',
        category='recall',
        fallback_tool='search_vector',
        priority=9,
        avg_latency_ms=45.0,
    ))

    # ── 3. kg_query: 知识图谱查询 ──
    registry.register(ToolSpec(
        name='kg_query',
        description='知识图谱拓扑遍历查询工具（含 Sub-graph Reasoning）。从目标电影出发，通过 DIRECTED_BY / ACTED_IN / BELONGS_TO 等关系进行多跳遍历，发现关联实体。同时进行子图推理，分析导演风格与用户偏好的语义重叠。',
        input_schema={
            'movie_title': 'str — 目标电影名称',
            'user_genres': 'list[str] — 用户偏好类型（用于 Sub-graph Reasoning）',
        },
        output_schema={
            'tool': 'str — 工具名称',
            'output': 'list[str] — 三元组列表，如 "《盗梦空间》(ID:456)--[导演:诺兰]-->《星际穿越》"',
            'reasoning_insights': 'list[str] — 子图推理结论',
            'count': 'int — 三元组数量',
        },
        reasoning_purpose='结构化关系推理：当用户提及特定电影（锚点电影）或需要多跳关系查询（如"同导演"、"合作演员"）时，KAG 通过知识图谱的 Cypher 查询执行确定性拓扑推理。擅长锚点电影关联、导演/演员关系查询、多跳逻辑约束。',
        category='reasoning',
        fallback_tool='search_vector',
        priority=10,
        avg_latency_ms=20.0,
    ))

    # ── 4. maan_rerank: MAAN 深度模型精排 ──
    registry.register(ToolSpec(
        name='maan_rerank',
        description='MAAN（Multi-Modal Attention Network）深度多模态模型精排工具。调用第四章训练的 MAAN 模型（GAUC 0.8898）对候选集进行最终打分与排序。模型特性：三路模态编码器（Text/Visual/KG）、跨模态注意力融合、门控多模态融合。',
        input_schema={
            'candidates': 'list[dict] — 候选电影列表',
            'user': 'User — Django 用户对象',
            'top_k': 'int — 返回结果数量，默认 15',
        },
        output_schema={
            'tool': 'str — 工具名称',
            'output': 'list[dict] — MAAN 精排后的候选列表，含 maan_score 字段',
            'stats': 'dict — 精排统计信息',
            'count': 'int — 结果数量',
        },
        reasoning_purpose='深度精排：在召回阶段获取候选集后，调用 MAAN 多模态深度模型进行最终排序。融合文本、视觉、知识图谱三路特征，输出个性化预测分数。确保推荐列表的排序质量。',
        category='rerank',
        dependencies=['search_vector', 'recall_hybrid'],  # 依赖召回工具先提供候选集
        priority=7,
        avg_latency_ms=8.0,
    ))

    # ── 5. rerank: 业务规则重排 ──
    registry.register(ToolSpec(
        name='rerank',
        description='业务过滤 + MMR 多样性排序工具。在精排结果基础上，应用业务规则（去重、黑名单、地区过滤等）和 MMR 算法确保推荐列表的多样性。',
        input_schema={
            'candidates': 'list[dict] — 候选电影列表',
            'user': 'User — Django 用户对象',
            'top_k': 'int — 返回结果数量，默认 15',
        },
        output_schema={
            'tool': 'str — 工具名称',
            'output': 'list[dict] — 重排后的候选列表',
            'stats': 'dict — 重排统计',
            'count': 'int — 结果数量',
        },
        reasoning_purpose='多样性与业务规则：确保最终推荐列表不过度集中于单一类型或导演，同时过滤不合规内容。',
        category='rerank',
        dependencies=['maan_rerank'],
        priority=5,
        avg_latency_ms=3.0,
    ))

    # ── 6. explain: 推荐解释生成 ──
    registry.register(ToolSpec(
        name='explain',
        description='推荐理由生成工具。基于最优锚点归因分析算法，从用户历史高分电影中选择最具说服力的参照物，通过多维关联强度计算（导演/演员/类型/视觉）确定归因类型，生成具备情感温度的自然语言推荐理由。',
        input_schema={
            'user': 'User — Django 用户对象',
            'movie_id': 'int — 推荐电影 ID',
        },
        output_schema={
            'tool': 'str — 工具名称',
            'output': 'str — 推荐理由文本',
            'reason_type': 'str — 归因类型（同导演/同主演/同类型/综合相似）',
            'strength': 'float — 归因强度',
        },
        reasoning_purpose='可解释推荐：为每条推荐结果生成知识驱动的归因解释。解释基于知识图谱真实关系（导演、演员、类型），杜绝幻觉。支持四种归因模板：同导演、同主演、同类型、综合美学相似。',
        category='explanation',
        priority=6,
        avg_latency_ms=5.0,
    ))

    return registry


# =============================================================
# 全局单例
# =============================================================

_global_registry = None


def get_global_registry() -> ToolRegistry:
    """获取全局工具注册表单例"""
    global _global_registry
    if _global_registry is None:
        _global_registry = build_movieagent_registry()
    return _global_registry


def get_tool_descriptions_for_prompt() -> str:
    """获取所有工具的 Prompt 描述（供 LLM 使用）"""
    return get_global_registry().generate_tool_prompt()