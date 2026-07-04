"""
SkillContext — Agent 统一上下文数据结构
=================================================
所有 Skill / Tool 之间传递的标准化数据容器。

设计原则:
  1. 不可变语义 — 通过 dataclass(frozen=False) 允许就地更新，
     但每个阶段应视为 snapshot，避免隐式副作用
  2. 渐进填充 — 各阶段按需填充字段，未填充的字段保持默认值
  3. 兼容层 — 支持 dict() 转换，兼容旧 Tool 的 kwargs 接口
  4. 可追踪 — 内置 trace 列表，记录每一步的数据变化
  5. 可序列化 — to_dict() 支持 JSON 序列化（用于日志和调试）

使用方式:
    ctx = SkillContext(user_input="推荐科幻电影")
    ctx.intent = "QUERY_MOVIE"
    ctx.constraints = {"genre": "科幻", "min_rating": 7.0}
    ctx.candidate_movies = [{"movie_id": 1, "score": 0.9}, ...]

    # 传给 Skill
    skill = registry.get("maan_rerank")
    result = skill.run(ctx.to_dict())

    # 兼容旧 Tool
    tool.execute(**ctx.to_tool_kwargs("maan_rerank"))
=================================================
"""

import time
import copy
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

logger = logging.getLogger('movie_agent')


@dataclass
class SkillContext:
    """
    Agent 推理链的统一上下文。

    生命周期:
      1. 创建: user_input 填充
      2. 意图分类: intent 填充
      3. 约束提取: constraints 填充
      4. 召回: candidate_movies 填充
      5. 精排: candidate_movies 更新
      6. 重排: candidate_movies 更新
      7. 解释: explanations 填充
      8. 输出: final_answer 填充
    """

    # ── 输入层 ──────────────────────────────────────────────
    user_input: str = ""
    intent: str = ""
    is_thinking_mode: bool = False

    # ── 约束层（来自 _micro_think 或 LLM 提取）────────────
    constraints: dict = field(default_factory=lambda: {
        'genre': None,
        'min_rating': None,
        'vibe': None,
        'year_filter': None,
        'director': None,
        'actor': None,
        'exclusions': [],
    })

    # ── 用户层 ──────────────────────────────────────────────
    user: Any = None                    # Django User 对象
    user_profile: dict = field(default_factory=dict)   # 长期画像
    memory_slots: dict = field(default_factory=dict)   # 短期槽位

    # ── 对话层 ──────────────────────────────────────────────
    conversation_history: list = field(default_factory=list)
    session_id: str = ""

    # ── 召回层 ──────────────────────────────────────────────
    candidate_movies: list = field(default_factory=list)
    retrieved_documents: list = field(default_factory=list)
    graph_facts: list = field(default_factory=list)

    # ── 工具执行层 ──────────────────────────────────────────
    tool_results: dict = field(default_factory=dict)
    tool_chain: list = field(default_factory=list)

    # ── 输出层 ──────────────────────────────────────────────
    final_answer: str = ""
    recommended_ids: list = field(default_factory=list)
    explanations: dict = field(default_factory=dict)

    # ── 追踪层 ──────────────────────────────────────────────
    trace: list = field(default_factory=list)
    metadata: dict = field(default_factory=lambda: {
        'created_at': None,
        'completed_at': None,
        'latency_ms': 0,
        'retry_count': 0,
        'fallback_used': False,
        'need_clarification': False,
        'clarification_options': [],
    })

    # ── 外部资源 ────────────────────────────────────────────
    neo_graph: Any = None               # Neo4j 图实例
    rag_resources: Any = None           # RAG 资源字典
    llm_config: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata.get('created_at') is None:
            self.metadata['created_at'] = time.time()

    # ── 工厂方法 ────────────────────────────────────────────

    @classmethod
    def from_agent_input(cls, user_input: str, user=None, **kwargs) -> "SkillContext":
        """
        从 MovieAgent.run() 的输入参数创建上下文。

        兼容现有入口:
            ctx = SkillContext.from_agent_input(
                user_input="推荐科幻电影",
                user=request.user,
            )
        """
        return cls(
            user_input=user_input,
            user=user,
            **kwargs,
        )

    @classmethod
    def from_dict(cls, data: dict) -> "SkillContext":
        """从字典创建上下文（用于反序列化）。"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    # ── 状态更新 ────────────────────────────────────────────

    def add_trace(self, step_type: str, content: str, **extra):
        """添加一个推理步骤到 trace。"""
        entry = {
            'step': len(self.trace),
            'type': step_type,
            'content': content,
            'timestamp': time.time(),
            **extra,
        }
        self.trace.append(entry)

    def set_tool_result(self, tool_name: str, result: dict):
        """记录一个工具的执行结果。"""
        self.tool_results[tool_name] = result
        # 同时更新 candidates（如果工具返回了候选列表）
        if 'output' in result and isinstance(result['output'], list):
            self.candidate_movies = result['output']

    def complete(self, final_answer: str = None, recommended_ids: list = None):
        """标记推理完成。"""
        if final_answer is not None:
            self.final_answer = final_answer
        if recommended_ids is not None:
            self.recommended_ids = recommended_ids
        self.metadata['completed_at'] = time.time()
        if self.metadata['created_at']:
            self.metadata['latency_ms'] = int(
                (self.metadata['completed_at'] - self.metadata['created_at']) * 1000
            )

    # ── 序列化 ──────────────────────────────────────────────

    def to_dict(self) -> dict:
        """
        转为字典，兼容旧 Tool 的 kwargs 接口。

        不可序列化的字段（user, neo_graph 等）会被替换为占位符。
        """
        d = {}
        for k, v in self.__dict__.items():
            if k in ('user', 'neo_graph', 'rag_resources'):
                d[k] = v  # 保留对象引用，不序列化
            elif k == 'llm_config':
                d[k] = {key: val for key, val in v.items()
                        if not key.startswith('_')}
            else:
                try:
                    json.dumps(v)
                    d[k] = v
                except (TypeError, ValueError):
                    d[k] = str(v)
        return d

    def to_tool_kwargs(self, tool_name: str) -> dict:
        """
        为指定 Tool 生成兼容的 kwargs。

        根据 tool_name 提取该工具需要的参数，避免传入多余字段。
        """
        kwarg_map = {
            'search_vector': lambda ctx: {
                'query': ctx.user_input,
                'k': ctx.metadata.get('k', 10),
            },
            'recall_hybrid': lambda ctx: {
                'user': ctx.user,
                'query_text': ctx.user_input,
                'top_k': ctx.metadata.get('top_k', 15),
                'neo_graph': ctx.neo_graph,
                'rag_resources': ctx.rag_resources,
            },
            'kg_query': lambda ctx: {
                'query': ctx.user_input,
                'constraints': ctx.constraints,
            },
            'maan_rerank': lambda ctx: {
                'candidates': ctx.candidate_movies,
                'user': ctx.user,
                'top_k': ctx.metadata.get('top_k', 10),
            },
            'rerank': lambda ctx: {
                'candidates': ctx.candidate_movies,
                'user': ctx.user,
                'top_k': ctx.metadata.get('top_k', 10),
            },
            'explain': lambda ctx: {
                'user': ctx.user,
                'movie_id': ctx.metadata.get('current_movie_id'),
            },
        }

        builder = kwarg_map.get(tool_name)
        if builder:
            return builder(self)
        # 未知工具：返回通用 kwargs
        return {
            'query': self.user_input,
            'user': self.user,
            'candidates': self.candidate_movies,
        }

    # ── 辅助方法 ────────────────────────────────────────────

    def get_slot(self, key: str, default=None):
        """从 memory_slots 获取槽位值。"""
        return self.memory_slots.get(key, default)

    def get_constraint(self, key: str, default=None):
        """从 constraints 获取约束值。"""
        return self.constraints.get(key, default)

    @property
    def has_candidates(self) -> bool:
        """是否有候选电影。"""
        return len(self.candidate_movies) > 0

    @property
    def has_graph_facts(self) -> bool:
        """是否有图谱事实。"""
        return len(self.graph_facts) > 0

    @property
    def latency_ms(self) -> int:
        """当前耗时（毫秒）。"""
        if self.metadata.get('created_at'):
            return int((time.time() - self.metadata['created_at']) * 1000)
        return 0

    def snapshot(self) -> dict:
        """
        生成当前状态的快照（用于 Trace 记录）。
        不包含不可序列化的对象。
        """
        return {
            'user_input': self.user_input,
            'intent': self.intent,
            'constraints': self.constraints,
            'candidate_count': len(self.candidate_movies),
            'graph_fact_count': len(self.graph_facts),
            'tool_count': len(self.tool_results),
            'trace_count': len(self.trace),
            'latency_ms': self.latency_ms,
        }

    def __repr__(self) -> str:
        return (
            f"SkillContext(input='{self.user_input[:30]}...', "
            f"intent='{self.intent}', "
            f"candidates={len(self.candidate_movies)}, "
            f"tools={list(self.tool_results.keys())})"
        )
