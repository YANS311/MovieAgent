"""
BaseSkill — Skill 抽象基类（v2）
=================================================
每个 Skill 封装一个独立的推荐子能力（召回/精排/解释等）。

v2 新增:
  - Skill Metadata: version, priority, latency_level, cost_level, tags, examples
  - SkillContext 支持: run() 接受 SkillContext 或 dict
  - 指标钩子: _record_metric() 自动记录调用统计
=================================================
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseSkill(ABC):
    """Skill 抽象基类，所有 Skill 必须继承此类。"""

    # ── 元数据（子类必须覆盖）──────────────────────────────
    name: str = "base_skill"
    description: str = "基础 Skill"

    # ── Metadata v2 ────────────────────────────────────────
    version: str = "1.0.0"
    author: str = "MovieAgent"
    priority: int = 50                   # 优先级，数值越大越优先
    latency_level: str = "medium"        # "low" / "medium" / "high"
    cost_level: str = "medium"           # "low" / "medium" / "high"
    tags: list = []                      # 标签，如 ["retrieval", "semantic"]
    examples: list = []                  # 示例输入输出

    # ── Schema（供 MCP Adapter / Multi-Agent 自动生成接口）──
    input_schema: dict = {}
    output_schema: dict = {}

    # ── 核心接口 ──────────────────────────────────────────

    @abstractmethod
    def can_handle(self, context) -> bool:
        """
        判断当前 Skill 是否能处理给定上下文。

        Args:
            context: SkillContext 实例 或 dict

        Returns:
            bool 是否能处理
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, context) -> dict:
        """
        执行 Skill 核心逻辑。

        Args:
            context: SkillContext 实例 或 dict

        Returns:
            dict 统一结果字典，至少包含:
                - skill: str Skill 名称
                - success: bool 是否成功
                - data: Any 核心数据
                - meta: dict 元数据（耗时、来源等）
        """
        raise NotImplementedError

    def fallback(self, context, error: Exception) -> dict:
        """
        降级处理。默认返回空结果 + 错误信息。

        Args:
            context: SkillContext 实例 或 dict
            error: 捕获到的异常

        Returns:
            dict 降级结果
        """
        return {
            'skill': self.name,
            'success': False,
            'data': [],
            'meta': {
                'fallback': True,
                'error': str(error),
            },
        }

    # ── 兼容层：让 Skill 可以当作 AgentTool 使用 ──────────

    @property
    def tool_name(self) -> str:
        """兼容 AgentTool.name"""
        return self.name

    @property
    def tool_description(self) -> str:
        """兼容 AgentTool.description"""
        return self.description

    def execute(self, **kwargs) -> dict:
        """
        兼容 AgentTool.execute() 接口。
        将散装 kwargs 转为 context 后调用 run()。
        """
        context = self._ensure_context(kwargs)
        try:
            return self.run(context)
        except Exception as e:
            return self.fallback(context, e)

    # ── Metadata 方法 ─────────────────────────────────────

    def get_metadata(self) -> dict:
        """返回完整的 Skill Metadata。"""
        return {
            'name': self.name,
            'description': self.description,
            'version': self.version,
            'author': self.author,
            'priority': self.priority,
            'latency_level': self.latency_level,
            'cost_level': self.cost_level,
            'tags': list(self.tags),
            'examples': list(self.examples),
            'input_schema': self.input_schema,
            'output_schema': self.output_schema,
        }

    def matches_tags(self, required_tags: list) -> bool:
        """检查是否匹配指定标签。"""
        if not required_tags:
            return True
        return bool(set(required_tags) & set(self.tags))

    # ── 辅助方法 ──────────────────────────────────────────

    def _ensure_context(self, context_or_dict) -> Any:
        """将 dict 转为 SkillContext（如果可用），否则返回原 dict。"""
        if isinstance(context_or_dict, dict):
            try:
                from myapp.agent.context import SkillContext
                return SkillContext.from_dict(context_or_dict)
            except ImportError:
                return context_or_dict
        return context_or_dict

    def _success(self, data, **meta) -> dict:
        """构造成功结果。"""
        return {
            'skill': self.name,
            'success': True,
            'data': data,
            'meta': meta,
        }

    def __repr__(self) -> str:
        return f"<Skill:{self.name} v{self.version} p{self.priority}>"
