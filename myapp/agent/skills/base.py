"""
BaseSkill — Skill 抽象基类
=================================================
每个 Skill 封装一个独立的推荐子能力（召回/精排/解释等）。

与原有 AgentTool 的区别:
  - 增加 input_schema / output_schema 供 MCP Adapter 自动生成 tool schema
  - 增加 can_handle() 供 SkillRegistry 自动路由
  - 增加 fallback() 优雅降级，而非直接抛异常
  - run() 接受统一的 context dict，而非散装 kwargs
=================================================
"""

from abc import ABC, abstractmethod


class BaseSkill(ABC):
    """Skill 抽象基类，所有 Skill 必须继承此类。"""

    # ── 元数据（子类必须覆盖）──────────────────────────────
    name: str = "base_skill"
    description: str = "基础 Skill"

    # ── Schema（供 MCP Adapter / Multi-Agent 自动生成接口）──
    input_schema: dict = {}
    output_schema: dict = {}

    # ── 核心接口 ──────────────────────────────────────────

    @abstractmethod
    def can_handle(self, context: dict) -> bool:
        """
        判断当前 Skill 是否能处理给定上下文。

        Args:
            context: 统一上下文字典，至少包含:
                - intent: str 意图分类结果
                - query: str 用户原始查询
                - user: User 对象（可选）
                - candidates: list 候选列表（可选）

        Returns:
            bool 是否能处理
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, context: dict) -> dict:
        """
        执行 Skill 核心逻辑。

        Args:
            context: 统一上下文字典

        Returns:
            dict 统一结果字典，至少包含:
                - skill: str Skill 名称
                - success: bool 是否成功
                - data: Any 核心数据
                - meta: dict 元数据（耗时、来源等）
        """
        raise NotImplementedError

    def fallback(self, context: dict, error: Exception) -> dict:
        """
        降级处理。默认返回空结果 + 错误信息。

        Args:
            context: 统一上下文字典
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
        将散装 kwargs 转为 context dict 后调用 run()。
        """
        context = kwargs.copy()
        try:
            return self.run(context)
        except Exception as e:
            return self.fallback(context, e)

    # ── 辅助方法 ──────────────────────────────────────────

    def _success(self, data, **meta) -> dict:
        """构造成功结果。"""
        return {
            'skill': self.name,
            'success': True,
            'data': data,
            'meta': meta,
        }

    def __repr__(self) -> str:
        return f"<Skill:{self.name}>"
