"""
SkillRegistry — Skill 注册与路由中心
=================================================
管理所有已注册的 Skill，支持:
  - 按名称查找
  - 按上下文自动选择
  - 列出全部可用 Skill
=================================================
"""

import logging
from typing import List, Optional

from .base import BaseSkill

logger = logging.getLogger('movie_agent')


class SkillRegistry:
    """Skill 注册中心。"""

    def __init__(self):
        self._skills: dict[str, BaseSkill] = {}

    def register(self, skill: BaseSkill) -> None:
        """注册一个 Skill。同名 Skill 会被覆盖。"""
        if not isinstance(skill, BaseSkill):
            raise TypeError(f"Expected BaseSkill, got {type(skill)}")
        self._skills[skill.name] = skill
        logger.debug(f"[SkillRegistry] Registered: {skill.name}")

    def get(self, name: str) -> Optional[BaseSkill]:
        """按名称获取 Skill。"""
        return self._skills.get(name)

    def select(self, context: dict) -> Optional[BaseSkill]:
        """
        根据上下文自动选择最合适的 Skill。

        遍历所有已注册 Skill，返回第一个 can_handle() 为 True 的。
        选择顺序按注册顺序（先注册优先）。

        Args:
            context: 统一上下文字典

        Returns:
            BaseSkill 或 None
        """
        for skill in self._skills.values():
            try:
                if skill.can_handle(context):
                    return skill
            except Exception as e:
                logger.warning(f"[SkillRegistry] can_handle error in {skill.name}: {e}")
        return None

    def select_all(self, context: dict) -> List[BaseSkill]:
        """返回所有能处理该上下文的 Skill。"""
        result = []
        for skill in self._skills.values():
            try:
                if skill.can_handle(context):
                    result.append(skill)
            except Exception:
                pass
        return result

    def list_skills(self) -> List[dict]:
        """列出所有已注册 Skill 的元信息。"""
        return [
            {
                'name': s.name,
                'description': s.description,
                'input_schema': s.input_schema,
                'output_schema': s.output_schema,
            }
            for s in self._skills.values()
        ]

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __repr__(self) -> str:
        names = ", ".join(self._skills.keys())
        return f"<SkillRegistry [{names}]>"
