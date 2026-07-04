"""
SkillRegistry — Skill 注册与路由中心（v2）
=================================================
管理所有已注册的 Skill，支持:
  - 按名称查找
  - 按上下文自动选择
  - 按 Metadata 排序和过滤
  - 列出全部可用 Skill（含 Metadata）
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
        logger.debug(f"[SkillRegistry] Registered: {skill.name} v{skill.version}")

    def get(self, name: str) -> Optional[BaseSkill]:
        """按名称获取 Skill。"""
        return self._skills.get(name)

    def select(self, context) -> Optional[BaseSkill]:
        """
        根据上下文自动选择最合适的 Skill。
        按 priority 降序遍历，返回第一个 can_handle() 为 True 的。
        """
        for skill in self._sorted_skills():
            try:
                if skill.can_handle(context):
                    return skill
            except Exception as e:
                logger.warning(f"[SkillRegistry] can_handle error in {skill.name}: {e}")
        return None

    def select_all(self, context) -> List[BaseSkill]:
        """返回所有能处理该上下文的 Skill（按 priority 降序）。"""
        result = []
        for skill in self._sorted_skills():
            try:
                if skill.can_handle(context):
                    result.append(skill)
            except Exception:
                pass
        return result

    def select_by_tags(self, tags: list, context=None) -> List[BaseSkill]:
        """按标签筛选 Skill，可选同时检查 can_handle。"""
        result = []
        for skill in self._sorted_skills():
            if skill.matches_tags(tags):
                if context is None:
                    result.append(skill)
                else:
                    try:
                        if skill.can_handle(context):
                            result.append(skill)
                    except Exception:
                        pass
        return result

    def list_skills(self) -> List[dict]:
        """列出所有已注册 Skill 的完整 Metadata。"""
        return [s.get_metadata() for s in self._sorted_skills()]

    def list_names(self) -> List[str]:
        """列出所有已注册 Skill 的名称。"""
        return [s.name for s in self._sorted_skills()]

    def _sorted_skills(self) -> List[BaseSkill]:
        """按 priority 降序排列。"""
        return sorted(self._skills.values(), key=lambda s: s.priority, reverse=True)

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __repr__(self) -> str:
        names = ", ".join(s.name for s in self._sorted_skills())
        return f"<SkillRegistry [{names}]>"
