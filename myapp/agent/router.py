"""
SkillRouter — 轻量级 Skill 路由器
=================================================
根据 Intent 选择 Skill，按 Metadata 排序，调用并合并结果。

职责:
  1. Intent → Skill 选择
  2. 按 priority 排序
  3. 调用 Skill（带指标记录）
  4. 合并结果

不实现:
  - Multi-Agent
  - LangGraph / AutoGen
  - 复杂的 DAG 调度
=================================================
"""

import time
import logging
from typing import List, Optional

from .skills.registry import SkillRegistry
from .skills.base import BaseSkill
from .context import SkillContext
from .metrics import get_global_metrics

logger = logging.getLogger('movie_agent')


# ── Intent → 默认 Skill 映射 ─────────────────────────────
INTENT_SKILL_MAP = {
    'QUERY_MOVIE': ['search_vector', 'maan_rerank', 'rerank'],
    'QUERY_COMPARISON': ['search_vector', 'maan_rerank', 'rerank'],
    'QUERY_PROFILE_REC': ['recall_hybrid', 'maan_rerank', 'rerank'],
    'QUERY_RANK': ['search_vector', 'maan_rerank'],
    'QUERY_NEW': ['search_vector', 'maan_rerank'],
    'QUERY_KG': ['kg_query'],
    'QUERY_VISUAL': ['search_vector'],
    'QUERY_SELF': [],
    'CHAT': [],
}

# ── 空结果纠偏映射 ───────────────────────────────────────
FALLBACK_CHAIN = {
    'recall_hybrid': 'search_vector',
    'search_vector': 'recall_hybrid',
    'kg_query': 'search_vector',
}


class SkillRouter:
    """
    轻量级 Skill 路由器。

    使用方式:
        router = SkillRouter(registry)
        result = router.route(context)
    """

    def __init__(self, registry: SkillRegistry):
        self.registry = registry
        self.metrics = get_global_metrics()

    def route(self, context: SkillContext) -> SkillContext:
        """
        执行完整的路由流程。

        1. 根据 intent 选择 skill 链
        2. 按顺序执行每个 skill
        3. 将结果写入 context
        4. 返回更新后的 context
        """
        intent = context.intent or 'CHAT'
        skill_names = INTENT_SKILL_MAP.get(intent, [])

        if not skill_names:
            context.add_trace('router', f'意图 {intent} 无需工具调用')
            return context

        context.tool_chain = list(skill_names)
        context.add_trace('router', f'选择工具链: {skill_names}')

        for skill_name in skill_names:
            skill = self.registry.get(skill_name)
            if not skill:
                logger.warning(f"[Router] Skill not found: {skill_name}")
                continue

            self._execute_skill(context, skill)

            # 空结果纠偏
            if not context.has_candidates and skill_name in FALLBACK_CHAIN:
                fb_name = FALLBACK_CHAIN[skill_name]
                fb_skill = self.registry.get(fb_name)
                if fb_skill:
                    context.metadata['fallback_used'] = True
                    context.add_trace('router', f'空结果纠偏: {skill_name} → {fb_name}')
                    self._execute_skill(context, fb_skill)

        return context

    def _execute_skill(self, context: SkillContext, skill: BaseSkill):
        """执行单个 Skill，记录指标。"""
        t0 = time.time()

        try:
            result = skill.run(context)
            elapsed_ms = (time.time() - t0) * 1000
            fallback = result.get('meta', {}).get('fallback', False)

            context.set_tool_result(skill.name, result)
            context.add_trace(
                'action',
                f'调用 {skill.name} → {"成功" if result.get("success") else "失败"}',
                elapsed_ms=round(elapsed_ms, 2),
            )

            self.metrics.record(
                skill.name, success=result.get('success', False),
                latency_ms=elapsed_ms, fallback=fallback,
            )

        except Exception as e:
            elapsed_ms = (time.time() - t0) * 1000
            logger.error(f"[Router] {skill.name} 异常: {e}")

            try:
                fallback_result = skill.fallback(context, e)
                context.set_tool_result(skill.name, fallback_result)
            except Exception:
                context.set_tool_result(skill.name, {
                    'skill': skill.name, 'success': False,
                    'data': [], 'meta': {'error': str(e)},
                })

            context.add_trace('error', f'{skill.name} 异常: {e}')
            self.metrics.record(
                skill.name, success=False,
                latency_ms=elapsed_ms, fallback=True, error=str(e),
            )

    def select_skills(self, context: SkillContext) -> List[BaseSkill]:
        """返回当前 context 下可用的 Skill 列表（按 priority 降序）。"""
        return self.registry.select_all(context.to_dict())

    def get_skill_chain(self, intent: str) -> List[str]:
        """获取指定 intent 的默认 Skill 链。"""
        return INTENT_SKILL_MAP.get(intent, [])
