"""
ReAct Agent — 真正的 ReAct 推理智能体
================================================
实现完整的 Thought → Action → Observation → Reflection 循环，
而非固定 Workflow 的 if-else 分支。

核心设计原则：
  1. 每一步决策都基于上一步的 Observation
  2. Agent 可以在运行时动态改变工具选择
  3. 支持自反馈纠偏（空结果重试）
  4. 支持多轮推理（最多 MAX_ITERATIONS 轮）
  5. 完整 Trace 记录
================================================
"""

import re
import time
import json
from typing import Dict, List, Optional, Tuple

from myapp.agent.tool_registry import get_global_registry
from myapp.agent.trace_logger import AgentTrace, TraceCollector


# =============================================================
# ReAct Agent 核心
# =============================================================

class ReActAgent:
    """
    基于 ReAct 范式的推荐智能体
    
    推理流程（动态，非固定）：
        Thought 1 → Action 1 → Observation 1
        → Reflection 1 → Thought 2 → Action 2 → Observation 2
        → ... → Final Answer
    
    与 Fixed Workflow 的核心区别：
        - Workflow: intent==A → 固定调用 tool1, tool2, tool3
        - ReAct:    Thought("需要语义搜索") → search_vector → 
                     Observation("返回空") → Thought("换图谱查询") → 
                     kg_query → Observation("找到结果") → ...
    """

    MAX_ITERATIONS = 6  # 最大推理轮数（防止无限循环）
    CANDIDATE_POOL_LIMIT = 200

    def __init__(self, user=None, neo_graph=None, rag_resources=None,
                 session_id=None, trace_enabled=True):
        self.user = user
        self.neo_graph = neo_graph
        self.rag_resources = rag_resources
        self.session_id = session_id or f"react_{int(time.time())}"
        self.trace_enabled = trace_enabled

        # 懒加载工具
        self._tools = None
        self._registry = get_global_registry()

        # 记忆管理器
        from myapp.agent.memory import MemoryManager
        self.memory = MemoryManager(
            user=user,
            session_id=self.session_id,
        )

    @property
    def tools(self):
        if self._tools is None:
            self._tools = self._init_tools()
        return self._tools

    def _init_tools(self) -> dict:
        from myapp.agent.movie_agent import (
            SearchVectorTool, RecallHybridTool, KGQueryTool,
            MAANRerankTool, RerankTool, ExplainTool,
        )
        return {
            'search_vector': SearchVectorTool(self.rag_resources),
            'recall_hybrid': RecallHybridTool(self.neo_graph, self.rag_resources),
            'kg_query': KGQueryTool(self.neo_graph),
            'maan_rerank': MAANRerankTool(),
            'rerank': RerankTool(),
            'explain': ExplainTool(),
        }

    # ── 主入口 ──

    def run(self, user_input: str) -> dict:
        """
        执行 ReAct 推理主流程
        
        Returns:
            dict: 与 MovieAgent.run() 兼容的返回格式
        """
        t_start = time.time()

        # 初始化 Trace
        trace = AgentTrace(
            query=user_input,
            session_id=self.session_id,
            user_id=getattr(self.user, 'id', 0),
            system_config='react_agent',
        ) if self.trace_enabled else None

        # Step 0: 更新记忆
        self.memory.update_slots(user_input)

        # Step 1: 意图分类（规则驱动）
        from myapp.agent.movie_agent import IntentClassifier
        intent = IntentClassifier.classify(user_input)
        if trace:
            trace.add_thought(f"意图分类: {intent}，用户输入: '{user_input}'")

        # Step 2: 感知环境状态
        env_state = self._perceive(user_input, intent)
        if trace:
            trace.add_thought(
                f"环境感知: 锚点电影={env_state.get('anchor_movie')}, "
                f"情感约束={env_state.get('sentiment')}, "
                f"追问={env_state.get('is_followup')}"
            )

        # Step 3: ReAct 循环
        candidates = []
        explanations = {}
        recommended_ids = []
        final_answer = ''
        iteration = 0
        current_tool_plan = self._initial_plan(intent, env_state)

        while iteration < self.MAX_ITERATIONS and current_tool_plan:
            iteration += 1
            tool_name = current_tool_plan.pop(0)

            # Thought: 分析当前状态，决定是否调整计划
            thought = self._think(iteration, tool_name, candidates, env_state)
            if trace:
                trace.add_thought(thought)

            # Action: 执行工具
            t_tool = time.time()
            action_result, observation = self._act(
                tool_name, user_input, candidates, env_state
            )
            tool_latency = (time.time() - t_tool) * 1000

            if trace:
                trace.add_action(
                    tool_name=tool_name,
                    input_text=action_result.get('input', ''),
                    latency_ms=tool_latency,
                )

            # Observation: 处理工具输出
            obs_count = observation.get('count', 0)
            new_candidates = self._merge_candidates(candidates, observation, tool_name)

            if trace:
                obs_summary = f"返回 {obs_count} 条结果"
                trace.add_observation(
                    tool_name=tool_name,
                    output_count=obs_count,
                    output_summary=obs_summary,
                )

            # Reflection: 基于 Observation 决定下一步
            reflection, plan_adjustment = self._reflect(
                tool_name, observation, candidates, new_candidates, env_state
            )
            candidates = new_candidates
            trace and trace.mark_candidate_pool(len(candidates))

            if trace and reflection:
                trace.add_reflection(reflection)

            # 应用计划调整（动态决策）
            if plan_adjustment == 'abort_and_fallback':
                # 当前工具完全失败，跳过后续依赖工具
                fallback = self._registry.get(tool_name)
                if fallback and fallback.fallback_tool:
                    current_tool_plan = [fallback.fallback_tool] + current_tool_plan
                    if trace:
                        trace.add_thought(
                            f"[动态调整] {tool_name} 失败，切换至 {fallback.fallback_tool}"
                        )
            elif plan_adjustment == 'skip_remaining_recall':
                # 召回已足够，跳过剩余召回工具
                current_tool_plan = [
                    t for t in current_tool_plan
                    if self._registry.get(t) and self._registry.get(t).category != 'recall'
                ]

        # Step 4: 精排（如果还没排过）
        has_reranked = any(
            s.stage == 'action' and s.tool_name in ('maan_rerank', 'rerank')
            for s in (trace.steps if trace else [])
        )
        if not has_reranked and candidates:
            if trace:
                trace.add_thought("候选集已就绪，执行 MAAN 精排")
            t_rerank = time.time()
            _, rerank_obs = self._act('maan_rerank', user_input, candidates, env_state)
            candidates = rerank_obs.get('output', candidates)
            if trace:
                trace.add_action('maan_rerank', f"{len(candidates)} candidates",
                                 latency_ms=(time.time() - t_rerank) * 1000)
                trace.add_observation('maan_rerank', output_count=len(candidates))

            t_rerank2 = time.time()
            _, rerank2_obs = self._act('rerank', user_input, candidates, env_state)
            candidates = rerank2_obs.get('output', candidates)
            if trace:
                trace.add_action('rerank', f"{len(candidates)} candidates",
                                 latency_ms=(time.time() - t_rerank2) * 1000)
                trace.add_observation('rerank', output_count=len(candidates))

        # Step 5: 提取推荐 ID
        for item in candidates:
            mid = item.get('movie_id') if isinstance(item, dict) else None
            if mid:
                recommended_ids.append(mid)

        # Step 6: 生成推荐理由
        if recommended_ids and self.user:
            for mid in recommended_ids[:5]:
                try:
                    explain_result = self.tools['explain'].execute(
                        user=self.user, movie_id=mid
                    )
                    explanations[mid] = explain_result.get('output', '')
                except Exception:
                    explanations[mid] = ''

        # Step 7: Final Answer
        from myapp.agent.movie_agent import VaguenessDetector
        is_vague, _ = VaguenessDetector.is_vague(user_input)
        if is_vague and intent in ('QUERY_MOVIE', 'CHAT'):
            options = VaguenessDetector.generate_clarification_options(
                user_input, self.memory.get_slots()
            )
            option_lines = [f"  {i}. {opt['label']}" for i, opt in enumerate(options, 1)]
            final_answer = (
                "🤔 您的需求我还需要进一步确认：\n" + "\n".join(option_lines)
            )
        else:
            final_answer = self._format_answer(
                user_input, intent, recommended_ids, explanations
            )

        if trace:
            trace.set_final_answer(final_answer, recommended_ids, explanations)
            trace.finalize()

        t_total = int((time.time() - t_start) * 1000)

        return {
            'intent': intent,
            'thought': trace.steps[0].input_summary if trace and trace.steps else '',
            'actions': [s.to_dict() for s in trace.steps if s.stage == 'action'] if trace else [],
            'observations': [s.to_dict() for s in trace.steps if s.stage == 'observation'] if trace else [],
            'final_answer': final_answer,
            'recommended_ids': recommended_ids,
            'explanations': explanations,
            'latency_ms': t_total,
            'need_clarification': is_vague and intent in ('QUERY_MOVIE', 'CHAT'),
            'clarification_options': [],
            'trace_steps': [s.to_dict() for s in trace.steps] if trace else [],
            'react_iterations': iteration,
        }

    # ── ReAct 核心方法 ──

    def _perceive(self, user_input: str, intent: str) -> dict:
        """感知环境状态：锚点电影、情感约束、追问状态等"""
        from myapp.agent.movie_agent import MultiIntentDetector

        env = {
            'intent': intent,
            'anchor_movie': self._detect_anchor(user_input),
            'sentiment': self._detect_sentiment(user_input),
            'is_followup': self.memory.is_followup(user_input),
            'slots': self.memory.get_slots(),
            'has_multi': MultiIntentDetector.detect(user_input)[0],
        }
        return env

    def _initial_plan(self, intent: str, env_state: dict) -> list:
        """
        生成初始工具调用计划（可被 Reflection 动态修改）
        """
        from myapp.agent.movie_agent import MovieAgent

        base_plan = list(MovieAgent.INTENT_TOOL_MAP.get(intent, []))

        # 锚点电影 → 前置 kg_query
        if env_state.get('anchor_movie') and intent in (
            'QUERY_MOVIE', 'QUERY_COMPARISON', 'QUERY_PROFILE_REC'
        ):
            if 'kg_query' not in base_plan:
                base_plan.insert(0, 'kg_query')

        return base_plan

    def _think(self, iteration: int, tool_name: str,
               candidates: list, env_state: dict) -> str:
        """
        Thought 阶段：分析当前状态，为即将执行的 Action 生成推理文本
        """
        tool_spec = self._registry.get(tool_name)
        purpose = tool_spec.reasoning_purpose if tool_spec else tool_name

        parts = [f"Thought (iter {iteration}): 准备调用 {tool_name}"]

        if tool_name == 'kg_query' and env_state.get('anchor_movie'):
            parts.append(f"用户提到了锚点电影《{env_state['anchor_movie']}》，通过知识图谱查询其关联实体")
        elif tool_name in ('search_vector', 'recall_hybrid'):
            if candidates:
                parts.append(f"当前候选池有 {len(candidates)} 条，继续召回以扩充候选集")
            else:
                parts.append("候选池为空，执行首次召回")
        elif tool_name == 'maan_rerank':
            parts.append(f"候选集已就绪（{len(candidates)} 条），调用 MAAN 模型进行深度精排")
        elif tool_name == 'rerank':
            parts.append("精排完成，进行业务规则过滤和多样性排序")

        return " | ".join(parts)

    def _act(self, tool_name: str, user_input: str,
             current_candidates: list, env_state: dict) -> Tuple[dict, dict]:
        """
        Action 阶段：执行工具调用
        """
        tool = self.tools.get(tool_name)
        if not tool:
            return ({'tool': tool_name, 'error': '工具不存在'},
                    {'output': [], 'count': 0, 'error': '工具不存在'})

        try:
            enhanced_query = self._build_query(user_input, env_state)

            if tool_name == 'search_vector':
                result = tool.execute(query=enhanced_query, k=60)
            elif tool_name == 'recall_hybrid':
                result = tool.execute(user=self.user, query_text=enhanced_query, top_k=60)
            elif tool_name == 'kg_query':
                movie_name = self._extract_movie_name(user_input)
                result = tool.execute(
                    movie_title=movie_name,
                    user_genres=env_state.get('slots', {}).get('genre'),
                )
            elif tool_name == 'maan_rerank':
                result = tool.execute(candidates=current_candidates, user=self.user, top_k=15)
            elif tool_name == 'rerank':
                result = tool.execute(candidates=current_candidates, user=self.user, top_k=15)
            elif tool_name == 'explain':
                result = tool.execute(user=self.user)
            else:
                result = {'tool': tool_name, 'output': [], 'count': 0}

            action = {'tool': tool_name, 'input': result.get('input', '')}
            observation = {
                'tool': tool_name,
                'output': result.get('output', []),
                'count': result.get('count', 0),
                'stats': result.get('stats', {}),
            }
            return action, observation

        except Exception as e:
            return (
                {'tool': tool_name, 'input': user_input, 'error': str(e)},
                {'output': [], 'count': 0, 'error': str(e)},
            )

    def _reflect(self, tool_name: str, observation: dict,
                 old_candidates: list, new_candidates: list,
                 env_state: dict) -> Tuple[str, str]:
        """
        Reflection 阶段：基于 Observation 进行反思，决定下一步策略
        
        Returns:
            (reflection_text, plan_adjustment)
            plan_adjustment: None | 'abort_and_fallback' | 'skip_remaining_recall'
        """
        obs_count = observation.get('count', 0)
        old_count = len(old_candidates)
        new_count = len(new_candidates)

        tool_spec = self._registry.get(tool_name)

        # 情况1: 工具返回空结果 → 触发纠偏
        if obs_count == 0 and tool_spec and tool_spec.fallback_tool:
            return (
                f"[反思] {tool_name} 返回空结果，"
                f"当前候选池 {old_count} 条。"
                f"决策: 切换至备用工具 {tool_spec.fallback_tool}",
                'abort_and_fallback',
            )

        # 情况2: 召回结果充足 → 跳过后续召回
        if tool_spec and tool_spec.category == 'recall' and new_count >= 30:
            return (
                f"[反思] {tool_name} 召回 {obs_count} 条，"
                f"候选池已达 {new_count} 条，足够精排。"
                f"决策: 跳过后续召回工具，直接进入精排阶段",
                'skip_remaining_recall',
            )

        # 情况3: 召回结果偏少 → 继续下一个召回
        if tool_spec and tool_spec.category == 'recall' and new_count < 10:
            return (
                f"[反思] {tool_name} 召回 {obs_count} 条，"
                f"候选池仅 {new_count} 条，偏少。"
                f"决策: 继续执行下一个召回工具以扩充候选集",
                None,
            )

        # 情况4: 精排/重排 → 继续后续流程
        if tool_spec and tool_spec.category in ('rerank', 'explanation'):
            return (
                f"[反思] {tool_name} 执行完成，结果 {obs_count} 条。继续后续流程",
                None,
            )

        # 默认: 无特殊反思
        return (f"[反思] {tool_name} 返回 {obs_count} 条，候选池 {new_count} 条", None)

    # ── 辅助方法 ──

    def _detect_anchor(self, text: str) -> Optional[str]:
        from myapp.agent.movie_agent import MovieAgent
        # 复用 MovieAgent 的锚点检测逻辑
        temp = MovieAgent(user=self.user)
        return temp._detect_anchor_movie(text)

    def _detect_sentiment(self, text: str) -> list:
        sentiments = []
        patterns = [
            (r'不要.*?(压抑|沉重|悲伤|黑暗)', '排除压抑'),
            (r'(轻松|愉快|欢快|温馨|治愈)', '偏好轻松'),
            (r'(刺激|紧张|惊悚)', '偏好紧张'),
        ]
        for p, label in patterns:
            if re.search(p, text):
                sentiments.append(label)
        return sentiments

    def _merge_candidates(self, existing: list, observation: dict, tool_name: str) -> list:
        """合并候选集（去重）"""
        import re as _re
        raw = observation.get('output', [])
        if not isinstance(raw, list) or not raw:
            return existing

        merged = list(existing)
        existing_ids = {c.get('movie_id', c.get('id')) for c in merged if isinstance(c, dict)}

        if tool_name == 'kg_query':
            for item in raw:
                if isinstance(item, str):
                    id_match = _re.search(r'ID[：:](\d+)', item)
                    if id_match:
                        mid = int(id_match.group(1))
                        if mid not in existing_ids:
                            title_match = _re.search(r'《([^》]+)》', item)
                            title = title_match.group(1) if title_match else ''
                            merged.append({'movie_id': mid, 'title': title})
                            existing_ids.add(mid)
        else:
            for item in raw:
                if isinstance(item, dict):
                    mid = item.get('movie_id', item.get('id'))
                    if mid and mid not in existing_ids:
                        merged.append(item)
                        existing_ids.add(mid)

        return merged[:self.CANDIDATE_POOL_LIMIT]

    def _build_query(self, user_input: str, env_state: dict) -> str:
        """构建语义查询（复用 MovieAgent 逻辑）"""
        from myapp.agent.movie_agent import MovieAgent
        temp = MovieAgent(user=self.user, session_id=self.session_id)
        temp.memory = self.memory
        return temp._build_enhanced_query(user_input)

    def _extract_movie_name(self, text: str) -> str:
        match = re.search(r'《([^》]+)》', text)
        if match:
            return match.group(1)
        return text[:10]

    def _format_answer(self, user_input, intent, recommended_ids, explanations):
        from myapp.agent.movie_agent import MovieAgent
        temp = MovieAgent(user=self.user)
        return temp._generate_final_answer(
            user_input, intent, recommended_ids, explanations, ''
        )