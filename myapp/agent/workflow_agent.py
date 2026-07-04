"""
Workflow Agent — 固定工作流基线
================================================
实现一个固定流水线（Fixed Workflow）的推荐 Agent，
用于与 ReAct Agent 进行对比实验。

核心特征（Workflow 的局限性）：
  1. 路径固定：intent → 固定工具链，不可动态调整
  2. 不可反思：无 Observation 驱动的规划调整
  3. 不可纠错：空结果时不切换备用路径
  4. 无 ReAct 循环：一次性按序执行所有工具

对比目的：
  证明 ReAct Agent 的动态规划能力优于固定 Workflow
================================================
"""

import re
import time
from typing import Dict, List, Optional


class WorkflowAgent:
    """
    固定工作流推荐 Agent（Baseline）
    
    推理流程（固定，不可动态调整）：
        intent → tool_chain[0] → tool_chain[1] → ... → Final Answer
    
    与 ReAct Agent 的核心区别：
        - 无 Thought / Reflection 阶段
        - 无动态工具切换
        - 无自反馈纠偏
        - 工具链在执行前完全确定
    """

    # 固定意图-工具链映射（不可运行时修改）
    INTENT_TOOL_MAP = {
        'QUERY_MOVIE': ['search_vector', 'maan_rerank', 'rerank'],
        'QUERY_COMPARISON': ['search_vector', 'maan_rerank', 'rerank'],
        'QUERY_PROFILE_REC': ['recall_hybrid', 'maan_rerank', 'rerank'],
        'QUERY_RANK': ['search_vector', 'maan_rerank'],
        'QUERY_NEW': ['search_vector', 'maan_rerank'],
        'QUERY_VISUAL': ['search_vector'],
        'QUERY_SELF': [],
        'CHAT': [],
    }

    def __init__(self, user=None, neo_graph=None, rag_resources=None,
                 session_id=None):
        self.user = user
        self.neo_graph = neo_graph
        self.rag_resources = rag_resources
        self.session_id = session_id or f"workflow_{int(time.time())}"

        self._tools = None

        # 记忆管理器（Workflow 也使用记忆，但不会动态调整计划）
        from myapp.agent.memory import MemoryManager
        self.memory = MemoryManager(
            user=user,
            session_id=self.session_id,
        )

    @property
    def tools(self):
        if self._tools is None:
            from myapp.agent.movie_agent import (
                SearchVectorTool, RecallHybridTool, KGQueryTool,
                MAANRerankTool, RerankTool, ExplainTool,
            )
            self._tools = {
                'search_vector': SearchVectorTool(self.rag_resources),
                'recall_hybrid': RecallHybridTool(self.neo_graph, self.rag_resources),
                'kg_query': KGQueryTool(self.neo_graph),
                'maan_rerank': MAANRerankTool(),
                'rerank': RerankTool(),
                'explain': ExplainTool(),
            }
        return self._tools

    def run(self, user_input: str) -> dict:
        """
        执行固定工作流推荐
        
        关键区别：无 ReAct 循环，无 Reflection，无动态调整
        """
        t_start = time.time()

        # Step 1: 更新记忆（仅用于查询增强，不用于动态规划）
        self.memory.update_slots(user_input)

        # Step 2: 意图分类
        from myapp.agent.movie_agent import IntentClassifier
        intent = IntentClassifier.classify(user_input)

        # Step 3: 获取固定工具链（核心区别：不可动态修改）
        tool_chain = list(self.INTENT_TOOL_MAP.get(intent, []))

        # ★ Workflow 的关键缺陷：即使检测到锚点电影，也不会动态插入 kg_query
        # （ReAct Agent 会在此处插入 kg_query）

        # Step 4: 按序执行工具链（无反思，无纠偏）
        candidates = []
        actions = []
        observations = []
        trace_steps = [{
            'step': 0, 'type': 'thought',
            'content': f'[Workflow] 意图: {intent}，固定工具链: {tool_chain}',
        }]
        step_counter = 1

        for tool_name in tool_chain:
            tool = self.tools.get(tool_name)
            if not tool:
                continue

            try:
                enhanced_query = self._build_query(user_input)

                if tool_name == 'search_vector':
                    result = tool.execute(query=enhanced_query, k=60)
                elif tool_name == 'recall_hybrid':
                    result = tool.execute(user=self.user, query_text=enhanced_query, top_k=60)
                elif tool_name == 'maan_rerank':
                    result = tool.execute(candidates=candidates, user=self.user, top_k=15)
                elif tool_name == 'rerank':
                    result = tool.execute(candidates=candidates, user=self.user, top_k=15)
                else:
                    result = {'tool': tool_name, 'output': [], 'count': 0}

                action = {'tool': tool_name, 'input': result.get('input', '')}
                observation = {
                    'tool': tool_name,
                    'output': result.get('output', []),
                    'count': result.get('count', 0),
                }
                actions.append(action)
                observations.append(observation)

                trace_steps.append({
                    'step': step_counter, 'type': 'action',
                    'content': f'[Workflow] 执行 {tool_name}',
                    'tool': tool_name,
                })
                step_counter += 1
                trace_steps.append({
                    'step': step_counter, 'type': 'observation',
                    'content': f'[Workflow] {tool_name} 返回 {result.get("count", 0)} 条',
                    'tool': tool_name,
                })
                step_counter += 1

                # ★ Workflow 的关键缺陷：即使返回空结果，也不重试
                # （ReAct Agent 会触发自反馈纠偏，切换至备用工具）

                raw = result.get('output', [])
                if isinstance(raw, list) and raw:
                    if tool_name in ('search_vector', 'recall_hybrid', 'kg_query'):
                        for item in raw:
                            if isinstance(item, dict):
                                mid = item.get('movie_id', item.get('id'))
                                if mid and mid not in {c.get('movie_id') for c in candidates}:
                                    candidates.append(item)
                    else:
                        candidates = raw

            except Exception as e:
                trace_steps.append({
                    'step': step_counter, 'type': 'error',
                    'content': f'[Workflow] {tool_name} 异常: {e}',
                })
                step_counter += 1

        # Step 5: 提取推荐 ID
        recommended_ids = []
        for item in candidates:
            mid = item.get('movie_id') if isinstance(item, dict) else None
            if mid:
                recommended_ids.append(mid)

        # Step 6: 生成推荐理由
        explanations = {}
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
        from myapp.agent.movie_agent import MovieAgent
        temp = MovieAgent(user=self.user)
        final_answer = temp._generate_final_answer(
            user_input, intent, recommended_ids, explanations, ''
        )

        trace_steps.append({
            'step': step_counter, 'type': 'final_answer',
            'content': final_answer,
        })

        t_total = int((time.time() - t_start) * 1000)

        return {
            'intent': intent,
            'thought': f'[Workflow] 固定流水线: {tool_chain}',
            'actions': actions,
            'observations': observations,
            'final_answer': final_answer,
            'recommended_ids': recommended_ids,
            'explanations': explanations,
            'latency_ms': t_total,
            'need_clarification': False,
            'clarification_options': [],
            'trace_steps': trace_steps,
            'agent_type': 'workflow',
        }

    def _build_query(self, user_input: str) -> str:
        """构建语义查询（复用 MovieAgent 逻辑）"""
        from myapp.agent.movie_agent import MovieAgent
        temp = MovieAgent(user=self.user, session_id=self.session_id)
        temp.memory = self.memory
        return temp._build_enhanced_query(user_input)


class SequentialPipelineAgent(WorkflowAgent):
    """
    顺序流水线 Agent — 比 Workflow 更简单的基线
    
    固定执行：search_vector → maan_rerank → rerank
    无论什么意图，都执行相同的工具链
    """

    INTENT_TOOL_MAP = {
        'QUERY_MOVIE': ['search_vector', 'maan_rerank', 'rerank'],
        'QUERY_COMPARISON': ['search_vector', 'maan_rerank', 'rerank'],
        'QUERY_PROFILE_REC': ['search_vector', 'maan_rerank', 'rerank'],
        'QUERY_RANK': ['search_vector', 'maan_rerank', 'rerank'],
        'QUERY_NEW': ['search_vector', 'maan_rerank', 'rerank'],
        'QUERY_VISUAL': ['search_vector', 'maan_rerank', 'rerank'],
        'QUERY_SELF': [],
        'CHAT': [],
    }

    def run(self, user_input: str) -> dict:
        result = super().run(user_input)
        result['agent_type'] = 'sequential_pipeline'
        return result