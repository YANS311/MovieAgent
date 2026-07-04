"""
Trace Logger — Agent 推理链追踪与持久化模块
================================================
用于记录 MovieAgent 每次推理的完整 ReAct 链路，
支持离线分析、论文实验数据采集和前端可视化。

记录内容：
  - Thought / Action / Observation / Reflection 各阶段
  - 工具调用详情（名称、输入、输出摘要、耗时）
  - 纠偏重试标记
  - 总延迟与 Token 统计

输出格式：
  JSON 结构，支持写入文件或数据库
================================================
"""

import json
import time
import os
from datetime import datetime
from collections import OrderedDict


class TraceStep:
    """
    单步推理记录
    """
    __slots__ = ['step_id', 'stage', 'tool_name', 'input_summary',
                 'output_summary', 'output_count', 'latency_ms',
                 'is_retry', 'retry_from', 'timestamp', 'metadata']

    def __init__(self, step_id: int, stage: str, tool_name: str = '',
                 input_summary: str = '', output_summary: str = '',
                 output_count: int = 0, latency_ms: float = 0.0,
                 is_retry: bool = False, retry_from: str = '',
                 metadata: dict = None):
        self.step_id = step_id
        self.stage = stage          # 'thought' | 'action' | 'observation' | 'reflection' | 'final_answer'
        self.tool_name = tool_name
        self.input_summary = input_summary
        self.output_summary = output_summary
        self.output_count = output_count
        self.latency_ms = latency_ms
        self.is_retry = is_retry
        self.retry_from = retry_from
        self.timestamp = int(time.time() * 1000)
        self.metadata = metadata or {}

    def to_dict(self):
        d = OrderedDict()
        d['step'] = self.step_id
        d['stage'] = self.stage
        if self.tool_name:
            d['tool'] = self.tool_name
        if self.input_summary:
            d['input'] = self.input_summary
        if self.output_summary:
            d['output'] = self.output_summary
        if self.output_count:
            d['count'] = self.output_count
        if self.latency_ms > 0:
            d['latency_ms'] = round(self.latency_ms, 1)
        if self.is_retry:
            d['is_retry'] = True
            d['retry_from'] = self.retry_from
        d['timestamp'] = self.timestamp
        if self.metadata:
            d['meta'] = self.metadata
        return d


class AgentTrace:
    """
    单次推理的完整 Trace 记录
    
    使用方式:
        trace = AgentTrace(query="推荐科幻片", intent="QUERY_MOVIE")
        trace.add_thought("用户想找科幻片...")
        trace.add_action("search_vector", "科幻片", latency_ms=12.5)
        trace.add_observation("search_vector", output_count=15)
        trace.add_reflection("召回结果充足，进入精排阶段")
        trace.set_final_answer("为您推荐以下电影...")
        trace.finalize()
        trace.save_to_file("traces/")
    """

    def __init__(self, query: str = '', intent: str = '',
                 session_id: str = '', user_id: int = 0,
                 system_config: str = 'full'):
        self.query = query
        self.intent = intent
        self.session_id = session_id
        self.user_id = user_id
        self.system_config = system_config  # 'full' | 'no_kag' | 'no_rag' | ...

        self.steps = []
        self.step_counter = 0
        self.start_time = time.time()
        self.end_time = None
        self.total_latency_ms = 0.0

        # 汇总统计
        self.tool_call_count = 0
        self.tools_used = []
        self.retry_count = 0
        self.candidate_pool_size = 0
        self.final_recommendation_count = 0

    # ── 添加各阶段步骤 ──

    def add_thought(self, content: str, metadata: dict = None):
        step = TraceStep(
            step_id=self.step_counter,
            stage='thought',
            input_summary=content[:500],
            metadata=metadata,
        )
        self.steps.append(step)
        self.step_counter += 1
        return step

    def add_action(self, tool_name: str, input_text: str = '',
                   latency_ms: float = 0.0, is_retry: bool = False,
                   retry_from: str = '', metadata: dict = None):
        step = TraceStep(
            step_id=self.step_counter,
            stage='action',
            tool_name=tool_name,
            input_summary=input_text[:300],
            latency_ms=latency_ms,
            is_retry=is_retry,
            retry_from=retry_from,
            metadata=metadata,
        )
        self.steps.append(step)
        self.step_counter += 1
        self.tool_call_count += 1
        if tool_name not in self.tools_used:
            self.tools_used.append(tool_name)
        if is_retry:
            self.retry_count += 1
        return step

    def add_observation(self, tool_name: str, output_count: int = 0,
                        output_summary: str = '', is_retry: bool = False,
                        retry_from: str = '', metadata: dict = None):
        step = TraceStep(
            step_id=self.step_counter,
            stage='observation',
            tool_name=tool_name,
            output_summary=output_summary[:500] if output_summary else '',
            output_count=output_count,
            is_retry=is_retry,
            retry_from=retry_from,
            metadata=metadata,
        )
        self.steps.append(step)
        self.step_counter += 1
        return step

    def add_reflection(self, content: str, metadata: dict = None):
        """Reflection 阶段：Agent 对当前观察结果的反思与下一步规划"""
        step = TraceStep(
            step_id=self.step_counter,
            stage='reflection',
            input_summary=content[:500],
            metadata=metadata,
        )
        self.steps.append(step)
        self.step_counter += 1
        return step

    def set_final_answer(self, content: str, recommended_ids: list = None,
                         explanations: dict = None):
        self.final_answer = content
        self.final_recommendation_count = len(recommended_ids) if recommended_ids else 0
        self._recommended_ids = recommended_ids or []
        self._explanations = explanations or {}

    def mark_candidate_pool(self, count: int):
        self.candidate_pool_size = max(self.candidate_pool_size, count)

    # ── 完成与导出 ──

    def finalize(self):
        self.end_time = time.time()
        self.total_latency_ms = (self.end_time - self.start_time) * 1000

    def to_dict(self) -> OrderedDict:
        d = OrderedDict()
        d['query'] = self.query
        d['intent'] = self.intent
        d['system_config'] = self.system_config
        d['session_id'] = self.session_id
        d['user_id'] = self.user_id
        d['timestamp'] = datetime.fromtimestamp(self.start_time).isoformat()
        d['total_latency_ms'] = round(self.total_latency_ms, 1)
        d['tool_calls'] = self.tool_call_count
        d['tools_used'] = self.tools_used
        d['retry_count'] = self.retry_count
        d['candidate_pool_size'] = self.candidate_pool_size
        d['final_recommendation_count'] = self.final_recommendation_count
        d['recommended_ids'] = self._recommended_ids
        d['explanations'] = self._explanations
        d['thoughts'] = [s.input_summary for s in self.steps if s.stage == 'thought']
        d['actions'] = [s.to_dict() for s in self.steps if s.stage == 'action']
        d['observations'] = [s.to_dict() for s in self.steps if s.stage == 'observation']
        d['reflections'] = [s.input_summary for s in self.steps if s.stage == 'reflection']
        d['steps'] = [s.to_dict() for s in self.steps]
        if hasattr(self, 'final_answer'):
            d['final_answer'] = self.final_answer
        return d

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    def to_react_text(self) -> str:
        """
        生成论文展示用的 ReAct 推理链可读文本
        """
        lines = []
        lines.append("=" * 60)
        lines.append(f"[Query] {self.query}")
        lines.append(f"[Intent] {self.intent}")
        lines.append("")

        for step in self.steps:
            stage = step.stage.upper()
            retry_flag = " ⚡[纠偏]" if step.is_retry else ""
            lines.append(f"── Step {step.step_id} [{stage}]{retry_flag} ──")

            if step.stage == 'thought':
                lines.append(f"  {step.input_summary}")
            elif step.stage == 'action':
                lines.append(f"  Tool: {step.tool_name}")
                lines.append(f"  Input: {step.input_summary}")
                if step.latency_ms > 0:
                    lines.append(f"  Latency: {step.latency_ms:.1f}ms")
            elif step.stage == 'observation':
                lines.append(f"  Tool: {step.tool_name}")
                lines.append(f"  Count: {step.output_count}")
                if step.output_summary:
                    lines.append(f"  Summary: {step.output_summary}")
            elif step.stage == 'reflection':
                lines.append(f"  {step.input_summary}")
            lines.append("")

        if hasattr(self, 'final_answer'):
            lines.append("── FINAL ANSWER ──")
            lines.append(self.final_answer)
        lines.append("=" * 60)
        return "\n".join(lines)

    def save_to_file(self, directory: str = 'traces/') -> str:
        """将 Trace 保存为 JSON 文件"""
        os.makedirs(directory, exist_ok=True)
        ts = datetime.fromtimestamp(self.start_time).strftime('%Y%m%d_%H%M%S')
        filename = f"trace_{ts}_{self.system_config}.json"
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
        return filepath


class TraceCollector:
    """
    批量 Trace 收集器
    用于实验中收集多条 Trace 并计算汇总指标
    """

    def __init__(self):
        self.traces = []

    def add(self, trace: AgentTrace):
        self.traces.append(trace)

    def summary(self) -> dict:
        if not self.traces:
            return {}

        n = len(self.traces)
        total_tool_calls = sum(t.tool_call_count for t in self.traces)
        total_retries = sum(t.retry_count for t in self.traces)
        total_latency = sum(t.total_latency_ms for t in self.traces)

        tool_usage = {}
        for t in self.traces:
            for tool in t.tools_used:
                tool_usage[tool] = tool_usage.get(tool, 0) + 1

        return {
            'trace_count': n,
            'avg_tool_calls': round(total_tool_calls / n, 2),
            'avg_retries': round(total_retries / n, 2),
            'avg_latency_ms': round(total_latency / n, 1),
            'tool_usage_distribution': tool_usage,
            'total_retries': total_retries,
        }

    def save_all(self, directory: str = 'traces/'):
        os.makedirs(directory, exist_ok=True)
        paths = []
        for trace in self.traces:
            path = trace.save_to_file(directory)
            paths.append(path)
        # 保存汇总
        summary_path = os.path.join(directory, 'trace_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(self.summary(), f, ensure_ascii=False, indent=2)
        paths.append(summary_path)
        return paths