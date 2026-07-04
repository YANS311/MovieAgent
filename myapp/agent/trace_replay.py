"""
AgentTrace — 推理链追踪与回放
=================================================
记录 Agent 推理过程中的完整状态变化，支持:
  - save(): 持久化为 JSON 文件
  - load(): 从 JSON 恢复
  - replay(): 回放推理过程（不重新调用工具，只重放上下文变化）
  - diff(): 对比两次 trace 的差异

设计目标:
  - 每一步记录 SkillContext 的 snapshot
  - 支持论文中的 "推理链可视化"
  - 支持 Debug 时的 "状态回溯"
  - 支持 Benchmark 的 "结果复现"
=================================================
"""

import os
import json
import time
import logging
from typing import List, Optional
from dataclasses import asdict

logger = logging.getLogger('movie_agent')


class AgentTrace:
    """
    单次 Agent 推理的完整追踪记录。

    使用方式:
        trace = AgentTrace(session_id="user_123_abc")

        # 在推理过程中记录
        trace.record_step("intent", context)
        trace.record_step("recall", context)
        trace.record_step("rerank", context)
        trace.record_step("output", context)

        # 保存
        trace.save("traces/user_123_abc.json")

        # 回放
        loaded = AgentTrace.load("traces/user_123_abc.json")
        loaded.replay()
    """

    def __init__(self, session_id: str = "", user_input: str = ""):
        self.session_id = session_id
        self.user_input = user_input
        self.created_at = time.time()
        self.steps: List[dict] = []
        self.final_result: dict = {}
        self.metadata: dict = {
            'total_latency_ms': 0,
            'step_count': 0,
            'intent': '',
            'skill_chain': [],
            'fallback_used': False,
            'success': False,
        }

    def record_step(self, step_type: str, context, **extra):
        """
        记录一个推理步骤。

        Args:
            step_type: 步骤类型（intent/recall/rerank/explain/output）
            context: SkillContext 实例
            **extra: 额外信息
        """
        snapshot = context.snapshot() if hasattr(context, 'snapshot') else {}

        step = {
            'step': len(self.steps),
            'type': step_type,
            'timestamp': time.time(),
            'snapshot': snapshot,
            'tool_results': {},
            **extra,
        }

        # 记录当前步骤的工具结果（只保留摘要）
        if hasattr(context, 'tool_results'):
            for name, result in context.tool_results.items():
                if name not in self.steps[-1].get('tool_results', {}) if self.steps else True:
                    step['tool_results'][name] = {
                        'success': result.get('success', False),
                        'count': result.get('meta', {}).get('count', len(result.get('data', []))),
                        'source': result.get('meta', {}).get('source', ''),
                        'fallback': result.get('meta', {}).get('fallback', False),
                    }

        self.steps.append(step)
        self.metadata['step_count'] = len(self.steps)

    def record_final(self, context):
        """记录最终结果。"""
        if hasattr(context, 'final_answer'):
            self.final_result = {
                'final_answer': context.final_answer,
                'recommended_ids': context.recommended_ids,
                'explanations': {
                    mid: {'text': exp.get('reason_text', ''), 'type': exp.get('reason_type', '')}
                    for mid, exp in context.explanations.items()
                } if hasattr(context, 'explanations') else {},
            }
            self.metadata['intent'] = context.intent
            self.metadata['total_latency_ms'] = context.latency_ms
            self.metadata['success'] = bool(context.final_answer)
            self.metadata['fallback_used'] = context.metadata.get('fallback_used', False)
            self.metadata['skill_chain'] = context.tool_chain if hasattr(context, 'tool_chain') else []

    def to_dict(self) -> dict:
        """序列化为字典。"""
        return {
            'session_id': self.session_id,
            'user_input': self.user_input,
            'created_at': self.created_at,
            'created_at_iso': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at)),
            'steps': self.steps,
            'final_result': self.final_result,
            'metadata': self.metadata,
        }

    def save(self, filepath: str):
        """
        保存 trace 到 JSON 文件。

        Args:
            filepath: 保存路径（自动创建目录）
        """
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"[AgentTrace] Saved to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "AgentTrace":
        """
        从 JSON 文件加载 trace。

        Args:
            filepath: JSON 文件路径

        Returns:
            AgentTrace 实例
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        trace = cls(
            session_id=data.get('session_id', ''),
            user_input=data.get('user_input', ''),
        )
        trace.created_at = data.get('created_at', 0)
        trace.steps = data.get('steps', [])
        trace.final_result = data.get('final_result', {})
        trace.metadata = data.get('metadata', {})
        return trace

    def replay(self) -> List[dict]:
        """
        回放推理过程。返回每一步的上下文快照。

        不重新调用工具，只重放已记录的状态变化。
        用于调试和论文中的推理链可视化。
        """
        replay_steps = []
        for step in self.steps:
            replay_steps.append({
                'step': step['step'],
                'type': step['type'],
                'snapshot': step['snapshot'],
                'tool_results': step.get('tool_results', {}),
                'timestamp': step.get('timestamp', 0),
            })
        return replay_steps

    def diff(self, other: "AgentTrace") -> dict:
        """
        对比两次 trace 的差异。

        用于:
          - 评估不同模型/配置对同一查询的处理差异
          - 回归测试：确保修改后行为一致
        """
        diff_result = {
            'session_ids': [self.session_id, other.session_id],
            'step_count_diff': len(self.steps) - len(other.steps),
            'intent_match': self.metadata.get('intent') == other.metadata.get('intent'),
            'skill_chain_match': self.metadata.get('skill_chain') == other.metadata.get('skill_chain'),
            'latency_diff_ms': (
                self.metadata.get('total_latency_ms', 0) - other.metadata.get('total_latency_ms', 0)
            ),
            'recommended_ids_match': (
                self.final_result.get('recommended_ids', []) ==
                other.final_result.get('recommended_ids', [])
            ),
            'step_diffs': [],
        }

        # 逐步对比
        max_steps = max(len(self.steps), len(other.steps))
        for i in range(max_steps):
            s1 = self.steps[i] if i < len(self.steps) else None
            s2 = other.steps[i] if i < len(other.steps) else None

            if s1 and s2:
                if s1['type'] != s2['type']:
                    diff_result['step_diffs'].append({
                        'step': i,
                        'type': 'changed',
                        'from': s1['type'],
                        'to': s2['type'],
                    })
            elif s1 and not s2:
                diff_result['step_diffs'].append({'step': i, 'type': 'removed', 'value': s1['type']})
            elif s2 and not s1:
                diff_result['step_diffs'].append({'step': i, 'type': 'added', 'value': s2['type']})

        return diff_result

    @property
    def summary(self) -> dict:
        """返回摘要信息。"""
        return {
            'session_id': self.session_id,
            'user_input': self.user_input[:50],
            'intent': self.metadata.get('intent', ''),
            'step_count': len(self.steps),
            'latency_ms': self.metadata.get('total_latency_ms', 0),
            'success': self.metadata.get('success', False),
            'fallback_used': self.metadata.get('fallback_used', False),
            'recommended_count': len(self.final_result.get('recommended_ids', [])),
        }

    def __repr__(self) -> str:
        return (
            f"AgentTrace(session='{self.session_id}', "
            f"steps={len(self.steps)}, "
            f"intent='{self.metadata.get('intent', '')}')"
        )


# ── Trace 存储管理 ───────────────────────────────────────

class TraceStore:
    """
    Trace 文件存储管理器。

    使用方式:
        store = TraceStore("traces/")
        store.save(trace)
        traces = store.list_traces()
        trace = store.load(traces[0])
    """

    def __init__(self, base_dir: str = "traces"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def save(self, trace: AgentTrace, filename: str = None) -> str:
        """保存 trace，返回文件路径。"""
        if not filename:
            ts = time.strftime('%Y%m%d_%H%M%S')
            filename = f"trace_{trace.session_id}_{ts}.json"
        filepath = os.path.join(self.base_dir, filename)
        trace.save(filepath)
        return filepath

    def load(self, filename: str) -> AgentTrace:
        """加载指定 trace。"""
        filepath = os.path.join(self.base_dir, filename)
        return AgentTrace.load(filepath)

    def list_traces(self, limit: int = 50) -> List[dict]:
        """列出所有 trace 文件（按修改时间降序）。"""
        files = []
        for f in os.listdir(self.base_dir):
            if f.endswith('.json'):
                filepath = os.path.join(self.base_dir, f)
                stat = os.stat(filepath)
                files.append({
                    'filename': f,
                    'size_kb': round(stat.st_size / 1024, 1),
                    'modified': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stat.st_mtime)),
                })
        files.sort(key=lambda x: x['modified'], reverse=True)
        return files[:limit]

    def get_summary(self) -> dict:
        """返回 trace 存储的统计摘要。"""
        traces = self.list_traces(limit=1000)
        return {
            'total_traces': len(traces),
            'total_size_kb': sum(t['size_kb'] for t in traces),
            'base_dir': self.base_dir,
        }
