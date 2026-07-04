"""
SkillMetrics — Skill 调用指标统计（内存实现）
=================================================
统计每个 Skill 的:
  - 调用次数
  - 成功 / 失败次数
  - 平均耗时
  - fallback 次数
  - 最近异常

不引入数据库，纯内存实现。
=================================================
"""

import time
import logging
from collections import defaultdict
from threading import Lock

logger = logging.getLogger('movie_agent')


class SkillMetrics:
    """Skill 调用指标统计器（线程安全，单例使用）。"""

    def __init__(self):
        self._lock = Lock()
        self._data: dict[str, dict] = defaultdict(lambda: {
            'call_count': 0,
            'success_count': 0,
            'fail_count': 0,
            'fallback_count': 0,
            'total_latency_ms': 0.0,
            'last_error': None,
            'last_error_time': None,
            'last_call_time': None,
        })

    def record(self, skill_name: str, success: bool, latency_ms: float,
               fallback: bool = False, error: str = None):
        """
        记录一次 Skill 调用。

        Args:
            skill_name: Skill 名称
            success: 是否成功
            latency_ms: 耗时（毫秒）
            fallback: 是否走了降级路径
            error: 错误信息（失败时）
        """
        with self._lock:
            d = self._data[skill_name]
            d['call_count'] += 1
            d['last_call_time'] = time.time()

            if success:
                d['success_count'] += 1
            else:
                d['fail_count'] += 1
                d['last_error'] = error
                d['last_error_time'] = time.time()

            if fallback:
                d['fallback_count'] += 1

            d['total_latency_ms'] += latency_ms

    def get(self, skill_name: str) -> dict:
        """获取指定 Skill 的指标。"""
        with self._lock:
            d = self._data.get(skill_name)
            if not d:
                return {}
            return self._enrich(skill_name, d)

    def summary(self) -> list:
        """返回所有 Skill 的指标摘要（按调用次数降序）。"""
        with self._lock:
            result = []
            for name, d in self._data.items():
                result.append(self._enrich(name, d))
            result.sort(key=lambda x: x['call_count'], reverse=True)
            return result

    def reset(self, skill_name: str = None):
        """重置指标。"""
        with self._lock:
            if skill_name:
                if skill_name in self._data:
                    del self._data[skill_name]
            else:
                self._data.clear()

    def _enrich(self, name: str, d: dict) -> dict:
        """计算衍生指标。"""
        call_count = d['call_count']
        avg_latency = (d['total_latency_ms'] / call_count) if call_count > 0 else 0
        success_rate = (d['success_count'] / call_count) if call_count > 0 else 0
        fallback_rate = (d['fallback_count'] / call_count) if call_count > 0 else 0

        return {
            'skill': name,
            'call_count': call_count,
            'success_count': d['success_count'],
            'fail_count': d['fail_count'],
            'success_rate': round(success_rate, 4),
            'avg_latency_ms': round(avg_latency, 2),
            'fallback_count': d['fallback_count'],
            'fallback_rate': round(fallback_rate, 4),
            'last_error': d['last_error'],
            'last_error_time': d['last_error_time'],
            'last_call_time': d['last_call_time'],
        }

    def __repr__(self) -> str:
        return f"<SkillMetrics skills={len(self._data)}>"


# ── 全局单例 ─────────────────────────────────────────────
_global_metrics = None


def get_global_metrics() -> SkillMetrics:
    """获取全局指标单例。"""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = SkillMetrics()
    return _global_metrics
