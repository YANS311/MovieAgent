"""
Agent Benchmark — Agent 离线评估框架
=================================================
30 个标准测试问题，覆盖 6 种意图。

评估指标:
  - Intent Accuracy: 意图分类准确率
  - Tool Selection Accuracy: 工具选择准确率
  - Recall Hit@K: 召回命中率
  - Reasoning Success: 推理成功率（有最终推荐结果）
  - Latency: 平均耗时
  - Fallback Rate: 降级使用率

使用方式:
    from myapp.agent.benchmark import AgentBenchmark, EVAL_SET

    benchmark = AgentBenchmark(agent)
    report = benchmark.run()
    benchmark.print_report(report)
    benchmark.save_report(report, "benchmark_report.json")
=================================================
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional

logger = logging.getLogger('movie_agent')


# ── 30 题评估集 ──────────────────────────────────────────

EVAL_SET: List[Dict] = [
    # ── QUERY_MOVIE（10 题）─────────────────────────────
    {"id": 1,  "query": "推荐几部诺兰导演的电影",         "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 2,  "query": "有没有好看的科幻片",             "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 3,  "query": "推荐评分8分以上的悬疑电影",       "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 4,  "query": "最近有什么好看的动作片",         "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 5,  "query": "推荐适合周末看的轻松喜剧",       "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector", "maan_rerank", "rerank"]},

    # ── QUERY_PROFILE_REC（5 题）───────────────────────
    {"id": 6,  "query": "根据我的喜好推荐电影",           "intent": "QUERY_PROFILE_REC", "expected_tools": ["recall_hybrid", "maan_rerank", "rerank"]},
    {"id": 7,  "query": "给我推荐几部电影",               "intent": "QUERY_PROFILE_REC", "expected_tools": ["recall_hybrid", "maan_rerank", "rerank"]},
    {"id": 8,  "query": "推荐我可能喜欢的",               "intent": "QUERY_PROFILE_REC", "expected_tools": ["recall_hybrid", "maan_rerank", "rerank"]},

    # ── QUERY_COMPARISON（5 题）────────────────────────
    {"id": 9,  "query": "推荐类似盗梦空间的电影",         "intent": "QUERY_COMPARISON",  "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 10, "query": "有没有像银翼杀手那样的",         "intent": "QUERY_COMPARISON",  "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 11, "query": "推荐和星际穿越风格类似的",       "intent": "QUERY_COMPARISON",  "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 12, "query": "有没有教父那种风格的犯罪片",     "intent": "QUERY_COMPARISON",  "expected_tools": ["search_vector", "maan_rerank", "rerank"]},

    # ── QUERY_KG（3 题）────────────────────────────────
    {"id": 13, "query": "诺兰导演和哪些演员合作最多",     "intent": "QUERY_KG",          "expected_tools": ["kg_query"]},
    {"id": 14, "query": "科幻片里评分最高的是哪部",       "intent": "QUERY_KG",          "expected_tools": ["kg_query"]},
    {"id": 15, "query": "汤姆汉克斯演过哪些高分电影",     "intent": "QUERY_KG",          "expected_tools": ["kg_query"]},

    # ── QUERY_RANK（3 题）──────────────────────────────
    {"id": 16, "query": "按评分排一下2024年的电影",       "intent": "QUERY_RANK",        "expected_tools": ["search_vector", "maan_rerank"]},
    {"id": 17, "query": "2023年票房最高的电影有哪些",     "intent": "QUERY_RANK",        "expected_tools": ["search_vector", "maan_rerank"]},

    # ── QUERY_NEW（2 题）───────────────────────────────
    {"id": 18, "query": "最近上映的新片有什么推荐",       "intent": "QUERY_NEW",         "expected_tools": ["search_vector", "maan_rerank"]},
    {"id": 19, "query": "2025年有什么值得期待的电影",     "intent": "QUERY_NEW",         "expected_tools": ["search_vector", "maan_rerank"]},

    # ── CHAT（2 题）────────────────────────────────────
    {"id": 20, "query": "你好",                           "intent": "CHAT",              "expected_tools": []},

    # ── 模糊查询（5 题）────────────────────────────────
    {"id": 21, "query": "推荐好看的",                     "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector"]},
    {"id": 22, "query": "有没有电影看",                   "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector"]},
    {"id": 23, "query": "推荐",                           "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector"]},

    # ── 复合约束（5 题）────────────────────────────────
    {"id": 24, "query": "推荐2020年以后评分8分以上的韩国悬疑片", "intent": "QUERY_MOVIE", "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 25, "query": "推荐适合和孩子一起看的动画片",   "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 26, "query": "推荐几部烧脑的科幻片，不要恐怖的", "intent": "QUERY_MOVIE",     "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 27, "query": "有没有类似疯狂动物城那种风格的", "intent": "QUERY_COMPARISON",  "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 28, "query": "推荐马丁斯科塞斯的黑帮片",       "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 29, "query": "喜欢宫崎骏的动画，有类似的吗",   "intent": "QUERY_COMPARISON",  "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
    {"id": 30, "query": "推荐几部一个人安静看的文艺片",   "intent": "QUERY_MOVIE",       "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
]


class AgentBenchmark:
    """
    Agent 离线评估框架。

    使用方式:
        benchmark = AgentBenchmark(agent, eval_set=EVAL_SET)
        report = benchmark.run()
    """

    def __init__(self, agent=None, eval_set: List[Dict] = None):
        """
        Args:
            agent: MovieAgent 实例（可选，为 None 时只做静态分析）
            eval_set: 评估集，默认使用内置 30 题
        """
        self.agent = agent
        self.eval_set = eval_set or EVAL_SET

    def run(self, max_questions: int = None) -> dict:
        """
        运行完整评估。

        Args:
            max_questions: 最大评估题目数（调试时可限制）

        Returns:
            评估报告字典
        """
        t_start = time.time()
        results = []
        questions = self.eval_set[:max_questions] if max_questions else self.eval_set

        for item in questions:
            result = self._evaluate_one(item)
            results.append(result)
            logger.info(f"[Benchmark] Q{item['id']}: intent={result['intent_correct']}, "
                       f"tools={result['tools_correct']}, latency={result['latency_ms']}ms")

        report = self._build_report(results, time.time() - t_start)
        return report

    def _evaluate_one(self, item: dict) -> dict:
        """评估单个问题。"""
        t0 = time.time()

        result = {
            'id': item['id'],
            'query': item['query'],
            'expected_intent': item['intent'],
            'expected_tools': item['expected_tools'],
            'actual_intent': '',
            'actual_tools': [],
            'intent_correct': False,
            'tools_correct': False,
            'recall_hit': False,
            'reasoning_success': False,
            'latency_ms': 0,
            'fallback_used': False,
            'recommended_count': 0,
            'error': None,
        }

        if not self.agent:
            result['latency_ms'] = int((time.time() - t0) * 1000)
            return result

        try:
            agent_result = self.agent.run(item['query'])
            elapsed_ms = int((time.time() - t0) * 1000)

            result['actual_intent'] = agent_result.get('intent', '')
            result['intent_correct'] = (result['actual_intent'] == item['intent'])

            # 工具链比较
            actual_tools = [a.get('tool', '') for a in agent_result.get('actions', [])]
            result['actual_tools'] = actual_tools
            result['tools_correct'] = self._tools_match(
                item['expected_tools'], actual_tools
            )

            # 召回命中
            recommended_ids = agent_result.get('recommended_ids', [])
            result['recall_hit'] = len(recommended_ids) > 0
            result['recommended_count'] = len(recommended_ids)

            # 推理成功
            result['reasoning_success'] = bool(agent_result.get('final_answer'))

            result['latency_ms'] = elapsed_ms
            result['fallback_used'] = any(
                a.get('fallback', False) for a in agent_result.get('actions', [])
            )

        except Exception as e:
            result['error'] = str(e)
            result['latency_ms'] = int((time.time() - t0) * 1000)

        return result

    def _tools_match(self, expected: list, actual: list) -> bool:
        """检查工具选择是否匹配（允许顺序不同）。"""
        return set(expected) == set(actual)

    def _build_report(self, results: list, total_time: float) -> dict:
        """构建评估报告。"""
        total = len(results)
        if total == 0:
            return {'error': 'No results'}

        intent_correct = sum(1 for r in results if r['intent_correct'])
        tools_correct = sum(1 for r in results if r['tools_correct'])
        recall_hits = sum(1 for r in results if r['recall_hit'])
        reasoning_success = sum(1 for r in results if r['reasoning_success'])
        fallback_count = sum(1 for r in results if r['fallback_used'])
        errors = sum(1 for r in results if r['error'])
        latencies = [r['latency_ms'] for r in results if r['latency_ms'] > 0]

        return {
            'summary': {
                'total_questions': total,
                'total_time_s': round(total_time, 2),
                'agent_score': round(
                    (intent_correct + tools_correct + recall_hits + reasoning_success)
                    / (total * 4) * 100, 1
                ),
            },
            'metrics': {
                'intent_accuracy': round(intent_correct / total, 4),
                'tool_selection_accuracy': round(tools_correct / total, 4),
                'recall_hit_rate': round(recall_hits / total, 4),
                'reasoning_success_rate': round(reasoning_success / total, 4),
                'fallback_rate': round(fallback_count / total, 4),
                'error_rate': round(errors / total, 4),
            },
            'latency': {
                'avg_ms': round(sum(latencies) / len(latencies), 1) if latencies else 0,
                'min_ms': min(latencies) if latencies else 0,
                'max_ms': max(latencies) if latencies else 0,
                'p50_ms': sorted(latencies)[len(latencies) // 2] if latencies else 0,
                'p95_ms': sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0,
            },
            'details': results,
        }

    def print_report(self, report: dict):
        """打印评估报告到控制台。"""
        s = report['summary']
        m = report['metrics']
        l = report['latency']

        print()
        print("=" * 60)
        print("  Agent Benchmark Report")
        print("=" * 60)
        print(f"  Questions:      {s['total_questions']}")
        print(f"  Total Time:     {s['total_time_s']}s")
        print(f"  Agent Score:    {s['agent_score']}")
        print("-" * 60)
        print(f"  Intent Accuracy:        {m['intent_accuracy']:.1%}")
        print(f"  Tool Selection:         {m['tool_selection_accuracy']:.1%}")
        print(f"  Recall Hit@K:           {m['recall_hit_rate']:.1%}")
        print(f"  Reasoning Success:      {m['reasoning_success_rate']:.1%}")
        print(f"  Fallback Rate:          {m['fallback_rate']:.1%}")
        print(f"  Error Rate:             {m['error_rate']:.1%}")
        print("-" * 60)
        print(f"  Latency Avg:            {l['avg_ms']}ms")
        print(f"  Latency P50:            {l['p50_ms']}ms")
        print(f"  Latency P95:            {l['p95_ms']}ms")
        print(f"  Latency Max:            {l['max_ms']}ms")
        print("=" * 60)
        print()

    def save_report(self, report: dict, filepath: str):
        """保存报告到 JSON 文件。"""
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info(f"[Benchmark] Report saved to {filepath}")


class SkillBenchmark:
    """
    Skill 级别的评估。

    针对每个 Skill 单独评估:
      - 调用成功率
      - 平均耗时
      - Fallback 频率
      - 结果质量（如有 ground truth）
    """

    def __init__(self, registry=None):
        self.registry = registry

    def evaluate_skills(self, test_cases: List[Dict] = None) -> dict:
        """
        评估所有已注册 Skill。

        Args:
            test_cases: 测试用例列表，每项包含:
                {"skill": "search_vector", "context": {...}, "expected_success": True}

        Returns:
            评估结果字典
        """
        if not self.registry:
            return {'error': 'No registry provided'}

        results = {}
        for skill_info in self.registry.list_skills():
            name = skill_info['name']
            results[name] = {
                'metadata': skill_info,
                'tests_passed': 0,
                'tests_failed': 0,
            }

        if test_cases:
            for case in test_cases:
                skill_name = case.get('skill')
                skill = self.registry.get(skill_name)
                if not skill:
                    continue

                try:
                    result = skill.run(case.get('context', {}))
                    if result.get('success') == case.get('expected_success', True):
                        results[skill_name]['tests_passed'] += 1
                    else:
                        results[skill_name]['tests_failed'] += 1
                except Exception:
                    results[skill_name]['tests_failed'] += 1

        return results

    @staticmethod
    def export_metrics(metrics_instance, filepath: str):
        """
        将 SkillMetrics 导出为 JSON 文件。

        Args:
            metrics_instance: SkillMetrics 实例
            filepath: 导出路径
        """
        data = {
            'exported_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'skills': metrics_instance.summary(),
        }
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
