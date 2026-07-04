# Agent Benchmark 评估框架

## 概述

MovieAgent 内置了 30 题标准评估集，用于离线评估 Agent 的推理质量。评估覆盖 7 种意图类型，输出包括意图准确率、工具选择准确率、召回命中率、推理成功率、延迟和降级率。

## 评估集覆盖场景

| 意图 | 题数 | 场景描述 | 期望工具链 |
|------|------|----------|-----------|
| `QUERY_MOVIE` | 10 | 类型/导演/评分/年份约束查询 | search_vector → maan_rerank → rerank |
| `QUERY_PROFILE_REC` | 5 | 基于用户画像的个性化推荐 | recall_hybrid → maan_rerank → rerank |
| `QUERY_COMPARISON` | 5 | 相似电影推荐（"类似X的"） | search_vector → maan_rerank → rerank |
| `QUERY_KG` | 3 | 知识图谱关系推理 | kg_query |
| `QUERY_RANK` | 2 | 排序类查询（"评分最高的"） | search_vector → maan_rerank |
| `QUERY_NEW` | 2 | 新片推荐 | search_vector → maan_rerank |
| `CHAT` | 3 | 闲聊/模糊查询/边界情况 | （无工具或 search_vector） |

### 特殊测试场景

- **模糊查询**（3 题）："推荐好看的"、"有没有电影看"、"推荐"
- **复合约束**（5 题）：多条件组合，如"2020年以后评分8分以上的韩国悬疑片"
- **风格类比**（2 题）："类似疯狂动物城那种风格的"、"喜欢宫崎骏的动画"
- **导演+类型**（2 题）："推荐诺兰导演的电影"、"推荐马丁斯科塞斯的黑帮片"

## 指标定义

### Intent Accuracy（意图准确率）

```
Intent Accuracy = 意图分类正确的题数 / 总题数
```

衡量 Agent 的意图分类器是否正确理解了用户需求。

### Tool Selection Accuracy（工具选择准确率）

```
Tool Selection = 工具链完全匹配的题数 / 总题数
```

衡量 Agent 是否为每种意图选择了正确的工具链（允许顺序不同）。

### Recall Hit@K（召回命中率）

```
Recall Hit@K = 有推荐结果的题数 / 总题数
```

衡量 Agent 是否能为用户查询返回非空的推荐列表。

### Reasoning Success（推理成功率）

```
Reasoning Success = 有最终回答的题数 / 总题数
```

衡量 Agent 是否完成了完整的推理链并生成了最终回答。

### Fallback Rate（降级率）

```
Fallback Rate = 使用了降级策略的题数 / 总题数
```

衡量 Agent 在正常路径失败时的降级频率。越低越好。

### Latency（延迟）

报告 P50、P95、Max 延迟，用于评估 Agent 的响应速度。

### Agent Score（综合分）

```
Agent Score = (Intent + Tool + Recall + Reasoning) / (4 × 总题数) × 100
```

四项指标的归一化综合分，满分 100。

## 运行方式

### 基本用法

```python
from myapp.agent.benchmark import AgentBenchmark

# 创建 benchmark（需要 MovieAgent 实例）
benchmark = AgentBenchmark(agent)

# 运行全部 30 题
report = benchmark.run()

# 打印报告
benchmark.print_report(report)

# 保存报告
benchmark.save_report(report, "benchmark_report.json")
```

### 限制题目数（调试）

```python
# 只跑前 5 题，用于快速验证
report = benchmark.run(max_questions=5)
```

### 自定义评估集

```python
custom_eval = [
    {"id": 1, "query": "推荐科幻片", "intent": "QUERY_MOVIE",
     "expected_tools": ["search_vector", "maan_rerank", "rerank"]},
]

benchmark = AgentBenchmark(agent, eval_set=custom_eval)
report = benchmark.run()
```

### 在 Django Shell 中运行

```bash
python manage.py shell

>>> from myapp.agent.movie_agent import MovieAgent
>>> from myapp.agent.benchmark import AgentBenchmark
>>> agent = MovieAgent(user=some_user)
>>> benchmark = AgentBenchmark(agent)
>>> report = benchmark.run(max_questions=5)
>>> benchmark.print_report(report)
```

## 示例输出

> 以下为示例格式，非真实评估结果。

```
============================================================
  Agent Benchmark Report
============================================================
  Questions:      30
  Total Time:     45.2s
  Agent Score:    86.7
------------------------------------------------------------
  Intent Accuracy:        90.0%
  Tool Selection:         83.3%
  Recall Hit@K:           96.7%
  Reasoning Success:      100.0%
  Fallback Rate:          6.7%
  Error Rate:             0.0%
------------------------------------------------------------
  Latency Avg:            1250ms
  Latency P50:            980ms
  Latency P95:            3200ms
  Latency Max:            4500ms
============================================================
```

## JSON 报告格式

```json
{
  "summary": {
    "total_questions": 30,
    "total_time_s": 45.2,
    "agent_score": 86.7
  },
  "metrics": {
    "intent_accuracy": 0.9,
    "tool_selection_accuracy": 0.8333,
    "recall_hit_rate": 0.9667,
    "reasoning_success_rate": 1.0,
    "fallback_rate": 0.0667,
    "error_rate": 0.0
  },
  "latency": {
    "avg_ms": 1250.0,
    "min_ms": 200,
    "max_ms": 4500,
    "p50_ms": 980,
    "p95_ms": 3200
  },
  "details": [
    {
      "id": 1,
      "query": "推荐几部诺兰导演的电影",
      "expected_intent": "QUERY_MOVIE",
      "actual_intent": "QUERY_MOVIE",
      "intent_correct": true,
      "expected_tools": ["search_vector", "maan_rerank", "rerank"],
      "actual_tools": ["search_vector", "maan_rerank", "rerank"],
      "tools_correct": true,
      "recall_hit": true,
      "reasoning_success": true,
      "latency_ms": 1100,
      "fallback_used": false,
      "recommended_count": 5
    }
  ]
}
```

## Skill Benchmark

除 Agent 级别的评估外，还可以对单个 Skill 进行评估：

```python
from myapp.agent.benchmark import SkillBenchmark
from myapp.agent.metrics import get_global_metrics

# 运行一段时间后导出指标
metrics = get_global_metrics()
SkillBenchmark.export_metrics(metrics, "skill_metrics.json")
```

导出格式：

```json
{
  "exported_at": "2026-07-04 19:30:00",
  "skills": [
    {
      "skill": "search_vector",
      "call_count": 150,
      "success_count": 148,
      "fail_count": 2,
      "success_rate": 0.9867,
      "avg_latency_ms": 22.5,
      "fallback_count": 3,
      "fallback_rate": 0.02,
      "last_error": null
    }
  ]
}
```

## Trace Replay

配合 AgentTrace 使用，可以回放 benchmark 中每道题的推理过程：

```python
from myapp.agent.trace_replay import AgentTrace, TraceStore

store = TraceStore("traces/")

# benchmark 运行时自动记录 trace
for item in report['details']:
    if not item['intent_correct']:  # 只看意图分类错误的
        trace = store.load(f"trace_{item['id']}.json")
        steps = trace.replay()
        for step in steps:
            print(f"  Step {step['step']}: {step['type']}")
            print(f"    Snapshot: {step['snapshot']}")
```
