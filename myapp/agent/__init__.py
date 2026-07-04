"""
Agent 智能推荐模块
================================================
基于 ReAct 范式的电影推荐智能体系统

核心组件：
  - movie_agent.py: MovieAgent 主引擎 + IntentClassifier + 工具集
  - react_agent.py: ReAct 推理智能体（Thought→Action→Observation→Reflection 循环）
  - workflow_agent.py: 固定工作流基线（用于对比实验）
  - memory.py: 多轮对话记忆管理
  - trace_logger.py: 推理链追踪与持久化
  - tool_registry.py: 工具注册中心（统一管理工具元信息）
  - evaluate_agent.py: Agent Benchmark 评测体系
  - evaluator.py: 经典推荐指标评估模块

推荐引擎（位于 recommender/ 目录）：
  - recall.py: 多路召回（向量/内容/模型/知识图谱/热门）
  - rank.py: 精排
  - rerank.py: 重排（多样性保障）
  - explain.py: 推荐解释生成
  - evaluate.py: 离线指标评估
================================================
"""

from myapp.agent.movie_agent import MovieAgent, IntentClassifier
from myapp.agent.memory import MemoryManager
from myapp.agent.trace_logger import AgentTrace, TraceCollector
from myapp.agent.tool_registry import ToolRegistry, ToolSpec, get_global_registry
from myapp.agent.react_agent import ReActAgent
from myapp.agent.workflow_agent import WorkflowAgent, SequentialPipelineAgent
from myapp.agent.evaluate_agent import AgentBenchmark, AGENT_EVAL_SET
from myapp.agent.evaluator import (
    run_full_evaluation,
    evaluate_rag_ablation,
    evaluate_kag_accuracy,
    evaluate_react_vs_workflow,
    evaluate_ablation,
    hit_rate, ndcg_at_k, mrr,
)