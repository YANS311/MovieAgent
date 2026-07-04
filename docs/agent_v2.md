# Agent v2 架构演进文档

## 演进目标

在不破坏现有接口的前提下，将 MovieAgent 架构逐步演进为更加现代的 Agent Engineering 设计。

**原则：** 渐进式演进，兼容层优先，不推倒重写。

## 架构总览

```
用户输入
  ↓
MovieAgent.run()          ← 现有入口，不变
  ↓
SkillRouter.route()       ← v2 新增：轻量路由器
  ↓
SkillRegistry.select()    ← v2 新增：按 Metadata 路由
  ↓
BaseSkill.run(ctx)        ← v2 新增：统一上下文
  ↓
┌─────────────────────────────────────────────┐
│  VectorSearchSkill     FAISS 向量语义召回    │
│  GraphReasoningSkill   Neo4j 知识图谱推理    │
│  HybridRecallSkill     五路并行召回          │
│  NeuralRerankSkill     MAAN 深度精排         │
│  ExplanationSkill      推荐理由生成          │
│  KnowledgeRetrieval    统一知识检索（新）    │
└─────────────────────────────────────────────┘
  ↓
SkillMetrics.record()     ← v2 新增：调用指标
  ↓
SkillContext 更新          ← v2 新增：统一数据流
  ↓
MovieAgent 输出            ← 现有输出格式，不变
```

## 为什么引入这些组件

### Skill 抽象层

**问题：** 现有 AgentTool 与 MovieAgent 紧密耦合，Tool 之间通过散装 kwargs 传递数据，无法独立复用。

**方案：** BaseSkill 提供统一接口（can_handle / run / fallback），每个 Skill 封装一个独立的推荐子能力。

**收益：**
- Skill 可独立测试和复用
- can_handle() 支持动态路由，替代硬编码的 INTENT_TOOL_MAP
- fallback() 提供优雅降级，而非直接抛异常
- input_schema / output_schema 为未来 MCP 适配做准备

### SkillContext

**问题：** 现有 Tool 之间通过 kwargs 传递数据，字段不统一，调试困难。

**方案：** SkillContext 作为统一的数据容器，贯穿整个推理链。

**收益：**
- 类型安全，字段有明确语义
- 内置 trace 记录，调试时可回溯完整数据流
- to_tool_kwargs() 兼容旧 Tool 接口
- snapshot() 支持序列化，可用于日志和评估

### Skill Metadata

**问题：** 路由时无法区分 Skill 的优先级、成本、延迟特征。

**方案：** 每个 Skill 声明 priority / latency_level / cost_level / tags。

**收益：**
- Router 可按 priority 排序选择
- 未来可按成本/延迟约束动态选择
- tags 支持按能力分类检索

### SkillRouter

**问题：** 现有 Intent → Tool 映射硬编码在 MovieAgent.INTENT_TOOL_MAP 中。

**方案：** SkillRouter 提供轻量级的路由层，将 Intent 映射到 Skill 链。

**收益：**
- 路由逻辑独立，可单独测试
- 支持空结果纠偏（FALLBACK_CHAIN）
- 自动记录调用指标

### SkillMetrics

**问题：** 无法追踪每个 Skill 的调用成功率、耗时、降级频率。

**方案：** 内存实现的指标统计器。

**收益：**
- 实时监控 Skill 健康状况
- 为未来的自适应路由提供数据基础
- 不引入数据库依赖

## 当前 Skill 列表

| Skill | 类型 | Priority | 延迟 | 成本 | Tags |
|-------|------|----------|------|------|------|
| `recall_hybrid` | Tool | 95 | medium | medium | retrieval, hybrid |
| `search_vector` | Tool | 90 | low | low | retrieval, semantic |
| `maan_rerank` | Tool | 85 | high | high | ranking, neural |
| `knowledge_retrieval` | Capability | 80 | medium | medium | retrieval, unified |
| `kg_query` | Tool | 70 | medium | low | retrieval, graph |
| `explain` | Tool | 60 | medium | low | explanation, xai |

### Tool vs Capability

- **Tool Skill：** 封装单个工具（如 FAISS、Neo4j），对应一个 AgentTool
- **Capability Skill：** 封装完整能力管线（如 KnowledgeRetrievalSkill 内部调用 SQL + Neo4j + Vector + Merge）

## 为什么暂时没有采用

### MCP（Model Context Protocol）

MCP 提供标准化的工具接口协议，但当前项目：
- 只有一个 Agent 实例，不需要跨进程的工具协议
- 所有 Tool 都在同一进程内，直接函数调用更高效
- MCP 需要额外的 Server 进程和网络通信

**计划：** 当需要接入外部工具（GitHub API、外部数据库）时，再引入 MCP Adapter。

### LangGraph / AutoGen

这些框架提供复杂的 Agent 编排能力，但当前项目：
- 推荐流程是线性的（召回 → 精排 → 重排 → 解释），不需要 DAG 调度
- 只有一个 Agent，不需要 Multi-Agent 协作
- 引入框架会增加依赖复杂度

**计划：** 当需要 Multi-Agent 协作时，再评估是否引入。

### Multi-Agent

当前推荐流程可以由单个 Agent 完成，Multi-Agent 会增加：
- Agent 间通信开销
- 状态同步复杂度
- 调试难度

**计划：** 当需要 Specialist Agents（如独立的推荐 Agent、解释 Agent、对话 Agent）时，再引入。

## 文件结构

```
myapp/agent/
├── context.py              SkillContext 统一上下文
├── router.py               SkillRouter 轻量路由器
├── metrics.py              SkillMetrics 调用指标
├── skills/
│   ├── __init__.py         统一导出
│   ├── base.py             BaseSkill 抽象基类（含 Metadata）
│   ├── registry.py         SkillRegistry 注册中心
│   ├── vector_search_skill.py
│   ├── graph_reasoning_skill.py
│   ├── hybrid_recall_skill.py
│   ├── neural_rerank_skill.py
│   ├── explanation_skill.py
│   └── knowledge_retrieval_skill.py   能力导向的统一检索
└── movie_agent.py          现有主引擎（不变）

docs/
├── agent_skills.md         Skill 层说明
└── agent_v2.md             v2 架构演进文档（本文件）
```

## 未来扩展路线图

### 阶段 A：MCP Adapter

```
Skill → MCP Adapter → 外部 MCP Server
                        ├── GitHub MCP（PR/Issue 管理）
                        ├── Neo4j MCP（图数据库查询）
                        └── Filesystem MCP（文件操作）
```

BaseSkill 的 input_schema / output_schema 可直接映射为 MCP tool 定义。

### 阶段 B：Multi-Agent

```
Router → Specialist Agents
           ├── RecommenderAgent（推荐专用）
           ├── ExplainerAgent（解释专用）
           └── ConversationalAgent（对话专用）
```

SkillRegistry 可作为能力发现中心，Agent 间通过 Skill 接口协作。

### 阶段 C：自适应路由

```
Router → SkillMetrics（历史数据）
       → Cost/Latency 约束
       → 动态选择最优 Skill 组合
```

SkillMetrics 的统计数据可驱动路由策略的自适应优化。
