# Agent Skill 抽象层

## 为什么引入 Skill

MovieAgent 当前的工具系统基于 `AgentTool` 基类，每个 Tool 直接在 `MovieAgent.__init__()` 中硬编码注册。这在单 Agent 场景下工作良好，但存在以下问题：

1. **耦合度高** — Tool 与 MovieAgent 紧密绑定，无法独立复用
2. **缺乏路由** — 意图到工具的映射通过 `INTENT_TOOL_MAP` 字典硬编码
3. **无降级机制** — 工具失败时需要在调用方手动 try-catch
4. **Schema 缺失** — 无法自动生成 MCP tool schema 或 Multi-Agent 协议接口

Skill 抽象层在不破坏现有 `AgentTool` 体系的前提下，提供了一条渐进式迁移路径。

## Skill 与 Tool 的区别

| 维度 | AgentTool | BaseSkill |
|------|-----------|-----------|
| 接口 | `execute(**kwargs)` | `run(context: dict)` |
| 路由 | `INTENT_TOOL_MAP` 硬编码 | `can_handle(context)` 动态选择 |
| 降级 | 调用方手动处理 | `fallback(context, error)` 自动降级 |
| Schema | 无 | `input_schema` / `output_schema` |
| MCP 适配 | 不支持 | Schema 可直接映射为 MCP tool |
| Multi-Agent | 不支持 | `SkillRegistry.select()` 可被外部 Agent 调用 |

## 当前 Skill 列表

| Skill | 对应 Tool | 能力 | 降级策略 |
|-------|-----------|------|----------|
| `search_vector` | `SearchVectorTool` | FAISS 向量语义召回 | 热门兜底 |
| `kg_query` | `KGQueryTool` | Neo4j 知识图谱推理 | 返回空结果 |
| `recall_hybrid` | `RecallHybridTool` | 五路并行召回 | 热门兜底 |
| `maan_rerank` | `MAANRerankTool` | MAAN 深度精排 | 按 score 排序 |
| `explain` | `ExplainTool` | 推荐理由生成 | 通用理由 |

## Skill 调用流程

```
用户查询
  ↓
MovieAgent（现有主流程，不变）
  ↓
INTENT_TOOL_MAP → AgentTool.execute()
  ↓
（可选迁移）SkillRegistry.select(context)
  ↓
BaseSkill.can_handle() → BaseSkill.run()
  ↓
成功 → 返回结果
失败 → BaseSkill.fallback() → 降级结果
```

**关键设计：Skill 是 Tool 的超集。** 每个 Skill 都有 `execute(**kwargs)` 方法，可以直接替换 `AgentTool` 使用，无需修改 `MovieAgent` 主流程。

## 使用示例

### 方式 1：作为独立模块使用

```python
from myapp.agent.skills import SkillRegistry, VectorSearchSkill, HybridRecallSkill

registry = SkillRegistry()
registry.register(VectorSearchSkill(rag_resources=rag))
registry.register(HybridRecallSkill(neo_graph=g, rag_resources=rag))

# 按名称调用
skill = registry.get("search_vector")
result = skill.run({"query": "推荐科幻电影", "k": 10})

# 自动选择
skill = registry.select({"intent": "QUERY_MOVIE", "query": "科幻"})
result = skill.run({"query": "科幻", "k": 10})
```

### 方式 2：兼容现有 AgentTool 接口

```python
# Skill 可以直接当作 AgentTool 使用
skill = VectorSearchSkill(rag_resources=rag)
result = skill.execute(query="科幻", k=10)  # 兼容旧接口
```

### 方式 3：渐进式迁移 MovieAgent

```python
# 在 MovieAgent.__init__ 中，逐步替换:
# 原来:
self.tools = {'search_vector': SearchVectorTool(rag_resources)}

# 迁移为:
from myapp.agent.skills import VectorSearchSkill
self.tools = {'search_vector': VectorSearchSkill(rag_resources)}
# execute() 接口完全兼容，无需修改其他代码
```

## 后续接入计划

### Multi-Agent

当引入多 Agent 协作时，`SkillRegistry` 可以作为能力发现中心：

```python
# Agent A 询问 Agent B 有哪些能力
skills = agent_b.registry.list_skills()
# [{'name': 'search_vector', 'description': '...'}, ...]

# Agent A 请求 Agent B 执行特定能力
result = agent_b.registry.get("search_vector").run(context)
```

### MCP Adapter

`BaseSkill` 的 `input_schema` / `output_schema` 可以直接映射为 MCP tool 定义：

```python
for skill in registry.list_skills():
    mcp_tool = {
        "name": skill['name'],
        "description": skill['description'],
        "inputSchema": skill['input_schema'],
    }
    # 注册到 MCP Server
```

### 可迁移的 Tool 列表

| 优先级 | 当前 Tool | 迁移状态 | 备注 |
|--------|-----------|----------|------|
| P0 | `SearchVectorTool` | ✅ 已迁移 | `VectorSearchSkill` |
| P0 | `RecallHybridTool` | ✅ 已迁移 | `HybridRecallSkill` |
| P0 | `KGQueryTool` | ✅ 已迁移 | `GraphReasoningSkill` |
| P0 | `MAANRerankTool` | ✅ 已迁移 | `NeuralRerankSkill` |
| P0 | `ExplainTool` | ✅ 已迁移 | `ExplanationSkill` |
| P1 | `SearchDatabaseTool` | 待迁移 | 可合并入 `VectorSearchSkill.fallback` |
| P2 | `RerankTool` | 待迁移 | 业务规则重排，独立性高 |

## 文件结构

```
myapp/agent/skills/
├── __init__.py              # 统一导出
├── base.py                  # BaseSkill 抽象基类
├── registry.py              # SkillRegistry 注册中心
├── vector_search_skill.py   # 向量语义搜索
├── graph_reasoning_skill.py # 知识图谱推理
├── hybrid_recall_skill.py   # 多路混合召回
├── neural_rerank_skill.py   # 神经网络精排
└── explanation_skill.py     # 推荐理由生成
```
