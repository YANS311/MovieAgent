# MovieAgent Redis 缓存 P0 重构报告

## 隔离与保护

- 原始目录：`/home/daylight/下载/DjangoProject3/DjangoProject3`。
- 新 Worktree：`/home/daylight/下载/DjangoProject3/MovieAgent-redis-cache`。
- 分支：`refactor/redis-cache`，基于 `d78390dede4a0371a1e7991f06568d8d390fbdbd`。
- 创建前只读检查：磁盘可用约 69G，目标路径和分支均不存在。
- 通过 `git -c core.hooksPath=/dev/null worktree add -b refactor/redis-cache ... HEAD`
  创建；禁用本次操作的 Git hooks，避免隐式启动任务。
- 仅应用原始 `memory.py` 的未提交补丁，修改前 `cmp` 确认逐字节一致。
- 原始 `settings.py` 的 dotenv/chapter4_v2 修改和训练脚本修改均未迁移。
- 最终核对原目录分支、HEAD、Git 状态及三个已有修改文件 SHA-256，均与开始一致。
  Git Worktree 创建按预期增加共享 Git 元数据中的分支和 Worktree 注册信息。
- 未启动服务/后台任务，未操作训练进程、真实缓存、数据库、模型或 GPU，未安装依赖。
- 未暂存、提交、推送、合并，未同步回原目录。

## 审计与修改

| 文件 | 改动和原因 |
| --- | --- |
| `requirements.txt` | 添加 `django-redis>=6.0,<7.0`，仅声明依赖；6.0 支持 Django 4.2+，与仓库声明的 Django 5.2 兼容。 |
| `myapp/utils/cache_keys.py`（新增） | 纯标准库 Key 构造；注入缓存对象管理推荐版本和用户级失效，不访问全局缓存。 |
| `myapp/agent/movie_agent.py` | 意图使用 SHA-256；移除用户级固定 Session 回退。 |
| `myapp/agent/memory.py` | 有效 Session 和用户身份共同构造摘要 Key；无有效标识只用实例内槽位；两种输入共用持久化路径。 |
| `myapp/agent_views.py` | 获取/创建真实 Django Session Key，Session 故障向上传播。 |
| `myapp/services/agent_chat_service.py` | 移除默认 `default` Session；显式会话仍沿用原来的 `user_ID_SESSION` 身份规则。 |
| `myapp/agent/react_agent.py`、`workflow_agent.py` | 移除秒级时间戳 Session 回退，避免同秒实例共享缓存；显式 ID 不变。 |
| `myapp/views.py` | KG Key 覆盖参数；混合推荐 Key 覆盖 top_k；现有 dislike/undislike 失效路径通过 on_commit 覆盖所有 top_k。 |
| `tests/test_redis_cache_p0_isolated.py`（新增） | 独立 unittest，以 AST、FakeCache、Mock 验证生产缓存函数。 |

依赖依据：[django-redis 6.0.0 官方 PyPI 页面](https://pypi.org/project/django-redis/6.0.0/)。

原 `update_slots()` 字典分支的输入判断、合并循环、忽略 None/空字符串和返回槽位规则完整保留；
将仅赋值 `_slots` 改为 `_save_slots()`，补上持久化。AST 复核确认合并循环一致。

## Key、TTL 与存储内容

下表为逻辑 Key；Django 的 `KEY_PREFIX=movieagent` 及后端版本包装保持原配置。

| 缓存 | 修改前 | 修改后 | TTL / 内容 |
| --- | --- | --- | --- |
| 意图 | `intent:{hash(trimmed_input)}` | `intent:v2:{SHA256(trimmed_input)}` | 600 秒；原解析字典，不含明文输入 Key。 |
| 会话 | `agent_memory_{session_id or default}` | `agent_memory:v2:{SHA256(JSON([user_id, session_id]))}` | 3600 秒；原槽位字典。无 ID、空 ID、default 或旧用户级回退仅用实例内记忆。 |
| KG 子图 | `kg_subgraph_v3_{topic_md5}_{history_repr_md5[:8]}` | `kg_subgraph:v4:{topic_sha256}:{history_sha256}:{max_triples}` | 1800 秒；原三元组文本。 |
| 混合推荐 | `hybrid_rec_{user_id}` | `hybrid_rec:v2:{user_id}:{generation}:top_k:{top_k}` | 1800 秒；原 `(Movie 列表, profile_text, source_stats)`，仍包含 ORM 对象。 |
| 推荐版本（新增） | 无 | `hybrid_rec:v2:{user_id}:generation` | 1800 秒；UUID 字符串。 |

KG 历史按每个 ID 的 JSON 表示去重排序，再用规范 JSON 摘要：列表/元组、顺序和重复不影响 Key；
保留 int/str 类型差异，避免 Neo4j 类型语义不同时错误命中。未改 Cypher、传入查询的历史或查询结果排序。

推荐版本在首次未命中时用 add 选择并发赢家，读取实际版本；反馈成功后用 on_commit 轮换。
旧版本的所有 top_k 条目不可再命中，仍按原 TTL 自然过期；只删除同用户旧版 hybrid 和 cb Key。
没有扫描、全局 clear、分布式锁。旧版本请求在失效后完成写入，也不会让旧缓存重新可达。
版本自身到期/被淘汰会产生新随机值，避免默认版本复用；这可能导致推荐条目实际满 TTL 前提前未命中。

## 安全测试与静态验证

执行前已审计：`views.py` 顶层导入 torch/deepctr_torch/RAG/model 模块；
`myapp.agent.__init__` 导入完整 Agent 链。因此禁止正常导入上述模块，也不运行 Django setup。

测试仅通过文件读取和 AST 执行选定的定义：MemoryManager、槽位常量、LLMIntentParser、
会话帮助函数、AgentChatService 类及缓存相关视图函数。类方法中的模型/服务路径不被调用。
所有 ORM、Neo4j、LLM、资源加载和事务回调均使用 Fake/Mock。
Key 模块通过文件路径载入，只有 hashlib/json/uuid 标准库依赖；未经过包初始化。
测试进程另设禁止模型模块导入的 guard，并使用 `-S` 禁用 site 启动初始化。
跨进程稳定性用前台等待的两个标准库 Python 子进程、不同 PYTHONHASHSEED 验证。

实际命令（在新 Worktree 前台执行）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
python3 -S tests/test_redis_cache_p0_isolated.py
```

最终结果：14 项全部通过。覆盖稳定 Key、不同用户/匿名 Session 隔离、缺失 ID 的本地多轮记忆、
字典输入合并和持久化、真实 Session 创建及故障传播、KG 历史和 max_triples 参数、推荐缓存命中/未命中
结构排序评分、top_k/force_refresh、反馈提交后失效、数据库写入失败不登记失效、版本淘汰和旧请求写入、
意图缓存异常的现有降级行为。

首轮 13 项中一项夹具断言失败：原槽位规则对“诺兰导演”先匹配中文名“诺兰”，并非英文名。
夹具改用无歧义 `nolan`，未修改槽位提取算法；增加旧版本延迟写入用例后重新运行并通过。

静态验证：9 个修改/新增 Python 文件 AST 解析和内存 compile 成功；`git diff --check` 通过。
AST 比较确认 vector/content/model recall 和 RRF 函数未变，全部 KG Cypher 字符串未变。

未执行：完整 Django 测试、真实 Redis/MySQL/Neo4j 集成、HTTP 服务和模型推理测试。
原因是零干扰约束及真实模块导入链风险。当前 tests 验证函数逻辑和 on_commit 注册/执行时机，
不声称验证了真实 Django 事务、Redis 并发或端到端接口。

## 剩余 P1 与兼容性风险

- ORM 对象缓存和序列化仍保持原状，轻量 ID/分数结构尚未实施。
- 通用 Redis 异常降级、日志和命中率指标尚未实施；混合推荐、KG、记忆和失效回调仍可能受缓存异常影响。
  Session 故障不被吞掉，未设置全局 IGNORE_EXCEPTIONS。
- 画像编辑、记忆写画像、其他反馈和偏好更新路径的全面失效整理尚未实施；此次只同步原有两处推荐失效路径。
- 版本化 Key 会导致首次冷缓存；旧会话槽位不迁移，以免延续错误共享。
- 未提供有效 Session ID 的调用方只保留实例内记忆，跨请求连续对话必须传入真实 ID。
  已审计在线聊天调用方，均通过统一 Session 帮助函数；rerank 的旧用户级槽位读取安全跳过，未修改算法文件。
- Session 创建需要现有后端可用；依赖声明未安装，实际环境兼容性和完整依赖解析尚未验证。
- 最终合并必须保留原目录 settings/训练脚本修改；本次未进行合并。

## 差异统计

实际 `git diff --stat`（不包含未跟踪新增文件）：

```text
 myapp/agent/memory.py                | 28 +++++++++++++++++++++++-----
 myapp/agent/movie_agent.py           |  5 +++--
 myapp/agent/react_agent.py           |  2 +-
 myapp/agent/workflow_agent.py        |  2 +-
 myapp/agent_views.py                 |  8 ++++++--
 myapp/services/agent_chat_service.py |  7 ++++---
 myapp/views.py                       | 29 +++++++++++++++--------------
 requirements.txt                    |  1 +
 8 files changed, 54 insertions(+), 28 deletions(-)
```

新增未跟踪文件：Key 帮助模块、隔离单元测试和本报告。未用 git add 制造统计。
