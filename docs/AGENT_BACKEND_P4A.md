# MovieAgent P4-A：Agent API 与持久化可靠性

日期：2026-10-14（Asia/Singapore）。Worktree：MovieAgent-agent-backend；分支：refactor/agent-backend。
基线：e33fdd4f63f6f50778e50d13bbed18433b02b15f。

> P4-A.1 当前结论：指定的 SSE EOF 成功误判阻断已修复，30 项 JavaScript 隔离测试与原 237 项 Python 隔离测试通过。详见第 12 节。以下 P4-A 初始审计结论保留为历史记录；其中 EOF 风险以第 12 节状态为准。

结论：后端代码级具备有条件人工验收条件。新增 42 项隔离测试，原 195 项保留，累计 237 项通过。
不代表真实 Django/MySQL/Redis 或模型推理集成验证通过；前端 SSE EOF 展示和冻结 Agent 的旧写库尝试仍有遗留风险。
本轮未暂存、提交、推送、合并或创建 PR，没有进入 P4-B/C/D。

## 1. 隔离与执行记录

从 P3 基线创建独立 Worktree 和 refactor/agent-backend 分支；目标路径/分支创建前均不存在。
创建前磁盘可用约 93 GiB，基线跟踪文件合计约 9.5 MB，没有大于 2 MB 的文件或常见模型权重后缀文件。
只检出 Git 基线，不迁移其他 Worktree 的未跟踪训练数据、模型、检查点或实验产物。

按本轮明确要求，用 git --no-optional-locks status --short 只读检查所有既有 Worktree：
P0/P2/P3 三个 Redis Worktree 干净；原始训练 Worktree 有 settings、memory、实验脚本三处跟踪修改及大量未跟踪文件；
另两个研究 Worktree 也有研究修改/未跟踪文件。没有读取、覆盖、暂存其训练文件或切换分支。
只检查共享 Git 元数据中的钩子：没有活动钩子，无自定义 hooksPath。
worktree add 仅对当前 Git 调用关闭自动维护，不禁用或绕过钩子。

语义搜索使用 autoUpdate=false 返回 INDEX_MISSING，随后使用 rg 精确定位；没有建立索引或生成 Embedding。
所有编辑仅在新 Worktree。没有 GPU/CUDA、训练进程、模型、Docker 或系统服务操作；没有 nvidia-smi/进程控制。
没有 Django.setup/manage.py、Ollama 探测、真实 agent.run、服务启动、Redis/MySQL/Neo4j 连接、迁移或依赖安装。
运行测试仅使用标准库、AST、FakeDB、Mock，并在当前终端前台逐个执行，限制 CPU 线程且禁用 GPU 可见性。

## 2. 修改前审计与源码证据

以下原始位置按 P3 基线记录，当前行号会随修改移动：

| 原位置 | 发现 |
|---|---|
| agent_views.agent_api_view（约 124 行） | 空输入/拦截返回 200；一般异常也返回 200，error 为 str(e)，可能泄露内部信息 |
| AgentChatService.process_chat（约 91 行） | 推理、过滤、详情/展示后分别调用两种持久化方法，没有共同事务 |
| _save_history（约 315 行） | 两次独立 create；第二次失败时第一条可能已经保存；捕获异常后只记日志 |
| _save_trace（约 324 行） | 第三次独立 create；失败被吞掉，上层仍可返回成功 |
| stream_chat（约 150 行） | 先输出步骤/答案块，后复用两种持久化方法；吞错后可能发出虚假的 done |
| agent_stream_view（约 185 行） | P0.1 已在生成器前初始化 Session；失败事件有 type/content |
| ChatHistory / AgentTrace 模型 | 同属 myapp，没有自定义写入别名；业务字段分别定义于 models.py / models_upgrade.py |
| movie/settings.py | 只有 default MySQL，未发现 DATABASE_ROUTERS 或 ATOMIC_REQUESTS 开启 |
| CommittedCacheFailureMiddleware | 只处理 RecommendationCacheInvalidationError/partial_success，不负责 Agent 持久化 |
| myapp/urls.py | chat/api/ 与 agent/api/ 共用 JSON View；agent/stream/ 对应 SSE View |
| templates/chat.html / agent_chat.html | JSON 依赖 response 等字段；SSE 依赖 type/content/meta，error 不靠 HTTP 状态表达 |

这些是源码静态结论，未读取真实数据库表引擎、路由运行状态或连接配置。
若部署实际采用不同别名、非事务表或额外路由，不能以本报告声称跨数据库/非事务表原子性。
运行时协调器核对两个模型写入别名，发现不同即拒绝写入，不尝试分布式事务。

### 2.1 额外发现、未修改的冻结代码

MovieAgent.run（约 2585 行）另有一次 ChatHistory.objects.create，使用 session_id/user_message/
agent_response/intent/latency_ms，并捕获异常直接 pass。当前 models.ChatHistory 的字段是 user/role/message/timestamp，
上述额外字段不匹配。按 Django 模型参数校验推断，该旧尝试应在构造模型时失败；本轮没有实际执行来验证。
它不属于本轮 Service 三条写入的事务，不声称所有 Agent 内部数据库/Redis 副作用都被纳入事务。
未修改 myapp/agent/movie_agent.py，因为本轮明确禁止修改 Agent 算法文件；该旧逻辑需单独审计授权。
若部署模型与仓库字段不同，使该旧尝试真的写入，则会出现额外事务外记录，本轮不能保证其回滚。

## 3. 修改文件与职责

| 文件 | 状态 | 职责 |
|---|---|---|
| myapp/agent_views.py | 修改 | JSON HTTP 状态/安全错误结构、服务端 UUID、阶段日志、SSE error 兼容 |
| myapp/services/agent_chat_service.py | 修改 | 单次执行包装、同别名短事务、异常链、JSON/SSE 共用协调器 |
| myapp/services/agent_contracts.py | 新增 | 标准库请求日志、异常类型、安全错误映射；无 Django/模型初始化 |
| tests/test_redis_cache_p0_isolated.py | 夹具扩展 | 按文件加载实际 contracts，验证导出/绑定，再向 AST 桩提供依赖 |
| tests/test_agent_backend_p4a_isolated.py | 新增 | FakeDB 提交/回滚、API/SSE、日志、符号绑定和冻结范围验证 |
| docs/AGENT_BACKEND_P4A.md | 新增 | 本报告 |

原 P0 14 项测试函数 AST 完整保留；没有修改其他八套 Redis 测试或删除/跳过断言。
夹具扩展没有替生产代码补缺失 import：新增测试先检查生产符号绑定，并以删除真实 import 的负向例子验证检查器。
DEFAULT_DB_ALIAS 的 default 只作为 Django API 桩，生产代码仍有真实 django.db 导入且接受缺失导入负向检查。

未修改 models、settings、urls、middleware、前端、Prompt、Ollama 配置、推荐/检索算法、P0～P3 缓存实现或 P2 Metrics。
没有 Migration、新表、依赖或后台架构。

## 4. 持久化事务边界

JSON 路径：

```text
View 生成 UUID / request_received / 输入校验 / Session
  -> Service agent_execution_started
  -> _get_agent + agent.run（无数据库事务）
  -> agent_execution_finished
  -> 排除过滤、电影详情查询、ReAct 展示（均不在新事务中）
  -> persistence_started
  -> 解析 ChatHistory/AgentTrace 写入别名并检查一致性/外层事务
  -> atomic(using=同一别名)
       ChatHistory(user)
       ChatHistory(ai)
       AgentTrace(原业务字段)
     退出 atomic，提交或失败
  -> persistence_succeeded(trace_id=主键) / persistence_failed
  -> XAI 展示 / JSON 序列化 / request_finished
```

_save_history / _save_trace 的异常不再被各自吞掉，只由协调器在 atomic 外捕获并包装：
raise AgentPersistenceError(...) from exc，保留原始异常链。
三条写入通过 objects.using(alias) 显式固定同一个连接；不依赖各次 create 单独重新路由。
事务内部没有本轮代码发起的 LLM、HTTP、Redis、生成器 yield 或重试。
模型导入/别名检查发生在新 atomic 之前；辅助方法内的模型 import 是相同已加载模块。

_execute_agent 在执行前拒绝 default 连接已有事务，防止当前单库配置下 ATOMIC_REQUESTS/外层 atomic 包住推理。
_persist_result 也拒绝目标连接已有外层事务，以免退出内层 savepoint 后误称已经提交。
本轮不支持由外层统一提交的 Agent 请求；此类调用需调整调用边界，不能静默接入。
这些防护基于当前单 default 配置；未来多库/定制路由/Manager 必须重新审查，不能把它当多库协调器。

### 4.1 失败时的可见状态与重试策略

| 场景 | 支持事务的同库预期 | 返回/重试 |
|---|---|---|
| 三次写入及提交成功 | 新增两条 history、一条 trace | JSON 200 / SSE done |
| 第一条 history 失败 | 三条均不新增 | persistence error，无重试 |
| 第二条 history 失败 | 第一条回滚，三条均不新增 | 同上 |
| AgentTrace 失败 | 两条 history 回滚，三条均不新增 | 同上 |
| 不同写入别名 | 新 atomic/写入前拒绝 | 同上 |
| 执行前已有 default 事务 | 不调用 Agent、不写入 | execution error |
| 持久化前已有目标外层事务 | 不开始三条写入 | persistence error |
| 提交失败且未提交 | 可能没有新增记录 | persistence not confirmed，不自动重试 |
| 提交已发生但确认丢失 | 可能已经完整存在三条记录 | 同样 not confirmed，不承诺已回滚 |
| 提交后展示/JSON 构造失败 | 三条记录可能已提交 | INTERNAL_ERROR，不自动重放 |

FakeDB 测试分别模拟未提交和已提交后确认丢失；后者明确断言三条记录已可见，且 Agent 只执行一次。
数据库写入异常、事务回滚与提交确认不确定是不同语义，不能把任意异常都解释成已回滚。
真实回滚依赖 MySQL 表引擎（例如 InnoDB）及数据库配置；未连接检查，因此报告只有预期和 Fake 合约验证。

本轮没有请求幂等表、唯一请求约束或事务重试。request_id 仅用于追踪，不是可重放的幂等键。
Agent 在事务前可能已更新记忆缓存或产生其他外部副作用；数据库回滚不撤销这些副作用。
禁止自动重新运行 Agent 或整次事务，以免未知提交状态下重复记录/副作用。客户端人工重发仍可能重复。

## 5. Agent API 错误契约

成功返回原 response/react_trace/recommended_ids/movie_details/explanations/intent/latency_ms/
need_clarification/clarification_options/trace_steps/user_friendly_trace 字段，View 额外附加 request_id。
Service.process_chat 的业务字典与基线逐字段比较相同；constructor 新增可选 request_id，旧两参数调用兼容。
便捷 process_agent_request 签名不变。

| 条件 | 修改前 | 修改后 |
|---|---|---|
| 正常请求 | 200，原业务字段 | 200，保留字段并新增 request_id |
| 空/无效消息 | 200，问候 response + react_trace=None | 400 INVALID_REQUEST，保留问候字段 |
| 安全检测拦截 | 200，原拦截提示 | 400 UNSAFE_INPUT，保留相同 response/提示 |
| Agent 执行异常 | 200，str(e) | 500 AGENT_EXECUTION_FAILED，无异常文本 |
| 三条记录持久化失败/未确认 | 可能吞错后 200 | 500 AGENT_PERSISTENCE_FAILED |
| 显式 AgentDependencyUnavailable | 原统一异常响应 | 503 AGENT_DEPENDENCY_UNAVAILABLE |
| 推理冒出的 MemoryCacheUnavailable | 原统一异常响应 | 映射上述显式依赖异常，再返回 503 |
| 未分类内部错误/Session 创建错误/展示失败 | 200 或框架错误 | View 内捕获时 500 INTERNAL_ERROR |
| 普通 ConnectionError/TimeoutError | 原统一异常 | 不猜原因；执行中包装为 execution 500，其余 internal 500 |

只有明确类型映射使用 503。没有基于异常字符串、状态码或 Ollama 探测结果推断错误分类。
不新增超时机制，不返回虚构的 504；Agent 原有 fallback 策略未改，内部吞掉的错误不会被本轮凭空识别。
所有本轮错误均 retryable=false，包括 503，避免暗示已经执行的 Agent 或不确定提交可以无风险重放。

示例：

```json
{
  "code": "AGENT_PERSISTENCE_FAILED",
  "message": "回复已生成，但保存未确认完成，请勿自动重试。",
  "request_id": "00000000-0000-4000-8000-000000000001",
  "retryable": false,
  "response": "回复已生成，但保存未确认完成，请勿自动重试。",
  "react_trace": null,
  "error": "AGENT_PERSISTENCE_FAILED"
}
```

error 旧字段改为固定 code，不能再含 SQL、Redis URL、密码、模型路径或 str(exc)。
安全检测仍调用原 sanitize_user_input，原判断算法/Prompt 完全未改。

### 5.1 框架边界与前端影响

login_required、require_POST、CSRF、URL 路由均保留。
UUID/统一 JSON 契约从 View 执行开始，针对进入 View 的请求；认证重定向、装饰器的原生 405、
CSRF 403、SessionMiddleware 在 View 之前的故障仍由 Django 原有机制处理，不保证它们也有 JSON request_id。
本轮没有为全站增加 middleware，也不把会话/认证故障伪装成匿名成功。

原前端对非 2xx 通常进入通用网络错误分支，可能不展示本轮 response/message 的具体提示；
空输入和拦截的 JSON 字段虽保留，HTTP 从 200 改为 400 是明确兼容变化。前端本身会拦截普通空输入，但直接 API 客户端需适配。
本轮遵守不修改前端渲染限制，后续如需友好展示，应单独授权前端读取安全 message/code/request_id。
用户反馈、重试失效接口、缓存 partial_success、CommittedCacheFailureMiddleware 均保持原 AST/文件不变。

## 6. request_id、阶段日志与数据库 Trace 关联

View 每次创建新的 UUID，不采纳客户端 X-Request-ID、Prompt 或 Session Key。
JSON 通过可选 constructor 参数传给 Service；SSE 为保留既有两参数构造契约，在调用 stream_chat 前赋值 service.request_id。
Service 首次使用时规范化 UUID，未传/非法值时生成新 UUID；直接旧调用保持兼容。
同一 View/Service 请求所有阶段用同一个 ID，request.request_id 供当前请求内部排查。

新增日志使用 movie_agent.request，属于既有 movie_agent 层级，不改全局配置/P2 Metrics。
固定阶段：request_received、agent_execution_started、agent_execution_finished、persistence_started、
persistence_succeeded 或 persistence_failed、request_finished。
耗时使用 perf_counter，按阶段或请求起点计算并截为非负；不使用墙钟计算新增耗时。
日志仅记录阶段、UUID、耗时、固定 outcome、成功记录主键；不记录新增原始 Prompt/回答、用户名、Session/Token、URL、SQL、异常字符串。
日志处理器失败会被隔离，不影响持久化或返回结果。旧 Agent/检测模块自身已有日志未在本轮全面清理。

持久化成功后记录 request_id + trace_id=AgentTrace.pk，建立可查日志关联；没有把 UUID 塞进 thought/actions 等业务字段。
AgentTrace 模型/Schema 未改，数据库记录本身没有新增 request_id，日志丢失或保留期过后关联可能不可恢复。
这是日志关联而非数据库永久外键；日志应作为内部数据限制访问。
将来如需要数据库持久关联，可独立设计 nullable/indexed request_id 与相应迁移、历史数据策略，再人工批准；本轮不执行。

## 7. SSE 兼容与限制

保留 intent/thought/action/observation/clarification/chunk/done 等正常事件的字段、顺序、分块及已有 sleep 行为；
基线与新 Service 的正常 SSE 事件列表逐项完全相等（Mock 时钟排除实际时间差）。
没有改变 LLM 生成方式、ReAct 流程、连接取消、断线检测或前端代码。

SSE 正常请求仍在构造 StreamingHttpResponse 前初始化 Session，P0.1 三项严格时序断言通过。
Session 创建失败继续在响应构造前向框架传播，不尝试无状态恢复；JSON 对 View 内同类错误返回安全 internal 500。
SSE HTTP 头新增 X-Request-ID；原 Cache-Control、X-Accel-Buffering、正常流 Connection 头保留，既有 middleware 仍按原职责移除 hop-by-hop 头。

持久化发生在步骤/答案块已经输出之后、done 之前。
任一持久化失败抛异常到 View，发已有 type=error/content 事件，附加 code/request_id/retryable=false；不发成功 done。
此时 HTTP 状态可能已经为 200，不能再尝试改成 500；测试明确验证 HTTP 200 + error 而非伪造状态变更。
空输入的单个 done 问候和安全拦截的单个 error 事件体原样保留，保持原 SSE 前端兼容；UUID 在响应头/日志中。
传输完成、Agent 执行成功、数据库持久化确认是三个不同状态。

### 7.1 现有前端的 EOF 展示风险

templates/agent_chat.html 在 reader.read() 的 EOF 分支无条件调用 onStreamDone，
后者可设置“推理完成”状态，即使此前收到了 error；这不是后端发送成功 done。
本轮只保证后端不会在持久化失败后发 done，不能宣称界面一定正确显示保存失败。
按本轮不改前端约束，保留该风险，需后续单独授权 UI 错误状态处理；未擅自实施 P4-B。
生成器未开始或未耗尽时，最终阶段日志也未必已产生；本轮不提供断线观测/取消保证。

## 8. 安全隔离测试与实际执行结果

测试开始前审计所有直接/间接导入：contracts 只用 logging/time/uuid；服务与 View 只提取选定 AST 定义，
transaction/router/models/request/response/Agent 均为明确 Fake。
_get_agent 在每次测试中覆盖为 FakeAgent，不会执行真实 MovieAgent、Ollama HTTP 或模型加载；
旧源码成功路径对照也只提取同一个 Service 类 AST，使用同样 Fake 依赖。
FakeAgent.run 是测试对象方法，不是实际 agent.run。

导入 Guard 在加载夹具前拒绝 Django、redis、torch、transformers、sentence_transformers、numpy、faiss、
tensorflow、MySQLdb、neo4j、requests、ollama、langchain。测试没有 Django.setup/AppConfig.ready。
FakeDB 区分 transaction-local rows 与 committed rows，提交/回滚更新最终可见状态；并模拟提交前失败/提交后确认丢失。
这不证明真实 MySQL 隔离级别、事务引擎或网络丢包行为。

执行命令前缀（每个脚本逐个执行，前台无并发）：

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/FILE.py
```

| 文件 | 测试数 | 结果 |
|---|---:|---|
| test_redis_cache_p0_isolated.py | 14 | OK / 0 |
| test_redis_cache_p01_isolated.py | 18 | OK / 0 |
| test_redis_cache_p1a_isolated.py | 18 | OK / 0 |
| test_redis_cache_p1b_isolated.py | 8 | OK / 0 |
| test_redis_cache_blockers_isolated.py | 23 | OK / 0 |
| test_redis_cache_p2_isolated.py | 24 | OK / 0 |
| test_redis_structures_p3_isolated.py | 41 | OK / 0 |
| test_redis_structures_p31_isolated.py | 26 | OK / 0 |
| test_redis_structures_p32_isolated.py | 23 | OK / 0 |
| test_agent_backend_p4a_isolated.py | 42 | OK / 0 |
| **合计** | **237** | **全部通过** |

新增覆盖：三次写入任一失败、已有记录保留、异常链、Agent 在事务外/只执行一次、同别名约束、外层事务拒绝、
提交不确定、400/500/503 分类、不假设超时、错误内容去敏、UUID 传播/日志非负/处理器失败、
业务字段与基线一致、SSE 正常协议、失败无 done、空/拦截事件原样、框架安全装饰器/反馈 partial_success 不变、
实际生产绑定/缺失 import 负向检查、冻结算法/模型/前端/缓存/配置逐字节不变。

开发时首次 P4-A 测试发现两个测试夹具问题：使用了不存在的失效视图名字、把 Python 自动提供的 __file__ 当缺失导入。
已改为真实 retry_cache_invalidation，并仅豁免 __file__；缺失业务 import 的负向检查保留，不修改生产代码来迎合桩。
所有原 195 项保留，没有删除、降低断言或 skip；P0 仅扩展真实标准库 helper 绑定。

静态检查：所有改动/新增 Python 的 ast.parse；生产完整符号表（仅排除模块加载器提供的 __file__）；
实际辅助导出检查；缺失 import 的负向测试；git diff --check；新增文件 UTF-8/无 NUL/非符号链接/无行尾空白。
未运行 py_compile 或完整 Django 测试套件，不产生字节码、不探测数据库。

## 9. 未完成的真实集成与风险

以下只设计，不在当前机器执行，需独立环境和人工授权：

1. 独立 Django/MySQL，安全测试 AppConfig，不加载模型；确认两个模型/用户都路由到同一个连接。
2. 查询表引擎为可回滚事务引擎，验证三条写入的真实 rollback/commit/隔离和信号行为。
3. 真实提交确认丢失、连接断开/锁超时；核对数据后人工恢复，不自动重放。
4. 实际 Agent 类型/返回对象与生产 Service 配合；保持推理 stub，不执行模型，并审计冻结 Agent 的旧字段写库尝试。
5. SessionMiddleware Cookie、认证重定向、CSRF/405、Session 后端故障、JSON 编码、XAI 展示异常。
6. 实际 WSGI/ASGI/SSE 错误帧、响应头和 EOF 行为，客户端显示是否准确；前端改动需另授权。
7. 部署日志 handler/权限/保留期，request_id 到 trace.pk 排查链与脱敏验证。
8. 与 P0～P3 真实 Redis 故障/缓存回源、partial_success 配合，禁止共享 Redis 全局清理。

本轮不保证 exactly-once、跨库原子性、Redis/数据库分布式提交、Agent 内部副作用回滚、硬超时或自动最终一致性。
拒绝外层事务是明确调用约束；多库、定制 Manager/路由、额外信号和非标准数据库后端要单独复审。
前端非 2xx 文案及 SSE EOF 状态、Agent 旧持久化尝试仍需后续授权，不能把代码级条件验收描述为生产上线验收。

## 10. 面试中的工程设计说明

- AI 推理/外部检索慢且有副作用，留在事务外；持久化只锁定三条必要写入，减少事务持有时间。
- Unit of Work 把一次请求的两条历史与一条 Trace 视作同库提交单元；异常只在边界包装，原始 cause 保留。
- 业务原子性与网络结果确认不同。COMMIT 应答丢失不能断言已回滚，retryable=false 优先避免重放副作用。
- UUID 帮助定位请求，不能替代幂等约束。没有幂等存储时不承诺安全自动重试或 exactly-once。
- API 用具体错误类型映射 HTTP 状态，不靠异常文本猜依赖/超时；客户端只看稳定 code/message。
- SSE HTTP 状态在响应开始后无法更改，因此用既有 error 帧与 done 分开表达持久化结果。
- FakeDB 验证最终可见状态比 assert_called 更强，但不能替代真实 MySQL/InnoDB/路由/网络集成测试。
- 修改只落在后端服务边界；推荐/模型策略、数据库 Schema、缓存一致性与前端保持冻结，遗留风险透明列出。

## 11. Git 与人工验收

本轮完整变更清单为第 3 节六个文件：三个跟踪修改、三个新增文件。
分支 refactor/agent-backend，HEAD 仍是 P3 基线；其他 Worktree 没有被本轮修改。
没有环境文件、密钥、模型权重或真实实验数据加入修改，没有全局配置/依赖变化。
空白、语法、生产符号检查和 237 项 CPU-only 回归通过；具备后端代码级有条件人工验收条件。
条件是接受第 5/7/9 节明确的框架、前端、冻结旧逻辑与真实集成边界，不宣称生产强一致性。
本轮不暂存、不提交、不推送、不合并、不创建 PR。完成后暂停，等待人工审查。


## 12. P4-A.1：SSE EOF 成功误判修复

日期：2026-10-14（Asia/Singapore）。只处理前端终态，不实施 P4-B。

### 12.1 原因与源码位置

P3 基线 templates/agent_chat.html 的 sendMessage 中，readChunk 在 result.done 分支无条件调用 onStreamDone；
handleStreamChunk 的 error 分支只追加气泡，不记录失败终态。
onStreamDone 会渲染 Trace/推荐卡片并设置“推理完成”，因此后端即使在持久化失败时只发 error 而不发 done，
流 EOF 仍可被前端误解为成功。第 7.1 节记录的 P1 阻断在本轮修复，历史证据保留。

本轮不改变后端 Service 的事务、异常、SSE 事件生产方式；不改变 Redis、LLM、Prompt、ReAct 或推荐算法。
在接收端区分传输 EOF、收到业务完成标记、已处理终态，不把它们混为一谈。

### 12.2 三个局部状态与终态转换

sendMessage 每次请求单独初始化 receivedDone=false、receivedError=false、terminalHandled=false，
不使用全局跨请求状态。fetch 返回 Promise，便于隔离测试等待完整链路，不改变现有 UI 调用者行为。

| 输入/阶段 | 修复前 | 修复后 |
|---|---|---|
| 正常 chunk + done + EOF | EOF 执行成功展示 | 记录 receivedDone，正常 EOF 且无错误时才调用原成功展示一次 |
| error + EOF | 错误气泡后仍可能显示推理完成 | receivedError=true，失败终态处理一次；EOF 不再触发成功或重复错误 |
| chunk + error + EOF | 回答存在，但最终误显示成功 | 已显示回答保留；明确显示本次请求未成功完成；不渲染最终成功元数据 |
| chunk/metadata + EOF 无 done | 当作完成 | 固定提示“响应不完整”，不调用 onStreamDone |
| 空流 EOF | 当作完成并显示默认回答 | 不完整响应失败 |
| done 后又 error | 仍可能成功 | EOF 前 error 优先，失败终态；没有成功完成展示 |
| done 后 reader 异常 | 网络错误分支与成功逻辑分散 | 保守判为失败，不用先收到 done 推导传输已正常完成 |
| 重复 error / error 后异常 | 可重复提示 | terminalHandled 防止重复处理 |
| 重复 done | 后一个可覆盖元数据 | 第一个 done 保留，最终展示一次 |
| 非 2xx | 直接调用 getReader | 先检查 resp.ok，不打开普通成功流解析，也不读取错误响应体 |

成功展示只在正常 EOF、receivedDone=true、receivedError=false、terminalHandled=false 时发生。
收到 done 时不立即渲染成功，避免后续 error/读取中断仍被成功状态覆盖。
成功 renderer 正常返回后才标记 terminalHandled；如果 renderer 抛异常，仍能转为一次失败展示。
失败后忽略后续普通消息，防止迟到的 thought/chunk/metadata/done 再次更新状态或增加成功卡片。
错误帧后的网络异常也不会再创建第二个错误气泡。

failStream 统一显示固定失败文本/状态。已展示的回答 DOM 不删除，也不会在失败时设置成功 data-response 属性。
AGENT_PERSISTENCE_FAILED 仅选择固定的“回答已生成，但保存未确认完成”提示，不把 chunk.content/原始异常字符串插入界面。
HTTP 错误体、fetch/reader 异常文本均不展示，避免 SQL、URL、Token、模型路径或内部堆栈泄露。
没有自动重试请求，提示先核对会话记录、勿自动重复提交。

### 12.3 保持原有渲染与前后端契约

thought/action/observation/reflection 和 chunk 分支保持原样，逐块文字累加/formatAiText/滚动逻辑不变。
onStreamDone 的正文及模板外部函数均与 P3 基线逐字节一致：
原 Trace、推荐卡片、追问选项、无 chunk 时的回答渲染仍由同一个 renderer 负责。
metadata、done 的 meta/data 原有字段与回退结构保留；done 后不再接受后来的普通元数据覆盖。
没有修改后端事件类型/字段，没有修改逐字符展示、Debug AJAX、打字机效果或推荐卡片函数。

沿用现有按行解包和 TextDecoder 流式解码；不重写 SSE 解析器。
新增对非法 JSON/非对象事件的固定失败处理，不能静默吞掉坏帧再宣称成功。
后端 _sse_data 输出完整 data 行与双换行；缺少可识别完整 done 行的 EOF 不作为成功。
fetch 响应没有可读 body 时同样失败，不解析成成功响应。

后端当前只有三条记录持久化确认后才产生 done；持久化失败产生 error，可能发生在答案 chunk 之后。
前端因此将 done 作为业务完成确认标记，EOF 仅作传输结束；本轮不独立查询数据库来验证持久化。
两侧协议在本轮范围内一致。提交确认不确定仍可能是“数据库已有记录 + error”，前端不会自动重发。

没有 reader.cancel/AbortController、重连、后台任务取消、WebSocket、异步架构或新定时器。
出错后仍可继续读到 EOF，但终态/渲染不会被后续帧覆盖；不会据此承诺连接生命周期或取消保障。

### 12.4 JavaScript 隔离测试（30 项）

新增 tests/test_agent_sse_terminal_isolated.cjs，只使用 Node 标准库 assert/strict、fs、path、vm。
从实际模板提取完整 sendMessage 函数，在 VM 中运行；fetch、ReadableStream reader、DOM 和渲染依赖全部为显式桩。
不会求值整个 Django 模板、启动浏览器/Django、发网络请求或访问模型；没有 npm 依赖安装、子进程或后台服务。
测试串行运行，合成 reader 在 EOF 后有调用上限，避免失败实现产生无限轮询。

覆盖结果：

- 正常 chunk+done+EOF：正文累加保留、只成功一次；Trace、推荐卡片、追问收到原数据。
- error+EOF、chunk+持久化 error+EOF：失败一次，保留回答，无虚假完成/成功元数据。
- chunk+EOF、metadata+EOF、空流：缺少 done 一律不完整失败。
- HTTP 400/401/403/500/503：均不调用 getReader，不访问错误响应体，不进入普通流解析。
- fetch 拒绝、reader 首次/中途拒绝、done 后读失败：安全失败，不泄露异常、不自动重试。
- 重复 error、error 后读取异常、error 后 done、done 后 error：无重复错误气泡或成功覆盖。
- 重复 done：不重复卡片/Trace，不覆盖第一份完成元数据。
- 迟到 thought/chunk/metadata：不能覆盖失败终态。
- metadata 回退、旧 React Trace 回退、无 chunk 的 done：保持原成功展示调用与数据。
- UTF-8 和事件跨包：正常解码/完成。
- 非法 JSON、null 事件、无 body、成功 renderer 异常：不误显示完成。
- 每次调用局部重建终态状态；前一请求失败不污染下一请求成功。

以上是真实客户端控制函数在 Fake 环境中的行为验证；没有真实浏览器 DOM、CSS、fetch 网络栈或推荐卡片视觉验收。
不能将“调用原 renderer 并传递相同数据”描述成已经验证浏览器中所有卡片显示/交互。

### 12.5 原 237 项回归及断言适配

原 237 项 Python 隔离测试在导入链审计后逐个重跑，全部通过。
原 P4-A 的 test_frozen_algorithms_cache_models_frontend_and_metrics_unchanged 曾冻结整个 agent_chat.html，
与本轮人工明确授权修改模板相冲突，已作必要的精确适配：
保留全部路径和测试方法，模板仍断言 sendMessage 之前、函数之后、Debug 分支、完整 onStreamDone、
推理步骤分支、chunk 渲染分支与基线完全一致；其余文件仍逐字节冻结。
不是删除失败断言或放宽算法/事务冻结范围，也没有 skip 用例。其余原测试函数不变。

安全执行命令：

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_THREADPOOL_SIZE=1 node --check tests/test_agent_sse_terminal_isolated.cjs
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_THREADPOOL_SIZE=1 node tests/test_agent_sse_terminal_isolated.cjs
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/FILE.py
```

| 测试脚本 | 数量 | 结果 |
|---|---:|---|
| test_redis_cache_p0_isolated.py | 14 | OK / 0 |
| test_redis_cache_p01_isolated.py | 18 | OK / 0 |
| test_redis_cache_p1a_isolated.py | 18 | OK / 0 |
| test_redis_cache_p1b_isolated.py | 8 | OK / 0 |
| test_redis_cache_blockers_isolated.py | 23 | OK / 0 |
| test_redis_cache_p2_isolated.py | 24 | OK / 0 |
| test_redis_structures_p3_isolated.py | 41 | OK / 0 |
| test_redis_structures_p31_isolated.py | 26 | OK / 0 |
| test_redis_structures_p32_isolated.py | 23 | OK / 0 |
| test_agent_backend_p4a_isolated.py | 42 | OK / 0 |
| test_agent_sse_terminal_isolated.cjs | 30 | 30 passed / 0 failed |
| 合计（Python + JS） | **267** | **全部通过** |

Node 脚本语法检查、提取的实际 sendMessage 编译、Python AST/生产符号检查和 git diff --check 均通过。
既有 Python import guard 保留；新增 JS VM 不提供真实网络、process/require 或浏览器 API，只注入合成依赖。

### 12.6 变更范围、剩余验证与结论

本轮实质变更仅四份文件：

1. templates/agent_chat.html：SSE 终态标记、HTTP 检查、统一安全失败提示。
2. tests/test_agent_sse_terminal_isolated.cjs：30 项实际控制函数的隔离测试。
3. tests/test_agent_backend_p4a_isolated.py：只适配一个模板冻结测试，增加精确的未授权范围冻结断言。
4. docs/AGENT_BACKEND_P4A.md：保留 P4-A 历史、更新当前结论并追加本节。

MySQL 事务设计、后端 API/SSE 生产逻辑、request_id、Redis P0～P3、LLM/Prompt/ReAct/推荐算法未修改。
原训练目录/其他 Worktree/GPU/CUDA 进程/模型/检查点未操作；没有服务、依赖或全局环境变化。
分支 refactor/agent-backend；HEAD 仍为 e33fdd4f63f6f50778e50d13bbed18433b02b15f，暂存区为空。

指定的 P1“error/不完整 EOF 显示推理完成”阻断已在代码与隔离测试范围内消除。
本轮范围内未发现新的 P0/P1 阻断，具备有条件人工验收条件；不宣称全项目或生产端到端无缺陷。
真实浏览器与代理断流/网络分包、实际页面状态/Trace/卡片视觉、认证重定向、真实 WSGI/ASGI SSE 行为尚未执行。
真实 Django/MySQL/InnoDB 三写回滚、提交确认丢失、Session/路由/信号和 Redis 配合仍未验证，第 9 节待办保留。
冻结 Agent 的旧字段写库尝试及框架提前拒绝请求的 request_id 边界仍未修改；不扩大本轮授权范围。
本轮不暂存、不提交、不推送、不合并，不进入其他 P4-B 功能。完成后暂停等待人工验收。
