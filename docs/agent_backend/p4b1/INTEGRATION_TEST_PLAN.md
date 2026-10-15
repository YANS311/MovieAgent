# 下一阶段独立 ASGI 集成计划（本轮未执行）

必须人工授权，在独立测试环境进行；不加载真实 Agent/模型，不接生产/共享数据库或 Redis，不修改当前训练环境。

## 前置条件

1. 锁定并记录 Python、Django 5.2.x、asgiref、Uvicorn 精确版本；建立无 apps.ready 模型预热的最小测试应用。
2. 注入有限 Fake Agent；事务层先用 Fake，再使用独立一次性数据库。生产路由、鉴权、Session、中间件最终单独覆盖；不以最小应用代替完整部署验证。
3. 固定线程池和准入、队列/事件字节上限、超时、关闭所有权及超限响应；不得每请求新建池。真实连接清理设计先过审。
4. 同输入 A 同步基线 / C 异步桥配对，保留原协议与短事务；禁止更改业务排序、模型或缓存语义。

## 验证矩阵

| 场景 | 必须观察 | 准入标准 |
|---|---|---|
| 真 StreamingHttpResponse 最小构造 | is_async；同步与异步 iterator 警告/路径 | C is_async=True，无完整 list 回退 |
| Uvicorn + Fake gate | 服务端 yield/send、客户端收到 intent、Agent 返回时间 | 客户端首事件早于业务 gate 释放；不能只测首个 yield |
| 浏览器 Fetch/ReadableStream | HTTP 非2xx、chunk、error/done、异常 EOF | 原30项前端契约在真实浏览器仍成立 |
| AbortController / TCP 断开 | receive disconnect、协程取消、源 close、worker 结束 | 分别记录检测/取消/同步工作完成；不宣称即时杀线程 |
| 可选代理 | 缓冲启闭、压缩、超时、事件间隔 | 明确哪些配置造成缓冲；无虚假首事件结论 |
| 慢客户端/满队列 | 队列深度、线程和任务数、RSS | 达上限背压/拒绝；取消不因终态队列满死锁 |
| 有界多连接 | 固定 W 的准入/拒绝，已关闭但仍运行 worker | 不提前复用残留工作 lease、不无界排队 |
| 持久化前/中/后断连 | 实际 SQL 提交、异常、终态和重试策略 | 三写同一事务；不以断连声称回滚 |
| commit 确认丢失、清理异常 | 服务端状态、客户端安全失败、真实行数 | 不自动重跑 Agent；无假 done、无异常详情泄露 |
| 中间件/Session/连接复用 | Cookie、鉴权惰性求值、线程ID与连接关闭 | 不跨线程使用 connection/cursor；不改变 Session 身份语义 |
| 关闭/重启 | 请求前/中/后关闭、pool 残留、ASGI 生命周期 | 无非有限 Fake；残留工作准确报告，不强杀 |

## 测量口径

TTFE 从客户端发起请求到解析首个完整 SSE；事件间隔按同一连接客户端时间戳；端到端耗时、断连检测延迟各自独立分布。另测 CPU/RSS、活跃线程、任务/连接峰值、队列字节数、准入拒绝数。固定种子、有限预热/样本/并发与总时长，记录 P50/P95/P99 和样本数；不相加各阶段 P95，不将 Fake 延迟宣传为生产性能。

Uvicorn 的 [HTTP h11 实现](https://github.com/encode/uvicorn/blob/master/uvicorn/protocols/http/h11_impl.py) 会维护断连状态；send 返回不是浏览器接收确认。测试必须记录实际服务器版本、receive 与 send 时间线，而不是把源码推断当实测。

只有该矩阵中流增量、取消、线程/连接清理、事务和前端终态都符合预先定义的标准后，才可评估生产 adapter 变更。本轮没有启动这套环境。
