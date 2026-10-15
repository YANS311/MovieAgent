# A / B / C 适配方案比较

调用锚点：agent_stream_view → lifecycle.wrap → AgentChatService.stream_chat → _execute_agent → 回放 → _persist_result → done。当前生产实现保持原样。

| 维度 | A：现有同步 iterator | B：async generator + 同步调用适配 | C：完整同步 producer + 有界桥 |
|---|---|---|---|
| 首事件 | Django ASGI 类型适配可能先收集全部 | 正确异步 iterator 可逐事件交付 | 正确异步 iterator 可逐事件交付 |
| 内部推理 | 同步完整 Agent | 仍同步 Agent，必须 offload | 仍同步 Agent；worker 运行完整源 |
| 资源 | 预消费列表、执行线程 | 默认线程池若无准入仍可能积压 | 共享固定池 W + 严格 W 个 lease + 有界队列 |
| 线程亲和 | 随 Django/asgiref 上下文 | 多次 to_thread 不保证同线程 | 单源 factory/next/close 固定在同一 worker |
| 背压 | 完整收集绕开网络侧背压 | 每步 await；须避免无界任务 | 队列满 producer 有界等待、合作停止 |
| 取消 | 本地 close 检查点 | 取消 await 不杀同步函数 | 停止信号 + 检查点；运行中工作仍可残留 |
| 数据库 | 原同步短事务 | 事务应封在一个同步函数 | 事务保留一个 worker 同步边界；仍需连接管理 |
| 侵入/复杂度 | 无新增代码但保留 ASGI 风险 | 表面简单；分步 next、异常、连接处理复杂 | 独立 adapter 复杂度较高，生命周期可显式测试 |
| 测试 | 原隔离测试不能证明网络流 | 需验证 offload/ORM/清理 | 本轮验证有限 Fake 桥；实际 Django 尚未验证 |

## B 的必要约束

`asyncio.to_thread` 复制 contextvars 并使用 loop 默认 executor；不保证多次调用的线程亲和或请求准入。`run_in_executor` 可指定固定池，但默认不替调用方复制上下文。`sync_to_async(thread_sensitive=True)` 按 asgiref 上下文串行/复用线程；False 可并发但不保证 ORM 连接线程安全。不能把它们当成等价的性能开关。见 [Python asyncio 线程接口](https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread)。

分步 offload `next()` 时不能让 StopIteration 穿过 Future；需显式 sentinel。factory、next、close 分散线程也可能破坏源的线程亲和性。异步 generator 内直接调用同步 Agent/ORM/sleep 会阻塞 loop，不能采用。同步事务整体 offload，不能把三次写入拆成独立异步步骤。

## 推荐

**C 用于独立 PoC；生产推广仍为 NO-GO。** 它更接近现有同步源的生命周期和事务边界，能明确测试满队列、取消和终态；必须承担固定池准入、连接释放、请求结束后的残留工作管理成本。池应按进程/loop共享且有显式寿命，不能每请求创建一个池或无限线程。

A 保留作原样基线；B 并非不可行，但当前调用链含完整同步 Agent、多阶段回放及同步事务，分步适配的所有权更难证明。C 不会让同步推理变成 Token Streaming，也没有性能更快的实测结论。线程数与队列上限只是 PoC 实验约束，不是推荐生产容量。
