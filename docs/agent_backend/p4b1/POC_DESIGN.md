# 独立 PoC 设计

只新增 experiments/asgi_streaming 和隔离测试，无生产导入方。

数据流：有限 FakeScenario.source(stop) → 一个稳定 executor worker → 有界 queue.Queue 的编码 bytes → BridgeStream.__anext__ → 可选 encoded_events 异步 generator。最后一步仅是本地交接，不是 ASGI Send 或浏览器接收。

## 所有权与状态

- BridgePool 显式创建，W=1 默认、最大2；流 lease 最大 W；第一次拉取才提交一个完整 source 工作，不为每事件建立任务。
- 状态：ADMITTED → RUNNING → PRODUCER_FINISHED → CONSUMER_EOF/CLOSED → RELEASED。取消可从任一状态发 STOP_REQUESTED；只有工作 Future 已结束且 consumer 已结束才释放 lease。已完成但未读完的流也继续占位。
- factory、next、源 close、FakeAgent 和 FakePersistence 在同一 worker。contextvars 显式复制一次。没有传递真实 Django 连接、request、惰性 ORM 对象。
- close() 幂等、非阻塞，只发 stop、丢弃待交付终态、排空队列。未开始的流不调用 factory。源 close 始终由所属 worker 执行。
- aclose(grace) 有界等候并报告 producer_finished；pool.aclose 先停止准入，再停止流，期限内工作结束才 join。超期返回 drained=False，并不杀线程；有限 Fake 全部最终 drain/join。非有限真实调用可能阻止 Python 退出，不允许据此直接接生产。

## 容量、背压与通知

队列容量默认2（1..8），编码事件上限4096字节（512..65536），已排队 bytes ≤ capacity × max_event_bytes。另有 producer 当前 blob、一个终态槽、consumer 当前 blob、源及序列化临时内存，不声称整个进程内存有此上界。

满队列 put(timeout=.01) 反复检查 stop/合作期限；没有无界提交。队列终态独立于数据队列，避免满队列时无法放入错误/退出 sentinel。loop 通知经 call_soon_threadsafe；worker 不直接操作 asyncio.Queue/Event。

当前沙箱的独立 Event 唤醒复现未及时被线程通知唤醒。因此 __anext__ 同时以最多5ms的 wait_for 定时片检查队列/完成状态；不是忙等。合作期限默认2秒、范围 .05..5秒，不能中断正在阻塞的 next/Agent/事务；consumer 超时抛出 PollDeadlineExceeded 并发 stop，不自动重试。

## 错误与终态

数据事件按源顺序流动。done 暂存到单独终态槽；源已正常结束且 close 成功后才交付。error 立即停止继续拉源；其后的 done 不被读取。重复 done、done 后异常、编码/清理错误都改为一个安全 error。BaseException 记录为异常 Future 并由池观察，不伪装正常完成；停止后不再交付终态。

错误仅含固定安全文案、受限 code、随机 request_id、retryable=False；不暴露异常详情。EOF 无 done 仍无成功含义。consumer 的 terminal_delivered 只表示本地交接，client_received_done 始终 None。

`encoded_events` 用 finally 调用 stream.aclose；本地 sender 需要 contextlib.aclosing 以覆盖 send 期间取消。未来 Django 响应需同时支持未开始迭代时的同步 close 注册和异步所有权传播；不能直接把本实验函数塞进生产认为已解决。

## 探针

LifecycleProbe 默认最多256条（上限1024）固定阶段、单调时间及线程标识；不记内容、画像、Session、异常详情。证据 JSON 只保留阶段序列、状态、合成行数和本地首次交接时间，非生产日志方案或性能数据。
