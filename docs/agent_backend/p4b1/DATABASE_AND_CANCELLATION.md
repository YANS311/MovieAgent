# 数据库与取消：保证边界

生产 AgentChatService.stream_chat:295–305 先检查生命周期，再调用 _persist_result，确认成功后才 done。_persist_result:366–386 在同一 alias 自有短事务中写两条 ChatHistory、一条 AgentTrace；拒绝跨库及已有外层事务。本轮没有修改该方法、提交策略或业务字段。

| 取消发生位置 | 本轮 Fake 验证 | 真实业务限制 |
|---|---|---|
| 源第一次执行前 | 未启动源不执行 Agent/保存；gate 前合作检查也能停止 | Django 响应 close 注册/首拉前断连待集成 |
| 同步 Agent 运行中 | 等待协程取消立即传播；worker 继续，gate 释放后合作退出 | 不能杀同步模型请求/线程；工具、Memory 副作用可能已经发生 |
| trace/chunk 回放 | 下一个合作检查停止；不开始保存 | 检查到开始保存之间仍有竞态窗口 |
| 保存前检查点 | 可观察到取消则不调用保存 | 不能保证所有并发断连都赶在事务开始前 |
| 事务中 | Fake 三次写入整体完成后观察 stop，不交付 done | 取消 await 不会让数据库 rollback；连接故障也可能造成结果不确定 |
| 提交后/终态前 | 已提交三条模拟记录保留，不补偿删除 | 客户端可能无 done；request_id 不是持久化幂等键 |
| done 已交接/网络未确认 | client_received_done=None | 不能以服务端交接证明用户收到；禁止盲目重试整个请求 |

FakePersistence 的 fail_at=1/2/3 模拟整体回滚；commit_ack_loss 模拟已提交但确认失败，发送 error 且不重试。它不导入事务包、不连接 MySQL，不能证明驱动/数据库事务行为。

同步工作放 worker 后的连接管理仍未完成：在工作线程内取得并使用连接，不跨线程传原始 connection/cursor，不在 event loop 中同步 ORM；整个 atomic 保持在同一同步函数。线程结束后负责本线程连接生命周期，不能关闭共享模型或其他请求的资源。ASGI 请求关闭信号可能在不同线程发生，必须真实测试；contextvars 复制不等于连接可安全迁移。

生产 Agent 内部另有 memory.update_slots（movie_agent.py:2550）与旧 ChatHistory.objects.create 尝试（:2585）；本轮冻结。协调层事务保证不能推广到这些副作用，更不能宣称取消后没有任何业务写入。

建议未来 adapter 延续“观察到取消后不再启动最终持久化；已开始的短事务自行完成，再判断是否发 done”的现有合作策略，但任何进一步改变提交时机都需单独批准。不拆事务、不自动补偿、不引入重试队列或幂等数据库表。
