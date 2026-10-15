# P4-B.1 准入结论

**CONDITIONAL GO：仅接受独立方案 C PoC 和后续集成实验设计。生产异步 SSE Adapter 推广为 NO-GO。**

建议下一独立阶段验证异步桥，而不是继续依赖当前同步 iterator 的 ASGI 预消费路径。正确的异步迭代可以解除框架整流收集问题，但不能把同步 Agent 内部推理变成实时Token Streaming，也不能保证立即停止同步调用。

## 已验证

- 有限Fake的逐事件本地交付、队列上限、严格执行资源准入、合作取消、异常终态互斥与关闭协议。
- producer线程稳定，Fake事务不中途拆分；取消协程后工作仍可残留，最终显式释放gate并join。
- 旧327项全部通过；新45通过/1真实Django层跳过；总372通过、0失败、1跳过。
- 当前Python 3.13.5；Django/asgiref/Uvicorn部署版本未确认。框架语义来自官方源码阅读。

## 仍阻断生产推广的P1

1. 尚无真实StreamingHttpResponse.is_async=True构造、ASGI send/disconnect与浏览器首事件证据。
2. Django包装层关闭传播、未首拉response.close、同步middleware、Session和鉴权惰性对象在线程切换后的行为待验证。
3. 真实ORM连接生命周期未实现；同步Agent/DB无法强制取消，残留池占位和进程退出策略须设计并验证。
4. 当前沙箱线程通知有唤醒限制；PoC的5ms定时回退只有本地正确性证据，没有部署开销/必要性证据。
5. 桥队列有界不等于Agent结果或序列化临时内存有界；多连接容量与拒绝策略没有真实性能数据。

## 未来可能涉及的生产位置（本轮未实施）

- myapp/agent_views.py:203：响应的异步iterator和显式close所有权；Session仍须在构造前完成。
- myapp/services/agent_contracts.py:197：保留现有同步包装器契约，另外设计异步桥接口，不能直接删旧实现。
- myapp/services/agent_chat_service.py:182：复用同步源/检查点；_persist_result:366短事务和算法不可改变。
- 新独立生产adapter及明确的进程/loop池寿命、连接回收；是否需ASGI生命周期包装另行设计。不能在apps.ready启动worker。

不需要也不授权修改Prompt、LLM、ReAct、推荐算法、Redis/Session语义、前端终态或数据库Schema。若提交策略需要改变，单独提案审批。

## Git和安全记录

分支 refactor/agent-sse-lifecycle，HEAD e60eb5a96662f9e448df571e5e0e495bd43ba737。未新建Worktree。本轮只新增14个文件：

- experiments/asgi_streaming/async_bridge_poc.py
- experiments/asgi_streaming/fake_agent.py
- experiments/asgi_streaming/lifecycle_probe.py
- experiments/asgi_streaming/README.md
- experiments/asgi_streaming/poc_evidence.json
- tests/test_asgi_streaming_poc_isolated.py
- docs/agent_backend/p4b1/ASGI_STREAMING_AUDIT.md
- docs/agent_backend/p4b1/ASYNC_ADAPTER_COMPARISON.md
- docs/agent_backend/p4b1/POC_DESIGN.md
- docs/agent_backend/p4b1/POC_TEST_REPORT.md
- docs/agent_backend/p4b1/DATABASE_AND_CANCELLATION.md
- docs/agent_backend/p4b1/INTEGRATION_TEST_PLAN.md
- docs/agent_backend/p4b1/P4B1_GO_NO_GO.md
- docs/agent_backend/p4b1/P4B_FROZEN_SHA256.json

P4B_FROZEN_SHA256保存开始时11个原文件与git diff --binary摘要；最终一致。git diff仍仅原三个生产文件（328插入/33删除），不是本轮新生产改动；git diff --stat不包含新增文件，不能用它代表完整范围。旧八个新增文件仍在，本轮14个新增文件均未暂存。原有跟踪文件、历史测试、前端、settings/asgi及其他Worktree没有改动。

只读官方网页、文件及Git元数据；执行审计过的前台CPU标准库/Fake/Node测试、有限临时执行池，均已回收。未访问原始训练目录，未查询/操作GPU或训练进程，未启动服务/模型、安装依赖、联网业务服务、迁移或清理缓存。未暂存、提交、推送、合并或创建PR。

等待人工验收；不继续生产替换、P4-C/P4-D或Go开发。真实验证矩阵见INTEGRATION_TEST_PLAN。
