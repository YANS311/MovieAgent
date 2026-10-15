# P4-B 最小实现

## 修改生产文件

1. myapp/services/agent_contracts.py
   - AgentRequestTrace提供sse_lifecycle工厂；新增纯标准库SSERequestLifecycle和_OwnedSSEIterator。
   - 状态与事实分离，固定字段日志，日志异常隔离；惰性factory支持未开始就关闭的记录/释放。
   - close幂等，消费锁防止关闭正在执行的generator；只在恢复控制后协作终止，不创建线程。
   - 统一SSE异常映射，error后不再pull，close/finalize不yield；持有引用释放、普通清理失败可观测。
   - 原JSON错误映射函数、AgentRequestTrace.stage和原有异常类型保持不变。
2. myapp/agent_views.py，仅agent_stream_view函数
   - Session仍在响应构造前；Service仍首次迭代才构造，仍使用原两个constructor参数。
   - 响应交给可登记close的包装器，内部响应属性暴露局部状态，不新增HTTP取消入口。
   - event_stream.finally显式关闭自己创建的Service producer，不依赖GC。
   - 预响应失败记录并继续抛出；原JSON和Feedback等所有其他View函数逐字冻结。
3. myapp/services/agent_chat_service.py，仅stream_chat方法
   - 可选注入请求lifecycle；Agent前/返回后、回放、chunk和保存前后设协作检查点。
   - 最终事务结束后才记录DB确认；取消中提交仍保留结果，不在atomic内检查/抛取消。
   - 无lifecycle的直接Service调用仍执行既有路径；process_chat及其余方法逐字冻结。

## 为什么选择此范围

基线可复现两个局部资源问题，纯标准库/Fake能证明修复，不需要运行模型或真实服务。
把生命周期放在已有纯标准库请求/异常契约模块，避免引入应用初始化、网络依赖或循环导入。
View和Service只持有请求级状态；测试调用实际契约工厂，不给夹具注入不存在的生产符号。
已有测试、包括AST夹具，**没有修改一行**；旧267项与新生命周期测试实际执行通过。

不关闭Agent._agent、全局RAG、Redis client、数据库连接池、其他请求的资源。
不设置__del__可靠性承诺；释放的是自己的factory/iterator/encoder和已挂起generator frame。
普通资源close失败固定代码/计数记录；不依赖日志成功，也不无界重试或删除已提交数据。

## 未采用方案

- 不把View改async、不增加sync_to_async逐next桥或私有ASGI receive监听：需要真实线程/ORM/断连集成证明。
- 不在阻塞Agent内部加取消、不杀线程/进程、不设置强制超时：无法安全保证模型或DB副作用。
- 不添加持久化任务队列/请求状态表、Outbox、幂等DB字段或新的消息系统。
- 不加前端停止按钮/AbortController：现有UI未定义取消协议，浏览器abort也不证明Agent停止。
- 不改sleep、Chunk Size、Token Streaming、LLM/Prompt/ReAct/工具/推荐算法。
- 不改Redis机制、Session backend、Schema/事务、部署/Worker或P4-PERF文件。

## 新增交付

tests/test_agent_sse_lifecycle_isolated.py；docs/agent_backend/下七份P4B文档。
新增测试、报告中没有真实用户数据、绝对本机路径、凭据、权重或训练产物。
Git只允许上述三个生产文件有跟踪差异，其他源码/前端/部署和原测试对基线保持不变。
未暂存、提交、推送、创建PR或合并。
