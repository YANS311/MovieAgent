# P4-B 准入结论

**CONDITIONAL GO：建议人工验收局部同步SSE生命周期修复；真实ASGI断连取消和上线验收未通过验证。**

## 已修复并隔离验证

- 外层结束/关闭时显式关闭内层producer，不再依赖GC；未开始迭代的关闭也有幂等记录和引用释放。
- close发生于活动next中只设协作标记，不对运行中的generator强行close，不创建/终止线程。
- Agent返回、回放、保存前/后检查取消；事务中取消保留其真实提交结果，之后不发done。
- 区分Agent已返回、DB确认、done交接和客户端未知；日志sse_iterator_*不宣称网络成功。
- error后结束来源，不继续pull done；未知/编码/清理异常有固定安全状态，BaseException不被吞掉。
- 预响应Session/sanitize/构造失败结束一次记录且不隐藏异常；Session身份故障未降级成普通cache miss。
- 原有267项 + 新增60项 = **327通过、0失败、0跳过**；无旧测试或夹具修改。

## 剩余P1边界（对可靠ASGI取消/上线仍为阻断）

1. 同步iterator在Django 5.2 ASGI适配中可被完整消费；没有改async桥，客户端断开不保证及时传给同步Agent。
2. 没有ASGI receive信号接入；http_disconnect_observed/client_received_done始终未知。
3. Agent/工具和DB同步调用不能强制中断；检查点与开始保存之间仍存在并发窗口。
4. 真实Django/中间件/Session、Uvicorn/代理/浏览器/线程/DB提交行为尚未集成验证。

这些不使已证明的局部close修复无效，但禁止宣传“关闭浏览器就停止推理”或“已完成真实SSE取消”。
如果人工验收要求本轮必须解决实时ASGI取消，则该更强目标是NO-GO，需要独立设计和授权。

## P2限制及历史边界

- closed表示包装器清理尝试结束；cleanup_errors可表明资源未真正释放；不自动重试失败清理。
- done之后的清理错误可仅在服务日志体现，客户端可能已经收到done，不能撤回或无条件删DB行。
- Agent内部Memory/工具副作用可能已发生，旧ChatHistory写入尝试保持冻结，不承诺取消补偿。
- request_id不是幂等键，commit应答丢失不能当作已回滚；没有新请求取消API或持久化任务。
- 直接Service无lifecycle保持旧行为；不把保证推广至其他调用方。
- 没有真实性能/并发数据，sleep和五字符展示未优化，不声称TTFT/P95/QPS提升。

## 冻结与Git

Worktree：MovieAgent-sse-lifecycle；branch：refactor/agent-sse-lifecycle。
HEAD：e60eb5a96662f9e448df571e5e0e495bd43ba737。
三个生产修改：agent_views.py（仅SSE View）、agent_chat_service.py（仅stream_chat）、agent_contracts.py。
新增一个测试文件、七份P4B报告。完整清单见P4B_IMPLEMENTATION。
JSON API、Agent/LLM/Prompt/ReAct/工具/推荐排序、Redis P0～P3、Feedback与原MySQL事务方法保持。
前端、Schema、settings/asgi、Worker/容器、历史测试和P4-PERF未修改。
AST/内存compile、生产符号绑定、JS语法、Git空白、敏感内容/路径与变更范围检查通过。
没有触及GPU/CUDA设备、训练进程/文件，没有启动模型或真实服务，没有安装/迁移/清理缓存。

未暂存、提交、推送、合并、Rebase或创建PR。等待人工验收，不继续P4-C/P4-D/Go。
