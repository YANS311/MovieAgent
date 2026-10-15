# P4-B 生命周期审计

日期：2026-10-14（Asia/Singapore）。基线：e60eb5a96662f9e448df571e5e0e495bd43ba737。
工作树：MovieAgent-sse-lifecycle；分支：refactor/agent-sse-lifecycle。
结论：**CONDITIONAL GO，局部同步迭代器可靠性；未实现或验证真实ASGI断连取消。**

## 隔离与审计范围

创建前确认P4-A工作树干净、HEAD与origin/refactor/agent-backend一致、origin为YANS311/MovieAgent。
目标路径/分支原先不存在，磁盘可用约94GB；原训练目录仅检查存在性，不读取其源码或未提交状态。
用指定commit创建--no-checkout工作树，稀疏排除experiments、golden_dataset、训练管理命令和旧评测产物。
未复制未跟踪内容、训练数据、权重或检查点。没有修改或暂存P4-PERF及其他工作树内容。
阅读P4-A报告、View/Service/契约、实际Agent调用、模板、URL、settings/asgi和既有测试。
语义索引缺失，使用精确源码检索，未创建索引或触发Embedding。

## 实际调用链与源码位置

行号为本轮源码；未变更Agent/模板位置以基线为准。

| 环节 | 文件、函数及位置 | 行为与持有范围 |
|---|---|---|
| 路由 | myapp/urls.py:93，agent/stream/ | login_required + require_POST；CSRF行为保留 |
| 请求接收 | myapp/agent_views.py:182 agent_stream_view | 服务端request_id、trace、独立lifecycle；不创建Agent |
| Session | myapp/agent_views.py:62 _get_session_key | 在构造有效业务响应前create/复用；失败仍抛，不伪装匿名恢复 |
| 外层持有者 | agent_contracts.py:197 _OwnedSSEIterator | 惰性factory、编码器和当前外层generator；Django可登记close |
| HTTP生成器 | agent_views.py:event_stream（agent_stream_view内） | 首次next才建Service，持有本请求user/input；finally显式关闭内层producer |
| 内层生成器 | agent_chat_service.py:182 stream_chat | 创建不执行；首个yield为intent（204），Agent尚未运行 |
| 真正执行 | agent_chat_service.py:212 → _execute_agent:58 → _get_agent:80 → MovieAgent.run:2524 | 同步完整执行；不在自有持久化事务内；不能强制中断 |
| 回放 | agent_chat_service.py:218/268 | Trace后每条sleep .05s（266）；答案仍每5字符（270），非Token Streaming |
| HTTP等待 | agent_views.py:event_stream | 每个正常被继续消费的事件后sleep .02s；取消后检查点可提前返回 |
| 持久化 | agent_chat_service.py:295 → _persist_result:366 | 保存前检查点；原同alias三次写入短atomic逐字保留 |
| 服务成功 | agent_chat_service.py:299/305 | 事务返回后记录确认；再检查取消，才产生done/meta |
| 编码/交接 | agent_contracts.py:236 __next__ → handoff:143 | 编码成功后交给迭代调用者；不证明HTTP发送或客户端接收 |
| 异常 | agent_contracts.py:236 __next__ → agent_error_payload | 保留原固定错误分类/字段，不自动重试；取消后不发错误帧 |
| 清理 | agent_contracts.py:215/226，agent_views.py:event_stream.finally | 外→内显式关闭，清除factory/encoder/iterator引用，幂等finalize |
| 前端EOF | templates/agent_chat.html:460 sendMessage | receivedDone/error/terminalHandled；保持P4-A.1，不改模板 |

Service仍请求级创建，不关闭共享模型/客户端/数据库连接池。只有已确认归属于该响应的generator被关闭。
直接Service调用无_sse_lifecycle时保持原有路径；JSON API不接入本状态机。

## 基线实际复现

在旧View的AST/Fake Service中保留内层generator强引用，读取一条后close外层，内层finally没有执行。
另以标准库原生generator确认：从未next的generator.close不会执行其body/finally。
本轮以可关闭的惰性iterator处理前者的显式关闭和后者的关闭记录，不依赖CPython回收时机。
原有日志把流遍历完标为success，没有区分Agent返回、DB确认、done交接和客户端接收。

## 状态模型

| 状态 | 本地含义 | 终止路径 |
|---|---|---|
| RECEIVED | 已接收，未完成业务校验 | 预响应异常→FAILED；关闭→CANCELLED |
| VALIDATED | 有效业务请求Session已检查；问候分支无需Agent/Session | intent后RUNNING；关闭→CANCELLED |
| RUNNING | 同步Agent调用正在执行 | 返回后REPLAYING；异常FAILED；协作取消CANCELLED |
| REPLAYING | Agent已返回，回放/准备详情 | 保存前PERSISTING；取消CANCELLED |
| PERSISTING | 已跨过保存前检查点，事务可能正在执行 | 返回记录DB确认；取消不强制打断事务 |
| COMPLETED | 生产者自然耗尽且done已从本迭代器交出 | 仅迭代完成，不证明传输/客户端成功 |
| FAILED | 执行、编码、持久化或清理失败 | 仍可交付时最多一个error；无重试 |
| CANCELLED | 本地cancel/close/GeneratorExit已观察 | 不再产生error/done；不撤销已提交记录 |
| INCOMPLETE | 无done/error的自然EOF | 不报告成功，前端按已有契约提示不完整 |

不设置DISCONNECTED状态：当前同步路径没有可靠的HTTP断开来源。
snapshot.http_disconnect_observed与client_received_done始终为None（未知），而非False或成功。
agent_completed、persistence_started/confirmed、done_handed_off、error_handed_off、iteration_exhausted分别记录。
cancel中事务仍可成功，因此CANCELLED与persistence_confirmed=True可同时成立。
close发生在done交接后但自然EOF前时，保持已提交事实，记录CANCELLED，不宣称客户端收到。
正常自然耗尽后的常规response.close不会把COMPLETED改成取消。

## 同步 / ASGI 核对

本机未定位可直接读取的已安装Django/Uvicorn源码；没有导入框架探测版本。
requirements仅声明django>=5.2，实际部署版本未锁定；以下是官方5.2参考源码，不是部署实测：

- [Django 5.2 response.py](https://github.com/django/django/blob/5.2/django/http/response.py#L460-L502)：
  _set_streaming_content登记close；__aiter__的同步适配走sync_to_async(list)，可能在发送前完整消费。
- [Django 5.2 ASGIHandler](https://github.com/django/django/blob/5.2/django/core/handlers/asgi.py#L173-L225)：
  disconnect监听取消响应协程，之后调用response.close；这是协程层，不等于终止同步Agent线程。
- [官方断连说明](https://docs.djangoproject.com/en/5.2/ref/request-response/#handling-disconnects)
  针对异步流捕获并重新抛出CancelledError；不能直接套在现有同步生成器宣称已取消。
- [Uvicorn h11源码](https://github.com/encode/uvicorn/blob/master/uvicorn/protocols/http/h11_impl.py#L473-L490)：
  ASGI receive返回disconnect；当前View没有receive访问，正常response_complete也会影响该返回。

因此，新wrapper仍是同步iterator，ASGI整段消费/线程占用风险未解决；close回调可能很晚才触发。
服务端COMPLETED可早于HTTP实际发送，日志使用sse_iterator_*明确该边界。
不把同步异常或普通close推断成物理网络断开，也不声明Uvicorn实际安装版本/代理行为已验证。
