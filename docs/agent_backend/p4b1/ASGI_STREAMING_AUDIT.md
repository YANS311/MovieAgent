# P4-B.1：ASGI 执行链审计

范围：MovieAgent-sse-lifecycle，refactor/agent-sse-lifecycle；HEAD 为 e60eb5a96662f9e448df571e5e0e495bd43ba737。P4-B 三个已有生产修改、八个已有新增文件按 SHA-256 冻结，本轮不改其字节。路径均相对该 Worktree。

## 本地源码证据

| 位置 | 实际行为与含义 |
|---|---|
| `movie/asgi.py:16` | get_asgi_application()；本轮只读，未导入执行 |
| `myapp/apps.py:10,25,29` | ready 导入 views 并 warmup_all_systems；禁止 django.setup()/完整应用初始化 |
| `myapp/agent_views.py:182,203–214` | 同步 view；respond 把 lifecycle.wrap 的同步对象交给 StreamingHttpResponse |
| `myapp/services/agent_contracts.py:197,212,226,236` | _OwnedSSEIterator 提供 iter/next/close，没有异步迭代协议；活动 next 中 close 只发协作信号 |
| `myapp/agent_views.py:247–254` | Session 在响应构造前解析，故障仍传播；本轮未改变 |
| `myapp/agent_views.py:256–275` | 第一次消费才构造服务；嵌套 producer 有 finally 关闭；每次回放后同步 sleep(0.02) |
| `myapp/services/agent_chat_service.py:182,204,212` | 先 intent，再完整同步 Agent.run，之后回放；不是 LLM Token Streaming |
| 同文件 `:266,270,295–305,366` | trace sleep(0.05)、五字符 chunk、短事务、确认保存后 done；本轮冻结 |
| `myapp/middleware.py:5–36`、`movie/settings.py:61` | 两个自定义 middleware 为同步类；无 async_capable 声明；均不遍历或替换 streaming_content |
| `templates/agent_chat.html:490–592` | done/error/EOF 终态标记及 HTTP 状态检查保持；EOF 不能补造成功 |

## 框架语义（官方源码只读，未运行）

Django 5.2 的 [_set_streaming_content / __iter__ / __aiter__](https://github.com/django/django/blob/5.2/django/http/response.py#L460-L502) 区分同步/异步迭代器；类型不匹配时会完整收集。因此当前对象预期 is_async=False，并在 ASGI 路径进入 sync_to_async(list)；首次业务 yield 不对应及时 body send。未来纯异步对象预期 is_async=True，但本轮没有实际 Django 构造证据。

[ASGIHandler](https://github.com/django/django/blob/5.2/django/core/handlers/asgi.py#L142-L225) 为请求设 ThreadSensitiveContext、并行等待断连和响应、取消未完成协程；[send_response](https://github.com/django/django/blob/5.2/django/core/handlers/asgi.py#L283-L325) 经异步迭代逐段 send。其关闭包装器不代表能杀掉同步工作。真实连接信号和清理延迟仍待验证。

[BaseHandler](https://github.com/django/django/blob/5.2/django/core/handlers/base.py) 适配同步 view/middleware。当前同步生成器预消费通常在请求对应的 thread-sensitive executor 内执行；确切线程调度与已部署 asgiref 版本有关，未经实测。同步 sleep 占用执行线程，不是 event-loop 上的 await。

`response.close()` 注册并调用 close；正常耗尽、显式 close 或异常释放可能触发 generator finally/GeneratorExit，但活动同步调用不能被并发强制 close。Django 的异步响应外层 aclosing 也不能替代对底层桥接对象关闭所有权的设计。

## Middleware 与真实版本边界

同步 middleware 会引入模式适配和线程成本；这两个自定义类本身不改变迭代器类型。不能因此声称整条中间件链已通过异步兼容验证。将来必须确认第三方 middleware、Session Cookie、鉴权惰性对象、异常路径、response.close 与 async finally。

实际执行 Python 3.13.5；审计活动解释器 sysconfig 安装目录及系统常见 dist/site-packages 的元数据，未找到可用 Django/asgiref/uvicorn 安装。requirements/environment 的 django>=5.2 不能锁定部署版本。官方 5.2 与部署版本的差异未知；asgiref 3.8.1、Uvicorn 官方源码仅供语义参考，不能冒充本机版本。

[asgiref SyncToAsync](https://github.com/django/asgiref/blob/3.8.1/asgiref/sync.py) 使用 executor、context 及取消协调；已运行同步函数没有通用强制终止机制。[Django async 指南](https://docs.djangoproject.com/en/5.2/topics/async/) 要求同步事务在同步函数内处理，并注意连接线程亲和性。

本轮没有导入 movie.asgi、构造真实 HTTP 请求或监听 socket；Fake close 不等于 ASGI disconnect。预测的预消费问题有源码证据，真实 TTFE、断连传播和线程释放没有生产测量数据。
