# P4-B SSE 兼容契约

URL仍为POST agent/stream/，登录、require_POST、CSRF、Session隔离和响应头保持。
SSE编码仍由_sse_data产生data: JSON及两个换行；无新事件类型或前端字段。

## 正常事件

Service先intent，再完整Agent执行，再按原顺序发送thought/action/observation/clarification、chunk和done/meta。
既有反思/其他trace信息展示格式不改变；当前Service不为所有trace类型单独yield，未扩展工具语义。
答案仍五字符分段；Trace sleep .05s，View事件后sleep .02s；正常路径等待顺序保留。
本地取消可在下一检查点跳过后续等待，这不是对已进入sleep的强制唤醒。
完整事件字典与P4-A历史输出对照测试通过；推荐卡片、Trace及逐字符前端完全未修改。

## 成功、失败与EOF

有效业务结果的done只在原短事务返回且取消检查通过后产生。
唯一历史例外是空输入问候done，它不执行Agent/数据库保存；该分支原样保留，不能把它算作持久化成功。
done_handed_off仅表示编码后的事件从本包装器返回给上层迭代调用者，不证明HTTP send或客户端接收。

Agent/持久化/未知异常仍调用同一个agent_error_payload：
type=error、content为固定安全消息，附code/request_id/retryable=false；不输出原异常/SQL/凭据。
首次error立即结束并关闭来源，不继续pull后面的done；本地取消后不再发送错误帧。
取消没有新增type=cancelled成功信号，也不会自动重试。
GeneratorExit、KeyboardInterrupt、SystemExit和注入同步路径的CancelledError不伪装业务成功或普通error；
按对应关闭/异常路径清理后继续传播BaseException。

极端情况下done之后生产者又抛异常，包装器仍可发一个error；前端P4-A.1本来就将done后error视为失败。
这不发生在正常Service末尾，新增Fake源用于验证防御语义；始终禁止error之后done。
若错误帧的编码器本身不可用，不能承诺能制造error帧；清理后传播异常，前端按原流中断语义处理。

无done的自然EOF记录INCOMPLETE；有error的EOF不再重复发error；本地关闭不yield。
前端只有done且无error、正常EOF才能完成成功渲染；HTTP非2xx、缺done或reader异常保持失败。
持久化前已经显示的chunk可以保留，但持久化失败必须显示本次请求未成功完成。
前端实际接收事实不回传服务器，所以snapshot.client_received_done始终未知。

## 清理与日志是另一条事实线

done交接后才发生的普通cleanup错误会计数/记录FAILED，不重试数据库，也不撤销已交出的事件。
不能保证客户端知道这种清理失败，不能据前端成功认定资源全已释放。
request_finished outcome现在为sse_iterator_completed/failed/cancelled/incomplete，避免把本地遍历当传输成功。
仅固定状态、布尔事实、错误代码、清理计数和服务端request_id；无聊天、画像、Session/用户标签。
预响应sanitize/Session失败仍向上抛，额外记录一次本地失败；不把Session故障转换为无状态成功。
JSON API、Feedback事务与缓存失效契约不变。
