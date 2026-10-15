# P4-B 取消与业务提交语义

## 信号来源与能力

只有本地显式response.sse_lifecycle.cancel()、包装迭代器close()、或实际抛入同步迭代的GeneratorExit。
response.sse_lifecycle是进程内诊断/协作钩子，不是新增HTTP接口；没有全局请求注册表或跨用户取消机制。
Django登记iterator.close的机制已审阅，是否在真实ASGI断连时及时调用尚未验证。
普通response.close也用于正常结束，不能据此断言网络断开。
没有增加前端取消按钮、AbortController、ASGI receive监听或后台任务。

| 信号发生位置 | 已实现的局部处理 | 不保证的能力 |
|---|---|---|
| 第一次next前close | 不执行factory/Service/Agent，立即释放factory并记录一次关闭 | 不证明HTTP可在这个时刻触发close |
| intent后、Agent前 | 停止后续执行，关闭内外generator | 不保证客户端看到intent时Agent尚未执行，ASGI可能预消费 |
| 同步Agent中 | 只设cancel标记；调用返回后停止回放/本Service最终三条写入 | 不能中断LLM、工具、网络、GPU或Agent内部副作用 |
| Trace/chunk中 | 下一协作边界停止，不发新的成功/错误帧 | 不撤回已交出的事件；不唤醒已经开始的sleep |
| 持久化前 | 若检查点已看到取消，跳过整个最终三写事务 | 检查点与事务进入之间存在普通并发窗口 |
| 持久化进行中 | 让原事务自然完成/失败，之后再读取消标记 | 不强制回滚、超时或中断数据库调用 |
| 提交确认后done前 | 保留两history/一trace，停止done | 不能以取消为由删除记录或重放请求 |
| done交接后、EOF前close | 保存done交接事实但本地标CANCELLED | 不知道客户端是否收到done |
| 完整迭代后close | 幂等资源关闭，保留COMPLETED | COMPLETED只是迭代完成，不是网络/用户确认 |

close与next共享非阻塞消费锁。next正在运行时close只标记，绝不对正在执行的generator调用close。
next重新取得控制后由其本身清理，避免generator already executing；没有创建/杀死线程或进程。
重复cancel/close/finalize不再次执行Agent/DB或重复登记request_finished。
清理函数不yield，普通清理异常计数、固定代码日志，不覆盖主业务错误；BaseException继续抛出。
cleanup_errors>0表示清理失败，**closed仅表示本包装器已结束持有/清理尝试，并非所有底层资源均成功释放**。
没有对失败清理进行无界重试。

## 提交与副作用边界

_execute_agent、_persist_result、_save_history、_save_trace和JSON process_chat逐字保留。
同库、独立短事务、三个写入和原异常cause链保持；取消检查不插入事务内部。
三写中失败回滚；提交确认后设置persistence_confirmed，不受晚到取消改写。
若commit应答丢失，可能已提交但persistence_confirmed=False，表示未知而不是确认未写。
不增加自动重试，不把request_id当幂等键；用户重复HTTP请求仍可能重复执行。

尤其不能把“跳过Service最终三写”说成“本请求没有任何持久化副作用”：
MovieAgent.run:2550先更新MemoryManager槽位，可能已写Redis；2585还有旧ChatHistory.create尝试并吞异常。
后者字段与当前模型不匹配是历史审计记录，本轮冻结Agent，未修复也不依赖它必然失败。
工具、画像、日志等Agent内部行为不在新协调器的撤销范围。取消不补偿已发生副作用。

## 断连与超时限制

当前没有可靠HTTP断连标志，http_disconnect_observed=None；不声称实现DISCONNECTED。
未消费/已缓冲事件、代理背压、正常HTTP完成、浏览器Abort与同步线程状态之间关系尚未集成验证。
本轮没有设置超时器；HTTP连接、SSE发送、Agent/工具调用和DB写入超时是四个不同边界。
普通计时不能中断同步工作；不通过杀线程、进程、CUDA操作、关闭共享池来模拟强取消。
同步ASGI适配完整消费的缺口仍在，详见审计与独立集成计划。
