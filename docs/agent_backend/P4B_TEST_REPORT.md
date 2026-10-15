# P4-B 隔离测试报告

这是CPU-only、标准库、AST、FakeDB/Fake Agent及Fake DOM测试，不是Django/ASGI/MySQL/浏览器集成验证。
原有测试与夹具未修改，没有删除/跳过/放宽断言；未读取或执行P4-PERF未提交测试。

## 导入链审查

新增入口test_agent_sse_lifecycle_isolated.py的直接import均为标准库，运行前AST/内存compile检查。
正常文件加载P4-A测试夹具，继而加载P0/P0.1/P3/P3.1安全夹具；只导入纯标准库cache/helper/contract模块。
AgentChatService/Views只提取实际AST定义；Agent、DB路由/atomic、HTTP响应、时间均Fake/Mock。
Guard在加载夹具前阻止Django/redis/torch/Transformers/FAISS/模型服务等真实导入。
asyncio仅标准库导入、注入CancelledError，没有创建event loop/后台任务；threading只创建同步锁/标记，没有start线程。
subprocess仅只读git show/diff；旧P0的纯key稳定性子进程不导入Django或模型。
JS测试只用Node内置assert/fs/path/vm和实际模板函数+Fake fetch/reader/DOM，无真实网络或浏览器。
没有django.setup、manage.py、真实agent.run、Redis/Lua/SQL或模型权重加载。

## 实际执行结果

| tests/ 下脚本 | 通过 | 失败 | 跳过 | 退出码 |
|---|---:|---:|---:|---:|
| test_redis_cache_p0_isolated.py | 14 | 0 | 0 | 0 |
| test_redis_cache_p01_isolated.py | 18 | 0 | 0 | 0 |
| test_redis_cache_p1a_isolated.py | 18 | 0 | 0 | 0 |
| test_redis_cache_p1b_isolated.py | 8 | 0 | 0 | 0 |
| test_redis_cache_blockers_isolated.py | 23 | 0 | 0 | 0 |
| test_redis_cache_p2_isolated.py | 24 | 0 | 0 | 0 |
| test_redis_structures_p3_isolated.py | 41 | 0 | 0 | 0 |
| test_redis_structures_p31_isolated.py | 26 | 0 | 0 | 0 |
| test_redis_structures_p32_isolated.py | 23 | 0 | 0 | 0 |
| test_agent_backend_p4a_isolated.py | 42 | 0 | 0 | 0 |
| test_agent_sse_lifecycle_isolated.py（新增） | 60 | 0 | 0 | 0 |
| test_agent_sse_terminal_isolated.cjs | 30 | 0 | 0 | 0 |
| 总计 | **327** | **0** | **0** | **全0** |

Python297（历史237+新增60），JS30；与P4-PERF的新增28项不是同一分支，不把其数量计入本轮。
真实服务项目未执行属于测试层次缺口，不是上述单元测试中的skip。
首版新增56项通过后，补充预响应失败、response closer API形态及清理失败共4项，最终60项全部通过。
最终原有全套重跑，无阻断性回归失败。

## 新增60项覆盖说明

- 正常：响应惰性、Session已建立、intent先于Agent、原事件/卡片/Trace兼容、恰好一个done、三写一次。
- 状态：RUNNING/REPLAYING/PERSISTING、Agent返回/DB确认/done交接/EOF独立；客户端接收和断连始终未知。
- 失败：Agent分类/依赖、三个写入逐项失败回滚、commit应答丢失、unknown/编码错误、request_id和安全消息。
- 取消：第一次yield前、intent后、Fake Agent内部、Trace/chunk中、保存前、事务中、提交后done前、done后EOF前。
- 关闭：强引用内层generator的finally、重复close/cancel、活动next中延迟清理、未开始factory释放、不恢复yield后代码。
- 异常：GeneratorExit、KeyboardInterrupt/SystemExit、同步路径注入CancelledError均不被伪装成功。
- 清理/日志：普通close错误计数、不泄漏原错误、BaseException传播、坏日志处理器不影响业务、不关闭共享资源。
- 协议：error后不拉取后续done、自然空EOF/缺done为INCOMPLETE、done后异常按已有前端失败契约处理。
- 兼容：问候/拦截原事件、JSON不接入lifecycle、原Service推理/JSON/事务方法逐字冻结，生产真实符号绑定。
- 预响应：sanitize/Session失败继续抛且只结束一次；response构造失败不创建Agent。

## Fake 能力边界

active-next测试是在Fake Agent/DB回调内重入close，证明锁与协作控制流，不证明真实多线程调度/HTTP时序。
FakeDB分离已提交行和事务帧，能验证代码顺序/三写一致；不是InnoDB、连接池、路由或网络提交证明。
response closer测试只模拟Django登记iterable.close的API形态，不运行中间件或ASGIHandler。
主动调用close/cancel，不是浏览器Abort/网线断开；没有证明真实断连能及时传到这些检查点。
直接Service无lifecycle仍保留旧语义，不能把新关闭保证推广到所有调用方。
Agent内部缓存/工具副作用未执行、未补偿；cleanup_errors>0不等于资源已释放。
没有测得TTFT、断连延迟、CPU/RSS、并发/QPS或生产P95提升。

## 实际命令口径

所有Python脚本逐个前台执行，使用以下环境；-S与这些标准库/AST入口兼容：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_agent_sse_lifecycle_isolated.py
```

表中其余Python文件用同一前缀逐个运行，无测试自动发现或完整Django套件。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_THREADPOOL_SIZE=1 node --check tests/test_agent_sse_terminal_isolated.cjs
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 UV_THREADPOOL_SIZE=1 node tests/test_agent_sse_terminal_isolated.cjs
git diff --check
```

Python语法采用ast.parse+内存compile，不生成pyc；实际生产helper导入/导出与symtable绑定一起检查。
跟踪差异限定三个生产文件；新增Python/报告做全文件空白、敏感内容和本机路径检查。
原有267项脚本及模板字节与基线相同；原MySQL事务函数与JSON路径冻结。
