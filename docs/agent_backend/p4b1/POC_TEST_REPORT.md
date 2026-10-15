# P4-B.1 隔离测试记录

日期：2026-10-14。Python 3.13.5。只有标准库、AST、Fake、Node builtin VM；未初始化 Django、真实服务、模型或 GPU。活动 Python 的审计安装位置未发现可安全使用的 Django，真实 StreamingHttpResponse 最小构造明确跳过。

## 最终实际结果

| 脚本（tests/） | 测试数 | Pass | Fail | Skip | 测试框架报告秒数 |
|---|---:|---:|---:|---:|---:|
| test_redis_cache_p0_isolated.py | 14 | 14 | 0 | 0 | 1.091 |
| test_redis_cache_p01_isolated.py | 18 | 18 | 0 | 0 | 2.000 |
| test_redis_cache_p1a_isolated.py | 18 | 18 | 0 | 0 | 0.691 |
| test_redis_cache_p1b_isolated.py | 8 | 8 | 0 | 0 | 1.385 |
| test_redis_cache_blockers_isolated.py | 23 | 23 | 0 | 0 | 1.886 |
| test_redis_cache_p2_isolated.py | 24 | 24 | 0 | 0 | 0.505 |
| test_redis_structures_p3_isolated.py | 41 | 41 | 0 | 0 | 0.529 |
| test_redis_structures_p31_isolated.py | 26 | 26 | 0 | 0 | 0.086 |
| test_redis_structures_p32_isolated.py | 23 | 23 | 0 | 0 | 0.101 |
| test_agent_backend_p4a_isolated.py | 42 | 42 | 0 | 0 | 2.984 |
| test_agent_sse_lifecycle_isolated.py | 60 | 60 | 0 | 0 | 3.975 |
| test_asgi_streaming_poc_isolated.py（新增） | 46 | 45 | 0 | 1 | 0.558 |
| test_agent_sse_terminal_isolated.cjs | 30 | 30 | 0 | 0 | 未单独计时 |

旧327项全部实际通过；新增45项通过、1项明确跳过。总计发现373项，**372通过、0失败、1跳过**。全部最终测试命令退出码0。三处模拟事务失败点在一个测试的三个 subTest 中验证，不额外虚增测试数。

## 新增覆盖与证据

增量/顺序：intent 在阻塞 Agent 返回前可被本地消费；Fake 同步 Agent 后才 trace/chunk，不能称 Token Streaming。正常得到 intent→thought→action→observation→4个chunk→done，合并回答 synthetic answer。

取消：源未启动、Agent 前、Agent 中、trace、chunk、保存前、事务中、提交后；CancelledError 传播。事务中取消不会取消 worker，最终保留三条模拟写入。同步运行中 stop 后 lease 继续占位。pool 超期 drained=False 以及释放有限 gate 后 drain/join 均验证。

背压/资源：容量1/2、慢消费者、满队列取消、错误唤醒、严格准入 W=1/2、完成但未读流占位、重复close、异步generator finally、send等待取消的aclosing、无额外pending任务、工作线程归零。所有异步测试 teardown 显式释放 gate、取消并等待自有task、join自有池。

终态：error后不读done；重复done、done后失败、编码失败、清理Exception/BaseException不成功；持久化三失败点/确认丢失不假成功不自动重试；空EOF/缺done无成功。done交接和客户端接收分开。

静态：标准库导入白名单、禁止服务/模型导入及socket连接、生产及PoC全局符号绑定、原11个文件SHA-256和生产diff摘要保持。旧夹具继续检查真实生产符号，未注入缺失符号来掩盖错误。

## 一次有限合成执行

原始状态见 `experiments/asgi_streaming/poc_evidence.json`，每场景一次，不计算分位数：

| 场景 | 首次本地交接ns | 取消时同步工作残留 | 最终模拟提交行数 | join后自有线程/额外pending task |
|---|---:|---|---:|---|
| normal | 603908 | 不适用 | 3 | 0 / 0 |
| cancel_agent | 390394 | 是 | 0 | 0 / 0 |
| cancel_transaction | 297156 | 是 | 3 | 0 / 0 |
| full_queue | 279760 | 本场景关闭返回时已结束 | 0 | 0 / 0 |

队列容量2，单池worker上限1；这些是当前沙箱的本地交接观察，不是 TTFE、真实ASGI延迟、SQL或性能提升。

## 开发中发现并处理

新增PoC首轮42项：13个error结果（含同一测试3个subTest错误）、1跳过，26.553秒。独立标准库复现发现线程通知未及时唤醒本环境 selector。新增有界5ms定时复查后通过；再增加通知缺失、并发pull、双worker准入、实际符号绑定测试。原测试与生产文件未改。未把失败归因于业务依赖，也未放宽断言。

## 导入链与可复现命令

先 AST 检查所有测试 imports、spec_from_file_location、exec、subprocess。原测试仅加载已审计的标准库辅助模块，业务类/方法由AST抽取并注入Fake；子进程为只读Git或稳定Key检查。JS只读模板并运行Node VM，fetch为Fake。

在仓库根目录，逐个以前台方式执行上表每个Python脚本：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/<上表脚本名>.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 UV_THREADPOOL_SIZE=1 node --check tests/test_agent_sse_terminal_isolated.cjs
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 UV_THREADPOOL_SIZE=1 node tests/test_agent_sse_terminal_isolated.cjs
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S experiments/asgi_streaming/lifecycle_probe.py
```

检查实际修改/新增Python：ast.parse + 内存compile（不产生pyc）；生产符号检查已执行；git diff --check；新增文本逐行尾空白、敏感路径、凭据及范围检查。

未验证：Django is_async实构造、中间件和Session、ASGI body发送、真实disconnect、Uvicorn/代理/浏览器、真实数据库线程连接/回滚/提交不确定性、并发性能。没有把Mock close称为网络断连。
