# ASGI streaming adapter：独立标准库 PoC

本目录没有生产调用方，不导入 Django、Agent、ORM、Redis 或模型。推荐方案 C 的可行性实验，不是部署实现。

在仓库根目录，审计导入后以前台方式执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_asgi_streaming_poc_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S experiments/asgi_streaming/lifecycle_probe.py
```

所有 Fake 同步等待最多 0.8 秒，固定延迟最多 0.1 秒。执行池最多两个线程，流准入数不超过线程数；队列最多八个事件，单事件字节数和合作式期限有明确限制。测试总是释放有限 gate 并关闭执行池。

- `async_bridge_poc.py`：同一 worker 完成 factory、next、close；异步消费编码后的事件。
- `fake_agent.py`：有限模拟 Agent、三次原子写入模型、各阶段取消检查点。
- `lifecycle_probe.py`：有界阶段探针与四个有限场景。`poc_evidence.json` 是一次合成证据，不是性能基准。
- `BridgePool` 由调用方显式创建/关闭；`open()` 不启动 producer，第一次异步拉取才启动。未拉取前也应调用 stream.close() 释放准入。
- 消费时用 `async with contextlib.aclosing(encoded_events(stream))`，尤其要覆盖 send 等待期间取消；仅有生成器 finally 不能证明 Django 所有包装层都会及时传播关闭。
- stream.close() 不阻塞，只发信号；aclose() 有界等待。池关闭期限后仍可能报告残留线程，不能强制终止。仅适用于这里已审计的有限 Fake，不得直接连接真实 Agent。
- `call_soon_threadsafe` 是跨线程通知途径；另有最多 5ms 的异步定时复查，防止当前沙箱线程唤醒限制使等待停滞。不是忙等。生产是否需要此回退须另行验证。
- 队列字节上界仅指已编码桥接队列；另有 producer 当前事件、单终态槽、consumer 当前事件。它不限制源对象/序列化临时内存、Agent 结果或数据库驱动内存。
- 无真实 ASGI、数据库、浏览器或代理证据；取消协程不等于停止同步函数，done 交付不等于客户端接收，模拟回滚不等于 MySQL 已验证。

审计、推荐及未决条件见 `docs/agent_backend/p4b1/`。不得将本实验自动注册为服务或生产启动钩子。
