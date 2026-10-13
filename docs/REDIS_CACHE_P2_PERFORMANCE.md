# MovieAgent Redis P2：性能基准、可观测性与优化评估

日期：2026-10-13（Asia/Singapore）。这是 CPU-only 合成微基准与 Mock 回归报告，不是生产性能报告。

## 1. 范围、基线与结论

工作目录：MovieAgent-redis-perf（Git Worktree，分支：perf/redis-benchmark）。
基线 Commit：`ccbc04d9e2e3208c906a32fba7da8dd4c7b24d25`。
冻结目录 `MovieAgent-redis-cache` 在创建前干净且 HEAD 匹配；目标路径和分支不存在。
磁盘可用约 69GB；基线跟踪文件约 9.26MB，无训练数据、模型权重、检查点或实验产物后缀/目录命中。
用禁用 hooks 的 worktree add 创建新目录，仅检出 Git 跟踪源码/静态资源；未复制原目录未跟踪内容。
没有访问原始训练工作区的源码、未提交文件或 Git 状态；新分支/Worktree 注册使用共享 Git 元数据。

**阶段结论：基准、默认关闭的观测工具与 105 项隔离回归完成；不新增生产性能优化。**
唯一测量的热路径优化候选收益不稳定，未应用。八个重点生产模块逐字节匹配冻结基线。
新增指标工具没有自动接入生产入口，因此默认生产路径没有新增指标日志、计时、序列化或导入开销。
没有生产 QPS、真实缓存命中率、接口延迟或数据库耗时数据，也不作生产性能提升承诺。

## 2. 混合推荐真实代码链路

`views.hybrid_recall_recommend` 的在线调用经 `_get_hybrid_candidates`（top_k=8）；
Agent 的 HybridRecallTool/HybridRecallSkill 另走 `recommender.recall.multi_channel_recall`。两者并非同一个缓存。

1. 先构造含 user_id、top_k、用户代际和目录代际的 key。已存在两个代际时执行两个 backend.get；
   两个代际都缺失时各 get/add/get，合计四个 get、两个 add。代际错误返回 None，真正绕过结果缓存。
2. 热路径：business_cache_get → 校验 schema、统计类型、每个 ID/有限分数 → 一次 Movie filter(id__in)
   + genres/actors/directors 的 prefetch_related → 构建 ID 映射、恢复顺序和四位小数 rrf_score →
   返回 (movies_list, profile_text, source_stats)，补回 stats.profile_text。
3. 热路径不会构建用户画像、加载 RAG、执行召回或 RRF；仍需从 ORM 读取当前字段和关联。
   Django 后端的 get 包含其反序列化；本次微基准将反序列化和恢复拆开测量。
4. 冷路径：画像/已看列表 → RAG 资源 → 三路召回 → 原 RRF 权重 1.2/1.0/1.5 → 已看过滤/
   热门兜底 → 批量电影和关联读取 → 原排序/分数 → 轻量 payload 构建 → business_cache_set，TTL 1800。
   代码注释称“并行”，实际三个调用在本函数内顺序执行；本轮未改其策略或并发。
5. 坏 payload 或缺失电影：best-effort 删除精确 key，再走原冷路径；ORM 错误继续抛出。
   用户/目录失效使旧代际不可再命中；本次计算只向开始时捕获的 key 写入，防止旧请求污染新代际。

FakeQuerySet 的一次 filter/一次 prefetch 调用不等于一条 SQL：真实 Django 关联预取通常会有额外 SQL。
本轮只确认调用形态；真实查询数、关联实例构建及实际网络等待待验证。

## 3. Agent 及其他缓存总览与潜在瓶颈

以下都是逻辑 key，部署后仍由 Django KEY_PREFIX 包装；未读取运行时 Redis 配置。

| 模块/入口 | key / payload / TTL | 热路径与风险 |
|---|---|---|
| views 混合推荐 | hybrid_rec_payload:v3:uid:ugen:top_k:k:catalog:cgen；hybrid_rec_payload_v1；1800s | 2 次代际读 + 结果读 + ORM 恢复；轻量数据减少快照，但没有消除 SQL |
| movie_agent.LLMIntentParser | intent:v2:SHA256(trimmed_input)；字典；600s | 不读 generation；成功解析才写；空输入立即返回、解析失败未负缓存，连续失败可重复回源 |
| views.get_kg_subgraph | kg_subgraph:v4:topic_sha:history_sha:max_triples:catalog:cgen；文本；1800s | 历史去重排序和摘要、目录读；成功查询空字符串也可命中，图未初始化/异常不写 |
| recall.multi_channel_recall | multi_recall:v3:uid:ugen:params:SHA256(JSON([query,top_k,weights])):catalog:cgen；列表+统计；1800s | 参数 JSON 与摘要、2 次代际读；缓存中的 latency_ms 是原冷路径统计，不是当前请求延迟 |
| QueryRewriter.rewrite | query_rewrite:v2:SHA256(trimmed_query)；字符串；3600s | 与意图缓存任务不同，不宜合并；key 未包含模型/改写配置，切换配置可能等 TTL 到期 |
| agent_service / ajax_explain_rec | explain:v3:uid:ugen:movie_id:catalog:cgen；解释字典；600s | 在线解释入口同一个 key 贯穿读写；便捷函数省略 key 时各自获取 generation |
| agent_service agent recall | agent_recall:v3:uid:ugen:空参数:catalog:cgen；列表；1800s | 精确检索仅找到两个定义，无活动调用方；未来使用时写入重新取代际的竞态仍需处理 |
| views CB | cb_recs:v3:uid:ugen:空参数:catalog:cgen；source+Movie 列表；3600s | 仍含 ORM 快照；非本轮重构对象，不扩大轻量化范围 |
| views 电影概念 | movie_concepts_jieba_SHA256(JSON([title,summary])):catalog:cgen；概念列表；86400s | 目录代际 TTL 1800s，可能提前失去可达性，旧值仍占空间到自身 TTL |
| cache_mutations | 用户/目录 generation 轮换，用户 legacy 精确删除 | on_commit 同步执行，批量用户工作量随用户数增长；失败聚合及恢复边界保留 |
| rerank.business_filter | 重排业务过滤，非结果缓存 | 静态代码逐候选 filter(id).values(is_sensitive).first，潜在 N+1；需真实查询统计后再考虑批量读取 |
| rerank.diversity_guarantee | 类型多样性 | 已使用批量 Movie filter + genres prefetch；没有改排序规则 |

混合推荐与多路召回可能重复 vector/content/model 工作，但 KG 路数、参数、画像来源、返回格式不同，
不能假设缓存可共用。没有新增缓存、负缓存、锁或后台合并。
高基数不同查询/历史自然产生未命中；本轮没有测量穿透率或生产访问分布。
代际 TTL 1800s 到期产生随机新 token，可能令长 TTL 条目提前冷缓存。保留其防旧代际复活的设计。

## 4. CPU-only 微基准方法

脚本：`benchmarks/redis_cache_microbench.py`，仅标准库。使用确定性的 SyntheticMovie、FakeCache 和 FakeQuerySet。
每部合成电影包含 ID/标题/评分/约 800 字节合成摘要、1 类型/3 演员/1 导演、模拟 state 和预取列表。
画像也为固定合成字符串；不读项目数据、模型或真实用户内容。
A：旧式代表性对象快照元组 (movies, profile_text, stats)，不是 Django Model。
B：当前生产代码 AST 提取的轻量 payload 构建，pickle 编码/解码。
C：B 加生产 _restore_hybrid_cache_payload 的 AST 执行，Movie 替换为 FakeManager，批量创建合成实例。
FakeQuerySet 故意逆序返回，恢复时必须还原缓存顺序；三个关联列表为静态合成数据，无 I/O。
A 反序列化恢复自带快照；C 读取合成 live rows 后创建对象。两者本轮字段/分数/关联值完全等价，
但数据库更新后的新鲜度语义不同；这是轻量方案的既有特征，不是基准证明的生产一致性。

环境：Python 3.13.5，Linux x86_64；pickle protocol 5。未安装/导入 django-redis，未确认生产 serializer/protocol。
每操作预热 5 次、采样 80 次；每样本后 sleep 2ms，休息不计入耗时；单进程、串行，无多线程/进程。
运行预算 20 秒，脚本拒绝超过 120 次/10 次预热/30 秒预算的设置；预算按采样边界检查，不是进程超时器。
P50/P95/P99 使用 nearest-rank（ceil(N*q)-1）；80 个样本的 P99 是最大值，尾部统计不稳定。
环境存在其他负载的可能，未查询或操作训练进程，未固定 CPU/调度优先级；不据此推断生产延迟。
500 部是合成规模扩展，不代表线上每次返回 500 部；生产各路 k=80 也限制实际候选池。

初始基线保留在 redis_cache_microbench_baseline.json。其 A 构建用了浅复制列表/统计；
最终复测改为与旧代码一致的元组引用组装，并保存原始 samples_ns，最终表格取自 validation.json。
其他八个生产模块没有更改；两次均不是“优化前后”的生产对照，而是同一冻结基线的重复微基准。

初始运行：wall=4.814s，CPU=0.685s。最终复测：wall=5.006s，CPU=0.857s。
CPU 时间含合成数据、AST 解析及统计；wall 包含节流休息。没有施加系统级 CPU quota，安全来自小规模、有界串行与主动休息。

## 5. 实测数据

字节数为 pickle blob 长度，C 使用与 B 相同 blob；不是 Redis key/对象内存、压缩后大小或网络包大小。

| 电影数 | A 对象快照 bytes | B/C 轻量 bytes |
|---:|---:|---:|
| 30 | 34664 | 1331 |
| 100 | 115045 | 2661 |
| 500 | 593258 | 10509 |

所有下表耗时单位为 **µs**，没有 SQL/Redis/模型执行。

| 电影数 | 操作 | P50 | P95 | P99 |
|---:|---|---:|---:|---:|
| 30 | A 元组构建 | 4.003 | 4.693 | 5.032 |
| 30 | A 序列化 | 129.585 | 158.053 | 242.1 |
| 30 | A 反序列化 | 141.093 | 169.327 | 259.577 |
| 30 | B payload 构建 | 13.378 | 30.834 | 55.244 |
| 30 | B 序列化 | 23.697 | 43.972 | 86.161 |
| 30 | B 反序列化 | 22.667 | 39.212 | 49.195 |
| 30 | C 批量恢复（不含解码） | 99.215 | 144.936 | 305.042 |
| 30 | C 解码 + 批量恢复 | 124.528 | 188.894 | 198.246 |
| 100 | A 元组构建 | 2.147 | 4.696 | 4.948 |
| 100 | A 序列化 | 456.48 | 616.778 | 768.051 |
| 100 | A 反序列化 | 466.902 | 480.141 | 537.626 |
| 100 | B payload 构建 | 28.473 | 44.862 | 62.212 |
| 100 | B 序列化 | 41.982 | 61.335 | 65.459 |
| 100 | B 反序列化 | 41.333 | 56.651 | 81.381 |
| 100 | C 批量恢复（不含解码） | 253.891 | 280.364 | 386.1 |
| 100 | C 解码 + 批量恢复 | 287.032 | 314.28 | 329.02 |
| 500 | A 元组构建 | 1.273 | 5.126 | 5.716 |
| 500 | A 序列化 | 1927.536 | 1976.809 | 2268.486 |
| 500 | A 反序列化 | 2309.761 | 2360.881 | 2907.307 |
| 500 | B payload 构建 | 86.248 | 109.844 | 137.028 |
| 500 | B 序列化 | 112.394 | 130.469 | 137.702 |
| 500 | B 反序列化 | 114.458 | 144.307 | 179.396 |
| 500 | C 批量恢复（不含解码） | 870.308 | 1162.366 | 1195.909 |
| 500 | C 解码 + 批量恢复 | 981.851 | 1052.818 | 1241.569 |

热 generation 构键 FakeCache P50/P95/P99：9.957/22.951/23.844µs；
首次两个 token 都不存在为 4 get + 2 add，已有 token 为 2 get。此计数不含业务 payload get。
结果说明在本合成对象中轻量 blob 明显更小，序列化工作量更少；构建 payload 和恢复对象也有额外 CPU 工作。
不同操作独立采样，不能把其 P95/P99 相加作为端到端分位数；C 解码+恢复为单独整体测量。
真实 MySQL/Redis 的网络、serializer、关联实例规模会改变结论，尤其不能把 C 推导成生产热缓存吞吐量。

## 6. 最小优化候选与未实施原因

候选：把恢复末尾 round(float(scores[movie.id]), 4) 改为 round(scores[movie.id], 4)。
理由：此前已用 float(score) 转换并检查有限值，scores 中是内置 float，第二次转换冗余。
仅在内存中生成候选 AST，未写生产源码。候选测量同输入、同 manager 类型、交替先后顺序，
每次计时连续执行 3 次取均值、预热 5 次、80 对样本、对之间 sleep 4ms。函数结果对照通过。

| 电影数 | 原版 P50/P95/P99 µs | 候选 P50/P95/P99 µs | 配对差值中位数 µs（正为省时） | 候选更快对数 |
|---:|---|---|---:|---:|
| 30 | 52.84/70.396/75.448 | 53.254/70.239/86.863 | 0.744 | 41/80 |
| 100 | 159.44/174.907/251.318 | 156.205/176.015/214.684 | -0.763 | 40/80 |
| 500 | 668.944/744.333/776.034 | 663.198/760.937/871.764 | 22.555 | 48/80 |

候选运行 wall=1.513s，CPU=0.531s。完整结果见 redis_cache_candidate.json。
30 部的独立 P50 变慢，100 部配对中位差值为负，500 部 P95/P99 变慢；没有稳定收益证据，**不应用**。
因此没有本轮生产优化后的延迟或性能收益数据；不存在可报告的生产加速比例。

其他暂缓项目：
- 合并用户/目录 generation 网络往返：Fake get 不能证明真实 Redis RTT；需要核对 get_many 的 alias/并发冷 token 语义。
- 请求内跨调用复用 generation：可能把新业务结果写入错误代际，不能为了微基准牺牲失效边界。
- 去掉 payload 校验、分数恢复或关联预取：会削弱坏缓存降级、返回契约或引入真实 N+1，不做。
- 重排敏感性查询批量化：静态发现查询模式，但无真实 SQL/工作负载证据；不借性能任务改业务规则。
- 扩大 ORM payload 轻量化到 CB、负缓存、锁、异步失效、修改 TTL：改变范围或一致性，未实施。

已实施的是基准复现性修复：P0.1 旧版对照原先取 HEAD，基线提交后变成新版导致 1 项失败；
现在固定到 d78390dede4a0371a1e7991f06568d8d390fbdbd，保留原输入和断言。
静态绑定回归也改为相对该旧基线的完整生产差异，避免提交后 git diff 为空导致检查漏掉已提交代码。
这两项是测试可靠性修复，不是推荐算法或性能优化。

## 7. 可观测性工具与接入边界

新模块 myapp/utils/cache_metrics.py，仅标准库，无 Django/模型/后台线程/日志/HTTP 接口。
CacheMetrics(enabled=False) 默认关闭；显式 enabled=True 才计数和采样。
固定 family 白名单 hybrid/intent/kg/multi_recall/query_rewrite/explain/agent_recall/generation/other，
不允许 user_id、完整 key、query、Session token 等标签；未知 family 归 other。
事件支持 hit/miss/error/bypass，计时支持 read/write/generation_lookup/orm_rehydration。
每个固定标签最多保存 256 条样本，可配置 1～1024；snapshot 仅含计数、ns 和字节数，不保存内容。
payload_bytes 只接受已经序列化的 bytes/bytearray/memoryview，不额外 pickle/JSON，不检查或记录内容。
ObservedBusinessCache 为显式诊断 adapter，保持 backend 异常/返回值，已有 cache_access 决定回源；
指标时钟、计数、记录器失败不影响业务，也不会因此重复调用 backend。该 adapter **不能作为 Session backend**。
None 值按业务 miss 统计；显式 default 与真实缓存值同一对象时无法无额外查询区分存在性，记语义 miss。
None key 在 adapter 内计 bypass；现有 business_cache_* 会在到达 adapter 前跳过 None，若诊断这些调用，
需调用端显式记录 bypass。不能用 adapter 计数自动宣称整个应用的命中率。
完整 generation 计时包住 key builder，而非仅一条 get；ORM 恢复计时包住整个 restore。
future 真实序列化大小需在配置 serializer 产出的 blob 处记录，不允许重复序列化来量大小。
实例应按请求/一次诊断创建；没有全局聚合或多线程同步，不能当作跨进程或可靠服务统计系统。

本轮不强制接入所有业务入口；新增模块没有被生产模块导入，没有默认日志噪声和线上数据采集。
示例（仅后续独立诊断环境使用，不改生产 Session/cache alias）：

```python
metrics = CacheMetrics(enabled=True)
observed = ObservedBusinessCache(fake_backend, metrics, family="hybrid")
key = metrics.measure("generation", "generation_lookup",
                      hybrid_recommendation_cache_key, fake_backend, user_id, top_k)
# 使用已有业务 get/set 降级函数；需显式收集 key=None 的 bypass。
result = metrics.measure("hybrid", "orm_rehydration", restore_function, payload)
metrics.payload_bytes("hybrid", serializer_blob)
```

观测示例 hit/miss/bypass 各 1 次仅来自脚本合成演示，不是生产命中率。
未引入外部监控服务。参考 redis-observability 技能的指标区分原则；未执行 INFO/SLOWLOG/MEMORY 等命令。

## 8. 全部隔离回归与安全检查

导入审计：六个测试、微基准及三个 cache helper 均仅标准库直接导入；按文件路径加载安全 helper/fixture。
生产代码仅 AST 选择定义，Movie/事务/缓存/模型服务全用 Fake/Mock，不正常导入 myapp.views/agent/Django app。
新脚本使用 -S，并在运行前 guard 禁止 torch/transformers/sentence_transformers/faiss/numpy/tensorflow/
django/redis/MySQLdb/pymysql/neo4j；旧测试自身模型 guard 也保留。未导入模型权重或大数据。

| 套件 | 实际项数 | 最终结果 |
|---|---:|---|
| P0 | 14 | OK / exit 0 |
| P0.1 | 18 | OK / exit 0 |
| P1-A | 18 | OK / exit 0 |
| P1-B | 8 | OK / exit 0 |
| 阻断修复 | 23 | OK / exit 0 |
| P2 新增 | 24 | OK / exit 0 |
| 总计 | 105 | 全部通过 |

初始 P0.1 执行为 17 通过、1 失败，原因为 HEAD 测试 oracle 漂移；修复测试基线后 18 项通过。
全部六套随后逐个完整执行；微基准脚本增加原始采样和修正 A 引用组装后再验证 P2/P0.1，均通过。
新增覆盖默认关闭、clock/observer 错误、backend 异常保持、事件/计时/size、限界/标签/隐私、
代际故障绕过、用户/top_k 隔离、30/100/500 冷 payload/热恢复的等价字段/顺序/分数/关联、
坏 payload/已删电影/DB 异常、候选等价、冻结生产模块逐字节比较及原 81 项保留。
这不是完整 Django 测试、真实 Session/事务/SQL/Redis 集成或性能压测。

实际命令（新 Worktree 前台、逐个执行，无 nohup/后台/多并发）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p0_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p01_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p1a_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p1b_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_blockers_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p2_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S benchmarks/redis_cache_microbench.py --iterations 80 --warmup 5 --max-seconds 20
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S benchmarks/redis_cache_microbench.py --compare-candidate --iterations 80 --max-seconds 20
git diff --check
```

微基准 stdout 分别保存为 benchmarks 下的 baseline/candidate/validation JSON；只含合成数据统计与源哈希。
AST 检查采用 python3 -S 读取所有本轮修改/新增 Python 源并 ast.parse + 内存 compile，无 pyc 或应用初始化。
git diff --check 不包含未跟踪文件，另对新增 Python/JSON/Markdown 做全文件语法/行尾空白检查。

## 9. 后续真实集成测试方案（设计，未执行）

前置条件：独立测试主机/虚拟机或完全隔离环境、明确人工授权、无现有训练任务，独立 Redis 与 MySQL 测试实例。
只用合成数据/匿名夹具；不得连接生产库，不清理共享 Redis，也不在本机启动测试服务。
冻结测试 commit、Python/Django/django-redis/redis-py 版本、serializer/protocol/compressor、数据库版本、
关联规模、数据 seed、资源预算、网络拓扑和采样策略；独立命名空间/alias 不复用 Session。
模型、LLM、Embedding、FAISS、Neo4j 召回全部替换为确定性的 cassette/Fake；实际推理不执行。
先审计导入链：当前 views 顶层仍导入 torch/deepctr/RAG，AppConfig 也须单独检查。若测试 settings/
导入隔离不能保证不初始化模型，停止，不用设置 CUDA_VISIBLE_DEVICES 代替安全证明。
可先做缓存 helper + 真实 Movie ORM/Redis 的窄集成，使用专用 test settings/AppConfig 和明确注入的
生产函数定义；这仅证明函数/中间件集成，不能称为整个 MovieAgent API 的端到端验证。
完整 HTTP 端到端验收要在上述导入条件满足后，使用真实路由、Session/CSRF 中间件和 Fake 召回执行。

| 验证项 | 方法与记录 | 一致性/验收要求 |
|---|---|---|
| 冷/热端到端 | 第一次使用全新测试 user/top_k/namespace，后续同输入；force_refresh 单独测 | 返回 ID 顺序/分数/字段相同；计时含序列化/ORM/HTTP，不计 setup |
| 真实 Redis get/set | 记录配置 alias 的真实读/写分段；禁用重复序列化探针 | P50/P95/P99、超时/异常数与 serializer 版本齐全 |
| SQL 查询数 | connection execute wrapper 或 CaptureQueriesContext，含响应序列化/模板访问 | 查询数不随每部电影线性增加；不是只数 Python filter 调用 |
| 关联预取 | 在恢复后访问 genres.all/actors.all/directors.all；另测无关联/大量关联 | 访问不触发每部额外 SQL；关联过滤调用可能绕开预取须单独测 |
| key/内存 | 仅精确测试 key 的 MEMORY USAGE，记录 value blob 大小及受控数据规模 | 不能用 pickle 字节数替代 Redis 内存，不能用全局扫描/清理 |
| 业务命中率 | 独立业务计数，hit/(hit+miss)，bypass/error 单列；明确分母与窗口 | 不混淆 Redis 全库 keyspace stats（含 Session/generation）与业务命中率 |
| 延迟分布 | 串行先预热，计划至少 1000 合格请求，样本/预算经授权后执行 | 分别给冷/热/故障 P50/P95/P99；不得加总分位数 |
| 反馈失效 | 写测试反馈、等实际 on_commit，再读各 top_k/另一个用户/目录版本 | 写成功后本人所有参数旧 key 不可达，其他用户不变；回滚不失效 |
| 超时/不可用 | 仅对专用测试 alias 注入超时或拒绝连接，不操作共享实例 | 业务缓存回源；Session/记忆故障显式失败；partial_success 不重复 DB 写 |
| 多用户/admin | 实际 atomic/savepoint/admin 生命周期 + 首用户失效失败 | 后续用户仍尝试、失败作用域完整、恢复仅缓存；无 outbox 最终保证 |
| 并发梯度 | 仅独立环境获授权后按 1/2/4，再评估更高档，固定运行时间与资源上限 | 首先正确性：generation add 竞争、失效期间旧请求写入、删电影 |

只回收独立测试 namespace 的精确已记录 key；不执行 FLUSHDB/FLUSHALL/cache.clear/宽泛通配符。
服务端、数据库或业务的任何故障测试不得指向生产和共享 Session；阶段结束保留失败样本和配置。
目前 Session Cookie、真实 SQL 数、事务回调、Redis 序列化/网络、内存与并发梯度全部未执行。

## 10. GPU/CUDA 零干扰执行记录

仅新 Worktree 源码/文档/合成统计有写入；未操作原始目录或冻结分支工作区。
没有启动/停止/暂停训练进程，没有调用 GPU 设备检查、CUDA 初始化、模型/权重/Embedding/FAISS/Neo4j 构建。
没有 Django/Celery/Redis/MySQL/Neo4j 服务启动、容器操作、pip/apt、迁移、真实数据连接或全局缓存清理。
微基准 wall/CPU 已记录，串行有界主动休息；测试逐个前台运行。CUDA_VISIBLE_DEVICES 和线程变量只作第二层防护。
语义检索返回 INDEX_MISSING 后采用 rg 精确查找；未建索引或生成 embedding。
没有读取训练日志/数据/检查点，没有复制未跟踪训练产物，没有提交/暂存/推送/合并。

## 11. Git 差异与完整新增清单

当前 HEAD 保持基线，分支 perf/redis-benchmark。原 refactor/redis-cache Commit 不变。
跟踪修改仅两份测试（共 7 插入/3 删除）：

- tests/test_redis_cache_p01_isolated.py：固定旧版本对照，不再随 HEAD 改变。
- tests/test_redis_cache_blockers_isolated.py：检查已提交及本轮修改的全部缓存绑定。

未跟踪新增：

- benchmarks/redis_cache_microbench.py：串行有界脚本与候选对照。
- benchmarks/redis_cache_microbench_baseline.json：初始试测统计，保留其 A 浅复制构建说明。
- benchmarks/redis_cache_candidate.json：未应用候选的配对统计。
- benchmarks/redis_cache_microbench_validation.json：最终复测及原始 ns 样本，本文主数据来源。
- myapp/utils/cache_metrics.py：默认关闭的独立观测工具。
- tests/test_redis_cache_p2_isolated.py：24 项新增隔离测试。
- docs/REDIS_CACHE_P2_PERFORMANCE.md：本报告。

不修改 settings.py、memory.py、生产算法/召回、依赖、Schema、迁移、训练代码或前端。
新增文件只含源码、报告与合成微基准统计；提交前需人工复核。真实环境变量/密钥和训练产物不在差异中。

最终静态检查结果：16 个相关 Python 文件 ast.parse/内存 compile 通过；git diff --check 退出码 0。
7 个新增文件逐个检查行尾空白、UTF-8/非二进制、非符号链接；3 个 JSON 解析通过，
最终复测的全部 80 个原始 ns 样本可重新计算并精确得到所列 P50/P95/P99。
新增文件未命中私钥、AWS access key 或 GitHub token 格式扫描；完整新增清单与预期一致。
暂存差异为空，HEAD 仍是基线；冻结 Worktree 最终 git status --short 无输出，分支/HEAD 未变。

```text
 tests/test_redis_cache_blockers_isolated.py | 4 +++-
 tests/test_redis_cache_p01_isolated.py      | 6 ++++--
 2 files changed, 7 insertions(+), 3 deletions(-)
```

以上 git diff --stat 仅统计跟踪文件；其余 7 个新增文件已在上面完整列出，未用暂存制造统计。

## 12. 验收与后续条件

CPU-only 基准/观测工具与代码隔离回归具备阶段验收条件。没有证据支持进一步生产微优化，暂缓实施。
生产性能与真实 ORM/Redis/MySQL 集成尚无验收依据；后续按第 9 节在独立环境获授权后验证。
默认关闭工具不是部署后的完整监控系统；接入/采样开销须在独立环境确认再决定 rollout。
没有自动提交、推送或合并；完成后暂停等待人工验收。
