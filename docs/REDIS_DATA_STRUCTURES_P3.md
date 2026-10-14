# MovieAgent P3：Hash、ZSet、Streams 工程实践

日期：2026-10-13（Asia/Singapore）。工作树：MovieAgent-redis-structures；分支：feat/redis-data-structures。
基线：f4037f90acd1132ae68652cf0e3d7b6dec58bf3a。

> P3.2 当前结论：已新增显式、有界管理命令，单次运行复用一个 Consumer；195 项隔离测试通过。未执行真实命令，未暂存、未提交，等待人工验收。详见第 15 节。
> 以下 P3/P3.1 结论与复现结果保留为历史记录：
> P3.1 当前结论：同实例消费公平性阻断已修复，172 项隔离测试通过；跨重启状态与可靠性限制见第 14 节。未暂存、未提交，等待人工验收。
> 以下为保留的 P3 最终审查历史：
> 最终针对性审查：需要修复，未提交。发现 P1 消费者推进缺陷，见第 13 节。
> 前 12 节保留为实现阶段记录；现有 146 项仍通过，但不能替代新增故障场景审查。

## 1. 范围与执行边界

实现辅助反馈事件统计和热度展示，不参与推荐主链路。默认 REDIS_ANALYTICS_ENABLED=False。
不修改 MAAN、SKB-FMLP、RRF、召回/重排公式、用户画像算法、数据库 Schema 或迁移。
既有 django-redis String 缓存、Session、Generation、缓存失效与故障回源代码保持原状。

创建前 P2 Worktree 干净，基线匹配；目标路径/分支不存在，磁盘可用约 94GB。
通过禁用创建 hooks 的 worktree add 创建独立目录，未复制原始训练目录的未跟踪数据。
没有操作原始训练目录、其他 Worktree 文件、训练/GPU 进程、CUDA、模型、容器或服务。
没有依赖安装、数据迁移、真实 Redis/MySQL/Neo4j 连接、缓存清理、后台任务、暂存、提交或推送。
语义检索返回 INDEX_MISSING 后用 rg 定位源码，未建索引或生成 Embedding。

本报告区分：生产代码设计、Python Fake 合约测试、待执行的真实 Redis/Django 集成。
没有真实 Lua 执行、并发原子性测量、生产命中率、QPS、内存或性能提升数据。

## 2. 数据模型与覆盖范围

UserFeedback 已有 click/like/dislike/collect/skip/share 六种选择、数据库主键、movie_id 和 auto_now_add created_at。
本轮仅接入 agent_views.recommend_feedback_view 的新增记录；反馈入库字段和返回结构不变。
没有新增数据库去重：客户端重复提交生成两个不同数据库 ID 时，仍是两个独立反馈事件。

**统计的是成功处理的新增反馈事件次数，不是当前有效收藏数、唯一用户数或模型预测分数。**
删除或撤销反馈不会减计数；该统计口径与数据库当前状态不同，是有意的事件口径。
未接入 dislike/undislike 另一组视图、评分、收藏 toggle、管理端反馈新增/删除、直接 ORM/维护脚本。
后续若扩展覆盖，需明确事件类型/撤销语义与重复来源，不能直接认为本轮已覆盖所有用户行为。

事件只包含以下五个字符串字段，schema_version 固定为 1：

```json
{"event_id":"123","movie_id":"9","feedback_type":"click",
 "event_timestamp":"2026-10-13T12:00:00.000000+00:00","schema_version":"1"}
```

event_id 来自数据库记录 ID；movie_id 是电影 ID；UTC 时间来自数据库 created_at，不用投递时钟替代。
事件不含 user_id/用户名、Session token、来源 source、聊天/画像、原始输入或整条 ORM 序列化对象。
多余字段、未知事件类型、无时区时间、非法 ID/schema 在写入/消费之前拒绝。
数据库 ID 仍可由持有数据库权限者关联记录，不能宣称该数据不可重新识别。

## 3. 连接、命名与 TTL

实现位于 myapp/services/redis_analytics.py，仅标准库顶层导入，redis-py 延迟到显式启用且配置验证后导入。
配置关闭时不读取 URL、不导入 redis、不构造连接、不注册 analytics 提交回调；既有失效照常执行。
启用必须显式提供 REDIS_ANALYTICS_URL，要求独立 standalone Redis 7+ 连接和明确 /DB。
没有 Redis URL 回退值，也不调用 django-redis 的 get_redis_connection 或复用 cache 对象。

配置会比较所有 CACHES LOCATION 的 host/port/DB；localhost/127.0.0.1/::1 归一化，凭据和 redis/rediss 差异不能掩盖同 DB。
URL query/fragment 覆盖参数、缺少显式 DB 或已知同缓存 DB 配置拒绝。缓存 URL 无路径按 DB 0 检查。
静态 URL 比较不能识别任意 DNS 别名、代理或部署拓扑；运维仍须确认物理隔离与专用 ACL。
本轮没有解析配置目标或访问服务器。客户端 socket/connect timeout 各 0.5s，连接池最多 2，关闭 timeout 重试选项。
没有安装 redis-py；实际依赖版本、默认其他重试行为和服务器版本需在独立环境验证。

| 数据结构 | 完整逻辑 Key | 保留策略 |
|---|---|---|
| Hash | movieagent:analytics:v1:movie:MID:day:YYYYMMDD | 成功处理新事件后 EXPIRE 8 天；按 UTC 事件日分桶 |
| ZSet | movieagent:analytics:v1:hot:YYYYMMDD | 同上；每次成功的新事件更新滑动 TTL |
| 去重 String | movieagent:analytics:v1:dedupe:event:EVENT_ID | 9 天；保存 pending/done + 五字段摘要，不保存用户数据 |
| Stream | movieagent:analytics:v1:feedback:events | 无过期、无任意长度裁剪；原子容量准入，确认后安全删除 |
| Consumer group | feedback-stats-v1 | 专用单组；支持多个显式消费者，不允许第二组无条件共享删除 |

直接 redis-py 不自动继承 django-redis KEY_PREFIX，因此所有 key 都明确含 movieagent:analytics:v1。
本版本使用多个独立 key 的 Lua，仅支持 standalone；不支持 Cluster 的跨 slot 多 key 脚本。
TTL 是从处理开始计算的滑动保留时间，不是自然日末截止；迟到消息可以重新创建较早日期的桶。
重复事件不会刷新计数、热度或正常去重 TTL。去重记录过期/被淘汰后重放可能重新累计。
不承诺无限期限 exactly-once，也不承诺 Redis 重启、数据恢复或淘汰策略下仍保持幂等。

## 4. Hash 与 ZSet 服务

process(event) 通过 Lua 做 HINCRBY 与 ZINCRBY；counts(movie_id, day) 使用 HGETALL，并为未写字段返回 0。
top(day, top_k=10) 使用 ZREVRANGE WITHSCORES，top_k 限 1～100。
返回 movie_id 和 analytics_score，不命名为 rrf_score、MAAN score 或推荐模型分数。

默认展示权重：click=1、like=3、collect=5、skip=0、dislike=0、share=2。
可通过 REDIS_ANALYTICS_WEIGHTS 覆盖，限定六个事件、有限且非负的 0～100 数值。
权重未经训练或推荐验证；修改配置只影响之后处理的事件，不追溯重算已有榜单，也不破坏已存在的去重标记。
skip/dislike 仍计入 Hash，会产生零增量的 ZSet 成员；不是负向推荐权重。

同分时保留 Redis 的逆字节字典序：字符串 ID 9、2、10 同分时是 9、2、10，不是数值降序。
这是辅助展示规则，没有重新排序推荐列表。[ZREVRANGE 官方语义](https://redis.io/docs/latest/commands/zrevrange/)

## 5. Producer 与数据库提交时序

视图中唯一业务改动：把原 create 返回值绑定为 feedback，调用 schedule_feedback_analytics(transaction, feedback, settings)。
原 create 参数、atomic 边界和原失效 on_commit lambda 保留。

1. 数据库事务内 create；关闭功能时直接跳过 Analytics，没有新增 Redis 调用。
2. 启用时捕获已保存记录的五字段快照，先登记 Analytics on_commit，再登记原推荐缓存失效回调。
3. 最外层提交成功才尝试投递；内层提交不提前投递，保存点/外层回滚丢弃对应回调。
4. Analytics 回调捕获配置、连接、脚本及日志处理失败，不向后续缓存回调传播异常。
5. Analytics 先尝试，随后缓存失效执行原逻辑；即使后者失败，前者不因非 robust 回调截断而遗漏。
6. 原缓存失效失败继续返回原 partial_success/database_committed 语义，不被 Analytics 吞掉。

schedule 返回 True 仅代表注册，不是送达确认。HTTP success 仍仅说明原反馈业务成功，未宣称事件已送达。
投递错误日志含操作、数据库 event_id、固定错误原因码/异常类型，不输出 URL、凭据或原始错误全文。
配置禁用发生于注册后会记录 disabled_after_commit；不伪造成功投递。
若日志处理器自身故障，保证主业务继续，但日志也可能丢失；没有可靠的失败账本。

**无 Outbox，属于 best-effort 投递**：提交后进程退出、网络错误、容量满均可能丢事件。
超时可能是“已 XADD 但未收到应答”，不能断言没写；即使重投形成重复 Stream 条目，按 event_id 去重。
不自动重复数据库写入，不提供自动补发/后台 Worker，不宣称最终一定送达。

## 6. Stream 长度与 ACK 保留策略

APPEND_LUA 在单次原子执行中检查 XLEN 与 capacity，然后 XADD。
默认最大未回收条目数 100000，可配置 1～1000000；满时明确报 ANALYTICS_BACKLOG_FULL，并由投递回调记录失败。
不会通过 MAXLEN、XTRIM、Stream TTL 或全局清理删除未确认消息。
容量准入是拒绝新事件的 best-effort 边界，不是可靠的背压/Outbox；拒绝后数据只在原数据库，没有自动重试。

消费统计成功（或确认重复）后执行 XACK。处理失败不 ACK；ACK 失败保持 Pending，后续重投去重。
ACK 成功后用 PRUNE_LUA 原子核对：只有当前专用单组、该条目没有 Pending 引用，才 XDEL 精确 ID。
多组出现时不删除，且新一轮 ensure_group 拒绝继续；不能暗中丢弃其他组未消费消息。
确认后的条目用于短暂传输而非长期审计归档，本轮没有提供长期事件历史查询。
删除出错在 report.prune_failed 和日志记录；已 ACK 条目可能留在 Stream 并占容量，需后续授权的精确维护。

XACK 只移除组的 Pending 引用，本身不移除 Stream payload；任意裁剪可能删除未 ACK 的 payload。
[Redis Streams 官方说明](https://redis.io/docs/latest/develop/data-types/streams/)

## 7. 有界消费、Pending 恢复与去重

FeedbackAnalytics.consume_once 是独立显式调用的消费服务，不在 Django 启动或请求中自动消费。
每次最多 100 条、默认 10s/最多 30s；先 XGROUP CREATE 0-0 MKSTREAM（已有组仅接受 BUSYGROUP），检查独占组。
先 XAUTOCLAIM 超过 min_idle_ms 的 Pending，再按剩余额度 XREADGROUP > 获取新消息。没有 BLOCK、循环或守护进程。
默认最小 idle=60000ms，避免把正常短批次过早转交；调用方可传回上次 reclaim_cursor 继续扫描。
XAUTOCLAIM 返回已删除 payload 的 Pending ID 时，报告 deleted_pending 并记录异常，不把它当作处理成功。
[XAUTOCLAIM 官方文档](https://redis.io/docs/latest/commands/xautoclaim/)

时间预算在消息边界检查，不是硬实时终止器；网络重试、连接超时或单条命令耗时仍须真实环境验证。
已拉取而因预算未处理的条目留在 Pending，report.deferred 记录数量；失败 ID 在 failed_ids，范围最多本批次 100。
processed/duplicates 表示统计脚本结果，acked 表示确认结果；ACK 失败时 processed 可能大于 acked，不能将其误读为数据库写入次数。

去重 key 全局按数据库 event_id，摘要包含规范化 UTC 时间、movie/type/schema。
相同 ID/相同内容已 done 返回 0 并 ACK；同 ID 不同内容报 ID_CONFLICT，不 ACK。
9 天窗口外或 key 淘汰后再投可能重计；Redis 专用实例建议保留足够内存、不淘汰去重/待处理数据，并由后续测试确认。

## 8. Lua 原子更新与错误边界

UPDATE_LUA 用 KEYS 明确传入 Hash、ZSet、去重 key，ARGV 只传字段/权重/ID/TTL/摘要。
先检查三类 key 类型、已有 marker、整数计数和有限热度，Redis 7 的 acl_check_cmd 预检查全部写命令权限。
设置 pending marker → HINCRBY/ZINCRBY → Hash/ZSet EXPIRE → done marker，整段脚本无其他客户端命令交错。
正常成功一次更新三者；并发重复消息由同一脚本看到 done 后直接跳过，无“先判断后自增”的客户端竞态。
[Redis 脚本原子执行说明](https://redis.io/docs/latest/develop/programmability/eval-intro/)

**Lua 的原子执行不等于运行时失败回滚。** 已写数据不会自动恢复；预检查只能消除常见确定性类型/数值/权限错误。
若在预检查后发生非预期资源/服务器错误，可能出现局部统计及 pending marker；下次回放拒绝再次累计，也不 ACK。
这时 Hash/ZSet 暂时不一致，需要人工核对，不能宣称所有故障下强一致或自动恢复。
局部更新还可能尚未设置最终 TTL；需与 Pending 标记一起检查，不能依靠成功路径 TTL 自动清理未知状态。
[Redis 事务及不回滚说明](https://redis.io/docs/latest/develop/using-commands/transactions/)

不确定状态的恢复流程（仅后续独立环境授权后）：记录 failed_ids/固定原因 → 查对应 event/marker/聚合状态 →
按原数据库事实核对并重建受影响 Analytics 桶或补偿；确认后再将 marker 完成及 ACK。
本轮不提供自动删 marker/重新累计命令，不反复执行数据库反馈写入。
marker 过期后不确定状态也可能再次累计；需在保留窗口内人工处理，长期故障不能靠重投解决。

## 9. 内存与积压风险

Raw 字段名和值合计最大约 136 字节/事件（两种 19 位 ID、7 字节类型、32 字节时间和版本），
100000 条原始字段文本最多约 13.6MB；这是字段文本长度上界估算，不是 Redis 内存上下界或 MEMORY USAGE 实测。
另有 Stream 节点/listpack、PEL、消费组、消费者元数据和字符串开销；ACK 后 XDEL 也不保证立即按条释放全部内存。
去重 key 数约为保留窗口内已处理的唯一事件数；没有由 Stream capacity 限定九天累计 marker 数量。
Hash 数约按电影/日期桶，ZSet 成员约按活跃电影/日期；迟到事件会延长旧日期桶寿命。
消费者名称过多也会增加组元数据，不用随意删除带 Pending 的消费者。

长时间未消费、毒消息、错误类型/ACL、ACK 后删除失败会占容量；到容量上限只拒绝新事件，保留待处理数据。
建议后续独立环境观察 XLEN、XPENDING、lag、deleted_pending、未确认时间、marker 数量和精确 key 内存。
没有真实内存测量、性能压测、广泛扫描或清理。物理数据库隔离仍不自动隔离共享实例的全局内存淘汰和故障。

## 10. 隔离测试与回归结果

执行前审计标准库、文件路径加载器及 AST 选择范围。新服务唯一非标准库 import redis 位于配置启用后的工厂分支。
测试全部注入 Fake 工厂或显式 False；不正常导入 Django app、views、ORM/AppConfig 或模型模块。
生产反馈函数由 AST 选择后注入 FakeTransaction/FakeFeedbackDB、实际 Analytics 帮助函数及明确 Fake settings。
静态绑定检查扩展为检查真实 Analytics 模块导出及生产导入，不能用夹具注入掩盖缺失生产符号。
原有测试断言和用例数量保留；P0 fixture 仅增加实际帮助函数与默认关闭配置。

| 套件 | 项数 | 最终结果 |
|---|---:|---|
| P0 | 14 | OK / exit 0 |
| P0.1 | 18 | OK / exit 0 |
| P1-A | 18 | OK / exit 0 |
| P1-B | 8 | OK / exit 0 |
| 阻断修复 | 23 | OK / exit 0 |
| P2 | 24 | OK / exit 0 |
| P3 | 41 | OK / exit 0 |
| 总计 | 146 | 全部通过 |

新增覆盖各计数字段/TTL、展示权重与同分、XADD/组/消费/ACK、处理/ACK 失败、Pending/失 payload 恢复、
重复及冲突 ID、单 eval/脚本预检查、非预期局部更新不重复计数、去重过期风险、第二组与容量保护、
时间预算、零连接、URL/权重约束、隐私/UTC、提交/回滚/嵌套保存点、投递故障与日志故障隔离、导入绑定及算法不变。

首轮新增 34 项中 5 项夹具错误：把已有 dict 型 Response 当作 response.data；修正字典访问后保留全部断言。
随后加入故障日志/失败 ID 等用例，最终七套完整顺序执行，41 项新增及原 105 项通过。
FakeRedis 是脚本预期语义的 Python 模型，**没有解释 Lua，也不能证明真实 Redis 原子性、ACL、TTL、PEL 或竞争行为**。
静态断言只证明单 eval 调用/脚本文本顺序；真实功能验收必须补第 11 节验证。

实际执行环境与命令（前台逐个，无后台/并发）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p0_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p01_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p1a_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p1b_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_blockers_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p2_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_structures_p3_isolated.py
git diff --check
```

## 11. 后续真实集成方案（未执行）

需要人工授权的独立测试环境、专用 standalone Redis 7+、明确 ACL/持久化/内存策略、redis-py 版本确认和合成数据。
不使用生产库或共享 Session DB，不修改原机器配置，不清理共享 Redis，不执行模型。
先仅按文件路径载入服务，显式 Fake settings + 专用 client，验证 Lua/Streams；再做真实事务/中间件测试。

关键验证：实际 Lua 类型/ACL/计数边界前置失败无写入；同 ID 并发消费与 XACK 应答丢失不重计；
server OOM/错误时 pending marker 的实际行为；两个组竞争/重创建、ACK 后 prune 保留、容量准入竞态；
XAUTOCLAIM 扫描游标、idle、删除 payload 返回值；Hash/ZSet/marker 过期、淘汰和重启恢复；
真实 Django atomic/savepoint/on_commit、Analytics/缓存回调时序和 HTTP partial_success 兼容；
专用 URL 与 ACL 边界、超时/重试上限、UTC 日桶/同分排序及内存积压测量。

重要启动风险：现有 MyappConfig.ready 会调用 views.warmup_all_systems。
因此没有新增需要当前 manage.py 运行的管理命令，仅提供消费服务；本轮也未运行 Django setup。
后续真实 Django 集成需专用不执行预热的测试 AppConfig/settings 并审计导入链，否则跳过。
不能仅依赖 CUDA_VISIBLE_DEVICES 来证明真实 Django 启动安全。

消费服务调用示意仅用于上述独立环境，未在本机执行：

```python
service = analytics_from_settings(explicit_test_settings)  # 开关 True、专用 URL、缓存连接清单
if service is not None:
    report = service.consume_once(consumer='manual-p3', count=100,
                                  min_idle_ms=60000, reclaim_cursor='0-0', max_seconds=10)
    # 根据 failed_ids/deferred/deleted_pending/prune_failed 人工处理，下一轮传回 reclaim_cursor。
```

没有 Outbox、可靠失败队列或常驻 Worker；增加这些机制属于后续授权范围。

## 12. Git 与验收

跟踪修改：

- movie/settings.py：默认关闭、显式 URL、展示权重和容量配置；原缓存/Session 配置不变。
- myapp/agent_views.py：只有新增反馈入口注册提交后事件；缓存失效 lambda、权限/POST/CSRF 保留。
- tests/test_redis_cache_p0_isolated.py：安全文件加载与真实导入绑定检查扩展，不删除断言。

新增未跟踪：

- myapp/services/redis_analytics.py：专用连接、三段固定 Lua、Hash/榜单/Stream 及有界消费服务。
- tests/test_redis_structures_p3_isolated.py：41 项标准库/Fake 测试。
- docs/REDIS_DATA_STRUCTURES_P3.md：本报告。

原推荐算法/模型、缓存 Key/Generation、P2 数据、依赖、数据库/迁移、前端和训练文件未改。
最终 18 个相关 Python 文件 AST/内存 compile 通过，git diff --check 退出码 0；
3 个新增文件的 UTF-8、非二进制/非符号链接、行尾空白、本机绝对路径及私钥/token 模式检查通过。
AST 比较确认原 P0 测试函数体不变；反馈 create 参数、atomic 调用和原失效 on_commit 完整保留。

```text
 movie/settings.py                     | 7 ++++++-
 myapp/agent_views.py                  | 5 ++++-
 tests/test_redis_cache_p0_isolated.py | 9 ++++++++-
 3 files changed, 18 insertions(+), 3 deletions(-)
```

上述 diff --stat 不包含新增文件，完整 3+3 文件清单见本节；没有为了统计执行暂存。
没有 auto-start、真实服务运行或全局 Redis 操作。分支 HEAD 保持基线；没有暂存/提交/推送/合并。
代码与 Fake 回归具备阶段审查条件；真实 Lua/Redis 与事务集成尚待授权验证，不能作为生产强一致性验收。
完成后暂停等待人工确认。

## 13. 最终针对性审查与提交阻断

本轮仅审查、运行 CPU-only 隔离回归及追加报告，没有修改生产源码、现有测试或 Lua。
没有执行真实 Redis/Lua/Django、创建提交或操作训练目录、其他 Worktree。

### R1：P1 消费者推进缺陷（本轮提交阻断）

位置：myapp/services/redis_analytics.py，FeedbackAnalytics.consume_once，恢复批次与新消息读取分支。
XAUTOCLAIM 允许一次认领 count 条，只有 len(recovered) < count 才执行 XREADGROUP >。
因此满批恢复消息占用全部额度；如果这些消息持续失败，每次调用可能只重试旧消息而不读取健康新消息。
continue 只让已获取批次内的下一条继续执行，没有保证新消息获取机会。

复现场景：先制造不确定更新的 Pending（有效事件留下 pending marker，不自动重计），
再投递不同 ID/不同电影的健康事件；每轮间隔 61 秒，超过默认 60 秒 idle 阈值。
恢复旧消息时它们再次失败，健康消息保持未读取。对 count=1 和默认 count=100 均重复两轮复现。

```text
count=1   clock=61   claimed=1   failed=1   acked=0 fresh_reads=0 fresh_unread=True
count=1   clock=122  claimed=1   failed=1   acked=0 fresh_reads=0 fresh_unread=True
count=100 clock=61   claimed=100 failed=100 acked=0 fresh_reads=0 fresh_unread=True
count=100 clock=122  claimed=100 failed=100 acked=0 fresh_reads=0 fresh_unread=True
```

诊断只执行实际 Python 消费控制流配合已有 FakeRedis；没有执行 Lua，Fake 中的中途错误用于设置一个需要人工处理的合法失败状态。
既有 Fake 的 xautoclaim 忽略 cursor，所以另外核对官方语义：本场景 PEL 正好等于 count，扫描末尾返回 0-0。
Redis 7.2 源码在循环结束后检查迭代器 EOF，确认为上述返回值；新消息未交付，不属于 PEL，不能令恢复游标自动跳出该循环。
[XAUTOCLAIM 文档](https://redis.io/docs/latest/commands/xautoclaim/)，[Redis 7.2 实现](https://github.com/redis/redis/blob/7.2/src/t_stream.c#L3267)。

这不是“所有调用时序下都永远阻塞”：在 idle 阈值以内立即再调用，可能有新消息额度；
但按每分钟或更长间隔运行时，没有代码层面的新消息推进保证。不能依赖操作员猜测调用频率解决调度缺陷。
影响范围仅为启用的辅助 Analytics；默认关闭、反馈数据库写入与推荐主业务不受影响，未发现 P0 GPU/主业务安全问题。
它仍是新的 P1 代码缺陷，不符合本轮“只有明确可靠性限制、没有新的代码缺陷才提交”的条件，所以不暂存/提交。

建议后续最小修复：在总条数/时间预算内为新消息保留明确机会，或有界轮换恢复与新消息；
同时定义 count=1 的公平行为。不得为解阻塞错误 ACK 毒消息、删除 Pending、裁剪 Stream 或重复数据库写入。
本轮未实施该修复，也未引入 dead-letter、Worker、Outbox 等新功能。

### 其他实际保证与限制

- Lua 更新顺序：预检查 → pending 去重标记 → Hash → ZSet → 两个 EXPIRE → done 标记。
  正常完成且标记仍存在时，同 ID/内容重投不重计；并发消费者仍可能处理同一事件，但脚本串行判断防止正常重复累加。
  Lua 非预期中途错误不回滚；pending 标记阻止窗口内盲目重算，却不能自动修复已发生的局部修改。
  必须按 event_id/failed_ids 核对数据库事实和受影响桶，不将人工恢复误称为自动最终一致性。
- ACK：只有 process 返回 0/1 后才调用；处理失败不 ACK。XACK 并不验证当前消费者所有权，
  所以旧消费者与重新认领的消费者可能发生 ACK 竞争，后者可能收到 0 并被报告失败，即使另一个消费者已经确认。
  这是诊断及租约限制，不等于统计一定重复；需真实并发集成验证。
- 九天窗口：当前没有旧消息拒绝/隔离策略。marker 过期后，旧 Pending/重复事件会按现有代码重新处理，
  可能重新创建历史日期桶或重复累计；不应把这称为永久幂等。窗口外恢复应先人工对账，而非无条件自动重投。
  未实现自动 age gate；本轮不擅自新增，保留为明确可靠性限制。
- 容量：XLEN 与 XADD 在同一 Lua 中，没有客户端“检查后写入”的并发竞态。
  仅在同一 standalone Redis、全部 producer 使用同一脚本且相同 capacity 时，可从 Redis 串行脚本语义推导条目数上限；
  外部直接 XADD、不同 capacity 配置、已有超限数据不在保证内。它不限制去重 key 数、PEL 或实际内存。
  Fake 单次 eval 只能验证调用形态，不能充当真实并发证明。
- Producer：event_id 取数据库主键，时间/字段快照稳定。Analytics 回调先注册且捕获自身异常，
  原缓存失效回调未修改；注册失败、配置错误、连接/容量错误也不导致再次写反馈。
  没有 Outbox：提交后退出、响应丢失、容量满可能丢事件或产生未确认送达状态；没有最终送达保证。
- 功能/隐私：False 分支在任何 URL/连接动作前返回；只有显式 URL 才创建专用 client，明显同缓存 DB 被拒绝。
  DNS 别名/代理的物理隔离无法靠字符串比较证明，仍需部署验证。Hash/ZSet 不作为推荐输入，事件仅五字段，日志不输出原始异常/凭据。

### 测试审查与补验项目

本轮重新审计七套测试及 helper 导入链：标准库顶层、按文件加载安全 helper，唯一 redis import 在启用工厂分支。
Fake 工厂/关闭配置与 AST 注入避免真实 Django、数据库或模型初始化；新增诊断额外在加载夹具前安装禁止导入 guard。
实际重新执行：14+18+18+8+23+24+41 = 146 项全部 OK，七个退出码为 0。
新增诊断成功复现 R1；它是独立故障探针，不加入“通过项数”，也不是证明消费者已满足新消息推进要求的测试。

缺口：Fake 不执行 Lua，不模拟所有 key 类型、ACL、ZSet 非有限值、OOM 或真实 Redis 事务故障；
xautoclaim 没有真实 COUNT×10 扫描/数值 ID 排序/游标模型，没有多消费者并发、ACK 竞争与容量并发测试；
现有 Pending 成功恢复测试没有覆盖满批持续失败消息与健康新事件并存。
测试符号检查先验证生产模块导出/绑定，未发现被注入掩盖的新增缺失符号；原 P0 用例未删除/跳过。

后续须先补 R1 的隔离回归，再在获授权的独立 Redis 7+ 环境验证实际 Lua、ACL、错误状态、过期/淘汰、
Pending 游标/同 ID 并发、ACK/Prune、统一容量下并发 producer 和真实 on_commit 时序。
当前禁止启动 AppConfig 预热或真实服务；这些集成没有执行。

最终结论：需要修复 R1 后重新审查；没有新 Commit SHA。仍为原来的 3 个跟踪修改和 3 个新增文件，暂存区为空。
本轮仅追加本报告；生产源码与此前 P3 实现相同，基线 HEAD/分支未变。暂停等待人工确认。


## 14. P3.1 消费者公平调度修复与回归

### 14.1 根因与修改范围

第 13 节 R1 是历史审查记录，本节说明其修复状态，未删除或改写历史复现结果。
旧 consume_once 先以整个 count 执行 XAUTOCLAIM；恢复满批时完全跳过 XREADGROUP >。
合法但不确定更新的 Pending 或毒消息可持续占满预算，导致新事件始终未交付。
旧实现每次默认使用 0-0，实例内也没有恢复游标及轮转状态。

本轮只修改独立 analytics 服务的实例状态/consume_once、P3 FakeRedis 与时间预算夹具，
新增 P3.1 隔离回归及本节报告。不修改 Lua、producer、事件 schema、TTL、容量机制、反馈视图或配置。
既有 P3 对 settings、agent_views、P0 夹具的修改保留；P3.1 没有再次改动它们。
没有引入 Redis 调度 key、Outbox、死信队列、数据库迁移、后台任务或新推荐行为。

### 14.2 新旧消费流程

| 项目 | 修复前 | P3.1 |
|---|---|---|
| count>=2 | Pending 可取完整 count，新消息只用余量 | 首类最多 ceil(count/2)，次类使用剩余预算；两类都有保留机会 |
| count=1 | 每次先 Pending，满额不读新消息 | 相邻调用交替 pending/fresh 优先；首类空时尝试另一类 |
| 一类空/不足 | 新消息只能使用 Pending 的余量 | 首轮余量给第二类；仍有余量时仅追加一次第一类获取 |
| 获取次数/条数 | 一次 claim，最多一次 fresh read | 至多三次非阻塞获取，合计返回条数不超过 count |
| 处理顺序 | 全部恢复消息在新消息之前 | 按本轮优先级在两类批次间交错处理 |
| 游标 | 缺省每次 0-0 | 保存并沿用 Redis 返回 cursor，EOF 0-0 下轮从头；支持显式覆盖 |
| 错误/ACK | 成功或重复才 ACK；失败保留 Pending | 保持同样规则，单条失败继续本批健康消息，读失败抛出 |

分配是获取/尝试机会保证，不保证每次 Redis 成功，也不保证毒消息能被处理成功。
max_seconds 为合作式时间预算：每次取消息和处理前检查 monotonic；到期不启动下一步。
它不是可以强行中止 Redis 调用或 Lua 的硬截止时间；配置客户端仍使用原 0.5 秒连接/读取超时。
XREADGROUP 不传 BLOCK（尤其不能传 BLOCK=0）；不存在无限读取循环、后台调度或无限等待分支。
时间到期时已交付但未处理的消息仍在 Pending；后续按 idle 门槛和恢复游标处理。
如果首条处理耗尽时间，另一类本轮可能只获取而未处理；下一轮优先级仍轮换。

### 14.3 状态、生命周期与重启边界

每个 FeedbackAnalytics 实例保存 _next_lane 和 _reclaim_cursor，面向单调用方串行使用；
不是跨线程安全调度器、Redis 租约或持久化协调状态。返回报告新增 fresh 与 next_lane 字段，
原 processed/duplicates/acked/failed/failed_ids/deferred/claimed/reclaim_cursor 等字段保留。
现有位置参数顺序保留；reclaim_cursor 默认 None 表示续用本实例游标，显式 0-0 仍表示从头扫描。
新增可选 first_lane 用于调用方显式续接轮转状态。Redis 返回 bytes cursor 会解码，非法 cursor 拒绝。
成功获取后立即保存返回游标，即使该批处理失败也不倒退；下一优先级在轮询开始设置，读失败也轮换。

调用方在另一个实例续接成功返回的状态可使用：

```python
report = analytics.consume_once(count=1, **previous_state)
previous_state = {
    "reclaim_cursor": report["reclaim_cursor"],
    "first_lane": report["next_lane"],
}
```

这段示例不实现持久化。调用方若只保存在内存，整个进程重启仍会丢失；发生异常未返回报告时，
外部检查点也可能滞后。实例内已保存成功获取的游标，但本轮没有提供可靠的异常/崩溃检查点协议。
没有新管理命令或消费者启动钩子；当前仓库没有服务之外的生产消费调用方，生命周期由显式调用者控制。
尤其不能使用 manage.py 启动当前 AppConfig 预热链路，未执行该链路。

**实际保证：** 同一串行长生命周期实例持续轮询时，count=1 交替给两类机会；count>=2 每轮分配配额，
游标向前推进使后续 Pending 也有被扫描机会。假设调用持续、Redis 返回且时间预算足以推进，不承诺处理成功。
重新实例化时显式续接两项状态，可保持该调度序列（只验证正常完成后的状态传递）。
**未保证：** 每次重新启动且不续接状态的 count=1 仍从 Pending 开始，可再次饿死新消息；测试明确复现这一限制。
count>=2 无状态重启仍给新消息配额，但反复重置游标可使 PEL 后部消息长期未扫描，也不能宣称全体 Pending 的跨重启公平性。
短生命周期人工调用宜使用 count>=2；要获得 PEL 全扫描和 count=1 连续轮转，调用方须可靠续接游标及 phase。

若需要自动跨重启/多进程公平性，应另行设计专用 analytics 调度状态、原子轮转和游标更新协议，
评估多消费者覆盖/并发抢占、ACL、状态 TTL、崩溃窗口与一致性。不能简单 GET/SET phase 就声称解决。
本轮未新增这样的 Redis 持久化状态；需人工另行授权与真实 Redis 集成验证。

### 14.4 针对性验证结果

均为生产 Python 控制流 + FakeRedis/Mock，**不是实际 Redis/Lua 原子性、并发或吞吐量验证**。

| 场景 | 实际隔离结果 |
|---|---|
| count=1，持续失败 Pending + 每轮新增事件 | 同一实例四轮 ACK 0/1/0/1；失败 1/0/1/0；实际读取顺序 claim/read/claim/read |
| count=2，两类持续存在 | 两轮均 claimed=1、fresh=1、failed=1、acked=1；第二轮 fresh 优先 |
| count=100，100 条毒 Pending + 100 条新消息 | 单轮 claimed=50、fresh=50、failed=50、acked=50；新消息按原顺序 ACK |
| 无 Pending / 无新事件 | 未使用的配额给另一类；测试证明全部 count 可被健康一类使用 |
| 游标 | 按 Redis 返回值前进，包括数值 ID 9/10 的排序、COUNT*10 扫描上限、EOF 0-0 与下一轮重用 |
| 重新实例化 | 显式状态续接仍为 0/1/0/1；未续接 count=1 反复失败、新消息不前进，作为限制验证 |
| 单条/ACK/Redis 读失败 | 毒消息不 ACK；其他健康消息继续；ACK 失败记录失败；读异常直接抛出，无虚假成功 |
| 重复事件 | Pending 和新投递的同 ID 只累计一次；重复投递确认后 ACK |
| 时间预算/空批 | 到期消息保留 Pending；空批最多三次获取；总 fetch/attempt 不超过 count；不传 BLOCK |

P3 FakeRedis 修正：数值 Stream ID 排序、遵循输入 cursor、COUNT*10 有界扫描与 EOF 返回值。
它仍只近似 Redis 的顺序读取契约，不执行 Lua，不模拟线程/多客户端并发、完整 Redis 类型检查、ACL 或 OOM。
原 P3 的 41 项测试、断言、失败与去重窗口用例全部保留。
唯一原测试夹具调整是时间预算：由固定四个时钟返回值改为首条 process 执行后推进假时钟；
原 acked=1、deferred=1、Pending 数=1 三条断言完整保留，没有放宽验证或跳过失败用例。
P3.1 新增 26 项测试，涵盖上述场景、生产符号绑定负向测试及原 41 项数量/断言结构检查。

首次执行 P3.1 时，新静态检查器在当前 CPython symtable 的内联推导式上把 local+global 标记误报为未绑定符号；
已按真实作用域排除局部参数/循环变量，并保留删除 import time 必须报错的负向断言。
没有为通过该测试注入生产缺失符号或修改生产导入。修正后整套 26 项通过。

### 14.5 安全导入审计与完整回归

运行前用 ast.parse 检查八套测试的 import、按文件加载和 exec/compile 路径。
cache_keys/cache_access/cache_metrics/analytics/微基准工具仅标准库顶层；analytics 唯一 redis 导入在延迟配置工厂中。
所有工厂调用用 Fake 注入或关闭开关；cache_mutations 等 Django 模块仅提取指定 AST 定义，未导入应用。
P3.1 在加载任何夹具之前安装拒绝 torch/transformers/sentence_transformers/numpy/faiss/tensorflow/
redis/django/MySQLdb/neo4j 的 import guard；全套没有真实数据库、模型加载、启动服务或后台任务。
原 P0 中的摘要子进程测试只执行标准库 key 模块；P2 只执行既有有界小样本 Mock 测试，没有新增性能压测。

在独立工作树前台逐个执行下列命令前缀加文件名，没有后台或并发：

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S
```

| 文件 | 项数 | 结果/退出码 |
|---|---:|---|
| tests/test_redis_cache_p0_isolated.py | 14 | OK / 0 |
| tests/test_redis_cache_p01_isolated.py | 18 | OK / 0 |
| tests/test_redis_cache_p1a_isolated.py | 18 | OK / 0 |
| tests/test_redis_cache_p1b_isolated.py | 8 | OK / 0 |
| tests/test_redis_cache_blockers_isolated.py | 23 | OK / 0 |
| tests/test_redis_cache_p2_isolated.py | 24 | OK / 0 |
| tests/test_redis_structures_p3_isolated.py | 41 | OK / 0 |
| tests/test_redis_structures_p31_isolated.py | 26 | OK / 0 |
| 合计 | **172** | 全部通过 |

额外执行八套测试/所有改动或未跟踪 Python 文件的 AST 语法检查，不用 py_compile，不产生字节码。
生产符号检查直接扫描实际 analytics 模块的 symtable，缺失 time import 的负向对照正常拒绝；
原生产辅助函数绑定检查和缓存/模型/算法/微基准数据与基线逐字节对照仍通过。
git diff --check 退出码 0；未跟踪文件另行验证 UTF-8、无 NUL/符号链接、无行尾空白。
原跟踪源码已有空白保留，只检查新增差异；不对生产文件执行格式化。
常见私钥、GitHub/AWS token 模式及本机绝对用户路径扫描未发现候选；不把静态扫描说成全面密钥审计。

### 14.6 可靠性限制与真实集成待办

第 13 节其他风险仍然成立，P3.1 没有弱化它们：

- 数据库提交后 Stream 投递是 best-effort；进程崩溃、配置/连接/容量失败可丢事件，没有 Outbox 最终送达保证。
- Lua 原子隔离不等于异常回滚；Hash/ZSet 更新途中运行时错误可能留下部分写入，pending 标记需要人工核对/修复。
- 去重仅九天；旧 Pending 超窗或 marker 被淘汰后可能重复统计，当前没有自动 age gate/隔离队列。
  超过安全恢复窗口的旧消息须暂停相关人工恢复调用，按 failed_ids/event_id 对账并由操作员隔离处理；不能无条件盲重投。
- 不承诺 Exactly-Once；claim 不禁止旧消费者继续处理，ACK 无所有权栅栏，实际并发竞态未验证。
- 容量的原子准入仅针对同一 standalone Redis、相同脚本/上限的 producer；外部 XADD/不同上限不在范围内。
  长期 Pending/毒消息仍占用 Stream 容量；公平调度不是清空积压或修复毒消息的机制。
- 默认关闭、独立 URL 和明确 namespace 保留；DNS/代理物理隔离、Redis 7+ Lua/ACL/淘汰/OOM 行为尚未验证。

后续只在独立环境、人工授权后测试真实 XAUTOCLAIM cursor/COUNT/删除 PEL 项、非阻塞 XREADGROUP、
quota/时间预算、双消费者 Pending 抢占、ACK/Prune、Lua 类型/ACL/部分更新、九天超窗人工恢复和 producer 容量并发。
真实 Django on_commit、Session/认证和数据库事务集成仍待验证；本轮没有执行真实服务或 Lua。

### 14.7 Git 完整差异与提交条件

当前分支 feat/redis-data-structures；HEAD 仍为 f4037f90acd1132ae68652cf0e3d7b6dec58bf3a。
暂存区为空，不提交、不推送、不合并、不修改 main/冻结分支，没有访问原始训练目录或其他 Worktree。
完整基线差异包含 **3 个跟踪修改 + 4 个未跟踪新增文件**：

| 文件 | 状态 | 实质差异 |
|---|---|---|
| movie/settings.py | M，原 P3 保留 | 追加默认关闭、专用连接、权重和容量配置 |
| myapp/agent_views.py | M，原 P3 保留 | 新增反馈提交后注册 best-effort producer；原失效回调不变 |
| tests/test_redis_cache_p0_isolated.py | M，原 P3 保留 | 安全 helper 实际加载/绑定与关闭配置夹具；原 P0 用例不变 |
| myapp/services/redis_analytics.py | ??，P3 + P3.1 | 独立 Hash/ZSet/Streams 服务；本轮只修公平消费与游标/phase |
| tests/test_redis_structures_p3_isolated.py | ??，P3 + P3.1 | 原 41 项保留；Fake cursor 修正和基于行为的时间预算 |
| tests/test_redis_structures_p31_isolated.py | ??，P3.1 新增 | 26 项公平性/有界性/异常/状态/符号绑定隔离测试 |
| docs/REDIS_DATA_STRUCTURES_P3.md | ??，历史保留 | 当前结论提示及本节 P3.1 审计/测试/边界记录 |

```text
 movie/settings.py                     | 7 ++++++-
 myapp/agent_views.py                  | 5 ++++-
 tests/test_redis_cache_p0_isolated.py | 9 ++++++++-
 3 files changed, 18 insertions(+), 3 deletions(-)
```

这份 git diff --stat 仅统计跟踪文件，不代表完整修改范围。完整 patch 同时包含上述四个新增文件，
由 git diff 与逐文件 git diff --no-index /dev/null FILE 组成，不需要暂存；交付时另提供只读 patch 制品。
没有模型、权重、训练数据、检查点、环境文件或密钥；没有误改推荐算法。

验收建议：R1 的单实例/有明确状态续接调用模式已通过代码级隔离回归，P3 可在接受上述限制后进入本地提交复审。
**不应批准无状态重启 count=1 的公平性承诺，也不应批准真实 Redis 并发/Exactly-Once/生产可靠性保证。**
若验收要求自动跨进程崩溃续接，当前边界内尚未满足，须另行设计并授权持久化状态。
本轮不自动暂存或提交；完成后暂停，等待人工验收。


## 15. P3.2 正式消费入口与单实例生命周期

日期：2026-10-14（Asia/Singapore）。本节记录代码级隔离验证，不代表实际执行管理命令。

### 15.1 入口审计、根因与修改范围

此前阻断：仓库没有生产 consume_once 调用循环；schedule_feedback_analytics 的提交回调
每次创建服务后只 publish，不消费。P3.1 的状态续接与长生命周期消费仅存在测试/示例中。
第 14 节的接口能力没有提供一个实际命令入口，因此不能把其 Fake 生命周期当作正式运行模式。

本轮新增 myapp/management/commands/consume_redis_feedback.py，使用 BaseCommand；
Command.handle 是正式、显式调用的消费控制入口。实例在有限循环前通过 analytics_from_settings 创建一次，
循环中仅调用同一 service.consume_once(count=..., max_seconds=...)，不传回固定 0-0 或重置 first_lane。
已有 Consumer、Lua、producer、连接工厂、反馈提交/缓存失效、推荐算法均未修改。
新增 tests/test_redis_structures_p32_isolated.py，仅验证控制循环和安全边界；本报告追加当前章节。

本轮没有注册 Cron/Systemd/Celery/开机服务/Django 启动钩子，没有依赖安装或服务运行。
默认 Analytics 关闭不变，未配置专用连接也拒绝运行。命令不操作 Session cache，不创建表或迁移。

### 15.2 创建、保留与销毁

1. 先验证参数、功能开关及专用 URL；关闭/缺配置时直接 CommandError，不调用连接工厂。
2. 开始总运行时钟，再调用已有工厂一次；工厂沿用专用 DB 隔离和原 socket/connect timeout 配置。
3. 在 for range(max_polls) 中串行复用同一个实例，_reclaim_cursor/_next_lane 自然随实例保留。
4. 单次正常返回后聚合数字计数；下一轮使用实例上更新的游标及轮转状态。
5. 达到次数/时间预算或遇到抛出的基础设施异常就退出；finally 关闭本命令专属 client。
   关闭的是客户端连接，不是停止 Redis 服务，也不发送清理命令。清理失败单独报错；不遮盖原轮询错误。
6. 没有全局实例或持久化调度状态；下一次命令运行创建新实例，游标/优先级从初始状态开始。

这使同一命令运行周期内的生命周期具备真实生产源码调用入口；测试验证的仍是 AST 提取的
Command 类配合 BaseCommand/settings 输出桩，而非真实 Django 发现命令、AppConfig 初始化或连接行为。

### 15.3 有界运行参数与空轮询

| 参数 | 默认 | 允许范围 | 作用 |
|---|---:|---|---|
| --count | 100 | 整数 1..100 | 沿用 consume_once 每轮获取/处理消息预算 |
| --max-polls | 10 | 整数 1..1000 | 最大轮询次数，不支持 0/无限循环 |
| --max-seconds | 30 | 有限数值 1..300 | 包含工厂初始化的合作式总时间预算 |

每轮调用前计算 remaining，传给 Consumer 的 max_seconds 为 min(10, remaining)。
已有 API 最小单轮时长是 1 秒，remaining<1 时提前正常退出，不把余量向上取整而突破总预算。
读取继续沿用 P3.1 的非阻塞方式，不传 XREADGROUP BLOCK，也不新实现 claim/read/ACK/Lua。
空轮询（claimed+fresh=0）且还有下一轮时，休眠 min(0.1 秒, 剩余时间)，最后一轮不休眠。
测试使用 Mock 时钟/睡眠，不实际等待，不进行 CPU 空转或并发压力测试。

时间限制仍是合作式的：不能强制中断正在执行的 Redis 命令、Lua、工厂或 client.close。
没有 BLOCK 并不等于任何网络操作必然在总截止前返回；原工厂 socket/connect 超时各 0.5 秒，
实际 redis-py 重试/超时、Redis 阻塞及关闭行为须独立集成验证，不能宣称硬实时退出保证。

### 15.4 错误退出、聚合统计与隐私

- 未启用/缺专用 URL/参数非法：运行前拒绝；没有 Redis 调用。已知同 Session DB 配置由原工厂拒绝。
- 初始化错误：固定提示，CommandError 退出，不做重试，不打印 URL、凭据或原异常文本。
- consume_once 抛出异常（组创建/校验、读取、连接或协议等）：退出原因 infrastructure_error，
  打印已完成轮询的聚合统计后 CommandError 退出，不高速重试，不重新创建 Consumer。
- 返回报告的 failed/failed_ids：消息级失败已经由 Consumer 保留 Pending；有限循环继续，
  不主动 ACK、删除、重发消息或重复数据库写入。聚合输出不包含 failed_ids 或事件内容。
  failed 是原 API 的消息处理层失败计数，可能含验证/Lua/写入/ACK 连接异常；它并非精确错误分类。
  本轮不改 API 来区分其全部根因，不把 failed>0 的批次称为全部成功。后续读错误会按抛出异常退出。
- prune_failed、deleted_pending、deferred 原计数保留；它们可能表示仍需人工核对的积压/不完整处理。
- 若抛出异常发生在一个尚未返回的轮询中，该轮可能已经交付消息，但无法从报告精确聚合，
  输出 polls 只计正常返回的轮次；不得凭 aggregates=0 判断该轮没有 Redis 副作用。

输出固定 JSON 字段：polls、exit_reason、processed、duplicates、acked、failed、prune_failed、
claimed、fresh、deleted_pending、deferred。无 consumer 标签、URL、cursor、event_id、Session Token、
聊天内容、画像或原始事件。底层原有安全错误日志机制未修改。
正常预算退出与抛出 CommandError 分开；消息失败仍可能正常预算退出，调用者必须检查 failed 等计数，
不能只凭命令退出码将毒消息标为处理成功。输出没有永久保存失败范围；恢复仍需原 Pending/日志和人工对账。

### 15.5 生命周期及公平性隔离结果

| 场景 | 实际 Mock/Fake 结果 |
|---|---|
| --max-polls=4 | 只创建一次 Consumer；同一对象调用 consume_once 四次；未显式重置 cursor/first_lane |
| count=1，2 条失败 Pending + 2 条新事件 | 四轮 failed=2、acked=2；读取 claim/read/claim/read；claim cursor 0-0→2-0；失败项仍在 Pending |
| count=2，两类各 2 条 | 两轮 claimed=2、fresh=2、failed=2、acked=2；新事件按序确认 |
| count=100，100 条毒 Pending + 100 条新事件 | 两轮累计 claimed=100、fresh=100、failed=100、acked=100；cursor 0-0→51-0 |
| 轮询次数上限 | 到达指定次数停止；最后一轮没有多余 sleep/poll |
| 时间预算 | 不把小于 1 秒的余量取整；初始化耗时也占预算；未开始的新轮不会获取消息 |
| 空轮询 | 四轮三次 0.1 秒假休眠；不足 0.1 秒时按余量休眠；无真实等待 |
| 关闭/缺 URL/共享 Session DB | 无 client factory 网络调用；共享 DB 由真实工厂逻辑静态注入 Fake 连接后拒绝 |
| 读取异常 | 单次尝试后 CommandError；清理一次；无无限重试/凭据输出 |
| ACK 失败/消息失败 | Pending 保留；统计不盲重计；有限轮询继续，聚合 failed 不被抹掉 |
| 清理失败 | 显式 cleanup_error；已有基础设施失败不会被清理错误遮盖 |

新增 **23 项**隔离测试。导入 guard 在加载夹具之前拒绝 Django、redis、torch、transformers、
sentence_transformers、numpy、faiss、tensorflow、MySQLdb、neo4j。
命令文件只 ast.parse/compile 选定 Command 类；Django 顶层 imports 未执行。
BaseCommand、CommandError、settings 使用最小桩，时钟与 sleep 使用 Mock；连接工厂替换为 Fake。
绑定检查先检查真实生产模块符号，再提供测试依赖；删除 analytics_from_settings 的真实 import 必须静态报错。
没有通过注入缺失生产符号掩盖错误，没有运行 manage.py、BaseCommand.execute 或 Django.setup。

### 15.6 完整隔离回归与静态检查

执行前审计九套测试、按文件加载/AST exec 路径及安全 helper 的 imports。新增命令不被实际导入。
逐个在前台执行，无后台、守护进程、多进程压测或真实数据库：

```text
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/FILE.py
```

| 文件 | 项数 | 结果/退出码 |
|---|---:|---|
| test_redis_cache_p0_isolated.py | 14 | OK / 0 |
| test_redis_cache_p01_isolated.py | 18 | OK / 0 |
| test_redis_cache_p1a_isolated.py | 18 | OK / 0 |
| test_redis_cache_p1b_isolated.py | 8 | OK / 0 |
| test_redis_cache_blockers_isolated.py | 23 | OK / 0 |
| test_redis_cache_p2_isolated.py | 24 | OK / 0 |
| test_redis_structures_p3_isolated.py | 41 | OK / 0 |
| test_redis_structures_p31_isolated.py | 26 | OK / 0 |
| test_redis_structures_p32_isolated.py | 23 | OK / 0 |
| **合计** | **195** | **全部通过，原 172 项保留** |

本轮没有改任何原测试、原 Consumer 或生产调用模块。静态 AST/生产符号绑定检查通过，
git diff --check 与新增文件 UTF-8/行尾空白/常见敏感模式检查通过。
没有模型加载、推理、训练、GPU/CUDA 进程操作、真实服务连接、依赖安装、缓存清理或迁移。
原训练目录、其他 Worktree 及冻结分支未操作。

### 15.7 手动运行方式（仅文档，禁止在当前环境执行）

后续在人工授权的独立 Redis 7+/Django 测试环境中，先审计所有导入和 AppConfig.ready，
配置一个不做模型预热的安全测试 AppConfig/settings，再启用 Analytics 并明确提供专用 URL。
只禁用 management system/migration checks 不会跳过 Django.setup/AppConfig.ready；当前训练机不可直接运行。
本节不是服务部署或运行授权，本轮没有执行以下命令：

```sh
python manage.py consume_redis_feedback --count 1 --max-polls 4 --max-seconds 30
python manage.py consume_redis_feedback --count 100 --max-polls 10 --max-seconds 30
```

两个命令分别是两个新的生命周期，不会共享实例状态；仅每个命令内部复用实例。
没有注册任何自动调度；不得把此示例改为 Cron 每次 count=1/max-polls=1 并宣称解决持续饥饿。
单次 count=1/max-polls=1 没有跨轮轮转机会，这种无状态频繁启动模式不在公平性验收保证内。

### 15.8 重启限制与真实集成缺口

公平性仅针对同一命令运行周期内连续轮询，或由调用方正确续接状态的既有 API 用法。
本命令不持久化轮转状态/游标；重启后恢复初始 pending/0-0，不保证 count=1 无状态频繁重启公平性，
重置游标也可能使 PEL 深层消息长时间未扫描；不保证跨进程游标共享。
不保证永久 Exactly-Once、数据库/Stream 原子提交、Outbox 最终送达或九天窗口外自动安全恢复。
旧 Pending 超窗、Lua 部分写入、ACK 所有权竞争和 best-effort 投递等风险仍按第 13/14 节人工处理。

独立环境待验证：Django 实际发现命令/execute/output/CommandError 退出码、无模型的 AppConfig、
redis-py socket/retry/close 及总预算超时行为、Redis 真正的 cursor/idle/Pending、双消费者抢占/ACK、
Lua ACL/部分写入/类型和容量并发、dedupe 超窗/淘汰、真实 on_commit 与 Session 隔离。
本轮 Fake/AST 不证明这些集成已通过，也不证明真实运行吞吐量或延迟。

### 15.9 Git 与验收建议

分支 feat/redis-data-structures；HEAD 仍为 f4037f90acd1132ae68652cf0e3d7b6dec58bf3a。
总差异为原 3 个跟踪修改和 **6 个未跟踪新增文件**（P3/P3.1 的四个，加本轮两份 Python 文件）。
本轮实质修改仅 command、P3.2 测试和报告三份文件，没有改动此前的生产代码或测试。
完整文件清单：

- movie/settings.py（原 P3 修改）
- myapp/agent_views.py（原 P3 修改）
- tests/test_redis_cache_p0_isolated.py（原 P3 夹具修改）
- myapp/services/redis_analytics.py（P3/P3.1 新增，P3.2 未改）
- myapp/management/commands/consume_redis_feedback.py（P3.2 新增正式入口）
- tests/test_redis_structures_p3_isolated.py（P3 新增，P3.2 未改）
- tests/test_redis_structures_p31_isolated.py（P3.1 新增，P3.2 未改）
- tests/test_redis_structures_p32_isolated.py（P3.2 新增 23 项）
- docs/REDIS_DATA_STRUCTURES_P3.md（保留历史并追加本节）

此前的正式调用入口缺失已在源码层面落实：循环外创建一次、循环内沿用实例状态、有限退出。
在接受本节重启及真实集成限制后，P3 具备本地提交前人工复审条件；不是生产上线验收。
最终 Git hooks/暂存内容检查属于下一轮提交前工作，本轮没有执行 hooks、暂存、提交、推送或合并。
完成后暂停，等待人工验收，不进入 P4。
