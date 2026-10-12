# MovieAgent Redis P0.1 回归修复报告

## 范围与保护

修改目录为 `/home/daylight/下载/DjangoProject3/MovieAgent-redis-cache`，分支 `refactor/redis-cache`。
本轮没有访问或操作原始训练工作目录；未操作任何训练进程、GPU/CUDA、容器、模型、数据或检查点。
未启动服务、执行真实缓存清理、安装依赖、暂存、提交、推送或合并。
在现有 P0 修改上增量修改，保留 memory.py 已迁移的字典输入和缓存隔离改动。
`REDIS_CACHE_P0_AUDIT.md` 是历史阶段报告；其中关于 rerank 跳过和遗漏反馈的说明以本报告为准。

## 本轮文件与关键差异

| 文件 | P0.1 修改 |
| --- | --- |
| `myapp/agent_views.py` | 在构造 SSE 响应之前获取真实 Session；通用反馈 API 使用 atomic/on_commit；失效失败明确返回503。 |
| `myapp/recommender/rerank.py` | business_filter/final_rerank 在签名末尾新增可选 memory_slots；取消模块内部构造 user_ID 和读取 MemoryManager。 |
| `myapp/agent/movie_agent.py` | 当前 Agent 的槽位快照传给 RerankTool，再传给 final_rerank。 |
| `myapp/agent/react_agent.py` | rerank 分支传递当前 memory.get_slots()。 |
| `myapp/agent/workflow_agent.py` | rerank 分支传递当前 memory.get_slots()。 |
| `myapp/utils/cache_keys.py` | 新增失效异常类型；Redis操作异常或 generation 写入明确返回False时，向调用方报告部分失败。 |
| `myapp/views.py` | dislike删除/创建置于同一事务；undislike也使用事务后失效；失效失败返回503；禁用全局清理并保留模型缓存。 |
| `tests/test_redis_cache_p0_isolated.py` | 原14项夹具补充atomic上下文和异常类型，保留原测试；注册型夹具明确假定存在外层事务。 |
| `tests/test_redis_cache_p01_isolated.py`（新增） | 18项隔离回归测试，有状态FakeTransaction/FakeFeedbackDB及AST生产函数。 |
| 本报告（新增） | 记录结果、边界和未覆盖写入口。 |

## Session 生命周期

修改前：视图返回 StreamingHttpResponse → SessionMiddleware 处理响应 → 消费 event_stream →
首次调用 _get_session_key/create。需要创建时已错过中间件设置Cookie的时机。

修改后：视图校验输入 → _get_session_key 获取/创建Session → 构造并返回响应 →
SessionMiddleware 处理响应 → event_stream 复用捕获的 session_key。
已有 Session 不创建；新 Session 的实际 cache SessionStore.create 会设置 modified 状态。
生成器不再访问创建路径。正常 SSE 事件内容、分块和 Agent 推理逻辑未修改。
Session创建失败在构造流响应前向上报告；没有吞掉Session故障。
空输入/被拦截输入仍沿用原来的直接响应路径，不启动Agent，也无需会话记忆。

测试用FakeSession验证create在响应构造前发生、modified已设置、已有Key复用、生成器不创建、
事件内容保持一致。未运行真实SessionMiddleware或浏览器Cookie往返；不能据此声称生产Cookie已验证。
依据：[Django 5.2 Session中间件源码](https://github.com/django/django/blob/stable/5.2.x/django/contrib/sessions/middleware.py)
及[cache Session后端源码](https://github.com/django/django/blob/stable/5.2.x/django/contrib/sessions/backends/cache.py)。

## 偏好传递与推荐兼容性

修改前：business_filter 自建 user_ID → 新Memory Key规则拒绝 → 新实例槽位为空 → genre优先补充失效。

修改后：当前MovieAgent/ReAct/Workflow的MemoryManager → memory_slots快照 → RerankTool →
final_rerank → business_filter → 原genre优先补充和安全池兜底。
ExecutionController及快速通道通过MovieAgent._act走同一路径；三个Agent的直接rerank调用均检查过。
原有位置参数保持不变，新参数为可选尾部参数。没有槽位时保留安全电影兜底，不查询任何其他会话。
无Session的Agent也只能提供自己实例内的槽位，不恢复固定用户级共享缓存。

测试将HEAD中的原business_filter以AST提取、用FakeMemory提供相同genre，
与新函数显式传槽位的结果逐项对照；敏感电影过滤、genre补充、评分字段、返回顺序一致。
不同会话科幻/剧情槽位分别得到对应补充，证明不会从重排模块读取其他会话。
截断、diversity_guarantee源码AST未变；MAANRerankTool/LLMRerankTool模型类及向量/内容/模型召回、RRF函数AST未变。
本轮没有修改排序公式、模型参数、训练代码或推荐模型。

## 数据库提交与缓存失效

三个入口：recommend_feedback_view、ajax_exclude_add、ajax_exclude_remove。

1. 进入transaction.atomic；dislike旧记录删除与新记录创建处于同一事务。
2. 写入成功后在事务内注册on_commit，使用默认非robust回调，不吞掉失效异常。
3. 创建失败时删除回滚；事务/保存点回滚时对应回调被丢弃。
4. 嵌套事务中内层退出不失效，最外层成功提交后才轮换generation并精确删除同用户legacy hybrid/cb Key。
5. 无外层事务时回调在atomic退出期间执行。失效异常被转换为RecommendationCacheInvalidationError，
   视图返回HTTP503，明确database_committed=true、cache_invalidated=false，并提示不要重复提交反馈。

静态检查当前settings和相关视图未配置ATOMIC_REQUESTS，也未发现这三个入口既有外层atomic包装。
新增atomic支持后续外层事务。如果将来外层事务包住视图，回调异常在外层退出时传播，
视图内不能将该异常转换成当前JSON503；外层调用者/框架须处理。不会用robust=True默默跳过。

保证边界：这不是数据库与Redis的分布式事务。数据库提交后Redis不可用，数据库不能回滚；
generation写入失败时，旧缓存仍可能被后续请求读到，直到成功失效或原TTL结束。
部分删除失败也会明确报错，不宣称失效完整。没有持久化重试队列或数据库版本号，因此不承诺强一致性。
本轮选择显式失败和记录异常，未引入全局开关、锁、后台重试、Schema或模型相关操作。

## 危险清理入口

admin_clear_cache的POST返回HTTP409及cache_clear_disabled提示；不调用cache.clear、delete_pattern或任何缓存操作。
也不重置MODEL_CACHE，避免下一次请求因这个入口触发模型重载。
Session与业务缓存共用后端，当前无法可靠枚举所有业务范围；待独立业务Cache Alias/DB后再设计清理。
此轮没有实际执行任何Redis命令。测试只执行提取的函数，使用Mock缓存并断言零调用。

## 安全验证

执行前导入链审计：P0.1测试 → 按绝对路径加载P0夹具 → 按文件路径加载cache_keys。
均为标准库依赖，避免myapp.agent和views包初始化。
生产函数/动作方法仅由AST提取；ORM、业务资源、LLM、事务和工具都替换为Fake/Mock。
旧函数对照的MemoryManager导入也是Fake；git show仅在当前Worktree只读读取HEAD源码。
启用-S禁用site启动初始化，禁止模型模块导入，测试限制GPU和CPU线程。

实际执行命令（前台、依次执行）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p0_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p01_isolated.py
```

最终：原14项 + 新18项 = 32项通过。
新增首轮4项夹具失败：Fake未成年人排除返回list，而真实get_minor_excluded_ids返回set；
改正Fake返回类型后18项通过，未改变生产过滤规则。

新测试覆盖SSE新/已有Session及创建时序、创建故障、三个入口各参数缓存失效、创建失败删除回滚、
外层事务回滚、嵌套提交延迟、嵌套回调失败传播、Redis写/删失败显式报告、禁止全局清理、
偏好补充新旧对照、无槽位安全兜底、不同会话补充、Tool/MovieAgent/ReAct槽位传递及Workflow静态调用检查。

静态检查：11个修改/新增Python文件解析及内存compile通过；git diff --check通过。
未运行完整Django测试、服务、模型推理、真实Session Cookie、Redis序列化/并发、MySQL事务及Neo4j集成。
FakeTransaction验证生产函数如何使用事务API，并不代替真实数据库和Django事务验证。
Workflow槽位传递使用AST断言，未执行完整Workflow流水线。

## 尚未覆盖的写入入口

以下在本轮审计范围内仍未登记统一推荐失效，不声称已实现全量一致性：

- `views.score_movie`、`views.movie_detail`：评分/评论写入。
- `views.add_collect`、`views.remove_collect`、`views.ajax_collect`：收藏变化。
- `views.center`：用户资料表单保存。
- `views.admin_comment_delete`：管理端评分删除。
- `agent_views.admin_user_feedback_delete`：管理端反馈删除。
- `agent_views.admin_user_profile_edit/delete`：画像编辑/删除。
- `MemoryManager.persist_to_profile`、`_merge_to_profile`：画像偏好写入；
  ajax_new_chat、ajax_summarize_profile等调用这些方法的入口也需要后续覆盖。

ORM缓存轻量化、通用缓存异常降级、指标、旧版本混跑的兼容失效仍未实施，未做性能压测。

## 验收与完整变更清单

P0代码及隔离单元测试具备验收条件；生产验收为有条件通过，条件包括真实Cookie/事务验证，
以及接受上述跨系统一致性和剩余写入入口边界。没有宣称强一致性或端到端验证完成。

相对当前HEAD的累计git diff --stat（含P0和P0.1，不含未跟踪文件）：

```text
 myapp/agent/memory.py                | 28 +++++++++--
 myapp/agent/movie_agent.py           | 16 ++++---
 myapp/agent/react_agent.py           |  7 +--
 myapp/agent/workflow_agent.py        |  7 +--
 myapp/agent_views.py                 | 43 +++++++++++++----
 myapp/recommender/rerank.py          | 23 ++++-----
 myapp/services/agent_chat_service.py |  7 +--
 myapp/views.py                       | 92 +++++++++++++++++++-----------------
 requirements.txt                    |  1 +
 9 files changed, 137 insertions(+), 87 deletions(-)
```

另有5个未跟踪文件：
`myapp/utils/cache_keys.py`、`tests/test_redis_cache_p0_isolated.py`、
`tests/test_redis_cache_p01_isolated.py`、`docs/REDIS_CACHE_P0_AUDIT.md`、本报告。
总计14个累计变更文件。未修改原始训练目录，未提交或推送。
