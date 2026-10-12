# MovieAgent Redis 最终审查与完整隔离回归

> 修复轮更新（2026-10-12）：F1/F2 阻断已修复，F3 中审查指出的状态修改入口已收紧 POST/CSRF，
> F4 的异常分数与统计结构校验已补齐。五套 CPU-only 隔离测试实际 81 项全部通过。
> 第 1～9 节及原差异附录保留为首次审查的历史快照；当前修复状态、HTTP 契约及恢复流程以第 10 节为准。

审查日期：2026-10-12（Asia/Singapore）。

工作目录：/home/daylight/下载/DjangoProject3/MovieAgent-redis-cache。
分支：refactor/redis-cache。
HEAD：d78390dede4a0371a1e7991f06568d8d390fbdbd。

## 1. 结论与本轮边界

**最终结论：需要修复，当前不建议作为完成成果提交，也不建议合并 main。**

四个指定隔离脚本按顺序在前台执行，14 + 18 + 18 + 8 = **58 项全部通过**，各进程退出码均为 0。
但静态审查和安全 AST 复现发现了测试夹具掩盖的实际运行时缺陷，以及多用户提交后回调的恢复范围缺口。
测试全绿不能替代实际模块命名空间、Django 中间件、事务与 ORM 集成验收。

本轮只新增本报告；没有修改源码、依赖声明、已有测试或历史报告，没有修复下列问题。
没有操作原始训练工作目录，也没有读取其当前 Git 状态或未提交文件。
本报告对原始 settings/memory 修改的描述仅引用之前的授权信息和历史阶段报告，不代表重新核验原目录。

## 2. P0～P1-B 完成清单

| 阶段 | 已实施内容 | 最终审查状态 |
|---|---|---|
| P0 | django-redis 依赖声明；意图 SHA-256；真实 Session 与用户身份摘要；无 ID 本地记忆；字典槽位持久化；KG 参数规范化；top_k 隔离 | 代码存在，隔离回归通过；依赖安装与真实后端兼容性未验证 |
| P0.1 | SSE 创建 Session 提前到响应构造前；显式 memory_slots 经 Agent → RerankTool → final_rerank → business_filter；反馈 atomic/on_commit；禁用全局清理 | 测试通过；真实 Cookie/中间件待验证；继承的部分写接口 CSRF 风险见第 5 节 |
| P1-A | 用户/目录 generation；在线评分、收藏、反馈、画像、管理端变更的提交后失效；业务缓存回源；严格记忆故障；partial_success；只重试失效接口 | 单用户路径覆盖较好；多用户失败恢复不完整，不能宣称全量一致性或最终完成保证 |
| P1-B | 混合推荐轻量 payload、新键族/schema、批量 ORM 恢复、顺序/分数/关联预取兼容 | 正常冷热路径测试通过；坏 payload/电影缺失路径存在缺失导入，阻断验收 |

### 算法与范围核对

相对 HEAD 的 AST 比较确认以下内容相同：

- views.py 的 vector_recall、content_recall、model_recall、rrf_merge。
- rerank.py 的 diversity_guarantee。
- movie_agent.py 的 MAANRerankTool、LLMRerankTool。
- get_kg_subgraph 内全部包含 MATCH 的 Cypher 字符串。
- hybrid_recall_recommend 从画像构建到 result 元组的冷路径计算语句。
- update_recommendations 在推荐记录持久化之前的计算前缀。

update_recommendations 的改动只把删除/批量创建记录移入 atomic，并登记失效，没有改变模型计算、选片公式或参数。
rerank 的差异是会话偏好传递；recall/query_rewriter 的差异是缓存键与故障路径，没有改召回/重排公式。
未发现训练脚本、模型实现、实验配置、数据集、检查点、索引、迁移或前端模板出现在变更清单中。
**当前差异属于 Redis 工程化及相关事务/会话兼容范围；现存缺陷尚不能说明成果已可合并。**

## 3. Key、payload、TTL、失效机制总览

以下为逻辑键；Django 后端仍使用 movieagent KEY_PREFIX、默认 alias，Redis URL 的配置回退值为 DB 0。
本轮没有读取运行时 REDIS_URL 或连接该后端，不能据此断定部署实际 DB。
CACHES 默认 TIMEOUT 为 3600 秒，下面显式 TTL 优先。

| 缓存 | 当前逻辑键/版本 | payload | 显式 TTL | 失效/故障行为 |
|---|---|---|---:|---|
| 意图 | intent:v2:SHA256(trimmed_input) | 解析字典 | 600 | 新前缀不读旧 hash 键；Redis 故障继续原解析路径 |
| 查询改写 | query_rewrite:v2:SHA256(trimmed_query) | 字符串 | 3600 | Redis 故障继续已有 LLM/规则逻辑；不缓存用户认证状态 |
| 会话槽位 | agent_memory:v2:SHA256(JSON([user_id, session_id])) | 槽位字典 | 3600 | 没有有效 ID 时只保留实例内槽位；读写异常报 MemoryCacheUnavailable；清槽位只删当前键 |
| KG 子图 | kg_subgraph:v4:topic_sha:history_sha:max_triples:catalog:cgen | 三元组文本 | 1800 | 历史去重、排序、保留 ID 类型；目录代际；缓存故障回源，Neo4j 原故障路径仍保留 |
| 混合推荐 | hybrid_rec_payload:v3:uid:ugen:top_k:k:catalog:cgen | hybrid_rec_payload_v1 字典 | 1800 | 新键族隔离旧 ORM 元组；用户/目录代际；坏数据恢复分支当前存在缺失导入 |
| 多路召回 | multi_recall:v3:uid:ugen:params:SHA256(JSON([query, top_k, weights])):catalog:cgen | 结果列表与统计元组 | 1800 | 用户、参数、目录隔离；缓存故障继续召回 |
| CB 推荐 | cb_recs:v3:uid:ugen:空参数:catalog:cgen | source 与 Movie 列表字典 | 3600 | 用户/目录代际；仍为 ORM payload，本轮目标仅轻量化混合推荐 |
| 推荐解释 | explain:v3:uid:ugen:movie_id:catalog:cgen | 解释结果字典 | 默认 600 | 在线调用捕获同一个键贯穿读写，防止计算后写入新代际 |
| Agent recall 帮助函数 | agent_recall:v3:uid:ugen:空参数:catalog:cgen | 列表 | 默认 1800 | 代际失效；当前没有活动调用方，写入重新取代际的潜在竞态见第 5 节 |
| 电影概念 | movie_concepts_jieba_SHA256(JSON([title, summary])):catalog:cgen | 概念列表 | 86400 | 目录代际；缓存故障走已有提取逻辑 |
| 用户代际 | hybrid_rec:v2:uid:generation | UUID 字符串 | 1800 | 缺失时 add 选赢家再读取；提交后 set 新 UUID |
| 目录代际 | recommendation_catalog:v1:generation | UUID 字符串 | 1800 | 内容变更提交后 set 新 UUID；不扫描/删除其他命名空间 |
| Django Session | 由 Django cache SessionStore 构造，使用 default alias | 身份/认证会话状态 | 由 SessionStore 过期规则决定 | 没有套用业务 fail-open；实际认证失败语义及 Cookie 行为未集成验证 |

代际查询/创建抛异常或创建后仍读不到时，键构造返回 None，业务缓存跳过。
随机代际避免 token 被淘汰/过期后复用固定旧版本；有代际的长 TTL 条目可能在 1800 秒代际过期时提前失去可达性。
“条目 TTL 保留”不等于“有效命中时长完全保持”。

用户失效会轮换 ugen，使所有 top_k/参数版本不可达，并精确删除该用户三个 legacy 键：
hybrid_rec_UID、cb_recs_UID、agent_recall_UID。
目录失效只轮换 cgen。没有 delete_pattern、全局 cache.clear 或 FLUSHDB；admin_clear_cache 返回 409。
Session 与业务共用 alias，危险全局清理仍应维持禁用。

### P1-B 恢复与兼容流程

冷路径计算仍返回 Movie 列表、画像文本、source_stats；电影赋予原四位小数 rrf_score。
写入缓存时只保存有序 movie_id/rrf_score、画像和紧凑统计字典，不保存 Movie 实例。
统计中的重复 profile_text 不入缓存，热路径会补回该字段。

命中时验证 schema、items、ID/分数类型，使用一次 filter(id__in=...) 批量读取 Movie，
保留 genres/actors/directors 预取，再按 items 的顺序和分数恢复原三元组。
电影字段来自当前 ORM；代码没有逐电影 filter/get 的模式，但真实 SQL 数量没有测量，
预取本身会有关系查询，不应把 Mock 的单次 filter 调用等同于“总共一条 SQL”。

字段非法/电影缺失时设计为删除精确缓存键并重算；目前删除函数没有导入，实际该分支会中断。
正常返回的语义及排序未变；坏数据路径的 P1-B 与先前 P0/P1-A 行为存在实际不兼容。

## 4. 完整隔离回归结果与安全检查

执行前完整读取了四个测试文件和两个实际加载的缓存工具模块。
实际项目文件导入仅通过文件路径加载 cache_keys.py、cache_access.py 及 P0 测试夹具；
这些文件使用标准库，不执行 Django setup。
生产函数/类通过 AST 读取后执行选定定义；ORM、事务、LLM、Neo4j、资源加载及 Agent 工具由 Fake/Mock 替代。
P0.1 中旧函数对照的 myapp/模型/Memory 导入同样被替换为假模块。

P0 的跨进程键检查只启动两个同步等待的标准库 python -S 子进程，运行纯 cache_keys 文件；
git show 只在当前 Worktree 读取 HEAD 源码。没有 nohup、后台服务或模型执行。
测试 guard 禁止 torch、transformers、sentence_transformers、faiss、tensorflow、numpy 导入。
guard 在文件路径加载之后安装，因此安全判断还依赖此次实际审计，而不是仅依赖环境变量或 guard。

四条命令分别前台执行，前一条退出后再运行下一条：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p0_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p01_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p1a_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p1b_isolated.py
```

| 脚本 | 实际运行数 | 结果 | 退出码 |
|---|---:|---|---:|
| P0 | 14 | OK | 0 |
| P0.1 | 18 | OK | 0 |
| P1-A | 18 | OK | 0 |
| P1-B | 8 | OK | 0 |
| 合计 | 58 | 全部通过 | — |

终端工具输出含有“Failed to create stream fd: Operation not permitted”启动提示；
各 unittest 进程均完成并输出上述 OK/退出码。本报告不把该提示当成业务测试失败，也不以测试耗时声称接口性能。

随后对全部 22 个已变更 Python 文件执行标准库 ast.parse 与内存 compile，通过；
其中 7 个未跟踪 Python 文件全部包括在内。没有 py_compile 写入字节码。
git diff --check 退出码为 0。完整指定脚本回归不是 Django 完整测试套件。

另执行两个一次性标准库/AST/Fake 检查（未新增测试文件）：

1. 按实际源码导入集合提供全局变量，放入旧元组 payload：复现 NameError: business_cache_delete is not defined。
2. 用实际 admin mixin 的 _schedule 登记两个 FakeTransaction 回调，第一个失效异常时只尝试用户 1，
   用户 2 的回调没有执行，异常只能报告用户 1。该检查说明控制流风险，不能冒充真实 Django 数据库集成测试。

### 58 项通过仍留下的测试缺口

- P0 extract 会自动注入 business_cache_delete 等辅助函数，并去除 HTTP 装饰器。
  P1-B 测试还显式注入 delete，掩盖了真实 views 命名空间缺失导入。
- P0.1 三个反馈入口执行的是去掉新装饰器的函数体；P1-A 使用合成视图测试装饰器，
  不能代替“真实入口 + 装饰器 + 中间件”的组合验证。
- P1-A 的 error JSON 回滚用例没有先写数据库状态或登记失效回调，只断言空缓存，
  不能充分证明之前的写入/回调确实被撤销。
- 默认 admin mixin 的真实 save/delete/bulk-delete、所有权变更及多回调失败没有专门回归。
- 重试函数的装饰器被移除，登录/POST/CSRF 仅静态审查；staff/catalog、非法参数和 503 重试故障没有完整运行覆盖。
- FakeSession 自行设置 modified；Cookie、安全属性、SessionStore 读取故障与 SessionMiddleware 保存没有验证。
- FakeCache 不模拟 TTL、序列化、Redis 原子 add 或实际并发。
- ORM 假对象不是新加载的 Django Model；SQL 数量、prefetch 缓存、字段实时变化与序列化还需真实验证。

## 5. Bug 与潜在回归风险

优先级含义：P1 为合并前必须处理或明确隔离的重要风险；P2 为需补充验证/限定适用范围的问题。
未发现此次改动引入可证实的跨用户缓存数据泄漏；这不等于端到端认证/隐私验证完成。

### F1 — P1，实际 P1-B 缺陷：失效重算分支缺少导入

位置：[views.py:59](/home/daylight/下载/DjangoProject3/MovieAgent-redis-cache/myapp/views.py:59)
与 [views.py:1067](/home/daylight/下载/DjangoProject3/MovieAgent-redis-cache/myapp/views.py:1067)。
模块只导入 business_cache_get/business_cache_set，却调用 business_cache_delete。
不存在另一个同名全局定义或导入。
旧/错误 schema、非法 payload 或缓存电影缺失触发该分支时，NameError 阻止正常重算；
外层画像推荐可能捕获异常并降为空结果，导致业务降级结果不符合 P1-B 目标。

本轮没有修复。后续需补导入并增加不自动补齐项目全局变量的依赖绑定检查，
保留删除失败依旧回源的测试，再执行 58 项隔离回归。

### F2 — P1，实际恢复缺口：多个受影响用户的回调可能被首个失败截断

位置：[cache_mutations.py:75](/home/daylight/下载/DjangoProject3/MovieAgent-redis-cache/myapp/services/cache_mutations.py:75)、
save_model、delete_queryset 和 schedule_user_invalidation。
所有权迁移或批量管理操作登记多个默认非 robust on_commit 回调。
第一个异常会阻止后续回调继续执行；数据库已提交，但其余受影响用户的代际可能未轮换。
partial_commit_response 仅返回一个 user_id，恢复接口也按一个 user_id 重试，不能说明其他用户已恢复。
集合遍历没有稳定顺序，不能预先假设哪个用户已完成。

建议后续授权修复时在提交后尝试全部受影响范围、聚合失败并返回全部待恢复用户；
不要仅用 robust=True 吞异常后声称成功，也不要重做已经提交的数据库变更。

### F3 — P1（继承的安全风险，并非此次新增）：部分变更入口绕过 CSRF 或可通过 GET 写入

位置：agent_views.recommend_feedback_view、views.ajax_exclude_add/remove 继承 csrf_exempt；
views.add_collect/remove_collect、admin_comment_delete 不要求 POST，GET 也能修改数据。
新 recommendation_write_view 对非 POST 直接旁路，因此没有给这些 GET 写入提供新增事务边界或请求方法保护。
原 HEAD 已存在这些行为，此次未修改前端或权限规则。
没有进行攻击测试；需在独立授权下复核/修复这些写入入口，或在合并决策中明确接受遗留风险。

新 /agent/cache/retry/ 不存在上述 CSRF 豁免：有 login_required、require_POST、
CsrfViewMiddleware 及 self/staff 范围检查。重复调用只改缓存，不重复用户数据库写入，
但 UUID 每次不同，属于结果安全的重复失效，不能宣称严格的重放去重。

### F4 — P2：payload 校验尚未完整覆盖所有非法数据

位置：[views.py:1026](/home/daylight/下载/DjangoProject3/MovieAgent-redis-cache/myapp/views.py:1026)。
score 为极大整数时 float(score) 可能抛 OverflowError，而不是返回无效 payload 并重算；
source_stats 只检查 dict 类型，没有验证统计字段类型。
缓存由应用自身写入时通常不会出现此输入，但“任意非法 payload 均安全回源”尚未证明。

### F5 — P2：partial_success 的前端恢复体验不完整

templates/agent_chat.html:929、movie_explain.html:172 只处理 status=success，
partial_success 不显示完整成功，也没有显式提醒。
movie_detail.html:176 与 front_center.html:332 会提示非成功消息，但保留旧按钮/列表状态。
尤其收藏为 toggle 操作，数据已经提交后再次点击可能反向修改。
没有发现前端自动调用 /agent/cache/retry/ 的入口；当前恢复依赖人工/API。
HTTP 200 + database_committed=true/retry_mutation=false 的语义较准确，
但不能宣称已有前端完成了部分成功的全面兼容。

### F6 — P2：一致性与并发边界

- generation 失效失败时旧值仍可能可达，直到失效重试成功、条目过期或 token 更新；没有持久化重试队列/最终完成保证。
- Session 槽位 get→merge→set 与 UserProfile 的读改写仍可能在并发请求下丢失更新；atomic 不能自动序列化这些读改写。
- get_slots 对非 dict 缓存值返回空槽位，严格异常策略并未覆盖缓存内容损坏。
- 默认 admin 与在线入口之外的直接 ORM update/bulk_create、导入/维护命令不触发这些 hook。
  已知命令包括 update_maan_scores、calc_recs、import_movielens、clean_data、
  merge_and_clean、clean_metadata、patch_missing_movies、enrich_tmdb_data、import_new_movies；
  本轮没有运行或修改它们。
- Agent recall 的 set_cached_agent_recall 会在写入时重新读取 generation，
  将来若接入“计算旧结果→用户失效→再写”的调用，可能把旧结果存入新代际；
  当前搜索仅发现定义，没有活动调用方。解释在线路径已使用读写同一个键，未出现该问题。
- 旧/新部署混跑时，旧代码仍读旧键，无法观察新 generation；版本切换需规划，不能宣称兼容旧进程长期共存。

### F7 — P2：配置、合并与历史文档

- settings.py 与原始目录的 dotenv/chapter4 配置修改在同一文件：未来合并必须保留原环境加载及路径配置，
  并把新增 MIDDLEWARE 项合入原列表，不应替换整份设置文件。
- memory.py 已含原授权迁移的字典输入分支及后续缓存/画像事务改动；未来与原未提交版本可能冲突。
  不应通过 checkout/reset/整文件覆盖解决，应在人工确认的安全环境逐块比较并保留双方逻辑。
  本轮没有读取原文件，不能确认原目录之后是否新增修改。
- 部分已有文件为 CRLF，本次新增行含 LF；git diff --check 通过，但未来合并可能有额外换行冲突。
- P0/P0.1/P1-A/P1-B 文档为历史快照。例如 P0.1 的 503 说明已被 P1-A 的 partial_success 200 取代；
  P1-B 的“删除无效缓存并重算”描述需结合 F1 看待。最终验收以本报告为准。
- HEAD 已有源码注释包含“2–5s→<10ms”等性能描述，本轮没有验证，不应引用为此次优化收益。
- requirements 的 Django 下限未设上限；django-redis 声明没有安装/解析实际环境。
  未验证将来 Django 版本、Python 版本或现有部署环境的完整兼容性。

## 6. 真实集成验证待办

以下均未在本轮执行，且不可从 58 个隔离测试推断已通过：

1. 在不会执行 MyappConfig.ready/模型预热的独立配置下验证真实 SessionMiddleware：
   首次 SSE Cookie、已有 Session、匿名到登录身份变化、Session 过期与 Redis 故读。
   新辅助函数没有吞 Session 创建异常，但配置的 Django SessionStore 自身读取故障行为尚未验证，
   不能宣称认证 Session 所有故障都会按相同方式向上传播。
2. 在专用可丢弃数据库/缓存范围内验证 Django atomic/savepoint/on_commit，
   创建失败删除回滚、外层回滚、实际 admin CRUD、多用户回调失败与正确恢复。
3. 验证 Retry Endpoint 的登录、POST、CSRF、self/staff/catalog 许可、非法 ID、503 重试响应，
   并明确是否需要真正的请求重放去重。
4. 真实 Movie/关联模型验证冷热字段、ID 排序、分数、删除与 M2M 修改、预取 SQL 数量、空结果。
5. 验证 django-redis 的序列化、原子 add、TTL、key prefix、token 过期/淘汰及并发失效。
   轻量 payload 是纯数据结构，但默认后端 serializer 没有改成本轮强制 JSON。
6. 定义 partial_success 的 UI 提示与只重试失效操作流程；避免重复收藏 toggle/反馈提交。
7. 明确维护命令/外部数据写入如何在成功后失效；接受无 outbox 的恢复保证上限，
   或另行授权持久化恢复设计，不能本轮新增表/Worker。

完整 Django 启动链当前会触发 apps.ready → views.warmup_all_systems；
views 与 agent 包还导入模型栈。因此不能在训练环境直接启动完整服务来补这些验证。
需要先准备独立安全测试配置，并获得对应环境/服务操作授权。

## 7. GPU/CUDA 零干扰执行记录

- 所有 shell 工作目录为独立 MovieAgent-redis-cache；没有读取/修改原始训练目录的 Git 状态或未提交内容。
- 未运行进程管理、GPU/CUDA 查询/修改、训练、推理、Embedding、FAISS/Neo4j 重建、模型/权重加载或下载。
- 未启动 Django、Celery、Redis/MySQL/Neo4j 或常驻后台任务；没有网络数据库连接。
- 未清理缓存/数据库，未运行容器操作、安装、升级、迁移或 Git 强制命令。
- 四脚本按指定 CPU 环境、-S、禁止字节码方式逐个前台执行。
- 静态检查只读源码、AST 和当前 Worktree 的 Git 对象；一次性复现只执行标准库 AST/Fake。
- zvec 检索返回索引缺失后使用 rg；没有创建索引、执行 Embedding 或启动检索后台进程。
- 原始目录未访问，所以不以读取训练日志/进程状态来证明训练进度；“零干扰”指此次执行没有上述干扰操作。

## 8. Git 提交前检查与建议

当前分支 refactor/redis-cache、HEAD d78390dede4a0371a1e7991f06568d8d390fbdbd。
原有 16 个跟踪文件修改、11 个未跟踪文件全部审查。
本轮仅新增本报告后，未跟踪文件总数为 12（源码/工具 3，隔离测试 4，文档 5）。

git diff --check 通过；22 个 Python 文件 AST/内存 compile 通过；58 项隔离回归通过。
本轮没有暂存、提交、推送或合并。报告保存后再次核对源码/既有文档和测试的 SHA-256、
跟踪差异摘要与 HEAD，确认除本报告外没有本轮修改。

新增文件与新增差异的定向静态扫描未发现私钥标记、常见硬编码凭据串或带用户名/密码的 URL；
未跟踪文件均为 .py/.md 文本，没有模型、二进制、实验数据、检查点或训练日志。
这属于有限静态检查，不是对整个原始项目或运行时环境的全面秘密审计。
日志/缓存 payload 仍可包含用户画像或偏好，应依赖项目访问权限及部署安全保护，不可误称匿名化数据。

**本地 Commit 建议：暂缓“完成/可发布”Commit。**
先在新的人工授权下修复 F1/F2、补真实全局依赖与多用户恢复用例，重新执行隔离回归；
若只是保存可恢复的 WIP checkpoint，应由人工明确决定并标注已知缺陷。本轮不提交。

**合并 main 建议：不建议现在合并。**
需满足：修复 F1/F2；对 F3 的遗留写接口风险作明确处理/批准；完成必要 Session/事务/ORM/权限集成验证；
确定 Redis 故障恢复及 partial_success 流程；人工处理 settings/memory 与原未提交修改的语义冲突。
不要求为了合并擅自改变训练代码、算法、Schema 或增加 Worker。

## 9. 完整 Git 差异与文件清单

以下附录由当前独立 Worktree 的只读 Git 输出生成。
跟踪文件差异含 P0～P1-B 的累计修改；未跟踪文件不在 git diff --stat 中，单独列出。
报告附录仅展示差异，不用于自动应用到原始工作目录。

### 实际 Git 状态



~~~~text
 M movie/settings.py
 M myapp/admin.py
 M myapp/agent/memory.py
 M myapp/agent/movie_agent.py
 M myapp/agent/react_agent.py
 M myapp/agent/workflow_agent.py
 M myapp/agent_views.py
 M myapp/middleware.py
 M myapp/recommender/query_rewriter.py
 M myapp/recommender/recall.py
 M myapp/recommender/rerank.py
 M myapp/services/agent_chat_service.py
 M myapp/services/agent_service.py
 M myapp/urls.py
 M myapp/views.py
 M requirements.txt
?? docs/REDIS_CACHE_FINAL_REVIEW.md
?? docs/REDIS_CACHE_P01_AUDIT.md
?? docs/REDIS_CACHE_P0_AUDIT.md
?? docs/REDIS_CACHE_P1A_AUDIT.md
?? docs/REDIS_CACHE_P1B_AUDIT.md
?? myapp/services/cache_mutations.py
?? myapp/utils/cache_access.py
?? myapp/utils/cache_keys.py
?? tests/test_redis_cache_p01_isolated.py
?? tests/test_redis_cache_p0_isolated.py
?? tests/test_redis_cache_p1a_isolated.py
?? tests/test_redis_cache_p1b_isolated.py
~~~~

### 跟踪差异统计（不含未跟踪文件）

~~~~text
 movie/settings.py                    |   1 +
 myapp/admin.py                       |  29 +++--
 myapp/agent/memory.py                | 222 ++++++++++++++++++---------------
 myapp/agent/movie_agent.py           |  21 ++--
 myapp/agent/react_agent.py           |   7 +-
 myapp/agent/workflow_agent.py        |   7 +-
 myapp/agent_views.py                 |  84 +++++++++++--
 myapp/middleware.py                  |  18 +++
 myapp/recommender/query_rewriter.py  |  10 +-
 myapp/recommender/recall.py          |  18 +--
 myapp/recommender/rerank.py          |  23 ++--
 myapp/services/agent_chat_service.py |   7 +-
 myapp/services/agent_service.py      |  26 ++--
 myapp/urls.py                        |   1 +
 myapp/views.py                       | 234 ++++++++++++++++++++++++++---------
 requirements.txt                     |   1 +
 16 files changed, 479 insertions(+), 230 deletions(-)
~~~~

### 全部跟踪修改文件

- movie/settings.py
- myapp/admin.py
- myapp/agent/memory.py
- myapp/agent/movie_agent.py
- myapp/agent/react_agent.py
- myapp/agent/workflow_agent.py
- myapp/agent_views.py
- myapp/middleware.py
- myapp/recommender/query_rewriter.py
- myapp/recommender/recall.py
- myapp/recommender/rerank.py
- myapp/services/agent_chat_service.py
- myapp/services/agent_service.py
- myapp/urls.py
- myapp/views.py
- requirements.txt

### 全部未跟踪新增文件

- docs/REDIS_CACHE_FINAL_REVIEW.md
- docs/REDIS_CACHE_P01_AUDIT.md
- docs/REDIS_CACHE_P0_AUDIT.md
- docs/REDIS_CACHE_P1A_AUDIT.md
- docs/REDIS_CACHE_P1B_AUDIT.md
- myapp/services/cache_mutations.py
- myapp/utils/cache_access.py
- myapp/utils/cache_keys.py
- tests/test_redis_cache_p01_isolated.py
- tests/test_redis_cache_p0_isolated.py
- tests/test_redis_cache_p1a_isolated.py
- tests/test_redis_cache_p1b_isolated.py

### 既有文件 SHA-256（报告保存前后核对基线）

| 文件 | SHA-256 |
|---|---|
| movie/settings.py | 4c055e92f2c112a15e7ead45da73dd5026dcf21568b033424d2ca799d9f56d69 |
| myapp/admin.py | 3833769db7c6d21ca4edf9f08592971918bdda3acc02482a8d26a09d3cc1c313 |
| myapp/agent/memory.py | f7d53337a0a29ef41014cbe08a8d0c52cc28713783680496d51c8b74ac654bc4 |
| myapp/agent/movie_agent.py | 364631aeebdfd9fa3def9f08f0a35d220692441fb56f0677dcfb27deba3cd408 |
| myapp/agent/react_agent.py | 433e2cbc218a9a9dfce842938b5316a76d519e9ff2ec78eb275bcc7be9282fda |
| myapp/agent/workflow_agent.py | b4d4b5df6ab67cc5375c3a7b1236c23dd1edfb7a8f2b669f84ca821c6dfecb14 |
| myapp/agent_views.py | fd7d19929827fddc39501658524b754792483c6c5fde8c7b3cf2084bba77f69e |
| myapp/middleware.py | c80cc6f176ba5c03226f1d8c800ff86d386c212794aab2030ef5382976504409 |
| myapp/recommender/query_rewriter.py | e9bd417cc6dd380f1d289f2993a9bd4a5516327000fde0f93394283fda4e6498 |
| myapp/recommender/recall.py | 9888b0de3057e3e40e78a6cdecb75b1cffa8a4dcdac052b86790b59c4c2cab00 |
| myapp/recommender/rerank.py | 116f64e29f896b52ebd72ab3ce4163e94496f5c88f5c22e6ef16b889e3e2bfe0 |
| myapp/services/agent_chat_service.py | 06cf0bb740f6f6ea55f11203b97a9d483ffaf90859bf8f537a2947d9c9382042 |
| myapp/services/agent_service.py | 6fad5775aaddb344f665830f262acff918d182c435a21d72d1d112577d9c8545 |
| myapp/urls.py | be3a5f777f4c545974ca716bf51e510e630c4000fadf069a106eb8d223a79b5f |
| myapp/views.py | 28dcb6028e59173806234feb78b9d8b844e93e5e5ec74a16aa76b7116e4e3440 |
| requirements.txt | 038a929d0f4a9da933f71a8cc53c437ca34e852f6036a6dd5124af5267c3d7dc |
| docs/REDIS_CACHE_P01_AUDIT.md | e4ef1f1fe2959a165325692b2ab317a63985ade41a70800c9390dccf7b440353 |
| docs/REDIS_CACHE_P0_AUDIT.md | 2d4b0210edaf439459015e7ac990e69a00fca41b944e63edaf8029b2eb1076e4 |
| docs/REDIS_CACHE_P1A_AUDIT.md | b89960e540ea508cf33c46afd34cdcf5e060f791b1d72f045643631790ad24ed |
| docs/REDIS_CACHE_P1B_AUDIT.md | 4413d0e87e4b211da108bbc4c93caf0769db985fce8d02a3e5a0601373b53fc7 |
| myapp/services/cache_mutations.py | 0ef1dd8acdbb5b181c75c8a341ef8d0f9ad9412e96128998cd35b5becb93ce31 |
| myapp/utils/cache_access.py | 2e049d4bc513542d0a4353b526888e368a5c5bf883be7a012aa61aab058deffb |
| myapp/utils/cache_keys.py | 3e5ea9b86a4ba66d4b5fa322840055120037cfb71b7dc4b18fb0ad9d54967a23 |
| tests/test_redis_cache_p01_isolated.py | b4a7d8c61217e90f66e1e7255bea11211080c9049eeeec78e890d2b68e63f922 |
| tests/test_redis_cache_p0_isolated.py | b86f4d4e601ab94b33cd0329b02d29864a2eba04df4999563a447423296aeb07 |
| tests/test_redis_cache_p1a_isolated.py | 7140a3062667616bb35cd61312b2f96d24898577b24157f0d9b9085f32e5a372 |
| tests/test_redis_cache_p1b_isolated.py | e15861d440c4374c152e8a8e5daabd1ef43ee305bace0eced08502901fd8c854 |

核对结果：上述 27 个既有变更文件内容均未由本轮修改；HEAD 与完整跟踪 diff 的 SHA-256 保持一致。

跟踪 diff SHA-256：9d01c155582cfea1f95a1d2f50d8300e0f665c41196471771974deb4a0cb5958。暂存差异为空。

### 58 项回归摘要（实际输出）

- test_redis_cache_p0_isolated.py：exit=0，Ran 14 tests in 0.427s / OK
- test_redis_cache_p01_isolated.py：exit=0，Ran 18 tests in 0.643s / OK
- test_redis_cache_p1a_isolated.py：exit=0，Ran 18 tests in 0.311s / OK
- test_redis_cache_p1b_isolated.py：exit=0，Ran 8 tests in 0.410s / OK

### 所有跟踪文件完整差异

<details>
<summary>展开 git diff --no-ext-diff --no-textconv 完整文本</summary>

格式说明（仅报告展示）：历史 diff 中的原始行尾空白改用可见标记 `⟦行尾:SP×N;TAB×N;CRLF⟧`；仅列出实际存在的项目，顺序与原行尾一致。SP/TAB 表示空格/制表符及数量，CRLF 表示原行使用 CRLF（未标记的行使用 LF）。标记不属于原始代码；仅有 diff 上下文前缀空格的行也显式标记。历史正文、代码及差异内容保留，报告统一使用 LF；此附录用于阅读，不能直接作为补丁应用。

~~~~diff
diff --git a/movie/settings.py b/movie/settings.py
index afd0685..a4db781 100644
--- a/movie/settings.py
+++ b/movie/settings.py
@@ -66,6 +66,7 @@ MIDDLEWARE = [
     'django.contrib.auth.middleware.AuthenticationMiddleware',⟦行尾:CRLF⟧
     'django.contrib.messages.middleware.MessageMiddleware',⟦行尾:CRLF⟧
     'django.middleware.clickjacking.XFrameOptionsMiddleware',⟦行尾:CRLF⟧
+    'myapp.middleware.CommittedCacheFailureMiddleware',
     'myapp.middleware.RemoveHopByHopHeaders',⟦行尾:CRLF⟧
 ]⟦行尾:CRLF⟧
⟦行尾:SP×1;CRLF⟧
diff --git a/myapp/admin.py b/myapp/admin.py
index 9817f6b..564a308 100644
--- a/myapp/admin.py
+++ b/myapp/admin.py
@@ -1,11 +1,13 @@
 # 文件: myapp/admin.py⟦行尾:CRLF⟧
⟦行尾:SP×1;CRLF⟧
 from django.contrib import admin⟦行尾:CRLF⟧
+from myapp.services.cache_mutations import RecommendationCacheAdminMixin
 from .models import UserInfo, Movie, Genre, Actor, Region, UserRating, Collect, Rec, ChatHistory⟦行尾:CRLF⟧
⟦行尾:SP×1;CRLF⟧
⟦行尾:SP×1;CRLF⟧
 # 1. 注册 UserInfo (修复版)⟦行尾:CRLF⟧
-class UserInfoAdmin(admin.ModelAdmin):⟦行尾:CRLF⟧
+class UserInfoAdmin(RecommendationCacheAdminMixin, admin.ModelAdmin):
+    cache_owner_field = 'pk'
     # --- ↓↓↓ 关键修复 ↓↓↓ ---⟦行尾:CRLF⟧
     # 原来的 user_type -> 改为 is_staff (是否管理员)⟦行尾:CRLF⟧
     # 原来的 user_status -> 改为 is_active (是否有效)⟦行尾:CRLF⟧
@@ -23,7 +25,8 @@ admin.site.register(UserInfo, UserInfoAdmin)
⟦行尾:SP×1;CRLF⟧
⟦行尾:SP×1;CRLF⟧
 # 2. 注册 Movie⟦行尾:CRLF⟧
-class MovieAdmin(admin.ModelAdmin):⟦行尾:CRLF⟧
+class MovieAdmin(RecommendationCacheAdminMixin, admin.ModelAdmin):
+    cache_scope = 'catalog'
     # 确保这里没有引用不存在的字段⟦行尾:CRLF⟧
     list_display = ('id', 'title', 'score', 'vote_count', 'date')⟦行尾:CRLF⟧
     search_fields = ('title',)⟦行尾:CRLF⟧
@@ -34,7 +37,7 @@ admin.site.register(Movie, MovieAdmin)
⟦行尾:SP×1;CRLF⟧
⟦行尾:SP×1;CRLF⟧
 # 3. 注册 UserRating⟦行尾:CRLF⟧
-class UserRatingAdmin(admin.ModelAdmin):⟦行尾:CRLF⟧
+class UserRatingAdmin(RecommendationCacheAdminMixin, admin.ModelAdmin):
     list_display = ('id', 'user', 'movie', 'score', 'discussion', 'comment_time')⟦行尾:CRLF⟧
     search_fields = ('user__username', 'movie__title', 'discussion')⟦行尾:CRLF⟧
     list_filter = ('score', 'comment_time')⟦行尾:CRLF⟧
@@ -44,7 +47,7 @@ admin.site.register(UserRating, UserRatingAdmin)
⟦行尾:SP×1;CRLF⟧
⟦行尾:SP×1;CRLF⟧
 # 4. 注册 Collect⟦行尾:CRLF⟧
-class CollectAdmin(admin.ModelAdmin):⟦行尾:CRLF⟧
+class CollectAdmin(RecommendationCacheAdminMixin, admin.ModelAdmin):
     list_display = ('id', 'user', 'movie', 'collect_time')⟦行尾:CRLF⟧
     search_fields = ('user__username', 'movie__title')⟦行尾:CRLF⟧
⟦行尾:SP×1;CRLF⟧
@@ -53,7 +56,7 @@ admin.site.register(Collect, CollectAdmin)
⟦行尾:SP×1;CRLF⟧
⟦行尾:SP×1;CRLF⟧
 # 5. 注册 Rec (推荐结果)⟦行尾:CRLF⟧
-class RecAdmin(admin.ModelAdmin):⟦行尾:CRLF⟧
+class RecAdmin(RecommendationCacheAdminMixin, admin.ModelAdmin):
     list_display = ('id', 'user', 'movie', 'rating')⟦行尾:CRLF⟧
     search_fields = ('user__username', 'movie__title')⟦行尾:CRLF⟧
⟦行尾:SP×1;CRLF⟧
@@ -61,9 +64,13 @@ class RecAdmin(admin.ModelAdmin):
 admin.site.register(Rec, RecAdmin)⟦行尾:CRLF⟧
⟦行尾:SP×1;CRLF⟧
 # 6. 注册其他基础表⟦行尾:CRLF⟧
-admin.site.register(Genre)⟦行尾:CRLF⟧
-admin.site.register(Actor)⟦行尾:CRLF⟧
-admin.site.register(Region)⟦行尾:CRLF⟧
+class CatalogAdmin(RecommendationCacheAdminMixin, admin.ModelAdmin):
+    cache_scope = 'catalog'
+
+
+admin.site.register(Genre, CatalogAdmin)
+admin.site.register(Actor, CatalogAdmin)
+admin.site.register(Region, CatalogAdmin)
 admin.site.register(ChatHistory)⟦行尾:CRLF⟧
⟦行尾:SP×1;CRLF⟧
⟦行尾:SP×1;CRLF⟧
@@ -84,14 +91,14 @@ class ConversationHistoryAdmin(admin.ModelAdmin):
⟦行尾:SP×1;CRLF⟧
⟦行尾:SP×1;CRLF⟧
 @admin.register(UserProfile)⟦行尾:CRLF⟧
-class UserProfileAdmin(admin.ModelAdmin):⟦行尾:CRLF⟧
+class UserProfileAdmin(RecommendationCacheAdminMixin, admin.ModelAdmin):
     list_display = ('user', 'total_ratings', 'avg_score', 'total_collections', 'is_cold_start', 'updated_at')⟦行尾:CRLF⟧
     search_fields = ('user__username',)⟦行尾:CRLF⟧
     list_filter = ('is_cold_start',)⟦行尾:CRLF⟧
⟦行尾:SP×1;CRLF⟧
⟦行尾:SP×1;CRLF⟧
 @admin.register(RecommendLog)⟦行尾:CRLF⟧
-class RecommendLogAdmin(admin.ModelAdmin):⟦行尾:CRLF⟧
+class RecommendLogAdmin(RecommendationCacheAdminMixin, admin.ModelAdmin):
     list_display = ('id', 'user', 'strategy', 'latency_ms', 'created_at')⟦行尾:CRLF⟧
     search_fields = ('user__username', 'query_text')⟦行尾:CRLF⟧
     list_filter = ('strategy', 'created_at')⟦行尾:CRLF⟧
@@ -105,7 +112,7 @@ class AgentTraceAdmin(admin.ModelAdmin):
⟦行尾:SP×1;CRLF⟧
⟦行尾:SP×1;CRLF⟧
 @admin.register(UserFeedback)⟦行尾:CRLF⟧
-class UserFeedbackAdmin(admin.ModelAdmin):⟦行尾:CRLF⟧
+class UserFeedbackAdmin(RecommendationCacheAdminMixin, admin.ModelAdmin):
     list_display = ('id', 'user', 'movie', 'feedback_type', 'source', 'created_at')⟦行尾:CRLF⟧
     search_fields = ('user__username', 'movie__title')⟦行尾:CRLF⟧
     list_filter = ('feedback_type', 'source', 'created_at')⟦行尾:CRLF⟧
diff --git a/myapp/agent/memory.py b/myapp/agent/memory.py
index c8258f8..1852103 100644
--- a/myapp/agent/memory.py
+++ b/myapp/agent/memory.py
@@ -29,6 +29,10 @@ import time
 import logging
 from collections import defaultdict
 from django.core.cache import cache
+from django.db import transaction
+from myapp.services.cache_mutations import schedule_user_invalidation
+from myapp.utils.cache_access import memory_cache_operation, MemoryCacheUnavailable
+from myapp.utils.cache_keys import memory_cache_key
⟦行尾:SP×1⟧
 logger = logging.getLogger('movie_agent')
⟦行尾:SP×1⟧
@@ -130,8 +134,9 @@ class MemoryManager:
             session_id: 会话ID（用于匿名用户或缓存键）
         """
         self.user = user
-        self.session_id = session_id or 'default'
-        self._cache_key = f"agent_memory_{self.session_id}"
+        self.session_id = session_id
+        self._cache_key = memory_cache_key(session_id, getattr(user, 'id', None))
+        self._slots = {}
⟦行尾:SP×5⟧
     def get_slots(self):
         """
@@ -140,9 +145,11 @@ class MemoryManager:
         Returns:
             Dict: 当前槽位字典
         """
-        cached = cache.get(self._cache_key)
+        if self._cache_key is None:
+            return dict(self._slots)
+        cached = memory_cache_operation(cache, 'get', self._cache_key)
         if cached and isinstance(cached, dict):
-            return cached
+            return dict(cached)
         return {}
⟦行尾:SP×5⟧
     def update_slots(self, user_input):
@@ -156,6 +163,13 @@ class MemoryManager:
             Dict: 更新后的槽位字典
         """
         slots = self.get_slots()
+        # 支持直接传入 dict 更新槽位（如 LLM 解析结果）
+        if isinstance(user_input, dict):
+            for key, value in user_input.items():
+                if value is not None and value != '':
+                    slots[key] = value
+            self._save_slots(slots)
+            return slots
         newly_extracted = self.extract_slots(user_input)
⟦行尾:SP×9⟧
         # 合并槽位（新值覆盖旧值，空值不覆盖）
@@ -170,12 +184,19 @@ class MemoryManager:
                     slots.pop(key, None)
⟦行尾:SP×9⟧
         # 保存到缓存
-        cache.set(self._cache_key, slots, self.SLOT_TTL)
+        self._save_slots(slots)
         return slots
⟦行尾:SP×5⟧
+    def _save_slots(self, slots):
+        if self._cache_key is not None:
+            memory_cache_operation(cache, 'set', self._cache_key, slots, self.SLOT_TTL)
+        self._slots = dict(slots)
+
     def clear_slots(self):
         """清除当前会话的所有槽位"""
-        cache.delete(self._cache_key)
+        if self._cache_key is not None:
+            memory_cache_operation(cache, 'delete', self._cache_key)
+        self._slots = {}
⟦行尾:SP×5⟧
     def extract_slots(self, text):
         """
@@ -319,39 +340,42 @@ class MemoryManager:
             return
⟦行尾:SP×9⟧
         try:
-            from myapp.models_upgrade import UserProfile
-            slots = self.get_slots()
-⟦行尾:SP×12⟧
-            profile, created = UserProfile.objects.get_or_create(user=self.user)
-⟦行尾:SP×12⟧
-            # 合并类型偏好到 genre_preferences (dict格式: {"科幻": 0.85, ...})
-            genre_prefs = profile.genre_preferences or {}
-            new_genre = slots.get('genre', '')
-            if new_genre:
-                genre_prefs[new_genre] = genre_prefs.get(new_genre, 0) + 1.0
-                profile.genre_preferences = genre_prefs
-⟦行尾:SP×12⟧
-            # 合并导演偏好
-            director_prefs = profile.director_preferences or {}
-            new_director = slots.get('director', '')
-            if new_director:
-                director_prefs[new_director] = director_prefs.get(new_director, 0) + 1.0
-                profile.director_preferences = director_prefs
-⟦行尾:SP×12⟧
-            # 更新画像文本（包含槽位摘要）
-            slots_text = ", ".join(f"{k}={v}" for k, v in slots.items() if v)
-            if slots_text:
-                profile_text = profile.profile_text or ''
-                if slots_text not in profile_text:
-                    profile.profile_text = f"{profile_text}\n最近偏好: {slots_text}".strip()
-⟦行尾:SP×12⟧
-            # 标记非冷启动
-            if slots:
-                profile.is_cold_start = False
-⟦行尾:SP×12⟧
-            profile.save()
+            with transaction.atomic():
+                from myapp.models_upgrade import UserProfile
+                slots = self.get_slots()
+
+                profile, created = UserProfile.objects.get_or_create(user=self.user)
+
+                # 合并类型偏好到 genre_preferences (dict格式: {"科幻": 0.85, ...})
+                genre_prefs = profile.genre_preferences or {}
+                new_genre = slots.get('genre', '')
+                if new_genre:
+                    genre_prefs[new_genre] = genre_prefs.get(new_genre, 0) + 1.0
+                    profile.genre_preferences = genre_prefs
+
+                # 合并导演偏好
+                director_prefs = profile.director_preferences or {}
+                new_director = slots.get('director', '')
+                if new_director:
+                    director_prefs[new_director] = director_prefs.get(new_director, 0) + 1.0
+                    profile.director_preferences = director_prefs
+
+                # 更新画像文本（包含槽位摘要）
+                slots_text = ", ".join(f"{k}={v}" for k, v in slots.items() if v)
+                if slots_text:
+                    profile_text = profile.profile_text or ''
+                    if slots_text not in profile_text:
+                        profile.profile_text = f"{profile_text}\n最近偏好: {slots_text}".strip()
+
+                # 标记非冷启动
+                if slots:
+                    profile.is_cold_start = False
+
+                profile.save()
+                schedule_user_invalidation(self.user.id)
         except Exception as e:
-            logger.error(f"[Memory] 持久化失败: {e}")
+            logger.error(f"[Memory] 画像操作未完整完成 error_type={type(e).__name__}")
+            raise
⟦行尾:SP×5⟧
     # ===========================================================
     # 会话总结与画像补充（类似 Gemini/ChatGPT 记忆机制）
@@ -522,69 +546,71 @@ class MemoryManager:
           - profile_text: 文本摘要
         """
         try:
-            from myapp.models_upgrade import UserProfile
-⟦行尾:SP×12⟧
-            profile, created = UserProfile.objects.get_or_create(user=self.user)
-⟦行尾:SP×12⟧
-            # ── 合并类型偏好 ──
-            genre_prefs = profile.genre_preferences or {}
-            for genre in preferences.get('genres', []):
-                if genre and genre not in preferences.get('exclude_genres', []):
-                    genre_prefs[genre] = genre_prefs.get(genre, 0) + 1.0
-            # 排除类型降低权重而非删除
-            for genre in preferences.get('exclude_genres', []):
-                if genre in genre_prefs:
-                    genre_prefs[genre] = max(0, genre_prefs[genre] - 0.5)
-            profile.genre_preferences = genre_prefs
-⟦行尾:SP×12⟧
-            # ── 合并导演偏好 ──
-            director_prefs = profile.director_preferences or {}
-            for director in preferences.get('directors', []):
-                if director:
-                    director_prefs[director] = director_prefs.get(director, 0) + 1.0
-            profile.director_preferences = director_prefs
-⟦行尾:SP×12⟧
-            # ── 合并演员偏好 ──
-            actor_prefs = profile.actor_preferences or {}
-            for actor in preferences.get('actors', []):
-                if actor:
-                    actor_prefs[actor] = actor_prefs.get(actor, 0) + 1.0
-            profile.actor_preferences = actor_prefs
-⟦行尾:SP×12⟧
-            # ── 更新画像文本（含锚点电影、情感偏好）──
-            summary_parts = []
-            if preferences.get('anchor_movies'):
-                summary_parts.append(f"参考电影: {'、'.join(preferences['anchor_movies'])}")
-            if preferences.get('sentiment'):
-                summary_parts.append(f"情感偏好: {'、'.join(preferences['sentiment'])}")
-            if preferences.get('score_min'):
-                summary_parts.append(f"最低评分: {preferences['score_min']}")
-⟦行尾:SP×12⟧
-            if summary_parts:
-                new_summary = "; ".join(summary_parts)
-                existing_text = profile.profile_text or ''
-                if new_summary not in existing_text:
-                    profile.profile_text = f"{existing_text}\n{new_summary}".strip()
-⟦行尾:SP×12⟧
-            # 标记非冷启动
-            profile.is_cold_start = False
-            profile.save()
+            with transaction.atomic():
+                from myapp.models_upgrade import UserProfile
+
+                profile, created = UserProfile.objects.get_or_create(user=self.user)
+
+                # ── 合并类型偏好 ──
+                genre_prefs = profile.genre_preferences or {}
+                for genre in preferences.get('genres', []):
+                    if genre and genre not in preferences.get('exclude_genres', []):
+                        genre_prefs[genre] = genre_prefs.get(genre, 0) + 1.0
+                # 排除类型降低权重而非删除
+                for genre in preferences.get('exclude_genres', []):
+                    if genre in genre_prefs:
+                        genre_prefs[genre] = max(0, genre_prefs[genre] - 0.5)
+                profile.genre_preferences = genre_prefs
⟦行尾:SP×13⟧
-            # 准备返回值
-            all_genres = sorted(genre_prefs.keys(), key=lambda g: genre_prefs[g], reverse=True)
-            all_directors = sorted(director_prefs.keys(), key=lambda d: director_prefs[d], reverse=True)
+                # ── 合并导演偏好 ──
+                director_prefs = profile.director_preferences or {}
+                for director in preferences.get('directors', []):
+                    if director:
+                        director_prefs[director] = director_prefs.get(director, 0) + 1.0
+                profile.director_preferences = director_prefs
⟦行尾:SP×13⟧
-            return {
-                'genres': all_genres,
-                'directors': all_directors,
-                'anchor_movies': preferences.get('anchor_movies', []),
-                'sentiment': preferences.get('sentiment', []),
-                'exclude_genres': preferences.get('exclude_genres', []),
-                'score_min': preferences.get('score_min'),
-            }
+                # ── 合并演员偏好 ──
+                actor_prefs = profile.actor_preferences or {}
+                for actor in preferences.get('actors', []):
+                    if actor:
+                        actor_prefs[actor] = actor_prefs.get(actor, 0) + 1.0
+                profile.actor_preferences = actor_prefs
+
+                # ── 更新画像文本（含锚点电影、情感偏好）──
+                summary_parts = []
+                if preferences.get('anchor_movies'):
+                    summary_parts.append(f"参考电影: {'、'.join(preferences['anchor_movies'])}")
+                if preferences.get('sentiment'):
+                    summary_parts.append(f"情感偏好: {'、'.join(preferences['sentiment'])}")
+                if preferences.get('score_min'):
+                    summary_parts.append(f"最低评分: {preferences['score_min']}")
+
+                if summary_parts:
+                    new_summary = "; ".join(summary_parts)
+                    existing_text = profile.profile_text or ''
+                    if new_summary not in existing_text:
+                        profile.profile_text = f"{existing_text}\n{new_summary}".strip()
+
+                # 标记非冷启动
+                profile.is_cold_start = False
+                profile.save()
+                schedule_user_invalidation(self.user.id)
+
+                # 准备返回值
+                all_genres = sorted(genre_prefs.keys(), key=lambda g: genre_prefs[g], reverse=True)
+                all_directors = sorted(director_prefs.keys(), key=lambda d: director_prefs[d], reverse=True)
+
+                return {
+                    'genres': all_genres,
+                    'directors': all_directors,
+                    'anchor_movies': preferences.get('anchor_movies', []),
+                    'sentiment': preferences.get('sentiment', []),
+                    'exclude_genres': preferences.get('exclude_genres', []),
+                    'score_min': preferences.get('score_min'),
+                }
         except Exception as e:
-            logger.error(f"[Memory] 画像合并失败: {e}")
-            return {}
+            logger.error(f"[Memory] 画像操作未完整完成 error_type={type(e).__name__}")
+            raise
⟦行尾:SP×5⟧
     def get_profile_summary(self):
         """
diff --git a/myapp/agent/movie_agent.py b/myapp/agent/movie_agent.py
index 79a08c2..abc12b3 100644
--- a/myapp/agent/movie_agent.py
+++ b/myapp/agent/movie_agent.py
@@ -18,6 +18,8 @@ import json
 import logging
 import traceback
 from collections import Counter
+from myapp.utils.cache_keys import intent_cache_key
+from myapp.utils.cache_access import business_cache_get, business_cache_set
⟦行尾:SP×1⟧
 logger = logging.getLogger('movie_agent')
⟦行尾:SP×1⟧
@@ -436,10 +438,10 @@ Output: {"user_id":"anonymous","query_text":"有什么好玩的 喜剧 电影吗
             return fallback
⟦行尾:SP×1⟧
         # Redis 缓存：相同查询复用意图解析结果（TTL 10 分钟）
-        cache_key = f"intent:{hash(user_input.strip())}"
+        cache_key = intent_cache_key(user_input)
         try:
             from django.core.cache import cache
-            cached = cache.get(cache_key)
+            cached = business_cache_get(cache, cache_key)
             if cached is not None:
                 return cached
         except Exception:
@@ -478,7 +480,7 @@ Output: {"user_id":"anonymous","query_text":"有什么好玩的 喜剧 电影吗
⟦行尾:SP×1⟧
             # 写入缓存（TTL 10 分钟）
             try:
-                cache.set(cache_key, result, timeout=600)
+                business_cache_set(cache, cache_key, result, 600)
             except Exception:
                 pass
⟦行尾:SP×1⟧
@@ -1835,9 +1837,10 @@ class RerankTool(AgentTool):
     name = "rerank"
     description = "对候选电影进行多样性重排"
⟦行尾:SP×5⟧
-    def execute(self, candidates=None, user=None, top_k=10, **kwargs):
+    def execute(self, candidates=None, user=None, top_k=10, memory_slots=None, **kwargs):
         from myapp.recommender.rerank import final_rerank
-        results, stats = final_rerank(candidates or [], user=user, top_k=top_k)
+        results, stats = final_rerank(candidates or [], user=user, top_k=top_k,
+                                     memory_slots=memory_slots)
         return {
             'tool': self.name,
             'input': f"{len(candidates or [])} candidates",
@@ -2515,7 +2518,7 @@ class MovieAgent:
         from myapp.agent.memory import MemoryManager
         self.memory = MemoryManager(
             user=user,
-            session_id=session_id or f"user_{getattr(user, 'id', 'anon')}"
+            session_id=session_id
         )
⟦行尾:SP×5⟧
     def run(self, user_input, is_thinking_mode=False):
@@ -3324,7 +3327,9 @@ class MovieAgent:
                 slots = self.memory.get_slots() if hasattr(self, 'memory') and self.memory else None
                 result = tool.execute(candidates=current_candidates, user=self.user, top_k=15, memory_slots=slots)
             elif tool_name == 'rerank':
-                result = tool.execute(candidates=current_candidates, user=self.user, top_k=15)
+                slots = self.memory.get_slots() if hasattr(self, 'memory') and self.memory else None
+                result = tool.execute(candidates=current_candidates, user=self.user, top_k=15,
+                                      memory_slots=slots)
             elif tool_name == 'explain':
                 result = tool.execute(user=self.user)
             else:
@@ -4177,4 +4182,4 @@ class MovieAgent:
         lines.append(result.get('final_answer', ''))
         lines.append("=" * 60)
⟦行尾:SP×9⟧
-        return "\n".join(lines)
\ No newline at end of file
+        return "\n".join(lines)
diff --git a/myapp/agent/react_agent.py b/myapp/agent/react_agent.py
index 109d97b..6e21511 100644
--- a/myapp/agent/react_agent.py
+++ b/myapp/agent/react_agent.py
@@ -50,7 +50,7 @@ class ReActAgent:
         self.user = user
         self.neo_graph = neo_graph
         self.rag_resources = rag_resources
-        self.session_id = session_id or f"react_{int(time.time())}"
+        self.session_id = session_id
         self.trace_enabled = trace_enabled
⟦行尾:SP×1⟧
         # 懒加载工具
@@ -352,7 +352,8 @@ class ReActAgent:
             elif tool_name == 'maan_rerank':
                 result = tool.execute(candidates=current_candidates, user=self.user, top_k=15)
             elif tool_name == 'rerank':
-                result = tool.execute(candidates=current_candidates, user=self.user, top_k=15)
+                result = tool.execute(candidates=current_candidates, user=self.user, top_k=15,
+                                      memory_slots=self.memory.get_slots())
             elif tool_name == 'explain':
                 result = tool.execute(user=self.user)
             else:
@@ -495,4 +496,4 @@ class ReActAgent:
         temp = MovieAgent(user=self.user)
         return temp._generate_final_answer(
             user_input, intent, recommended_ids, explanations, ''
-        )
\ No newline at end of file
+        )
diff --git a/myapp/agent/workflow_agent.py b/myapp/agent/workflow_agent.py
index 69ab4a0..6161ff4 100644
--- a/myapp/agent/workflow_agent.py
+++ b/myapp/agent/workflow_agent.py
@@ -51,7 +51,7 @@ class WorkflowAgent:
         self.user = user
         self.neo_graph = neo_graph
         self.rag_resources = rag_resources
-        self.session_id = session_id or f"workflow_{int(time.time())}"
+        self.session_id = session_id
⟦行尾:SP×1⟧
         self._tools = None
⟦行尾:SP×1⟧
@@ -125,7 +125,8 @@ class WorkflowAgent:
                 elif tool_name == 'maan_rerank':
                     result = tool.execute(candidates=candidates, user=self.user, top_k=15)
                 elif tool_name == 'rerank':
-                    result = tool.execute(candidates=candidates, user=self.user, top_k=15)
+                    result = tool.execute(candidates=candidates, user=self.user, top_k=15,
+                                          memory_slots=self.memory.get_slots())
                 else:
                     result = {'tool': tool_name, 'output': [], 'count': 0}
⟦行尾:SP×1⟧
@@ -250,4 +251,4 @@ class SequentialPipelineAgent(WorkflowAgent):
     def run(self, user_input: str) -> dict:
         result = super().run(user_input)
         result['agent_type'] = 'sequential_pipeline'
-        return result
\ No newline at end of file
+        return result
diff --git a/myapp/agent_views.py b/myapp/agent_views.py
index 6b31685..7cfb5bc 100644
--- a/myapp/agent_views.py
+++ b/myapp/agent_views.py
@@ -30,7 +30,16 @@ from django.contrib.auth.decorators import login_required
 from django.views.decorators.csrf import csrf_exempt
 from django.views.decorators.http import require_POST
 from django.contrib import messages
+from django.core.cache import cache
+from myapp.services.cache_mutations import (
+    recommendation_write_view, schedule_user_invalidation, schedule_catalog_invalidation,
+    partial_commit_response,
+)
+from django.db import transaction
 from django.db.models import Q
+from myapp.utils.cache_keys import (
+    invalidate_recommendation_cache, invalidate_catalog_cache, RecommendationCacheInvalidationError,
+)
⟦行尾:SP×1⟧
 from myapp.models import Movie, UserRating, ChatHistory, Genre, Region
⟦行尾:SP×1⟧
@@ -46,8 +55,12 @@ def _sanitize_input(raw_input):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 def _get_session_key(request):
-    """获取会话标识"""
-    return request.session.session_key or 'default'
+    """获取真实会话标识；Session 后端故障向上报告。"""
+    if not request.session.session_key:
+        request.session.create()
+    if not request.session.session_key:
+        raise RuntimeError("Session backend did not create a session key")
+    return request.session.session_key
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 def _admin_required_api(view_func):
@@ -212,11 +225,15 @@ def agent_stream_view(request):
             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
         )
⟦行尾:SP×5⟧
+    # SessionMiddleware sees the new session before processing the response.
+    # Never create a session while consuming streaming_content.
+    session_key = _get_session_key(request)
+
     def event_stream():
         try:
             from myapp.services.agent_chat_service import AgentChatService
⟦行尾:SP×13⟧
-            service = AgentChatService(request.user, _get_session_key(request))
+            service = AgentChatService(request.user, session_key)
             is_thinking = request.POST.get('is_thinking') == 'true'
⟦行尾:SP×13⟧
             for chunk in service.stream_chat(user_input, is_thinking):
@@ -280,6 +297,7 @@ def movie_explain_view(request, pk):
 @csrf_exempt
 @login_required
 @require_POST
+@recommendation_write_view
 def recommend_feedback_view(request):
     """推荐反馈收集接口"""
     movie_id = request.POST.get('movie_id')
@@ -293,18 +311,24 @@ def recommend_feedback_view(request):
         from myapp.models_upgrade import UserFeedback
         movie = get_object_or_404(Movie, pk=movie_id)
⟦行尾:SP×9⟧
-        UserFeedback.objects.create(
-            user=request.user,
-            movie=movie,
-            feedback_type=feedback_type,
-            source=source,
-        )
+        with transaction.atomic():
+            UserFeedback.objects.create(
+                user=request.user,
+                movie=movie,
+                feedback_type=feedback_type,
+                source=source,
+            )
+            transaction.on_commit(
+                lambda user_id=request.user.id: invalidate_recommendation_cache(cache, user_id)
+            )
⟦行尾:SP×9⟧
         return JsonResponse({
             'status': 'success',
             'msg': '感谢您的反馈！',
             'feedback_type': feedback_type,
         })
+    except RecommendationCacheInvalidationError as exc:
+        return partial_commit_response(exc)
     except Exception as e:
         logger.error(f"[Feedback] 保存失败: {e}")
         return JsonResponse({'status': 'error', 'msg': str(e)})
@@ -314,6 +338,35 @@ def recommend_feedback_view(request):
 # 视图 5: 对话管理 API
 # =============================================================
⟦行尾:SP×1⟧
+@login_required
+@require_POST
+def retry_cache_invalidation(request):
+    """Retry invalidation only; never re-run a committed database mutation."""
+    scope = request.POST.get('scope', 'user')
+    if scope not in ('user', 'catalog'):
+        return JsonResponse({'status': 'error', 'msg': '无效缓存范围'}, status=400)
+    try:
+        user_id = int(request.POST.get('user_id', request.user.id)) if scope == 'user' else None
+    except (TypeError, ValueError):
+        return JsonResponse({'status': 'error', 'msg': '无效用户标识'}, status=400)
+    if scope == 'user' and user_id <= 0:
+        return JsonResponse({'status': 'error', 'msg': '无效用户标识'}, status=400)
+    if scope == 'catalog' or user_id != request.user.id:
+        if not request.user.is_staff:
+            return JsonResponse({'status': 'error', 'msg': '无权失效其他用户或目录缓存'}, status=403)
+    try:
+        if scope == 'catalog':
+            invalidate_catalog_cache(cache)
+        else:
+            invalidate_recommendation_cache(cache, user_id)
+    except RecommendationCacheInvalidationError:
+        return JsonResponse({
+            'status': 'error', 'database_mutation': False, 'retry_mutation': False,
+            'retry_cache_invalidation': True, 'cache_invalidated': False,
+            'msg': '缓存失效重试未完成；未重复写入用户数据。',
+        }, status=503)
+    return JsonResponse({'status': 'ok', 'database_mutation': False, 'cache_invalidated': True})
+
 @csrf_exempt
 @login_required
 def ajax_clear_chat(request):
@@ -337,6 +390,7 @@ def ajax_clear_chat(request):
⟦行尾:SP×1⟧
 @csrf_exempt
 @login_required
+@recommendation_write_view
 def ajax_new_chat(request):
     """新建对话：总结当前会话 → 补充画像 → 清除历史"""
     if request.method != 'POST':
@@ -435,6 +489,7 @@ def ajax_chat_history(request):
⟦行尾:SP×1⟧
 @csrf_exempt
 @login_required
+@recommendation_write_view
 def ajax_summarize_profile(request):
     """手动触发会话总结 → 画像补充"""
     if request.method != 'POST':
@@ -481,6 +536,7 @@ def admin_user_profile_list(request):
⟦行尾:SP×1⟧
 @login_required
 @_admin_required_view
+@recommendation_write_view
 def admin_user_profile_edit(request, pk):
     from myapp.models_upgrade import UserProfile
     profile = get_object_or_404(UserProfile, pk=pk)
@@ -488,7 +544,9 @@ def admin_user_profile_edit(request, pk):
     if request.method == 'POST':
         profile.profile_text = request.POST.get('profile_text', '')
         profile.is_cold_start = request.POST.get('is_cold_start') == 'on'
+        affected_user_id = profile.user_id
         profile.save()
+        schedule_user_invalidation(affected_user_id)
         messages.success(request, f"用户 {profile.user.username} 的画像已更新")
         return redirect('admin_user_profile_list')
⟦行尾:SP×1⟧
@@ -497,13 +555,16 @@ def admin_user_profile_edit(request, pk):
⟦行尾:SP×1⟧
 @login_required
 @_admin_required_view
+@recommendation_write_view
 def admin_user_profile_delete(request, pk):
     from myapp.models_upgrade import UserProfile
     profile = get_object_or_404(UserProfile, pk=pk)
⟦行尾:SP×1⟧
     if request.method == 'POST':
         username = profile.user.username
+        affected_user_id = profile.user_id
         profile.delete()
+        schedule_user_invalidation(affected_user_id)
         messages.success(request, f"用户 {username} 的画像已删除")
         return redirect('admin_user_profile_list')
⟦行尾:SP×1⟧
@@ -588,12 +649,15 @@ def admin_user_feedback_list(request):
⟦行尾:SP×1⟧
 @login_required
 @_admin_required_view
+@recommendation_write_view
 def admin_user_feedback_delete(request, pk):
     from myapp.models_upgrade import UserFeedback
     feedback = get_object_or_404(UserFeedback, pk=pk)
⟦行尾:SP×1⟧
     if request.method == 'POST':
+        affected_user_id = feedback.user_id
         feedback.delete()
+        schedule_user_invalidation(affected_user_id)
         messages.success(request, "反馈记录已删除")
         return redirect('admin_user_feedback_list')
⟦行尾:SP×1⟧
@@ -1189,4 +1253,4 @@ def ajax_agent_kg_query(request):
         return JsonResponse({
             "nodes": [], "links": [], "categories": [],
             "triples": [], "error": str(e)
-        })
\ No newline at end of file
+        })
diff --git a/myapp/middleware.py b/myapp/middleware.py
index 81b1cdb..8665e90 100644
--- a/myapp/middleware.py
+++ b/myapp/middleware.py
@@ -1,3 +1,21 @@
+from myapp.utils.cache_keys import RecommendationCacheInvalidationError
+from myapp.services.cache_mutations import partial_commit_response
+
+
+class CommittedCacheFailureMiddleware:
+    """Also classify callback failures from an outer view transaction."""
+    def __init__(self, get_response):
+        self.get_response = get_response
+
+    def __call__(self, request):
+        return self.get_response(request)
+
+    def process_exception(self, request, exception):
+        if isinstance(exception, RecommendationCacheInvalidationError):
+            return partial_commit_response(exception)
+        return None
+
+
 class RemoveHopByHopHeaders:
     """移除 wsgiref 不允许的 hop-by-hop 响应头，修复 Django 4.2 的 AssertionError"""
⟦行尾:SP×1⟧
diff --git a/myapp/recommender/query_rewriter.py b/myapp/recommender/query_rewriter.py
index 0b3d029..9020c53 100644
--- a/myapp/recommender/query_rewriter.py
+++ b/myapp/recommender/query_rewriter.py
@@ -14,6 +14,8 @@ Query Rewrite 模块 (Query Rewriter Module)
 import re
 import logging
 from django.core.cache import cache
+from myapp.utils.cache_access import business_cache_get, business_cache_set
+from myapp.utils.cache_keys import query_rewrite_cache_key
⟦行尾:SP×1⟧
 logger = logging.getLogger('movie_agent')
⟦行尾:SP×1⟧
@@ -77,9 +79,9 @@ class QueryRewriter:
         query_clean = query.strip()
⟦行尾:SP×1⟧
         # 1. 尝试缓存命中
-        cache_key = f"query_rewrite_{hash(query_clean)}"
-        cached_result = cache.get(cache_key)
-        if cached_result:
+        cache_key = query_rewrite_cache_key(query_clean)
+        cached_result = business_cache_get(cache, cache_key)
+        if cached_result is not None:
             return cached_result
⟦行尾:SP×1⟧
         rewritten = ""
@@ -109,7 +111,7 @@ class QueryRewriter:
             rewritten = self._rule_based_rewrite(query_clean)
⟦行尾:SP×1⟧
         # 4. 写入缓存 (1小时)
-        cache.set(cache_key, rewritten, 3600)
+        business_cache_set(cache, cache_key, rewritten, 3600)
         return rewritten
⟦行尾:SP×1⟧
     def _rule_based_rewrite(self, query: str) -> str:
diff --git a/myapp/recommender/recall.py b/myapp/recommender/recall.py
index 4e35ae8..68de3f1 100644
--- a/myapp/recommender/recall.py
+++ b/myapp/recommender/recall.py
@@ -20,6 +20,8 @@ import numpy as np
 from collections import Counter
 from django.core.cache import cache
 from django.db.models import Q, Count, Max
+from myapp.utils.cache_access import business_cache_get, business_cache_set
+from myapp.utils.cache_keys import multi_recall_cache_key
⟦行尾:SP×1⟧
 from myapp.models import Movie, UserRating, Collect, Genre, Actor, Rec
⟦行尾:SP×1⟧
@@ -315,11 +317,13 @@ def multi_channel_recall(user, query_text=None, top_k=30,
     """
     t_start = time.time()
⟦行尾:SP×1⟧
-    # 缓存
-    cache_key = f"multi_recall_{user.id}_{hash(query_text or '')}"
-    if not force_refresh:
-        cached = cache.get(cache_key)
-        if cached:
+    # Cache only the recomputable result. User/catalog generations invalidate
+    # this family alongside other recommendation entries; parameters are part
+    # of the stable digest so top_k/weights cannot reuse an incompatible value.
+    cache_key = multi_recall_cache_key(cache, user.id, query_text, top_k, weights)
+    if cache_key is not None and not force_refresh:
+        cached = business_cache_get(cache, cache_key)
+        if cached is not None:
             return cached
⟦行尾:SP×1⟧
     # 排除已看过的电影
@@ -369,7 +373,7 @@ def multi_channel_recall(user, query_text=None, top_k=30,
     stats['latency_ms'] = int((time.time() - t_start) * 1000)
⟦行尾:SP×1⟧
     result = (final_results, stats)
-    cache.set(cache_key, result, 1800)  # 缓存30分钟
+    business_cache_set(cache, cache_key, result, 1800)  # 缓存30分钟
     return result
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
@@ -415,4 +419,4 @@ def _build_profile_query(user, liked_ids, top_n_genres=5, top_n_actors=5):
     if recent_high:
         parts.append(f"近期高分电影：{'、'.join(recent_high)}")
⟦行尾:SP×1⟧
-    return "；".join(parts) if parts else "优质电影 高评分 经典"
\ No newline at end of file
+    return "；".join(parts) if parts else "优质电影 高评分 经典"
diff --git a/myapp/recommender/rerank.py b/myapp/recommender/rerank.py
index ed6951a..f71d66a 100644
--- a/myapp/recommender/rerank.py
+++ b/myapp/recommender/rerank.py
@@ -21,7 +21,8 @@ from myapp.recommender.recall import _get_user_liked_ids
 logger = logging.getLogger('movie_agent')
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
-def business_filter(candidates, user=None, exclude_sensitive=True, exclude_ids=None):
+def business_filter(candidates, user=None, exclude_sensitive=True, exclude_ids=None,
+                    memory_slots=None):
     """
     业务规则过滤：
       1. 排除已推荐过的电影（去重）
@@ -36,6 +37,7 @@ def business_filter(candidates, user=None, exclude_sensitive=True, exclude_ids=N
         user: 用户对象
         exclude_sensitive: 是否排除敏感内容
         exclude_ids: 额外排除的ID集合
+        memory_slots: 调用方提供的当前会话槽位；缺省时不读取其他会话
⟦行尾:SP×5⟧
     Returns:
         List[Dict] 过滤后的候选列表（自动补充安全内容）
@@ -98,15 +100,9 @@ def business_filter(candidates, user=None, exclude_sensitive=True, exclude_ids=N
⟦行尾:SP×13⟧
             # 获取用户偏好类型（优先匹配）
             user_genres = []
-            try:
-                from myapp.agent.memory import MemoryManager
-                session_id = f"user_{getattr(user, 'id', 'anon')}"
-                memory = MemoryManager(user=user, session_id=session_id)
-                slots = memory.get_slots()
-                if slots.get('genre'):
-                    user_genres.append(slots['genre'])
-            except Exception:
-                pass
+            slots = memory_slots if isinstance(memory_slots, dict) else {}
+            if slots.get('genre'):
+                user_genres.append(slots['genre'])
⟦行尾:SP×13⟧
             # 优先从用户偏好类型中补充
             safe_qs = Movie.objects.filter(is_sensitive=False).exclude(id__in=existing_ids)
@@ -205,7 +201,7 @@ def diversity_guarantee(candidates, min_genre_diversity=3, top_k=15):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 def final_rerank(candidates, user=None, top_k=15,
-                  exclude_sensitive=True, ensure_diversity=True):
+                  exclude_sensitive=True, ensure_diversity=True, memory_slots=None):
     """
     重排主函数。
⟦行尾:SP×5⟧
@@ -215,6 +211,7 @@ def final_rerank(candidates, user=None, top_k=15,
         top_k: 最终返回数量
         exclude_sensitive: 是否过滤敏感内容
         ensure_diversity: 是否保障多样性
+        memory_slots: 当前会话的槽位快照
⟦行尾:SP×5⟧
     Returns:
         Tuple[List[Dict], Dict] - (最终推荐列表, 统计信息)
@@ -224,7 +221,7 @@ def final_rerank(candidates, user=None, top_k=15,
     # 1. 业务过滤
     filtered = business_filter(
         candidates, user=user,
-        exclude_sensitive=exclude_sensitive
+        exclude_sensitive=exclude_sensitive, memory_slots=memory_slots
     )
⟦行尾:SP×5⟧
     # 2. 多样性保障
@@ -244,4 +241,4 @@ def final_rerank(candidates, user=None, top_k=15,
         'latency_ms': int((time.time() - t_start) * 1000),
     }
⟦行尾:SP×5⟧
-    return final, stats
\ No newline at end of file
+    return final, stats
diff --git a/myapp/services/agent_chat_service.py b/myapp/services/agent_chat_service.py
index a5cb9dd..5856c3c 100644
--- a/myapp/services/agent_chat_service.py
+++ b/myapp/services/agent_chat_service.py
@@ -37,9 +37,10 @@ class AgentChatService:
     5. 排除"不喜欢"电影
     """
⟦行尾:SP×5⟧
-    def __init__(self, user, session_key: str = 'default'):
+    def __init__(self, user, session_key: Optional[str] = None):
         self.user = user
-        self.session_id = f"user_{user.id}_{session_key}"
+        self.session_id = (f"user_{user.id}_{session_key}"
+                           if session_key and session_key != 'default' else None)
         self._agent = None
⟦行尾:SP×5⟧
     def _get_agent(self):
@@ -343,7 +344,7 @@ class AgentChatService:
             logger.error(f"[Service] 保存 AgentTrace 失败: {e}")
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
-def process_agent_request(user, user_input: str, session_key: str = 'default', is_thinking: bool = False) -> Dict:
+def process_agent_request(user, user_input: str, session_key: Optional[str] = None, is_thinking: bool = False) -> Dict:
     """
     便捷函数：一步完成 Agent 聊天处理。
     适用于不想实例化 Service 的场景。
diff --git a/myapp/services/agent_service.py b/myapp/services/agent_service.py
index d5141af..29bb453 100644
--- a/myapp/services/agent_service.py
+++ b/myapp/services/agent_service.py
@@ -16,6 +16,10 @@ import time
 from typing import Optional, Tuple
⟦行尾:SP×1⟧
 from django.core.cache import cache
+from myapp.utils.cache_keys import user_business_cache_key
+from myapp.utils.cache_access import business_cache_get, business_cache_set
+
+_AUTO_KEY = object()
⟦行尾:SP×1⟧
 logger = logging.getLogger('agent_service')
⟦行尾:SP×1⟧
@@ -64,7 +68,7 @@ def safe_neo4j_query(graph, cypher: str, params: dict = None, timeout_ms: int =
         return []
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
-def get_cached_explain(user_id: int, movie_id: int) -> Optional[dict]:
+def get_cached_explain(user_id: int, movie_id: int, cache_key=_AUTO_KEY) -> Optional[dict]:
     """
     获取缓存的推荐解释结果
⟦行尾:SP×5⟧
@@ -74,29 +78,31 @@ def get_cached_explain(user_id: int, movie_id: int) -> Optional[dict]:
     【修改原因】：ajax_explain_rec 每次都要调用 LLM，对热门电影造成大量重复计算。
     【性能收益】：缓存命中时响应时间从 2-5s 降低到 <10ms。
     """
-    cache_key = f"explain_v1_{user_id}_{movie_id}"
-    return cache.get(cache_key)
+    if cache_key is _AUTO_KEY:
+        cache_key = user_business_cache_key(cache, user_id, 'explain', movie_id)
+    return business_cache_get(cache, cache_key)
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
-def set_cached_explain(user_id: int, movie_id: int, data: dict, ttl: int = 600):
+def set_cached_explain(user_id: int, movie_id: int, data: dict, ttl: int = 600, cache_key=_AUTO_KEY):
     """
     缓存推荐解释结果（默认 TTL=10 分钟）
⟦行尾:SP×5⟧
     【修改原因】：热门推荐电影（如首页推荐）会被大量用户反复查看，
                 缓存可显著降低 LLM 调用次数。
     """
-    cache_key = f"explain_v1_{user_id}_{movie_id}"
-    cache.set(cache_key, data, ttl)
+    if cache_key is _AUTO_KEY:
+        cache_key = user_business_cache_key(cache, user_id, 'explain', movie_id)
+    business_cache_set(cache, cache_key, data, ttl)
     logger.debug(f"[Cache] 写入解释缓存 key={cache_key} ttl={ttl}s")
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 def get_cached_agent_recall(user_id: int) -> Optional[list]:
     """获取缓存的 Agent 召回结果"""
-    cache_key = f"agent_recall_{user_id}"
-    return cache.get(cache_key)
+    cache_key = user_business_cache_key(cache, user_id, 'agent_recall')
+    return business_cache_get(cache, cache_key)
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 def set_cached_agent_recall(user_id: int, data: list, ttl: int = 1800):
     """缓存 Agent 召回结果（默认 TTL=30 分钟）"""
-    cache_key = f"agent_recall_{user_id}"
-    cache.set(cache_key, data, ttl)
\ No newline at end of file
+    cache_key = user_business_cache_key(cache, user_id, 'agent_recall')
+    business_cache_set(cache, cache_key, data, ttl)
\ No newline at end of file
diff --git a/myapp/urls.py b/myapp/urls.py
index 39a17e3..fd18c08 100644
--- a/myapp/urls.py
+++ b/myapp/urls.py
@@ -97,6 +97,7 @@ urlpatterns = [
⟦行尾:SP×1;CRLF⟧
     # 推荐反馈收集⟦行尾:CRLF⟧
     path('agent/feedback/', agent_views.recommend_feedback_view, name='recommend_feedback'),⟦行尾:CRLF⟧
+    path('agent/cache/retry/', agent_views.retry_cache_invalidation, name='retry_cache_invalidation'),
⟦行尾:SP×1;CRLF⟧
     # Agent推理链展示⟦行尾:CRLF⟧
     path('agent/trace/', agent_views.agent_trace_view, name='agent_trace_list'),⟦行尾:CRLF⟧
diff --git a/myapp/views.py b/myapp/views.py
index b01d6e9..df4d465 100644
--- a/myapp/views.py
+++ b/myapp/views.py
@@ -19,6 +19,7 @@ def _safe_cuda_available():
⟦行尾:SP×1⟧
 import hashlib
 import json
+import math
 import pickle
 import re
 import threading
@@ -46,6 +47,16 @@ from django.conf import settings
 from urllib.parse import quote
 from django.contrib.auth.forms import SetPasswordForm, UserCreationForm
 from django.core.cache import cache
+from myapp.services.cache_mutations import (
+    recommendation_write_view, schedule_user_invalidation, schedule_catalog_invalidation,
+    partial_commit_response,
+)
+from myapp.utils.cache_keys import (
+    kg_subgraph_cache_key, hybrid_recommendation_cache_key,
+    invalidate_recommendation_cache,
+    RecommendationCacheInvalidationError, user_business_cache_key, catalog_business_cache_key,
+)
+from myapp.utils.cache_access import business_cache_get, business_cache_set
 from django.core.management import call_command
 from django.shortcuts import render, redirect, get_object_or_404
 from django.http import HttpResponseRedirect, JsonResponse, HttpResponseBadRequest
@@ -797,12 +808,14 @@ def update_recommendations(user, top_k=15):
                     best_idx = i
         final_rec_indices.append(best_idx)
⟦行尾:SP×1⟧
-    Rec.objects.filter(user=user).delete()
     rec_objs = [
         Rec(user=user, movie_id=cand_100_raw[idx], rating=float(cand_100_preds[idx]))
         for idx in final_rec_indices
     ]
-    Rec.objects.bulk_create(rec_objs)
+    with transaction.atomic():
+        Rec.objects.filter(user=user).delete()
+        Rec.objects.bulk_create(rec_objs)
+        schedule_user_invalidation(user.id)
     return len(rec_objs)
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
@@ -984,6 +997,58 @@ def rrf_merge(ranked_lists, k_rrf=60, weights=None):
     return merged  # [(movie_id, rrf_score), ...]
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
+_HYBRID_CACHE_SCHEMA = 'hybrid_rec_payload_v1'
+
+
+def _restore_hybrid_cache_payload(payload):
+    """Validate a lightweight cache value and hydrate movies in one query.
+
+    Return None for an obsolete/malformed payload or when any cached movie has
+    been deleted. Database errors intentionally propagate to the caller.
+    """
+    if not isinstance(payload, dict) or payload.get('schema') != _HYBRID_CACHE_SCHEMA:
+        return None
+    items = payload.get('items')
+    profile_text = payload.get('profile_text')
+    source_stats = payload.get('source_stats')
+    if not isinstance(items, list) or not isinstance(profile_text, str) or not isinstance(source_stats, dict):
+        return None
+
+    movie_ids = []
+    scores = {}
+    for item in items:
+        if not isinstance(item, dict):
+            return None
+        movie_id = item.get('movie_id')
+        score = item.get('rrf_score')
+        if (not isinstance(movie_id, int) or isinstance(movie_id, bool)
+                or movie_id in scores or not isinstance(score, (int, float))
+                or isinstance(score, bool) or not math.isfinite(float(score))):
+            return None
+        movie_ids.append(movie_id)
+        scores[movie_id] = score
+
+    if not movie_ids:
+        restored_stats = dict(source_stats)
+        restored_stats['profile_text'] = profile_text
+        return ([], profile_text, restored_stats)
+
+    movies = list(
+        Movie.objects.filter(id__in=movie_ids)
+        .prefetch_related('genres', 'actors', 'directors')
+    )
+    by_id = {movie.id: movie for movie in movies}
+    if len(by_id) != len(movie_ids) or any(movie_id not in by_id for movie_id in movie_ids):
+        return None
+
+    ordered_movies = [by_id[movie_id] for movie_id in movie_ids]
+    for movie in ordered_movies:
+        movie.rrf_score = round(float(scores[movie.id]), 4)
+    restored_stats = dict(source_stats)
+    restored_stats['profile_text'] = profile_text
+    return ordered_movies, profile_text, restored_stats
+
+
 def hybrid_recall_recommend(user, top_k=30, force_refresh=False):
     """
     混合检索主函数：
@@ -992,11 +1057,14 @@ def hybrid_recall_recommend(user, top_k=30, force_refresh=False):
     3. RRF 融合 & 取 top_k
     4. 返回 (Movie QuerySet, profile_text, source_stats)
     """
-    cache_key = f"hybrid_rec_{user.id}"
-    if not force_refresh:
-        cached = cache.get(cache_key)
-        if cached:
-            return cached
+    cache_key = hybrid_recommendation_cache_key(cache, user.id, top_k)
+    if cache_key is not None and not force_refresh:
+        cached = business_cache_get(cache, cache_key)
+        if cached is not None:
+            restored = _restore_hybrid_cache_payload(cached)
+            if restored is not None:
+                return restored
+            business_cache_delete(cache, cache_key)
⟦行尾:SP×1⟧
     profile_text, liked_ids = build_user_profile_text(user)
     liked_set = set(liked_ids)
@@ -1048,7 +1116,19 @@ def hybrid_recall_recommend(user, top_k=30, force_refresh=False):
         m.rrf_score = round(id_score.get(m.id, 0.0), 4)
⟦行尾:SP×1⟧
     result = (movies_qs, profile_text, source_stats)
-    cache.set(cache_key, result, 1800)  # 缓存 30 分钟
+    if cache_key is not None:
+        compact_stats = dict(source_stats)
+        compact_stats.pop('profile_text', None)
+        payload = {
+            'schema': _HYBRID_CACHE_SCHEMA,
+            'items': [
+                {'movie_id': movie.id, 'rrf_score': movie.rrf_score}
+                for movie in movies_qs
+            ],
+            'profile_text': profile_text,
+            'source_stats': compact_stats,
+        }
+        business_cache_set(cache, cache_key, payload, 1800)  # 缓存 30 分钟
     return result
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
@@ -1073,6 +1153,8 @@ def recommend(request):
         try:
             # 🚀 实时调用 KAG 模型预测！
             update_recommendations(user, top_k=40)
+        except RecommendationCacheInvalidationError as exc:
+            return partial_commit_response(exc)
         except Exception as e:
             logger.error(f"模型推理失败: {e}")
⟦行尾:SP×1⟧
@@ -1107,8 +1189,8 @@ def recommend(request):
     total_liked_count = len(liked_ids)
⟦行尾:SP×1⟧
     # 🔥 优化二：引入缓存。如果用户只是在翻页，直接从缓存拿兜底数据，避免重算
-    cache_key = f"cb_recs_{user.id}"
-    cached_cb_data = cache.get(cache_key) if not force_refresh else None
+    cache_key = user_business_cache_key(cache, user.id, 'cb_recs')
+    cached_cb_data = business_cache_get(cache, cache_key) if not force_refresh else None
⟦行尾:SP×1⟧
     if cached_cb_data:
         rec_source = cached_cb_data['source']
@@ -1149,7 +1231,7 @@ def recommend(request):
                     'genres', 'actors').order_by('-vote_count', '-score')[:50])
⟦行尾:SP×1⟧
         # 写入缓存，保留 1 小时
-        cache.set(cache_key, {'source': rec_source, 'list': final_rec_list}, 3600)
+        business_cache_set(cache, cache_key, {'source': rec_source, 'list': final_rec_list}, 3600)
⟦行尾:SP×1⟧
     cb_paginator = Pagination(
         request,
@@ -1191,6 +1273,7 @@ def recommend(request):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @login_required
+@recommendation_write_view
 def score_movie(request):
     """
     处理用户在电影详情页提交的打分与评论
@@ -1216,6 +1299,7 @@ def score_movie(request):
                             'discussion': discussion_val if discussion_val else None
                         }
                     )
+                    schedule_user_invalidation(request.user.id)
                     messages.success(request, "评价提交成功！系统将根据您的最新反馈调整推荐。")
             except ValueError:
                 messages.error(request, "无效的评分格式。")
@@ -1226,6 +1310,7 @@ def score_movie(request):
     return redirect('movie_detail', pk=movie_id)
⟦行尾:SP×1⟧
 @never_cache  # (这个我们之前加过，防止缓存)
+@recommendation_write_view
 def movie_detail(request, pk):
     # 🔥 优化：使用 prefetch_related 预加载所有 M2M 字段 (含 directors)
     # 这样在模板里循环时，不会产生几十条 SQL 查询
@@ -1256,6 +1341,7 @@ def movie_detail(request, pk):
                     'discussion': rating_form.cleaned_data['discussion']
                 }
             )
+            schedule_user_invalidation(request.user.id)
             messages.success(request, "评论发表成功！")
             return redirect('movie_detail', pk=pk)
⟦行尾:SP×1⟧
@@ -1279,6 +1365,7 @@ def movie_detail(request, pk):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @login_required  # 2. 确保只有登录用户才能调用
+@recommendation_write_view
 def add_collect(request, pk):
     # 获取电影, 如果不存在则 404
     movie = get_object_or_404(Movie, pk=pk)
@@ -1290,11 +1377,14 @@ def add_collect(request, pk):
         movie=movie  # 当前电影
     )
⟦行尾:SP×1⟧
+    schedule_user_invalidation(request.user.id)
+
     # 4. 操作完成后, 重定向回电影详情页
     return redirect('movie_detail', pk=pk)
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @login_required  # 2. 确保只有登录用户才能调用
+@recommendation_write_view
 def remove_collect(request, pk):
     movie = get_object_or_404(Movie, pk=pk)
⟦行尾:SP×1⟧
@@ -1304,6 +1394,8 @@ def remove_collect(request, pk):
         movie=movie  # 当前电影
     ).delete()
⟦行尾:SP×1⟧
+    schedule_user_invalidation(request.user.id)
+
     # 4. 操作完成后, 重定向回电影详情页
     return redirect('movie_detail', pk=pk)
⟦行尾:SP×1⟧
@@ -1311,6 +1403,7 @@ def remove_collect(request, pk):
 # --- ↓↓↓ 新增 "收藏" 视图 ↓↓↓ ---
 @login_required
 @require_POST
+@recommendation_write_view
 def ajax_collect(request):
     movie_id = request.POST.get('movie_id')
     if not movie_id:
@@ -1324,8 +1417,10 @@ def ajax_collect(request):
         if not created:
             # 如果已经存在 (created=False)，说明是"取消收藏"操作
             collect_obj.delete()
+            schedule_user_invalidation(request.user.id)
             return JsonResponse({'status': 'success', 'action': 'uncollected', 'msg': '已取消收藏'})
         else:
+            schedule_user_invalidation(request.user.id)
             # 如果是新建 (created=True)，说明是"收藏"操作
             return JsonResponse({'status': 'success', 'action': 'collected', 'msg': '收藏成功'})
⟦行尾:SP×1⟧
@@ -2898,10 +2993,10 @@ def get_kg_subgraph(topic_keyword, user_history_mids=None, max_triples=12):
     if not topic_keyword:
         return ""
⟦行尾:SP×1⟧
-    _key_hash = hashlib.md5(topic_keyword.encode("utf-8")).hexdigest()
-    hist_hash = hashlib.md5(str(user_history_mids).encode()).hexdigest()[:8]
-    cache_key = f"kg_subgraph_v3_{_key_hash}_{hist_hash}"
-    cached = cache.get(cache_key)
+    cache_key = catalog_business_cache_key(
+        cache, kg_subgraph_cache_key(topic_keyword, user_history_mids, max_triples)
+    )
+    cached = business_cache_get(cache, cache_key)
     if cached is not None:
         return cached
⟦行尾:SP×1⟧
@@ -3004,7 +3099,7 @@ def get_kg_subgraph(topic_keyword, user_history_mids=None, max_triples=12):
                 )
⟦行尾:SP×1⟧
         result = "\n".join(triples[:max_triples]) if triples else ""
-        cache.set(cache_key, result, 60 * 30)  # 缓存 30 分钟
+        business_cache_set(cache, cache_key, result, 60 * 30)  # 缓存 30 分钟
         return result
⟦行尾:SP×1⟧
     except Exception as e:
@@ -3482,8 +3577,9 @@ def ajax_explain_rec(request):
     # 🔥 [工程优化] 推荐解释缓存：避免同一用户对同一电影重复调用 LLM
     # 【性能收益】：缓存命中时响应时间从 2-5s 降低到 <10ms
     force_refresh = request.GET.get('refresh') == 'true'
+    explain_cache_key = user_business_cache_key(cache, request.user.id, 'explain', int(movie_id))
     if not force_refresh:
-        cached_result = get_cached_explain(request.user.id, int(movie_id))
+        cached_result = get_cached_explain(request.user.id, int(movie_id), cache_key=explain_cache_key)
         if cached_result:
             logger.info(f"[ExplainCache] 缓存命中 user={request.user.id} movie={movie_id}")
             return JsonResponse(cached_result)
@@ -3811,7 +3907,7 @@ def ajax_explain_rec(request):
⟦行尾:SP×1⟧
         # 🔥 [工程优化] 缓存解释结果，避免同一用户对同一电影重复调用 LLM
         # 【论文可写点】：推荐解释缓存策略，TTL=10分钟
-        set_cached_explain(request.user.id, int(movie_id), response_data, ttl=600)
+        set_cached_explain(request.user.id, int(movie_id), response_data, ttl=600, cache_key=explain_cache_key)
⟦行尾:SP×1⟧
         # 【论文可写点】：AgentTracer 记录推理完成，输出结构化 trace
         trace_log_simple(
@@ -3835,9 +3931,9 @@ def extract_concepts_jieba(movie_title, summary):
     import jieba.analyse
⟦行尾:SP×1⟧
     # 1. 缓存检查（🔥 同样使用 md5 避免中文标题直接进入 cache key）
-    _title_hash = hashlib.md5(movie_title.encode("utf-8")).hexdigest()
-    cache_key = f"movie_concepts_jieba_{_title_hash}"
-    cached_data = cache.get(cache_key)
+    _title_hash = hashlib.sha256(json.dumps([movie_title, summary or ""], ensure_ascii=False).encode("utf-8")).hexdigest()
+    cache_key = catalog_business_cache_key(cache, f"movie_concepts_jieba_{_title_hash}")
+    cached_data = business_cache_get(cache, cache_key)
     if cached_data: return cached_data
⟦行尾:SP×1⟧
     # 2. 文本清洗与预处理
@@ -3859,7 +3955,7 @@ def extract_concepts_jieba(movie_title, summary):
             concepts = list(jieba.cut(movie_title))[:2]
⟦行尾:SP×1⟧
         # 5. 写入缓存 (1天)
-        cache.set(cache_key, concepts, 60 * 60 * 24)
+        business_cache_set(cache, cache_key, concepts, 60 * 60 * 24)
         return concepts
⟦行尾:SP×1⟧
     except Exception as e:
@@ -4349,6 +4445,7 @@ def search_results(request):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @login_required
+@recommendation_write_view
 def center(request):
     """
     个人中心视图 (V4 - 双表单版)
@@ -4384,6 +4481,7 @@ def center(request):
⟦行尾:SP×1⟧
             if profile_form.is_valid():
                 profile_form.save()
+                schedule_user_invalidation(request.user.id)
                 messages.success(request, "📝 个人资料已更新！")
                 # 更新 Session 中的用户信息 (防止页面显示旧数据)
                 update_session_auth_hash(request, request.user)
@@ -4464,6 +4562,7 @@ def admin_movie_list(request):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @admin_required
+@recommendation_write_view
 def admin_movie_add(request):
     """
     后台：添加电影 (文本输入版)
@@ -4480,6 +4579,7 @@ def admin_movie_add(request):
             _process_m2m_text(movie, form.cleaned_data['genres'], Genre, 'genres')
             _process_m2m_text(movie, form.cleaned_data['directors'], Actor, 'directors')
⟦行尾:SP×1⟧
+            schedule_catalog_invalidation()
             messages.success(request, f"电影《{movie.title}》添加成功！")
             return redirect('admin_movie_list')
     else:
@@ -4492,6 +4592,7 @@ def admin_movie_add(request):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @admin_required
+@recommendation_write_view
 def admin_movie_edit(request, pk):
     """
     后台：编辑电影 (文本输入版)
@@ -4510,6 +4611,7 @@ def admin_movie_edit(request, pk):
             _process_m2m_text(movie, form.cleaned_data['genres'], Genre, 'genres')
             _process_m2m_text(movie, form.cleaned_data['directors'], Actor, 'directors')
⟦行尾:SP×1⟧
+            schedule_catalog_invalidation()
             messages.success(request, f"电影《{movie.title}》更新成功！")
             return redirect('admin_movie_list')
     else:
@@ -4533,12 +4635,14 @@ def admin_movie_edit(request, pk):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @admin_required
+@recommendation_write_view
 def admin_movie_delete(request, pk):
     movie_obj = get_object_or_404(Movie, pk=pk)
⟦行尾:SP×1⟧
     if request.method == 'POST':
         # (POST 请求) 确认删除
         movie_obj.delete()
+        schedule_catalog_invalidation()
         messages.success(request, "电影删除成功。")
         return redirect('admin_movie_list')
⟦行尾:SP×1⟧
@@ -4566,6 +4670,7 @@ def admin_user_list(request):
 # --- ↓↓↓ 在这里添加 3 个新视图 ↓↓↓ ---
⟦行尾:SP×1⟧
 @admin_required
+@recommendation_write_view
 def admin_user_add(request):
     """
     C - 创建新用户 (管理员)
@@ -4574,6 +4679,7 @@ def admin_user_add(request):
         form = AdminUserCreationForm(request.POST)
         if form.is_valid():
             user = form.save()  # UserCreationForm 会自动哈希密码
+            schedule_user_invalidation(user.id)
             messages.success(request, f"用户 {user.username} 创建成功！")
             return redirect('admin_user_list')
     else:
@@ -4583,6 +4689,7 @@ def admin_user_add(request):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @admin_required
+@recommendation_write_view
 def admin_user_edit(request, pk):
     """
     U - 编辑用户信息 (管理员)
@@ -4593,6 +4700,7 @@ def admin_user_edit(request, pk):
         form = AdminUserChangeForm(request.POST, instance=user_obj)
         if form.is_valid():
             form.save()
+            schedule_user_invalidation(user_obj.id)
             messages.success(request, f"用户 {user_obj.username} 信息更新成功！")
             return redirect('admin_user_list')
     else:
@@ -4624,6 +4732,7 @@ def admin_user_reset_password(request, pk):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @admin_required
+@recommendation_write_view
 def admin_user_delete(request, pk):
     user_obj = get_object_or_404(UserInfo, pk=pk)
⟦行尾:SP×1⟧
@@ -4633,7 +4742,9 @@ def admin_user_delete(request, pk):
         return redirect('admin_user_list')
⟦行尾:SP×1⟧
     if request.method == 'POST':
+        affected_user_id = user_obj.id
         user_obj.delete()
+        schedule_user_invalidation(affected_user_id)
         messages.success(request, f"用户 {user_obj.username} 删除成功。")
         return redirect('admin_user_list')
⟦行尾:SP×1⟧
@@ -4679,6 +4790,7 @@ def admin_comments(request):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @login_required
+@recommendation_write_view
 def admin_comment_delete(request, rating_id):
     """
     后台：删除评论
@@ -4687,7 +4799,9 @@ def admin_comment_delete(request, rating_id):
         return redirect('front_index')
⟦行尾:SP×1⟧
     # 获取并删除
+    affected_user_id = get_object_or_404(UserRating, id=rating_id).user_id
     UserRating.objects.filter(id=rating_id).delete()
+    schedule_user_invalidation(affected_user_id)
⟦行尾:SP×1⟧
     # 提示并跳回列表
     messages.success(request, "评论/评分已删除。")
@@ -4744,6 +4858,7 @@ def admin_director_list(request):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @admin_required
+@recommendation_write_view
 def admin_actor_add(request):
     """
     后台：添加演员
@@ -4752,6 +4867,7 @@ def admin_actor_add(request):
         form = ActorModelForm(request.POST)
         if form.is_valid():
             form.save()
+            schedule_catalog_invalidation()
             messages.success(request, "演员添加成功！")
             return redirect('admin_actor_list')
     else:
@@ -4764,6 +4880,7 @@ def admin_actor_add(request):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @admin_required
+@recommendation_write_view
 def admin_actor_edit(request, pk):
     """
     后台：编辑演员
@@ -4774,6 +4891,7 @@ def admin_actor_edit(request, pk):
         form = ActorModelForm(request.POST, instance=actor_obj)
         if form.is_valid():
             form.save()
+            schedule_catalog_invalidation()
             messages.success(request, f"演员 {actor_obj.name} 更新成功！")
             return redirect('admin_actor_list')
     else:
@@ -4786,6 +4904,7 @@ def admin_actor_edit(request, pk):
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
 @admin_required
+@recommendation_write_view
 def admin_actor_delete(request, pk):
     """
     后台：删除演员
@@ -4794,6 +4913,7 @@ def admin_actor_delete(request, pk):
⟦行尾:SP×1⟧
     if request.method == 'POST':
         actor_obj.delete()
+        schedule_catalog_invalidation()
         messages.success(request, "演员已删除。")
         return redirect('admin_actor_list')
⟦行尾:SP×1⟧
@@ -4839,22 +4959,12 @@ def admin_trigger_train(request):
⟦行尾:SP×1⟧
 @admin_required
 def admin_clear_cache(request):
-    """
-    [接口] 清除系统缓存 (Redis/LocalMem)
-    """
+    """全局清理已禁用；保留 Session 和模型缓存。"""
     if request.method == 'POST':
-        try:
-            cache.clear()
-            # 同时也清除模型缓存，强制下次推理时重新加载新权重
-            global MODEL_CACHE
-            MODEL_CACHE = {
-                'model': None,
-                'feature_columns': None,
-                'encoders': {},
-            }
-            return JsonResponse({'status': 'success', 'msg': '🧹 系统缓存 & 模型权重缓存已清除！'})
-        except Exception as e:
-            return JsonResponse({'status': 'error', 'msg': str(e)})
+        return JsonResponse({
+            'status': 'error', 'code': 'cache_clear_disabled',
+            'msg': '全局缓存清理已禁用：业务缓存与 Session 共用后端。请待独立缓存隔离后再启用。',
+        }, status=409)
     return JsonResponse({'status': 'error', 'msg': '不支持的请求方式'})
⟦行尾:SP×1⟧
⟦行尾:SP×1⟧
@@ -5404,6 +5514,7 @@ def get_excluded_movie_ids(user):
⟦行尾:SP×1⟧
 @csrf_exempt
 @login_required
+@recommendation_write_view
 def ajax_exclude_add(request):
     """
     将电影加入"不喜欢"排除列表
@@ -5423,20 +5534,21 @@ def ajax_exclude_add(request):
     except Movie.DoesNotExist:
         return JsonResponse({'status': 'error', 'msg': '电影不存在'})
⟦行尾:SP×1⟧
-    # 先移除已有的 dislike 记录（如有），再创建（防重复）
-    UserFeedback.objects.filter(
-        user=request.user, movie=movie, feedback_type='dislike'
-    ).delete()
-    UserFeedback.objects.create(
-        user=request.user, movie=movie,
-        feedback_type='dislike', source=request.POST.get('source', 'recommend_page')
-    )
-
-    # 清除推荐缓存，确保下次推荐时排除生效
-    cache_key = f"hybrid_rec_{request.user.id}"
-    cache.delete(cache_key)
-    cache_key2 = f"cb_recs_{request.user.id}"
-    cache.delete(cache_key2)
+    try:
+        with transaction.atomic():
+            # 创建失败时，已有 dislike 的删除也会回滚。
+            UserFeedback.objects.filter(
+                user=request.user, movie=movie, feedback_type='dislike'
+            ).delete()
+            UserFeedback.objects.create(
+                user=request.user, movie=movie,
+                feedback_type='dislike', source=request.POST.get('source', 'recommend_page')
+            )
+            transaction.on_commit(
+                lambda user_id=request.user.id: invalidate_recommendation_cache(cache, user_id)
+            )
+    except RecommendationCacheInvalidationError as exc:
+        return partial_commit_response(exc)
⟦行尾:SP×1⟧
     # ★ 查找候补电影：从 Rec 表中排除所有已显示+已不喜欢的电影，取下一个
     excluded_ids = set(get_excluded_movie_ids(request.user))
@@ -5486,6 +5598,7 @@ def ajax_exclude_add(request):
⟦行尾:SP×1⟧
 @csrf_exempt
 @login_required
+@recommendation_write_view
 def ajax_exclude_remove(request):
     """
     将电影从"不喜欢"排除列表中移除（恢复推荐）
@@ -5505,15 +5618,16 @@ def ajax_exclude_remove(request):
     except Movie.DoesNotExist:
         return JsonResponse({'status': 'error', 'msg': '电影不存在'})
⟦行尾:SP×1⟧
-    deleted, _ = UserFeedback.objects.filter(
-        user=request.user, movie=movie, feedback_type='dislike'
-    ).delete()
-
-    # 清除推荐缓存
-    cache_key = f"hybrid_rec_{request.user.id}"
-    cache.delete(cache_key)
-    cache_key2 = f"cb_recs_{request.user.id}"
-    cache.delete(cache_key2)
+    try:
+        with transaction.atomic():
+            deleted, _ = UserFeedback.objects.filter(
+                user=request.user, movie=movie, feedback_type='dislike'
+            ).delete()
+            transaction.on_commit(
+                lambda user_id=request.user.id: invalidate_recommendation_cache(cache, user_id)
+            )
+    except RecommendationCacheInvalidationError as exc:
+        return partial_commit_response(exc)
⟦行尾:SP×1⟧
     return JsonResponse({
         'status': 'ok',
diff --git a/requirements.txt b/requirements.txt
index c93a781..a4a1115 100644
--- a/requirements.txt
+++ b/requirements.txt
@@ -1,4 +1,5 @@
 django>=5.2
+django-redis>=6.0,<7.0
 mysqlclient
 pillow
 requests

~~~~

</details>

本轮结束：只新增本报告，暂停等待人工确认；不提交、推送或合并。

## 10. 阻断修复轮：当前状态与验收

以下为独立 Worktree 中的后续授权修复，前文“只新增报告”等表述均为历史执行记录。
本轮没有操作原始训练目录，没有启动服务、连接真实数据库/Redis、初始化模型、安装依赖或执行迁移/压测。
仍未提交、推送或合并。

### 10.1 历史问题的修复状态

| 问题 | 当前状态 | 修复/边界 |
|---|---|---|
| F1 缺失 business_cache_delete 导入 | 已修复 | views 明确导入真实工具函数；静态生产绑定检查先于 AST 夹具注入，模拟去掉导入会失败 |
| F2 首个失效失败阻断其他用户 | 已修复 | 每个受影响用户都尝试；admin 请求协调提交后回调，全部尝试完才聚合报错；保留完整失败范围 |
| F3 审查指出的 HTTP 写入风险 | 已修复设计，待真实中间件验证 | 收藏/评分删除/历史清除要求 POST；反馈、会话写入、debug 去掉 CSRF 豁免；同步两处前端 POST 调用 |
| F4 非法 payload 校验 | 已修复本次明确缺口 | 极大整数/非有限分数安全拒绝；vector/content/model 统计要求非负整数 |
| F5 partial_success UI 提示不完整 | 保留风险 | 未扩大为前端流程重构；数据库提交语义仍准确，收藏 toggle 仍需避免重复提交 |
| F6 并发、outbox、维护命令等 | 保留边界 | 不新增表、Worker、锁或持久化恢复机制；不宣称最终一致性 |
| F7 原目录合并与环境兼容 | 仍待人工处理 | 未访问原目录、未安装依赖、未做真实集成；历史性能注释不代表实测收益 |

结论：**首次审查中的 F1/F2 代码阻断已消除，并修复了指定 HTTP 状态修改入口和 F4。
当前可进入人工验收；真实集成、残余一致性和 UI 风险仍不能由隔离测试替代。**

### 10.2 修改文件与根因

- myapp/views.py：补全 business_cache_delete 导入；非法 payload 分数/统计校验；
  收藏/评分删除/历史清除要求 POST，dislike/undislike 保留 CSRF；
  旧异步 ajax_chat 保留协程与内部 POST guard，并关闭 CSRF 豁免。
- myapp/agent_views.py：扩展授权的 users 批量缓存重试，校验 JSON 正整数 ID、去重、每批最多 100；
  反馈、会话写入及 debug 使用 POST/CSRF，保持登录及原管理员限制。
- myapp/utils/cache_keys.py：单用户错误记录 scope/user_id/operation/error_type；
  新增批量失效函数，逐个执行并汇总每个失败，不输出底层连接细节或密钥。
- myapp/services/cache_mutations.py：admin 回调协调、聚合错误及 retry_requests。
  所有权变更覆盖旧/新用户；bulk delete 和 list_editable 均不会被首个 Redis 故障截断。
- tests/test_redis_cache_p0_isolated.py：注入前通过 symtable/AST 校验缓存工具的真实生产绑定与导出；
  所有现有提取用例均执行该检查，夹具不能再补救缺失的生产导入。
- tests/test_redis_cache_p1a_isolated.py：纳入真实恢复帮助定义；
  原 error JSON 回滚测试先登记回调再断言取消，消除空断言问题。
- tests/test_redis_cache_blockers_isolated.py：新增 23 项隔离测试。
- templates/admin_comments.html：删除链接改为带 CSRF 的 POST 表单。
- templates/chat.html：清除历史由 GET 跳转改为带 token 的 POST，成功后刷新。
- 本报告：保留历史，追加当前状态及恢复步骤。

没有改变推荐公式、RRF、模型参数、训练代码、Schema、缓存 TTL 或 P1-B 的正常冷热返回契约。
模板变更仅同步已授权的 HTTP 方法和 token，不重构业务界面。

### 10.3 实际多用户恢复流程

1. Admin 仍在 atomic 内写数据库，并按成功写入登记 on_commit；保存点/外层事务回滚时，
   对应回调被丢弃，Redis 不会因这些回滚写入而失效。
2. 同一 admin 请求中每个回调捕获并记录失效异常，继续后续回调。
   同一用户在该请求提交后只尝试一次；目录范围亦去重。
3. 最后一个回调汇总错误并抛出 RecommendationCacheInvalidationError。
   普通包装或已有 CommittedCacheFailureMiddleware 以 HTTP 200 partial_success 返回，
   database_committed=true、cache_invalidated=false、retry_mutation=false。
   数据库提交成功不会被报告为回滚。
4. 响应新增 failed_scopes，每条记录含用户/目录范围、失败操作阶段和异常类型。
   generation_set 与 legacy_delete 家族可以区分；不包含原始后端异常字符串。
5. retry_requests 列出恢复请求。单个用户使用 user/user_id，用户本人可恢复；
   多用户使用 users/user_ids，只允许 staff。超过 100 个失败用户拆成多个请求。
   目录范围只允许 staff 使用 catalog。
6. 管理员仅针对失败列表调用 POST /agent/cache/retry/；不重做保存、删除、评分或收藏操作。
   成功返回 database_mutation=false、cache_invalidated=true。
   某些用户仍失败则返回 503、database_mutation=false、retry_mutation=false、
   新的 failed_scopes/retry_requests，仅列本次仍失败的范围。

示例：原操作涉及用户 1/2/3，1 和 3 失效失败，2 已成功。

```json
{
  "status": "partial_success",
  "database_committed": true,
  "cache_invalidated": false,
  "retry_mutation": false,
  "failed_scopes": [
    {"scope": "user", "user_id": 1, "operation": "generation_set", "error_type": "ConnectionError"},
    {"scope": "user", "user_id": 3, "operation": "generation_set", "error_type": "ConnectionError"}
  ],
  "retry_requests": [{"scope": "users", "user_ids": [1, 3]}]
}
```

管理员在登录状态下向该接口发 form POST，scope=users，
user_ids 的表单值为 JSON 字符串 [1,3]，并携带正常 CSRF token。
只失效 1/3，不修改数据库，也不失效无关用户 2。
若只剩用户 1 失败，下次 retry_requests 改用 scope=user/user_id=1；
管理员仍可恢复，普通用户只能恢复自己的 ID。

限制：失败列表只保存在提交回调内存、HTTP 响应和日志中，不是持久化 outbox。
响应丢失或进程在数据库提交与所有回调完成之间退出，可能无法知道完整待恢复集合；
日志也不是可靠队列。重复恢复会再次轮换 generation，数据正确性安全，但没有重放去重或最终完成保证。

### 10.4 HTTP 审计与兼容性变化

| URL | 当前 Method | 权限/范围 | 状态变更/调用方 |
|---|---|---|---|
| /movie/ID/collect/add/、remove/ | POST | 登录，收藏绑定 request.user | 当前用户收藏；旧 GET 客户端必须迁移 |
| /admin_panel/comments/delete/ID/ | POST | 登录 + staff | 删除指定评分，按评分所属用户失效；模板已改 POST 表单 |
| /ajax/exclude/add/、remove/ | POST | 登录，反馈绑定 request.user | dislike/undislike；已有 AJAX 发送 token |
| /agent/feedback/ | POST | 登录，反馈绑定 request.user | 当前用户反馈；已有两处反馈页面发送 token |
| /agent/cache/retry/ | POST | 本人单用户或 staff 跨用户/批量/目录 | 只缓存失效，零数据库修改 |
| /chat/api/、/agent/api/ | POST | 登录，当前用户 | 对话历史/trace/记忆；两个 URL 都绑定 agent_api_view |
| /agent/stream/ | POST | 登录，当前 Session/用户 | SSE 对话历史/trace/记忆；已有 fetch 发送 X-CSRFToken |
| /agent/clear_chat/、new_chat/、edit_last/、summarize_profile/ | POST | 登录，当前用户及 Session | 历史/槽位/画像；现有 AJAX 发送 token |
| /agent/debug/ | POST | 登录 + staff，当前用户 | Agent 调试涉及记忆写入；前端发送 token |
| /chat/clear/ | POST | 登录，删除当前用户历史 | chat 模板已改 POST/token |

以上当前可路由写入口均未豁免 CSRF，保留 CsrfViewMiddleware。
旧 views.ajax_chat 协程目前未直接绑定上述 URL；保留其内部 POST 检查、关闭属性豁免，避免同步包装破坏协程。
只读历史读取、KG 查询和健康查询的遗留豁免未扩大修改；未执行健康接口或任何模型检查。

兼容性风险已在修改前告知：旧 GET 收藏/删除/清历史会收到 405；缺少 CSRF 的 POST 会收到 403。
本轮没有保留不安全 GET 兼容模式。现有模板已发送 token，且删除/清历史两处已同步；
外部 API 客户端须自行迁移并进行真实中间件验收。
当前 POST/鉴权/CSRF 证据为静态设计检查与授权函数体 Mock 测试，不能声称真实 Cookie/403 响应已集成通过。

### 10.5 测试与安全执行结果

执行前审查新增脚本的导入链：仅标准库、原 P0/P0.1 安全夹具、按文件路径加载的标准库缓存工具。
生产方法通过 AST 执行；没有 Django app、AppConfig.ready、模型模块、真实 Redis/数据库导入或调用。

按相同环境逐个前台运行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p0_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p01_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p1a_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_p1b_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_redis_cache_blockers_isolated.py
```

实际结果：14 + 18 + 18 + 8 + 23 = **81 项全部 OK，五个进程退出码均为 0**。
新增覆盖缺失导入负例、真实声明导入恢复坏缓存、非法 payload、
多用户首个失败后继续、bulk delete、list_editable、所有权变更、
数据库/外层/保存点回滚、外层提交失败汇总、
管理员恢复、本人恢复、非法 ID 与越权拒绝、目录保留 Session、
HTTP 方法/权限/CSRF 声明及模板 POST/token。

仍未验证：真实 Django Admin 生命周期、SessionMiddleware/CSRF 中间件、MySQL 事务、
Redis 并发/TTL/serializer、真实 ORM SQL 数及浏览器行为。
没有运行完整 Django 应用、GPU/模型、迁移、压力测试或后台任务。

### 10.6 当前 Git 与验收建议

当前分支仍为 refactor/redis-cache。修改范围限于缓存、相关事务/HTTP 安全修复及相应模板/测试/报告。
未访问原始训练目录，未改训练/推荐公式，未安装依赖，未提交/推送/合并。
以下当前 Git 快照与前面首次审查的快照分开记录。
建议人工先验收当前修复，再决定本地提交；main 合并仍需真实安全集成验证、残余风险接受，
以及对原始 settings/memory 未提交修改的人工语义合并。


### 10.7 修复后的实际 Git 快照

分支：refactor/redis-cache；HEAD：d78390dede4a0371a1e7991f06568d8d390fbdbd。

~~~~text
 movie/settings.py                    |   1 +
 myapp/admin.py                       |  29 ++--
 myapp/agent/memory.py                | 222 ++++++++++++++++-------------
 myapp/agent/movie_agent.py           |  21 +--
 myapp/agent/react_agent.py           |   7 +-
 myapp/agent/workflow_agent.py        |   7 +-
 myapp/agent_views.py                 | 118 +++++++++++++---
 myapp/middleware.py                  |  18 +++
 myapp/recommender/query_rewriter.py  |  10 +-
 myapp/recommender/recall.py          |  18 ++-
 myapp/recommender/rerank.py          |  23 ++-
 myapp/services/agent_chat_service.py |   7 +-
 myapp/services/agent_service.py      |  26 ++--
 myapp/urls.py                        |   1 +
 myapp/views.py                       | 261 ++++++++++++++++++++++++++---------
 requirements.txt                     |   1 +
 templates/admin_comments.html        |  14 +-
 templates/chat.html                  |   9 +-
 18 files changed, 538 insertions(+), 255 deletions(-)
~~~~

当前跟踪修改文件（含先前阶段；本轮直接文件见 10.2）：

- movie/settings.py
- myapp/admin.py
- myapp/agent/memory.py
- myapp/agent/movie_agent.py
- myapp/agent/react_agent.py
- myapp/agent/workflow_agent.py
- myapp/agent_views.py
- myapp/middleware.py
- myapp/recommender/query_rewriter.py
- myapp/recommender/recall.py
- myapp/recommender/rerank.py
- myapp/services/agent_chat_service.py
- myapp/services/agent_service.py
- myapp/urls.py
- myapp/views.py
- requirements.txt
- templates/admin_comments.html
- templates/chat.html

当前全部未跟踪文件：

- docs/REDIS_CACHE_FINAL_REVIEW.md
- docs/REDIS_CACHE_P01_AUDIT.md
- docs/REDIS_CACHE_P0_AUDIT.md
- docs/REDIS_CACHE_P1A_AUDIT.md
- docs/REDIS_CACHE_P1B_AUDIT.md
- myapp/services/cache_mutations.py
- myapp/utils/cache_access.py
- myapp/utils/cache_keys.py
- tests/test_redis_cache_blockers_isolated.py
- tests/test_redis_cache_p01_isolated.py
- tests/test_redis_cache_p0_isolated.py
- tests/test_redis_cache_p1a_isolated.py
- tests/test_redis_cache_p1b_isolated.py

修复后 git diff --check、23 个 Python AST/内存 compile 及未跟踪 Python 空白检查通过。
暂存差异为空；只修改独立 Worktree，未提交、推送或合并。
