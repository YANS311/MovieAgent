# P4-B.2-R0：冻结测试阻断修复

日期：2026-10-14。**GO：两个 Git 状态阻断已修复，真实临时 Git 三态和负向验证通过，可以重新启动 P4-B.2 冻结审查。**

本结论只针对冻结测试；不代表真实 Django/ASGI、网络断连或生产 SSE 适配通过。本轮没有暂存/提交真实项目，没有创建后续 Worktree。

## 1. 根因与实际修改

上轮阻断报告为临时文件 movieagent-p4b2-freeze-review.md。核对原测试和修复前快照后确认：

| 测试 | 原断言 | 修复后的约束 |
|---|---|---|
| test_asgi_streaming_poc_isolated.StaticBoundaryTests.test_protected_p4b_files_byte_identical | SHA(git diff --binary) 等于原未暂存差异摘要；暂存/提交后变空 | 固定P4-A基线、明确三个生产路径、禁用外部diff/textconv，保留原差异SHA和文件SHA |
| test_agent_sse_lifecycle_isolated.test_only_authorized_production_files_changed | 从基线获取全部tracked差异，集合却仅允许三个生产文件；忽略untracked | 对worktree、index、HEAD分别按固定基线检查状态，另外枚举全部untracked，包括ignored；生产与新增成果分别使用精确路径白名单 |

固定基线仍为 `e60eb5a96662f9e448df571e5e0e495bd43ba737`，没有使用动态HEAD替代。

修改/新增清单：

1. tests/test_asgi_streaming_poc_isolated.py：新增固定基线内容检查；原11文件保护保留，生命周期测试获授权R0修订使用明确摘要；验证manifest自身原始摘要；额外核对index/HEAD中受保护blob和模式，避免坏暂存内容被恢复后的工作树掩盖。
2. tests/test_agent_sse_lifecycle_isolated.py：精确三生产路径、22个原新增路径、两个R0可选新增路径；NUL分隔状态，关闭rename detection；不允许删除、类型变化、未知路径，不仅比较默认diff。
3. tests/test_git_freeze_states_isolated.py：新增32项标准库/微型Git测试，无Django/模型/网络。
4. docs/agent_backend/p4b2/FREEZE_BLOCKER_FIX_REPORT.md：本报告。

两个原测试文件均尚未跟踪，因此普通git diff不显示它们的修改；与修复前快照比，分别是生命周期测试+83/-4、PoC测试+76/-6。本轮没有生产代码增删。

## 2. 冻结约束没有降低

生产允许修改路径严格为：

- myapp/agent_views.py
- myapp/services/agent_chat_service.py
- myapp/services/agent_contracts.py

新增成果要求原22个路径全部存在且为普通文件。只有本轮的新Git测试和本报告可以额外存在；没有放开整个tests/docs/experiments目录。未知已有文件修改、未知新增、必需文件缺失、删除/重命名、symlink均拒绝。

baseline→worktree、baseline→index、baseline→HEAD只允许生产M、已授权新增A；index→worktree另外允许已授权新增文件M，以兼容正在修订的已暂存新增测试。此分类不会放过其他生产文件。

文件内容检查仍要求原生产SHA和原binary diff SHA完全相同。HEAD/index中的生产blob只能为原基线或批准的P4-B内容；原保护文档也要求冻结内容。模式须为100644，index stage须为0。不是只看路径名字，更不是只看最终工作树。

原manifest保护的生命周期测试此次明确获授权修改。没有改manifest：其原始测试摘要保持历史证据，检查器同时固定原摘要和唯一批准R0修订摘要；任意额外测试篡改仍失败。没有无条件刷新生产期望值。

## 3. Diff与SHA完整性

禁用external diff/textconv并固定基线后，得到的生产patch与修复前保存的patch**逐字节相同**。没有格式变化，没有重新生成宽松期望值。

| 证据 | 原摘要，仍保持 |
|---|---|
| 三个生产文件binary diff | ebfac9f7a758fed1605af08234474a3ca35bdce7751a458d0dea101ba6d7646e |
| agent_views.py | fab03562816a043fdd39848dae8a2a2e52ceb94b653f9059c55263db2c524212 |
| agent_chat_service.py | 4cc09196b84ae13a218c8b1b66112a6ee7f7a920a9c4f75b7cb487cca293179f |
| agent_contracts.py | 6e26c3c584ba809ed41d199de999ca052719eb5af4bf0227a77c087659c913ed |
| 原P4B_FROZEN_SHA256.json字节 | 5e97348f2b4c586f39f83f36cb796db9b442855aa1795d526c7d9b98c7fa3c99 |

修复前25文件中，除两个获授权测试外，其他23文件全部与修复前快照SHA一致：三个生产文件、所有原文档、PoC代码/README/JSON均保持原状。没有修改原始测量样本或冻结摘要。

## 4. Git三态验证方法

每个Fixture位于独立临时目录，git init建立自己的.git，不克隆/引用项目.git、不使用alternates、不读训练目录。只复制三个允许生产文件的基线/批准文本和已审查的小型测试/报告/PoC文本；其他控制文件为合成内容，不执行这些业务代码。

Fixture禁读系统/全局Git配置，剔除继承的GIT_DIR/INDEX等变量；作者为fixture@example.invalid，hooks路径为临时空目录，不签名、不联网。Git add/commit/update-index只发生于Fixture，串行有限执行，每条子进程5秒超时，结束后清理自身TemporaryDirectory。

微型仓库的基线提交SHA必然不同。测试执行实际修复函数，包装器只将命令中的固定P4-A token译为该Fixture的合成基线；其他输出来自真实Git，不伪造diff结果。真实项目检查另有独立只读测试；静态断言确保生产BASELINE固定、无动态HEAD diff。

| 场景 | Unstaged | Staged | Committed |
|---|---|---|---|
| 同一批准源码及原新增成果 | 通过 | 通过，默认diff为空 | 通过，diff HEAD为空 |
| 固定基线生产diff摘要 | 原摘要 | 原摘要 | 原摘要 |
| 非白名单生产修改 | 拒绝 | 拒绝 | 拒绝 |
| 冻结生产内容篡改 | 拒绝 | 拒绝 | 拒绝 |
| 未知文件添加 | 拒绝 | 拒绝 | 拒绝 |
| 冻结生产文件删除 | 拒绝 | 拒绝 | 拒绝 |
| 原冻结报告删除 | 拒绝 | 拒绝 | 拒绝 |

上述多态负向组合有subTest，计入所在测试方法，不虚增32项的统计。

## 5. 其他负向与边界验证

全部通过：

- HEAD前进到另一个合成Commit后，仍比较原固定基线；把BASELINE错误改成HEAD会失败。
- 原测试和报告正确提交后不误报；混合暂存同一批准内容通过。
- 不安全production blob已暂存、工作树恢复批准内容：拒绝。
- HEAD中坏production blob、index和工作树恢复批准内容：拒绝。
- 暂存生产文件可执行模式变化、工作树保持原状：拒绝。
- 暂存受保护报告坏内容、工作树恢复：拒绝。
- 未知路径只存在index或HEAD但已从工作树/index删除：拒绝。
- ignored未知.env、已有非生产文件修改、含换行的未知文件名：拒绝。
- 生产/报告重命名、必需untracked文件缺失、symlink替代：拒绝。
- manifest替换、未批准的生命周期测试额外篡改、受保护报告内容改动：拒绝。
- 仅两个精确R0可选路径在三态下通过，相似未知路径不通过。

没有删、跳过或放宽旧用例。与修复前AST对照：原60个生命周期测试方法和46个PoC测试方法均保留；除了两个获授权冻结方法，其余104个方法AST完全一致。

## 6. 实际测试结果

安全导入审计：测试仅标准库；原业务依赖使用已审计Fake/AST抽取；本轮Git测试只抽取指定实际辅助函数，不导入应用或运行模型。所有命令使用CPU环境限制、Python -S、PYTHONDONTWRITEBYTECODE=1，逐个前台执行。

| 原有脚本（tests/） | 实际通过 | 失败 | 跳过 | 框架报告秒数 |
|---|---:|---:|---:|---:|
| test_redis_cache_p0_isolated.py | 14 | 0 | 0 | 1.098 |
| test_redis_cache_p01_isolated.py | 18 | 0 | 0 | 1.992 |
| test_redis_cache_p1a_isolated.py | 18 | 0 | 0 | 0.691 |
| test_redis_cache_p1b_isolated.py | 8 | 0 | 0 | 1.437 |
| test_redis_cache_blockers_isolated.py | 23 | 0 | 0 | 1.872 |
| test_redis_cache_p2_isolated.py | 24 | 0 | 0 | 0.502 |
| test_redis_structures_p3_isolated.py | 41 | 0 | 0 | 0.501 |
| test_redis_structures_p31_isolated.py | 26 | 0 | 0 | 0.079 |
| test_redis_structures_p32_isolated.py | 23 | 0 | 0 | 0.096 |
| test_agent_backend_p4a_isolated.py | 42 | 0 | 0 | 2.991 |
| test_agent_sse_lifecycle_isolated.py | 60 | 0 | 0 | 4.144 |
| test_asgi_streaming_poc_isolated.py | 45 | 0 | 1 | 0.672 |
| test_agent_sse_terminal_isolated.cjs | 30 | 0 | 0 | 未单独计时 |

历史口径：**372通过、0失败、1跳过**，本轮实际重跑。跳过仍为真实Django响应构造，没有新增跳过；P4-PERF测试未混入。

新增Git状态测试单独统计：**32通过、0失败、0跳过**，2.490秒。累计405发现，**404通过、0失败、1跳过**。完整回归、JS syntax、最终三态执行退出码均为0。

开发首轮新增测试有一处AST工具参数TypeError：30项中的1项error，2.698秒；正确三态/负向检查已通过。修正新测试中工具调用，并增加暂存blob/mode负向案例后，32项全部通过。没有删除失败用例或降低断言；该初次结果保留于记录，不替代最终结果。

复现命令（仓库根目录，Python逐个执行上表脚本）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/test_git_freeze_states_isolated.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 python3 -S tests/<原有Python脚本>.py
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 UV_THREADPOOL_SIZE=1 node --check tests/test_agent_sse_terminal_isolated.cjs
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 UV_THREADPOOL_SIZE=1 node tests/test_agent_sse_terminal_isolated.cjs
```

另外完成Python AST/内存compile、生产符号绑定、JSON测量样本结构、git diff --check及未跟踪文件尾空白/敏感格式检查。没有重新运行性能基准、改写JSON样本或宣称真实性能收益。

## 7. Git与停止状态

当前Worktree：MovieAgent-sse-lifecycle。分支refactor/agent-sse-lifecycle，HEAD仍为e60eb5a96662f9e448df571e5e0e495bd43ba737。真实暂存区为空。

Git累计状态仍有原三个生产跟踪修改，但其内容没有本轮变化；原22个未跟踪文件保留，两个目标测试修订仍表现为untracked；另外新增Git测试及本报告，共24个untracked。普通git diff --stat仍仅显示原生产差异，不能代表本轮测试修复范围。

以后重新启动冻结流程必须将这两个R0新增文件明确纳入审核/提交归属，不能沿用旧25文件总数漏交。当前没有执行任何真实git add/commit/push/merge/rebase，也没有创建P4-B.2 Worktree。

## 8. 验收边界

两个阻断的修复及三态/负向验证为GO，可以重新审查P4-B/B.1冻结；仍需独立授权真实项目暂存/提交，并核查Git hooks安全。没有用Fixture提交冒充P4-B/B.1 Commit。

真实Django 5.2、ASGIHandler、Uvicorn、浏览器、代理及真实ORM验证尚未开展，不能据R0结果宣称生产SSE可靠性通过。同步Agent残留、事务与客户端可见终态不一致等既有边界未改变。

未访问/修改原训练目录、GPU/CUDA、运行中模型或其他Worktree；未启动真实服务、联网数据库/Redis、安装依赖、迁移、清理缓存。完成后停止，等待人工验收。

## 9. 报告新增后的最终复核

报告写入后，再实际执行32项Git状态测试（2.958秒）、60项生命周期测试（4.113秒）和46项PoC测试（0.711秒）：全部通过，PoC仍只有原1项Django层跳过。重复执行不重复计入累计数量。确认真实工作区中的R0报告和新测试路径也能通过精确范围检查。

最终静态结果：9个修改/未跟踪Python文件AST、内存compile和全局符号绑定通过；历史三份微基准JSON的样本数、分位数及原字节检查通过；PoC证据JSON结构、队列界限、join/任务字段通过；所有24个未跟踪文件尾空白和敏感格式筛查通过。原HEAD/branch不变、真实index为空，git diff --check退出0。
