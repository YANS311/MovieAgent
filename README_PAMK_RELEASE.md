# 第五章本地PAMK服务发布入口

本分支是独立本地验收版本；不切换生产数据库，不包含权重、数据库内容或真实凭证。
推荐入口为 `movie.release_settings` + `research_recommendation.product_urls`。
Web进程不导入PyTorch，Worker通过显式配置的独立Python启动，科学模型源码随Git固定在scientific_pamk。

## 依赖与配置

1. MySQL、Redis和NVIDIA驱动须已安装。Web使用Python3.13；Worker使用Python3.9与torch2.4.0+cu121。
2. 可用 `scripts/bootstrap_release.sh /path/python3.13 /path/python3.9 /path/new-runtime`创建两套依赖环境。不会修改已有环境。
3. 将config/assets.example.json、encoder_manifest.example.json及release.env.example复制到Git外的私密目录，替换实际路径和数据库凭证，env文件chmod600。外部资产必须满足模板内完整SHA；不能更换为旧异常Full2024。
4. 保持NEW_TEST_ACCESS=0。不得配置Test/Validation标签作为在线历史。所有科研资产只读。

## 本地启动

下列`$WEB`是Web虚拟环境的Python，`$ENV`是私密env文件绝对路径。

```bash
"$WEB" scripts/local_service.py init-db --env "$ENV"
"$WEB" scripts/local_service.py migrate --env "$ENV"
"$WEB" scripts/local_service.py check --env "$ENV"
"$WEB" scripts/local_service.py start --env "$ENV" --port 8770
"$WEB" scripts/local_service.py status --env "$ENV"
"$WEB" scripts/local_service.py stop --env "$ENV"
```

空库还须导入合法Train研究目录、可信业务电影及映射，不能把科研用户直接导入为可登录账户。
`import_pamk_train --assets <prepared_train_metadata>`支持按用户块恢复与幂等导入；输入由Train清单、research_users.json、research_movies.json组成，需事先准备并校验其来源。
已有独立验收库可在**目标空库**上执行`seed-from --source-db ch5_pamk_source --env "$ENV"`进行流式复制，源和目标均须ch5_pamk_前缀，禁止生产库、覆盖非空目标。真实业务内容属于外部私有资产，不提交Git。
科研导入回滚命令`rollback_pamk_import --confirm-train-sha256 <SHA>`默认dry-run；仅独立库、确认后加--apply，仅回滚对应科研导入，不删业务账户/电影/评分。本轮不对实际验收库执行此删除；相应行为在自动化临时测试库验证。

## 推荐身份和回滚

- 原生研究授权：PAMK-Net，固定科研用户参数，仅受控演示，不证明注册者与科研用户同一人。
- 普通账户默认：PAMK-HistoryMean-v1；本人真实规范电影正反馈（含未匹配旧库电影）的物品Embedding均值，再经过冻结PAMK兼容评分。不是训练编码器。
- 显式model_name=PAMK-Inductive-v1或配置PAMK_INDUCTIVE_STRATEGY=learned：已训练编码器；其真实内部留出效果低于均值方案，保留研究对照。
- 空/全不可信历史：明确的METADATA_COLD_START或METADATA_PREFERENCE。
- 回滚普通模型分流：先stop，把PAMK_INDUCTIVE_ENABLED设0，再start；恢复设1。研究原生路径仍须授权。原生产默认配置不变。

所有API请求须登录、CSRF及Trace所有权校验。PAMK-AV禁用。模型错误明确503，不静默换模型。

## 实际验收

`acceptance_http.py --env "$ENV" --accounts <private_synthetic_research_accounts.json>`创建合成普通账号并通过真实HTTP评分/模型/Trace验收。
`acceptance_faults.py`仅供独立库和独占本地端口：测试缺失Worker、真实超时、独立MySQL代理断连/恢复、功能开关回滚；不关停真实MySQL、不故意耗尽GPU。
测试会新增合成账户/Trace，不要对生产运行。私密测试账号文件不进Git。
自动化回归用`manage_release.py --env <flag-off-unit-env> test research_recommendation.tests --settings=movie.release_test_settings --noinput`，单元/契约含Mock，不能代替上述真实GPU验收。

当前范围是本机单Web进程＋串行单GPU Worker；完整Agent、公开部署、多进程GPU调度未验收。内部512用户留出继承冻结基座的Train信息，不是严格独立新用户泛化基准。

## 前端与静态资源

新本地入口包含紫色主题首页、/rankings/榜单、/depot/类型筛选与分页、电影详情反馈、海报推荐卡片、Trace可视页和/assistant/受限工具命令界面。静态CSS/JS/SVG随Git固定，无CDN依赖；本地URLconf仅在loopback release settings中使用Django静态视图，不作为公开生产静态方案。

PAMK_POSTER_ROOT指向既有可信业务海报目录，必须是公开图片专用目录；只支持单层jpg/jpeg/png/webp文件。海报和数据库内容为外部资产，不伪造海报映射。缺图统一占位；未映射研究电影不借用业务海报。/assistant/仅支持“推荐电影”“查看来源”，真实调用同一推荐API，不声明LLM、RAG和知识图谱闭环。

停止脚本根据私密状态文件中的源码路径与进程启动标识核验自身服务，允许发布版本切换时从另一交付检出执行stop；不能只杀端口上的任意服务。启动前检测端口，忙碌时保留旧状态。


## 统一目录与原生工具Agent（2026-11-14）

迁移0005增加规范/业务二择一反馈及会话、工具调用、待确认操作表，不重编号科研电影。
`/depot/`与`/api/catalog/`统一检索；可信业务别名折叠为同一规范身份，未匹配或歧义对象使用`/research/movies/<cid>/`，ID 0有效。在线反馈不回写科研Train。
`/history/`查看本人反馈；`/api/catalog/feedback/`保存规范身份交互并改变历史上下文缓存版本。

`/assistant/`通过`/api/agent/`调用已有ToolRegistry的产品适配器。安装受控Ollama 0.17.7及配置中固定digest的qwen3:4b-instruct模型；不会自动下载。工具为search_movies、recommend_movies、explain_recommendation、update_preferences。原生工具选择至多两次规划、每次最多四个调用；无无限循环、无关键词伪装规划。偏好写入先创建10分钟有效的待确认动作，再由本人会话CSRF确认。解释限于所有者Trace事实，不输出因果KG结论。
LLM与PAMK共享单机GPU；keep_alive=0及时释放LLM，首载延迟独立记录。该Agent是现有Registry上的产品适配器，不等同于验收了所有旧MAAN/RAG分支。

本轮真实验收入口（均须独立数据库、登录测试账号、NEW_TEST_ACCESS=0）：
```bash
"$WEB" scripts/acceptance_completion.py --env "$ENV" --base http://127.0.0.1:8770
"$WEB" scripts/acceptance_http.py --env "$ENV" --accounts /private/research-demo-accounts.json
```
前者实际使用本地LLM、MySQL和GPU，不属于Mock测试。无历史用户仍明确冷启动；PAMK-AV仍503。
新空库明确使用utf8mb4_unicode_ci；复制既有独立库时继承Trace UUID列排序规则，避免新增外键类型不一致。目标非空时seed-from拒绝覆盖。只在临时测试库验证迁移逆向，已含新反馈的库不直接逆迁移。
