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
- 普通账户默认：PAMK-HistoryMean-v1；真实映射正反馈的物品Embedding均值，再经过冻结PAMK兼容评分。不是训练编码器。
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
