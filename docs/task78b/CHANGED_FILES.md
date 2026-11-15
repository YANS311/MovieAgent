# Task78-B 改动文件

仅模板、静态样式/导航及独立部署覆盖配置发生变化。没有Python业务代码改动。

| 文件 | 作用 | SHA-256 |
|---|---|---|
| [templates/research_recommendation/base.html](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/templates/research_recommendation/base.html>) | 提供可覆写header/footer、内容页菜单按钮和容器；加载navigation.js、本地favicon及CSS版本参数。 | `524c46265ef92f705c7b49d780c65d65cdc3b7bfe4210c3e495e0cf64d9f5fef` |
| [templates/research_recommendation/auth_base.html](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/templates/research_recommendation/auth_base.html>) | 新增auth专用品牌/返回首页页头及紧凑页脚，无完整业务导航。 | `d09fd275c4a55ea9dfe72de3ab0393734790526516603ba6ac036e2ff77d457c` |
| [templates/research_recommendation/login.html](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/templates/research_recommendation/login.html>) | 仅改为继承auth_base，表单字段、CSRF、next及认证行为不变。 | `0922cf1ad3f8c6902df7da71f7f6fa2a8168a7b324bbd86de994fd37a1909723` |
| [templates/research_recommendation/register.html](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/templates/research_recommendation/register.html>) | 仅改为继承auth_base，既有注册校验和表单字段不变。 | `e7a798f4bc21dce1d4366b1c4f683c2553926f8786407321aaa23f2046ad4381` |
| [static/research_recommendation/classic.css](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/static/research_recommendation/classic.css>) | 从组件与断点源头调整导航、auth flex高度、卡片间距及海报占位、详情文字防溢出。 | `299fd38b229e7c19af7a4492d3d13289d5f16d107731a51bc64accdd65a9c844` |
| [static/research_recommendation/navigation.js](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/static/research_recommendation/navigation.js>) | 新增小型渐进增强菜单控制，aria-expanded、Esc及媒体断点关闭；不接触业务API。 | `c3e556d3eab947bae1a37c7379e865903770e41885eaf81551911cae353be7da` |
| [static/research_recommendation/favicon.svg](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/static/research_recommendation/favicon.svg>) | 新增本地图标，消除旧容器浏览器首次favicon404。 | `be5e89dcabaa93bccb099f4a148fd1b9e22a7cbd7f335d7426392e351564dc7b` |

部署文件：[compose.task78b.yml](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/compose.task78b.yml>)。它仅选择独立Web镜像标签，并以read_only挂载既有`docs/task77/assets/movie_details.json`至其原容器路径；源SHA `d1792b0a6125c027d61f3db1b8c0ddd7779a84c0a90039562259cae3110bc87d` 不变。MySQL与Redis配置、命名卷、凭据及应用权限未改动。

修改前模板和classic.css保存在backups目录。[FRONTEND.diff](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/FRONTEND.diff>)为完整最小差异；[BUILD_CONTEXT_FINAL.json](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/BUILD_CONTEXT_FINAL.json>)为最终有限构建上下文SHA。没有覆盖Task72或Task78历史报告和截图。

验收代码为[browser_regression.js](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/browser_regression.js>)与[browser_supplement.js](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/browser_supplement.js>)，采用已有Playwright CLI运行，没有引入应用依赖或新的测试框架。公开脚本中的凭据为占位标识，实际执行文件与原始含密日志仅存放于Git忽略的private目录并设置限制权限；公开证据已脱敏。

如需退回上轮已验收的蓝紫版本，仅替换Web：

```bash
docs/task72/tools/docker-compose --env-file docs/task72/private/compose.env -f docs/task72/compose.task72.yml -f docs/task72/frontend_restore_20261022/compose.frontend.yml --profile web up -d --no-build --no-deps --force-recreate --wait django
```

当前没有执行回退。不得使用down -v或任何卷清理命令。
