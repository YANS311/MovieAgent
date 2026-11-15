# Task78-B 布局精修与最终视觉验收

日期：2026-10-22。结论：**PASS；本轮范围内的布局、真实浏览器回归和Docker静态资源确认均完成。** 保留蓝紫渐变、白色圆角卡片、电影图标及渐变主按钮，没有恢复绿色主题。

## 六项最终结论

| 问题 | 结论 |
|---|---|
| 1 顶部导航是否修复？ | 是。auth页只有品牌与返回首页，去掉完整菜单及重复登录入口。内容页桌面单行导航，900/1024px也无重叠，手机折叠菜单可操作并支持Esc关闭。 |
| 2 桌面登录卡片是否完整可见？ | 是。1440×900及1366×768中标题、47px输入框、登录按钮、注册入口及整张卡片都在视口内。 |
| 3 移动端是否无横向溢出？ | 是。390×844、375×667登录页及本轮检查的首页、目录、Top20、详情、错误状态均无横向溢出。注册及长内容自然滚动，按钮可达。 |
| 4 Docker是否使用新CSS？ | 是。Web镜像已替换且healthy；实际HTTP取回CSS和导航JS与源码逐字节一致，主CSS带v=78b版本参数。 |
| 5 原有登录和推荐功能是否正常？ | 是。73项真实浏览器断言通过，既有账户登录/退出/重登、错误提示、原20项结果和分数及权限隔离正常。无新模型推理。 |
| 6 哪些截图可替换论文旧图？ | 01桌面登录、03首页、04普通账户Top20、05公开详情；02展示移动端适配。04明确是普通账户规则推荐，不能替换成UKMAP真实结果图；14只用于权限边界。 |

## 修复源头

新增auth_base.html，在模板层显式分开auth与内容页头。auth主体采用body最小视口高度与flex分配可用空间，删除依赖固定页头/页脚高度的calc减法。卡片没有负margin、transform或裁切处理；输入仍保持47px。注册帮助较长时允许整页自然滚动。

内容页导航继续复用原有五项入口与身份操作，桌面保持同一行；781—1050px缩减间距及用户名可视宽度，780px及以下使用原生按钮与小型独立JS控制菜单。CSS直接修改classic.css的断点和组件规则，没有新增前端框架、UI依赖、内联样式或!important补丁。

首页横幅及功能卡片保留已有统一间距；电影卡片使用固定比例海报区域和相同圆角、间距，缺失占位维持既有真实fallback。详情的简介、演员、导演、类型与来源字段均实际显示。智能助手仍展示当前未启用Agent的真实边界，不虚构LLM能力。

原容器缺少Task77公开详情文件，本轮以只读bind mount接入已冻结、SHA匹配的文件，使既有详情展示能力可在容器中验收；没有修改Python业务代码、映射、评分或数据库电影资料。

## 实测卡片与视口

| 视口 | auth页头高度 | 登录卡片上沿/下沿 | 横向溢出 | 结果 |
|---|---:|---:|---:|---|
| 1440×900 | 65px | 213.9 / 713.0px | 0px | PASS |
| 1366×768 | 65px | 147.9 / 647.0px | 0px | PASS |
| 390×844 | 65px | 192.3 / 678.6px | 0px | PASS |
| 375×667 | 65px | 103.8 / 590.1px | 0px | PASS |

## Docker与保护

当前Web镜像为`movieagent-task72-web:20261022-task78b`，ID `sha256:31fa60513423a1e36dde3db244c42331a49384266dc287b02a7981ad02f2aff6`。仅重建/替换Web；MySQL、Redis容器与命名卷没有重启、重建或删除。两次镜像构建均退出0，最终部署退出0并健康。原Task72与Task78镜像标签保留。

实际CSS SHA-256：`299fd38b229e7c19af7a4492d3d13289d5f16d107731a51bc64accdd65a9c844`；实际navigation.js SHA-256：`c3e556d3eab947bae1a37c7379e865903770e41885eaf81551911cae353be7da`。二者与源文件相同，验收使用独立Chrome会话并重新加载带版本参数的资源，排除了旧页面缓存冒充更新的情况。

已有账户标识、偏好、电影ID、授权及13条Trace内容前后完全一致，浏览器读取的20项ID/顺序/分数逐项对应原Trace。访问详情在本次隔离测试库新增1条正常view交互（3→4），未写入宿主机正式业务库。没有更改模型、canonical映射、Agent Trace、研究身份或第四章资产；NEW_TEST_ACCESS=0；无CUDA、commit/push或Git清理。

当前查看入口：[登录页面](http://127.0.0.1:18072/cinema/login/)。后续操作应使用覆盖配置：

```bash
docs/task72/tools/docker-compose --env-file docs/task72/private/compose.env -f docs/task72/compose.task72.yml -f docs/task78b/compose.task78b.yml --profile web ps
```

## 验收与证据边界

主批次65项、补充批次8项，共73项通过；不与Task72的76项或Task78的15项累加。144次静态响应事件无404，pageerror异常0。Chrome版本155.0.8059.84，独立headless真实浏览器，非模拟网页、非Mock响应。14张PNG已逐张视觉核对。

内置浏览器技能连接因宿主桥接不可用而未建立；未绕过该限制控制用户标签页。使用用户明确允许的Playwright独立Chrome完成验收，现已正常退出。本轮没有重新执行授权研究画像的Full实时评分，也没有修改其排序代码；研究页证据为普通账户无授权的真实状态。当前样本没有触发缺失海报状态，本轮未为该状态修改电影或伪造图片响应，不将其算作新增已验收故障用例。

补充脚本初次误在浏览器外引用innerHeight而中止；应用无故障，改为返回视口高度后仅重跑8项补充批次，旧输出保留但不累计。首次旧版浏览器打开时观察到favicon.ico的404，新增本地SVG图标后最终批次无静态404。

详见[BROWSER_REGRESSION.md](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/BROWSER_REGRESSION.md>)、[CHANGED_FILES.md](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/CHANGED_FILES.md>)、[SCREENSHOT_INDEX.md](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/SCREENSHOT_INDEX.md>)及[FINAL_RESULT.json](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/FINAL_RESULT.json>)。本轮到此停止，不继续扩展美化或业务功能。
