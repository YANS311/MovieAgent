# Task78-B 真实浏览器回归

最终状态：**73/73通过**，其中主批次65、补充最终批次8。证据来源是实际Docker端口18072与独立Chrome155.0.8059.84（headless），使用指定CSS视口；不将其描述为真实手机硬件测试。没有TestClient、响应Mock、图片拦截替换或GPU执行。

## 覆盖矩阵

| 范围 | 真实操作与判定 | 结果 |
|---|---|---|
| A 桌面登录 | 1440×900、1366×768；完整卡片、标题、47px输入、按钮、注册入口，auth页头无五项导航/重复登录 | PASS |
| B 手机登录 | 390×844、375×667；输入聚焦、横向宽度、完整初始卡片；错误状态可自然滚动 | PASS |
| C 注册与错误 | 真实浏览器提交不匹配密码；服务端errorlist可读；四种宽度滚动到注册按钮并确认在视口内 | PASS；未创建新账户 |
| D 导航 | 现有隔离账户登录；桌面单行；900/1024px横向几何不相交；手机展开五链接、点击及Esc回焦 | PASS |
| E 影片与Top20 | 目录12张当前页卡片；既有个人Trace20项、rank1—20、真实海报、来源标识及绑定详情链接 | PASS |
| F 权限 | 普通账户研究页无授权表单、profiles为空；读取第二账户个人Trace404 | PASS；无研究身份绑定或授权写入 |
| G 详情 | 从原第一名卡片真实点击；核对排名/分数、海报及演员导演实际加载；移动详情无横向溢出 | PASS |
| H 退出与重登 | 真正POST退出、个人页重定向登录，再用原账户重新登录并退出 | PASS |
| 静态与JS | 收集主批次144个静态响应事件，HTTP均<400；pageerror 0；本地CSS/JS与源字节一致 | PASS |
| 智能助手 | 点击实际入口并核对“未启用Agent”的现有边界 | PASS；不宣称LLM生成 |

布局记录共20条，各已检查视口的scrollWidth均不大于innerWidth。长页面采用正常滚动，未使用overflow-x:hidden掩盖错误。

## 结果保护与范围

[DATABASE_BEFORE.json](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/DATABASE_BEFORE.json>)与[DATABASE_AFTER.json](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/DATABASE_AFTER.json>)证明既有users/preferences/traces/movie_ids/grants字段相同。完整Trace provenance与13条原记录保持不变。[FINAL_RESULT.json](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/FINAL_RESULT.json>)记录从真实DOM提取20项电影ID、顺序和原分数与存储快照一致；这组数据来源明确为普通账户规则推荐，不冒充UKMAP Full。

详情GET按原业务行为新增1条隔离库view事件，交互数量3→4。没有改写正式业务库、模型参数、原映射、研究身份或Agent Trace。Task77公开详情文件只读挂载并通过原SHA核对；演员/导演/类型/已核验关系由既有代码展示，未连接新的Neo4j或LLM服务。

## 批次与故障记录

- 主批次：[BROWSER_ATTEMPT1.json](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/BROWSER_ATTEMPT1.json>)，completed=true，65项全通过。
- 补充最终批次：[BROWSER_SUPPLEMENT.json](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/BROWSER_SUPPLEMENT.json>)，completed=true，8项全通过。
- 补充初次脚本：[BROWSER_SUPPLEMENT_ATTEMPT1.json](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/BROWSER_SUPPLEMENT_ATTEMPT1.json>)，4项通过后因验收代码作用域错误停止；仅修复测试脚本并重新运行该小批次，不重复累计。
- 最终构建/部署退出0：[BUILD_FINAL.exit](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/BUILD_FINAL.exit>)、[DEPLOY_FINAL.exit](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/DEPLOY_FINAL.exit>)。
- 浏览器正常关闭：[BROWSER_CLOSE.exit](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/BROWSER_CLOSE.exit>)；用户内置浏览器和所有Docker核心服务保留。

初始旧页面favicon404已用本地SVG图标修复；最终批次无此静态错误。注册校验错误、无授权界面及越权404是预期用例结果，不是JavaScript异常。本轮不重跑Full GPU评分，不为未出现的缺失海报状态构造Mock或改写目录；对应范围不宣称新增通过。

## 主批次断言清单

| 断言 | 结果 |
|---|---|
| `login_1440x900_no_horizontal_overflow` | PASS |
| `login_1440x900_compact_auth_header` | PASS |
| `login_1440x900_no_duplicate_login_link` | PASS |
| `login_1440x900_usable_input_height` | PASS |
| `login_1440x900_whole_card_and_register_visible` | PASS |
| `login_1440x900_input_focus` | PASS |
| `login_1366x768_no_horizontal_overflow` | PASS |
| `login_1366x768_compact_auth_header` | PASS |
| `login_1366x768_no_duplicate_login_link` | PASS |
| `login_1366x768_usable_input_height` | PASS |
| `login_1366x768_whole_card_and_register_visible` | PASS |
| `login_1366x768_input_focus` | PASS |
| `login_390x844_no_horizontal_overflow` | PASS |
| `login_390x844_compact_auth_header` | PASS |
| `login_390x844_no_duplicate_login_link` | PASS |
| `login_390x844_usable_input_height` | PASS |
| `login_390x844_input_focus` | PASS |
| `login_375x667_no_horizontal_overflow` | PASS |
| `login_375x667_compact_auth_header` | PASS |
| `login_375x667_no_duplicate_login_link` | PASS |
| `login_375x667_usable_input_height` | PASS |
| `login_375x667_input_focus` | PASS |
| `invalid_login_error_readable` | PASS |
| `invalid_login_mobile_no_horizontal_overflow` | PASS |
| `registration_form_present` | PASS |
| `register_mobile_no_horizontal_overflow` | PASS |
| `registration_server_validation_visible` | PASS |
| `registration_error_mobile_no_horizontal_overflow` | PASS |
| `real_existing_account_login` | PASS |
| `mobile_menu_expands` | PASS |
| `mobile_five_clickable_links` | PASS |
| `mobile_menu_no_horizontal_overflow` | PASS |
| `home_1440_no_horizontal_overflow` | PASS |
| `desktop_nav_single_row_1440` | PASS |
| `home_1366_no_horizontal_overflow` | PASS |
| `desktop_nav_single_row_1366` | PASS |
| `home_390_no_horizontal_overflow` | PASS |
| `home_375_no_horizontal_overflow` | PASS |
| `home_visible_real_posters_loaded` | PASS |
| `catalogue_visible_real_posters_loaded` | PASS |
| `catalogue_desktop_no_horizontal_overflow` | PASS |
| `catalogue_cards_visible` | PASS |
| `catalogue_mobile_no_horizontal_overflow` | PASS |
| `existing_trace_available` | PASS |
| `top20_twenty_cards` | PASS |
| `source_rule_not_ukmap` | PASS |
| `rank_numbers_preserved` | PASS |
| `top20_visible_real_posters_loaded` | PASS |
| `top20_desktop_no_horizontal_overflow` | PASS |
| `top20_mobile_no_horizontal_overflow` | PASS |
| `detail_visible_real_posters_loaded` | PASS |
| `detail_desktop_no_horizontal_overflow` | PASS |
| `detail_original_trace_rank_score` | PASS |
| `verified_existing_cast_asset` | PASS |
| `detail_mobile_no_horizontal_overflow` | PASS |
| `ordinary_account_no_research_form` | PASS |
| `research_permissions_no_horizontal_overflow` | PASS |
| `ordinary_account_profiles_empty` | PASS |
| `cross_account_trace_denied` | PASS |
| `assistant_actual_boundary_visible` | PASS |
| `assistant_no_horizontal_overflow` | PASS |
| `logout_requires_login` | PASS |
| `relogin_succeeds` | PASS |
| `no_frontend_js_exceptions` | PASS |
| `no_static_404` | PASS |
| `medium_nav_no_overlap_900` | PASS |
| `medium_nav_no_overlap_1024` | PASS |
| `twenty_bound_ranked_results` | PASS |
| `mobile_menu_escape_returns_focus` | PASS |
| `register_scroll_button_accessible_1440` | PASS |
| `register_scroll_button_accessible_1366` | PASS |
| `register_scroll_button_accessible_390` | PASS |
| `register_scroll_button_accessible_375` | PASS |
