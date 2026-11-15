# MovieAgent 前端工程与多视口回归验证说明（`feat/ch5-frontend-playwright-regression`）

本分支基于 `release/ch5-final-20261114`（`2422b53`）构建，汇总了第五章 `/cinema/*` 模块的原生前端页面、模块化 JavaScript/CSS 静态资源、POST-SSE 流式终态加固代码、纯 CPU 隔离单元/契约测试，以及对应的 Playwright 多视口浏览器回归脚本与脱敏验证记录。

---

## 一、前端架构概览

本项目采用 **Django 5.2 Templates 服务端渲染（MPA）+ 模块化原生 ES6+ JavaScript** 架构（无 Vue/React 及 Node 构建打包依赖），包含两类核心前端实现：

1. **第五章可核验推荐前端（`/cinema/*` 路由族）**
   - **模板层**：位于 `templates/research_recommendation/`（`base.html`、`auth_base.html`、`_movie_card.html`、`home.html`、`movies.html`、`recommend.html`、`record.html`、`detail.html`、`personal.html`、`preferences.html`、`interactions.html`、`login.html`、`register.html`、`error.html` 等）。
   - **模块化脚本（IIFE + `'use strict'` + `<script defer>`）**：
     - `static/research_recommendation/pages.js`：实现模型与数据集联动（`alignModel`）、`busy` 并发互斥锁、`AbortController` 20 秒超时控制、响应契约校验（`validPayload` 校验 64 位 SHA-256 指纹、36 位 UUID `trace_id`、单调分数与映射枚举），以及基于 `document.createElement` + `textContent` + `replaceChildren()` 的零 `innerHTML` 安全 DOM 渲染、图片 `loading="lazy"` 与占位回退 `posterFallback()`。
     - `static/research_recommendation/detail.js`：绑定 `trace_id` 发起 `/cinema/explain/` POST 请求，采用 `busy` 锁与 `textContent` 安全更新。
     - `static/research_recommendation/navigation.js`：移动端折叠菜单渐进增强，支持 `aria-expanded` 状态同步、`Escape` 键关闭菜单并归还焦点，以及 `matchMedia('(min-width: 781px)')` 视口断点自动复位。
   - **响应式样式与可访问性**：`static/research_recommendation/classic.css` 与 `pages.css`，覆盖桌面/移动端多断点（`781–1050px`、`<=780px`、`<=650px`）、`.skip-link`、`:focus-visible` 焦点环与 `@media (prefers-reduced-motion: reduce)`。

2. **Agent 对话 POST-SSE 流式交互与终态状态机（`templates/agent_chat.html`）**
   - 使用原生 `fetch('/agent/stream/', { method: 'POST' })` + `resp.body.getReader()` + `TextDecoder` 逐帧解析 SSE 事件流，配合 ECharts 展示 ReAct 推理流程图与知识图谱子图。
   - 引入单次请求局部终态标志（`receivedDone`、`receivedError`、`terminalHandled`）与 `failStream()` 收口函数，拦截非 2xx HTTP 响应、半包 JSON 异常、未收到 `done` 的异常断流（EOF）以及持久化失败（`AGENT_PERSISTENCE_FAILED`）。

---

## 二、多视口浏览器回归与 73 项历史验证记录说明

### 1. 测试工具与执行方式声明（诚实边界）
- `docs/task78b/browser_regression.js`（主批次 65 项）与 `docs/task78b/browser_supplement.js`（补充批次 8 项）是在开发验收阶段**借助 Playwright CLI 及 AI/MCP 自动化工具辅助编写与驱动**的浏览器回归脚本，直接运行于独立 Headless Chrome（`155.0.8059.84`），**并未自行开发独立的测试框架或引入重型前端工程依赖**。
- 回归测试通过 `page.setViewportSize()` 覆盖 `1440×900`、`1366×768`、`390×844`、`375×667`（及补充 `900×768`、`1024×768`）CSS 视口；属于桌面 Headless 浏览器的多视口 CSS 布局与交互验证，不等同于移动真机硬件测试。

### 2. 73 项历史回归覆盖范围與源码一致性
- **验证范围**：覆盖 `/cinema/*` 登录、注册校验反馈、导航折叠与 `Escape` 焦点回退、首页与影片库渲染、普通账户个人 Top-20 推荐 Trace 保序展示、电影详情页原排名/得分/演职员核对、普通账户对科研接口的越权拦截（无表单、空画像、跨账户 Trace 404）、登出重登闭环。
- **核心指标**：在历史隔离 Docker 验收环境（端口 `18072`）中，**73/73 项检查通过**（65 主项 + 8 补充项），捕获 **`144` 次 `/static/` 静态资源响应 `0` 次 404**，全程 **`pageerror` 运行时异常为 `0`**。
- **版本字节级对应**：本分支中的 `classic.css`（SHA-256 `299fd38b229e7c19af7a4492d3d13289d5f16d107731a51bc64accdd65a9c844`）、`navigation.js`（SHA-256 `c3e556d3eab947bae1a37c7379e865903770e41885eaf81551911cae353be7da`）及 `base.html`/`auth_base.html`/`login.html`/`register.html` 与 `docs/task78b/CHANGED_FILES.md` 和 `docs/task78b/evidence/FINAL_RESULT.json` 记录的 SHA-256 完全一致。
- **复现条件说明**：73 项完整浏览器回归依赖当时隔离的 Docker Web/MySQL 容器（`127.0.0.1:18072`）与预置合成账户数据；**本分支仅归档脱敏后的脚本与证据 JSON（不含私密凭据、数据库快照、海报大图及模型权重），未在新分支上重新启动 Docker/GPU 复跑完整 73 项浏览器流程**。

---

## 三、本分支可直接复跑的纯 CPU 自动化测试（零 GPU / 零外部库依赖）

本分支提供 4 套无需 MySQL、无需 GPU、无需私有权重的纯 CPU 自动化测试，可在数秒内直接验证前端 SSE 状态机与 `/cinema/*` 视图/模板契约：

```bash
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# 1. Agent POST-SSE 终态状态机隔离单测（Node.js node:vm，30 项用例）
node tests/test_agent_sse_terminal_isolated.cjs

# 2. 推荐请求与返回契约单测（Django SimpleTestCase，35 项用例）
python manage.py test research_recommendation.tests.test_contracts --settings=movie.task57_test_settings

# 3. /cinema/* 页面渲染、搜索转义、安全海报协议与授权边界测试（SQLite :memory:，25 项用例）
python manage.py test research_recommendation.task60_support.tests --settings=movie.task60_test_settings

# 4. /cinema/* 注册登录、偏好更新、Top-20 确定性排序、CSRF 与 HTML 转义测试（SQLite :memory:，26 项用例）
python manage.py test research_recommendation.task62_tests --settings=movie.task62_test_settings
```
