# Task78-B 真实截图索引

14张均为运行中的Docker服务经真实Chrome渲染保存的PNG，未经拼接、生成或重绘，已逐张视觉核对。URL、浏览器视口与PNG尺寸分别登记：fullPage图片可以高于视口。浏览器为Chrome155.0.8059.84，独立headless验收；账户为Task72合成测试身份。

论文替换建议：01替换登录图，03替换首页图，05替换电影详情图；02可用于移动适配。04为普通账户规则Top20完整长图，适合附录或按论文版式选择展示位置，不能标成UKMAP模型推理。14只能证明无授权状态，不能充当有授权推荐结果图。08/09保留为错误反馈测试证据。

| 文件 | 视口 | 实际PNG尺寸 | 状态 | SHA-256 |
|---|---|---|---|---|
| [01_login_desktop.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/01_login_desktop.png>) | 1440×900 | 1440×900 | PASS | `75849d6e66a917630f528cb2a5fc564ad15330edd0ac5da7b2a281936dcd5aa4` |
| [06_login_1366.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/06_login_1366.png>) | 1366×768 | 1366×768 | PASS | `c7a9a73f4ce7fca8fb3b595575a6fc2622b4ee259a0fe2c6f0934609c6c99795` |
| [02_login_mobile.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/02_login_mobile.png>) | 390×844 | 390×844 | PASS | `f49996c2a0d36102adfad66f0941cd60082608185386a166ed47cdcf4a302563` |
| [07_login_375.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/07_login_375.png>) | 375×667 | 375×667 | PASS | `467086b01c7445c7200b671dbfc0382e2d5ba421d0753bcd62a93078eba677d9` |
| [08_login_error.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/08_login_error.png>) | 375×667 | 375×705 | PASS | `a7caa3f6610c44e872c965e9205bc24f6c02d5763e3e96f8fc8cce9b325d71b5` |
| [09_register_error.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/09_register_error.png>) | 375×667 | 375×984 | PASS | `305c009da30675f88c9e58e1e31d388a55c42b05ffdedae9ae09410c48f9ded0` |
| [10_mobile_menu.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/10_mobile_menu.png>) | 375×667 | 375×1380 | PASS | `20097b8d42d5aa39aecf77ee65f6451fc926546feee34de54bbc17cf47f70aba` |
| [03_home.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/03_home.png>) | 1440×900 | 1440×900 | PASS | `7c44025cacf458671650fb9308fad35b738d6cae89e4d2972339c556ecf4f39d` |
| [11_catalogue_mobile.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/11_catalogue_mobile.png>) | 375×667 | 375×667 | PASS | `8c4156eb0449907abddd5d70749b0cf5fbf3e4e2268b6c1c18a50b9b3698b937` |
| [04_recommend_top20.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/04_recommend_top20.png>) | 1440×900 | 1440×3437 | PASS | `9abc0e8d6d53acc30ee510f7b2698a3f6126935bb12241299ada245571c22994` |
| [12_top20_mobile.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/12_top20_mobile.png>) | 375×667 | 375×667 | PASS | `6f3a33568bee12b206f00a2e75dc82748a1bc209733a9b69aba1207b267f2631` |
| [05_movie_detail.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/05_movie_detail.png>) | 1440×900 | 1440×900 | PASS | `fa84cb4dbacd0d3129f44e1f165f041bc104ed53fe2cf1999e97768aebb9678d` |
| [13_detail_mobile.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/13_detail_mobile.png>) | 375×667 | 375×2795 | PASS | `603eb09cb0f63985246599365c3933e49b08f599ec72a098b5664ac94c7a0a4e` |
| [14_research_permissions.png](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/screenshots/14_research_permissions.png>) | 1440×900 | 1440×900 | PASS | `e544046209d70219bc682eaeba1bddaf651714583281a659018e7798098a89a9` |

## URL与捕获时间

- **01_login_desktop.png**：`http://127.0.0.1:18072/cinema/login/`；文件时间UTC `2026-10-22T10:04:28.248191+00:00`。
- **06_login_1366.png**：`http://127.0.0.1:18072/cinema/login/`；文件时间UTC `2026-10-22T10:04:29.001210+00:00`。
- **02_login_mobile.png**：`http://127.0.0.1:18072/cinema/login/`；文件时间UTC `2026-10-22T10:04:29.660228+00:00`。
- **07_login_375.png**：`http://127.0.0.1:18072/cinema/login/`；文件时间UTC `2026-10-22T10:04:30.287244+00:00`。
- **08_login_error.png**：`http://127.0.0.1:18072/cinema/login/`；文件时间UTC `2026-10-22T10:04:31.198268+00:00`。
- **09_register_error.png**：`http://127.0.0.1:18072/cinema/register/`；文件时间UTC `2026-10-22T10:04:31.488276+00:00`。
- **10_mobile_menu.png**：`http://127.0.0.1:18072/cinema/personal/`；文件时间UTC `2026-10-22T10:04:32.430300+00:00`。
- **03_home.png**：`http://127.0.0.1:18072/cinema/`；文件时间UTC `2026-10-22T10:04:36.140397+00:00`。
- **11_catalogue_mobile.png**：`http://127.0.0.1:18072/cinema/movies/`；文件时间UTC `2026-10-22T10:04:36.296401+00:00`。
- **04_recommend_top20.png**：`http://127.0.0.1:18072/cinema/personal/traces/0bd9ded5-57d2-4a52-89ad-a62b4ef58185/`；文件时间UTC `2026-10-22T10:07:42.673336+00:00`。
- **12_top20_mobile.png**：`http://127.0.0.1:18072/cinema/personal/traces/0bd9ded5-57d2-4a52-89ad-a62b4ef58185/`；文件时间UTC `2026-10-22T10:04:37.329428+00:00`。
- **05_movie_detail.png**：`http://127.0.0.1:18072/cinema/movies/75425/?personal_trace=0bd9ded5-57d2-4a52-89ad-a62b4ef58185`；文件时间UTC `2026-10-22T10:04:37.570435+00:00`。
- **13_detail_mobile.png**：`http://127.0.0.1:18072/cinema/movies/75425/?personal_trace=0bd9ded5-57d2-4a52-89ad-a62b4ef58185`；文件时间UTC `2026-10-22T10:04:37.674437+00:00`。
- **14_research_permissions.png**：`http://127.0.0.1:18072/cinema/recommend/`；文件时间UTC `2026-10-22T10:04:37.844442+00:00`。

机器可读索引：[SCREENSHOT_INDEX.json](</home/daylight/下载/DjangoProject3/MovieAgent-task57-delivery/docs/task78b/evidence/SCREENSHOT_INDEX.json>)。业务状态和截图均为本轮实际现场，不将以前截图重新命名充作新结果。
