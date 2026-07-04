# MovieAgent

基于 ReAct Agent 与知识图谱的可解释电影推荐系统。

用户通过自然语言对话获取推荐，Agent 自主完成意图识别、多路召回、精排、重排、生成推荐理由的完整推理链，全程可追溯。

## 系统架构

```
用户（浏览器）
  ↓
Django View（HTTP / SSE 流式）
  ↓
AgentChatService（业务服务层）
  ↓
MovieAgent（ReAct 推理引擎）
  ├── 意图分类器（规则优先 + LLM 兜底）
  ├── Thought → Action → Observation 循环（最多 3 轮）
  │     ├── search_vector   FAISS 向量语义召回
  │     ├── recall_hybrid   多路召回（向量+内容+模型+图谱+热门）
  │     ├── kg_query        Neo4j 知识图谱查询
  │     ├── maan_rerank     MMAN 多模态精排
  │     ├── rerank          业务规则重排
  │     └── explain         生成个性化推荐理由
  └── Final Answer（LLM 生成最终回复）
  ↓
持久化（ChatHistory + AgentTrace）
  ↓
用户（推荐结果 + 推荐理由 + 推理链展示）
```

## 技术栈

| 层级 | 技术 |
|------|------|
| Backend | Python 3.11 / Django 5.2 / Uvicorn (ASGI) |
| Frontend | HTML / Bootstrap 3.4 / ECharts / AJAX |
| Database | MySQL 8.0 / Neo4j 5 / Redis 7 / FAISS |
| AI / LLM | Ollama + qwen3:4b-instruct / LangChain / BGE-small-zh-v1.5 / CLIP-ViT |
| Deep Learning | PyTorch / DeepCTR-Torch / SKB-FMLP / MMAN |
| Deployment | Docker Compose（5 服务编排）|

## 核心功能

- **ReAct Agent 智能推荐** — 自然语言对话，Agent 自主调用工具完成推荐全流程
- **五路并行召回** — 向量语义 + 内容特征 + 深度模型 + 知识图谱 + 热门兜底
- **知识图谱问答（GraphRAG）** — Neo4j 图谱 Cypher 自动生成，支持导演风格、演员合作等关系推理
- **向量语义检索（Vector RAG）** — FAISS + BGE 中文语义检索
- **可解释推荐** — 每部推荐电影附带个性化推荐理由，推理链全程可视化
- **多模态特征融合** — CLIP 视觉向量 + 文本嵌入，MMAN 注意力网络精排
- **管理员后台** — 电影 / 用户 / 演员 / 导演 / 评论 / 图谱 / 画像 / 推理链 / 反馈的完整 CRUD
- **离线评估** — AUC / NDCG / HR@K / MRR 指标计算

## 项目亮点

- **ReAct 推理链** — Thought → Action → Observation → Final Answer，每轮记录完整 Trace
- **多意图分支检测** — 自动识别"推荐科幻和爱情片"等并列意图，生成分支选项
- **自反馈纠偏** — 工具返回空结果时自动切换备用策略，模糊查询时主动追问
- **Importance-aware 知识图谱** — 融合模型特征贡献度，对导演/类型/演员差异化赋权
- **拟人化 XAI** — 将机器推理步骤翻译为用户友好的白话文
- **SSE 流式对话** — 打字机效果实时展示 Agent 推理过程
- **未成年保护** — 重排阶段自动过滤敏感内容，候选不足时从安全池补充

## 快速开始

### Docker 部署（推荐）

```bash
# 1. 克隆项目
git clone https://github.com/YANS311/MovieAgent.git
cd MovieAgent

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填写必填项：DJANGO_SECRET_KEY、DB_PASSWORD、NEO4J_PASSWORD

# 3. 启动全部服务
docker-compose up -d

# 4. 访问
# http://localhost:8000
```

Docker Compose 自动编排 5 个服务：

| 服务 | 镜像 | 端口 |
|------|------|------|
| mysql | mysql:8.0 | 3306 |
| redis | redis:7 | 6379 |
| neo4j | neo4j:5 | 7474 / 7687 |
| ollama | ollama/ollama:latest | 11434 |
| django | 自建（Dockerfile） | 8000 |

entrypoint.sh 会自动等待 MySQL 就绪、执行数据库迁移、收集静态文件、启动 Uvicorn（默认 8 Worker）。

### 本地开发

```bash
# 1. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env

# 4. 确保以下服务已启动：
#    - MySQL 8.0
#    - Redis 7
#    - Neo4j 5
#    - Ollama（已拉取 qwen3:4b-instruct 模型）

# 5. 数据库迁移
python manage.py migrate

# 6. 导入数据（可选）
python manage.py import_movielens        # MovieLens-1M 数据集
python manage.py import_new_movies       # TMDB 新片（需 TMDB_API_KEY）
python manage.py build_kg                # 构建知识图谱
python manage.py build_rag_index         # 构建 RAG 向量索引

# 7. 启动开发服务器
python manage.py runserver
```

## 环境变量说明

复制 `.env.example` 为 `.env` 并填写：

| 变量 | 必填 | 说明 |
|------|------|------|
| `DJANGO_SECRET_KEY` | 生产必填 | Django 密钥，开发环境可留空自动生成 |
| `DJANGO_DEBUG` | 可选 | 调试模式，默认 False |
| `ALLOWED_HOSTS` | 可选 | 允许访问的主机名，逗号分隔 |
| `DB_PASSWORD` | **必填** | MySQL 数据库密码 |
| `NEO4J_PASSWORD` | **必填** | Neo4j 图数据库密码 |
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` | 可选 | MySQL 连接配置 |
| `NEO4J_URI` / `NEO4J_USER` | 可选 | Neo4j 连接配置 |
| `REDIS_URL` | 可选 | Redis 连接地址 |
| `OLLAMA_BASE` | 可选 | Ollama 服务地址 |
| `AGENT_LLM_MODEL` | 可选 | LLM 模型名称，默认 qwen3:4b-instruct |
| `TMDB_API_KEY` | 可选 | TMDB API Key，用于导入新电影 |

## 管理命令

```bash
# 数据导入
python manage.py import_movielens          # 导入 MovieLens-1M
python manage.py import_new_movies         # TMDB 爬取新电影
python manage.py enrich_tmdb_data          # TMDB 数据补全
python manage.py merge_and_clean           # 合并清洗多源数据

# 知识图谱与索引
python manage.py build_kg                  # 构建知识图谱（Neo4j）
python manage.py build_rag_index           # 构建 RAG 向量索引（FAISS）
python manage.py build_visual_index        # 构建视觉索引

# 模型与推荐
python manage.py train_hybrid              # 训练混合推荐模型
python manage.py calc_recs                 # 离线计算推荐结果

# 评估
python manage.py run_agent_eval            # Agent 离线评估
python manage.py run_llm_comparison        # LLM 模型对比

# 基准测试
python manage.py system_benchmark          # 系统综合基准
python manage.py benchmark_latency         # 延迟基准
python manage.py load_test                 # 负载测试
```

## 项目结构

```
MovieAgent/
├── manage.py                    Django 入口
├── movie/                       Django 项目配置
│   ├── settings.py              数据库 / 缓存 / LLM / 安全配置
│   └── urls.py                  根路由
├── myapp/                       Django 主应用
│   ├── models.py                数据模型
│   ├── views.py                 前台视图
│   ├── agent_views.py           Agent 视图
│   ├── agent/                   ReAct Agent 引擎
│   │   ├── movie_agent.py       MovieAgent 核心
│   │   └── memory.py            对话记忆管理
│   ├── recommender/             推荐管线
│   │   ├── recall.py            五路并行召回
│   │   ├── rank.py              精排
│   │   ├── rerank.py            重排
│   │   └── explain.py           推荐理由生成
│   ├── services/                业务服务层
│   ├── utils/                   工具模块（GraphRAG / VectorRAG / XAI）
│   ├── management/commands/     Django 管理命令（30+）
│   └── migrations/              数据库迁移
├── templates/                   HTML 模板
├── static/                      静态资源
├── experiments/                 实验脚本（run_*.py / 消融 / 基准测试）
├── tests/                       测试脚本
├── scripts/                     数据处理与工具脚本
├── Dockerfile                   Docker 镜像定义
├── docker-compose.yml           Docker 编排（5 服务）
├── entrypoint.sh                Docker 启动脚本
├── requirements.txt             Python 依赖
├── .env.example                 环境变量模板
└── LICENSE                      MIT License
```

## 许可证

[MIT License](LICENSE)
