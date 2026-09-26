# Disaster Assessment Agent

面向遥感影像的灾害评估智能体。用户可以在网页中上传影像或地理数据，用自然语言提出分析任务；后端通过 LangGraph 调用经过筛选的 MCP 工具，返回文字结论、可下载的分析产物，并在有空间范围时将结果显示在地图上。

当前默认工具涵盖建筑损毁、洪水淹没、火烧迹地、溢油和藻华识别，以及遥感指数、变化趋势、空间统计、GeoAI 分析和联网搜索。默认接入范围以 [`agent/tool_policy.py`](agent/tool_policy.py) 为准；工具文件存在并不代表已接入智能体。需要报告时，可以在对话中明确要求生成 PDF。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `backend_api.py` | FastAPI 接口、对话执行、会话恢复和报告生成 |
| `agent/` | 智能体工具、工具路由、错误反馈、数据库与产物存储 |
| `frontend/chatDisaster/` | Vue 3 + Vite 前端，包含聊天、历史会话、地图和报告预览 |
| `migrations/` | PostgreSQL / PostGIS 数据库迁移 |
| `model/` | 各分析工具使用的模型及权重；该目录内容不随 Git 提交 |
| `data/artifacts/` | 上传文件、分析结果和报告的持久化文件目录 |
| `scripts/migrate_legacy_artifacts.py` | 将旧会话中的文件迁移到统一产物目录 |

## 运行要求

- Python：当前运行环境为 **3.10.20**，使用 Conda 环境 `earthagent_cpython`。仓库尚无统一的 Python 依赖清单或环境文件；在新机器上部署时，需要另外准备后端及所用模型工具的依赖。
- Node.js：**20.19+ 或 22.12+**，以及 npm。
- PostgreSQL + PostGIS：示例使用 `postgis/postgis:16-3.4`，本机端口 `5433`。项目没有 Docker Compose 文件。
- 可访问的 OpenAI 兼容模型服务。图像分析工具还需要对应的本地模型权重；`model/` 未包含在 Git 仓库中。

## 快速启动

以下命令均从项目根目录开始。数据库、后端和前端依次启动；后两者分别占用一个终端。

### 1. 启动数据库

如果本机已有 `disaster_agent_db` 容器：

```bash
docker start disaster_agent_db
```

首次创建数据库时可使用：

```bash
docker run -d --name disaster_agent_db \
  -e POSTGRES_DB=disaster_agent \
  -e POSTGRES_USER=disaster_agent \
  -e POSTGRES_PASSWORD=change-this-password \
  -p 127.0.0.1:5433:5432 \
  -v disaster_agent_pg_data:/var/lib/postgresql/data \
  postgis/postgis:16-3.4
```

数据库数据保存在 Docker 卷中。首次启动可能需要等待数据库完成初始化。

### 2. 配置后端

在项目根目录创建 `.env`，填写与数据库一致的密码，以及实际可用的模型地址和密钥：

```dotenv
DB_HOST=127.0.0.1
DB_PORT=5433
DB_NAME=disaster_agent
DB_USER=disaster_agent
DB_PASSWORD=change-this-password

QWEN35_API_KEY=your-model-api-key
QWEN_SERVER_URL=http://your-model-host:8001/v1
```

设置 `QWEN35_API_KEY` 后，后端会使用 `qwen3.5-9b` 和 `QWEN_SERVER_URL`。如果使用其他模型，请移除 `QWEN35_API_KEY`，再调整 [`agent/config.json`](agent/config.json) 中的模型配置；可通过 `AGENT_CONFIG` 指定 `agent/` 下的其他配置文件。后端会读取根目录 `.env`，其中的值会覆盖同名进程环境变量。不要将密钥提交到 Git。

### 3. 启动后端

在已安装项目所需 Python 依赖的环境中执行：

```bash
conda activate earthagent_cpython
python -m uvicorn backend_api:app --host 127.0.0.1 --port 8000
```

后端启动时会自动执行 `migrations/*.sql`；数据库无法连接或迁移失败时，服务不会启动。

验证后端：

```bash
curl --noproxy '*' http://127.0.0.1:8000/api/health
```

返回结果中的 `ok` 和 `db_ok` 应为 `true`。健康检查确认的是服务与数据库状态，实际对话还需要模型服务和所用工具可用。

### 4. 启动前端

另开终端执行：

```bash
cd frontend/chatDisaster
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

打开 <http://127.0.0.1:5173>。Vite 将 `/api` 请求代理到本机 `8000` 端口。需要使用地图时，在 `frontend/chatDisaster/.env.local` 中配置：

```dotenv
VITE_ARCGIS_API_KEY=your-arcgis-api-key
```

修改前端环境变量后需重启 Vite。没有该密钥时，聊天仍可使用，但地图无法初始化。

### 5. 配置检索向量化服务

项目会从已接入的工具说明和当前会话的旧文件中检索参考资料。文本向量化调用服务器上的 `embed_server.py`，它提供 `POST /embed` 接口。脚本默认监听 `8000` 端口；在根目录 `.env` 中可指定：

```dotenv
EMBED_SERVER_URL=http://172.31.233.78:8000
EMBED_MODEL_NAME=Qwen3-Embedding-4B
```

确保后端机器可访问该地址，并可通过 `curl --noproxy '*' http://172.31.233.78:8000/health` 检查。向量保存在 PostgreSQL 的 `retrieval_embeddings` 表中，服务暂时不可用时会使用关键词检索。当前语料规模较小，相似度在 Python 中计算，不要求安装 `pgvector`。检索到的文件严格限制在当前会话；回答下方会显示本轮使用的工具说明和文件引用。

### 6. 配置联网搜索

`web_search` 已接入智能体。询问“今天的地震新闻”“搜索最新洪水预警”等实时问题时，路由会提供这个工具。搜索结果包含标题、摘要、原始链接、发布时间和检索时间；本轮链接会保存在会话历史中，并显示在回答下方。外部摘要仅作参考，重要灾情应打开来源核实。

未配置搜索服务时，默认使用 Google News RSS，**只能搜索新闻**，不代表全网网页。通用网页搜索可在根目录 `.env` 选择以下一种配置：

```dotenv
# Brave Search API（需要自行申请密钥）
WEB_SEARCH_PROVIDER=brave
BRAVE_SEARCH_API_KEY=your-brave-search-key

# 或自建 SearXNG，并启用 JSON 返回格式
# WEB_SEARCH_PROVIDER=searxng
# SEARXNG_URL=http://your-searxng-host:8080

# 可选：每次请求的超时秒数，限制在 2–20 秒
WEB_SEARCH_TIMEOUT_SECONDS=10

# 可选：后端作为 systemd 服务运行且需通过代理访问外网时
# WEB_SEARCH_PROXY=socks5h://127.0.0.1:your-proxy-port
```

也可以省略 `WEB_SEARCH_PROVIDER`：后端依次选用已配置的 Brave、SearXNG，否则使用新闻检索。若明确设置 `WEB_SEARCH_PROVIDER=news`，则始终使用免密钥新闻检索。修改 `.env` 后重启后端。SearXNG 的 `settings.yml` 须将 `json` 加入 `search.formats`；其 [搜索 API 文档](https://docs.searxng.org/dev/search_api.html) 和 [Brave Web Search API 文档](https://api-dashboard.search.brave.com/app/documentation/web-search) 提供服务端配置说明。若机器通过 SOCKS 代理访问外网，Python 环境还需要 `requests[socks]`。`WEB_SEARCH_PROXY` 只影响搜索请求；不设置时沿用进程的标准代理环境变量。

可先在项目根目录验证搜索工具：

```bash
python -c "from agent.web_search import search_web; print(search_web('地震 灾害', 3))"
```

## 使用方式

在页面上传遥感影像或相关数据，描述要识别的灾害和希望得到的结果。涉及灾前灾后对比时，注明各文件的时间和角色。执行完成后，页面会展示结论和可用的结果文件；开启轨迹显示后还能查看工具调用过程。具有地理范围的评估可在地图中查看。需要 PDF 时，在消息中明确提出“生成报告”。

主要接口：

| 接口 | 作用 |
| --- | --- |
| `GET /api/health` | 服务、数据库及启用工具状态 |
| `POST /api/chat/stream` | 流式对话；使用 `multipart/form-data`，包含 UUID 格式的 `session_id`、`message` 和可选的 `files` |
| `POST /api/chat` | 非流式对话，字段同上 |
| `GET /api/sessions` | 最近会话 |
| `GET /api/sessions/{session_id}/messages` | 会话消息 |
| `GET /api/sessions/{session_id}/turns` | 各轮的 `running`、`completed` 或 `failed` 状态 |
| `GET /api/sessions/{session_id}/assessments` | 评估结果及产物链接 |
| `GET /api/files/{file_id}` | 下载或预览产物 |

## 数据与迁移

会话、消息、轮次状态、评估结果和产物元数据保存在 PostgreSQL；上传文件与生成文件保存在 `data/artifacts/`。部署时需要同时持久化数据库卷和该文件目录。后端可从数据库恢复历史会话；失败轮次会记录失败状态。实现细节见 [数据库与产物持久化](docs/数据库持久化.md)。

已有旧数据需要迁移文件引用时，先预览，再执行：

```bash
python scripts/migrate_legacy_artifacts.py
python scripts/migrate_legacy_artifacts.py --apply
```

新数据库无需执行此脚本。旧记录若只保留了已失效的文件路径，无法凭路径恢复文件内容。

## 验证与更多文档

```bash
python -m unittest test_error_feedback test_tool_policy test_tool_router test_persistence test_retrieval test_web_search
RUN_DB_INTEGRATION=1 python -m unittest test_persistence.DatabaseIntegrationTests test_retrieval.RetrievalDatabaseTests
cd frontend/chatDisaster && npm run build
```

数据库集成测试需要可用的本地数据库。更多工具选择依据见 [Agent 工具接入筛选](docs/工具接入筛选.md)。当前接口按会话 ID 访问数据；如需向不受信任的用户开放服务，还应增加身份认证和会话权限控制。
