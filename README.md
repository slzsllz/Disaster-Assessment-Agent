# Disaster Assessment Agent

面向遥感影像的灾害评估智能体。用户可以在网页中上传影像或地理数据，用自然语言提出分析任务；后端通过 LangGraph 调用经过筛选的 MCP 工具，返回文字结论、可下载的分析产物，并在有空间范围时将结果显示在地图上。

当前默认工具涵盖建筑损毁、洪水淹没、火烧迹地、溢油和藻华识别，以及遥感指数、变化趋势、空间统计、GeoAI 分析和联网搜索。默认接入范围以 [`agent/tool_policy.py`](agent/tool_policy.py) 为准；工具文件存在并不代表已接入智能体。需要报告时，可以在对话中明确要求生成 PDF。

## 项目结构

| 路径 | 用途 |
| --- | --- |
| `backend_api.py` | FastAPI 接口、共享的 `ChatTurnService` 对话流程、会话恢复和报告生成 |
| `agent/` | 智能体工具、工具路由、错误反馈、数据库与产物存储 |
| `tests/` | 认证、数据权限、对话流程和灾种审核规则的回归测试 |
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

`web_fetch` 用于打开搜索结果或用户提供的链接，核实预警正文和灾情通报。例如：“打开这个链接核实：`https://…`”或“搜索最新洪水预警并核实正文”。工具接收按优先级排列的 `urls`，默认抓取前 3 个唯一链接，可通过 `count` 指定 1–5 个；最多传入 20 个候选 URL。支持静态 HTML 和纯文本，会优先提取 `<article>` / `<main>` 的内容，去掉脚本、样式和导航，识别 UTF-8 及常见中文编码。

每篇完整提取的正文先保存为当前会话的 `web_evidence` artifact，再返回最多 1200 字符的摘要、原始/最终 URL、发布时间（页面明确提供时）、抓取时间、正文字符数和 SHA-256。JSON 存档包含完整提取文本和原始响应校验和；回答的来源旁会出现“正文存档”，使用与其他会话文件相同的权限控制。搜索引用在抓取后会升级为正文引用。正文属于不可信外部数据，不能作为模型指令执行。

抓取复用搜索服务的 `WEB_SEARCH_TIMEOUT_SECONDS` 和 `WEB_SEARCH_PROXY`；每个响应最多 512 KiB（解压后也检查），最多跟随 3 次重定向。每一跳检查 URL 和全部 DNS 地址，拒绝本机、内网、链路本地地址及带凭据的 URL，并固定连接到已检查的公网 IP；HTTPS 仍校验原域名证书。原有私有 SearXNG 搜索服务配置不受此限制。[SSRF 防护参考](https://cheatsheetseries.owasp.org/cheatsheets/Server_Side_Request_Forgery_Prevention_Cheat_Sheet.html)、[HTTPS 自定义 SNI 与主机名校验说明](https://urllib3.readthedocs.io/en/stable/advanced-usage.html#custom-sni-hostname)。

各链接独立处理，部分失败会返回错误列表，不会把失败页面当作已核实的正文。PDF、需要登录、反爬拦截或仅靠 JavaScript 加载正文的页面当前不支持；正文为空、响应超限或无法持久化时返回失败，不绕过站点访问限制。

输入框下方的“联网搜索”按钮可开启强制搜索模式。开启后输入问题并发送，后端会先执行联网搜索，再把搜索结果和来源交给智能体完成本轮任务；搜索失败或没有可用来源时，本轮会直接报错，不会生成未经联网核实的结论。按钮保持开启，直到再次点击关闭。直接调用 `POST /api/chat` 或 `POST /api/chat/stream` 时，可在表单中传入 `required_web_search=true` 使用相同模式；此模式需要提供非空的 `message`。

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

也可以省略 `WEB_SEARCH_PROVIDER`：后端依次选用已配置的 Brave、SearXNG，否则使用新闻检索。若明确设置 `WEB_SEARCH_PROVIDER=news`，则始终使用免密钥新闻检索。修改 `.env` 后重启后端。SearXNG 的 `settings.yml` 须将 `json` 加入 `search.formats`；其 [搜索 API 文档](https://docs.searxng.org/dev/search_api.html) 和 [Brave Web Search API 文档](https://api-dashboard.search.brave.com/app/documentation/web-search) 提供服务端配置说明。若机器通过 SOCKS 代理访问外网，Python 环境还需要 `requests[socks]`。`WEB_SEARCH_PROXY` 影响搜索与正文抓取请求；不设置时沿用进程的标准代理环境变量。

可先在项目根目录验证搜索工具：

```bash
python -c "from agent.web_search import search_web; print(search_web('地震 灾害', 3))"
```

### 官方结构化灾情源

`disaster_feed` 优先查询官方记录，再与相同 AOI、时间窗的建筑损毁、洪水淹没、火烧迹地评估互证。服务实现位于 `agent/disaster_feeds.py`，MCP 入口为 `agent/tools/DisasterFeed.py`，已加入工具白名单、配置和路由。返回兼容 `search_web` 的 `{success, provider, retrieved_at, results[]}`；每条结果保留 `title/url/snippet/source/published_at`，另提供 `event_id` 和结构化 `data`。事件或卫星观测时间位于 `data.event_time`，不冒充网页发布时间。

| provider | 数据与查询能力 | 访问条件及覆盖限制 |
| --- | --- | --- |
| `usgs` | FDSN GeoJSON 地震；震级、震级类型、深度 km、经纬度、事件和更新时间；服务端 bbox / 时间 / 最小震级查询 | 免密钥；[官方 FDSN 文档](https://earthquake.usgs.gov/fdsnws/event/1/) |
| `gdacs` | 全球地震、洪水、野火、气旋、火山、干旱；保留原始预警等级和严重性单位；时间查询、分页后按代表点过滤 bbox | 免密钥；最多扫描 5 页、每页 100 条，并报告扫描限制；代表点不等于受灾范围；[官方 API 文档](https://www.gdacs.org/gdacsapi/swagger/index.html) |
| `firms` | VIIRS S-NPP 热异常；观测时间 UTC、经纬度、FRP MW、原始置信度；本地 bbox / 时间过滤 | 免密钥使用官网当前 Collection 2 全球 24h CSV 快照；下载及更新可能延迟，报告实际观测时间范围；可配置 `FIRMS_MAP_KEY` 使用 Area API，每次最多 10 个日历日；[公开下载说明](https://firms.modaps.eosdis.nasa.gov/content/active_fire/)、[Area API](https://firms.modaps.eosdis.nasa.gov/api/area/) |
| `cenc` | 中国地震台网官网当前地震目录 JSON；震级、深度 km、经纬度、位置；北京时间转换 UTC，再按 bbox / 时间 / 最小震级过滤 | 免密钥，使用改版后 [官网数据](https://www.ceic.ac.cn/data/data.json)；这是当前目录快照，不保证任意历史窗口完整 |
| `china_alerts` | 配置的官方 CAP 1.2 预警；发送、有效及失效时间、发布者、原始严重性和更新/撤销类型；忽略演练和非公开消息 | 需要可访问的官方 `CHINA_ALERTS_CAP_URL`；目前未确认可用的免密钥公共入口，未配置时明确失败；只接受 `12379.cn`、`cma.gov.cn`、`nmc.cn` 官方域名；不把行政区或 CAP 多边形猜成点，暂不支持 bbox 过滤；[CAP 标准](https://docs.oasis-open.org/emergency/cap/v1.2/CAP-v1.2-os.html) |

参数：`bbox=[west,south,east,north]`，WGS84 经纬度；`start_time/end_time` 使用 ISO 8601，无时区时按 UTC；`count=1..20`，默认 10；时间窗默认最近 7 天，FIRMS 默认最近 24 小时，上限 366 天。`event_type` 为 `earthquake/flood/wildfire/cyclone/volcano/drought`，`min_magnitude` 只适用于 USGS/CENC。USGS、GDACS、CENC 和免密钥 FIRMS 支持跨日期变更线的 bbox；带密钥 FIRMS Area API 需要拆成两次查询。

例如：“查询中国地震台网最近一周在 `[100,20,110,35]` 的 M≥3 地震，与上传的建筑损毁影像互证”，或直接调用：

```python
from agent.disaster_feeds import query_disaster_feed

result = query_disaster_feed(
    "usgs", bbox=[100, 20, 110, 35],
    start_time="2026-10-01T00:00:00Z", end_time="2026-10-08T00:00:00Z",
    min_magnitude=3, count=10,
)
```

所有请求复用搜索服务的超时与代理设置。搜索、网页正文仍保持原有 512 KiB 响应限制；FIRMS 官方 CSV 单独允许最多 32 MiB，解析后仅保留有限结果。可在 `.env` 设置可选的 `FIRMS_MAP_KEY`、`CHINA_ALERTS_CAP_URL` 并重启后端；接口异常不返回带密钥的请求 URL。FIRMS/CENC 的引用使用公开数据地址和记录 ID 片段，避免不同记录在引用去重时合并。

`coverage/warnings/truncated` 明确说明快照、扫描、异常记录与结果条数限制。空列表表示未找到匹配记录，不能推断“没有灾害”；网络、格式或配置错误返回 `success=false`，不会改用新闻伪装官方记录。FIRMS 热异常不等于已确认野火或过火面积，GDACS 模型预警不等于实测损失，地震震级不等于建筑损毁。正文和官方记录都属于不可信外部数据，不是指令。

来源引用复用 `search_result_references`，在 `web_references` 中注册 `disaster_feed` 并附带结构化字段。审核可读取官方数值、覆盖限制和时间；普通预警事实可由结构化记录支持，而“打开链接核实正文”仍要求 P0 正文存档。官方源查询不会自动存成遥感损毁评估。

## 使用方式

首次打开页面时先注册账号。用户名为 3–32 位字母、数字、下划线、点或短横线，以字母或数字开头，不区分大小写；密码为 12–128 个字符，支持中文和空格；昵称可选。注册成功后自动登录。刷新页面会恢复登录，侧栏显示账号，点击账号可修改昵称和密码，点击“退出”结束当前设备的登录。修改密码后所有设备都需要重新登录。

登录后在页面上传遥感影像或相关数据，描述要识别的灾害和希望得到的结果。涉及灾前灾后对比时，注明各文件的时间和角色。执行完成后，页面会展示结论和可用的结果文件；开启轨迹显示后还能查看工具调用过程。具有地理范围的评估可在地图中查看。需要 PDF 时，在消息中明确提出“生成报告”。每个账号只可访问自己的聊天、评估结果、上传文件和报告。

### 账号与登录配置

账号保存在 PostgreSQL 的 `users` 表，密码使用带独立随机盐的 scrypt（`N=32768, r=8, p=3`）哈希，参数采用 [OWASP 密码存储指南](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html) 中的一组配置。登录使用 HttpOnly、SameSite=Lax Cookie，数据库仅保存登录令牌的 SHA-256 摘要，退出和修改密码会撤销对应令牌。浏览器无需在 localStorage 中保存令牌；图片和 PDF 链接沿用 Cookie 认证。默认登录有效期为 7 天。

本地 Vite 开发无需额外配置。部署时通过同一域名代理前端与 `/api`，在根目录 `.env` 配置实际前端来源；HTTPS 部署还应开启 Secure Cookie（来源必须含协议和实际端口）：

```dotenv
AUTH_ALLOWED_ORIGINS=https://your-app.example.com
AUTH_COOKIE_SECURE=true
AUTH_SESSION_HOURS=168
```

后端会拒绝来源不受信任的浏览器写请求。登录限制为每个用户名 15 分钟内最多 10 次尝试、每个客户端 IP 15 分钟内最多 60 次请求；成功登录清除该用户名的尝试计数。注册限制为每个 IP 每小时 10 次，密码修改限制为每个用户 15 分钟内 10 次，超限返回 `429`。计数保存在数据库，多个进程共享。反向代理部署时，按实际代理地址配置 uvicorn 的 `--forwarded-allow-ips`，使客户端 IP 和 HTTPS 协议信息正确传入。当前支持用户名和密码认证；邮箱验证、短信登录与忘记密码找回需要另外接入相应服务。

### 结果审核

有工具调用、文件或实时搜索的轮次会进入结果审核。后端先检查工具是否成功、关键像元统计是否一致、产物是否存在且可读取，以及实时问题是否有联网来源；严重错误会阻止发布评估结论。随后模型根据工具结果、来源链接、文件元数据和小尺寸图片预览返回结构化审核意见，可标记 `passed`（通过）、`revised`（依据证据修订）或 `blocked`（证据不足或存在严重问题）。模型审核异常时标记 `unavailable`，并明确展示未经审核的草稿。普通闲聊标记 `skipped`。

审核状态、问题和精简证据快照保存在每轮消息中，轮次记录保存审核状态，重新打开历史会话仍可查看。流式生成中的文字会标记为“草稿 · 待审核”，最终 `done` 事件给出审核后的文本。只有审核通过或完成修订的评估结果才写入 `assessment_results`；审核阻断或不可用时不会生成 PDF 报告。审核仍依赖模型判断复杂语义；`passed` 不等于现场核验或人工审批。

联网审核区分搜索摘要和已存档正文。后端核对正文 artifact 的会话归属、来源和校验和，并给审核模型提供有长度限制的正文片段，优先保留与问题相关的后续段落。要求打开链接核实、查看正文，或询问实时预警/灾情通报时，只有搜索摘要会触发 `web_body_missing` 阻断；抓取了其他链接而没有抓取要求核实的链接，也会阻断。存档 ID、正文校验和及抓取时间会保留在审核证据中，便于回查。抓取和搜索工具提供的是来源证据，不作为灾害检测模型的评估结果入库。

默认由主模型执行二次审核。如需使用独立的多模态审核模型，可在 `.env` 中配置 `REVIEW_MODEL_NAME`、`REVIEW_MODEL_URL` 和 `REVIEW_MODEL_API_KEY`；审核模型需要兼容 OpenAI Chat Completions 接口并支持图片输入。未设置这些变量时沿用主模型。

主要接口：

| 接口 | 作用 |
| --- | --- |
| `GET /api/health` | 服务、数据库及启用工具状态 |
| `POST /api/auth/register` | 注册并登录；JSON：`username`、`password`、可选 `display_name` |
| `POST /api/auth/login` | 登录；JSON：`username`、`password` |
| `GET /api/auth/me` | 当前登录账号 |
| `PATCH /api/auth/me` | 修改昵称；JSON：`display_name` |
| `POST /api/auth/password` | 修改密码并撤销所有登录；JSON：`current_password`、`new_password` |
| `POST /api/auth/logout` | 退出当前设备 |
| `POST /api/chat/stream` | 流式对话；使用 `multipart/form-data`，包含 UUID 格式的 `session_id`、`message` 和可选的 `files` |
| `POST /api/chat` | 非流式对话，字段同上 |
| `GET /api/sessions` | 当前用户的最近会话 |
| `GET /api/sessions/{session_id}/messages` | 会话消息 |
| `GET /api/sessions/{session_id}/turns` | 各轮的 `running`、`completed` 或 `failed` 状态 |
| `GET /api/sessions/{session_id}/assessments` | 评估结果及产物链接 |
| `GET /api/files/{file_id}` | 下载或预览产物 |

除健康检查、注册和登录外，业务数据接口需要登录 Cookie。无登录或已过期返回 `401`，访问不属于自己的会话或文件返回 `404`。用 curl 调试时，在登录请求中用 `-c cookies.txt` 保存 Cookie，并在后续请求中用 `-b cookies.txt` 携带 Cookie。

## 数据与迁移

会话、消息、轮次状态、评估结果和产物元数据保存在 PostgreSQL；上传文件与生成文件保存在 `data/artifacts/`。部署时需要同时持久化数据库卷和该文件目录。后端可从数据库恢复历史会话；失败轮次会记录失败状态。实现细节见 [数据库与产物持久化](docs/数据库持久化.md)。

重启后端会自动应用 `005_auth.sql`，增加账号、登录令牌、限流表和会话所有者字段。旧匿名会话会保留，但不会出现在任何新账号下。管理员确认旧记录归属后，可先预览，再分配给已注册的账号；`--session-id` 可重复指定，省略时选取全部未分配的旧会话：

```bash
python scripts/assign_legacy_sessions.py --username your-account --session-id UUID
python scripts/assign_legacy_sessions.py --username your-account --session-id UUID --apply
```

这个操作只设置原会话的所有者，其消息、文件、报告和评估结果会一起变为该用户可访问，不会改动已有所有者的会话。

已有旧数据需要迁移文件引用时，先预览，再执行：

```bash
python scripts/migrate_legacy_artifacts.py
python scripts/migrate_legacy_artifacts.py --apply
```

新数据库无需执行此脚本。旧记录若只保留了已失效的文件路径，无法凭路径恢复文件内容。

## 验证与更多文档

```bash
python -m unittest discover -s tests -p 'test_*.py'
AUTH_TEST_DATABASE=1 python -m unittest discover -s tests -p 'test_auth_database.py' -v
cd frontend/chatDisaster && npm run build
```

数据库认证测试会创建临时独立 schema，验证迁移、账号唯一性、令牌失效、密码变更、限流和用户隔离，完成后清理；默认测试不需要连接数据库。对话流程的扩展方式见 [后端对话执行架构](docs/后端执行架构.md)，工具选择依据见 [Agent 工具接入筛选](docs/工具接入筛选.md)。
