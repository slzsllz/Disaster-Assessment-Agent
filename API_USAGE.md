# Disaster-Assessment-Agent 远程调用接口说明文档

> 本文档面向需要远程调用或集成本系统的用户/开发者，介绍服务部署方式、HTTP API 接口定义、请求/响应示例及典型调用流程。

---

## 1. 服务部署

### 1.1 环境要求

- Docker Engine 20.10+
- Docker Compose v2+
- 建议操作系统：Linux / macOS / Windows WSL2

### 1.2 启动 PostgreSQL + PostGIS（数据持久化）

```bash
cd docker
docker compose up -d
```

启动后：

- 容器名：`disaster_agent_db`
- 外部端口：`5433`
- 默认数据库：`disaster_agent`
- 默认用户/密码：`disaster_agent` / `disaster_agent_pwd`

### 1.3 启动后端 API 服务

```bash
# 在项目根目录执行
uvicorn backend_api:app --host 0.0.0.0 --port 8000
```

- API 基地址：`http://<服务器IP>:8000/api`
- 自动接口文档（Swagger UI）：`http://<服务器IP>:8000/docs`
- 备用文档（ReDoc）：`http://<服务器IP>:8000/redoc`

### 1.4 启动前端（可选）

```bash
cd frontend/chatDisaster
npm install
npm run dev
```

前端默认地址：`http://localhost:5173`

---

## 2. 基础约定

### 2.1 基地址

```text
http://<host>:8000/api
```

### 2.2 跨域（CORS）

后端默认允许以下前端源：

- `http://localhost:5173`
- `http://127.0.0.1:5173`

若需远程网页端调用，请修改 `backend_api.py` 中的 `allow_origins` 配置，或统一通过后端反向代理访问。

### 2.3 认证

当前版本未实现独立认证中间件，采用**会话 ID（session_id）**进行状态隔离。远程调用时请自行生成并妥善保存 `session_id`（建议使用 UUID）。

### 2.4 内容类型

- 普通 JSON 请求：`application/json`
- 文件上传请求：`multipart/form-data`

---

## 3. 接口列表

### 3.1 健康检查

**请求**

```http
GET /api/health
```

**响应示例**

```json
{
  "ok": true,
  "model": "qwen3-235b-a22b",
  "tools": ["GeoAI", "Index", "FloodSegmentation", "DamageAssessment", "..."],
  "db_ok": true
}
```

---

### 3.2 创建/发送聊天消息（非流式）

**请求**

```http
POST /api/chat
Content-Type: multipart/form-data
```

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `session_id` | string | 是 | 会话唯一标识，建议 UUID |
| `message` | string | 否 | 用户自然语言指令 |
| `system_prompt` | string | 否 | 系统提示词，默认使用内置提示 |
| `recursion_limit` | int | 否 | Agent 最大递归步数，默认 40 |
| `max_execution_time` | int | 否 | 单次调用最大执行秒数，默认 600 |
| `show_trace` | bool | 否 | 是否在响应中返回工具调用轨迹，默认 false |
| `files` | file[] | 否 | 用户上传的遥感影像等附件，支持多文件 |

**cURL 示例**

```bash
curl -X POST http://localhost:8000/api/chat \
  -F "session_id=550e8400-e29b-41d4-a716-446655440000" \
  -F "message=请分析这张遥感影像中的洪水淹没范围" \
  -F "files=@/path/to/sentinel2_image.tif"
```

**响应示例**

```json
{
  "answer": "影像中检测到约 3.2 km² 的洪水淹没区域，主要分布在河流下游低洼地带...",
  "elapsed": 18.5,
  "tool_calls": 2,
  "images": [
    {"name": "flood_overlay.png", "url": "/api/files/a1b2c3d4e5f6"}
  ],
  "files": [
    {"name": "flood_mask.geojson", "url": "/api/files/b2c3d4e5f6a1"}
  ],
  "geometry": {
    "type": "Polygon",
    "coordinates": [[[...]]]
  },
  "legend": [
    {"label": "洪水", "color": "#0066ff"}
  ],
  "trace": [],
  "report": null
}
```

---

### 3.3 创建/发送聊天消息（流式 SSE）

**请求**

```http
POST /api/chat/stream
Content-Type: multipart/form-data
```

参数与 `/api/chat` 完全一致。

**cURL 示例**

```bash
curl -N -X POST http://localhost:8000/api/chat/stream \
  -F "session_id=550e8400-e29b-41d4-a716-446655440000" \
  -F "message=请分析这张遥感影像中的洪水淹没范围" \
  -F "files=@/path/to/sentinel2_image.tif" \
  -H "Accept: text/event-stream"
```

**SSE 事件类型**

| 事件 | 说明 |
|------|------|
| `status` | 状态通知，例如 "正在思考..."、"正在生成报告..." |
| `delta` | 增量文本，用于实时展示 AI 回复 |
| `done` | 最终完整结果，结构与 `/api/chat` 响应一致 |
| `report` | 当用户要求生成报告时，返回 PDF 报告信息 |
| `error` | 调用异常 |

**SSE 数据示例**

```text
event: status
data: {"message": "正在思考..."}

event: delta
data: {"text": "正在"}

event: delta
data: {"text": "分析影像..."}

event: done
data: {"answer": "...", "elapsed": 18.5, "tool_calls": 2, "images": [...], "files": [...], "geometry": {...}, "legend": [...], "trace": []}
```

**前端 JavaScript 示例**

```javascript
const form = new FormData();
form.append("session_id", "550e8400-e29b-41d4-a716-446655440000");
form.append("message", "请分析这张遥感影像");
form.append("files", fileInput.files[0]);

const evtSource = new EventSource("");
const response = await fetch("http://localhost:8000/api/chat/stream", {
  method: "POST",
  body: form,
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  const parts = buffer.split("\n\n");
  buffer = parts.pop();
  for (const part of parts) {
    const lines = part.split("\n");
    const event = lines[0].replace("event: ", "");
    const data = JSON.parse(lines[1].replace("data: ", ""));
    console.log(event, data);
  }
}
```

---

### 3.4 获取会话列表

**请求**

```http
GET /api/sessions?limit=30
```

**响应示例**

```json
{
  "sessions": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "title": "洪水淹没范围分析",
      "model_name": "qwen3-235b-a22b",
      "message_count": 4,
      "created_at": "2026-07-24T08:30:00",
      "updated_at": "2026-07-24T08:35:00",
      "first_message": "请分析这张遥感影像中的洪水淹没范围"
    }
  ]
}
```

---

### 3.5 获取会话消息历史

**请求**

```http
GET /api/sessions/{session_id}/messages
```

**响应示例**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "messages": [
    {
      "id": 1,
      "role": "user",
      "content": "请分析这张遥感影像中的洪水淹没范围",
      "attachments": [{"name": "sentinel2_image.tif", "path": "..."}],
      "images": [],
      "legend": [],
      "tool_trace": [],
      "elapsed_seconds": null,
      "tool_call_count": null,
      "created_at": "2026-07-24T08:30:00",
      "report": null
    },
    {
      "id": 2,
      "role": "assistant",
      "content": "影像中检测到约 3.2 km² 的洪水淹没区域...",
      "attachments": [],
      "images": [{"name": "flood_overlay.png", "url": "/api/files/..."}],
      "legend": [{"label": "洪水", "color": "#0066ff"}],
      "tool_trace": [...],
      "elapsed_seconds": 18.5,
      "tool_call_count": 2,
      "created_at": "2026-07-24T08:30:20",
      "report": null
    }
  ]
}
```

---

### 3.6 获取会话评估结果

**请求**

```http
GET /api/sessions/{session_id}/assessments
```

**响应示例**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "assessments": [
    {
      "id": 1,
      "session_id": "550e8400-e29b-41d4-a716-446655440000",
      "task": "FloodSegmentation",
      "description": "洪水淹没范围",
      "raster_path": "...",
      "geojson_path": "...",
      "overlay_path": "...",
      "summary_path": "...",
      "summary": {"area_km2": 3.2, "pixel_count": 12580},
      "num_objects": 1,
      "geom": {"type": "Polygon", "coordinates": [[[...]]]},
      "created_at": "2026-07-24T08:30:20",
      "overlay_url": "/api/files/..."
    }
  ]
}
```

---

### 3.7 获取会话最新空间范围

**请求**

```http
GET /api/sessions/{session_id}/latest-geometry
```

**响应示例**

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "found": true,
  "assessment": {...},
  "geom": {
    "type": "Polygon",
    "coordinates": [[[...]]]
  }
}
```

---

### 3.8 获取全局评估结果

**请求**

```http
GET /api/assessments?task=FloodSegmentation&limit=50
```

**响应示例**

```json
{
  "assessments": [...]
}
```

---

### 3.9 清空会话聊天记录

**请求**

```http
POST /api/sessions/{session_id}/clear
```

**响应示例**

```json
{"ok": true}
```

---

### 3.10 删除会话

**请求**

```http
DELETE /api/sessions/{session_id}
```

**响应示例**

```json
{"ok": true}
```

---

### 3.11 获取文件资源

**请求**

```http
GET /api/files/{file_id}
```

用于下载 `/api/chat` 或 `/api/sessions/{id}/messages` 返回的图片、GeoJSON、PDF 报告等文件。

---

## 4. 典型调用流程

### 4.1 单次评估流程

```text
1. POST /api/chat
   发送用户问题 + 遥感影像文件
   ↓
2. 后端 Agent 自动选择并调用 MCP 工具
   ↓
3. 返回 answer / images / files / geometry
   ↓
4. GET /api/files/{file_id}
   下载结果图片或矢量文件
```

### 4.2 多轮对话流程

```text
1. 固定 session_id
2. 重复调用 POST /api/chat 或 POST /api/chat/stream
3. 每次调用都会自动追加到同一会话历史
4. GET /api/sessions/{session_id}/messages 可查看完整上下文
```

### 4.3 报告生成流程

在 `message` 中明确要求“生成报告”，例如：

```text
请分析这张影像的洪水情况并生成报告
```

响应中的 `report` 字段将包含 PDF 下载信息：

```json
{
  "report": {
    "url": "/api/files/report_abc123.pdf",
    "name": "report.pdf",
    "description": "基于本次对话生成的评估报告"
  }
}
```

---

## 5. 错误处理

非流式接口在异常时返回：

```json
{
  "answer": "后端调用失败：...",
  "elapsed": 0,
  "tool_calls": 0,
  "images": [],
  "files": [],
  "geometry": null,
  "legend": [],
  "trace": [],
  "error": "...详细堆栈...",
  "memory_suggestions": []
}
```

流式接口通过 `event: error` 推送错误信息。

常见错误：

| 状态码 | 原因 |
|--------|------|
| 404 | 文件不存在（`/api/files/{file_id}`） |
| 422 | 请求参数校验失败 |
| 500 | Agent 推理或工具执行异常 |

---

## 6. Python 调用示例

```python
import requests
import uuid

base_url = "http://localhost:8000/api"
session_id = str(uuid.uuid4())

# 1. 健康检查
r = requests.get(f"{base_url}/health")
print(r.json())

# 2. 发送消息并上传文件
with open("/path/to/image.tif", "rb") as f:
    r = requests.post(
        f"{base_url}/chat",
        data={
            "session_id": session_id,
            "message": "请分析这张遥感影像",
        },
        files={"files": ("image.tif", f, "image/tiff")},
    )
result = r.json()
print(result["answer"])

# 3. 下载结果图片
for img in result.get("images", []):
    img_data = requests.get(f"{base_url}{img['url']}").content
    with open(img["name"], "wb") as out:
        out.write(img_data)
```

---

## 7. 注意事项

1. **session_id 务必唯一且稳定**：多轮对话必须复用同一个 `session_id`，否则上下文丢失。
2. **文件大小限制**：上传大影像时请确保后端服务器和反向代理的 `max_request_size` / `client_max_body_size` 足够大。
3. **推理耗时**：灾害评估涉及深度学习推理，单次调用可能需要数十秒到数分钟，建议使用流式接口并设置合理超时。
4. **生产部署**：建议在后端前增加 Nginx 反向代理，并配置 HTTPS、访问控制及文件大小限制。
5. **数据持久化**：PostgreSQL 数据默认保存在 Docker 卷 `docker_pg_data` 中，请勿随意执行 `docker compose down -v`，否则会删除数据。
