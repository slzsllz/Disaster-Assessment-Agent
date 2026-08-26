# Disaster-Assessment-Agent Docker 镜像部署包制作与使用说明

> 本文档介绍如何将本项目打包为 Docker 镜像部署包，并在目标服务器上一键部署运行（含数据库、后端、前端）。

---

## 1. 部署包内容

标准部署包目录结构建议如下：

```text
disaster-assessment-agent-deploy/
├── docker-compose.yml          # 全栈编排文件
├── .env                        # 环境变量配置
├── images/                     # 离线镜像 tar 包
│   ├── disaster-agent-backend.tar
│   ├── disaster-agent-frontend.tar
│   └── postgis-16-3.4.tar      # 可选，如目标机无法联网拉取
├── initdb/                     # 数据库初始化脚本
│   ├── 01_schema.sql
│   ├── 02_add_legend_column.sql
│   └── 03_add_report_files_column.sql
├── model/                      # 模型权重目录（体积大，通常单独挂载）
│   ├── Xview2_Strong_Baseline/
│   ├── sen1floods11-segmentation/
│   ├── FLOGA-main/
│   ├── landslide_l4s/
│   ├── oilspill_cbdnet/
│   ├── pest_yolo/
│   ├── algalbloom_ndci/
│   └── GeoAIModels/
└── README.md                   # 快速启动说明
```

---

## 2. 环境准备

### 2.1 构建机 / 目标机要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux x86_64（推荐 Ubuntu 22.04+） |
| Docker | 20.10+ |
| Docker Compose | v2+ |
| 内存 | 建议 ≥ 16 GB |
| 磁盘 | 后端镜像 + 模型权重 ≥ 50 GB 可用空间 |
| GPU | 非必须；CUDA 支持需额外配置 nvidia-docker |

### 2.2 准备模型权重

将预训练模型文件放置到 `model/` 目录，结构如下：

```text
model/
├── Xview2_Strong_Baseline/      # 地震建筑损伤评估
├── sen1floods11-segmentation/   # 洪水淹没提取
├── FLOGA-main/                  # 野火过火面积变化
├── landslide_l4s/               # 滑坡分割
├── oilspill_cbdnet/             # 海上溢油检测
├── pest_yolo/                   # 作物病虫害检测
├── algalbloom_ndci/             # 藻华候选检测
└── GeoAIModels/                 # 通用 GeoAI 模型
```

> 模型权重体积较大，不建议打包进 Docker 镜像，应通过挂载方式加载。

### 2.3 配置环境变量

复制模板并填写实际值：

```bash
cp docker/.env .env
```

编辑 `.env`：

```ini
# 模型配置：使用 config_deploy.json，通过环境变量传入 API 密钥
AGENT_CONFIG=config_deploy.json

# LLM API（默认 Qwen / 阿里云 DashScope）
QWEN_API_KEY=your_qwen_api_key_here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL_NAME=qwen3.7-plus

# ArcGIS API Key（地理上下文查询 + 前端地图）
ARCGIS_API_KEY=your_arcgis_api_key_here

# 数据库
DB_PORT=5433
DB_NAME=disaster_agent
DB_USER=disaster_agent
DB_PASSWORD=disaster_agent_pwd

# 服务端口
FRONTEND_PORT=80
BACKEND_PORT=8000

# 模型权重挂载路径（绝对路径）
MODEL_DIR=/opt/disaster-models
```

---

## 3. 构建 Docker 镜像

### 3.1 构建前端生产包

如果 `frontend/chatDisaster/dist/` 不存在或已过期，先重新构建：

```bash
cd frontend/chatDisaster
npm install
npm run build
cd ../..
```

构建产物将输出到 `frontend/chatDisaster/dist/`。

### 3.2 构建后端镜像

```bash
docker build -f docker/Dockerfile.backend -t disaster-agent-backend:latest .
```

该镜像基于 `continuumio/miniconda3`，使用 `docker/environment.yml` 安装 GDAL、PyTorch、LangChain 等全部 Python 依赖。首次构建时间较长（约 20–60 分钟），请确保网络畅通。

### 3.3 构建前端镜像

```bash
docker build -f docker/Dockerfile.frontend -t disaster-agent-frontend:latest .
```

### 3.4 拉取数据库镜像（如目标机无法联网，需提前下载）

```bash
docker pull postgis/postgis:16-3.4
```

---

## 4. 导出镜像部署包

### 4.1 导出镜像为 tar

```bash
mkdir -p deploy/images

docker save -o deploy/images/disaster-agent-backend.tar disaster-agent-backend:latest
docker save -o deploy/images/disaster-agent-frontend.tar disaster-agent-frontend:latest
docker save -o deploy/images/postgis-16-3.4.tar postgis/postgis:16-3.4
```

### 4.2 复制编排文件与初始化脚本

```bash
cp docker/docker-compose.full.yml deploy/docker-compose.yml
cp docker/.env deploy/.env
cp -r docker/initdb deploy/initdb
cp DOCKER_DEPLOY.md deploy/README.md
```

### 4.3 复制模型权重（如与目标机同机部署可跳过，直接挂载原目录）

```bash
cp -r /path/to/model deploy/model
```

### 4.4 打包压缩

```bash
cd deploy
tar czvf ../disaster-assessment-agent-deploy.tar.gz .
cd ..
```

> 提示：模型权重通常单独传输，不与代码镜像一起压缩，便于版本管理。

---

## 5. 目标机部署

### 5.1 解压部署包

```bash
tar xzvf disaster-assessment-agent-deploy.tar.gz -C /opt/disaster-assessment-agent
cd /opt/disaster-assessment-agent
```

### 5.2 加载离线镜像

```bash
docker load -i images/disaster-agent-backend.tar
docker load -i images/disaster-agent-frontend.tar
docker load -i images/postgis-16-3.4.tar
```

### 5.3 修改环境变量

```bash
vim .env
```

确保以下关键变量正确：

- `QWEN_API_KEY` / `QWEN_BASE_URL`
- `ARCGIS_API_KEY`
- `MODEL_DIR`（指向实际模型权重目录）
- `FRONTEND_PORT`、`BACKEND_PORT`、`DB_PORT`

### 5.4 启动服务

```bash
docker compose up -d
```

### 5.5 查看状态

```bash
docker compose ps
docker compose logs -f backend
```

---

## 6. 验证部署

### 6.1 健康检查

```bash
curl http://localhost:8000/api/health
```

预期响应：

```json
{
  "ok": true,
  "model": "qwen3.7-plus",
  "tools": ["GeoAI", "Index", "FloodSegmentation", "DamageAssessment", "..."],
  "db_ok": true
}
```

### 6.2 前端访问

浏览器打开：

```text
http://<服务器IP>
```

### 6.3 测试聊天接口

```bash
curl -X POST http://localhost:8000/api/chat \
  -F "session_id=$(uuidgen)" \
  -F "message=请做一次健康检查"
```

---

## 7. 常用运维命令

| 操作 | 命令 |
|------|------|
| 查看日志 | `docker compose logs -f backend` |
| 重启服务 | `docker compose restart` |
| 停止服务 | `docker compose down` |
| 停止并删除数据卷（危险） | `docker compose down -v` |
| 进入后端容器 | `docker exec -it disaster_agent_backend bash` |
| 进入数据库 | `docker exec -it disaster_agent_db psql -U disaster_agent -d disaster_agent` |

---

## 8. 更新部署

### 8.1 更新代码后重新构建

```bash
docker build -f docker/Dockerfile.backend -t disaster-agent-backend:latest .
docker build -f docker/Dockerfile.frontend -t disaster-agent-frontend:latest .
docker compose up -d --build
```

### 8.2 仅更新模型权重

替换 `MODEL_DIR` 指向的模型文件后重启后端：

```bash
docker compose restart backend
```

---

## 9. 注意事项

1. **模型权重不外置于镜像**  
   模型文件体积大且更新频繁，通过 `volumes` 挂载到 `/app/model`，避免重复构建镜像。

2. **数据库数据持久化**  
   PostgreSQL 数据保存在 Docker 卷 `pg_data` 中。除非明确需要清空数据，否则不要使用 `docker compose down -v`。

3. **大文件上传**  
   遥感影像通常较大，若通过 Nginx 反向代理上传失败，请调整 `client_max_body_size`（已在 `nginx.conf` 中默认开启）。

4. **推理超时**  
   后端默认 `max_execution_time=600` 秒，Nginx 也配置了 600 秒读取超时。如模型推理更慢，请同步调整。

5. **联网依赖**  
   后端运行时需要访问 LLM API（Qwen/DashScope 或 DeepSeek），请确保目标服务器能访问对应地址。

6. **临时文件清理**  
   后端 `/app/tmp` 用于存放上传文件和推理输出，已挂载为命名卷。长期运行后可能占用较多磁盘空间，建议定期清理或设置 cron 任务。

---

## 10. 故障排查

### 10.1 后端无法连接数据库

检查后端日志：

```bash
docker compose logs backend | grep -i "database\|psycopg"
```

确认 `.env` 中 `DB_HOST=db`，且数据库服务健康：

```bash
docker compose ps db
```

### 10.2 MCP 工具启动失败

进入后端容器检查工具脚本是否存在：

```bash
docker exec -it disaster_agent_backend ls -la /app/agent/tools/
```

### 10.3 模型加载失败

检查模型挂载路径是否正确：

```bash
docker exec -it disaster_agent_backend ls -la /app/model/
```

### 10.4 前端无法访问后端

浏览器 F12 查看网络请求，确认 `/api/health` 是否返回 200。若 502/504，检查 backend 容器是否正常运行。
