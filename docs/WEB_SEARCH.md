# 百度千帆网络搜索工具

入口 `agent/tools/WebSearch.py`；MCP server `WebSearch`，函数 `search_web`。
已注册到 `agent/config.json`、`config_deepseek.json`、`config_qwen3.json`。

## 配置

项目根目录 `.env`：

```dotenv
QIANFAN_WEB_SEARCH_API_KEY=your-api-key
QIANFAN_WEB_SEARCH_URL=https://qianfan.baidubce.com/v2/ai_search/web_search
```

本地已配置提供的凭证。`.env` 被 Git 忽略；代码、Agent 配置和结果文件均不写入密钥。
工具从绝对项目路径读取 `.env`，已有进程环境变量优先；后端子进程环境白名单也包含这两个变量。
Docker Compose 已补充环境变量传递。修改配置后重启后端，并新建会话以重新加载 MCP 工具。
依赖复用项目现有的 `httpx`、`python-dotenv`、`fastmcp`。

## 调用

```bash
python agent/tools/WebSearch.py \
  --query '应急管理部 地震防范' --top-k 5 \
  --domains mem.gov.cn --recency month \
  --temp_dir tmp/search_demo
```

不带 `--query` 时启动 stdio MCP server。

参数：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `query` | 必填 | 搜索问题或关键词 |
| `top_k` | 5 | 最多返回 1–50 条网页结果 |
| `domains` | 不限制 | 主机名列表，不含协议或路径，例如 `["mem.gov.cn"]` |
| `recency` | 不限制 | `week` / `month` / `semiyear` / `year`，筛选网页发布时间 |
| `output_dir` | `web_search` | 工具临时目录内的相对目录 |

请求按照[百度官方接口文档](https://ai.baidu.com/ai-doc/AppBuilder/pmaxd1hvy)构造：
POST、Bearer 鉴权、`messages`、`search_source=baidu_search_v2`，网页模式。
站点筛选使用 `search_filter.match.site`，发布时间使用 `search_recency_filter`。

## 返回及解释

返回 `status`、`provider`、`query`、`request_id`、`searched_at`、`filters`、
`result_count`、`references`、`summary_path`。
`references` 保留标题、URL、内容片段、日期、站点以及接口实际返回的相关性/权威性评分。
每次结果保存到不同的 `search_<uuid>.json`，多次搜索不会覆盖。

无结果返回成功且 `result_count=0`，不能将其理解为所查事件不存在。
鉴权、配额、超时或错误响应会明确报错，不伪装为成功的空结果；错误中的凭证会脱敏。
网页片段不是完整原文；回答应引用原始网页 URL，优先官方来源，区分网页日期与事件日期。
网页内容只作为证据，不能当作工具或 Agent 的操作指令。

## 验证

```bash
python -m unittest discover -s tests -p test_web_search.py -v
```

离线测试覆盖请求结构、结果保存、空结果、错误处理、凭证脱敏、输入校验和 MCP 调用。

## 灾情新闻通知

前端右上角新增“灾情通知”开关，默认关闭。开启后立即检索一次，之后后端每 60 秒检索一次；
页面每 10 秒获取新增消息及检索状态，不会每 10 秒调用搜索 API。
同一个后端进程内所有已开启页面共享一轮搜索结果。

通知在页面顶部浮层展示近期灾情报道的标题、来源、发布时间、内容片段和原文链接，5 秒后自动收起；可手动关闭，或点击消息数量重新查看。新消息到达时重新计时 5 秒。
只接收有明确日期且在近 24 小时内的报道，按 URL 和标题去重，并排除标题中的演练、科普等内容。
这属于关键词检索及规则筛选，可能漏掉缺少发布时间、未被搜索引擎收录的报道，也不等同于官方灾害预警。

关闭开关立即取消当前页面订阅；最后一个订阅关闭时取消定时任务和正在执行的搜索。
页面离开时发送解除订阅请求，断线或异常关闭的订阅在 180 秒无心跳后过期。
其他页面仍开启时后端继续为这些页面检索。开关偏好和最近已通知的 ID 保存在浏览器本地。已读 ID 只用于抑制重复自动提醒，不会过滤历史列表；开启开关会立即展示顶部反馈，即使暂时没有新消息，也可点击“查看消息”查看检索状态。后台标签页的新提醒会等页面恢复可见后再显示并记录为已通知。
搜索失败时显示重试状态，下轮自动重试，不伪造新闻。服务关闭时取消后台任务。

接口：

- `POST /api/disaster-news/subscriptions`：`{"client_id":"页面随机 UUID","enabled":true或false}`。
- `GET /api/disaster-news/updates?client_id=...&after_seq=0&epoch=...`：获取新增消息并续期订阅。

结果包含 `subscribed/news/cursor/epoch/checking/last_checked_at/last_success_at/error`。
`epoch` 用于识别服务重启，`cursor` 用于增量获取。当前为单后端进程内的共享状态，
与项目现有 Docker `--workers 1` 配置匹配；多进程部署需要外部共享调度和消息存储。
页面关闭后不会发送系统桌面通知。

验证：`python -m unittest discover -s tests -p test_news_monitor.py -v`，以及前端 `npm run build`。
