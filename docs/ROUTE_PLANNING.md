# 高德救援路线规划

MCP server：`RoutePlanning`；工具函数：`plan_emergency_routes`。
服务端入口 `agent/tools/RoutePlanning.py`，坐标/文件处理 `agent/route_geometry.py`。
已注册到三份 Agent 配置，支持在聊天中直接说明灾害、提供坐标或上传地理数据后规划。

本次实现前已在 `SpringBreeze` 执行 `git pull --ff-only`，远端返回 Already up to date。

## 配置

项目根目录 `.env`：

```dotenv
AMAP_WEB_KEY=your-amap-web-service-key
```

本机已将提供的 Key 配入 Git 忽略的 `.env`。后端 MCP 子进程白名单和 Docker Compose
均已加入该变量。密钥不作为工具参数暴露，也不写入结果和导航链接。
依赖复用项目环境的 `httpx`、`python-dotenv`、`fastmcp`、`shapely`、`pyproj`、`rasterio`。
修改配置后重启后端并新建对话，以加载新增 MCP 工具。

## 在聊天中使用

- “某处发生洪涝，GPS 坐标为东经 116.4、北纬 39.9，规划到附近医院和应急避难场所的驾车路线。”
- “这个 GeoJSON 是受灾区，请用第 0 个要素作为起点，规划到医院的路线。”
- “高德坐标 116.4,39.9，找附近消防站，按驾车距离推荐路线。”

工具不根据一句地址或无地理信息的图片猜坐标。经纬度顺序始终为 `[经度,纬度]`。
消防站类别表示**从灾害位置到消防站**；如需消防站到灾区的出警路线，不能将本结果反向使用，
因为单行线/通行条件可能不同，当前工具需要另行提供明确起终点的扩展。

## 输入

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `origin` | 无 | 灾害位置 `[longitude, latitude]` |
| `disaster_path` | 无 | 地理文件路径；与 origin 二选一 |
| `coord_system` | `wgs84` | GPS/普通 GeoJSON 为 wgs84；明确来自高德的坐标用 gcj02 |
| `destination_types` | `["hospital","shelter"]` | 可选 hospital、shelter、fire_station |
| `search_radius_m` | 10000 | 周边检索半径，100–50000 米 |
| `max_candidates` | 3 | 每类最多比较最近的 1–5 个 POI |
| `rank_by` | `duration` | duration 按驾车秒数，distance 按驾车米数排序 |
| `feature_index` | 无 | 多要素/多行时必须明确选择，索引从 0 开始 |
| `avoid_polygons` | 无 | 已知不可通行区域，坐标系与 coord_system 相同 |
| `output_dir` | `route_planning` | 工具临时目录下的相对输出目录 |

文件格式：

- GeoJSON Point：直接使用点位；FeatureCollection 多个要素时必须提供 feature_index。
- Polygon/MultiPolygon：使用选中区域内部代表点，明确标记为推断位置。
- JSON：`{"longitude":116.4,"latitude":39.9}`，也接受 lon/lat、lng/lat；记录数组必须选择索引。
- CSV：longitude/latitude、lon/lat 或 lng/lat 表头；多行必须选择索引。
- GeoTIFF：必须声明 CRS，计算图像范围中心并转为 WGS84；它不是检测出的实际灾情点。
- 不接受无 CRS 的栅格或未声明坐标系的投影坐标。

避让区域为 `[[[lon,lat],...],...]`；最多 32 个，每个 3–16 个顶点且面积不超过 81 平方公里。
起点在避让区域内会报错，需提供灾区外可达的接驳点。返回路径若仍穿过已知避让区则被剔除。
灾害范围**不会自动等同于封路区域**；已知阻断区由使用者另行提供。

## 计算过程与返回

1. 输入 GPS 坐标通过高德官方坐标转换 API 转为 GCJ02；高德坐标直接使用。
2. 周边 POI 按直线距离检索前 25 条并筛选候选：医院使用综合医院分类 090100，额外排除社区卫生站、诊所、口腔/美容/体检等名称；避难场所和消防站使用关键词与名称筛选。名称筛选不等同于临床接诊能力确认。
3. 对每个候选请求普通驾车路线（策略 10，不走轮渡），比较实际路网距离/预计耗时。
4. 推荐**已评估候选中**耗时最短或驾车距离最短的路线，保留其余候选与失败原因。

这不保证找到全市/全路网最优设施，也不代表 POI 实际具有急诊接收能力或已开放。

返回 `origin/search/destinations/errors/avoidance/coordinate_systems/outputs/summary_path`。
每类包含 `recommended.facility` 和 `recommended.route`：名称、地址、电话、驾车距离、
预计耗时、分步指示、GCJ02 路线点串、高德导航链接。
`status` 为 success、partial、no_route 或 error；不伪造无结果/失败的路线。

每次独立保存 `summary.json` 和 `routes.geojson`，避免多次规划互相覆盖。
GeoJSON 使用公开 GCJ 模型的近似数值逆变换得到 WGS84，仅用于地图展示；
原始导航坐标保持高德返回的 GCJ02。聊天结果自动打开现有地图显示起点、路线和设施点：
蓝色医院、绿色避难场所、橙色消防站。底图仍使用项目现有 ArcGIS 配置。

## CLI

```bash
python agent/tools/RoutePlanning.py \
  --origin 116.4,39.9 --coord-system wgs84 \
  --destinations hospital shelter --max-candidates 3 \
  --temp_dir tmp/routes_demo

python agent/tools/RoutePlanning.py \
  --data /path/to/disaster.geojson --feature-index 0 \
  --destinations hospital --rank-by distance
```

不带起点/文件参数时启动 stdio MCP server。

## 使用边界

这是普通小客车道路规划，不是消防车/救护车调度或灾后道路通行验证系统。
未核实消防车尺寸/载重限制、灾损桥梁、积水深度、实时封路、医院床位和避难场所开放状态。
出发前应由现场指挥核实道路和设施。导航链接打开后会重新算路，不能保留本次避让区域。
需要复核本次结果时使用保存的路线和指示。

## 验证

```bash
python -m unittest discover -s tests -p test_route_planning.py -v
cd frontend/chatDisaster
node --test tests/mapGeometry.test.js
npm run build
```

测试覆盖实际 HTTP 请求结构、排序、GeoJSON/CSV/投影栅格、无结果、失败处理、
密钥不进入结果、MCP 调用、避让约束及地图线/点绘制。

接口依据：
[周边检索](https://developer.amap.com/api/webservice/guide/api/search/)、
[驾车路径规划](https://developer.amap.com/api/webservice/guide/api/direction)、
[坐标转换](https://developer.amap.com/api/webservice/guide/api/convert)、
[导航 URI](https://developer.amap.com/api/uri-api/guide/travel/route)。
近似显示坐标模型参考 [coordtransform](https://github.com/wandergis/coordtransform)，非官方逆变换或测绘精度保证。
