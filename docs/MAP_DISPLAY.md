# 国内外地图展示

聊天产生空间结果后自动打开地图。国内地点（含港澳台）使用高德地图；国外或混合跨境数据
继续使用原有 ArcGIS，标准/卫星图层切换仍可用。手动打开空地图时默认显示中国范围的高德底图。

高德展示复用根目录 `.env` 的 `AMAP_WEB_KEY`（Web 服务 Key），无需把密钥写入前端。
`GET /api/maps/amap/static` 代理高德官方静态底图 API，校验坐标、尺寸、缩放级别，限速并缓存
最多 16 张图片。Key、上游 URL 和原始错误不返回浏览器。需为账号开通静态地图服务权限/配额。
部署变更后重启后端并刷新前端；Docker/生产环境还需重新构建前端。

`AMapDisplay.vue` 使用底图图片配合 SVG 叠加位置、路线和灾害面，支持拖动后平移、滚轮及按钮缩放、
“定位全部”和失败重试。缩放级别 1–17；超出 1024 像素的容器按比例请求并显示图片。
底图保留高德返回的标识。国内视图目前提供标准底图；国外卫星底图由 ArcGIS 提供。
后台只在用户打开地图或收到空间结果时请求底图，不定时刷新。

高德返回的路线通过 Feature 外部成员 `geometry_gcj02` 保留原始坐标，供高德底图直接叠加；
标准 `geometry` 继续保存近似 WGS84，兼容 ArcGIS 和现有数据提取。
旧 GeoJSON 和其他 WGS84 灾害几何使用近似 GCJ 正变换，不能作为测绘级定位依据。
国内外选择依据标准 WGS84 几何，不依赖工具名、模型文字或一个覆盖邻国的矩形。

### 静态地图像素标定

2026-10-23 修复了“路线穿建筑”的显示问题：SVG 先前使用 `256 × 2^zoom` 的世界像素宽度，
而当前高德静态图 `scale=1` 与原生 `paths` 绘制使用 `512 × 2^zoom`，导致叠加图形相对底图
缩小一半。投影、逆投影、自动缩放范围与拖动必须使用相同的 512 基准；ArcGIS 仍由 SDK 处理。

用用户深圳大学附近的真实 53 点路线对比高德静态图原生折线，缩放 15/16/17 级均确认该基准，
省略 scale 参数也相同。原错误绘制在 16 级的中心线像素偏差中位数约 42、95 分位约 87；
修复后中位数为 0、95 分位不超过 1 像素（线宽、取整和抗锯齿容差）。
测试固定了 6 个从原生折线实际观测到的控制点，避免仅做投影/逆投影自洽测试而漏掉比例错误。
本次没有改动高德返回的路线点、强行插点或移动起点；实际通行与出入口仍以现场道路及定位为准。

选择底图用的离线多边形来自 [Natural Earth 1:10m 数据](https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_10m_admin_0_countries.geojson)，
提取 CHN/TWN/HKG/MAC 几何，按 0.001 度保拓扑简化并保留六位小数，保存在 `src/utils/chinaBoundary.js`。
这是概略地图范围数据，仅用于界面选择底图，边境、细小离岛、海上位置可能存在误差；
不用于行政界线认定。[Natural Earth 数据属于公有领域](https://www.naturalearthdata.com/about/terms-of-use/)。

高德接口参数依据 [官方静态地图文档](https://developer.amap.com/api/webservice/guide/api/staticmaps)。
这是 Web 服务 API，不将 Web Key 用作需要独立 Key/安全配置的 JS SDK。

验证命令：

```bash
python -m unittest discover -s tests -p test_amap_display.py -v
python -m unittest discover -s tests -p test_route_planning.py -v
cd frontend/chatDisaster
node tests/mapProvider.test.js
node tests/mapGeometry.test.js
npm run build
```
