# 地理上下文查询

统一入口仍为 MCP 工具 `query_geo_context`，不新增需要大模型自行选择的供应商工具。

## 服务分工

- 查询中心在国内离线范围内：高德 Web 服务，读取后端 `AMAP_WEB_KEY`。
- 查询中心在国外：保留 ArcGIS 逆地理编码、Places，以及显式配置的 Feature Layers，读取 `ARCGIS_API_KEY`。
- 缺少对应密钥或接口失败时如实返回，不跨供应商兜底。国内查询不发送 ArcGIS 请求，国外查询不发送高德请求。
- 国内不执行 ArcGIS Feature Layers，显式传入时以 `partial` 和局限说明返回，不能声称已获得建筑轮廓或完整路网。

服务选择依据显式 `center`，否则使用 `bbox` 或带 CRS 栅格变换后的 WGS84 范围中心。
离线范围沿用地图已有的 Natural Earth CHN/TWN/HKG/MAC 概略多边形，包含港澳台；
保存在 `agent/china_boundary.json`。测试检查它与前端 `chinaBoundary.js` 一致。
数据仅用于服务选择，不是权威行政边界；边境、沿海、细小离岛和海上地点可能有近似误差。
跨境或大范围数据应拆分查询，不将中心周围的结果描述为全范围覆盖。

## 输入与坐标

输入为 `raster_path`、WGS84 `bbox=[西,南,东,北]` 或 WGS84 `center=[经度,纬度]`。
不得传入未标识的 GCJ02/BD09 坐标；缺少 CRS 的栅格会拒绝定位，不猜测为经纬度。
显式中心须位于所提供的范围内，跨日期变更线的范围须拆分。

国内使用高德官方坐标转换接口将中心和 POI 检索矩形转换为 GCJ02，再调用逆地理编码和矩形 POI 搜索。
返回 POI 同时包含 `location_gcj02` 和 `lon`/`lat`，后者一般是近似 WGS84 逆变换，明确标注 `WGS84_APPROX`，不用于测绘级定位。
若官方转换对查询中心保持原坐标，则不人为引入逆变换偏移，并在结果中注明使用恒等转换。
半径范围为 1–50000 米；POI 查询沿用最多 20 km × 20 km 的采样范围上限。
每类仅取首页 1–20 条，空结果不是“当地不存在设施”的证据。

## 查询内容与状态

支持医院、学校、公园、小区、政府机构、避难场所、消防站、公安机构、商业建筑等类别。
国内 `road` 来自逆地理编码返回的附近道路参考点，半径至多 3000 米，不是完整道路矢量；
地址、行政区归属同样来自逆地理编码，而不是完整行政区边界下载。
`query_types=[]` 只查地址/行政上下文，不逐类搜索 POI。

保留 `location`、`nearby_places`、`intersecting_feature_layers`、`summary` 等原有字段；
新增 `provider`、`provider_selection`、`coordinate_systems` 和 `status`。
`status` 为 `success`、`partial` 或 `error`，区分服务失败、部分成功和正常空结果。
每次查询保存独立摘要 JSON，避免后续国内外查询覆盖前一次产物。

位置和周边设施只支持初步背景研判。附近存在、位于影像矩形内、与灾害掩膜相交、
已经受损或停运，是不同判断，不得混同。完整暴露统计需权威要素与实际灾害范围的空间叠加。

高德密钥仅在服务器发往固定官方接口；请求错误不输出包含 Key 的 URL 或上游原始错误文本。
无需新增前端 Key 或改变地图展示组件。重启后端后，新初始化的 Agent 会读取更新后的工具说明和系统提示词。

## 验证

```bash
python -m unittest discover -s tests -p test_geo_context.py -v
```

测试使用模拟接口和临时 GeoTIFF，不消耗实际配额。
协议依据：[逆地理编码](https://developer.amap.com/api/webservice/guide/api/georegeo/)、
[POI 范围搜索](https://developer.amap.com/api/webservice/guide/api/search/)、
[坐标转换](https://developer.amap.com/api/webservice/guide/api/convert)。
离线数据来源和公有领域说明见 [MAP_DISPLAY.md](MAP_DISPLAY.md)。
