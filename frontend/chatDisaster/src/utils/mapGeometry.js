// GeoJSON -> ArcGIS graphic definitions. Route coordinates are approximate WGS84.
const colors = { hospital: [37, 99, 235, 1], shelter: [22, 163, 74, 1], fire_station: [234, 88, 12, 1] }
const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;' })[c])
const isPoint = point => Array.isArray(point) && point.length >= 2 &&
  Number.isFinite(point[0]) && Number.isFinite(point[1]) && Math.abs(point[0]) <= 180 && Math.abs(point[1]) <= 90

export function mapGraphicDefinitions(value) {
  const source = typeof value === 'string' ? JSON.parse(value) : value
  const graphics = []
  function add(geometry, properties = {}) {
    if (!geometry) return
    const { type, coordinates } = geometry
    const color = colors[properties.destination_type] || [239, 68, 68, 1]
    const base = { spatialReference: { wkid: 4326 } }
    const popup = properties.name ? { title: escape(properties.name), content: '路线坐标为近似 WGS84，仅作展示；请核实现场道路和设施状态。' } : null
    if (type === 'Point' && isPoint(coordinates)) {
      graphics.push({ geometry: { ...base, type: 'point', longitude: coordinates[0], latitude: coordinates[1] },
        symbol: { type: 'simple-marker', color, size: properties.role === 'incident' ? 13 : 10, outline: { color: 'white', width: 1.5 } }, popupTemplate: popup })
    } else if (type === 'LineString' && Array.isArray(coordinates) && coordinates.length >= 2 && coordinates.every(isPoint)) {
      graphics.push({ geometry: { ...base, type: 'polyline', paths: [coordinates] }, symbol: { type: 'simple-line', color, width: 4 }, popupTemplate: popup })
    } else if (type === 'MultiLineString') {
      for (const line of coordinates || []) add({ type: 'LineString', coordinates: line }, properties)
    } else if (type === 'Polygon' && Array.isArray(coordinates) && coordinates.length && coordinates.every(ring => Array.isArray(ring) && ring.length >= 3 && ring.every(isPoint))) {
      graphics.push({ geometry: { ...base, type: 'polygon', rings: coordinates }, symbol: { type: 'simple-fill', color: [239, 68, 68, .18], outline: { color: [239, 68, 68, .95], width: 2 } } })
    } else if (type === 'MultiPolygon') {
      for (const polygon of coordinates || []) add({ type: 'Polygon', coordinates: polygon }, properties)
    }
  }
  if (source?.type === 'FeatureCollection') {
    for (const feature of source.features || []) add(feature.geometry, feature.properties || {})
  } else if (source?.type === 'Feature') add(source.geometry, source.properties || {})
  else add(source)
  return { graphics, isRoute: source?.properties?.task === 'route_planning', hasRoutes: graphics.some(item => item.geometry.type === 'polyline') }
}
