import chinaBoundary from './chinaBoundary.js'

export function mapFeatures(value) {
  const source = typeof value === 'string' ? JSON.parse(value) : value
  if (!source) return []
  return source.type === 'FeatureCollection' ? source.features || []
    : source.type === 'Feature' ? [source] : [{ geometry: source }]
}

export const validPoint = p => Array.isArray(p) && p.length >= 2 &&
  Number.isFinite(p[0]) && Number.isFinite(p[1]) && Math.abs(p[0]) <= 180 && Math.abs(p[1]) <= 90

function drawableGeometry(geometry) {
  const { type, coordinates: c } = geometry || {}
  const line = (v, minimum) => Array.isArray(v) && v.length >= minimum && v.every(validPoint)
  const polygon = v => Array.isArray(v) && v.length > 0 && v.every(ring => line(ring, 3))
  if (type === 'Point') return validPoint(c)
  if (type === 'LineString') return line(c, 2)
  if (type === 'Polygon') return polygon(c)
  if (type === 'MultiLineString') return Array.isArray(c) && c.length > 0 && c.every(v => line(v, 2))
  if (type === 'MultiPolygon') return Array.isArray(c) && c.length > 0 && c.every(polygon)
  return false
}

export function geometryPoints(geometry) {
  const points = []
  function visit(value) {
    if (validPoint(value)) points.push(value.slice(0, 2))
    else if (Array.isArray(value)) value.forEach(visit)
  }
  visit(geometry?.coordinates)
  return points
}

function inRing([x, y], ring) {
  let inside = false
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [xi, yi] = ring[i], [xj, yj] = ring[j]
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

const regions = chinaBoundary.map(rings => {
  const xs = rings[0].map(p => p[0]), ys = rings[0].map(p => p[1])
  return { rings, bounds: [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)] }
})

// Country polygons, rather than a bounding rectangle that also contains nearby countries.
export function isDomesticPoint(point) {
  if (!validPoint(point)) return false
  const [x, y] = point
  return regions.some(({ rings, bounds: [west, south, east, north] }) =>
    x >= west && x <= east && y >= south && y <= north && inRing(point, rings[0]) &&
    !rings.slice(1).some(hole => inRing(point, hole)))
}

export function selectMapProvider(value) {
  const points = mapFeatures(value).filter(f => drawableGeometry(f.geometry)).flatMap(feature => geometryPoints(feature.geometry))
  return !points.length || points.every(isDomesticPoint) ? 'amap' : 'arcgis'
}

// Approximation for legacy WGS84 artifacts. New routes carry native GCJ02 geometry.
export function wgs84ToGcj02([lon, lat]) {
  if (lon < 72.004 || lon > 137.8347 || lat < .8293 || lat > 55.8271) return [lon, lat]
  const x = lon - 105, y = lat - 35, pi = Math.PI
  const common = (20 * Math.sin(6 * x * pi) + 20 * Math.sin(2 * x * pi)) * 2 / 3
  let north = -100 + 2*x + 3*y + .2*y*y + .1*x*y + .2*Math.sqrt(Math.abs(x)) + common
  let east = 300 + x + 2*y + .1*x*x + .1*x*y + .1*Math.sqrt(Math.abs(x)) + common
  for (const [a, d] of [[20,1],[40,3],[160,12],[320,30]]) north += a * Math.sin(y*pi/d) * 2/3
  for (const [a, d] of [[20,1],[40,3],[150,12],[300,30]]) east += a * Math.sin(x*pi/d) * 2/3
  const rad = lat*pi/180, eccentricity = .006693421622965943, ellipsoid = 6378245
  const magic = 1 - eccentricity * Math.sin(rad)**2
  return [lon + east*180/((ellipsoid/Math.sqrt(magic))*Math.cos(rad)*pi),
    lat + north*180/((ellipsoid*(1-eccentricity)/magic**1.5)*pi)]
}

export function amapFeatures(value) {
  function convert(coordinates) {
    return validPoint(coordinates) ? wgs84ToGcj02(coordinates) : coordinates.map(convert)
  }
  return mapFeatures(value).filter(f => drawableGeometry(f.geometry)).map(feature => {
    const native = feature.geometry_gcj02
    const geometry = native?.type === feature.geometry.type && drawableGeometry(native)
      ? native : { ...feature.geometry, coordinates: convert(feature.geometry.coordinates) }
    return { ...feature, geometry, properties: feature.properties || {} }
  })
}

export function worldPixel([lon, lat], zoom) {
  const size = 256 * 2**zoom, rad = Math.max(-85.05112878, Math.min(85.05112878, lat)) * Math.PI / 180
  return [(lon + 180) / 360 * size, (1 - Math.log(Math.tan(rad) + 1 / Math.cos(rad)) / Math.PI) / 2 * size]
}

export function pixelLocation([x, y], zoom) {
  const size = 256 * 2**zoom
  return [((x / size * 360) % 360 + 360) % 360 - 180,
    Math.max(-85, Math.min(85, Math.atan(Math.sinh(Math.PI * (1 - 2*y/size))) * 180 / Math.PI))]
}

export function fitMap(points, width, height) {
  if (!points.length) return { center: [104.1954, 35.8617], zoom: 4 }
  let west = Infinity, east = -Infinity, north = Infinity, south = -Infinity
  for (const p of points) {
    const [x, y] = worldPixel(p, 0)
    west = Math.min(west, x); east = Math.max(east, x)
    north = Math.min(north, y); south = Math.max(south, y)
  }
  const scale = Math.min(Math.max(32, width-96)/Math.max(.001, east-west), Math.max(32, height-96)/Math.max(.001, south-north))
  return { center: pixelLocation([(west+east)/2, (north+south)/2], 0),
    zoom: Math.max(1, Math.min(points.length === 1 ? 14 : 17, Math.floor(Math.log2(scale)))) }
}
