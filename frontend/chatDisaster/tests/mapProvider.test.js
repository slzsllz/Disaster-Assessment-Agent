import test from 'node:test'
import assert from 'node:assert/strict'
import { amapFeatures, fitMap, isDomesticPoint, pixelLocation, selectMapProvider, wgs84ToGcj02, worldPixel } from '../src/utils/mapProvider.js'

test('domestic cities including Hong Kong, Macau and Taipei use AMap', () => {
  for (const point of [[116.4,39.9],[114.0579,22.5431],[114.1694,22.3193],[113.5439,22.1987],[121.5654,25.033]]) {
    assert.ok(isDomesticPoint(point), point.join(','))
    assert.equal(selectMapProvider({type:'Point',coordinates:point}),'amap')
  }
})
test('overseas and neighboring countries inside the coarse rectangle keep ArcGIS', () => {
  for (const point of [[-74,40.7],[-.12,51.5],[139.69,35.69],[126.978,37.5665],[105.8342,21.0278],[102.6331,17.9757],[85.324,27.7172],[106.917,47.918]]) {
    assert.equal(isDomesticPoint(point),false,point.join(','))
    assert.equal(selectMapProvider({type:'Point',coordinates:point}),'arcgis')
  }
})
test('mixed collections keep ArcGIS while old polygon JSON selects AMap', () => {
  assert.equal(selectMapProvider({type:'FeatureCollection',features:[
    {geometry:{type:'Point',coordinates:[116.4,39.9]}}, {geometry:{type:'Point',coordinates:[126.978,37.5665]}},
  ]}),'arcgis')
  assert.equal(selectMapProvider(JSON.stringify({type:'Polygon',coordinates:[[[116,39],[117,39],[117,40],[116,39]]]})),'amap')
})
test('native GCJ02 is preserved exactly; legacy WGS84 uses the known transform', () => {
  const shifted=wgs84ToGcj02([116.404,39.915])
  assert.ok(Math.abs(shifted[0]-116.41024449916938)<1e-8)
  assert.ok(Math.abs(shifted[1]-39.91640428150164)<1e-8)
  const native={type:'LineString',coordinates:[[116.41,39.92],[116.42,39.93]]}
  const feature={type:'Feature',geometry:{type:'LineString',coordinates:[[116.404,39.915],[116.414,39.925]]},geometry_gcj02:native}
  assert.deepEqual(amapFeatures(feature)[0].geometry,native)
  const old={type:'Point',coordinates:[116.404,39.915]}
  assert.deepEqual(amapFeatures(old)[0].geometry.coordinates,shifted)
})
test('Mercator pan and fit keep route points within the image', () => {
  const points=[[116.4,39.9],[116.5,40]]
  const view=fitMap(points,640,480), center=worldPixel(view.center,view.zoom)
  for(const p of points){
    const pixel=worldPixel(p,view.zoom)
    assert.ok(Math.abs(pixel[0]-center[0])<=640/2-40)
    assert.ok(Math.abs(pixel[1]-center[1])<=480/2-40)
    const back=pixelLocation(pixel,view.zoom)
    assert.ok(Math.abs(back[0]-p[0])<1e-9 && Math.abs(back[1]-p[1])<1e-9)
  }
  assert.equal(fitMap([],640,480).zoom,4)
})

test('malformed artifact geometry does not break other valid map features', () => {
  const geometry={type:'Point',coordinates:[116.404,39.915]}
  const result=amapFeatures({type:'FeatureCollection',features:[
    {geometry,geometry_gcj02:{type:'Point',coordinates:[NaN,39]}},
    {geometry:{type:'LineString',coordinates:[[116,39],[300000,4000000]]}},
    {geometry:{type:'Polygon',coordinates:[null]}},
  ]})
  assert.equal(result.length,1)
  assert.deepEqual(result[0].geometry.coordinates,wgs84ToGcj02(geometry.coordinates))
})
