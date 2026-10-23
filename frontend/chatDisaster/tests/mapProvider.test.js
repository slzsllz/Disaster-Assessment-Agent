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

test('AMap projected route matches independently observed native static-map pixels', () => {
  // Official /v3/staticmap paths at location=113.937409,22.527149, zoom=15,
  // size=934*811, scale=1. These are observed blue-line pixels, not a round-trip test.
  const center = worldPixel([113.937409,22.527149],15)
  const reference = [
    { coordinates:[113.938865,22.521864], pixel:[535,672] },
    { coordinates:[113.934225,22.522616], pixel:[319,634] },
    { coordinates:[113.938302,22.529187], pixel:[509,303] },
    { coordinates:[113.940625,22.531118], pixel:[617,205] },
    { coordinates:[113.94055,22.531199], pixel:[613,201] },
    { coordinates:[113.939057,22.532359], pixel:[544,143] },
  ]
  for (const {coordinates,pixel} of reference) {
    const projected=worldPixel(coordinates,15).map((value,i)=>value-center[i]+[467,405.5][i])
    assert.ok(Math.hypot(projected[0]-pixel[0],projected[1]-pixel[1])<2,coordinates.join(','))
  }
})

test('one level zoom doubles AMap pixel offsets and panning preserves the drag distance', () => {
  const center=[113.937409,22.527149],point=[113.940625,22.531118]
  const offset=z=>worldPixel(point,z).map((v,i)=>v-worldPixel(center,z)[i])
  assert.ok(Math.abs(offset(16)[0]-offset(15)[0]*2)<1e-6)
  assert.ok(Math.abs(offset(16)[1]-offset(15)[1]*2)<1e-6)
  const pixels=worldPixel(center,15)
  const after=pixelLocation([pixels[0]-80,pixels[1]-30],15)
  const next=worldPixel(after,15)
  assert.ok(Math.abs(next[0]-pixels[0]+80)<1e-6)
  assert.ok(Math.abs(next[1]-pixels[1]+30)<1e-6)
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
