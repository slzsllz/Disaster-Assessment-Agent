import test from 'node:test'
import assert from 'node:assert/strict'
import { mapGraphicDefinitions } from '../src/utils/mapGeometry.js'

test('route collection renders incident, two colored routes and destinations', () => {
  const route = (kind, lat) => [
    { type: 'Feature', properties: { destination_type: kind, name: '<img src=x onerror=alert(1)>' }, geometry: { type: 'LineString', coordinates: [[116.4,39.9],[116.41,lat]] } },
    { type: 'Feature', properties: { destination_type: kind, role: 'destination' }, geometry: { type: 'Point', coordinates: [116.41,lat] } },
  ]
  const result = mapGraphicDefinitions({ type: 'FeatureCollection', properties: { task: 'route_planning' }, features: [
    { type: 'Feature', properties: { role: 'incident' }, geometry: { type: 'Point', coordinates: [116.4,39.9] } },
    ...route('hospital',39.91), ...route('shelter',39.92),
  ] })
  assert.equal(result.graphics.length,5)
  assert.equal(result.isRoute,true)
  assert.equal(result.hasRoutes,true)
  assert.equal(result.graphics[1].geometry.type,'polyline')
  assert.notDeepEqual(result.graphics[1].symbol.color,result.graphics[3].symbol.color)
  assert.equal(result.graphics[0].geometry.spatialReference.wkid,4326)
  assert.ok(result.graphics[1].popupTemplate.title.includes('&lt;img'))
})

test('existing disaster polygons still render', () => {
  const result = mapGraphicDefinitions(JSON.stringify({ type:'Polygon',coordinates:[[[116,39],[117,39],[117,40],[116,39]]] }))
  assert.equal(result.graphics.length,1)
  assert.equal(result.graphics[0].geometry.type,'polygon')
  assert.equal(result.hasRoutes,false)
})

test('invalid/projected route positions are not drawn as WGS84', () => {
  const result = mapGraphicDefinitions({ type:'LineString',coordinates:[[1,2],[300000,4000000]] })
  assert.deepEqual(result.graphics,[])
})
