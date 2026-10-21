"""AMap protocol, coordinate correctness, route ranking, and failure behavior."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
from agent.route_geometry import coordinate, gcj02_to_wgs84_approx, resolve_origin, wgs84_to_gcj02_approx

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('route_planning_test', ROOT / 'agent/tools/RoutePlanning.py')
route = importlib.util.module_from_spec(spec)
spec.loader.exec_module(route)
ORIGIN = [116.4, 39.9]


def poi(index):
    return {'id': f'hospital-{index}', 'name': f'综合医院{index}', 'location': f'{116.4+index*.01:.6f},39.900000',
            'distance': str(index*1000), 'address': '测试地址', 'tel': '010-12345678', 'type': '医疗保健服务;综合医院'}


def driving(destination):
    first = destination.startswith('116.410000')
    return {'status': '1', 'route': {'paths': [{'distance': '2000' if first else '4000',
        'duration': '600' if first else '200', 'strategy': '测试策略', 'steps': [{
            'instruction': '沿测试路向东行驶', 'road': '测试路', 'distance': '2000', 'duration': '200',
            'polyline': f'116.400000,39.900000;{destination}',
        }]}]}}


class GeometryTests(unittest.TestCase):
    def test_invalid_coordinates(self):
        for value in ([91, 181], [float('nan'), 30], [120, True], [120], [500000, 3000000]):
            with self.subTest(value=value), self.assertRaises(ValueError): coordinate(value)

    def test_coordinate_reference_and_inverse(self):
        # Published coordtransform reference; inverse is numerical, not a survey guarantee.
        shifted = wgs84_to_gcj02_approx([116.404, 39.915])
        self.assertAlmostEqual(shifted[0], 116.41024449916938, places=8)
        self.assertAlmostEqual(shifted[1], 39.91640428150164, places=8)
        inverse = gcj02_to_wgs84_approx(shifted)
        self.assertAlmostEqual(inverse[0], 116.404, places=7)
        self.assertAlmostEqual(inverse[1], 39.915, places=7)
        self.assertEqual(gcj02_to_wgs84_approx([-73, 40]), [-73, 40])

    def test_file_inputs_and_explicit_feature_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'incident.geojson'
            p.write_text(json.dumps({'type': 'FeatureCollection', 'features': [
                {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': ORIGIN}},
                {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [117,40]}},
            ]}))
            with self.assertRaisesRegex(ValueError, 'Multiple locations'):
                resolve_origin(None, str(p), 'wgs84')
            point, system, meta = resolve_origin(None, str(p), 'wgs84', 1)
            self.assertEqual(point, [117,40])
            self.assertEqual(system, 'wgs84')
            p.write_text(json.dumps({'type':'Polygon','coordinates':[[[116,39],[117,39],[117,40],[116,40],[116,39]]]}))
            self.assertEqual(resolve_origin(None,str(p),'wgs84')[2]['kind'], 'polygon_representative_point')
            csv = Path(tmp) / 'incident.csv'
            csv.write_text('longitude,latitude\n116.4,39.9\n')
            self.assertEqual(resolve_origin(None,str(csv),'wgs84')[0], ORIGIN)

    def test_projected_raster_and_missing_crs(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'incident.tif'
            profile = dict(driver='GTiff',width=2,height=2,count=1,dtype='uint8',transform=from_origin(12958000,4852000,10,10))
            with rasterio.open(p,'w',crs='EPSG:3857',**profile) as dst: dst.write(np.zeros((1,2,2),np.uint8))
            point, system, meta = resolve_origin(None,str(p),'wgs84')
            self.assertTrue(116 < point[0] < 117 and 39 < point[1] < 40)
            self.assertEqual(meta['kind'], 'raster_extent_center')
            with rasterio.open(p,'w',**profile) as dst: dst.write(np.zeros((1,2,2),np.uint8))
            with self.assertRaisesRegex(ValueError, 'no CRS'): resolve_origin(None,str(p),'wgs84')


class RoutingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        old = route.TEMP_DIR
        route.TEMP_DIR = Path(self.temp.name)
        self.addCleanup(setattr,route,'TEMP_DIR',old)
        env = patch.dict(os.environ, {'AMAP_WEB_KEY':'amap-test-secret'})
        env.start(); self.addCleanup(env.stop)
        self.requests=[]

    def mock(self, handler=None):
        def default(request):
            self.requests.append(request)
            params=request.url.params
            if request.url.path.endswith('/convert'):
                return httpx.Response(200,json={'status':'1','locations':';'.join(params['locations'].split('|'))})
            if request.url.path.endswith('/around'):
                return httpx.Response(200,json={'status':'1','pois':[poi(1),poi(2)]})
            return httpx.Response(200,json=driving(params['destination']))
        original = httpx.AsyncClient
        transport = httpx.MockTransport(handler or default)
        return patch.object(route.httpx,'AsyncClient',side_effect=lambda **kwargs: original(transport=transport,**kwargs))

    async def test_duration_and_distance_ranking_and_artifacts(self):
        with self.mock():
            result=await route.plan_emergency_routes(origin=ORIGIN,coord_system='gcj02',destination_types=['hospital'],max_candidates=2)
            distance=await route.plan_emergency_routes(origin=ORIGIN,coord_system='gcj02',destination_types=['hospital'],max_candidates=2,rank_by='distance')
        self.assertEqual(result['status'],'success')
        recommended=result['destinations'][0]['recommended']
        self.assertEqual(recommended['facility']['id'],'hospital-2')
        self.assertEqual(recommended['route']['duration_s'],200)
        self.assertEqual(distance['destinations'][0]['recommended']['facility']['id'],'hospital-1')
        self.assertEqual(self.requests[0].url.params['sortrule'],'distance')
        saved=Path(result['summary_path']).read_text()
        self.assertEqual(json.loads(saved),result)
        self.assertNotIn('amap-test-secret',saved)
        self.assertNotIn('key=',recommended['route']['navigation_url'])
        data=json.loads(Path(result['outputs']['geojson_path']).read_text())
        self.assertEqual([f['geometry']['type'] for f in data['features']],['Point','LineString','Point'])
        self.assertNotEqual(data['features'][0]['geometry']['coordinates'],ORIGIN)
        self.assertEqual(data['features'][0]['geometry_gcj02']['coordinates'], ORIGIN)
        self.assertEqual(data['features'][1]['geometry_gcj02']['coordinates'], recommended['route']['polyline_gcj02'])
        self.assertEqual(data['features'][2]['geometry_gcj02']['coordinates'], recommended['facility']['location_gcj02'])

    async def test_community_station_is_not_recommended_as_hospital(self):
        def handler(request):
            if request.url.path.endswith('/around'):
                station = {**poi(1), 'name': '某社区卫生服务站'}
                return httpx.Response(200,json={'status':'1','pois':[station,poi(2)]})
            return httpx.Response(200,json=driving(request.url.params['destination']))
        with self.mock(handler):
            result=await route.plan_emergency_routes(origin=ORIGIN,coord_system='gcj02',destination_types=['hospital'],max_candidates=1)
        self.assertEqual(result['destinations'][0]['recommended']['facility']['id'],'hospital-2')
        self.assertEqual(result['destinations'][0]['returned_poi_count'],2)
        self.assertEqual(result['destinations'][0]['eligible_poi_count'],1)

    async def test_wgs84_uses_official_conversion_endpoint(self):
        with self.mock():
            await route.plan_emergency_routes(origin=ORIGIN,destination_types=['hospital'],max_candidates=1)
        self.assertTrue(self.requests[0].url.path.endswith('/coordinate/convert'))
        self.assertEqual(self.requests[0].url.params['coordsys'],'gps')
        self.assertEqual(self.requests[0].url.params['locations'],'116.400000,39.900000')

    async def test_no_facilities_is_not_a_fabricated_route(self):
        with self.mock(lambda _: httpx.Response(200,json={'status':'1','pois':[]})):
            result=await route.plan_emergency_routes(origin=ORIGIN,coord_system='gcj02')
        self.assertEqual(result['status'],'no_route')
        self.assertTrue(all(d['status']=='no_facilities' for d in result['destinations']))
        data=json.loads(Path(result['outputs']['geojson_path']).read_text())
        self.assertEqual(len(data['features']),1)

    async def test_missing_paths_and_avoidance_check(self):
        def handler(request):
            if request.url.path.endswith('/around'):
                return httpx.Response(200,json={'status':'1','pois':[poi(1)]})
            return httpx.Response(200,json=driving(request.url.params['destination']))
        ring=[[116.404,39.899],[116.406,39.899],[116.406,39.901],[116.404,39.901]]
        with self.mock(handler):
            result=await route.plan_emergency_routes(origin=ORIGIN,coord_system='gcj02',destination_types=['hospital'],avoid_polygons=[ring])
        self.assertEqual(result['status'],'no_route')
        self.assertEqual(result['destinations'][0]['candidates'][0]['status'],'no_acceptable_route')
        def no_path(request):
            return httpx.Response(200,json={'status':'1','pois':[poi(1)]} if request.url.path.endswith('/around') else {'status':'1','route':{'paths':[]}})
        with self.mock(no_path):
            result=await route.plan_emergency_routes(origin=ORIGIN,coord_system='gcj02',destination_types=['hospital'])
        self.assertIsNone(result['destinations'][0]['recommended'])

    async def test_validation_before_network(self):
        cases=[dict(origin=ORIGIN,disaster_path='x'),dict(origin=ORIGIN,destination_types=[]),
               dict(origin=ORIGIN,destination_types='hospital'),dict(origin=ORIGIN,search_radius_m=50001),
               dict(origin=ORIGIN,max_candidates=True),dict(origin=ORIGIN,output_dir='../escape'),
               dict(origin=ORIGIN,coord_system='bd09'),dict(origin=ORIGIN,avoid_polygons=[[[116.39,39.89],[116.42,39.89],[116.42,39.92],[116.39,39.92]]])]
        with patch.object(route,'AMapClient',side_effect=AssertionError('network must not run')):
            for options in cases:
                with self.subTest(options=options),self.assertRaises(ValueError):
                    await route.plan_emergency_routes(**options)

    async def test_auth_failure_and_redaction(self):
        with self.mock(lambda _: httpx.Response(200,json={'status':'0','infocode':'10001','info':'amap-test-secret'})):
            with self.assertRaises(route.AMapError) as ctx:
                await route.plan_emergency_routes(origin=ORIGIN,coord_system='gcj02')
        self.assertIn('10001',str(ctx.exception))
        self.assertNotIn('amap-test-secret',str(ctx.exception))
        self.assertFalse(list(route.TEMP_DIR.rglob('summary.json')))
        with patch.dict(os.environ,{'AMAP_WEB_KEY':''}):
            with self.assertRaisesRegex(RuntimeError,'Missing AMAP_WEB_KEY'):
                await route.plan_emergency_routes(origin=ORIGIN)

    async def test_timeout(self):
        def timeout(request): raise httpx.ReadTimeout('amap-test-secret',request=request)
        with self.mock(timeout):
            result=await route.plan_emergency_routes(origin=ORIGIN,coord_system='gcj02',destination_types=['hospital'])
        self.assertEqual(result['status'],'error')
        self.assertNotIn('amap-test-secret',json.dumps(result))

    @unittest.skipIf(route.mcp is None,'FastMCP unavailable')
    async def test_mcp_discovery_and_execution(self):
        from fastmcp import Client
        with self.mock(lambda _: httpx.Response(200,json={'status':'1','pois':[]})):
            async with Client(route.mcp,timeout=10) as client:
                tool=(await client.list_tools())[0]
                self.assertEqual(tool.name,'plan_emergency_routes')
                self.assertIn('disaster_path',tool.inputSchema['properties'])
                result=await client.call_tool(tool.name,{'origin':ORIGIN,'coord_system':'gcj02','destination_types':['hospital']})
                self.assertFalse(result.is_error)
                self.assertIn('no_route',result.content[0].text)


if __name__=='__main__': unittest.main()
