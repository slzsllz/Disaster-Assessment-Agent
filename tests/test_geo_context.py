import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx
import numpy as np
import rasterio
from rasterio.transform import from_origin

from agent.geo_region import context_provider
from agent import amap_context

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('geo_context_test', ROOT / 'agent/tools/GeoContextQuery.py')
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)
DOMESTIC = [114.06, 22.54]
FOREIGN = [-73.98, 40.75]


class RegionTests(unittest.TestCase):
    def test_domestic_and_neighboring_foreign_cities(self):
        for point in (DOMESTIC, [116.4,39.9], [87.45,28.5], [114.1694,22.3193], [113.5439,22.1987], [121.5654,25.033]):
            with self.subTest(point=point):
                self.assertEqual(context_provider(point), 'amap')
        for point in (FOREIGN, [100.5,13.75], [139.69,35.68], [85.32,27.72], [126.98,37.57], [104.92,11.56]):
            with self.subTest(point=point):
                self.assertEqual(context_provider(point), 'arcgis')

    def test_boundary_data_matches_existing_map_geometry(self):
        source = (ROOT / 'frontend/chatDisaster/src/utils/chinaBoundary.js').read_text()
        coordinates, _ = json.JSONDecoder().raw_decode(source.split('export default ',1)[1])
        backend = json.loads((ROOT / 'agent/china_boundary.json').read_text())
        self.assertEqual(backend['coordinates'], coordinates)


class GeographicContextTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        patches = [
            patch.object(geo, 'TEMP_DIR', self.root),
            patch.object(geo, 'PROJECT_ROOT', self.root),
            patch.dict(os.environ, {'AMAP_WEB_KEY':'test-amap-secret', 'ARCGIS_API_KEY':'test-arcgis-secret', 'ARCGIS_FEATURE_LAYERS_JSON':''}),
            patch.object(amap_context.time, 'sleep'),
        ]
        for item in patches:
            item.start()
            self.addCleanup(item.stop)
        self.requests = []

    def handler(self, request):
        self.requests.append(request)
        self.assertEqual(request.url.host, 'restapi.amap.com')
        self.assertEqual(request.url.params['key'], 'test-amap-secret')
        self.assertNotIn('Authorization', request.headers)
        path = request.url.path
        if path.endswith('/convert'):
            self.assertEqual(request.url.params['coordsys'], 'gps')
            points = [list(map(float,p.split(','))) for p in request.url.params['locations'].split('|')]
            locations = ';'.join(f'{x+.006:.6f},{y+.006:.6f}' for x,y in points)
            return httpx.Response(200,json={'status':'1','locations':locations})
        if path.endswith('/regeo'):
            return httpx.Response(200,json={'status':'1','regeocode':{
                'formatted_address':'广东省深圳市福田区测试地址',
                'addressComponent':{'country':'中国','province':'广东省','city':'深圳市','district':'福田区','adcode':'440304','township':'测试街道'},
                'roads':[{'id':'road1','name':'测试道路','location':'114.066000,22.546000','distance':'25'}],
            }})
        if path.endswith('/polygon'):
            nw,se=[list(map(float,p.split(','))) for p in request.url.params['polygon'].split('|')]
            self.assertLess(nw[0],se[0])
            self.assertGreater(nw[1],se[1])
            return httpx.Response(200,json={'status':'1','count':'2','pois':[
                {'id':'poi1','name':'测试医院','location':'114.066000,22.546000','address':'测试地址','tel':'0000','type':'医疗保健服务'},
                {'id':'bad','name':'无坐标地点','location':[]},
            ]})
        raise AssertionError('Unexpected endpoint')

    def client(self, handler=None):
        client_class = httpx.Client
        transport = httpx.MockTransport(handler or self.handler)
        return patch.object(amap_context.httpx, 'Client', side_effect=lambda **kwargs:client_class(transport=transport,**kwargs))

    def test_domestic_uses_amap_only_and_returns_tagged_coordinates(self):
        with self.client(), patch.object(geo,'_arcgis_token',side_effect=AssertionError('ArcGIS key accessed')), patch.object(geo,'_get_json',side_effect=AssertionError('ArcGIS called')):
            result = geo.query_geo_context(center=DOMESTIC,query_types=['hospital','school','road'],max_results_per_type=2)
        self.assertEqual(result['provider'],'amap')
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['location']['city'],'深圳市')
        self.assertEqual(result['coordinate_systems']['service'],'GCJ02')
        self.assertEqual(len(self.requests),4)
        item = result['nearby_places']['hospital']['items'][0]
        self.assertEqual(item['location_gcj02'],[114.066,22.546])
        self.assertEqual(item['coordinate_system'],'WGS84_APPROX')
        self.assertNotEqual([item['lon'],item['lat']],item['location_gcj02'])
        self.assertEqual(result['nearby_places']['road']['items'][0]['distance_from_center_m'],25)
        saved = Path(result['summary_path']).read_text()
        self.assertNotIn('test-amap-secret',saved)
        self.assertNotIn('test-arcgis-secret',saved)
        self.assertIn('not complete building footprints',saved)

    def test_foreign_uses_arcgis_only_and_preserves_feature_layers(self):
        layer = [{'name':'buildings','url':'https://example.invalid/FeatureServer/0'}]
        with patch.object(geo,'query_amap_context',side_effect=AssertionError('AMap called')), \
             patch.object(geo,'_reverse_geocode',return_value={'formatted_address':'New York'}) as reverse, \
             patch.object(geo,'_query_places',return_value={'hospital':{'count':1,'items':[{'name':'Hospital'}]}}), \
             patch.object(geo,'_query_feature_layers',return_value={'buildings':{'count':1,'items':[{}]}}) as features:
            result=geo.query_geo_context(center=FOREIGN,query_types=['hospital'],feature_layers=layer)
        self.assertEqual(result['provider'],'arcgis')
        self.assertEqual(result['status'],'success')
        reverse.assert_called_once_with(FOREIGN,'test-arcgis-secret')
        self.assertEqual(features.call_args.args[2],layer)
        self.assertEqual(result['summary']['feature_counts_by_layer'],{'buildings':1})

    def test_missing_domestic_key_does_not_use_arcgis(self):
        with patch.dict(os.environ,{'AMAP_WEB_KEY':''}), patch.object(geo,'_arcgis_token',side_effect=AssertionError('Fallback')):
            with self.assertRaisesRegex(RuntimeError,'AMAP_WEB_KEY'):
                geo.query_geo_context(center=DOMESTIC)

    def test_missing_foreign_key_does_not_use_amap(self):
        with patch.object(geo,'_arcgis_token',return_value=''),patch.object(geo,'query_amap_context',side_effect=AssertionError('Fallback')):
            with self.assertRaisesRegex(RuntimeError,'ARCGIS_API_KEY'):
                geo.query_geo_context(center=FOREIGN)

    def test_amap_key_can_be_read_from_dotenv_with_export(self):
        (self.root/'.env').write_text('export AMAP_WEB_KEY="test-amap-secret" # operator setting\n')
        with patch.dict(os.environ,{'AMAP_WEB_KEY':''}),self.client():
            self.assertEqual(geo.query_geo_context(center=DOMESTIC,query_types=[])['status'],'success')

    def test_domestic_does_not_execute_arcgis_layers(self):
        with self.client(),patch.object(geo,'_query_feature_layers',side_effect=AssertionError('ArcGIS queried')):
            result=geo.query_geo_context(center=DOMESTIC,query_types=[],feature_layers=[{'url':'https://example.invalid'}])
        self.assertEqual(result['status'],'partial')
        self.assertEqual(result['intersecting_feature_layers'],{})
        self.assertIn('not executed',' '.join(result['limitations']))

    def test_amap_error_is_not_reported_as_success_and_does_not_leak_key(self):
        def reject(request):
            return httpx.Response(200,json={'status':'0','infocode':'10003','info':'test-amap-secret quota'})
        with self.client(reject),patch.object(geo,'_get_json',side_effect=AssertionError('Fallback')):
            result=geo.query_geo_context(center=DOMESTIC,query_types=['hospital'])
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['errors'][0]['code'],'10003')
        self.assertNotIn('test-amap-secret',json.dumps(result))

    def test_partial_failure_and_empty_results_are_distinct(self):
        def handler(request):
            if request.url.path.endswith('/polygon'):
                return httpx.Response(200,json={'status':'1','count':'0','pois':[]})
            if request.url.path.endswith('/regeo'):
                return httpx.Response(200,json={'status':'0','infocode':'10005'})
            return self.handler(request)
        with self.client(handler):
            result=geo.query_geo_context(center=DOMESTIC,query_types=['hospital'])
        self.assertEqual(result['status'],'partial')
        self.assertNotIn('error',result['nearby_places']['hospital'])
        self.assertEqual(result['nearby_places']['hospital']['count'],0)

    def test_timeout_does_not_echo_request_url_or_key(self):
        def timeout(request):
            raise httpx.ReadTimeout(str(request.url),request=request)
        with self.client(timeout):
            result=geo.query_geo_context(center=DOMESTIC,query_types=[])
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['errors'][0]['code'],'TIMEOUT')
        self.assertNotIn('key=',json.dumps(result))

    def test_unchanged_official_conversion_is_not_shifted_again(self):
        def handler(request):
            if request.url.path.endswith('/convert'):
                return httpx.Response(200,json={'status':'1','locations':request.url.params['locations'].replace('|',';')})
            return self.handler(request)
        with self.client(handler):
            result=geo.query_geo_context(center=DOMESTIC,query_types=['hospital'])
        item=result['nearby_places']['hospital']['items'][0]
        self.assertEqual([item['lon'],item['lat']],item['location_gcj02'])
        self.assertEqual(result['coordinate_systems']['poi_wgs84_conversion'],'identity')

    def test_invalid_provider_responses_fail_explicitly(self):
        for payload in ([], {'status':'1','locations':'bad'}, {'status':'1','locations':'114,22'}):
            with self.subTest(payload=payload), self.client(lambda _:httpx.Response(200,json=payload)):
                result=geo.query_geo_context(center=DOMESTIC,query_types=[])
                self.assertEqual(result['status'],'error')

    def test_empty_poi_result_does_not_imply_service_failure(self):
        def handler(request):
            if request.url.path.endswith('/polygon'):
                return httpx.Response(200,json={'status':'1','count':'0','pois':[]})
            return self.handler(request)
        with self.client(handler):
            result=geo.query_geo_context(center=DOMESTIC,query_types=['hospital'])
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['nearby_places']['hospital']['count'],0)

    def test_foreign_failures_are_not_silently_replaced_by_amap(self):
        failure={'error':{'code':498,'message':'Invalid token'}}
        with patch.object(geo,'query_amap_context',side_effect=AssertionError('Fallback')), \
             patch.object(geo,'_reverse_geocode',return_value=failure), \
             patch.object(geo,'_query_places',return_value={'hospital':{**failure,'items':[]}}), \
             patch.object(geo,'_query_feature_layers',return_value={}):
            result=geo.query_geo_context(center=FOREIGN,query_types=['hospital'])
        self.assertEqual(result['status'],'error')
        self.assertEqual(result['provider'],'arcgis')

    def test_validation_precedes_requests(self):
        cases=[{'center':[float('nan'),22]}, {'center':[22,114]}, {'center':[114,True]},
               {'bbox':[115,22,114,23]}, {'bbox':[114,22,115,23],'center':[116,25]},
               {'center':DOMESTIC,'radius_m':float('inf')}, {'center':DOMESTIC,'radius_m':0},
               {'center':DOMESTIC,'query_types':['unknown']}, {'center':DOMESTIC,'query_types':'hospital'},
               {'center':DOMESTIC,'max_results_per_type':True}, {'center':DOMESTIC,'max_results_per_type':21},
               {'center':DOMESTIC,'output_dir':'../escape'}, {'center':DOMESTIC,'output_dir':'/tmp/escape'},
               {'center':DOMESTIC,'feature_layers':['bad']}, {}]
        with patch.object(geo,'query_amap_context') as amap,patch.object(geo,'_get_json') as arcgis:
            for kwargs in cases:
                with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                    geo.query_geo_context(**kwargs)
            amap.assert_not_called()
            arcgis.assert_not_called()

    def test_projected_raster_routing_and_missing_crs(self):
        from rasterio.warp import transform
        x,y=transform('EPSG:4326','EPSG:3857',[DOMESTIC[0]],[DOMESTIC[1]])
        path=self.root/'input.tif'
        profile=dict(driver='GTiff',width=2,height=2,count=1,dtype='uint8',transform=from_origin(x[0]-10,y[0]+10,10,10))
        with rasterio.open(path,'w',crs='EPSG:3857',**profile) as dst:
            dst.write(np.zeros((1,2,2),np.uint8))
        with self.client():
            result=geo.query_geo_context(raster_path=str(path),query_types=[])
        self.assertEqual(result['provider'],'amap')
        self.assertEqual(result['provider_selection']['basis'],'bbox_center')
        np.testing.assert_allclose(result['input']['center'],DOMESTIC,atol=1e-5)
        with rasterio.open(path,'w',**profile) as dst:
            dst.write(np.zeros((1,2,2),np.uint8))
        with self.assertRaisesRegex(ValueError,'no CRS'):
            geo.query_geo_context(raster_path=str(path))

    def test_large_extent_is_bounded_and_summaries_are_unique(self):
        with self.client():
            first=geo.query_geo_context(bbox=[113,21,115,24],center=DOMESTIC,query_types=[])
            second=geo.query_geo_context(center=DOMESTIC,query_types=[])
        self.assertTrue(first['query_extent']['places_bbox_was_limited_to_service_max_extent'])
        self.assertNotEqual(first['summary_path'],second['summary_path'])
        self.assertTrue(Path(first['summary_path']).is_file())


if __name__=='__main__':
    unittest.main()
