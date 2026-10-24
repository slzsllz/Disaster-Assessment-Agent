"""Server-side AMap geographic context with explicit GCJ02/WGS84 metadata."""

import math
import re
import time

import httpx

from agent.route_geometry import coordinate, gcj02_to_wgs84_approx


SEARCHES = {
    'hospital': {'types': '090100'},
    'school': {'types': '141200'},
    'park': {'types': '110101'},
    'residential': {'types': '120302'},
    'community': {'types': '120302'},
    'government': {'types': '130000'},
    'shelter': {'keywords': '避难场所'},
    'emergency': {'keywords': '应急'},
    'fire_station': {'keywords': '消防站'},
    'police': {'keywords': '派出所'},
    'bridge': {'keywords': '桥'},
    'building': {'types': '120000'},
    'commercial': {'keywords': '商场'},
}


class AMapContextError(RuntimeError):
    def __init__(self, code):
        self.code = str(code)
        super().__init__(f'AMap context request failed (code={self.code}); check service permissions, quota and inputs.')

    def result(self):
        return {'error': {'code': self.code, 'message': str(self)}}


def _string(value):
    return value if isinstance(value, str) else ''


def _number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def _pair(point):
    return ','.join(f'{v:.6f}' for v in point)


class AMapContextClient:
    def __init__(self, key, client):
        self.key = key
        self.client = client
        self.last_request = 0.

    def get(self, path, **params):
        time.sleep(max(0., .35 - (time.monotonic() - self.last_request)))
        self.last_request = time.monotonic()
        try:
            response = self.client.get('https://restapi.amap.com' + path,
                                       params={**params, 'key': self.key, 'output': 'JSON'})
        except httpx.TimeoutException:
            raise AMapContextError('TIMEOUT') from None
        except httpx.HTTPError:
            raise AMapContextError('NETWORK') from None
        if not response.is_success:
            raise AMapContextError(f'HTTP_{response.status_code}')
        try:
            data = response.json()
        except ValueError:
            raise AMapContextError('INVALID_RESPONSE') from None
        if not isinstance(data, dict):
            raise AMapContextError('INVALID_RESPONSE')
        if str(data.get('status')) != '1':
            code = str(data.get('infocode', ''))
            # Provider messages and request URLs can echo the key: return neither.
            raise AMapContextError(code if re.fullmatch(r'\d{5}', code) else 'INVALID_RESPONSE')
        return data

    def convert(self, points):
        data = self.get('/v3/assistant/coordinate/convert',
                        locations='|'.join(_pair(p) for p in points), coordsys='gps')
        try:
            converted = [coordinate(p.split(',')) for p in data['locations'].split(';')]
            if len(converted) != len(points):
                raise ValueError('inconsistent coordinate count')
            return converted
        except (KeyError, TypeError, ValueError, AttributeError):
            raise AMapContextError('INVALID_COORDINATES') from None


def _item(raw, category, *, native_is_wgs84=False):
    if not isinstance(raw, dict) or not _string(raw.get('name')):
        return None
    try:
        native = coordinate(raw['location'].split(','))
    except (KeyError, ValueError, TypeError, AttributeError):
        return None
    lon, lat = native if native_is_wgs84 else gcj02_to_wgs84_approx(native)
    return {
        'name': raw['name'], 'place_id': _string(raw.get('id')),
        'lon': lon, 'lat': lat, 'coordinate_system': 'WGS84' if native_is_wgs84 else 'WGS84_APPROX',
        'location_gcj02': native, 'native_coordinate_system': 'GCJ02',
        'categories': [_string(raw.get('type')) or category],
        'address': _string(raw.get('address')), 'telephone': _string(raw.get('tel')),
        'distance_from_center_m': _number(raw.get('distance')),
    }


def query_amap_context(center, bbox, query_types, limit, radius_m, key):
    location, places = {}, {}
    # All queries use fixed official endpoints and a server-side Web Service key.
    with httpx.Client(timeout=httpx.Timeout(20., connect=10.), trust_env=False) as http:
        api = AMapContextClient(key, http)
        try:
            native_center, northwest, southeast = api.convert([
                center, [bbox[0], bbox[3]], [bbox[2], bbox[1]],
            ])
        except AMapContextError as exc:
            return exc.result(), {name: {**exc.result(), 'items': []} for name in query_types}, {}
        native = {'center_gcj02': native_center, 'bbox_gcj02': [northwest[0], southeast[1], southeast[0], northwest[1]]}
        # The official converter can leave coordinates unchanged outside its
        # distortion coverage. Do not introduce an artificial inverse shift.
        native_is_wgs84 = all(abs(a - b) <= 1e-6 for a, b in zip(center, native_center))
        native['poi_wgs84_conversion'] = 'identity' if native_is_wgs84 else 'approximate_inverse'
        try:
            data = api.get('/v3/geocode/regeo', location=_pair(native_center),
                           radius=min(3000, int(radius_m)), extensions='all', roadlevel=0)
            regeo = data.get('regeocode')
            if not isinstance(regeo, dict):
                raise AMapContextError('INVALID_RESPONSE')
            component = regeo.get('addressComponent') or {}
            if not isinstance(component, dict):
                raise AMapContextError('INVALID_RESPONSE')
            roads = regeo.get('roads') or []
            if not isinstance(roads, list):
                raise AMapContextError('INVALID_RESPONSE')
            location = {
                'formatted_address': _string(regeo.get('formatted_address')),
                'country': _string(component.get('country')),
                'region': _string(component.get('province')), 'city': _string(component.get('city')),
                'district': _string(component.get('district')), 'adcode': _string(component.get('adcode')),
                'neighborhood': _string(component.get('township')),
                'roads': [item for raw in roads if (item := _item(raw, 'road', native_is_wgs84=native_is_wgs84))],
                'road_query_radius_m': min(3000, int(radius_m)),
            }
        except AMapContextError as exc:
            location = exc.result()
        for category in query_types:
            if category == 'road':
                if 'error' in location:
                    places[category] = {'error': location['error'], 'items': []}
                else:
                    roads = location.get('roads', [])
                    places[category] = {'count': len(roads[:limit]), 'items': roads[:limit],
                                        'source': 'AMap reverse geocoding road references, not a road network'}
                continue
            try:
                data = api.get('/v3/place/polygon',
                               polygon=_pair(northwest) + '|' + _pair(southeast),
                               offset=limit, page=1, extensions='all', **SEARCHES[category])
                if not isinstance(data.get('pois'), list):
                    raise AMapContextError('INVALID_RESPONSE')
                items = [item for raw in data['pois'] if (item := _item(raw, category, native_is_wgs84=native_is_wgs84))]
                places[category] = {'count': len(items[:limit]), 'items': items[:limit],
                                    'total_reported': _number(data.get('count')), 'page': 1,
                                    'note': 'First page only, not an exhaustive inventory; no absence inferred from empty results.'}
            except AMapContextError as exc:
                places[category] = {**exc.result(), 'items': []}
    return location, places, native
