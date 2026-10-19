"""AMap nearby hospital/shelter/fire-station search and emergency route planning."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import json
import math
import os
import re
from pathlib import Path
import sys
from typing import Optional
from urllib.parse import urlencode
from uuid import uuid4

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from agent.route_geometry import coordinate, gcj02_to_wgs84_approx, resolve_origin

try:
    from fastmcp import FastMCP
except ImportError as exc:
    FastMCP = None
    FASTMCP_IMPORT_ERROR = exc

load_dotenv(PROJECT_ROOT / '.env', override=False)
TEMP_DIR = Path('tmp/tmp/out')
mcp = FastMCP() if FastMCP is not None else None
BASE_URL = 'https://restapi.amap.com'
FACILITIES = {
    'hospital': {'label': '综合医院', 'types': '090100'},
    'shelter': {'label': '应急避难场所', 'keywords': '避难场所'},
    'fire_station': {'label': '消防站', 'keywords': '消防站'},
}
TOOL_DESCRIPTION = """
Plan road-network routes from a disaster/incident location to nearby hospitals,
emergency shelters, or fire stations using AMap Web Services. Use for evacuation,
medical transfer, and responder logistics after the user provides incident coordinates/data.
Supply exactly one of origin=[longitude,latitude] or disaster_path (GeoJSON/JSON,
CSV longitude/latitude records, or georeferenced GeoTIFF). Multiple features/rows require
feature_index (zero-based). Polygon inputs use a representative point; GeoTIFF uses its
extent center, which may differ from a real incident or accessible pickup location.
coord_system defaults to wgs84 (GPS/GeoJSON); use gcj02 only for known AMap coordinates.
Never swap coordinate systems or infer coordinates from an ungeoreferenced image.
destination_types defaults to [hospital,shelter]; also supports fire_station.
Search within search_radius_m (100..50000, default 10000). Compare max_candidates (1..5,
default 3) nearest-by-straight-line facilities per type, and recommend the reachable
candidate by rank_by=duration (default, seconds) or distance (driving meters).
This is the best among evaluated candidates, not a guarantee of the globally nearest
facility. No results or no drivable paths must be reported honestly; do not invent routes.
avoid_polygons optionally supplies known impassable areas as coordinate rings in the
same coord_system (max 32 polygons, 3..16 vertices and <=81 square km each).
Origin must be outside avoidance polygons; use an accessible pickup point when necessary.
Returns facility names/addresses/phones, estimated driving distance/time, instructions,
AMap navigation links, native GCJ02 routes, approximate WGS84 GeoJSON and a saved summary.
Output paths go under output_dir within the tool temp directory. The API key is server-side.
Important: ordinary passenger-car routing, NOT certified fire-engine dispatch/navigation.
No real-time disaster blockage, hospital capacity, shelter opening status or vehicle clearance
is verified. A POI match is not confirmation of emergency receiving capability. Do not call
routes safe or guaranteed passable. Navigation links recalculate and do not retain avoidance
polygons; use the saved instructions and verify local closures with responders.
"""


class AMapError(RuntimeError):
    def __init__(self, message, code=''):
        super().__init__(message)
        self.code = str(code)


def _text(value):
    return value if isinstance(value, str) else ''


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number >= 0 else None
    except (TypeError, ValueError):
        return None


def _pair(point):
    return ','.join(f'{v:.6f}' for v in point)


class AMapClient:
    def __init__(self, client):
        self.client = client
        self.key = os.getenv('AMAP_WEB_KEY', '').strip()
        self.last_request = 0.
        if not self.key:
            raise RuntimeError('Missing AMAP_WEB_KEY. Configure it in the project .env or process environment.')

    async def get(self, path, **params):
        # Sequential, paced calls keep a route comparison from bursting the POI/route APIs.
        loop = asyncio.get_running_loop()
        await asyncio.sleep(max(0., .35 - (loop.time() - self.last_request)))
        self.last_request = loop.time()
        try:
            response = await self.client.get(BASE_URL + path, params={**params, 'key': self.key, 'output': 'JSON'})
        except httpx.TimeoutException:
            raise AMapError('AMap request timed out; retry later.', 'TIMEOUT') from None
        except httpx.RequestError:
            raise AMapError('Cannot connect to AMap Web Services; check network access.', 'NETWORK') from None
        if not response.is_success:
            raise AMapError(f'AMap returned HTTP {response.status_code}.', f'HTTP_{response.status_code}')
        try:
            data = response.json()
        except ValueError:
            raise AMapError('AMap returned a non-JSON response.', 'INVALID_RESPONSE') from None
        if not isinstance(data, dict):
            raise AMapError('AMap returned an invalid response object.', 'INVALID_RESPONSE')
        if str(data.get('status')) != '1':
            value = _text(data.get('infocode'))
            code = value if re.fullmatch(r'\d{5}', value) else 'INVALID_RESPONSE'
            # No provider response text or request URLs: those can echo credentials.
            raise AMapError(f'AMap rejected the request (infocode={code or "unknown"}); check key permissions/quota and input parameters.', code)
        return data

    async def convert(self, points, system):
        if system == 'gcj02':
            return [p[:] for p in points]
        converted = []
        for offset in range(0, len(points), 40):
            data = await self.get('/v3/assistant/coordinate/convert',
                                  locations='|'.join(_pair(p) for p in points[offset:offset+40]), coordsys='gps')
            try:
                converted.extend(coordinate(s.split(',')) for s in data['locations'].split(';'))
            except (KeyError, ValueError, TypeError, AttributeError):
                raise AMapError('AMap coordinate conversion returned invalid locations.', 'INVALID_RESPONSE') from None
        if len(converted) != len(points):
            raise AMapError('AMap coordinate conversion returned an inconsistent count.', 'INVALID_RESPONSE')
        return converted


def _avoidance(rings, origin):
    from pyproj import Geod
    from shapely.geometry import Point, Polygon
    if rings is None:
        return []
    if not isinstance(rings, list) or len(rings) > 32:
        raise ValueError('avoid_polygons must contain at most 32 coordinate rings.')
    result = []
    for ring in rings:
        points = [coordinate(p) for p in ring]
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if not 3 <= len(points) <= 16:
            raise ValueError('Each avoidance ring requires 3..16 vertices.')
        polygon = Polygon(points)
        if polygon.is_empty or not polygon.is_valid or polygon.area == 0:
            raise ValueError('Invalid avoidance polygon.')
        area, _ = Geod(ellps='WGS84').polygon_area_perimeter(*zip(*points))
        if abs(area) > 81_000_000:
            raise ValueError('AMap avoidance polygons must each be no larger than 81 square kilometers.')
        if polygon.covers(Point(origin)):
            raise ValueError('Origin is inside an avoidance polygon. Supply an accessible pickup point outside the blocked area.')
        result.append(points)
    return result


def _poi(raw):
    try:
        point = coordinate(raw['location'].split(','))
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    name = _text(raw.get('name'))
    if not name:
        return None
    return {'id': _text(raw.get('id')), 'name': name, 'location_gcj02': point,
            'address': _text(raw.get('address')), 'telephone': _text(raw.get('tel')),
            'type': _text(raw.get('type')), 'straight_line_distance_m': _number(raw.get('distance'))}


def _eligible_facility(kind, poi):
    name = poi['name']
    if kind == 'hospital':
        # AMap sometimes tags community service stations as 090100. They must
        # not silently become hospital transfer destinations.
        return bool(re.search(r'医院|医疗中心|卫生院', name)) and not re.search(
            r'社区卫生|卫生服务站|卫生服务中心|诊所|门诊部|口腔|牙科|美容|整形|体检|宠物|兽医|停车场|出入口', name)
    if kind == 'shelter':
        return bool(re.search(r'避难|避险|安置', name))
    return '消防' in name and not re.search(r'器材|培训|学校|检测', name)


def _parse_path(path, origin, destination, blocked):
    from shapely.geometry import LineString
    distance, duration = _number(path.get('distance')), _number(path.get('duration'))
    if distance is None or duration is None:
        return None
    line, instructions = [], []
    for step in path.get('steps') or []:
        if not isinstance(step, dict):
            return None
        instructions.append({'instruction': _text(step.get('instruction')), 'road': _text(step.get('road')),
                             'distance_m': _number(step.get('distance')), 'duration_s': _number(step.get('duration'))})
        try:
            points = [coordinate(p.split(',')) for p in _text(step.get('polyline')).split(';') if p]
        except ValueError:
            return None
        for point in points:
            if not line or point != line[-1]:
                line.append(point)
    if len(line) < 2:
        return None
    if any(area.intersects(LineString(line)) for area in blocked):
        return None
    navigation_url = 'https://uri.amap.com/navigation?' + urlencode({
        'from': f'{_pair(origin)},灾害位置', 'to': f'{_pair(destination["location_gcj02"])},{destination["name"]}',
        'mode': 'car', 'coordinate': 'gaode', 'src': 'DisasterAssessmentAgent', 'callnative': '0',
    })
    return {'distance_m': distance, 'duration_s': duration, 'instructions': instructions,
            'polyline_gcj02': line, 'navigation_url': navigation_url,
            'provider_strategy': _text(path.get('strategy')), 'traffic_restriction': path.get('restriction'),
            'navigation_link_retains_avoidance': False}


def _fatal(error):
    # Invalid credentials, disabled APIs, or exhausted quota affect every candidate.
    return error.code.startswith('100') or error.code in ('HTTP_401', 'HTTP_403', 'HTTP_429')


async def plan_emergency_routes(
    origin: Optional[list[float]] = None,
    disaster_path: Optional[str] = None,
    coord_system: str = 'wgs84',
    destination_types: Optional[list[str]] = None,
    search_radius_m: int = 10000,
    max_candidates: int = 3,
    rank_by: str = 'duration',
    feature_index: Optional[int] = None,
    avoid_polygons: Optional[list[list[list[float]]]] = None,
    output_dir: str = 'route_planning',
) -> dict:
    location, system, source = resolve_origin(origin, disaster_path, coord_system, feature_index)
    if destination_types is not None and (not isinstance(destination_types, list) or any(not isinstance(k, str) for k in destination_types)):
        raise ValueError('destination_types must be a list of facility type names.')
    kinds = ['hospital', 'shelter'] if destination_types is None else list(dict.fromkeys(destination_types))
    if not kinds or any(kind not in FACILITIES for kind in kinds):
        raise ValueError('destination_types must be a nonempty list of hospital, shelter, or fire_station.')
    for name, value, low, high in (('search_radius_m', search_radius_m, 100, 50000), ('max_candidates', max_candidates, 1, 5)):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in [{low},{high}].')
    if rank_by not in ('duration', 'distance'):
        raise ValueError('rank_by must be duration or distance.')
    root = TEMP_DIR.resolve()
    directory = (root / output_dir).resolve()
    if Path(output_dir).is_absolute() or not directory.is_relative_to(root):
        raise ValueError('output_dir must be relative and inside the tool temp directory.')
    rings = _avoidance(avoid_polygons, location)
    # Files with their own CRS are WGS84, but explicit avoidance rings still use coord_system.
    if rings and system != coord_system:
        raise ValueError('Avoidance rings must use the same CRS as the resolved incident location; use wgs84 for georeferenced files.')
    async with httpx.AsyncClient(timeout=httpx.Timeout(25., connect=10.), trust_env=False) as http:
        api = AMapClient(http)
        flattened = [location] + [p for ring in rings for p in ring]
        transformed = await api.convert(flattened, system)
        incident = transformed[0]
        gcj_rings, cursor = [], 1
        for ring in rings:
            gcj_rings.append(transformed[cursor:cursor+len(ring)])
            cursor += len(ring)
        from shapely.geometry import Polygon, Point
        blocked = [Polygon(ring) for ring in gcj_rings]
        avoidance_arg = '|'.join(';'.join(_pair(p) for p in ring) for ring in gcj_rings)
        destinations, errors = [], []
        for kind in kinds:
            spec = FACILITIES[kind]
            try:
                response = await api.get('/v3/place/around', location=_pair(incident), radius=search_radius_m,
                                         sortrule='distance', offset=25, page=1, extensions='all',
                                         **{k:v for k,v in spec.items() if k != 'label'})
            except AMapError as exc:
                if _fatal(exc): raise
                errors.append({'destination_type': kind, 'stage': 'facility_search', 'error': str(exc)})
                destinations.append({'destination_type': kind, 'status': 'search_failed', 'candidates': []})
                continue
            raw_pois = response.get('pois')
            if not isinstance(raw_pois, list):
                raise AMapError('AMap POI response is missing its pois list.', 'INVALID_RESPONSE')
            pois = [p for p in (_poi(raw) for raw in raw_pois if isinstance(raw, dict))
                    if p and _eligible_facility(kind, p)]
            pois.sort(key=lambda p: p['straight_line_distance_m'] if p['straight_line_distance_m'] is not None else math.inf)
            candidates, seen_ids = [], set()
            for poi in pois[:max_candidates]:
                identity = poi['id'] or str(poi['location_gcj02'])
                if identity in seen_ids:
                    continue
                seen_ids.add(identity)
                if any(area.covers(Point(poi['location_gcj02'])) for area in blocked):
                    candidates.append({'facility': poi, 'status': 'inside_avoidance_area'})
                    continue
                try:
                    params = dict(origin=_pair(incident), destination=_pair(poi['location_gcj02']),
                                  strategy=10, extensions='all', ferry=1)
                    if avoidance_arg: params['avoidpolygons'] = avoidance_arg
                    data = await api.get('/v3/direction/driving', **params)
                    details = data.get('route') or {}
                    if not isinstance(details, dict):
                        raise AMapError('AMap returned invalid driving route details.', 'INVALID_RESPONSE')
                    paths = details.get('paths') or []
                    if not isinstance(paths, list) or any(not isinstance(p, dict) for p in paths):
                        raise AMapError('AMap returned invalid driving paths.', 'INVALID_RESPONSE')
                    parsed = [p for p in (_parse_path(p, incident, poi, blocked) for p in paths) if p]
                    metric = 'duration_s' if rank_by == 'duration' else 'distance_m'
                    parsed.sort(key=lambda p: (p[metric], p['distance_m']))
                    candidates.append({'facility': poi, 'status': 'success' if parsed else 'no_acceptable_route',
                                       'route': parsed[0] if parsed else None})
                except AMapError as exc:
                    if _fatal(exc): raise
                    errors.append({'destination_type': kind, 'facility_id': poi['id'], 'stage': 'driving', 'error': str(exc)})
                    candidates.append({'facility': poi, 'status': 'route_failed'})
            reachable = [p for p in candidates if p['status'] == 'success']
            metric = 'duration_s' if rank_by == 'duration' else 'distance_m'
            reachable.sort(key=lambda p: (p['route'][metric], p['route']['distance_m']))
            destinations.append({'destination_type': kind, 'label': spec['label'],
                                 'status': 'success' if reachable else 'no_acceptable_route' if candidates else 'no_facilities',
                                 'returned_poi_count': len(raw_pois), 'eligible_poi_count': len(pois),
                                 'candidates': candidates, 'recommended': reachable[0] if reachable else None})
    features = [{'type': 'Feature', 'properties': {'role': 'incident', 'name': '灾害位置'},
                 'geometry': {'type': 'Point', 'coordinates': gcj02_to_wgs84_approx(incident)}}]
    for group in destinations:
        recommended = group.get('recommended')
        if not recommended: continue
        route, poi = recommended['route'], recommended['facility']
        properties = {'role': 'route', 'destination_type': group['destination_type'], 'name': poi['name'],
                      'distance_m': route['distance_m'], 'duration_s': route['duration_s']}
        features.append({'type': 'Feature', 'properties': properties, 'geometry': {
            'type': 'LineString', 'coordinates': [gcj02_to_wgs84_approx(p) for p in route['polyline_gcj02']]}})
        features.append({'type': 'Feature', 'properties': {**properties, 'role': 'destination'},
                         'geometry': {'type': 'Point', 'coordinates': gcj02_to_wgs84_approx(poi['location_gcj02'])}})
    successful = sum(g.get('status') == 'success' for g in destinations)
    result = {
        'task': 'route_planning', 'status': 'success' if successful == len(kinds) else 'partial' if successful else 'error' if errors else 'no_route',
        'provider': 'amap', 'planned_at': datetime.now(timezone.utc).isoformat(),
        'origin': {'input': location, 'input_coord_system': system, 'location_gcj02': incident, 'source': source},
        'search': {'radius_m': search_radius_m, 'max_candidates_per_type': max_candidates, 'rank_by': rank_by,
                   'scope': 'Filter the first 25 nearby POIs, then compare the nearest eligible candidates; not globally optimal.'},
        'destinations': destinations, 'errors': errors,
        'avoidance': {'polygon_count': len(rings), 'checked_returned_paths': True},
        'coordinate_systems': {'native_routes': 'GCJ02', 'geojson': 'WGS84 (approximate inverse for visualization)'},
        'limitations': ['普通小客车路网规划，未验证消防车尺寸、载重、应急通行或灾后道路可通行性。',
                        '未核实医院急诊接收能力、床位或避难场所开放状态，请联系设施和现场指挥确认。',
                        '图像中心或灾害面代表点不一定可供车辆抵达，优先提供可达的救援接驳点。',
                        '高德导航链接会重新算路，不保留本次避让区域；GeoJSON为近似WGS84，仅作地图展示。'],
    }
    out = directory / uuid4().hex
    out.mkdir(parents=True, exist_ok=True)
    geojson_path = out / 'routes.geojson'
    geojson_path.write_text(json.dumps({'type': 'FeatureCollection', 'properties': {
        'task': 'route_planning', 'coordinate_conversion': 'approximate_gcj02_inverse'}, 'features': features},
        ensure_ascii=False, allow_nan=False), encoding='utf-8')
    result['outputs'] = {'geojson_path': str(geojson_path)}
    result['summary_path'] = str(out / 'summary.json')
    Path(result['summary_path']).write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return result


if mcp is not None:
    mcp.tool(description=TOOL_DESCRIPTION)(plan_emergency_routes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--temp_dir', default=str(TEMP_DIR))
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument('--origin', help='longitude,latitude')
    inputs.add_argument('--data', help='GeoJSON/JSON, CSV or GeoTIFF')
    parser.add_argument('--coord-system', choices=['wgs84','gcj02'], default='wgs84')
    parser.add_argument('--destinations', nargs='+', choices=list(FACILITIES), default=['hospital','shelter'])
    parser.add_argument('--radius', type=int, default=10000)
    parser.add_argument('--max-candidates', type=int, default=3)
    parser.add_argument('--rank-by', choices=['duration','distance'], default='duration')
    parser.add_argument('--feature-index', type=int)
    parser.add_argument('--avoid-polygons', help='JSON array of coordinate rings')
    parser.add_argument('--output', default='route_planning')
    args = parser.parse_args()
    TEMP_DIR = Path(args.temp_dir)
    if args.origin or args.data:
        try:
            result = asyncio.run(plan_emergency_routes(
                origin=coordinate(args.origin.split(',')) if args.origin else None,
                disaster_path=args.data, coord_system=args.coord_system, destination_types=args.destinations,
                search_radius_m=args.radius, max_candidates=args.max_candidates, rank_by=args.rank_by,
                feature_index=args.feature_index, avoid_polygons=json.loads(args.avoid_polygons) if args.avoid_polygons else None,
                output_dir=args.output))
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            parser.exit(1, f'{exc}\n')
    else:
        if mcp is None: raise FASTMCP_IMPORT_ERROR
        mcp.run()
