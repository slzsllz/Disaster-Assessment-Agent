"""Input geometry and explicitly approximate display coordinate conversion for routes."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def coordinate(value) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2 or any(isinstance(v, bool) for v in value):
        raise ValueError('Coordinates must be [longitude, latitude].')
    try:
        lon, lat = map(float, value)
    except (TypeError, ValueError):
        raise ValueError('Coordinates must be numeric [longitude, latitude].') from None
    if not math.isfinite(lon) or not math.isfinite(lat) or not -180 <= lon <= 180 or not -90 <= lat <= 90:
        raise ValueError('Invalid longitude/latitude; projected coordinates require a file with a declared CRS.')
    return [lon, lat]


def _select(items, index):
    if not items:
        raise ValueError('The input has no features or coordinate records.')
    if index is None:
        if len(items) != 1:
            raise ValueError('Multiple locations found. Supply feature_index or an explicit incident origin; no location was guessed.')
        return items[0]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(items):
        raise ValueError('feature_index is outside the available records.')
    return items[index]


def _record_point(record):
    for x, y in (('longitude', 'latitude'), ('lon', 'lat'), ('lng', 'lat')):
        if x in record and y in record:
            return coordinate([record[x], record[y]])
    raise ValueError('Coordinate records require longitude/latitude, lon/lat, or lng/lat columns.')


def resolve_origin(origin, disaster_path, coord_system, feature_index=None):
    if (origin is None) == (disaster_path is None):
        raise ValueError('Supply exactly one of origin or disaster_path.')
    if coord_system not in ('wgs84', 'gcj02'):
        raise ValueError('coord_system must explicitly be wgs84 or gcj02; BD09 is not interchangeable.')
    if origin is not None:
        if feature_index is not None:
            raise ValueError('feature_index only applies to disaster_path.')
        return coordinate(origin), coord_system, {'kind': 'explicit_incident_location'}
    if feature_index is not None and (isinstance(feature_index, bool) or not isinstance(feature_index, int) or feature_index < 0):
        raise ValueError('feature_index must be a nonnegative integer.')
    path = Path(disaster_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() in ('.tif', '.tiff'):
        import rasterio
        from rasterio.warp import transform
        with rasterio.open(path) as src:
            if not src.crs:
                raise ValueError('Raster has no CRS; cannot infer a reliable geographic incident location.')
            x, y = src.transform * (src.width / 2, src.height / 2)
            lon, lat = transform(src.crs, 'EPSG:4326', [x], [y])
            return coordinate([lon[0], lat[0]]), 'wgs84', {
                'kind': 'raster_extent_center', 'path': str(path), 'source_crs': str(src.crs),
                'note': 'Image extent center, not a detected incident point. Override with an accessible incident origin when known.',
            }
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        return _record_point(_select(rows, feature_index)), coord_system, {'kind': 'csv_record', 'path': str(path), 'feature_index': feature_index or 0}
    if path.suffix.lower() not in ('.json', '.geojson'):
        raise ValueError('Supported data: GeoJSON/JSON, CSV longitude/latitude records, or GeoTIFF with CRS.')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if isinstance(data, list):
        data = _select(data, feature_index)
    if not isinstance(data, dict):
        raise ValueError('Input JSON must contain a geographic geometry or a coordinate record.')
    crs = data.get('crs')
    if crs:
        name = str((crs.get('properties') or {}).get('name', '')).upper()
        if name not in ('EPSG:4326', 'URN:OGC:DEF:CRS:OGC:1.3:CRS84'):
            raise ValueError('GeoJSON with a projected or unsupported CRS must first be reprojected to WGS84.')
        if coord_system != 'wgs84':
            raise ValueError('GeoJSON CRS declares WGS84 but coord_system is not wgs84.')
    if data.get('type') == 'FeatureCollection':
        data = _select(data.get('features') or [], feature_index)
    if data.get('type') == 'Feature':
        data = data.get('geometry')
    if not isinstance(data, dict):
        raise ValueError('Feature has no geometry.')
    if not data.get('type'):
        return _record_point(data), coord_system, {'kind': 'coordinate_record', 'path': str(path)}
    if data['type'] == 'Point':
        return coordinate(data.get('coordinates')), coord_system, {'kind': 'geojson_point', 'path': str(path)}
    if data['type'] not in ('Polygon', 'MultiPolygon'):
        raise ValueError('Use a Point or a Polygon/MultiPolygon incident footprint, or supply an explicit origin.')
    from shapely.geometry import shape
    geometry = shape(data)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError('Incident geometry is empty or invalid.')
    point = geometry.representative_point()
    return coordinate([point.x, point.y]), coord_system, {
        'kind': 'polygon_representative_point', 'path': str(path),
        'note': 'Representative point of the selected footprint; not necessarily an accessible pickup point.',
    }


def wgs84_to_gcj02_approx(point):
    """Public GCJ distortion model, for display conversion, not authoritative surveying.

    Model reference: https://github.com/wandergis/coordtransform
    API input conversion uses AMap's official conversion endpoint instead.
    """
    lon, lat = coordinate(point)
    if not 72.004 <= lon <= 137.8347 or not .8293 <= lat <= 55.8271:
        return [lon, lat]
    x, y = lon - 105., lat - 35.
    common = sum(20 * math.sin(k * x * math.pi) for k in (6, 2)) * 2 / 3
    north = -100 + 2*x + 3*y + .2*y*y + .1*x*y + .2*math.sqrt(abs(x)) + common
    east = 300 + x + 2*y + .1*x*x + .1*x*y + .1*math.sqrt(abs(x)) + common
    north += sum(a * math.sin(y * math.pi / d) for a, d in ((20,1),(40,3),(160,12),(320,30))) * 2 / 3
    east += sum(a * math.sin(x * math.pi / d) for a, d in ((20,1),(40,3),(150,12),(300,30))) * 2 / 3
    radians = math.radians(lat)
    eccentricity = .006693421622965943
    ellipsoid = 6378245.
    magic = 1 - eccentricity * math.sin(radians)**2
    dy = north * 180 / ((ellipsoid * (1-eccentricity) / magic**1.5) * math.pi)
    dx = east * 180 / ((ellipsoid / math.sqrt(magic)) * math.cos(radians) * math.pi)
    return [lon + dx, lat + dy]


def gcj02_to_wgs84_approx(point):
    target = coordinate(point)
    guess = target[:]
    for _ in range(6):
        shifted = wgs84_to_gcj02_approx(guess)
        error = [shifted[i] - target[i] for i in range(2)]
        guess = [guess[i] - error[i] for i in range(2)]
        if max(map(abs, error)) < 1e-8:
            break
    return guess
