import argparse
import json
import math
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import dotenv_values

try:
    from fastmcp import FastMCP
except Exception as exc:  # noqa: BLE001
    FastMCP = None
    FASTMCP_IMPORT_ERROR = exc
else:
    FASTMCP_IMPORT_ERROR = None


mcp = FastMCP() if FastMCP is not None else None

GEO_CONTEXT_TOOL_DESCRIPTION = """
Query geographic context around a disaster assessment area. The backend automatically
uses AMap for domestic centers (including Hong Kong/Macao/Taiwan in the routing data)
and ArcGIS for foreign centers. Do not select a provider or fall back between them.
Routing uses an approximate offline land boundary, not authoritative jurisdiction data.

This tool complements disaster-detection tools. It does not detect disasters; it describes
what is around or inside a georeferenced disaster area, such as address/admin context and
nearby POIs. For foreign queries, if ArcGIS Feature Layer URLs are supplied, it can query configured
building, road, administrative boundary, or other vector layers by spatial intersection.

Parameters:
- raster_path (str | None): Optional georeferenced raster/GeoTIFF with a declared CRS; its WGS84 bounding box is used.
- bbox (list[float] | None): Optional WGS84 bbox [min_lon, min_lat, max_lon, max_lat].
- center (list[float] | None): Optional WGS84 center [lon, lat]. Used for reverse geocoding and as fallback search center.
- radius_m (float): Search radius (1..50000 meters) when bbox is omitted or too large. Default 1000.
- query_types (list[str] | None): Context types to query. Common values:
  ["hospital", "school", "park", "residential", "building", "road", "shelter", "government"].
- max_results_per_type (int): Maximum places/features per type, 1..20. Only the first page is queried.
- feature_layers (list[dict] | None): Optional ArcGIS Feature Layer configs for true spatial queries.
  Each item: {"name": "buildings", "url": "https://.../FeatureServer/0", "out_fields": ["name"]}.
  These are not called for domestic queries; AMap does not supply complete feature layers.

All input coordinates must be WGS84, never raw GCJ02/BD09. Domestic inputs are converted
through AMap's official GPS conversion API. Returned domestic POIs include native GCJ02
and approximate WGS84 coordinates; no survey-grade accuracy is implied.
Provider selection uses the supplied center, or the bbox center. Split cross-border areas
into separate queries; a center-based query does not cover every jurisdiction in a large bbox.
query_types=[] requests address/admin context only. Default queries are bounded samples.

Returns:
- dict: Address/admin context, POI results from ArcGIS Places, optional feature-layer intersection results,
  query extent, limitations, and a saved summary JSON path.

Interpretation guidance:
- Both providers return POI samples, not complete building footprints or road networks.
- Domestic roads are nearby reference points from reverse geocoding, not road polylines.
- Nearby POIs or features within the query extent are not evidence of actual damage,
  flooding, road closure, or loss of service. Describe them as nearby or potentially exposed.
- Raster input and Feature Layer intersection queries use a bounding box, not the actual
  disaster-mask footprint. A bounding-box intersection does not prove disaster-mask intersection.
- Reliable exposure statistics require authoritative feature data and spatial intersection
  with the actual disaster footprint, using separate spatial analysis such as PostGIS.
  Exposure alone does not confirm damage or operational status.
- Places results are useful for situational awareness, nearby facilities, and preliminary impact interpretation.
"""

parser = argparse.ArgumentParser()
parser.add_argument("--temp_dir", type=str)
parser.add_argument("--raster", type=str, default=None)
parser.add_argument("--bbox", type=str, default=None)
parser.add_argument("--center", type=str, default=None)
parser.add_argument("--radius", type=float, default=1000.0)
parser.add_argument("--output", type=str, default="geo_context_query")
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir or "tmp/tmp/out")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from agent.amap_context import query_amap_context
from agent.geo_region import context_provider
from agent.route_geometry import coordinate

GEOCODE_URL = "https://geocode-api.arcgis.com/arcgis/rest/services/World/GeocodeServer/reverseGeocode"
PLACES_WITHIN_EXTENT_URL = (
    "https://places-api.arcgis.com/arcgis/rest/services/places-service/v1/places/within-extent"
)

DEFAULT_QUERY_TYPES = [
    "hospital",
    "school",
    "park",
    "residential",
    "government",
    "shelter",
    "road",
    "building",
]

PLACE_SEARCH_TEXT = {
    "hospital": "hospital",
    "school": "school",
    "park": "park",
    "residential": "residential",
    "community": "residential community",
    "government": "government",
    "shelter": "shelter",
    "emergency": "emergency",
    "fire_station": "fire station",
    "police": "police",
    "road": "road",
    "bridge": "bridge",
    "building": "building",
    "commercial": "commercial",
}


def _read_env_file(path: Path) -> dict[str, str]:
    try:
        return {key: value for key, value in dotenv_values(path).items() if value}
    except OSError:
        return {}


def _arcgis_token() -> str:
    root_env = _read_env_file(PROJECT_ROOT / ".env")
    return (
        os.getenv("ARCGIS_API_KEY")
        or os.getenv("VITE_ARCGIS_API_KEY")
        or root_env.get("ARCGIS_API_KEY")
        or root_env.get("VITE_ARCGIS_API_KEY")
        or ""
    )


def _get_json(url: str, params: dict[str, Any], token: str, timeout: float = 20.0) -> dict:
    clean_params = {k: v for k, v in params.items() if v is not None and v != ""}
    query = urllib.parse.urlencode(clean_params, doseq=True)
    request_url = f"{url}?{query}" if query else url
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(request_url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            body = response.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return {"error": {"message": str(exc), "url": url}}
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return {"error": {"message": "ArcGIS response is not valid JSON", "body_preview": body[:500]}}
    if "error" in data:
        return data
    return data


def _bbox_from_raster(raster_path: str) -> list[float]:
    try:
        import rasterio
        from rasterio.warp import transform_bounds
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Reading raster bounds requires rasterio in the current environment.") from exc

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        if src.crs:
            min_lon, min_lat, max_lon, max_lat = transform_bounds(
                src.crs,
                "EPSG:4326",
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                densify_pts=21,
            )
        else:
            raise ValueError("Raster has no CRS; cannot determine a reliable geographic context or provider.")
    return [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]


def _parse_float_list(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        data = json.loads(value)
        if isinstance(data, list):
            return [float(x) for x in data]
    except Exception:
        pass
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def _bbox_center(bbox: list[float]) -> list[float]:
    return [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]


def _bbox_from_center_radius(center: list[float], radius_m: float) -> list[float]:
    lon, lat = center
    radius_m = max(1.0, float(radius_m))
    delta_lat = radius_m / 111_320.0
    delta_lon = radius_m / max(1.0, 111_320.0 * math.cos(math.radians(lat)))
    return [max(-180., lon - delta_lon), max(-90., lat - delta_lat),
            min(180., lon + delta_lon), min(90., lat + delta_lat)]


def _haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    lon1, lat1 = a
    lon2, lat2 = b
    r = 6_371_000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    h = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(h))


def _limit_bbox_for_places(bbox: list[float], center: list[float], radius_m: float) -> tuple[list[float], bool]:
    width = _haversine_m((bbox[0], center[1]), (bbox[2], center[1]))
    height = _haversine_m((center[0], bbox[1]), (center[0], bbox[3]))
    if 0 < width <= 20_000 and 0 < height <= 20_000:
        return bbox, False
    capped_radius = min(max(float(radius_m), 100.0), 9_900.0)
    return _bbox_from_center_radius(center, capped_radius), True


def _normalize_place(place: dict, center: list[float]) -> dict:
    loc = place.get("location") or {}
    lon = loc.get("x")
    lat = loc.get("y")
    distance_m = None
    if lon is not None and lat is not None:
        distance_m = round(_haversine_m((center[0], center[1]), (float(lon), float(lat))), 1)
    categories = []
    for category in place.get("categories") or []:
        label = category.get("label") or category.get("name")
        if label:
            categories.append(label)
    return {
        "name": place.get("name"),
        "place_id": place.get("placeId"),
        "lon": lon,
        "lat": lat,
        "distance_from_center_m": distance_m,
        "categories": categories,
    }


def _reverse_geocode(center: list[float], token: str) -> dict:
    data = _get_json(
        GEOCODE_URL,
        {
            "f": "json",
            "location": f"{center[0]},{center[1]}",
            "outFields": "*",
            "returnInputLocation": "true",
        },
        token,
    )
    if "error" in data:
        return {"error": data["error"]}
    address = data.get("address") or {}
    return {
        "formatted_address": address.get("LongLabel") or address.get("Match_addr"),
        "match_address": address.get("Match_addr"),
        "address_type": address.get("Addr_type"),
        "place_name": address.get("PlaceName"),
        "neighborhood": address.get("Neighborhood"),
        "district": address.get("District"),
        "city": address.get("City"),
        "subregion": address.get("Subregion"),
        "region": address.get("Region"),
        "country": address.get("CntryName"),
        "country_code": address.get("CountryCode"),
        "structure_type": address.get("StrucType"),
        "structure_detail": address.get("StrucDet"),
        "raw_address": address,
    }


def _query_places(
    bbox: list[float],
    center: list[float],
    query_types: list[str],
    token: str,
    max_results_per_type: int,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    page_size = max(1, min(int(max_results_per_type), 20))
    for query_type in query_types:
        search_text = PLACE_SEARCH_TEXT.get(query_type, query_type)
        if not search_text or len(search_text) < 3:
            continue
        data = _get_json(
            PLACES_WITHIN_EXTENT_URL,
            {
                "f": "json",
                "xmin": bbox[0],
                "ymin": bbox[1],
                "xmax": bbox[2],
                "ymax": bbox[3],
                "searchText": search_text,
                "pageSize": page_size,
            },
            token,
        )
        if "error" in data:
            results[query_type] = {"error": data["error"], "items": []}
            continue
        places = [_normalize_place(place, center) for place in data.get("results") or []]
        results[query_type] = {
            "search_text": search_text,
            "count": len(places),
            "items": places,
        }
    return results


def _feature_layer_configs(feature_layers: list[dict] | None) -> list[dict]:
    configs = list(feature_layers or [])
    env_raw = os.getenv("ARCGIS_FEATURE_LAYERS_JSON") or ""
    if env_raw:
        try:
            env_layers = json.loads(env_raw)
            if isinstance(env_layers, list):
                configs.extend(item for item in env_layers if isinstance(item, dict))
        except Exception:
            pass
    return configs


def _query_feature_layers(
    bbox: list[float],
    token: str,
    feature_layers: list[dict] | None,
    max_results_per_type: int,
) -> dict[str, Any]:
    queried: dict[str, Any] = {}
    envelope = {
        "xmin": bbox[0],
        "ymin": bbox[1],
        "xmax": bbox[2],
        "ymax": bbox[3],
        "spatialReference": {"wkid": 4326},
    }
    for layer in _feature_layer_configs(feature_layers):
        name = str(layer.get("name") or layer.get("id") or "feature_layer")
        url = str(layer.get("url") or "").rstrip("/")
        if not url:
            queried[name] = {"error": "Missing Feature Layer query URL.", "items": []}
            continue
        query_url = url if url.endswith("/query") else f"{url}/query"
        out_fields = layer.get("out_fields") or layer.get("outFields") or ["*"]
        if isinstance(out_fields, list):
            out_fields = ",".join(str(field) for field in out_fields)
        data = _get_json(
            query_url,
            {
                "f": "json",
                "where": layer.get("where") or "1=1",
                "geometry": json.dumps(envelope),
                "geometryType": "esriGeometryEnvelope",
                "inSR": 4326,
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": out_fields,
                "returnGeometry": str(bool(layer.get("return_geometry", False))).lower(),
                "resultRecordCount": max(1, min(int(max_results_per_type), 200)),
                "token": token,
            },
            token,
        )
        if "error" in data:
            queried[name] = {"error": data["error"], "items": []}
            continue
        features = data.get("features") or []
        queried[name] = {
            "count": len(features),
            "exceeded_transfer_limit": bool(data.get("exceededTransferLimit")),
            "items": [
                {
                    "attributes": feature.get("attributes") or {},
                    "geometry": feature.get("geometry") if layer.get("return_geometry", False) else None,
                }
                for feature in features
            ],
        }
    return queried


def query_geo_context(
    raster_path: str | None = None,
    bbox: list[float] | None = None,
    center: list[float] | None = None,
    radius_m: float = 1000.0,
    query_types: list[str] | None = None,
    max_results_per_type: int = 10,
    feature_layers: list[dict] | None = None,
    output_dir: str = "geo_context_query",
) -> dict:
    if isinstance(radius_m, bool) or not isinstance(radius_m, (float, int)) or not math.isfinite(radius_m) or not 1 <= radius_m <= 50_000:
        raise ValueError("radius_m must be a finite number within [1, 50000].")
    if isinstance(max_results_per_type, bool) or not isinstance(max_results_per_type, int) or not 1 <= max_results_per_type <= 20:
        raise ValueError("max_results_per_type must be an integer within [1, 20].")
    if feature_layers is not None and (not isinstance(feature_layers, list) or any(not isinstance(layer, dict) for layer in feature_layers)):
        raise ValueError("feature_layers must be a list of layer configuration objects.")
    if query_types is None:
        query_types = list(DEFAULT_QUERY_TYPES)
    if not isinstance(query_types, list) or any(not isinstance(name, str) or name not in PLACE_SEARCH_TEXT for name in query_types):
        raise ValueError("query_types must contain only the documented context categories.")
    query_types = list(dict.fromkeys(query_types))
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("output_dir must be a non-empty relative directory.")
    out_dir = (TEMP_DIR / output_dir).resolve()
    if Path(output_dir).is_absolute() or not out_dir.is_relative_to(TEMP_DIR.resolve()):
        raise ValueError("output_dir must be relative and inside the tool temp directory.")

    if raster_path and bbox is None:
        bbox = _bbox_from_raster(raster_path)
    if bbox is not None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat].")
        west, south = coordinate(bbox[:2])
        east, north = coordinate(bbox[2:])
        if west >= east or south >= north:
            raise ValueError("bbox must have increasing bounds; split antimeridian-spanning areas.")
        bbox = [west, south, east, north]
    selection_basis = 'explicit_center' if center is not None else 'bbox_center'
    if center is not None:
        center = coordinate(center)
    if center is None and bbox is not None:
        center = _bbox_center(bbox)
    if center is None:
        raise ValueError("Provide raster_path, bbox, or center.")
    if bbox is None:
        bbox = _bbox_from_center_radius(center, radius_m)
    if not bbox[0] <= center[0] <= bbox[2] or not bbox[1] <= center[1] <= bbox[3]:
        raise ValueError("center must lie inside bbox; do not mix unrelated locations.")

    provider = context_provider(center)
    places_bbox, bbox_was_limited = _limit_bbox_for_places(bbox, center, radius_m)
    limitations = [
        "POIs are bounded samples, not complete building footprints, road networks, or evidence of damage/service availability.",
        "An image bbox or nearby POI does not imply intersection with the actual disaster mask; verify authoritative data and actual footprints separately.",
        "Provider selection uses the query center and approximate offline land polygons; coastal, border and small-island routing may be imperfect. Split cross-border queries.",
        "Large extents are limited to a center-based sample no larger than 20 km by 20 km, not full-area coverage.",
    ]
    native = {}
    if provider == 'amap':
        key = os.getenv('AMAP_WEB_KEY') or _read_env_file(PROJECT_ROOT / '.env').get('AMAP_WEB_KEY', '')
        if not key.strip():
            raise RuntimeError("Domestic geographic context requires AMAP_WEB_KEY; ArcGIS is not used as a fallback.")
        location, places, native = query_amap_context(
            center, places_bbox, query_types, max_results_per_type, radius_m, key.strip(),
        )
        feature_layer_results = {}
        source = 'AMap reverse geocoding and POI search'
        limitations.extend([
            "AMap input conversion is official WGS84-to-GCJ02; returned WGS84 POI coordinates are approximate inverses, not survey-grade.",
            "Roads are reference points within at most 3000 m of the center, not road geometry or the entire bbox road network.",
            "ArcGIS Feature Layers are not called for domestic queries, including layers configured in the environment.",
        ])
        if feature_layers:
            limitations.append("Requested feature_layers were not executed: domestic queries use AMap, which does not provide these vector layers.")
    else:
        token = _arcgis_token()
        if not token:
            raise RuntimeError("Foreign geographic context requires ARCGIS_API_KEY; AMap is not used as a fallback.")
        location = _reverse_geocode(center, token)
        places = _query_places(places_bbox, center, query_types, token, max_results_per_type)
        feature_layer_results = _query_feature_layers(bbox, token, feature_layers, max_results_per_type)
        source = 'ArcGIS Geocoding, ArcGIS Places, optional ArcGIS Feature Layers'

    operations = [location, *places.values(), *feature_layer_results.values()]
    errors = [item['error'] for item in operations if isinstance(item, dict) and item.get('error')]
    status = 'error' if len(errors) == len(operations) else 'partial' if errors or (provider == 'amap' and feature_layers) else 'success'

    place_counts = {
        key: value.get("count", 0) if isinstance(value, dict) else 0
        for key, value in places.items()
    }
    feature_counts = {
        key: value.get("count", 0) if isinstance(value, dict) else 0
        for key, value in feature_layer_results.items()
    }
    result = {
        "task": "geo_context_query",
        "status": status,
        "provider": provider,
        "source": source,
        "provider_selection": {"basis": selection_basis, "center_wgs84": center},
        "coordinate_systems": {"input": "WGS84", "service": "GCJ02" if provider == 'amap' else 'WGS84', **native},
        "input": {
            "raster_path": raster_path,
            "bbox": bbox,
            "center": center,
            "radius_m": radius_m,
            "query_types": query_types,
            "max_results_per_type": max_results_per_type,
        },
        "query_extent": {
            "bbox": bbox,
            "places_bbox": places_bbox,
            "places_bbox_was_limited_to_service_max_extent": bbox_was_limited,
        },
        "location": location,
        "nearby_places": places,
        "intersecting_feature_layers": feature_layer_results,
        "summary": {
            "place_counts_by_type": place_counts,
            "feature_counts_by_layer": feature_counts,
            "location_label": location.get("formatted_address") if isinstance(location, dict) else None,
        },
        "limitations": limitations,
    }
    if errors:
        result['errors'] = errors

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"geo_context_summary_{uuid4().hex}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    result["summary_path"] = str(summary_path)
    return result


if mcp is not None:
    mcp.tool(description=GEO_CONTEXT_TOOL_DESCRIPTION)(query_geo_context)


if __name__ == "__main__":
    parsed_bbox = _parse_float_list(args.bbox)
    parsed_center = _parse_float_list(args.center)
    if args.raster or parsed_bbox or parsed_center:
        print(json.dumps(
            query_geo_context(
                raster_path=args.raster,
                bbox=parsed_bbox,
                center=parsed_center,
                radius_m=args.radius,
                output_dir=args.output,
            ),
            indent=2,
            ensure_ascii=False,
        ))
    else:
        if mcp is None:
            raise FASTMCP_IMPORT_ERROR
        mcp.run()
