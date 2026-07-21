import argparse
import json
import math
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

try:
    from fastmcp import FastMCP
except Exception as exc:  # noqa: BLE001
    FastMCP = None
    FASTMCP_IMPORT_ERROR = exc
else:
    FASTMCP_IMPORT_ERROR = None


mcp = FastMCP() if FastMCP is not None else None

GEO_CONTEXT_TOOL_DESCRIPTION = """
Query geographic context around a disaster assessment area using ArcGIS REST services.

This tool complements disaster-detection tools. It does not detect disasters; it describes
what is around or inside a georeferenced disaster area, such as address/admin context and
nearby POIs. If ArcGIS Feature Layer URLs are supplied, it can also query configured
building, road, administrative boundary, or other vector layers by spatial intersection.

Parameters:
- raster_path (str | None): Optional georeferenced raster/GeoTIFF path. If provided, its WGS84 bounding box is used.
- bbox (list[float] | None): Optional WGS84 bbox [min_lon, min_lat, max_lon, max_lat].
- center (list[float] | None): Optional WGS84 center [lon, lat]. Used for reverse geocoding and as fallback search center.
- radius_m (float): Search radius in meters when bbox is omitted or too large. Default 1000.
- query_types (list[str] | None): Context types to query. Common values:
  ["hospital", "school", "park", "residential", "building", "road", "shelter", "government"].
- max_results_per_type (int): Maximum places/features returned per type. ArcGIS Places page size is capped at 20.
- feature_layers (list[dict] | None): Optional ArcGIS Feature Layer configs for true spatial queries.
  Each item: {"name": "buildings", "url": "https://.../FeatureServer/0", "out_fields": ["name"]}.

Returns:
- dict: Address/admin context, POI results from ArcGIS Places, optional feature-layer intersection results,
  query extent, limitations, and a saved summary JSON path.

Interpretation guidance:
- ArcGIS Places returns POIs, not complete building footprints or road networks.
- For reliable affected-building/road/admin statistics, pass authoritative Feature Layer URLs
  or use a local spatial database such as PostGIS.
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
    values: dict[str, str] = {}
    if not path.exists():
        return values
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception:
        return values
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _arcgis_token() -> str:
    root_env = _read_env_file(PROJECT_ROOT / ".env")
    frontend_env = _read_env_file(PROJECT_ROOT / "frontend" / "chatDisaster" / ".env.local")
    return (
        os.getenv("ARCGIS_API_KEY")
        or os.getenv("VITE_ARCGIS_API_KEY")
        or root_env.get("ARCGIS_API_KEY")
        or root_env.get("VITE_ARCGIS_API_KEY")
        or frontend_env.get("ARCGIS_API_KEY")
        or frontend_env.get("VITE_ARCGIS_API_KEY")
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
            min_lon, min_lat, max_lon, max_lat = bounds.left, bounds.bottom, bounds.right, bounds.top
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
    return [lon - delta_lon, lat - delta_lat, lon + delta_lon, lat + delta_lat]


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
    token = _arcgis_token()
    if not token:
        raise RuntimeError(
            "ArcGIS API key is not configured. Set ARCGIS_API_KEY or VITE_ARCGIS_API_KEY "
            "in the backend environment, .env, or frontend/chatDisaster/.env.local."
        )

    if raster_path and not bbox:
        bbox = _bbox_from_raster(raster_path)
    if bbox:
        bbox = [float(value) for value in bbox]
        if len(bbox) != 4:
            raise ValueError("bbox must be [min_lon, min_lat, max_lon, max_lat].")
    if center:
        center = [float(value) for value in center]
        if len(center) != 2:
            raise ValueError("center must be [lon, lat].")
    if not center and bbox:
        center = _bbox_center(bbox)
    if not bbox and center:
        bbox = _bbox_from_center_radius(center, radius_m)
    if not bbox or not center:
        raise ValueError("Provide raster_path, bbox, or center.")

    query_types = query_types or DEFAULT_QUERY_TYPES
    places_bbox, bbox_was_limited = _limit_bbox_for_places(bbox, center, radius_m)
    location = _reverse_geocode(center, token)
    places = _query_places(places_bbox, center, query_types, token, max_results_per_type)
    feature_layer_results = _query_feature_layers(
        bbox,
        token,
        feature_layers,
        max_results_per_type,
    )

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
        "source": "ArcGIS Geocoding, ArcGIS Places, optional ArcGIS Feature Layers",
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
        "limitations": [
            "ArcGIS Places provides POIs and place records, not exhaustive building footprints or road networks.",
            "For reliable affected-building, road, or administrative-area statistics, configure authoritative ArcGIS Feature Layers or a local spatial database.",
            "Places service queries are limited to extents no larger than 20,000 meters by 20,000 meters; large raster extents are searched around the center with the requested radius.",
        ],
    }

    out_dir = TEMP_DIR / output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "geo_context_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
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
