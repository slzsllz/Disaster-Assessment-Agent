"""Offline service routing using the same approximate region as the map UI.

Natural Earth public-domain CHN/TWN/HKG/MAC polygons, copied mechanically from
frontend/chatDisaster/src/utils/chinaBoundary.js. Not an authoritative boundary.
"""

import json
from functools import lru_cache
from pathlib import Path

from agent.route_geometry import coordinate


@lru_cache(maxsize=1)
def _domestic_region():
    from shapely.geometry import shape
    from shapely.prepared import prep

    geometry = json.loads(Path(__file__).with_name('china_boundary.json').read_text())
    return prep(shape(geometry))


def context_provider(point) -> str:
    from shapely.geometry import Point

    lon, lat = coordinate(point)
    return 'amap' if _domestic_region().covers(Point(lon, lat)) else 'arcgis'
