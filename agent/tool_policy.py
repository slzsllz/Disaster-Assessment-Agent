"""Reviewed tools exposed to the disaster-assessment agent by default.

MCP registration alone is not an endorsement. Keep this allowlist narrow: a tool
must contribute to disaster detection, remote-sensing preparation, or measurable
impact assessment, and its required runtime must be present on this server.
"""

from __future__ import annotations

from typing import Any, Sequence


CURATED_TOOLS_BY_SERVER: dict[str, tuple[str, ...]] = {
    # Bundled hazard models with their required weights and imports present.
    "DamageAssessment": ("assess_building_damage",),
    "FloodSegmentation": ("extract_flood_inundation",),
    "FireBurnedAreaChange": ("detect_fire_burned_area_change",),
    "OilSpillSegmentation": ("extract_oil_spill_area",),
    "AlgalBloomDetection": ("detect_algal_bloom",),
    "GeoAI": ("geoai_object_detection", "geoai_semantic_segmentation"),
    # Hazard trends and spatial hotspots; omit generic ACF/spike helpers.
    "Analysis": (
        "compute_linear_trend", "mann_kendall_test", "sens_slope",
        "detect_change_points", "getis_ord_gi_star",
        "analyze_hotspot_direction",
    ),
    # Vegetation, water, fire, snow and drought remote-sensing indices.
    "Index": (
        "calculate_batch_ndvi", "calculate_batch_ndwi", "calculate_batch_ndbi",
        "calculate_batch_evi", "calculate_batch_nbr", "calculate_batch_fvc",
        "calculate_batch_wri", "calculate_batch_ndti", "calculate_batch_frp",
        "calculate_batch_ndsi", "calc_extreme_snow_loss_percentage_from_binary_map",
        "compute_tvdi",
    ),
    # Heat, drought/soil moisture and water-quality measurements.
    "Inversion": (
        "lst_single_channel", "modis_day_night_lst",
        "dual_polarization_ratio", "calculate_water_turbidity_ntu",
    ),
    # Mask preparation and simple raster measurement.
    "Perception": ("threshold_segmentation", "count_above_threshold"),
    "Statistics": (
        "calc_batch_image_hotspot_percentage", "calc_batch_fire_pixels",
        "calculate_threshold_ratio", "calculate_intersection_percentage",
        "calculate_multi_band_threshold_ratio", "count_pixels_satisfying_conditions",
        "calculate_band_mean_by_condition", "get_percentile_value_from_image",
    ),
}

CURATED_TOOL_NAMES = frozenset(
    name for names in CURATED_TOOLS_BY_SERVER.values() for name in names
)


def select_curated_tools(tools: Sequence[Any]) -> list[Any]:
    """Prevent unreviewed MCP tools from entering the router or agent graph."""
    selected = [tool for tool in tools if tool.name in CURATED_TOOL_NAMES]
    names = [tool.name for tool in selected]
    if len(names) != len(set(names)):
        raise ValueError("Curated MCP tool names must be unique")
    missing = CURATED_TOOL_NAMES.difference(names)
    if missing:
        raise RuntimeError(f"Curated MCP tools unavailable: {', '.join(sorted(missing))}")
    return selected
