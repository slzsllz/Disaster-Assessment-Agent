"""Open-Meteo weather query MCP server and CLI (no API key required)."""
import argparse
import asyncio
from datetime import datetime, timezone as utc_timezone
import json
import math
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from fastmcp import FastMCP


mcp = FastMCP()
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
CURRENT = (
    "temperature_2m", "relative_humidity_2m", "apparent_temperature",
    "precipitation", "rain", "weather_code", "cloud_cover",
    "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m",
)
DAILY = (
    "weather_code", "temperature_2m_max", "temperature_2m_min",
    "precipitation_sum", "rain_sum", "precipitation_probability_max",
    "wind_speed_10m_max", "wind_gusts_10m_max",
)
HOURLY = (
    "temperature_2m", "relative_humidity_2m", "precipitation_probability",
    "precipitation", "weather_code", "wind_speed_10m", "wind_gusts_10m",
)
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Depositing rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snowfall", 73: "Moderate snowfall", 75: "Heavy snowfall",
    77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    97: "Heavy thunderstorm", 99: "Thunderstorm with heavy hail",
}
TOOL_DESCRIPTION = """
Query current weather and a 1..16 day forecast from Open-Meteo worldwide.
Use this tool, not model recollection, for current weather or future rainfall/wind.
Provide either a location name (e.g. Shenzhen, Guangdong), optionally country_code
(ISO alpha-2, e.g. CN), OR both WGS84 latitude and longitude. Never fabricate
coordinates; coordinates from a georeferenced upload must first be in WGS84,
not projected coordinates or GCJ-02 map coordinates.
forecast_days includes today in local time (default 3). include_hourly adds hourly
data for the same days; otherwise return current and daily data only.
timezone defaults to auto, resolving the location's local timezone.
If status=location_ambiguous, inspect candidates and resolve using user context,
or ask the user; never silently use an arbitrary candidate. Repeat with coordinates.
location_not_found means no matching location, NOT no weather or no rain.
Return temperatures in Celsius, precipitation in mm, wind in m/s, timestamps,
units, source URL and retrieval time. Report queried place, date and timezone;
missing/null values are unknown, not zero. Attribute the data to Open-Meteo.
Current conditions are weather-model estimates, not guaranteed station observations.
Hourly precipitation refers to the preceding hour; current precipitation uses the
returned current.interval (seconds); daily precipitation_sum is a daily total.
This tool does not query historical weather, official warnings, flood depth, actual
damage, or evacuation orders. A forecast or WMO weather code is NOT an official
warning or a locally calibrated disaster severity classification. Explain uncertainty
when using forecasts as disaster context; do not claim rain proves flooding.
No output files are generated: summarize weather directly, without artifact/file lists.
"""


def _coordinate(value, name, limit):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not -limit <= value <= limit):
        raise ValueError(f"{name} must be a finite number in [-{limit}, {limit}].")
    return float(value)


async def _get_json(client, url, params):
    try:
        response = await client.get(url, params=params)
    except httpx.TimeoutException:
        raise RuntimeError("Open-Meteo request timed out; retry later.") from None
    except httpx.RequestError:
        raise RuntimeError("Cannot connect to Open-Meteo; check server network access.") from None
    if response.status_code == 429:
        raise RuntimeError("Open-Meteo rate limit exceeded (HTTP 429); retry later.")
    if not response.is_success:
        raise RuntimeError(f"Open-Meteo request failed (HTTP {response.status_code}).")
    try:
        data = response.json()
    except ValueError:
        raise RuntimeError("Open-Meteo returned a non-JSON response.") from None
    if not isinstance(data, dict) or data.get("error"):
        raise RuntimeError("Open-Meteo returned an invalid or error response.")
    return data


def _validate_weather(data, forecast_days, include_hourly):
    try:
        _coordinate(data.get("latitude"), "response latitude", 90)
        _coordinate(data.get("longitude"), "response longitude", 180)
        if (not isinstance(data.get("timezone"), str) or not data["timezone"]
                or type(data.get("utc_offset_seconds")) is not int):
            raise ValueError("Missing timezone metadata.")
        sections = [("current", CURRENT), ("daily", DAILY)]
        if include_hourly:
            sections.append(("hourly", HOURLY))
        for name, fields in sections:
            section, units = data.get(name), data.get(f"{name}_units")
            if not isinstance(section, dict) or not isinstance(units, dict):
                raise ValueError(f"Missing {name} data or units.")
            times = [section.get("time")] if name == "current" else section.get("time")
            if not isinstance(times, list) or not times:
                raise ValueError(f"Missing {name} timestamps.")
            for stamp in times:
                if not isinstance(stamp, str):
                    raise ValueError("Invalid timestamp.")
                datetime.fromisoformat(stamp)
            if name == "daily" and len(times) != forecast_days:
                raise ValueError("Incomplete daily forecast.")
            if name == "current" and (
                type(section.get("interval")) is not int or section["interval"] <= 0
            ):
                raise ValueError("Missing current aggregation interval.")
            available = False
            for field in fields:
                if field not in section or not isinstance(units.get(field), str):
                    raise ValueError(f"Missing {name} variable or unit: {field}.")
                values = [section[field]] if name == "current" else section[field]
                if not isinstance(values, list) or len(values) != len(times):
                    raise ValueError(f"Misaligned {name} variable: {field}.")
                for value in values:
                    if value is None:
                        continue
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value)):
                        raise ValueError(f"Invalid {name} value: {field}.")
                    available = True
            if not available:
                raise ValueError(f"All {name} values are unavailable.")
    except (ValueError, OverflowError) as exc:
        raise RuntimeError(f"Invalid Open-Meteo weather response: {exc}") from None


async def query_weather(
    location: Optional[str] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    country_code: Optional[str] = None,
    forecast_days: int = 3,
    include_hourly: bool = False,
    timezone: str = "auto",
) -> dict:
    """Look up local weather by place name or explicit WGS84 coordinates."""
    if type(forecast_days) is not int or not 1 <= forecast_days <= 16:
        raise ValueError("forecast_days must be an integer within [1, 16].")
    if type(include_hourly) is not bool:
        raise ValueError("include_hourly must be a boolean.")
    if not isinstance(timezone, str) or not timezone.strip():
        raise ValueError("timezone must be auto or a valid IANA timezone name.")
    timezone = timezone.strip()
    if timezone != "auto":
        try:
            ZoneInfo(timezone)
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("timezone must be auto or a valid IANA timezone name.") from None
    if country_code is not None:
        if (not isinstance(country_code, str) or len(country_code.strip()) != 2
                or not country_code.strip().isascii() or not country_code.strip().isalpha()):
            raise ValueError("country_code must be an ISO alpha-2 code, e.g. CN.")
        country_code = country_code.strip().upper()
    if location is not None:
        if not isinstance(location, str) or not 2 <= len(location.strip()) <= 200:
            raise ValueError("location must contain 2..200 characters.")
        location = location.strip()
    has_coordinates = latitude is not None or longitude is not None
    if has_coordinates:
        latitude = _coordinate(latitude, "latitude", 90)
        longitude = _coordinate(longitude, "longitude", 180)
        if location or country_code:
            raise ValueError("Use either location/country_code or coordinates, not both.")
    elif not location:
        raise ValueError("Provide location or both latitude and longitude.")

    result = {
        "task": "weather_query", "provider": "Open-Meteo",
        "requested_location": location,
        "sources": [{"title": "Weather data by Open-Meteo", "url": "https://open-meteo.com/"}],
    }
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0), trust_env=False,
    ) as client:
        if not has_coordinates:
            params = {"name": location, "count": 5, "language": "zh", "format": "json"}
            if country_code:
                params["countryCode"] = country_code
            geocoded = await _get_json(client, GEOCODING_URL, params)
            candidates = geocoded.get("results", [])
            if not isinstance(candidates, list):
                raise RuntimeError("Invalid Open-Meteo geocoding response.")
            fields = ("id", "name", "latitude", "longitude", "country", "country_code",
                      "admin1", "admin2", "admin3", "admin4", "timezone")
            places = []
            for item in candidates:
                if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                    raise RuntimeError("Invalid Open-Meteo location candidate.")
                try:
                    _coordinate(item.get("latitude"), "candidate latitude", 90)
                    _coordinate(item.get("longitude"), "candidate longitude", 180)
                except ValueError as exc:
                    raise RuntimeError(f"Invalid Open-Meteo location candidate: {exc}") from None
                places.append({key: item[key] for key in fields if key in item})
            if len(places) != 1:
                return {
                    **result, "status": "location_ambiguous" if places else "location_not_found",
                    "candidates": places,
                    "message": ("Resolve the location using user context or ask for clarification; "
                                "then query with WGS84 coordinates. No weather was queried."),
                    "queried_at": datetime.now(utc_timezone.utc).isoformat(),
                }
            place = places[0]
            latitude, longitude = place["latitude"], place["longitude"]
        else:
            place = {"latitude": latitude, "longitude": longitude}
        params = {
            "latitude": latitude, "longitude": longitude, "timezone": timezone,
            "forecast_days": forecast_days, "current": ",".join(CURRENT),
            "daily": ",".join(DAILY), "temperature_unit": "celsius",
            "wind_speed_unit": "ms", "precipitation_unit": "mm",
        }
        if include_hourly:
            params["hourly"] = ",".join(HOURLY)
        data = await _get_json(client, FORECAST_URL, params)
    _validate_weather(data, forecast_days, include_hourly)
    sections = ("current", "daily", "hourly") if include_hourly else ("current", "daily")
    for name in sections:
        code = data[name]["weather_code"]
        data[name]["weather_description"] = (
            WEATHER_CODES.get(code, "Unknown") if name == "current" else
            [WEATHER_CODES.get(value, "Unknown") for value in code]
        )
    return {
        **result, "status": "success", "location": place,
        "queried_at": datetime.now(utc_timezone.utc).isoformat(),
        "source_url": str(httpx.URL(FORECAST_URL, params=params)),
        "forecast_days": forecast_days,
        "timezone": data["timezone"], "utc_offset_seconds": data["utc_offset_seconds"],
        "grid_location": {key: data.get(key) for key in ("latitude", "longitude", "elevation")},
        **{key: data[key] for name in sections for key in (name, f"{name}_units")},
        "limitations": [
            "Current conditions and forecasts are model estimates, not official warnings or guaranteed station measurements.",
            "Weather grid coordinates may differ from the requested point; this is not parcel-level data.",
            "Null means unavailable, never zero. Forecast uncertainty increases with lead time.",
            "Weather alone cannot determine flood depth, actual damage, or an official disaster response level.",
        ],
    }


mcp.tool(description=TOOL_DESCRIPTION)(query_weather)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temp_dir", help="Accepted for compatibility; this tool writes no files.")
    parser.add_argument("--location")
    parser.add_argument("--latitude", type=float)
    parser.add_argument("--longitude", type=float)
    parser.add_argument("--country-code")
    parser.add_argument("--forecast-days", type=int, default=3)
    parser.add_argument("--include-hourly", action="store_true")
    parser.add_argument("--timezone", default="auto")
    args = parser.parse_args()
    if args.location is not None or args.latitude is not None or args.longitude is not None:
        try:
            result = asyncio.run(query_weather(
                args.location, args.latitude, args.longitude, args.country_code,
                args.forecast_days, args.include_hourly, args.timezone,
            ))
        except (ValueError, RuntimeError) as exc:
            parser.exit(1, f"{exc}\n")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        mcp.run()
