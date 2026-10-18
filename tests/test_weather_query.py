"""Open-Meteo contracts, failure handling and real MCP discovery."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from fastmcp import Client
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("weather_test_tool", ROOT / "agent/tools/WeatherQuery.py")
weather = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weather)


def forecast(days=3, hourly=False):
    data = {
        "latitude": 22.55, "longitude": 114.05, "elevation": 15,
        "timezone": "Asia/Shanghai", "utc_offset_seconds": 28800,
        "current": {"time": "2026-10-18T13:15", "interval": 900},
        "daily": {"time": [f"2026-10-{18 + i:02d}" for i in range(days)]},
    }
    sections = [("current", weather.CURRENT, 1), ("daily", weather.DAILY, days)]
    if hourly:
        data["hourly"] = {"time": ["2026-10-18T13:00", "2026-10-18T14:00"]}
        sections.append(("hourly", weather.HOURLY, 2))
    for name, fields, count in sections:
        data[f"{name}_units"] = {"time": "iso8601"}
        for field in fields:
            value = 61 if field == "weather_code" else 2.5
            data[name][field] = value if name == "current" else [value] * count
            data[f"{name}_units"][field] = "wmo code" if field == "weather_code" else "test unit"
    return data


PLACE = {
    "id": 1795565, "name": "Shenzhen", "latitude": 22.54554, "longitude": 114.0683,
    "country_code": "CN", "country": "China", "admin1": "Guangdong",
    "timezone": "Asia/Shanghai",
}


class WeatherQueryTests(unittest.IsolatedAsyncioTestCase):
    def client_mock(self, handler):
        client_class = httpx.AsyncClient
        transport = httpx.MockTransport(handler)
        return patch.object(weather.httpx, "AsyncClient", side_effect=lambda **kw: client_class(transport=transport, **kw))

    async def test_coordinates_request_units_source_and_descriptions(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json=forecast())

        with self.client_mock(handler):
            result = await weather.query_weather(latitude=22.54554, longitude=114.0683)
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request.url.host, "api.open-meteo.com")
        self.assertNotIn("authorization", request.headers)
        self.assertEqual(request.url.params["wind_speed_unit"], "ms")
        self.assertEqual(request.url.params["timezone"], "auto")
        self.assertNotIn("hourly", request.url.params)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["location"]["latitude"], 22.54554)
        self.assertEqual(result["grid_location"]["latitude"], 22.55)
        self.assertEqual(result["current"]["weather_description"], "Slight rain")
        self.assertEqual(result["current"]["interval"], 900)
        self.assertEqual(result["daily"]["weather_description"], ["Slight rain"] * 3)
        self.assertEqual(result["timezone"], "Asia/Shanghai")
        self.assertEqual(result["utc_offset_seconds"], 28800)
        self.assertEqual(result["sources"][0]["url"], "https://open-meteo.com/")
        self.assertNotIn("summary_path", result)
        json.dumps(result, allow_nan=False)

    async def test_city_geocoding_country_and_hourly(self):
        requests = []

        def handler(request):
            requests.append(request)
            if request.url.host == "geocoding-api.open-meteo.com":
                return httpx.Response(200, json={"results": [PLACE]})
            return httpx.Response(200, json=forecast(days=1, hourly=True))

        with self.client_mock(handler):
            result = await weather.query_weather(" Shenzhen ", country_code=" cn ",
                                                 forecast_days=1, include_hourly=True,
                                                 timezone="Asia/Shanghai")
        self.assertEqual(requests[0].url.params["name"], "Shenzhen")
        self.assertEqual(requests[0].url.params["countryCode"], "CN")
        self.assertEqual(requests[1].url.params["latitude"], str(PLACE["latitude"]))
        self.assertEqual(requests[1].url.params["hourly"], ",".join(weather.HOURLY))
        self.assertEqual(result["location"]["admin1"], "Guangdong")
        self.assertEqual(result["hourly"]["weather_description"], ["Slight rain"] * 2)

    async def test_ambiguous_and_missing_places_never_query_forecast(self):
        for candidates in ([], [PLACE, {**PLACE, "id": 42, "admin1": "Other province"}]):
            requests = []

            def handler(request):
                requests.append(request)
                return httpx.Response(200, json={"results": candidates})

            with self.subTest(candidates=len(candidates)), self.client_mock(handler):
                result = await weather.query_weather("Somewhere")
            self.assertEqual(len(requests), 1)
            self.assertEqual(result["status"], "location_ambiguous" if candidates else "location_not_found")
            self.assertNotIn("current", result)

    async def test_validation_precedes_network(self):
        cases = [
            {}, {"location": " "}, {"location": "A"},
            {"latitude": 22}, {"longitude": 114},
            {"latitude": 91, "longitude": 0}, {"latitude": 0, "longitude": 181},
            {"latitude": float("nan"), "longitude": 0},
            {"latitude": True, "longitude": 0},
            {"location": "Shenzhen", "latitude": 22, "longitude": 114},
            {"latitude": 22, "longitude": 114, "country_code": "CN"},
            {"location": "Shenzhen", "forecast_days": 17},
            {"location": "Shenzhen", "forecast_days": True},
            {"location": "Shenzhen", "forecast_days": 0},
            {"location": "Shenzhen", "country_code": "China"},
            {"location": "Shenzhen", "include_hourly": "yes"},
            {"location": "Shenzhen", "timezone": "not-a-zone"},
        ]
        with patch.object(weather.httpx, "AsyncClient") as client:
            for options in cases:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    await weather.query_weather(**options)
            client.assert_not_called()

    async def test_provider_failures_are_explicit_errors(self):
        responses = [
            httpx.Response(429, json={"error": True}),
            httpx.Response(500, text="down"), httpx.Response(200, text="html"),
            httpx.Response(200, json={"error": True, "reason": "bad input"}),
            httpx.Response(200, json=[]),
        ]
        for response in responses:
            with self.subTest(status=response.status_code), self.client_mock(lambda _: response):
                with self.assertRaises(RuntimeError):
                    await weather.query_weather(latitude=22, longitude=114)
        for failure in (httpx.ReadTimeout, httpx.ConnectError):
            def handler(request):
                raise failure("network failure", request=request)

            with self.subTest(failure=failure), self.client_mock(handler):
                with self.assertRaises(RuntimeError):
                    await weather.query_weather(latitude=22, longitude=114)

    async def test_malformed_geocoding_response_is_not_no_match(self):
        for body in ({"results": {}}, {"results": [None]}, {"results": [{"name": "X"}]}):
            with self.client_mock(lambda _: httpx.Response(200, json=body)):
                with self.assertRaisesRegex(RuntimeError, "geocoding|candidate"):
                    await weather.query_weather("Somewhere")

    async def test_missing_weather_is_not_fabricated_success(self):
        original = forecast()
        cases = []
        for key in ("current", "daily_units", "timezone", "utc_offset_seconds"):
            data = copy.deepcopy(original)
            del data[key]
            cases.append(data)
        data = copy.deepcopy(original)
        data["daily"]["rain_sum"] = [1]
        cases.append(data)
        data = copy.deepcopy(original)
        data["daily"]["time"] = []
        cases.append(data)
        data = copy.deepcopy(original)
        data["current"]["time"] = "not-a-time"
        cases.append(data)
        data = copy.deepcopy(original)
        for field in weather.CURRENT:
            data["current"][field] = None
        cases.append(data)
        for data in cases:
            with self.client_mock(lambda _: httpx.Response(200, json=data)):
                with self.assertRaisesRegex(RuntimeError, "Invalid Open-Meteo weather response"):
                    await weather.query_weather(latitude=22, longitude=114)

    async def test_partial_nulls_remain_unknown_and_non_finite_rejected(self):
        data = forecast()
        data["current"]["rain"] = None
        data["daily"]["precipitation_probability_max"][0] = None
        with self.client_mock(lambda _: httpx.Response(200, json=data)):
            result = await weather.query_weather(latitude=22, longitude=114)
        self.assertIsNone(result["current"]["rain"])
        self.assertIsNone(result["daily"]["precipitation_probability_max"][0])
        data["current"]["rain"] = float("nan")
        with self.client_mock(lambda _: httpx.Response(200, text=json.dumps(data))):
            with self.assertRaisesRegex(RuntimeError, "Invalid current value"):
                await weather.query_weather(latitude=22, longitude=114)

    async def test_real_mcp_discovery_and_http_mocked_query(self):
        with self.client_mock(lambda _: httpx.Response(200, json=forecast())):
            async with Client(weather.mcp, timeout=10) as client:
                discovered = await client.list_tools()
                self.assertEqual([tool.name for tool in discovered], ["query_weather"])
                self.assertIn("latitude", discovered[0].inputSchema["properties"])
                self.assertIn("model estimates", discovered[0].description)
                result = await client.call_tool("query_weather", {"latitude": 22, "longitude": 114})
                self.assertFalse(result.is_error)
                self.assertIn("Slight rain", result.content[0].text)

    async def test_configured_stdio_server_starts_and_rejects_bad_input(self):
        config = json.loads((ROOT / "agent/config_qwen3.json").read_text())
        with tempfile.TemporaryDirectory(prefix="weather-mcp-test-") as temporary:
            arguments = config["mcpServers"]["WeatherQuery"]["args"]
            parameters = StdioServerParameters(
                command=sys.executable, args=arguments, cwd=str(ROOT / "agent"),
            )
            with (Path(temporary) / "stderr.log").open("w") as log:
                async with stdio_client(parameters, errlog=log) as (reader, writer):
                    async with ClientSession(reader, writer) as session:
                        await session.initialize()
                        discovered = await session.list_tools()
                        self.assertEqual([tool.name for tool in discovered.tools], ["query_weather"])
                        result = await session.call_tool("query_weather", {"latitude": 99, "longitude": 0})
                        self.assertTrue(result.isError)
                        self.assertIn("latitude", result.content[0].text)

    def test_all_configs_register_weather_without_extra_dependencies(self):
        for name in ("config.json", "config_qwen3.json", "config_deepseek.json"):
            config = json.loads((ROOT / "agent" / name).read_text())
            server = config["mcpServers"]["WeatherQuery"]
            self.assertEqual(server["args"][0], "tools/WeatherQuery.py")
            self.assertEqual(server["command"], "python")


if __name__ == "__main__":
    unittest.main()
