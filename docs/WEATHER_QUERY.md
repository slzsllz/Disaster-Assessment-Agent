# Open-Meteo Weather Tool

`agent/tools/WeatherQuery.py` exposes the MCP tool `query_weather`. It is enabled
in `agent/config_qwen3.json`, `agent/config_deepseek.json` and `agent/config.json`.
Restart the Python backend to load the new tool. No frontend changes, model
weights, additional Python dependencies, or API key are required.

## Queries

- Place name: `query_weather(location="Shenzhen, Guangdong", country_code="CN")`.
- Coordinates: `query_weather(latitude=22.54554, longitude=114.0683)`.
- Forecast: `forecast_days=1..16`, including today (default 3).
- Hourly details: `include_hourly=True` (default false).
- Timezone: `timezone="auto"` by default, or an IANA name such as `Asia/Shanghai`.

Use place name/country or coordinates, not both. Coordinates must be WGS84.
Place lookup returns candidates without fetching weather when a name is
ambiguous. Resolve the place using user context or ask the user, then query its
coordinates. No match is distinguished from network/provider failure.

Example chat requests:

- "查询深圳未来三天天气，重点说明降水和风速。"
- "查询北纬 22.54554、东经 114.0683 的天气，并给出逐小时降水预报。"

## Results And Limits

Current conditions include temperature, apparent temperature, humidity,
precipitation, rain, weather code, cloud cover, wind direction/speed/gusts.
Daily forecasts include temperature extremes, precipitation/rain totals,
maximum precipitation probability and maximum wind/gust speed.
Optional hourly forecasts include temperature, humidity, precipitation amount
and probability, weather code, wind speed and gusts.

Results retain provider units, local timestamps, UTC offset, current aggregation
interval, query time, requested coordinates, weather grid coordinates, source
URL and WMO weather descriptions. Units are Celsius, mm and m/s. Missing values
remain null, not zero. No files are generated or offered for download, and this
tool alone does not trigger artifact review.

Current conditions are **weather-model estimates**, not guaranteed station
observations. Hourly precipitation is the preceding-hour total; current
precipitation uses the returned interval; daily precipitation is a daily total.
Forecasts are not historical-image-date observations, official warnings, flood
depth measurements or disaster response classifications. Longer-range forecasts
are more uncertain. Use weather as supporting disaster context, not proof of
flooding, damage, or a reason to issue evacuation orders.

## Verification

```bash
python -m unittest discover -s tests -p 'test_weather_query.py' -v
python agent/tools/WeatherQuery.py --latitude 22.54554 --longitude 114.0683 --forecast-days 3
python agent/tools/WeatherQuery.py --location 'Shenzhen, Guangdong' --country-code CN
```

The tests mock HTTP responses, including rate limits, timeouts, null values,
malformed responses and ambiguous locations; MCP discovery/invocation and a real
stdio subprocess are also tested without external API calls.

## Sources And Licence

- [Forecast API](https://open-meteo.com/en/docs)
- [Geocoding API](https://open-meteo.com/en/docs/geocoding-api)
- [Terms](https://open-meteo.com/en/terms)

Attribute weather data to Open-Meteo in answers. The public free endpoint is for
non-commercial use, with usage limits and CC BY 4.0 attribution requirements.
Commercial deployment requires an appropriate subscription; this integration
uses the public endpoints, not the paid customer endpoints. Open-Meteo receives
the queried place/coordinates, not uploaded file contents or the chat history.
