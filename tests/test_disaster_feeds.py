"""Official-feed contracts: spatial/time filters, provenance and honest coverage."""

import json
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta, timezone

import requests
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agent.disaster_feeds as feeds
from agent.review import audit_evidence, gather_review_evidence
from agent.tool_policy import CURATED_TOOLS_BY_SERVER
from agent.tool_router import ToolRouter
from agent.web_search import search_result_references, web_references

START = "2026-10-07T00:00:00Z"
END = "2026-10-09T00:00:00Z"


def usgs_feature(event_id="us-example", lon=104.65, lat=28.5, magnitude=3):
    return {"type": "Feature", "id": event_id,
            "geometry": {"type": "Point", "coordinates": [lon, lat, 5]},
            "properties": {"time": 1791367200000, "updated": 1791367300000,
                           "mag": magnitude, "magType": "ml", "status": "reviewed", "alert": None,
                           "title": "M 3.0 — test earthquake",
                           "url": "https://earthquake.usgs.gov/earthquakes/eventpage/" + event_id}}


def gdacs_feature(event_id=1001335, lon=104.65, lat=28.5, start=START, end=END, kind="FL"):
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"eventtype": kind, "eventid": event_id, "episodeid": 6,
                           "name": "Official test alert", "fromdate": start, "todate": end,
                           "datemodified": END, "alertlevel": "Orange", "country": "Test",
                           "severitydata": {"severity": 12, "severityunit": "test_unit", "severitytext": "reported severity"},
                           "url": {"report": f"https://www.gdacs.org/report.aspx?eventtype={kind}&eventid={event_id}&episodeid=6"}}}


def collection(features):
    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


CAP = b'''<alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">
 <identifier>official-test-1</identifier><sender>official</sender><sent>2026-10-08T09:00:00+08:00</sent>
 <status>Actual</status><scope>Public</scope><msgType>Update</msgType>
 <info><language>zh-CN</language><event>Flood</event><headline>Official flood warning</headline>
 <severity>Severe</severity><urgency>Immediate</urgency><certainty>Likely</certainty>
 <expires>2026-10-09T09:00:00+08:00</expires><web>https://www.12379.cn/alert/test-1</web>
 <area><areaDesc>Test administrative area</areaDesc><polygon>28,104 29,104 29,105 28,104</polygon></area></info></alert>'''


class DisasterFeedTests(unittest.TestCase):
    def query(self, provider="usgs", **kwargs):
        return feeds.query_disaster_feed(provider, start_time=START, end_time=END, **kwargs)

    def test_invalid_queries_fail_before_network_and_keep_envelope(self):
        cases = [{"provider": "invented"}, {"provider": []}, {"count": 0}, {"count": True},
                 {"bbox": [1, 2, 3]}, {"bbox": [0, -91, 10, 30]}, {"bbox": [0, 20, 0, 30]},
                 {"bbox": [float("nan"), 0, 10, 10]}, {"bbox": [False, 0, 10, 10]},
                 {"event_type": "tsunami"}, {"event_type": "wildfire"},
                 {"min_magnitude": 11}, {"min_magnitude": float("inf")},
                 {"start_time": "invalid"}, {"start_time": END, "end_time": START},
                 {"start_time": "2020-01-01", "end_time": END}]
        with patch.object(feeds, "_get") as get:
            for case in cases:
                with self.subTest(case=case):
                    result = feeds.query_disaster_feed(**case)
                    self.assertFalse(result["success"])
                    self.assertEqual(result["error_code"], "invalid_query")
                    self.assertTrue({"success", "provider", "retrieved_at", "results"} <= result.keys())
                    self.assertEqual(result["results"], [])
            get.assert_not_called()

    def test_usgs_fdsn_filters_and_structured_units_reuse_search_citations(self):
        with patch.object(feeds, "_get", return_value=collection([usgs_feature()])) as get:
            result = self.query(bbox=[100, 20, 110, 30], min_magnitude=2.5, count=5)
        self.assertTrue(result["success"])
        params = get.call_args.kwargs["params"]
        self.assertEqual((params["minlongitude"], params["maxlatitude"]), (100, 30))
        self.assertEqual(params["minmagnitude"], 2.5)
        self.assertEqual(params["limit"], 6)
        data = result["results"][0]["data"]
        self.assertEqual(data["depth_km"], 5)
        self.assertEqual(data["magnitude"], 3)
        self.assertTrue(data["event_time"].endswith("Z"))
        self.assertIsNone(result["results"][0]["published_at"])
        self.assertEqual(len(search_result_references(result)), 1)
        self.assertNotIn("casualties", data)

    def test_usgs_antimeridian_and_limit_have_no_false_total(self):
        features = [usgs_feature(str(i), lon=179 if i % 2 else -179) for i in range(3)]
        with patch.object(feeds, "_get", return_value=collection(features)) as get:
            result = self.query(bbox=[170, 20, -170, 30], count=2)
        self.assertEqual(get.call_args.kwargs["params"]["maxlongitude"], 190)
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(result["truncated"])
        self.assertTrue(result["warnings"])

    def test_schema_failure_and_network_failure_are_not_empty_success(self):
        for body in (b"<html>login required</html>", b'{"error": "rate limited"}', collection([{}])):
            with patch.object(feeds, "_get", return_value=body):
                self.assertFalse(self.query()["success"])
        with patch.object(feeds, "_get", side_effect=requests.HTTPError("secret-token-url")):
            result = self.query()
            self.assertFalse(result["success"])
            self.assertNotIn("secret-token", json.dumps(result))
        with patch.object(feeds, "_get", return_value=collection([])):
            result = self.query()
            self.assertTrue(result["success"])
            self.assertEqual(result["results"], [])

    def test_time_and_bbox_filters_drop_unrelated_records_even_if_upstream_ignores_query(self):
        good = usgs_feature()
        outside = usgs_feature("outside", lon=0, lat=0)
        old = usgs_feature("old")
        old["properties"]["time"] = 0
        below = usgs_feature("small", magnitude=1)
        with patch.object(feeds, "_get", return_value=collection([good, outside, old, below])):
            result = self.query(bbox=[100, 20, 110, 30], min_magnitude=2)
        self.assertEqual([row["event_id"] for row in result["results"]], ["us-example"])

    def test_gdacs_preserves_native_severity_and_interval_overlap(self):
        feature = gdacs_feature(start="2026-09-01T00:00:00Z")
        with patch.object(feeds, "_get", return_value=collection([feature])) as get:
            result = self.query("gdacs", event_type="flood", bbox=[100, 20, 110, 30])
        self.assertTrue(result["success"])
        self.assertEqual(get.call_args.kwargs["params"]["eventlist"], "FL")
        data = result["results"][0]["data"]
        self.assertEqual(data["alert_level"], "Orange")
        self.assertEqual(data["severity_unit"], "test_unit")
        self.assertNotIn("flood_area", data)
        self.assertEqual(result["coverage"]["spatial_filter"], "representative_point_not_impact_footprint")

    def test_gdacs_bbox_scan_pages_and_limit_are_explicit(self):
        outside = [gdacs_feature(i, lon=0, lat=0) for i in range(100)]
        with patch.object(feeds, "_get", side_effect=[collection(outside), collection([gdacs_feature()])]) as get:
            result = self.query("gdacs", bbox=[100, 20, 110, 30])
        self.assertEqual(get.call_count, 2)
        self.assertEqual(len(result["results"]), 1)
        with patch.object(feeds, "_get", return_value=collection(outside)) as get:
            result = self.query("gdacs", bbox=[100, 20, 110, 30])
        self.assertEqual(get.call_count, 5)
        self.assertTrue(result["success"])
        self.assertTrue(result["truncated"])
        self.assertEqual(result["results"], [])

    def test_cenc_uses_current_official_schema_and_beijing_time(self):
        rows = [{"id": "CD.20261007185908.494", "time": "2026-10-07 18:56:00", "latitude": 28.5,
                 "longitude": 104.65, "depth": 5, "magnitude": 3, "location": "四川宜宾市高县"}]
        with patch.object(feeds, "_get", return_value=json.dumps(rows).encode()) as get:
            result = self.query("cenc", bbox=[100, 20, 110, 30])
        self.assertEqual(get.call_args.args[0], "https://www.ceic.ac.cn/data/data.json")
        self.assertEqual(result["results"][0]["data"]["event_time"], "2026-10-07T10:56:00Z")
        self.assertFalse(result["coverage"]["historical_completeness"])
        self.assertTrue(result["warnings"])

    def firms_csv(self, date, rows=1):
        header = "latitude,longitude,acq_date,acq_time,satellite,confidence,frp,daynight\n"
        return (header + "".join(f"28.5,{104.6+i/1000},{date},0013,N,nominal,0.82,N\n" for i in range(rows))).encode()

    def test_firms_keyless_csv_filters_points_and_preserves_frp_mw(self):
        now = datetime.now(timezone.utc)
        with patch.dict(os.environ, {"FIRMS_MAP_KEY": ""}), patch.object(
            feeds, "_get", return_value=self.firms_csv(now.date().isoformat(), 4)
        ) as get:
            result = feeds.query_disaster_feed("firms", bbox=[100, 20, 110, 30], count=2,
                                                start_time=now.date().isoformat(),
                                                end_time=(now + timedelta(days=1)).date().isoformat())
        self.assertTrue(result["success"])
        self.assertEqual(len(result["results"]), 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(get.call_args.args[0], feeds.FIRMS_CSV_URL)
        self.assertEqual(get.call_args.kwargs["max_response_bytes"], 32 * 1024 * 1024)
        data = result["results"][0]["data"]
        self.assertEqual(data["frp_mw"], 0.82)
        self.assertEqual(data["observation_type"], "thermal_anomaly")
        self.assertEqual(len(search_result_references(result)), 2)

    def test_firms_historical_query_requires_key_and_does_not_leak_it(self):
        with patch.dict(os.environ, {"FIRMS_MAP_KEY": ""}), patch.object(feeds, "_get") as get:
            result = feeds.query_disaster_feed("firms", start_time="2020-01-01", end_time="2020-01-02")
            self.assertFalse(result["success"])
            get.assert_not_called()
        key = "secret-test-map-key"
        with patch.dict(os.environ, {"FIRMS_MAP_KEY": key}), patch.object(
            feeds, "_get", return_value=self.firms_csv("2026-10-08")
        ) as get:
            result = self.query("firms", bbox=[100, 20, 110, 30])
            self.assertTrue(result["success"])
            self.assertIn(key, get.call_args.args[0])
            self.assertNotIn(key, json.dumps(result))
        with patch.dict(os.environ, {"FIRMS_MAP_KEY": key}), patch.object(
            feeds, "_get", side_effect=requests.HTTPError("https://official/" + key)
        ):
            self.assertNotIn(key, json.dumps(self.query("firms")))

    def test_bbox_empty_match_with_a_bad_row_is_still_a_valid_empty_query(self):
        features = [gdacs_feature(lon=0, lat=0), {}]
        with patch.object(feeds, "_get", return_value=collection(features)):
            result = self.query("gdacs", bbox=[100, 20, 110, 30])
        self.assertTrue(result["success"])
        self.assertEqual(result["results"], [])
        self.assertEqual(result["coverage"]["invalid_records"], 1)

    def test_china_cap_requires_real_official_endpoint_and_preserves_message_type(self):
        with patch.dict(os.environ, {"CHINA_ALERTS_CAP_URL": ""}), patch.object(feeds, "_get") as get:
            self.assertFalse(self.query("china_alerts")["success"])
            get.assert_not_called()
        with patch.dict(os.environ, {"CHINA_ALERTS_CAP_URL": "http://127.0.0.1/cap"}), patch.object(feeds, "_get") as get:
            self.assertFalse(self.query("china_alerts")["success"])
            get.assert_not_called()
        with patch.dict(os.environ, {"CHINA_ALERTS_CAP_URL": "https://www.12379.cn/cap.xml"}), patch.object(
            feeds, "_get", return_value=CAP
        ) as get:
            result = self.query("china_alerts")
            self.assertTrue(result["success"])
            self.assertTrue(get.call_args.kwargs["public_only"])
            data = result["results"][0]["data"]
            self.assertEqual(data["message_type"], "Update")
            self.assertEqual(data["alert_level"], "Severe")
            self.assertEqual(data["event_time"], "2026-10-08T01:00:00Z")
            self.assertNotIn("latitude", data)
            self.assertFalse(self.query("china_alerts", bbox=[100, 20, 110, 30])["success"])

    def test_cap_rejects_html_entities_and_exercises(self):
        endpoint = {"CHINA_ALERTS_CAP_URL": "https://www.12379.cn/cap.xml"}
        for body in (b"<html>Login</html>", b'<!DOCTYPE x [<!ENTITY a "test">]><alert/>'):
            with patch.dict(os.environ, endpoint), patch.object(feeds, "_get", return_value=body):
                self.assertFalse(self.query("china_alerts")["success"])
        with patch.dict(os.environ, endpoint), patch.object(feeds, "_get", return_value=CAP.replace(b"Actual", b"Exercise")):
            result = self.query("china_alerts")
            self.assertTrue(result["success"])
            self.assertEqual(result["results"], [])

    def test_references_and_review_use_official_fields_but_link_verification_still_needs_body(self):
        with patch.object(feeds, "_get", return_value=collection([gdacs_feature()])):
            result = self.query("gdacs")
        messages = [AIMessage(content="", tool_calls=[{"name": "disaster_feed", "args": {}, "id": "feed-1"}]),
                    ToolMessage(content=[{"type": "text", "text": json.dumps(result)}], tool_call_id="feed-1")]
        references = web_references(messages)
        self.assertEqual(references[0]["evidence_type"], "official_record")
        self.assertEqual(references[0]["structured_data"]["severity"], 12)
        evidence, issues = gather_review_evidence({"messages": messages}, [], references, "最新洪水预警")
        self.assertNotIn("web_body_missing", {issue["code"] for issue in issues})
        audit = next(item for item in audit_evidence(evidence) if item["kind"] == "web")
        self.assertEqual(audit["structured_data"]["alert_level"], "Orange")
        _, issues = gather_review_evidence({"messages": messages}, [], references,
                                          "打开这个链接核实 https://www.gdacs.org/report.aspx")
        self.assertIn("web_body_missing", {issue["code"] for issue in issues})
        result["results"][0]["url"] = "https://unofficial.example/alert"
        self.assertEqual(feeds.feed_result_references(result), [])


class DisasterFeedRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_registration_and_fallback_route_official_sources(self):
        names = ("disaster_feed", "web_fetch", "web_search", "assess_building_damage", "extract_flood_inundation", "detect_fire_burned_area_change")
        tools = []
        for name in names:
            tool = Mock()
            tool.name, tool.description = name, name
            tools.append(tool)
        llm = Mock()
        llm.ainvoke.side_effect = RuntimeError("offline")
        router = ToolRouter(llm, tools)
        for question in ("查询官方洪水预警", "用 USGS 查询地震", "获取中国地震台网目录", "最新 FIRMS 火点"):
            with self.subTest(question=question):
                selected = await router.select([HumanMessage(content=question)])
                self.assertIn("disaster_feed", [tool.name for tool in selected])
        self.assertEqual(CURATED_TOOLS_BY_SERVER["DisasterFeed"], ("disaster_feed",))
        config = json.loads((Path(__file__).resolve().parents[1] / "agent/config.json").read_text())
        self.assertIn("tools/DisasterFeed.py", config["mcpServers"]["DisasterFeed"]["args"])
        selected = await router.select([HumanMessage(content="用 GDACS 官方通报与洪水淹没遥感影像互证")])
        self.assertTrue({"disaster_feed", "extract_flood_inundation"} <= {tool.name for tool in selected})


if __name__ == "__main__":
    unittest.main()
