import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from agent.review import gather_review_evidence, parse_review_response, review_required
from backend_api import (execute_review, extract_artifact_selection, review_image_data_url,
                         review_model_config)


def response_with_tool(name, payload):
    return {"messages": [
        AIMessage(content="", tool_calls=[{"id": "call-1", "name": name, "args": {}}]),
        ToolMessage(content=json.dumps(payload), name=name, tool_call_id="call-1"),
        AIMessage(content="<Conclusion>原始结论</Conclusion>"),
    ]}


class FakeHandle:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def review_answer_and_artifacts(self, *args):
        self.calls += 1
        return self.result


class ReviewTests(unittest.TestCase):
    def test_deterministic_damage_mismatch_blocks_before_llm(self):
        response = response_with_tool("assess_building_damage", {
            "building_total": 10, "no_damage": 2, "minor_damage": 1,
            "major_damage": 1, "destroyed": 1, "damaged": 3,
        })
        handle = FakeHandle("{}")
        answer, images, files, review = execute_review(
            handle, "评估损毁", response, "原始结论", [], [], [],
        )
        self.assertEqual(review["status"], "blocked")
        self.assertIn("统计不一致", answer)
        self.assertEqual(handle.calls, 0)
        self.assertEqual((images, files), ([], []))

    def test_live_question_without_sources_is_blocked(self):
        answer, _, _, review = execute_review(
            FakeHandle("{}"), "搜索今天的地震新闻", {"messages": []},
            "今天发生了地震", [], [], [],
        )
        self.assertEqual(review["status"], "blocked")
        self.assertNotIn("今天发生了地震", answer)

    def test_revised_answer_requires_known_evidence_and_selects_only_own_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.txt"
            path.write_text("result", encoding="utf-8")
            response = response_with_tool("calculate_batch_ndvi", {"result": 3})
            structured = {
                "status": "revised", "issues": [{
                    "code": "wrong_number", "severity": "warning",
                    "message": "数值已更正", "evidence_ids": ["tool:call-1"],
                }],
                "revised_answer": "修订结论 3", "supplement": "文件说明：可核对计算结果。",
                "artifacts": {"display": [], "download": ["result.txt", "unknown.txt"]},
            }
            answer, images, files, review = execute_review(
                FakeHandle(json.dumps(structured)), "计算指数", response,
                "原始结论 2", [], [str(path)], [],
            )
            self.assertEqual(review["status"], "revised")
            self.assertIn("修订结论 3", answer)
            self.assertEqual(images, [])
            self.assertEqual(files, [str(path)])
            self.assertIn("artifact_selection_invalid", [x["code"] for x in review["issues"]])

    def test_malformed_review_is_visible_and_does_not_expose_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.txt"
            path.write_text("result", encoding="utf-8")
            answer, images, files, review = execute_review(
                FakeHandle('{"status":"passed","issues":[],"artifacts":[]}'),
                "评估", response_with_tool("calculate_batch_ndvi", {"result": 3}),
                "原始结论", [], [str(path)], [],
            )
            self.assertEqual(review["status"], "unavailable")
            self.assertIn("尚未完成审核", answer)
            self.assertEqual((images, files), ([], []))

    def test_parser_rejects_non_object_and_unknown_evidence(self):
        with self.assertRaises(ValueError):
            parse_review_response("[]", {"tool:call-1"})
        payload = {"status": "revised", "issues": [{
            "severity": "warning", "message": "wrong", "evidence_ids": ["invented"],
        }], "revised_answer": "fixed", "artifacts": {"display": [], "download": []}}
        with self.assertRaises(ValueError):
            parse_review_response(json.dumps(payload), {"tool:call-1"})
        self.assertIsNone(extract_artifact_selection("<Artifacts>[]</Artifacts>", []))

    def test_tool_evidence_and_risk_trigger(self):
        response = response_with_tool("extract_flood_inundation", {
            "total_valid_pixels": 4, "flood_pixels": 5,
        })
        evidence, checks = gather_review_evidence(response, [], [], "洪水评估")
        self.assertEqual(evidence[0]["id"], "tool:call-1")
        self.assertIn("flood_count_mismatch", [item["code"] for item in checks])
        self.assertTrue(review_required("洪水评估", response, [], []))
        self.assertFalse(review_required("你好", {"messages": []}, [], []))

    def test_damage_pixel_counts_cannot_be_reported_as_buildings(self):
        response = response_with_tool("assess_building_damage", {
            "count_unit": "pixels", "building_total": 10, "no_damage": 7,
            "minor_damage": 1, "major_damage": 1, "destroyed": 1,
            "damaged": 3,
        })
        _, issues = gather_review_evidence(response, [], [], "损毁评估", "共损毁 3 栋建筑")
        self.assertIn("pixel_count_as_buildings", [item["code"] for item in issues])

    def test_missing_reported_artifact_is_blocked(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.tif")
            response = response_with_tool("extract_flood_inundation", {
                "total_valid_pixels": 10, "flood_pixels": 2,
                "flood_mask_path": missing,
            })
            _, checks = gather_review_evidence(response, [], [], "洪水评估")
            self.assertIn("reported_artifact_missing", [item["code"] for item in checks])

    def test_geotiff_is_previewed_at_small_size(self):
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mask.tif"
            with rasterio.open(path, "w", driver="GTiff", width=4, height=4,
                               count=1, dtype="uint8", crs="EPSG:4326",
                               transform=from_origin(0, 4, 1, 1)) as dst:
                dst.write(np.arange(16, dtype="uint8").reshape(4, 4), 1)
            self.assertTrue(review_image_data_url(str(path)).startswith("data:image/png;base64,"))
            response = response_with_tool("extract_flood_inundation", {
                "total_valid_pixels": 16, "flood_pixels": 1,
                "flood_mask_path": str(path),
            })
            evidence, checks = gather_review_evidence(response, [str(path)], [], "洪水评估")
            self.assertFalse([item for item in checks if item["severity"] == "critical"])
            self.assertEqual(evidence[-1]["raster"]["crs"], "EPSG:4326")

    def test_independent_reviewer_config_is_optional(self):
        config = {"model_name": "main", "base_url": "http://main", "api_key": "main-key",
                  "generate_args": {"sample": True}}
        with patch.dict("os.environ", {"REVIEW_MODEL_NAME": "reviewer",
                                    "REVIEW_MODEL_URL": "http://reviewer",
                                    "REVIEW_MODEL_API_KEY": "review-key"}):
            selected = review_model_config(config)
        self.assertEqual(selected["model_name"], "reviewer")
        self.assertEqual(selected["generate_args"], {})


if __name__ == "__main__":
    unittest.main()
