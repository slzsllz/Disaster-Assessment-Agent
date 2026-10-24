import json
from pathlib import Path
import unittest

from evaluation.scoring import aggregate, decode_result, failed_result, objective_checks, selection, trace_messages


class EvaluationTests(unittest.TestCase):
    def test_dataset(self):
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "evaluation/questions_50.json").read_text())
        self.assertEqual(len(data["cases"]), 50)
        self.assertEqual(len({case["id"] for case in data["cases"]}), 50)
        self.assertEqual(sum(bool(case["required_tool_groups"]) for case in data["cases"]), 46)
        self.assertEqual(sum(bool(case.get("guard")) for case in data["cases"]), 4)

    def test_correlates_parallel_tool_ids(self):
        from langchain_core.messages import AIMessage, ToolMessage
        messages = [AIMessage(content="", tool_calls=[
            {"id": "a", "name": "query_weather", "args": {"latitude": 22}},
            {"id": "b", "name": "query_geo_context", "args": {"center": [114, 22]}}]),
            ToolMessage(content='{"provider":"amap"}', tool_call_id="b"),
            ToolMessage(content='{"forecast":[]}', tool_call_id="a")]
        trace = trace_messages(messages)
        self.assertEqual(trace[0]["name"], "query_weather")
        self.assertEqual(trace[0]["result"], {"forecast": []})
        self.assertEqual(trace[1]["result"], {"provider": "amap"})

    def test_nested_mcp_failure(self):
        value = [{"type": "text", "text": '{"success":false,"error":"bad input"}'}]
        self.assertTrue(failed_result(value))
        self.assertEqual(decode_result([{"type": "text", "text": "12"}]), 12)

    def test_right_choice_is_not_execution_success(self):
        case = {"required_tool_groups": [["extract_flood_inundation"]]}
        trace = [{"name": "extract_flood_inundation", "success": False, "args": {}}]
        self.assertTrue(selection(case, trace)["correct"])
        checks = objective_checks(case, trace, [], {})
        self.assertFalse(checks["passed"])

    def test_wrong_additional_disaster_tool(self):
        case = {"required_tool_groups": [["extract_flood_inundation"]], "forbidden_tools": ["extract_landslide_area"]}
        self.assertFalse(selection(case, [{"name": "extract_flood_inundation"}, {"name": "extract_landslide_area"}])["correct"])

    def test_guard_rejects_guessed_weather(self):
        case = {"required_tool_groups": [], "guard": True}
        self.assertTrue(selection(case, [])["correct"])
        self.assertFalse(selection(case, [{"name": "query_weather"}])["correct"])

    def test_missing_coordinates_are_failed_not_crash(self):
        case = {"required_tool_groups": [["query_weather"]], "coordinates": {"query_weather": [114, 22]}}
        checks = objective_checks(case, [{"name": "query_weather", "success": True, "args": {"latitude": None}}], [], {})
        self.assertFalse(checks["passed"])

    def test_unknown_judge_is_not_success(self):
        cases = [{"id": "a", "required_tool_groups": [["x"]]}, {"id": "b", "required_tool_groups": []}]
        results = [{"id": "a", "task_success": True, "selection": {"correct": True}, "task_elapsed_s": 10},
                   {"id": "b", "task_success": None, "selection": {"correct": True}, "task_elapsed_s": 20}]
        result = aggregate(cases, results)
        self.assertIsNone(result["task_success_rate"])
        self.assertEqual(result["task_success_lower_bound"], .5)
        self.assertEqual(result["major_tool_accuracy"], 1.)
        self.assertEqual(result["mean_task_elapsed_s"], 15.)


if __name__ == "__main__":
    unittest.main()
