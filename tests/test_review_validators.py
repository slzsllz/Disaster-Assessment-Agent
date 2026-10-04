"""Domain audit rules must keep blocking inconsistent disaster measurements."""

import json
import unittest

from langchain_core.messages import AIMessage, ToolMessage

from agent.review import gather_review_evidence


def issue_codes(tool_name: str, data: dict, answer: str = "") -> set[str]:
    response = {"messages": [
        AIMessage(content="", tool_calls=[
            {"name": tool_name, "args": {}, "id": "call-1", "type": "tool_call"}
        ]),
        ToolMessage(content=json.dumps(data), tool_call_id="call-1", name=tool_name),
    ]}
    _, issues = gather_review_evidence(response, [], [], "", answer)
    return {issue["code"] for issue in issues}


class ReviewValidatorTests(unittest.TestCase):
    def test_damage_validator_checks_counts_and_pixel_units(self):
        data = {
            "building_total": 9, "no_damage": 5, "minor_damage": 2,
            "major_damage": 2, "destroyed": 1, "damaged": 5,
            "count_unit": "pixels",
        }
        codes = issue_codes("assess_building_damage", data, "受损 5 栋")
        self.assertIn("damage_count_mismatch", codes)
        self.assertIn("pixel_count_as_buildings", codes)

    def test_flood_validator_checks_count_and_shared_ratio_rule(self):
        codes = issue_codes("extract_flood_inundation", {
            "total_valid_pixels": 5, "flood_pixels": 6, "flood_ratio": 1.2,
        })
        self.assertIn("flood_count_mismatch", codes)
        self.assertIn("invalid_ratio", codes)


if __name__ == "__main__":
    unittest.main()
