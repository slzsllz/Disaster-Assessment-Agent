import unittest
from types import SimpleNamespace

from agent.tool_policy import CURATED_TOOL_NAMES, CURATED_TOOLS_BY_SERVER, select_curated_tools
from backend_api import load_model_config


class ToolPolicyTests(unittest.TestCase):
    def test_default_configuration_starts_only_reviewed_servers(self):
        configured = set(load_model_config()["mcp_servers"])
        self.assertEqual(configured, set(CURATED_TOOLS_BY_SERVER))
        self.assertEqual(len(CURATED_TOOL_NAMES), 39)
        self.assertNotIn("PestDetection", configured)
        self.assertNotIn("LandslideSegmentation", configured)
        self.assertNotIn("GeoContextQuery", configured)

    def test_unreviewed_tools_are_not_passed_to_agent(self):
        inventory = [SimpleNamespace(name=name) for name in CURATED_TOOL_NAMES]
        inventory.append(SimpleNamespace(name="mean"))
        selected = select_curated_tools(inventory)
        self.assertEqual({tool.name for tool in selected}, CURATED_TOOL_NAMES)

    def test_missing_approved_tool_fails_closed(self):
        with self.assertRaises(RuntimeError):
            select_curated_tools([])


if __name__ == "__main__":
    unittest.main()
