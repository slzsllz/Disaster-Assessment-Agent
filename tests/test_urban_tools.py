import ast
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from rasterio.transform import from_origin

import backend_api


ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "Analysis": {"compute_linear_trend", "mann_kendall_test", "sens_slope", "detect_change_points"},
    "Index": {"calculate_batch_ndvi", "calculate_batch_ndbi", "calculate_batch_nbr"},
    "Statistics": {
        "calc_batch_image_mean", "calc_batch_image_std", "calc_batch_image_min", "calc_batch_image_max",
        "calculate_tif_difference", "calculate_threshold_ratio", "count_pixels_satisfying_conditions",
        "calculate_intersection_percentage", "mean", "percentage_change",
        "get_percentile_value_from_image", "grayscale_to_colormap",
    },
    "Perception": {"threshold_segmentation", "bbox_expansion", "bboxes2centroids"},
}


def write_raster(path, data):
    bands = data[np.newaxis] if data.ndim == 2 else data
    with rasterio.open(
        path, "w", driver="GTiff", height=bands.shape[1], width=bands.shape[2],
        count=bands.shape[0], dtype="float32", crs="EPSG:3857",
        transform=from_origin(0, 100, 10, 10),
    ) as destination:
        destination.write(bands.astype("float32"))


class UrbanToolConfigurationTests(unittest.TestCase):
    def test_only_approved_functions_are_registered(self):
        total = 0
        for module, expected in EXPECTED.items():
            tree = ast.parse((ROOT / "agent/tools" / f"{module}.py").read_text(encoding="utf-8"))
            registered = {
                node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                and any(
                    isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "tool" for decorator in node.decorator_list
                )
            }
            self.assertEqual(registered, expected, module)
            total += len(registered)
        self.assertEqual(total, 22)
        self.assertFalse((ROOT / "agent/tools/Inversion.py").exists())

    def test_configurations_enable_all_retained_services(self):
        for path in (ROOT / "agent").glob("config*.json"):
            configuration = json.loads(path.read_text(encoding="utf-8"))
            servers = configuration.get("mcpServers", {})
            if not any(name in servers for name in EXPECTED):
                continue
            self.assertTrue(EXPECTED.keys() <= servers.keys(), path.name)
            self.assertNotIn("Inversion", servers, path.name)
            for server in servers.values():
                for argument in server.get("args", []):
                    if argument.startswith("tools/") and argument.endswith(".py"):
                        self.assertTrue((ROOT / "agent" / argument).is_file(), argument)


class UrbanToolMCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_22_tools_through_real_mcp_servers(self):
        with tempfile.TemporaryDirectory(prefix="urban-tools-test-") as temporary:
            directory = Path(temporary)
            data = np.arange(1, 17, dtype="float32").reshape(4, 4)
            rasters = {}
            for name, values in {
                "a": data, "b": data * 2, "multi": np.stack([data, data * 2]),
                "nir": np.full((4, 4), 0.8), "red": np.full((4, 4), 0.2),
                "swir": np.full((4, 4), 0.4),
            }.items():
                rasters[name] = str(directory / f"{name}.tif")
                write_raster(rasters[name], values)
            called = set()

            async def invoke(session, name, arguments):
                result = await session.call_tool(name, arguments)
                self.assertFalse(result.isError, f"{name}: {result.content}")
                called.add(name)
                values = []
                for block in result.content:
                    if block.type == "text":
                        try:
                            values.append(json.loads(block.text))
                        except ValueError:
                            values.append(block.text)
                return values[0] if len(values) == 1 else values

            for module, expected in EXPECTED.items():
                output = directory / module
                parameters = StdioServerParameters(
                    command=backend_api.CONDA_PYTHON,
                    args=[str(ROOT / "agent/tools" / f"{module}.py"), "--temp_dir", str(output)],
                    env=backend_api.build_mcp_child_env(),
                    cwd=str(ROOT / "agent"),
                )
                with (directory / f"{module}.log").open("w", encoding="utf-8") as log:
                    async with stdio_client(parameters, errlog=log) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            await session.initialize()
                            discovered = await session.list_tools()
                            self.assertEqual({tool.name for tool in discovered.tools}, expected, module)
                            if module == "Analysis":
                                result = await invoke(session, "compute_linear_trend", {"y": [1, 3, 5, 7]})
                                np.testing.assert_allclose(result, [2, 1])
                                result = await invoke(session, "mann_kendall_test", {"x": list(range(10))})
                                self.assertEqual(result[0], "increasing")
                                result = await invoke(session, "sens_slope", {"x": [1, 3, 5, 7]})
                                self.assertEqual(result[0], 2)
                                result = await invoke(session, "detect_change_points", {
                                    "signal": [0] * 20 + [10] * 20, "penalty": 1,
                                })
                                self.assertIn(20, result)
                            elif module == "Index":
                                for name, extra, expected_value in [
                                    ("calculate_batch_ndvi", {"input_red_paths": [rasters["red"]]}, 0.6),
                                    ("calculate_batch_ndbi", {"input_swir_paths": [rasters["swir"]]}, -1 / 3),
                                    ("calculate_batch_nbr", {"input_swir_paths": [rasters["swir"]]}, 1 / 3),
                                ]:
                                    filename = f"indices/{name}.tif"
                                    await invoke(session, name, {
                                        "input_nir_paths": [rasters["nir"]], "output_paths": [filename], **extra,
                                    })
                                    with rasterio.open(output / filename) as source:
                                        np.testing.assert_allclose(source.read(1), expected_value, rtol=1e-5)
                                        self.assertEqual(source.crs.to_epsg(), 3857)
                            elif module == "Statistics":
                                for name, expected_value in [
                                    ("calc_batch_image_mean", 8.5), ("calc_batch_image_std", np.std(data, ddof=1)),
                                    ("calc_batch_image_min", 1), ("calc_batch_image_max", 16),
                                ]:
                                    result = await invoke(session, name, {"file_list": [rasters["a"]]})
                                    np.testing.assert_allclose(result, [expected_value])
                                await invoke(session, "calculate_tif_difference", {
                                    "image_a_path": rasters["a"], "image_b_path": rasters["b"],
                                    "output_path": "difference.tif",
                                })
                                with rasterio.open(output / "difference.tif") as source:
                                    np.testing.assert_allclose(source.read(1), data)
                                result = await invoke(session, "calculate_threshold_ratio", {
                                    "image_paths": rasters["a"], "threshold": 8,
                                })
                                self.assertEqual(result, 50)
                                result = await invoke(session, "count_pixels_satisfying_conditions", {
                                    "image_path": rasters["multi"], "band_conditions": [[0, 8, "above"], [1, 24, "below"]],
                                })
                                self.assertEqual(result, 3)
                                result = await invoke(session, "calculate_intersection_percentage", {
                                    "path1": rasters["a"], "threshold1": 8, "path2": rasters["b"], "threshold2": 16,
                                })
                                self.assertEqual(result, 50)
                                self.assertEqual(await invoke(session, "mean", {"x": [1, 2, 3]}), 2)
                                self.assertEqual(await invoke(session, "percentage_change", {"a": 100, "b": 150}), 50)
                                self.assertEqual(await invoke(session, "get_percentile_value_from_image", {
                                    "image_path": rasters["a"], "percentile": 50,
                                }), 8.5)
                                result = await invoke(session, "grayscale_to_colormap", {
                                    "image_path": rasters["a"], "save_name": "maps/colormap.png",
                                })
                                self.assertTrue((output / "maps/colormap.png").exists(), result)
                                result = await invoke(session, "grayscale_to_colormap", {
                                    "image_path": rasters["a"], "save_name": "maps/georeferenced.png", "preserve_geo": True,
                                })
                                self.assertEqual(result, f"Result save at {output / 'maps/georeferenced.tif'}")
                                with rasterio.open(output / "maps/georeferenced.tif") as source:
                                    self.assertEqual(source.count, 3)
                                    self.assertEqual(source.crs.to_epsg(), 3857)
                            else:
                                await invoke(session, "threshold_segmentation", {
                                    "input_image_path": rasters["a"], "threshold": 8, "output_path": "mask.tif",
                                })
                                with rasterio.open(output / "mask.tif") as source:
                                    np.testing.assert_array_equal(source.read(1), (data > 8).astype("uint8") * 255)
                                result = await invoke(session, "bbox_expansion", {
                                    "bboxes": [[0, 0, 10, 20]], "radius": 5, "gsd": 1,
                                })
                                np.testing.assert_allclose(result, [[-5, -5, 15, 25]])
                                result = await invoke(session, "bboxes2centroids", {"bboxes": [[0, 0, 10, 20]]})
                                np.testing.assert_allclose(result, [[5, 10]])
            self.assertEqual(called, set().union(*EXPECTED.values()))


if __name__ == "__main__":
    unittest.main()
