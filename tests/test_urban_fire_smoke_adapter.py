"""Standard-library unit checks for the urban smoke/fire MCP bridge (no GPU)."""
from __future__ import annotations
import ast
import asyncio
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ADAPTER = Path(__file__).resolve().parents[1] / "agent" / "tools" / "UrbanFireSmoke.py"
spec = importlib.util.spec_from_file_location("urban_fire_adapter_under_test", ADAPTER)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)

class UrbanFireBridgeTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.image_dir = self.root / "tmp"
        self.image_dir.mkdir()
        self.image = self.image_dir / "smoke.jpg"
        self.image.write_bytes(b"\xff\xd8\xff\xe0fake-jpeg")
        self.package = self.root / "model" / "urban_fire_smoke_yolov5l"
        (self.package / "weights").mkdir(parents=True)
        for name in ("infer.py", "fixed_model.py", "infer_cpu.py", "PACKAGE_MANIFEST.json"):
            (self.package / name).write_text("stub")
        for name in ("tensor_index.json", "tensors_numeric_only.npz"):
            (self.package / "weights" / name).write_bytes(b"stub")
        (self.package / "weights" / "metadata.json").write_text(json.dumps({
            "source_filename": "yolov5l.pt",
            "source_sha256": module.ORIGINAL_CHECKPOINT_SHA256,
        }))
        self.python = self.root / "model_python"
        self.python.write_text("stub")
        self.patches = [
            patch.object(module, "PROJECT_ROOT", self.root),
            patch.object(module.ARGS, "temp_dir", str(self.image_dir / "out")),
            patch.object(module.ARGS, "fire_python", str(self.python)),
            patch.object(module.ARGS, "l_model_dir", None),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)

    def test_invalid_model_and_threshold(self):
        for name in ("s", "l", "base", "m", "L", "../", ""):
            with self.assertRaises(TypeError), patch.object(module.subprocess, "run") as child:
                module.detect_urban_fire_smoke(str(self.image), model_name=name)
            child.assert_not_called()
        for v in (0.0, float("nan"), float("inf"), True, 1.2, "0.25", None):
            with self.assertRaises(ValueError):
                module.detect_urban_fire_smoke(str(self.image), conf=v)

    def test_input_root_and_symlink(self):
        outside = self.root / "out-of-root.jpg"
        outside.write_bytes(self.image.read_bytes())
        with self.assertRaises(ValueError):
            module.detect_urban_fire_smoke(str(outside))
        symlink = self.image_dir / "linked.jpg"
        symlink.symlink_to(self.image)
        with self.assertRaises(ValueError):
            module.detect_urban_fire_smoke(str(symlink))

    def test_bad_model_provenance(self):
        metadata = self.package / "weights" / "metadata.json"
        metadata.write_text(json.dumps({"source_filename": "yolov5s.pt", "source_sha256": "wrong"}))
        with self.assertRaises(ValueError):
            module.detect_urban_fire_smoke(str(self.image))

    def test_child_command_and_summary(self):
        def fake_run(cmd, **kw):
            self.assertEqual(cmd[:3], [str(self.python), "-I", "-B"])
            self.assertEqual(kw["env"]["CUDA_VISIBLE_DEVICES"], "-1")
            self.assertEqual(kw["timeout"], 180)
            self.assertEqual(kw["cwd"], str(self.package))
            self.assertEqual(cmd[3], str(self.package / "infer.py"))
            self.assertNotIn("yolov5s", " ".join(cmd))
            self.assertEqual(cmd[cmd.index("--conf") + 1], "0.25")
            self.assertEqual(cmd[cmd.index("--iou") + 1], "0.45")
            out = Path(cmd[cmd.index("--output") + 1])
            out.mkdir(parents=True)
            (out / "annotated.jpg").write_bytes(b"fake")
            (out / "results.json").write_text("{}")
            result = {
                "status": "success", "model": "yolov5l", "detections": [],
                "summary": {"smoke_count": 0, "fire_count": 0, "fire_confirmed": False},
                "annotated_image_path": str(out / "annotated.jpg"),
                "results_json_path": str(out / "results.json"),
                "summary_path": str(out / "summary.json"),
            }
            (out / "summary.json").write_text(json.dumps(result))
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        with patch.object(module.subprocess, "run", side_effect=fake_run) as mocked:
            result = module.detect_urban_fire_smoke(str(self.image))
            self.assertEqual(result["model"], "yolov5l")
            self.assertEqual(Path(result["summary_path"]).name, "summary.json")
            self.assertEqual(mocked.call_count, 1)

    def test_local_runtime_config(self):
        local = self.root / "model" / "urban_fire_runtime.json"
        local.write_text(json.dumps({"python_executable": str(self.python)}))
        with patch.object(module.ARGS, "fire_python", None):
            with patch.dict(module.os.environ, {"URBAN_FIRE_CPU_PYTHON": ""}):
                self.assertEqual(module._runtime_python(), str(self.python))

    def test_nonzero_subprocess_is_failure(self):
        with patch.object(module.subprocess, "run", return_value=subprocess.CompletedProcess([], 2, "bad", "")):
            with self.assertRaisesRegex(RuntimeError, "exit 2"):
                module.detect_urban_fire_smoke(str(self.image))

    def test_only_l_runtime_and_public_signature(self):
        self.assertEqual(module.MODEL_NAME, "yolov5l")
        self.assertEqual(module._model_dir(), self.package)
        self.assertEqual(list(inspect.signature(module.detect_urban_fire_smoke).parameters),
                         ["image_path", "conf"])
        self.assertFalse(hasattr(module.ARGS, "model"))
        self.assertFalse(hasattr(module.ARGS, "s_model_dir"))

    def test_no_fallback_when_l_is_missing(self):
        self.package.rename(self.package.with_name("urban_fire_smoke_yolov5s"))
        with patch.object(module.subprocess, "run") as child:
            with self.assertRaises(FileNotFoundError):
                module.detect_urban_fire_smoke(str(self.image))
        child.assert_not_called()

    def test_l_override_with_s_metadata_rejected(self):
        (self.package / "weights" / "metadata.json").write_text(json.dumps({
            "source_filename": "yolov5s.pt",
            "source_sha256": module.ORIGINAL_CHECKPOINT_SHA256,
        }))
        with self.assertRaises(ValueError):
            module._model_dir()

    def test_missing_model_file_rejected(self):
        (self.package / "fixed_model.py").unlink()
        with self.assertRaises(FileNotFoundError):
            module._model_dir()

    def test_symlinked_model_file_rejected(self):
        target = self.package / "fixed_model.py"
        target.unlink()
        target.symlink_to(self.image)
        with self.assertRaises(ValueError):
            module._model_dir()

    def test_symlinked_parent_input_directory_rejected(self):
        link = self.image_dir / "linked-directory"
        link.symlink_to(self.image_dir, target_is_directory=True)
        with self.assertRaises(ValueError):
            module._input_image(str(link / self.image.name))

    def test_directory_and_invalid_extension_rejected(self):
        with self.assertRaises(ValueError):
            module._input_image(str(self.image_dir))
        wrong = self.image_dir / "input.tif"
        wrong.write_bytes(b"image")
        with self.assertRaises(ValueError):
            module._input_image(str(wrong))

    def test_empty_and_oversized_image_rejected(self):
        self.image.write_bytes(b"")
        with self.assertRaises(ValueError):
            module._input_image(str(self.image))
        with self.image.open("wb") as stream:
            stream.truncate(30 * 1024 * 1024 + 1)
        with self.assertRaises(ValueError):
            module._input_image(str(self.image))

    def test_custom_input_root(self):
        outside = self.root / "trusted"
        outside.mkdir()
        image = outside / "test.jpg"
        image.write_bytes(b"image")
        with patch.object(module.ARGS, "input_root", [str(outside)]):
            self.assertEqual(module._input_image(str(image)), image)

    def test_output_parent_symlink_rejected(self):
        link = self.root / "output-link"
        link.symlink_to(self.image_dir, target_is_directory=True)
        with patch.object(module.ARGS, "temp_dir", str(link / "out")):
            with self.assertRaises(ValueError):
                module._output_dir()

    def test_runtime_config_symlink_rejected(self):
        local = self.root / "model" / "urban_fire_runtime.json"
        local.symlink_to(self.image)
        with patch.object(module.ARGS, "fire_python", None), patch.dict(
                module.os.environ, {"URBAN_FIRE_CPU_PYTHON": ""}):
            with self.assertRaises(ValueError):
                module._runtime_python()

    def test_timeout_propagated(self):
        with patch.object(module.subprocess, "run", side_effect=subprocess.TimeoutExpired([], 180)):
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                module.detect_urban_fire_smoke(str(self.image))

    def test_missing_summary_is_failure(self):
        with patch.object(module.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "", "")):
            with self.assertRaisesRegex(RuntimeError, "summary.json"):
                module.detect_urban_fire_smoke(str(self.image))

    def test_wrong_model_and_confirmed_fire_results_rejected(self):
        for model_name, confirmed in (("yolov5s", False), ("yolov5l", True)):
            with self.subTest(model=model_name, confirmed=confirmed):
                def fake_run(cmd, **kw):
                    out = Path(cmd[cmd.index("--output") + 1])
                    out.mkdir()
                    (out / "summary.json").write_text(json.dumps({
                        "status": "success", "model": model_name,
                        "summary": {"fire_confirmed": confirmed},
                    }))
                    return subprocess.CompletedProcess(cmd, 0, "", "")
                with patch.object(module.subprocess, "run", side_effect=fake_run):
                    with self.assertRaisesRegex(RuntimeError, "unexpected result schema"):
                        module.detect_urban_fire_smoke(str(self.image))

    def test_inconsistent_result_path_rejected(self):
        def fake_run(cmd, **kw):
            out = Path(cmd[cmd.index("--output") + 1])
            out.mkdir()
            (out / "annotated.jpg").write_bytes(b"fake")
            (out / "summary.json").write_text(json.dumps({
                "status": "success", "model": "yolov5l",
                "summary": {"fire_confirmed": False},
                "annotated_image_path": str(self.root / "outside.jpg"),
            }))
            return subprocess.CompletedProcess(cmd, 0, "", "")
        with patch.object(module.subprocess, "run", side_effect=fake_run):
            with self.assertRaisesRegex(RuntimeError, "absent or inconsistent"):
                module.detect_urban_fire_smoke(str(self.image))

    @unittest.skipIf(module.mcp is None, "FastMCP is not installed")
    def test_mcp_schema_and_legacy_parameter_rejection(self):
        async def check():
            tools = await module.mcp.get_tools()
            self.assertEqual(list(tools), ["detect_urban_fire_smoke"])
            tool = tools["detect_urban_fire_smoke"]
            self.assertEqual(set(tool.parameters["properties"]), {"image_path", "conf"})
            with patch.object(module.subprocess, "run") as child:
                with self.assertRaises(ValueError):
                    await tool.run({"image_path": str(self.image), "model_name": "s"})
                child.assert_not_called()
        asyncio.run(check())


class FirePromptTests(unittest.TestCase):
    def setUp(self):
        # Extract only the real prompt/function; avoid importing backend services.
        path = ADAPTER.parents[2] / "backend_api.py"
        tree = ast.parse(path.read_text())
        nodes = [
            node for node in tree.body
            if (
                isinstance(node, ast.FunctionDef)
                and node.name in ("load_system_prompt", "build_system_prompt")
            ) or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(t, ast.Name) and t.id == "DEFAULT_SYSTEM_PROMPT"
                    for t in node.targets
                )
            )
        ]
        self.ns = {
            "Path": Path,
            "SYSTEM_PROMPT_PATH": path.parent / "agent/prompts/system.md",
            "ERROR_MEMORY": SimpleNamespace(format_prompt_block=lambda: ""),
            "RAG_INSTRUCTIONS": "",
        }
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), self.ns)

    def test_default_prompt_has_only_l_and_retains_other_tool_rules(self):
        default = self.ns["DEFAULT_SYSTEM_PROMPT"]
        result = self.ns["build_system_prompt"](default, [])
        self.assertEqual(result.count("detect_urban_fire_smoke"), 1)
        self.assertIn("固定使用 YOLOv5l", result)
        self.assertNotIn("model_name='s'", result)
        self.assertIn("FireBurnedAreaChange", result)
        for line in default.splitlines():
            if "query_weather" in line or "search_web" in line:
                self.assertIn(line, result)

    def test_custom_prompt_gets_fire_rule(self):
        result = self.ns["build_system_prompt"]("自定义系统提示词", [])
        self.assertIn("自定义系统提示词", result)
        self.assertIn("固定使用 YOLOv5l", result)

    def test_stale_and_duplicate_fire_rules_replaced(self):
        old = "Use detect_urban_fire_smoke for smoke/fire detection; model_name='s' by default"
        result = self.ns["build_system_prompt"]("CUSTOM\n" + old + "\n" + old, [])
        self.assertIn("CUSTOM", result)
        self.assertEqual(result.count("detect_urban_fire_smoke"), 1)
        self.assertNotIn("model_name='s'", result)
        again = self.ns["build_system_prompt"](result, [])
        self.assertEqual(again.count("detect_urban_fire_smoke"), 1)

if __name__ == "__main__":
    unittest.main()
