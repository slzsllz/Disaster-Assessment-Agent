"""Auditable, predeclared checks independent of the evaluated LLM."""
import json
from pathlib import Path


def decode_result(value):
    if isinstance(value, str):
        try:
            return decode_result(json.loads(value))
        except (ValueError, TypeError):
            return value
    if isinstance(value, list):
        parts = [decode_result(item.get("text", item) if isinstance(item, dict) else item) for item in value]
        return parts[0] if len(parts) == 1 else parts
    if isinstance(value, dict) and set(value) == {"result"}:
        return decode_result(value["result"])
    return value


def failed_result(value):
    value = decode_result(value)
    if isinstance(value, dict):
        return (value.get("success") is False or value.get("isError") is True
                or str(value.get("status", "")).lower() in {"error", "failed", "failure"}
                or bool(value.get("error") and value.get("success") is not True))
    if isinstance(value, list):
        return any(failed_result(item) for item in value)
    if isinstance(value, str):
        return value.strip().lower().startswith(("error", "failed", "错误", "失败"))
    return False


def trace_messages(messages):
    from langchain_core.messages import AIMessage, ToolMessage

    calls, by_id = [], {}
    for message in messages:
        if isinstance(message, AIMessage):
            tool_calls = getattr(message, "tool_calls", None) or []
            if not tool_calls:
                for call in (getattr(message, "additional_kwargs", {}) or {}).get("tool_calls", []):
                    function = call.get("function", {})
                    tool_calls.append({"id": call.get("id"), "name": function.get("name"),
                                       "args": decode_result(function.get("arguments", {}))})
            for call in tool_calls:
                entry = {"id": call.get("id"), "name": call.get("name"), "args": call.get("args", {}),
                         "result": None, "success": False, "completed": False}
                calls.append(entry)
                if entry["id"]:
                    by_id[entry["id"]] = entry
        elif isinstance(message, ToolMessage):
            entry = by_id.get(getattr(message, "tool_call_id", ""))
            if entry is None:
                entry = {"id": getattr(message, "tool_call_id", ""), "name": getattr(message, "name", ""), "args": {}}
                calls.append(entry)
            result = decode_result(message.content)
            entry.update(result=result, completed=True,
                         success=getattr(message, "status", "success") != "error" and not failed_result(result))
    return calls


def strings(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [text for item in value.values() for text in strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in strings(item)]
    return []


def dictionaries(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from dictionaries(item)
    elif isinstance(value, list):
        for item in value:
            yield from dictionaries(item)


def selection(case, trace):
    names = {item.get("name") for item in trace}
    missing = [group for group in case["required_tool_groups"] if not names.intersection(group)]
    prohibited = set(case.get("forbidden_tools", []))
    if case.get("guard"):
        prohibited.update({"geoai_object_detection", "geoai_semantic_segmentation", "geoai_image_similarity",
                           "geoai_extract_features", "query_geo_context", "query_weather", "plan_emergency_routes"})
    forbidden = sorted(names.intersection(prohibited))
    return {"correct": not missing and not forbidden, "called": sorted(name for name in names if name),
            "missing_groups": missing, "forbidden_called": forbidden}


def near_coordinate(value, expected):
    try:
        return abs(float(value) - expected) < .001
    except (TypeError, ValueError):
        return False


def objective_checks(case, trace, output_paths, fixture_paths):
    selected = selection(case, trace)
    checks = {"major_tool_selection": selected["correct"]}
    groups = case["required_tool_groups"]
    if groups:
        checks["required_tools_succeeded"] = all(
            any(item.get("name") in group and item.get("success") for item in trace) for group in groups)
        supplied = {Path(path).resolve() for path in fixture_paths.values()}
        main_names = {name for group in groups for name in group}
        used = {Path(text).resolve() for item in trace if item.get("name") in main_names
                for text in strings(item.get("args", {})) if text.startswith("/")}
        checks["uploaded_inputs_used"] = supplied.issubset(used)
    if case.get("require_output"):
        checks["generated_file_exists"] = any(Path(path).is_file() and Path(path).stat().st_size > 0 for path in output_paths)
    for name, expected in case.get("input_arguments", {}).items():
        checks["input_roles_" + name] = any(
            item.get("name") == name and isinstance(item.get("args"), dict)
            and all(Path(str(item["args"].get(key, "__missing__"))).resolve() == Path(fixture_paths[value]).resolve()
                    for key, value in expected.items()) for item in trace)
    for name, expected in case.get("argument_values", {}).items():
        checks["arguments_" + name] = any(
            item.get("name") == name and isinstance(item.get("args"), dict)
            and all(item["args"].get(key) == value for key, value in expected.items()) for item in trace)
    if case.get("provider"):
        checks["correct_geo_provider"] = any(
            item.get("name") == "query_geo_context" and item.get("success")
            and any(part.get("provider") == case["provider"] for part in dictionaries(item.get("result")))
            for item in trace)
    for name, (longitude, latitude) in case.get("coordinates", {}).items():
        checks["coordinates_" + name] = any(
            item.get("name") == name and isinstance(item.get("args"), dict)
            and near_coordinate(item["args"].get("longitude"), longitude)
            and near_coordinate(item["args"].get("latitude"), latitude) for item in trace)
    if case.get("raster_check"):
        import numpy as np
        import rasterio

        expected, valid = case["raster_check"], False
        for path in output_paths:
            if Path(path).suffix.lower() not in {".tif", ".tiff"}:
                continue
            try:
                with rasterio.open(path) as source:
                    values = source.read(1, masked=True).compressed()
                if not len(values):
                    continue
                correct = abs(float(np.mean(values)) - expected["mean"]) <= expected["tolerance"]
                if "unique" in expected:
                    correct &= set(np.unique(values).tolist()) == set(expected["unique"])
                valid |= bool(correct)
            except (OSError, ValueError):
                pass
        checks["numerical_raster_oracle"] = valid
    return {"passed": all(checks.values()), "checks": checks, "selection": selected}


def aggregate(cases, results):
    import statistics

    by_id = {item["id"]: item for item in results}
    required = [case for case in cases if case["required_tool_groups"]]
    guards = [case for case in cases if not case["required_tool_groups"]]
    scored = [item for item in results if item.get("task_success") is not None]
    successes = sum(item.get("task_success") is True for item in results)
    times = [item["task_elapsed_s"] for item in results if item.get("task_elapsed_s") is not None]
    success_times = [item["task_elapsed_s"] for item in results if item.get("task_success") is True]
    main_correct = sum(bool(by_id.get(case["id"], {}).get("selection", {}).get("correct")) for case in required)
    guard_correct = sum(bool(by_id.get(case["id"], {}).get("selection", {}).get("correct")) for case in guards)
    return {"questions": len(cases), "completed": len(results), "graded": len(scored),
            "task_successes": successes, "task_success_rate": successes / len(cases) if len(scored) == len(cases) else None,
            "task_success_lower_bound": successes / len(cases),
            "graded_task_success_rate": successes / len(scored) if scored else None,
            "required_tool_tasks": len(required), "major_tool_correct": main_correct,
            "major_tool_accuracy": main_correct / len(required) if required else None,
            "guard_tasks": len(guards), "guard_correct": guard_correct,
            "guard_accuracy": guard_correct / len(guards) if guards else None,
            "mean_task_elapsed_s": statistics.mean(times) if times else None,
            "mean_successful_task_elapsed_s": statistics.mean(success_times) if success_times else None,
            "timeouts": sum(item.get("status") == "timeout" for item in results),
            "ungraded": len(cases) - len(scored)}
