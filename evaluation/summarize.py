"""Produce a Chinese audit summary without re-running or re-grading answers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation.scoring import aggregate


def summarize(directory):
    directory = Path(directory).resolve()
    cases = json.loads((directory / "questions.json").read_text())["cases"]
    results = json.loads((directory / "results.json").read_text())
    settings = json.loads((directory / "settings.json").read_text())
    assert hashlib.sha256((directory / "questions.json").read_bytes()).hexdigest() == settings["question_sha256"]
    by_id = {row["id"]: row for row in results}
    metrics = aggregate(cases, results)
    effective = sum(bool(by_id.get(case["id"], {}).get("objective", {}).get("passed"))
                    for case in cases if case["required_tool_groups"])
    metrics["major_tool_effective_tasks"] = effective
    metrics["major_tool_effective_rate"] = effective / metrics["required_tool_tasks"]
    times = [row["task_elapsed_s"] for row in results]
    metrics["median_task_elapsed_s"] = statistics.median(times)
    metrics["p90_task_elapsed_s"] = statistics.quantiles(times, n=10, method="inclusive")[8] if len(times) > 1 else times[0]
    warm = [row["warm_task_elapsed_s"] for row in results if row.get("warm_task_elapsed_s") is not None]
    metrics["mean_warm_task_elapsed_s"] = statistics.mean(warm) if warm else None
    metrics["model_names"] = dict(Counter(row.get("model_name", "unknown") for row in results))
    categories = []
    for category in dict.fromkeys(case["category"] for case in cases):
        subset = [case for case in cases if case["category"] == category]
        entries = [by_id[case["id"]] for case in subset if case["id"] in by_id]
        categories.append({"category": category, **aggregate(subset, entries)})

    def pct(value):
        return f"{value*100:.2f}%" if value is not None else "未完成核验"

    lines = ["# Agent 50题自动评测报告", "", "## 统计口径", "",
             "这是自建单轮开发评测集，不是公开独立基准。所有任务仅执行一次；试运行不计入结果。",
             "使用当前工作区模型配置、系统提示词、全部注册工具和生产主流程/review方法，独立运行，不创建生产聊天记录。",
             f"每题独立对话且只执行一次，时限{settings['timeout']}秒，并发数{settings['concurrency']}，可用模型使用物理GPU {settings['gpu']}（本机A40）；城市火灾按适配器使用CPU。",
             "评分规则先固定再执行，正确工具名和评分标准不发给被测Agent。模型级损毁真值、掩膜IoU、人员伤亡和真实建筑栋数不在本评测范围。", "",
             "## 核心指标", "", "| 指标 | 结果 |", "|---|---|",
             f"| 任务成功率 | {metrics['task_successes']}/{metrics['questions']}，{pct(metrics['task_success_rate'])} |",
             f"| 主要工具选择正确率 | {metrics['major_tool_correct']}/{metrics['required_tool_tasks']}，{pct(metrics['major_tool_accuracy'])} |",
             f"| 主要工具客观执行达标率 | {effective}/{metrics['required_tool_tasks']}，{pct(metrics['major_tool_effective_rate'])} |",
             f"| 缺少/不兼容输入题未误调用率 | {metrics['guard_correct']}/{metrics['guard_tasks']}，{pct(metrics['guard_accuracy'])} |",
             f"| 平均任务处理耗时（全部尝试） | {metrics['mean_task_elapsed_s']:.2f} 秒 |",
             f"| 任务处理耗时中位数 | {metrics['median_task_elapsed_s']:.2f} 秒 |",
             f"| P90任务处理耗时 | {metrics['p90_task_elapsed_s']:.2f} 秒 |", "",
             "任务成功要求客观检查和最终答案评分点全部通过；选择正确只代表主要工具匹配，不能替代执行成功。",
             "客观执行达标率包含工具选择、实际执行、输入文件/顺序/指定参数、服务提供方，以及适用题目的产物和精确数值检查。",
             "需要工具的46题和需要澄清的4题分开计算工具指标。任务成功率仍包含全部50题。",
             "计时包含每题新建Agent/工具发现、主模型与工具循环及多模态review；不含评分、样本准备、HTTP传输、数据库持久化或前端渲染。失败和超时计入均值。", "",
             "## 分场景结果", "", "| 场景 | 通过/题数 | 任务成功率 | 平均耗时（秒） |", "|---|---:|---:|---:|"]
    for row in categories:
        lines.append(f"| {row['category']} | {row['task_successes']}/{row['questions']} | {pct(row['task_success_rate'])} | {row['mean_task_elapsed_s']:.2f} |")
    lines.extend(["", "## 未通过或未核验题", ""])
    failed = [row for row in results if row.get("task_success") is not True]
    for row in failed:
        bad = [name for name, valid in row.get("objective", {}).get("checks", {}).items() if not valid]
        reason = row.get("judge", {}).get("reason") or row.get("judge_error") or row.get("error") or ", ".join(bad)
        lines.append(f"- {row['id']}（{row['category']}）：{reason}")
    if not failed:
        lines.append("本轮所有题均通过预定自动检查；不代表真实灾害识别精度或其他问题上也能达到相同比例。")
    if (directory / "diagnostics.json").exists():
        lines.extend(["", "## 本轮已复核的问题", "",
                      "- E027、E028：交通模型选择正确，上传NPZ路径也正确，但输入保护因model_profile包含file而把合法类别值ccd误当作文件路径，阻止了调用。",
                      "- E043：阈值三次均以字符串0.5传入，NumPy无法将浮点栅格与字符串比较，没有生成二值栅格。没有在评测中自动转型或修补工具。",
                      "- E045：三个主要工具均成功完成，但主流程提取的raw_answer长度为0，review只提供文件说明与建议，最终没有主体分析和天气影响分析。此证据不能证明非空天气内容被过滤；未记录finish_reason，不能断言空回答的模型端原因。",
                      "- 诊断不改变原始判定；详细证据见diagnostics.json和对应case目录。本轮未修复业务代码，也没有重跑失败题取最好结果。"])
    lines.extend(["", "## 局限与复核", "",
                  "- 地震图像和交通特征为合成流程测试输入，交通不包含原始视频处理；其余检测优先使用模型自带样本。部分输入重复使用不同问法，不能把50题当作50个独立灾害事件。",
                  "- 有9类模型任务及天气、地理、路径、检索和计算任务，但没有专项覆盖RAG、多轮记忆、并发压测或全部39个MCP函数。",
                  "- 主要工具选择允许预先列出的等价工具；不计算所有辅助调用的全量precision/recall。",
                  "- 对最终答案的语义核验使用单独请求的同源LLM，存在同源偏好和误判风险。它不是人工专家标注；论文或比赛报告需复核原始回答与工具证据。",
                  "- 本评测采用严格任务定义，错误调用不相关的灾害工具也会影响任务成功，不能把该比例直接等同其他论文的成功率。",
                  "- 样本少、单次采样、缺乏外部基线；不得据此声称优于其他Agent。",
                  "- 问题快照及系统提示词/运行器/评分器hash见settings.json，逐题完整证据见cases/E*/result.json，表格见results.csv。"])
    (directory / "summary_zh.md").write_text("\n".join(lines)+"\n")
    (directory / "summary_metrics.json").write_text(json.dumps({"overall": metrics, "categories": categories}, ensure_ascii=False, indent=2)+"\n")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), ensure_ascii=False, indent=2))
