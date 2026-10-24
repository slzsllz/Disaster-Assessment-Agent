# Agent Task Evaluation

This directory contains a fixed 50-question development benchmark and an isolated
runner. It evaluates task completion and major-tool selection, not the scientific
accuracy of the underlying disaster models.

Questions and scoring criteria are fixed before running. Expected tools and grading
rubrics are never included in the messages sent to the evaluated agent.

Each question uses a fresh conversation and the production prompt, tool guards,
error recovery, multimodal input processing and artifact-review helpers. Existing
chat records are not modified. Only bundled sample data and explicitly labelled
synthetic fixtures are used. No training or model download is required.

## Metrics

- Task success: all objective requirements AND all answer criteria pass. Objective
  requirements cover required tools, successful execution, supplied inputs, roles,
  requested parameters, provider, artifacts and numerical raster oracles where
  applicable. A separate request to the same configured LLM grades the answer
  against frozen criteria and actual tool results. This is automated grading,
  not independent expert annotation or model accuracy evaluation.
- Task success rate: passed tasks / 50, only when all 50 can be graded. Judge API
  failure leaves a task ungraded; report coverage, scored-task rate and the
  conservative lower bound, never silently label it successful or unsuccessful.
- Major-tool selection accuracy: correct tool-required tasks / 46. A correct task
  must call one accepted tool from every required group, with no explicitly
  unrelated disaster tool. Selection does NOT imply that execution or arguments
  succeeded. Argument correctness also contributes to task success.
- Guard accuracy: correct clarification/no-inappropriate-tool behavior / 4.
- Mean task processing time: mean of all attempted tasks, including failures and
  timeouts. It includes fresh Agent/tool discovery, main LLM/tool loop and artifact
  review, but excludes grading, fixture creation, HTTP, database persistence and
  frontend rendering. Also retain setup, main, review and warm-task times.

## Protocol

The fixed set includes 28 single-model tasks, 9 weather/geography/route/search
tasks, 7 numerical/remote-sensing tasks, 2 multi-tool tasks and 4 guard tasks.
There is no RAG-specific evaluation. Earthquake images and traffic features are
explicitly synthetic; the other model inputs are bundled samples. Traffic tests
do not cover raw video. Some numerical inputs have exact known output oracles.

Questions never name the exact required function or expose the scoring rubric to
the agent. Every task has a fresh conversation, no prior answers, the production
system prompt and all currently registered tools. The evaluator freezes local
error-memory learning and uses an isolated temp directory/GPU environment only
inside its own processes. No production service is restarted or patched.

Default: 240-second task limit, two concurrent tasks, one attempt per question;
never keep the best of repeated answers. Judge time is recorded separately.
The same-backbone judge has self-preference/bias risks; manually audit raw
answers and evidence before using these development-set numbers in a paper.

## Run

```bash
python evaluation/run.py --output tmp/agent_eval/20261024 --prepare-only
python -m unittest discover -s tests -p test_agent_evaluation.py
python evaluation/run.py --output tmp/agent_eval/20261024 --concurrency 2 --gpu 2
```

Set `--gpu` to an idle physical GPU index (`-1` for CPU). Runtime data uses the
project disk via `TMPDIR`, not the system `/tmp`. Resume identical questions with
`--resume`. Changing the question snapshot is rejected on resume.

Questions are in `questions_50.json`. The run stores question/prompt hashes,
fixture provenance, per-task answers, ID-correlated tool traces, objective checks,
LLM judge decisions, a CSV and aggregate metrics. No credentials are copied.
