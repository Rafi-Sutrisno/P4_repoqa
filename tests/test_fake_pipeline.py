"""End to end with a fake LLM that returns the ground truth: pass@1 must be 1.0."""

import json
import os

import pytest

from repoqa_ctx import run
from repoqa_ctx import tasks as T


@pytest.mark.parametrize(
    "strategy, budget, stages",
    [
        ("full", None, ["answer"]),
        ("index_select", 2048, ["select", "answer"]),
        ("cascade", 2048, ["select", "answer"]),
    ],
)
def test_fake_pipeline(strategy, budget, stages, mini_dataset, task_cache, tmp_path, monkeypatch):
    _, path = mini_dataset
    monkeypatch.setattr(T, "TASK_CACHE_DIR", task_cache)
    argv = [
        "--strategy", strategy,
        "--llm", "fake",
        "--fake-mode", "oracle",
        "--dataset-path", path,
        "--result-dir", str(tmp_path / "ctx"),
    ]  # fmt: skip
    if budget is not None:
        argv += ["--budget", str(budget)]
    run_dir = run.run_experiment(run.parse_args(argv))

    with open(os.path.join(run_dir, "scores.json"), encoding="utf-8") as f:
        scores = json.load(f)
    # Threshold 0.0: the oracle answer is the best match for every task. At 0.8 even a
    # perfect answer cannot reach 1.0: RepoQA's BLEU gives 0.576 to an exact copy of a
    # 3-token needle (cpp UTF8CharsetEncoder in this subset). See docs/DECISIONS.md.
    assert scores["fake-oracle"]["scores"]["all"]["0.0"]["pass@1"] == 1.0

    # One log line per LLM call, tagged with its stage.
    calls = run.read_jsonl(os.path.join(run_dir, "calls.jsonl"))
    assert len(calls) == 60 * len(stages)
    for uid in {c["uid"] for c in calls}:
        assert [c["stage"] for c in calls if c["uid"] == uid] == stages


# Phase 5 strategies: light check on 12 tasks (2 per language). `stages` is the exact stage
# list per task; None = checked by `check` instead.
PHASE5 = [
    ("adaptive_k", {"signal": "margin"}, 2048, ["answer"]),
    ("adaptive_k", {"signal": "units"}, 2048, ["answer"]),
    ("adaptive_k", {"signal": "llm"}, 2048, ["confidence", "answer"]),
    ("hyde_code", {"then": "s1"}, 2048, ["hyde", "answer"]),
    ("hyde_code", {"then": "s3"}, 2048, ["hyde", "select", "answer"]),
    ("pair_verify", {"both_orders": True}, 2048, None),
    ("graph_neighbors", {}, 2048, ["answer"]),
    ("pointer", {}, None, ["select"]),
    ("index_select", {"index": "sum"}, 2048, None),
]


@pytest.mark.parametrize(
    "strategy, sargs, budget, stages", PHASE5, ids=[f"{s}-{json.dumps(a)}" for s, a, _, _ in PHASE5]
)
def test_fake_pipeline_phase5(
    strategy, sargs, budget, stages, mini_dataset, task_cache, tmp_path, monkeypatch
):
    from repoqa_ctx import summary_cache

    _, path = mini_dataset
    monkeypatch.setattr(T, "TASK_CACHE_DIR", task_cache)
    monkeypatch.setattr(summary_cache, "SUMMARY_DIR", str(tmp_path / "summaries"))
    argv = [
        "--strategy", strategy,
        "--strategy-args", json.dumps(sargs),
        "--llm", "fake",
        "--fake-mode", "oracle",
        "--limit", "12",
        "--dataset-path", path,
        "--result-dir", str(tmp_path / "ctx"),
    ]  # fmt: skip
    if budget is not None:
        argv += ["--budget", str(budget)]
    run_dir = run.run_experiment(run.parse_args(argv))

    records = run.read_jsonl(os.path.join(run_dir, "outputs.jsonl"))
    calls = run.read_jsonl(os.path.join(run_dir, "calls.jsonl"))
    assert len(records) == 12
    for record in records:
        got = [c["stage"] for c in calls if c["uid"] == record["uid"]]
        if stages is not None:
            assert got == stages
        elif strategy == "pair_verify":  # verify only when the top two are close
            assert got in (["answer"], ["verify", "verify", "answer"])
        else:  # X1-sum: summaries for uncached units, then select and answer
            assert got[-2:] == ["select", "answer"] and set(got[:-2]) <= {"index_build"}
        if budget is not None:
            assert record["ctx"]["ctx_tokens"] <= budget
    assert any(c["stage"] == "index_build" for c in calls) == (sargs.get("index") == "sum")

    # Both scorings are written from the same answers.
    for name in ("scores.json", "scores_ignore_comments.json"):
        with open(os.path.join(run_dir, name), encoding="utf-8") as f:
            scores = json.load(f)["fake-oracle"]["scores"]["all"]["0.0"]["pass@1"]
        if strategy == "pointer":  # the system copies the picked unit, not the gold
            assert all(r["output"][0].startswith("```") for r in records)
        else:
            assert scores == 1.0
