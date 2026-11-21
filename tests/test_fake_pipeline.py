"""End to end with a fake LLM that returns the ground truth: pass@1 must be 1.0."""

import json
import os

from repoqa_ctx import run
from repoqa_ctx import tasks as T


def test_fake_pipeline(mini_dataset, task_cache, tmp_path, monkeypatch):
    _, path = mini_dataset
    monkeypatch.setattr(T, "TASK_CACHE_DIR", task_cache)
    args = run.parse_args(
        [
            "--strategy", "full",
            "--llm", "fake",
            "--fake-mode", "oracle",
            "--dataset-path", path,
            "--result-dir", str(tmp_path / "ctx"),
        ]
    )  # fmt: skip
    run_dir = run.run_experiment(args)

    with open(os.path.join(run_dir, "scores.json"), encoding="utf-8") as f:
        scores = json.load(f)
    # Threshold 0.0: the oracle answer is the best match for every task. At 0.8 even a
    # perfect answer cannot reach 1.0: RepoQA's BLEU gives 0.576 to an exact copy of a
    # 3-token needle (cpp UTF8CharsetEncoder in this subset). See docs/DECISIONS.md.
    assert scores["fake-oracle"]["scores"]["all"]["0.0"]["pass@1"] == 1.0

    # One log line per LLM call: 60 tasks, one answer call each.
    calls = run.read_jsonl(os.path.join(run_dir, "calls.jsonl"))
    assert len(calls) == 60 and all(c["stage"] == "answer" for c in calls)
