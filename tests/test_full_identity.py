"""Strategy `full` must be identical to the original RepoQA pipeline."""

import json

import pytest

import repoqa.search_needle_function as snf
from repoqa_ctx import run
from repoqa_ctx import tasks as T
from repoqa_ctx.llm import FakeClient


def test_full_identity(mini_dataset, task_cache, tmp_path, monkeypatch):
    _, path = mini_dataset

    # Original pipeline: evaluate_model caches every task before it picks a backend, so an
    # unknown backend gives us the original tasks without calling any model.
    monkeypatch.setattr(snf, "CACHE_DIR", str(tmp_path / "orig_cache"))
    with pytest.raises(ValueError, match="Unknown backend"):
        snf.evaluate_model(
            model="identity-check",
            backend="none",
            dataset_path=path,
            result_dir=str(tmp_path / "orig_results"),
        )
    with open(tmp_path / "orig_cache" / "cache_ntoken_16384_v1.jsonl", encoding="utf-8") as f:
        original = [json.loads(line) for line in f]
    for task in original:
        task.pop("cache_id")

    # Our task builder: same tasks, same order, every original field equal.
    monkeypatch.setattr(T, "TASK_CACHE_DIR", task_cache)
    ours = T.build_tasks(T.load_dataset(path))
    assert len(ours) == len(original) == 60
    for orig, mine in zip(original, ours):
        assert {key: mine[key] for key in orig} == orig

    # Runner with strategy full: the prompt sent to the LLM equals the original prompt.
    fake = FakeClient("empty")
    args = run.parse_args(
        [
            "--strategy", "full",
            "--llm", "fake",
            "--fake-mode", "empty",
            "--dataset-path", path,
            "--result-dir", str(tmp_path / "ctx"),
        ]
    )  # fmt: skip
    run.run_experiment(args, client=fake)
    expected = ["".join(t[key] for key in t["template"].split("\n")) for t in original]
    assert fake.prompts == expected
