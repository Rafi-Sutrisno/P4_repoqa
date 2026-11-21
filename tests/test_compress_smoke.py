"""Smoke test of the compress stage on CPU with small models, 3 tasks per strategy: it runs,
stays within the budget, and the runner can read the result. Skipped without torch (the main
.venv has none); run it with .venv-compress:
    .venv-compress/Scripts/python -m pytest tests/test_compress_smoke.py
"""

import json
import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from repoqa_ctx import compress_cache  # noqa: E402
from repoqa_ctx import run  # noqa: E402
from repoqa_ctx.tokens import count_tokens  # noqa: E402

BUDGET = 2048
SMALL = {
    "rag_fn_unixcoder": {},
    "rag_sw_unixcoder": {},
    "longcodezip": {"compressor_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct"},
    "llmlingua": {"compressor_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct"},
    "longllmlingua": {"compressor_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct"},
    "llmlingua2": {"compressor_model": "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"},
}


@pytest.mark.parametrize("strategy", list(SMALL))
def test_compress_smoke(strategy, tmp_path, monkeypatch):
    if strategy in ("llmlingua", "llmlingua2", "longllmlingua"):
        pytest.importorskip("llmlingua")
    if strategy == "longcodezip":
        from repoqa_ctx.compress.methods import OFFICIAL_DIR

        if not os.path.isdir(OFFICIAL_DIR):
            pytest.skip("third_party/LongCodeZip missing (scripts/setup_compress.sh)")
    from repoqa_ctx.compress.__main__ import parse_args, run_compress

    monkeypatch.setattr(compress_cache, "COMPRESS_DIR", str(tmp_path / "cache"))
    sargs = json.dumps(SMALL[strategy])
    path = run_compress(
        parse_args(
            ["--strategy", strategy, "--budget", str(BUDGET), "--limit", "3",
             "--device", "cpu", "--strategy-args", sargs]
        )  # fmt: skip
    )
    entries = [json.loads(line) for line in open(path, encoding="utf-8")]
    assert len(entries) == 3
    for entry in entries:
        assert 0 < count_tokens(entry["code_context"]) <= BUDGET
        assert entry["compress_tokens"] > 0 and entry["compress_s"] > 0

    # The runner's strategy of the same name reads the cache (dry-run, no LLM).
    run_dir = run.run_experiment(
        run.parse_args(
            ["--strategy", strategy, "--budget", str(BUDGET), "--limit", "3", "--dry-run",
             "--strategy-args", sargs, "--result-dir", str(tmp_path / "ctx")]
        )  # fmt: skip
    )
    records = run.read_jsonl(os.path.join(run_dir, "dryrun.jsonl"))
    assert len(records) == 3
    for record in records:
        assert record["ctx"]["ctx_tokens"] <= BUDGET
        assert record["ctx"]["meta"]["compress"]["model"] == entries[0]["compressor_model"]
