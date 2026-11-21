"""Cache of compressed contexts produced by the separate "compress" stage.

The compress stage (python -m repoqa_ctx.compress, run in .venv-compress with torch) writes
one JSONL file per (strategy, params incl. compressor model, budget) with one line per task.
The runner's compressed strategies only read this cache, so the main venv needs no torch.
Lines are only appended, never rewritten.
"""

import hashlib
import json
import os
import re
from typing import Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPRESS_DIR = os.environ.get("REPOQA_CTX_COMPRESS_DIR") or os.path.join(
    REPO_ROOT, "results", "compress_cache"
)


def cache_path(strategy: str, params: Dict, budget: int) -> str:
    blob = json.dumps(params, sort_keys=True)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
    model = re.sub(r"[^A-Za-z0-9._-]+", "_", params.get("compressor_model", "none"))
    return os.path.join(COMPRESS_DIR, f"{strategy}_B{budget}_{model}_{digest}.jsonl")


def load(strategy: str, params: Dict, budget: int) -> Dict[str, Dict]:
    """uid -> entry; a truncated last line is skipped."""
    path = cache_path(strategy, params, budget)
    entries: Dict[str, Dict] = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                entries[entry["uid"]] = entry
    return entries


def append(strategy: str, params: Dict, budget: int, entry: Dict) -> None:
    path = cache_path(strategy, params, budget)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                with open(path, "ab") as g:
                    g.write(b"\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
