"""X1-sum: one-sentence LLM summaries of function units, cached per (model, hash of the unit
text) in results/summary_cache/<model>.jsonl (append-only). Only cache misses call the LLM,
with stage "index_build", so the index-building cost is logged separately and falls on the
first tasks that need a unit; later windows of the same repo reuse it.
"""

import hashlib
import json
import os
import re
from typing import Dict, List

from repoqa_ctx.codeparse import Unit

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY_DIR = os.environ.get("REPOQA_CTX_SUMMARY_DIR") or os.path.join(
    REPO_ROOT, "results", "summary_cache"
)
SUMMARY_MAX_TOKENS = 48
SUMMARY_INSTRUCTION = "Summarize in one sentence what this function does. Output only the sentence."
_loaded: Dict[str, Dict[str, str]] = {}


def _path(model: str) -> str:
    return os.path.join(SUMMARY_DIR, re.sub(r"[^A-Za-z0-9._-]+", "_", model) + ".jsonl")


def _key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\n{text}".encode("utf-8")).hexdigest()


def _cache(model: str) -> Dict[str, str]:
    path = _path(model)
    if path not in _loaded:
        entries: Dict[str, str] = {}
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    entries[item["key"]] = item["summary"]
        _loaded[path] = entries
    return _loaded[path]


def summaries(units: List[Unit], cands: List[int], llm) -> Dict[int, str]:
    """unit index -> one-line summary, calling the LLM only for uncached units."""
    model = llm.model
    cache = _cache(model)
    out = {}
    for i in cands:
        key = _key(model, units[i].text)
        if key not in cache:
            prompt = f"{SUMMARY_INSTRUCTION}\n```\n{units[i].text}\n```\n{SUMMARY_INSTRUCTION}"
            reply = llm.chat(prompt, stage="index_build", max_tokens=SUMMARY_MAX_TOKENS)
            text = " ".join((reply.text or "").split())[:300]
            cache[key] = text
            os.makedirs(SUMMARY_DIR, exist_ok=True)
            with open(_path(model), "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "summary": text}) + "\n")
        out[i] = cache[key]
    return out
