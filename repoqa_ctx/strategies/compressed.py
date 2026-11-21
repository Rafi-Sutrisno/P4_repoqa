"""Shared build_context for strategies whose context comes from the compress stage."""

from typing import Dict

from repoqa_ctx import compress_cache
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.tokens import count_tokens

_loaded: Dict[str, Dict[str, Dict]] = {}


def from_cache(name: str, inp: StrategyInput, defaults: Dict) -> StrategyOutput:
    params = merge_params(defaults, inp.params)
    path = compress_cache.cache_path(name, params, inp.budget)
    entries = _loaded.get(path)
    if entries is None or inp.uid not in entries:
        entries = _loaded[path] = compress_cache.load(name, params, inp.budget)
    entry = entries.get(inp.uid)
    if entry is None:
        raise RuntimeError(
            f"no compressed context for task {inp.uid} in {path}; run the compress stage first: "
            f"python -m repoqa_ctx.compress --strategy {name} --budget {inp.budget} ..."
        )
    context = entry["code_context"]
    if count_tokens(context) > inp.budget:
        raise RuntimeError(f"cached context for {inp.uid} exceeds the budget {inp.budget}")
    return StrategyOutput(
        code_context=context,
        meta={
            **entry.get("meta", {}),
            "compress": {
                "model": entry["compressor_model"],
                "device": entry.get("device"),
                "compress_s": entry["compress_s"],
                "compress_tokens": entry["compress_tokens"],
            },
        },
    )
