"""LongLLMLingua (rank_method longllmlingua over line chunks), target_token = budget.

The context is produced by the compress stage (repoqa_ctx/compress, run in .venv-compress)
and read here from the compress cache; see docs/DECISIONS.md (Fase 4).
"""

from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput
from repoqa_ctx.strategies.compressed import from_cache

NEEDS_BUDGET = True
DEFAULT_PARAMS = {"compressor_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "chunk_lines": 20, "overlap_lines": 10}


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    return from_cache("longllmlingua", inp, DEFAULT_PARAMS)
