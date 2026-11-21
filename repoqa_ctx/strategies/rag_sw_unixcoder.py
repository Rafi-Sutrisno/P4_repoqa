"""RAG Sliding Window (lines) with UniXcoder (cosine), following LongCodeZip's RepoQA code.

The context is produced by the compress stage (repoqa_ctx/compress, run in .venv-compress)
and read here from the compress cache; see docs/DECISIONS.md (Fase 4).
"""

from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput
from repoqa_ctx.strategies.compressed import from_cache

NEEDS_BUDGET = True
DEFAULT_PARAMS = {"compressor_model": "microsoft/unixcoder-base", "window_lines": 20, "overlap_lines": 10}


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    return from_cache("rag_sw_unixcoder", inp, DEFAULT_PARAMS)
