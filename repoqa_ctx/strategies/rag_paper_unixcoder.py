"""RAG with UniXcoder, PAPER VERSION: exactly the official LongCodeZip RepoQA code
(experiments/repo-qa/main.py:130-207): mean pooling, truncation 512, cosine, top-5 chunks
WITHOUT a token budget, in similarity order, 20-line windows with 10 lines overlap.

Kept apart from our rag_*_unixcoder (budgeted). Its context length is not controlled by B,
so it is compared by its own token count. Produced by the compress stage.
"""

from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput
from repoqa_ctx.strategies.compressed import from_cache

NEEDS_BUDGET = False
TRACK = "paper"
DEFAULT_PARAMS = {
    "compressor_model": "microsoft/unixcoder-base",
    "chunking": "sliding_window",  # main.py:406 offers function_based / sliding_window
    "window_lines": 20,
    "overlap_lines": 10,
    "top_k": 5,
}


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    return from_cache("rag_paper_unixcoder", inp, DEFAULT_PARAMS)
