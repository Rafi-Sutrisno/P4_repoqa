"""RAG Function Chunking with BM25, defined by IR practice rather than tuned on data.

Differences from rag_fn_bm25: only function units are candidates (function chunking), and
the standard English stop set is removed from query and chunks.
"""

from repoqa_ctx.bm25 import ENGLISH_STOP_WORDS, bm25_scores, rank, tokenize
from repoqa_ctx.codeparse import split_window
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.strategies.common import greedy_select

NEEDS_BUDGET = True
DEFAULT_PARAMS = {"k1": 1.5, "b": 0.75}


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    units = split_window(inp.code_context, inp.language)
    cands = [i for i, u in enumerate(units) if u.kind == "function"]
    scores = bm25_scores(
        tokenize(inp.description, ENGLISH_STOP_WORDS),
        [tokenize(units[i].text, ENGLISH_STOP_WORDS) for i in cands],
        k1=params["k1"],
        b=params["b"],
    )
    ranked = [cands[j] for j in rank(scores)]
    context, selected = greedy_select(units, ranked, inp.budget)
    return StrategyOutput(
        code_context=context,
        meta={"n_functions": len(cands), "n_selected": len(selected)},
    )
