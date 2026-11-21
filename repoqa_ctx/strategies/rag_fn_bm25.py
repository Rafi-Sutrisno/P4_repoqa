"""RAG Function Chunking with BM25.

Chunks = function units plus the non-empty code between them (codeparse). Chunks are ranked
by BM25 against the description and added greedily while they fit the budget. The result
keeps the original order, the Path header of every file that contributes a chunk, and a
"...\n" marker wherever code was skipped.
"""

from repoqa_ctx.bm25 import bm25_scores, rank, tokenize
from repoqa_ctx.codeparse import split_window
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.strategies.common import greedy_select

NEEDS_BUDGET = True
DEFAULT_PARAMS = {"k1": 1.5, "b": 0.75}


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    units = split_window(inp.code_context, inp.language)
    cands = [i for i, u in enumerate(units) if u.kind != "header" and u.text.strip()]
    scores = bm25_scores(
        tokenize(inp.description),
        [tokenize(units[i].text) for i in cands],
        k1=params["k1"],
        b=params["b"],
    )
    ranked = [cands[j] for j in rank(scores)]
    context, selected = greedy_select(units, ranked, inp.budget)
    return StrategyOutput(
        code_context=context,
        meta={
            "n_chunks": len(cands),
            "n_selected": len(selected),
            "n_functions_selected": sum(units[i].kind == "function" for i in selected),
        },
    )
