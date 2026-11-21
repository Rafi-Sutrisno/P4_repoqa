"""S3 cascade: S1 (struct_bm25) keeps the top m function units; S2 (index_select) lets the
LLM pick at most k of them. The index lists the m candidates in original order, so the S1
rank itself is not shown to the LLM. Fallback = S1 order.
"""

from repoqa_ctx.codeparse import split_window
from repoqa_ctx.strategies import index_select, struct_bm25
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params

NEEDS_BUDGET = True
DEFAULT_PARAMS = {"m": 20, **index_select.DEFAULT_PARAMS}


def build_context(inp: StrategyInput, llm) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    units = split_window(inp.code_context, inp.language)
    ranked = struct_bm25.rank_units(inp.description, units, {})
    cands = sorted(ranked[: params["m"]])
    out = index_select.select_and_build(inp, llm, units, cands, ranked, params)
    out.meta["m"] = len(cands)
    return out
