"""S1 struct_bm25: field-wise BM25 over function units, fused with Reciprocal Rank Fusion.

Fields: name (camelCase/snake_case split), signature, doc (docstring and comments) and body
(the whole unit). Each field is ranked by BM25 against the description, with the standard
English stop set removed; a unit with a non-zero score in a field gets 1 / (rrf_k + rank)
from that field (rrf_k = 60, the constant of the original RRF paper; no field weights).
Whole units are taken in fused order while they fit the budget and rendered in original
order with Path headers. skeleton=true: the remaining budget shows other functions as one
signature line each, in fused order.
"""

from typing import Dict, List

from repoqa_ctx.bm25 import ENGLISH_STOP_WORDS, bm25_scores, rank, tokenize
from repoqa_ctx.codeparse import Unit, split_window
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.strategies.common import assemble, greedy_select, skeleton_line, with_newline
from repoqa_ctx.tokens import count_tokens

NEEDS_BUDGET = True
DEFAULT_PARAMS = {"skeleton": False, "rrf_k": 60, "k1": 1.5, "b": 0.75}
FIELDS = ("name", "signature", "doc", "body")


def _field(unit: Unit, field: str) -> str:
    return unit.text if field == "body" else getattr(unit, field)


def rank_units(description: str, units: List[Unit], params: Dict) -> List[int]:
    """Function-unit indices in fused order (also used by index_select and cascade)."""
    params = merge_params(DEFAULT_PARAMS, params)
    cands = [i for i, u in enumerate(units) if u.kind == "function"]
    query = tokenize(description, ENGLISH_STOP_WORDS)
    fused = {i: 0.0 for i in cands}
    for field in FIELDS:
        docs = [tokenize(_field(units[i], field), ENGLISH_STOP_WORDS) for i in cands]
        scores = bm25_scores(query, docs, k1=params["k1"], b=params["b"])
        for position, j in enumerate(rank(scores), 1):
            if scores[j] <= 0:
                break
            fused[cands[j]] += 1.0 / (params["rrf_k"] + position)
    return sorted(cands, key=lambda i: (-fused[i], i))


def _add_skeleton(units: List[Unit], ranked: List[int], selected: set, budget: int):
    """Fill the budget left after the whole units with signature lines, in fused order."""
    context = assemble(units, selected, set())
    used = count_tokens(context)
    segments = {units[i].segment for i in selected}
    header_cost = {
        u.segment: count_tokens(with_newline(u.text)) for u in units if u.kind == "header"
    }
    skeleton = set()
    for i in ranked:
        if i in selected or not units[i].signature:
            continue
        cost = count_tokens(skeleton_line(units[i]))
        if units[i].segment not in segments:
            cost += header_cost.get(units[i].segment, 0)
        if used + cost <= budget:
            skeleton.add(i)
            segments.add(units[i].segment)
            used += cost
    context = assemble(units, selected, skeleton)
    kept = [i for i in ranked if i in skeleton]
    while kept and count_tokens(context) > budget:
        skeleton.discard(kept.pop())
        context = assemble(units, selected, skeleton)
    return context, skeleton


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    units = split_window(inp.code_context, inp.language)
    ranked = rank_units(inp.description, units, params)
    context, selected = greedy_select(units, ranked, inp.budget)
    skeleton = set()
    if params["skeleton"]:
        context, skeleton = _add_skeleton(units, ranked, selected, inp.budget)
    return StrategyOutput(
        code_context=context,
        meta={
            "n_functions": len(ranked),
            "n_selected": len(selected),
            "n_skeleton": len(skeleton),
        },
    )
