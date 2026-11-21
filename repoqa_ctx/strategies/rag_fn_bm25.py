"""RAG Function Chunking with BM25.

Chunks = function units plus the non-empty code between them (codeparse). Chunks are ranked
by BM25 against the description and added greedily while they fit the budget. The result
keeps the original order, the Path header of every file that contributes a chunk, and a
"...\n" marker wherever code was skipped.
"""

from typing import List, Set

from repoqa_ctx.bm25 import bm25_scores, tokenize
from repoqa_ctx.codeparse import Unit, split_window
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.tokens import count_tokens

NEEDS_BUDGET = True
DEFAULT_PARAMS = {"k1": 1.5, "b": 0.75}

GAP = "...\n"


def _with_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def assemble(units: List[Unit], selected: Set[int]) -> str:
    out = []
    segments = {}
    for i, unit in enumerate(units):
        segments.setdefault(unit.segment, []).append(i)
    for idxs in segments.values():
        if not any(i in selected for i in idxs):
            continue
        skipped = False
        for i in idxs:
            unit = units[i]
            if unit.kind == "header":
                out.append(_with_newline(unit.text))
            elif i in selected:
                if skipped:
                    out.append(GAP)
                    skipped = False
                out.append(_with_newline(unit.text))
            elif unit.text.strip():
                skipped = True
        if skipped:
            out.append(GAP)
    return "".join(out)


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
    ranked = [cands[j] for j in sorted(range(len(cands)), key=lambda j: (-scores[j], j))]

    gap_cost = count_tokens(GAP)
    header_cost = {
        u.segment: count_tokens(_with_newline(u.text)) for u in units if u.kind == "header"
    }
    selected: Set[int] = set()
    used_segments = set()
    used = 0
    for i in ranked:  # greedy: take every chunk that still fits
        seg = units[i].segment
        cost = count_tokens(_with_newline(units[i].text)) + gap_cost
        if seg not in used_segments:
            cost += header_cost.get(seg, 0) + gap_cost
        if used + cost <= inp.budget:
            selected.add(i)
            used_segments.add(seg)
            used += cost

    # Exact check; drop the lowest-ranked chunks if the estimate was too optimistic.
    context = assemble(units, selected)
    kept = [i for i in ranked if i in selected]
    while kept and count_tokens(context) > inp.budget:
        selected.discard(kept.pop())
        context = assemble(units, selected)
    return StrategyOutput(
        code_context=context,
        meta={
            "n_chunks": len(cands),
            "n_selected": len(selected),
            "n_functions_selected": sum(units[i].kind == "function" for i in selected),
        },
    )
