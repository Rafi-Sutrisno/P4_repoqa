"""Shared helpers: fill a token budget with whole units and render them in original order."""

from typing import Dict, List, Optional, Set, Tuple

from repoqa_ctx.codeparse import Unit
from repoqa_ctx.tokens import count_tokens

GAP = "...\n"  # marks skipped code


def with_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def skeleton_line(unit: Unit) -> str:
    indent = unit.text[: len(unit.text) - len(unit.text.lstrip(" \t"))]
    return f"{indent}{unit.signature} ...\n"


def assemble(units: List[Unit], selected: Set[int], skeleton: Optional[Set[int]] = None) -> str:
    """Selected units (whole) in original order, with the Path header of each used file.

    Without ``skeleton``, "...\n" marks every place where non-empty code was skipped. With
    ``skeleton``, those function units are shown as one signature line and no markers are
    added.
    """
    out = []
    segments: Dict[int, List[int]] = {}
    for i, unit in enumerate(units):
        segments.setdefault(unit.segment, []).append(i)
    shown = selected | (skeleton or set())
    for idxs in segments.values():
        if not any(i in shown for i in idxs):
            continue
        skipped = False
        for i in idxs:
            unit = units[i]
            if unit.kind == "header":
                out.append(with_newline(unit.text))
            elif i in selected:
                if skipped:
                    out.append(GAP)
                    skipped = False
                out.append(with_newline(unit.text))
            elif skeleton is not None:
                if i in skeleton:
                    out.append(skeleton_line(unit))
            elif unit.text.strip():
                skipped = True
        if skipped:
            out.append(GAP)
    return "".join(out)


def greedy_select(units: List[Unit], ranked: List[int], budget: int) -> Tuple[str, Set[int]]:
    """Take every unit in ``ranked`` order that still fits; exact check at the end."""
    gap_cost = count_tokens(GAP)
    header_cost = {
        u.segment: count_tokens(with_newline(u.text)) for u in units if u.kind == "header"
    }
    selected: Set[int] = set()
    used_segments = set()
    used = 0
    for i in ranked:
        seg = units[i].segment
        cost = count_tokens(with_newline(units[i].text)) + gap_cost
        if seg not in used_segments:
            cost += header_cost.get(seg, 0) + gap_cost
        if used + cost <= budget:
            selected.add(i)
            used_segments.add(seg)
            used += cost

    # The estimate can be slightly off; drop the lowest-ranked units until it fits.
    context = assemble(units, selected)
    kept = [i for i in ranked if i in selected]
    while kept and count_tokens(context) > budget:
        selected.discard(kept.pop())
        context = assemble(units, selected)
    return context, selected
