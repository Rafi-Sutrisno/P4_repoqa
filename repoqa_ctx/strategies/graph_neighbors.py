"""S7 graph_neighbors (X2): a graph inside the window over function units: callees (a name
of another unit followed by "(" in the body), callers (the reverse) and members of the same
class/struct/impl (codeparse ``container``). The top `seeds` units of the S1 ranking come
first, then their 1-hop neighbours (in S1 order), then the rest of the S1 ranking; whole
units are taken in that order until the budget is used. No extra LLM call.
"""

import re
from typing import Dict, List, Set

from repoqa_ctx.codeparse import Unit, split_window
from repoqa_ctx.strategies import struct_bm25
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.strategies.common import greedy_select

NEEDS_BUDGET = True
DEFAULT_PARAMS = {"weights": [1, 1, 1, 1], "seeds": 3}
_CALL = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _short(name: str) -> str:
    return re.split(r"::|\.", name)[-1]


def neighbours(units: List[Unit], funcs: List[int]) -> Dict[int, Set[int]]:
    by_name: Dict[str, Set[int]] = {}
    for i in funcs:
        if units[i].name:
            by_name.setdefault(_short(units[i].name), set()).add(i)
    graph: Dict[int, Set[int]] = {i: set() for i in funcs}
    for i in funcs:
        for called in set(_CALL.findall(units[i].text)):
            for j in by_name.get(called, ()):
                if j != i:
                    graph[i].add(j)  # callee
                    graph[j].add(i)  # caller
    by_container: Dict[str, Set[int]] = {}
    for i in funcs:
        if units[i].container:
            by_container.setdefault(units[i].container, set()).add(i)
    for members in by_container.values():
        for i in members:
            graph[i] |= members - {i}
    return graph


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    units = split_window(inp.code_context, inp.language)
    ranked = struct_bm25.rank_units(inp.description, units, {"weights": params["weights"]})
    position = {i: p for p, i in enumerate(ranked)}
    graph = neighbours(units, ranked)
    seeds = ranked[: params["seeds"]]
    near = sorted({j for s in seeds for j in graph.get(s, ())} - set(seeds), key=position.get)
    order = seeds + near + [i for i in ranked if i not in set(seeds) | set(near)]
    context, selected = greedy_select(units, order, inp.budget)
    return StrategyOutput(
        code_context=context,
        meta={
            "n_neighbours": len(near),
            "n_selected": len(selected),
            "n_neighbours_selected": len(selected & set(near)),
        },
    )
