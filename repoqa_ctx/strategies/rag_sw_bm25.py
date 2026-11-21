"""RAG Sliding Window with BM25.

The window is cut into fixed-size token windows (RepoQA tokenizer) with 50% overlap.
Windows are ranked by BM25 against the description and added greedily while their union
fits the budget. Overlapping or touching windows are merged; "...\n" marks skipped text.
Same retriever settings as rag_fn_bm25 (no stop words).
"""

from typing import List, Set, Tuple

from repoqa_ctx.bm25 import bm25_scores, rank, tokenize
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.strategies.common import GAP, with_newline
from repoqa_ctx.tokens import count_tokens, get_tokenizer

NEEDS_BUDGET = True
DEFAULT_PARAMS = {"window": 256, "k1": 1.5, "b": 0.75}


def _token_starts(text: str) -> List[int]:
    enc = get_tokenizer()(text, add_special_tokens=False, return_offsets_mapping=True)
    starts, last = [], 0
    for start, _ in enc["offset_mapping"]:
        last = max(last, start)
        starts.append(last)
    if starts:
        starts[0] = 0
    return starts


def _intervals(windows: List[Tuple[int, int]], chosen: Set[int]) -> List[Tuple[int, int]]:
    merged: List[Tuple[int, int]] = []
    for lo, hi in sorted(windows[w] for w in chosen):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def _render(text: str, starts: List[int], spans: List[Tuple[int, int]]) -> str:
    n = len(starts)

    def char(t: int) -> int:
        return starts[t] if t < n else len(text)

    out = []
    for k, (lo, hi) in enumerate(spans):
        if lo > 0 and (k == 0 or spans[k - 1][1] < lo):
            out.append(GAP)
        out.append(with_newline(text[char(lo) : char(hi)]))
    if spans and spans[-1][1] < n:
        out.append(GAP)
    return "".join(out)


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    size = params["window"]
    stride = size // 2
    text = inp.code_context
    starts = _token_starts(text)
    n = len(starts)
    windows = []  # token ranges [lo, hi)
    lo = 0
    while True:
        windows.append((lo, min(lo + size, n)))
        if lo + size >= n:
            break
        lo += stride
    char = lambda t: starts[t] if t < n else len(text)  # noqa: E731
    scores = bm25_scores(
        tokenize(inp.description),
        [tokenize(text[char(a) : char(b)]) for a, b in windows],
        k1=params["k1"],
        b=params["b"],
    )
    ranked = rank(scores)

    gap_cost = count_tokens(GAP)
    chosen: Set[int] = set()
    for w in ranked:  # greedy on the token union of the chosen windows
        spans = _intervals(windows, chosen | {w})
        cost = sum(hi - lo for lo, hi in spans) + gap_cost * (len(spans) + 1)
        if cost <= inp.budget:
            chosen.add(w)

    context = _render(text, starts, _intervals(windows, chosen))
    kept = [w for w in ranked if w in chosen]
    while kept and count_tokens(context) > inp.budget:
        chosen.discard(kept.pop())
        context = _render(text, starts, _intervals(windows, chosen))
    return StrategyOutput(
        code_context=context,
        meta={"n_windows": len(windows), "n_selected": len(chosen)},
    )
