"""S4 adaptive_k (X3): the number k of whole units kept is chosen per task, then the top-k
units of the S1 ranking (struct_bm25) are rendered within the budget.

signal="margin": relative gap of the two best S1 scores, (s1 - s2) / s1:
                 >= margin_high -> k_levels[0], >= margin_low -> k_levels[1], else k_levels[2]
signal="units":  k = ceil(number of function units / units_per_k), clamped to [1, k_levels[2]]
signal="llm":    one extra call (stage "confidence") shows the top-m S1 candidates (index as in
                 S3) and asks for the best ID with high/medium/low confidence -> k = 1/3/6;
                 the LLM's ID goes first, then the S1 order. Unparseable -> medium, S1 order.
Thresholds are our a-priori choice (docs/DECISIONS.md, R1), not tuned.
"""

import json
import math

from repoqa_ctx.codeparse import split_window
from repoqa_ctx.selection import parse_confidence
from repoqa_ctx.strategies import index_select, struct_bm25
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.strategies.common import greedy_select

NEEDS_BUDGET = True
DEFAULT_PARAMS = {
    "signal": "margin",
    "weights": [1, 1, 1, 1],
    "k_levels": [1, 3, 6],
    "margin_high": 0.5,
    "margin_low": 0.2,
    "units_per_k": 20,
    "m": 20,
    "index": "sig",
    "confidence_max_tokens": 32,
}
CONFIDENCE_INSTRUCTION = (
    "Below is a numbered list of functions from a code repository, followed by a function"
    " description. Return the ID of the function that best matches the description and how"
    ' confident you are, as JSON such as {"id": 7, "confidence": "high"}; confidence is'
    " high, medium or low. Output only the JSON."
)


def confidence_prompt(units, cands, description, index):
    lines = index_select.index_lines(units, cands, index)
    return (
        f"{CONFIDENCE_INSTRUCTION}\n\nFunctions:\n{lines}\n{description}\n{CONFIDENCE_INSTRUCTION}"
    )


def build_context(inp: StrategyInput, llm) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    units = split_window(inp.code_context, inp.language)
    fused = struct_bm25.fused_scores(inp.description, units, {"weights": params["weights"]})
    ranked = sorted(fused, key=lambda i: (-fused[i], i))
    low, mid, high = params["k_levels"]
    meta = {"signal": params["signal"]}
    order = ranked
    if params["signal"] == "margin":
        s1 = fused[ranked[0]] if ranked else 0.0
        s2 = fused[ranked[1]] if len(ranked) > 1 else 0.0
        gap = (s1 - s2) / s1 if s1 > 0 else 0.0
        k = low if gap >= params["margin_high"] else mid if gap >= params["margin_low"] else high
        meta["margin"] = round(gap, 4)
    elif params["signal"] == "units":
        k = min(high, max(1, math.ceil(len(ranked) / params["units_per_k"])))
    elif params["signal"] == "llm":
        cands = sorted(ranked[: params["m"]])
        prompt = confidence_prompt(units, cands, inp.description, params["index"])
        reply = llm.chat(prompt, stage="confidence", max_tokens=params["confidence_max_tokens"])
        best, level = parse_confidence(reply.text, len(cands))
        k = {"high": low, "medium": mid, "low": high}[level or "medium"]
        if best is not None:
            first = cands[best - 1]
            order = [first] + [i for i in ranked if i != first]
        meta.update({"confidence": level, "best_id": best, "reply": reply.text[:200]})
    else:
        raise ValueError(f"signal must be margin, units or llm, got {params['signal']!r}")
    context, selected = greedy_select(units, order[:k], inp.budget)
    meta.update({"k": k, "n_selected": len(selected)})
    return StrategyOutput(code_context=context, meta=json.loads(json.dumps(meta)))
