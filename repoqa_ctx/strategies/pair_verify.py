"""S6 pair_verify (X4): only when the two best S1 candidates are close, i.e. their relative
score gap (s1 - s2) / s1 < margin, one extra call (stage "verify") shows both full functions
and the description and asks for "A" or "B". both_orders=true asks a second time with A/B
swapped; if the two answers disagree, nothing changes. The winner goes first; with
drop_loser=true the loser is left out of the context. Then whole units in that order within
the budget (like S1). Margin is our a-priori choice (docs/DECISIONS.md, R3).
"""

from repoqa_ctx.codeparse import split_window
from repoqa_ctx.selection import parse_choice
from repoqa_ctx.strategies import struct_bm25
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.strategies.common import greedy_select

NEEDS_BUDGET = True
DEFAULT_PARAMS = {
    "weights": [1, 1, 1, 1],
    "margin": 0.2,
    "both_orders": False,
    "drop_loser": True,
    "verify_max_tokens": 8,
}
VERIFY_INSTRUCTION = (
    "Which candidate is the function described below? Answer with the letter A or B only."
)


def verify_prompt(code_a: str, code_b: str, description: str) -> str:
    return (
        f"{VERIFY_INSTRUCTION}\n\nCandidate A:\n```\n{code_a}\n```\n\n"
        f"Candidate B:\n```\n{code_b}\n```\n{description}\n{VERIFY_INSTRUCTION}"
    )


def build_context(inp: StrategyInput, llm) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    units = split_window(inp.code_context, inp.language)
    fused = struct_bm25.fused_scores(inp.description, units, {"weights": params["weights"]})
    ranked = sorted(fused, key=lambda i: (-fused[i], i))
    meta = {"verified": False}
    order = ranked
    if len(ranked) >= 2 and fused[ranked[0]] > 0:
        first, second = ranked[0], ranked[1]
        gap = (fused[first] - fused[second]) / fused[first]
        meta["margin"] = round(gap, 4)
        if gap < params["margin"]:
            a, b = units[first].text, units[second].text
            winners = []
            reply = llm.chat(
                verify_prompt(a, b, inp.description),
                stage="verify",
                max_tokens=params["verify_max_tokens"],
            )
            winners.append({"A": first, "B": second}.get(parse_choice(reply.text)))
            if params["both_orders"]:
                reply = llm.chat(
                    verify_prompt(b, a, inp.description),
                    stage="verify",
                    max_tokens=params["verify_max_tokens"],
                )
                winners.append({"A": second, "B": first}.get(parse_choice(reply.text)))
            decided = winners[0] if len(set(winners)) == 1 else None
            meta.update({"verified": True, "winner_changed": decided == second})
            if decided is not None:
                loser = second if decided == first else first
                rest = [i for i in ranked if i not in (first, second)]
                order = [decided] + ([] if params["drop_loser"] else [loser]) + rest
    context, selected = greedy_select(units, order, inp.budget)
    meta["n_selected"] = len(selected)
    return StrategyOutput(code_context=context, meta=meta)
