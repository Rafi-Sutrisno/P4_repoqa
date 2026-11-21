"""S8 pointer, TOOL-ASSISTED TRACK ("dibantu alat"): the LLM only answers one ID (stage
"select") from the index of the top-m S1 candidates (as in S3); the system then copies that
unit's text as the final answer, so no answer call is made. Reported separately: the copy is
done by the system, not by the model (docs/PLAN.md S8). Unparseable reply -> S1 top unit.
"""

import textwrap

from repoqa_ctx.codeparse import split_window
from repoqa_ctx.selection import parse_ids
from repoqa_ctx.strategies import index_select, struct_bm25
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.strategies.common import assemble

NEEDS_BUDGET = False
TRACK = "tool-assisted"
DEFAULT_PARAMS = {"weights": [1, 1, 1, 1], "m": 20, "index": "sig", "select_max_tokens": 32}


def build_context(inp: StrategyInput, llm) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    units = split_window(inp.code_context, inp.language)
    ranked = struct_bm25.rank_units(inp.description, units, {"weights": params["weights"]})
    cands = sorted(ranked[: params["m"]])
    prompt = index_select.select_prompt(units, cands, inp.description, {**params, "k": 1})
    reply = llm.chat(prompt, stage="select", max_tokens=params["select_max_tokens"])
    ids = parse_ids(reply.text, len(cands), 1)
    pick = cands[ids[0] - 1] if ids else (ranked[0] if ranked else None)
    if pick is None:
        return StrategyOutput(code_context="", meta={"fallback": True}, final_answer="")
    answer = "```\n" + textwrap.dedent(units[pick].text) + "\n```"
    return StrategyOutput(
        code_context=assemble(units, {pick}),  # what the answer was copied from
        meta={"picked_id": ids[0] if ids else None, "fallback": not ids},
        final_answer=answer,
    )
