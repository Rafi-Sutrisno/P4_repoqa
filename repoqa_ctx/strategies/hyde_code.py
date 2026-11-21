"""S5 hyde_code: one cheap extra call (stage "hyde") sees only the description and writes a
guessed signature and body. The guess is an extra BM25 query, fused by RRF with the
description query in the S1 field rankings (struct_bm25). Then:
  then="s1": whole units in fused order within the budget (like S1)
  then="s3": top-m fused candidates -> LLM select (like S3, stage "select")
"""

import re

from repoqa_ctx.codeparse import split_window
from repoqa_ctx.strategies import index_select, struct_bm25
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.strategies.common import greedy_select

NEEDS_BUDGET = True
DEFAULT_PARAMS = {
    "then": "s1",
    "weights": [1, 1, 1, 1],
    "hyde_max_tokens": 256,
    "m": 20,
    **index_select.DEFAULT_PARAMS,
}
_FENCE = re.compile(r"```[^\n]*\n(.*?)(?:```|$)", re.S)


def hyde_prompt(language: str, description: str) -> str:
    instruction = (
        f"Write a plausible {language} implementation (signature and body) of the function"
        " described below. Output only the code in one code block."
    )
    return f"{instruction}\n{description}\n{instruction}"


def guessed_code(text: str) -> str:
    match = _FENCE.search(text or "")
    return match.group(1) if match else (text or "")


def build_context(inp: StrategyInput, llm) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    reply = llm.chat(
        hyde_prompt(inp.language, inp.description),
        stage="hyde",
        max_tokens=params["hyde_max_tokens"],
    )
    guess = guessed_code(reply.text)
    units = split_window(inp.code_context, inp.language)
    ranked = struct_bm25.rank_units(
        inp.description, units, {"weights": params["weights"]}, extra_queries=[guess]
    )
    if params["then"] == "s1":
        context, selected = greedy_select(units, ranked, inp.budget)
        out = StrategyOutput(code_context=context, meta={"n_selected": len(selected)})
    elif params["then"] == "s3":
        cands = sorted(ranked[: params["m"]])
        out = index_select.select_and_build(inp, llm, units, cands, ranked, params)
    else:
        raise ValueError(f"then must be s1 or s3, got {params['then']!r}")
    out.meta.update({"then": params["then"], "guess_chars": len(guess)})
    return out
