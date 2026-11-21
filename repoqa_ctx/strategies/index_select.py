"""S2 index_select (X1): an LLM picks functions from a numbered index; the answer prompt
then contains only the picked functions.

Call 1 (stage "select"): the function units of the window in original order, one line each,
    [7] src/pkg/file.py: def foo(a, b):
plus, with index="doc", the first line of the unit's docstring/comments as "    doc: ...".
The LLM returns at most k IDs as a JSON list. If no usable ID comes back, the S1
(struct_bm25) ranking is used instead. Call 2 (stage "answer") is made by the runner with the
original RepoQA prompt; its code_context holds the picked units (LLM order, within budget),
rendered in original order with Path headers.
"""

import re
from typing import Dict, List

from repoqa_ctx.codeparse import Unit, split_window
from repoqa_ctx.selection import parse_ids
from repoqa_ctx.strategies import struct_bm25
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput, merge_params
from repoqa_ctx.strategies.common import greedy_select

NEEDS_BUDGET = True
DEFAULT_PARAMS = {"index": "sig", "k": 10, "select_max_tokens": 128}

SELECT_INSTRUCTION = (
    "Below is a numbered list of functions from a code repository, followed by a function"
    " description. Return the IDs of the functions that best match the description, best"
    " first, at most {k} IDs, as a JSON list of integers such as [3, 17]."
    " Output only the JSON list."
)
_COMMENT_MARKS = re.compile(r"^\s*(?://[/!]?|/\*+|\*+/?|#+|\"\"\"|'''|r\"\"\")\s*")
MAX_LINE_CHARS = 300


def first_doc_line(doc: str) -> str:
    for line in doc.split("\n"):
        cleaned = _COMMENT_MARKS.sub("", line).strip().strip("\"'").strip()
        cleaned = cleaned.removesuffix("*/").strip()
        if cleaned:
            return cleaned[:MAX_LINE_CHARS]
    return ""


def index_lines(units: List[Unit], cands: List[int], index: str) -> str:
    lines = []
    for number, i in enumerate(cands, 1):
        unit = units[i]
        lines.append(f"[{number}] {unit.path or '?'}: {unit.signature[:MAX_LINE_CHARS]}")
        if index == "doc":
            doc = first_doc_line(unit.doc)
            if doc:
                lines.append(f"    doc: {doc}")
    return "\n".join(lines)


def select_prompt(units: List[Unit], cands: List[int], description: str, params: Dict) -> str:
    instruction = SELECT_INSTRUCTION.format(k=params["k"])
    index = index_lines(units, cands, params["index"])
    return f"{instruction}\n\nFunctions:\n{index}\n{description}\n{instruction}"


def select_and_build(
    inp: StrategyInput, llm, units: List[Unit], cands: List[int], fallback: List[int], params
) -> StrategyOutput:
    """Ask the LLM to pick from ``cands`` (unit indices, original order); build the context."""
    if params["index"] not in ("sig", "doc"):
        raise ValueError(f"index must be 'sig' or 'doc', got {params['index']!r}")
    prompt = select_prompt(units, cands, inp.description, params)
    reply = llm.chat(prompt, stage="select", max_tokens=params["select_max_tokens"])
    ids = parse_ids(reply.text, len(cands), params["k"])
    picked = [cands[i - 1] for i in ids]
    used_fallback = not picked
    if used_fallback:
        picked = fallback[: params["k"]]
    context, selected = greedy_select(units, picked, inp.budget)
    return StrategyOutput(
        code_context=context,
        meta={
            "n_index": len(cands),
            "picked_ids": ids,
            "fallback": used_fallback,
            "n_selected": len(selected),
            "select_reply": reply.text[:300],
        },
    )


def build_context(inp: StrategyInput, llm) -> StrategyOutput:
    params = merge_params(DEFAULT_PARAMS, inp.params)
    units = split_window(inp.code_context, inp.language)
    cands = [i for i, u in enumerate(units) if u.kind == "function"]
    fallback = struct_bm25.rank_units(inp.description, units, {})
    return select_and_build(inp, llm, units, cands, fallback, params)
