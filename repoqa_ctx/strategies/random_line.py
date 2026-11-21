"""Random Line: drop random lines (seeded per task) until the window fits the budget.

Path header lines are never dropped; line order is kept.
"""

import re

from repoqa_ctx.codeparse import header_regex
from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput
from repoqa_ctx.tokens import count_tokens

NEEDS_BUDGET = True
DEFAULT_PARAMS = {}

_LINES = re.compile(r"[^\n]*\n|[^\n]+$")


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    lines = _LINES.findall(inp.code_context)
    header = header_regex(inp.language)
    removable = [i for i, line in enumerate(lines) if not header.fullmatch(line)]
    inp.rng().shuffle(removable)

    # Same per-line counting as make_code_context, then an exact check on the result.
    costs = [count_tokens(line) for line in lines]
    total = sum(costs)
    dropped = set()
    order = iter(removable)
    for i in order:
        if total <= inp.budget:
            break
        dropped.add(i)
        total -= costs[i]

    def render() -> str:
        return "".join(line for i, line in enumerate(lines) if i not in dropped)

    context = render()
    for i in order:
        if count_tokens(context) <= inp.budget:
            break
        dropped.add(i)
        context = render()
    return StrategyOutput(
        code_context=context,
        meta={"total_lines": len(lines), "kept_lines": len(lines) - len(dropped)},
    )
