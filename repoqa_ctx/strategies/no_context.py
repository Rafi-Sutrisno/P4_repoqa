"""No Context: empty code window (prompt = instruction + description + instruction)."""

from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput

NEEDS_BUDGET = False
DEFAULT_PARAMS = {}


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    return StrategyOutput(code_context="")
