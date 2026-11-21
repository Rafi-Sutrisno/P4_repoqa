"""No Compression: the original RepoQA window, unchanged."""

from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput

NEEDS_BUDGET = False
DEFAULT_PARAMS = {}


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    return StrategyOutput(code_context=inp.code_context)
