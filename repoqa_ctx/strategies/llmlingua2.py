"""LLMLingua-2 (token classification), target_token = budget.

The context is produced by the compress stage (repoqa_ctx/compress, run in .venv-compress)
and read here from the compress cache; see docs/DECISIONS.md (Fase 4).
"""

from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput
from repoqa_ctx.strategies.compressed import from_cache

NEEDS_BUDGET = True
DEFAULT_PARAMS = {
    "compressor_model": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    "force_tokens": ["\n", "?"],  # llmlingua README example; variant: ["?"]
}


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    return from_cache("llmlingua2", inp, DEFAULT_PARAMS)
