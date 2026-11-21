"""Random Token: keep a random subset of the window's CodeLlama tokens (order kept) so that
the decoded text has at most `budget` tokens. Seeded per task."""

from repoqa_ctx.strategies.base import StrategyInput, StrategyOutput
from repoqa_ctx.tokens import count_tokens, get_tokenizer

NEEDS_BUDGET = True
DEFAULT_PARAMS = {}


def build_context(inp: StrategyInput, llm=None) -> StrategyOutput:
    tok = get_tokenizer()
    ids = tok.encode(inp.code_context, add_special_tokens=False)
    rng = inp.rng()
    order = list(range(len(ids)))
    rng.shuffle(order)
    keep = min(len(ids), inp.budget)
    while True:
        kept = sorted(order[:keep])
        context = tok.decode([ids[i] for i in kept])
        n = count_tokens(context)
        if n <= inp.budget or keep == 0:
            break
        # Re-tokenizing the decoded text can merge or split tokens; shrink and retry.
        keep = max(0, keep - max(1, n - inp.budget))
    return StrategyOutput(code_context=context, meta={"total_tokens": len(ids), "kept_tokens": keep})
