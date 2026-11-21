from importlib import import_module

NAMES = ("full", "no_context", "random_line", "rag_fn_bm25")


def get_strategy(name: str):
    if name not in NAMES:
        raise ValueError(f"unknown strategy {name!r}; choose from {NAMES}")
    return import_module(f"repoqa_ctx.strategies.{name}")
