from importlib import import_module

NAMES = (
    "full",
    "no_context",
    "rag_fn_bm25",
    "rag_fn_bm25_plus",
    "struct_bm25",
    "index_select",
    "cascade",
    "rag_fn_unixcoder",
    "longcodezip",
    "rag_sw_bm25",
    "rag_sw_unixcoder",
    "llmlingua",
    "llmlingua2",
    "longllmlingua",
    "random_line",
    "random_token",
)


def get_strategy(name: str):
    if name not in NAMES:
        raise ValueError(f"unknown strategy {name!r}; choose from {NAMES}")
    return import_module(f"repoqa_ctx.strategies.{name}")
