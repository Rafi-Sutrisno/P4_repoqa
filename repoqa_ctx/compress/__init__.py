"""Compress stage: torch-based baselines (UniXcoder RAG, LongCodeZip, LLMLingua family).

Run in .venv-compress:
    python -m repoqa_ctx.compress --strategy longcodezip --budget 2048 [--limit 30]
        [--strategy-args JSON] [--device auto|cpu|cuda] [--languages ...]
Results go to the compress cache (repoqa_ctx/compress_cache.py); the runner's strategies of
the same name read them. Already compressed tasks are skipped; nothing is overwritten.
"""
