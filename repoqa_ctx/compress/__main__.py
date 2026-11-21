import argparse
import json
import platform
import time
from datetime import datetime

import torch

from repoqa_ctx import compress_cache
from repoqa_ctx import tasks as T
from repoqa_ctx.compress.methods import OFFICIAL_COMMIT, make_compressor, resolve_device
from repoqa_ctx.strategies import get_strategy
from repoqa_ctx.strategies.base import merge_params

COMPRESS_STRATEGIES = (
    "rag_fn_unixcoder",
    "rag_sw_unixcoder",
    "longcodezip",
    "llmlingua",
    "llmlingua2",
    "longllmlingua",
    "rag_paper_unixcoder",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="compress stage (needs torch)")
    p.add_argument("--strategy", required=True, choices=COMPRESS_STRATEGIES)
    p.add_argument("--budget", type=int, help="required unless the strategy has no budget")
    p.add_argument("--strategy-args", default="{}")
    p.add_argument("--limit", type=int)
    p.add_argument("--languages", type=lambda s: [x for x in s.split(",") if x])
    p.add_argument("--device", default="auto")
    p.add_argument("--code-context-size", type=int, default=16 * 1024)
    p.add_argument("--dataset-path")
    return p.parse_args(argv)


def run_compress(args) -> str:
    module = get_strategy(args.strategy)
    if module.NEEDS_BUDGET and args.budget is None:
        raise SystemExit(f"{args.strategy} needs --budget")
    if not module.NEEDS_BUDGET:
        args.budget = None
    params = merge_params(module.DEFAULT_PARAMS, json.loads(args.strategy_args))
    dataset = T.load_dataset(args.dataset_path)
    tasks = T.build_tasks(dataset, args.code_context_size, args.languages, args.limit)
    done = compress_cache.load(args.strategy, params, args.budget)
    todo = [t for t in tasks if t["uid"] not in done]
    path = compress_cache.cache_path(args.strategy, params, args.budget)
    print(f"{args.strategy} B={args.budget}: {len(tasks)} tasks, {len(todo)} to compress -> {path}")
    if not todo:
        return path
    device = resolve_device(args.device)
    start = time.perf_counter()
    compress = make_compressor(args.strategy, params, device)
    load_s = time.perf_counter() - start
    for k, task in enumerate(todo, 1):
        start = time.perf_counter()
        context, meta, used = compress(task, args.budget)
        entry = {
            "uid": task["uid"],
            "strategy": args.strategy,
            "params": params,
            "budget": args.budget,
            "compressor_model": params["compressor_model"],
            "device": device,
            "compress_s": round(time.perf_counter() - start, 3),
            "compress_tokens": used,
            "model_load_s": round(load_s, 2),
            "code_context": context,
            "meta": meta,
            "versions": {
                "torch": torch.__version__,
                "python": platform.python_version(),
                "longcodezip": OFFICIAL_COMMIT,
            },
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        compress_cache.append(args.strategy, params, args.budget, entry)
        print(f"[{k}/{len(todo)}] {task['language']} {entry['compress_s']}s")
    return path


if __name__ == "__main__":
    run_compress(parse_args())
