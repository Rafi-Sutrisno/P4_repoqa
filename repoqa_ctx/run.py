"""Run a context strategy on RepoQA Search Needle Function.

Every run writes to a new timestamped folder under --result-dir and never overwrites older
results. Resuming:
  --resume auto   continue the latest unfinished run with identical settings, or skip it
                  if it already finished (makes batch scripts safe to re-run)
  --resume DIR    continue that run folder with the settings stored in it

Folder contents:
  run_config.json   settings, git commit, Ollama version, `ollama ps` offload, seed (once)
  resume_*.json     git commit / Ollama version at each resume
  outputs.jsonl     one record per task (RepoQA output format + extra fields)
  dryrun.jsonl      same records without "output" (--dry-run)
  calls.jsonl       one line per LLM call
  scores.json       original repoqa.compute_score output, ignore_comments=False
  scores_ignore_comments.json   same answers, ignore_comments=True (as the paper)
  complete.json     written when all tasks are done
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from repoqa_ctx import tasks as T
from repoqa_ctx.llm import CallLogger, FakeClient, LLMClient, OllamaClient, model_num_ctx
from repoqa_ctx.strategies import get_strategy
from repoqa_ctx.strategies.base import StrategyInput, merge_params
from repoqa_ctx.tokens import count_tokens

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--strategy", required=True)
    p.add_argument("--strategy-args", default="{}", help="JSON dict of strategy parameters")
    p.add_argument("--budget", type=int, help="max code-context tokens (CodeLlama tokenizer)")
    p.add_argument("--dry-run", action="store_true", help="build contexts only, no LLM call")
    p.add_argument("--limit", type=int, help="N tasks spread evenly over the languages")
    p.add_argument("--languages", type=lambda s: [x for x in s.split(",") if x])
    p.add_argument("--model", help="Ollama model name, e.g. qwen2.5-coder:7b-instruct")
    p.add_argument("--llm", choices=["ollama", "fake"], default="ollama")
    p.add_argument(
        "--fake-mode",
        choices=FakeClient.MODES,
        default="oracle",
        help="fake LLM behaviour (--llm fake, and the select stage of --dry-run)",
    )
    p.add_argument("--ollama-url", default="http://localhost:11434")
    p.add_argument("--num-ctx", type=int, help="default: per model, repoqa_ctx/models.json")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--code-context-size", type=int, default=16 * 1024)
    p.add_argument("--result-dir", default=os.path.join("results", "ctx"))
    p.add_argument("--dataset-path")
    p.add_argument("--resume", help="'auto' or a run folder")
    return p.parse_args(argv)


def read_jsonl(path: str) -> List[Dict]:
    """Read JSONL, skipping a truncated line left by an interrupted run."""
    if not os.path.exists(path):
        return []
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"warning: skipping unreadable line in {path}")
    return items


def ensure_trailing_newline(path: str) -> None:
    """After an interrupted write, start new lines on a fresh line (never rewrite old data)."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
        if last != b"\n":
            with open(path, "ab") as f:
                f.write(b"\n")


def build_prompt(task: Dict, code_context: str) -> str:
    """Same assembly as evaluate_model: template keys concatenated without separator."""
    fields = {**task, "code_context": code_context}
    return "".join(fields[key] for key in task["template"].split("\n"))


def needle_retained(gold: str, code_context: str) -> bool:
    """Is the whole needle still present in the context, ignoring all whitespace?

    Whitespace is removed entirely because make_code_context itself inserts a newline at
    the needle start, which splits a token when the byte offset is misaligned.
    """
    return "".join(gold.split()) in "".join(code_context.split())


def model_name(args: argparse.Namespace) -> Optional[str]:
    if args.dry_run:
        return None
    if args.llm == "fake":
        return f"fake-{args.fake_mode}"
    if not args.model:
        raise SystemExit("--model is required with --llm ollama")
    return args.model


def resolve_num_ctx(args: argparse.Namespace) -> int:
    return args.num_ctx if args.num_ctx is not None else model_num_ctx(args.model)


def run_identity(args: argparse.Namespace, strategy) -> Dict:
    """Settings that define a run; --resume auto only continues runs with equal identity."""
    llm = "none" if args.dry_run else args.llm
    return {
        "strategy": args.strategy,
        "params": merge_params(strategy.DEFAULT_PARAMS, json.loads(args.strategy_args)),
        "budget": args.budget,
        "model": model_name(args),
        "llm": llm,
        "fake_mode": args.fake_mode if llm in ("fake", "none") else None,
        "num_ctx": resolve_num_ctx(args) if llm == "ollama" else None,
        "max_new_tokens": None if args.dry_run else args.max_new_tokens,
        "seed": args.seed,
        "code_context_size": args.code_context_size,
        "languages": args.languages,
        "limit": args.limit,
        "dataset_path": args.dataset_path,
        "dry_run": args.dry_run,
    }


def git_info():
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return commit, bool(dirty)
    except Exception:
        return None, None


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def new_run_dir(result_dir: str, ident: Dict) -> str:
    parts = [datetime.now().strftime("%Y%m%d-%H%M%S"), ident["strategy"]]
    if ident["budget"] is not None:
        parts.append(f"B{ident['budget']}")
    parts.append("dry" if ident["dry_run"] else _safe(ident["model"]))
    base = os.path.join(result_dir, "_".join(parts))
    path, k = base, 2
    while True:
        try:
            os.makedirs(path)  # fails if it exists: never reuse a folder
            return path
        except FileExistsError:
            path, k = f"{base}-{k}", k + 1


def find_resumable(result_dir: str, ident: Dict) -> Optional[str]:
    if not os.path.isdir(result_dir):
        return None
    for name in sorted(os.listdir(result_dir), reverse=True):
        cfg_path = os.path.join(result_dir, name, "run_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                if json.load(f)["identity"] == ident:
                    return os.path.join(result_dir, name)
    return None


def _write_new(path: str, obj: Dict) -> None:
    with open(path, "x", encoding="utf-8") as f:  # "x": never overwrite
        json.dump(obj, f, indent=2)
        f.write("\n")


def make_client(args: argparse.Namespace) -> LLMClient:
    if args.dry_run or args.llm == "fake":
        return FakeClient(args.fake_mode)  # dry-run: only select stages, never the answer
    return OllamaClient(
        model_name(args), url=args.ollama_url, num_ctx=resolve_num_ctx(args), seed=args.seed
    )


def ollama_stamp(client: LLMClient) -> Dict:
    """Ollama version, warm-up time and offload status (`ollama ps`) after loading."""
    if not isinstance(client, OllamaClient):
        return {"ollama_version": None}
    start = time.perf_counter()
    client.chat("Reply with OK.", max_tokens=1)  # loads the model with this num_ctx
    return {
        "ollama_version": client.version(),
        "warmup_s": round(time.perf_counter() - start, 2),
        "ollama_ps": client.ps(),
    }


def run_experiment(args: argparse.Namespace, client: Optional[LLMClient] = None) -> str:
    if args.resume and args.resume != "auto":
        with open(os.path.join(args.resume, "run_config.json"), encoding="utf-8") as f:
            saved = json.load(f)["args"]
        run_dir = args.resume
        args = argparse.Namespace(**{**saved, "resume": run_dir})
    else:
        run_dir = None

    strategy = get_strategy(args.strategy)
    if strategy.NEEDS_BUDGET and args.budget is None:
        raise SystemExit(f"strategy {args.strategy} needs --budget")
    if not strategy.NEEDS_BUDGET and args.budget is not None:
        raise SystemExit(f"strategy {args.strategy} does not take --budget")
    ident = run_identity(args, strategy)

    if args.resume == "auto":
        run_dir = find_resumable(args.result_dir, ident)
    if run_dir and os.path.exists(os.path.join(run_dir, "complete.json")):
        print(f"Already complete, skipping: {run_dir}")
        return run_dir

    client = client or make_client(args)
    commit, dirty = git_info()
    stamp = {"git_commit": commit, "git_dirty": dirty, **ollama_stamp(client)}
    if run_dir is None:
        run_dir = new_run_dir(args.result_dir, ident)
        _write_new(
            os.path.join(run_dir, "run_config.json"),
            {
                "identity": ident,
                "args": vars(args),
                "seed": args.seed,
                **stamp,
                "python": sys.version,
                "started_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
        print(f"New run: {run_dir}")
    else:
        now = datetime.now()
        _write_new(
            os.path.join(run_dir, f"resume_{now.strftime('%Y%m%d-%H%M%S')}.json"),
            {**stamp, "resumed_at": now.isoformat(timespec="seconds")},
        )
        print(f"Resuming: {run_dir}")

    dataset = T.load_dataset(args.dataset_path)
    tasks = T.build_tasks(dataset, args.code_context_size, args.languages, args.limit)
    split = T.load_split()
    gold = T.gold_functions(dataset)
    records_path = os.path.join(run_dir, "dryrun.jsonl" if args.dry_run else "outputs.jsonl")
    done = {r["uid"] for r in read_jsonl(records_path)}
    calls_path = os.path.join(run_dir, "calls.jsonl")
    for path in (records_path, calls_path):
        ensure_trailing_newline(path)
    logger = CallLogger(client, calls_path)

    todo = [t for t in tasks if t["uid"] not in done]
    print(f"{len(tasks)} tasks, {len(done)} done, {len(todo)} to go")
    for k, task in enumerate(todo, 1):
        uid = task["uid"]
        inp = StrategyInput(
            uid=uid,
            language=task["language"],
            code_context=task["code_context"],
            description=task["description"],
            budget=args.budget,
            params=ident["params"],
            seed=args.seed,
        )
        logger.context = {"uid": uid, "strategy": args.strategy}
        logger.task_calls = []
        if isinstance(client, FakeClient):
            client.gold = gold[uid]  # fake only; strategies never see it
        start = time.perf_counter()
        out = strategy.build_context(inp, logger)
        build_s = time.perf_counter() - start

        prompt = build_prompt(task, out.code_context)
        record = {key: task[key] for key in T.ORIGINAL_KEYS}
        record["code_context"] = out.code_context
        record.update(
            {
                "uid": uid,
                "split": T.split_of(task, split),
                "strategy": args.strategy,
                "params": ident["params"],
                "budget": args.budget,
                "ctx": {
                    "orig_ctx_tokens": count_tokens(task["code_context"]),
                    "ctx_tokens": count_tokens(out.code_context),
                    "prompt_tokens": count_tokens(prompt),
                    "needle_retained": needle_retained(gold[uid], out.code_context),
                    "build_s": build_s,
                    "meta": out.meta,
                },
            }
        )
        if out.final_answer is not None:  # tool-assisted: the system copies the answer
            record["ctx"]["meta"]["final_answer"] = True
            if not args.dry_run:
                record["output"] = [out.final_answer]
        elif not args.dry_run:
            reply = logger.chat(prompt, stage="answer", max_tokens=args.max_new_tokens)
            record["output"] = [reply.text]
        record["calls"] = logger.task_calls
        with open(records_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        print(
            f"[{k}/{len(todo)}] {task['language']} ctx={record['ctx']['ctx_tokens']} "
            f"retained={record['ctx']['needle_retained']}"
        )

    records = read_jsonl(records_path)
    if not args.dry_run:
        from repoqa.compute_score import compute_score

        # Same answers, two scorings: RepoQA default and comment-free (as the LongCodeZip paper).
        for name, ignore in (("scores", False), ("scores_ignore_comments", True)):
            scores = compute_score(ident["model"], dataset, records, ignore)
            scores_path = os.path.join(run_dir, f"{name}.json")
            if os.path.exists(scores_path):
                scores_path = os.path.join(run_dir, f"{name}_{datetime.now():%Y%m%d-%H%M%S}.json")
            _write_new(scores_path, scores)
    _write_new(
        os.path.join(run_dir, "complete.json"),
        {"finished_at": datetime.now().isoformat(timespec="seconds"), "n_records": len(records)},
    )
    return run_dir


def main():
    run_experiment(parse_args())


if __name__ == "__main__":
    main()
