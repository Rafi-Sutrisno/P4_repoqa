"""Summarize run folders into a Markdown table (threshold 0.8) and a CSV (all thresholds).

    python -m repoqa_ctx.summarize results/ctx --out reports/baselines.md [--split test]

Definitions (per run):
  ctx tok      mean code-context tokens sent (CodeLlama tokenizer)
  ratio        sum of original window tokens / sum of context tokens
  input tok    mean input tokens per task over all LLM calls (dry-run: answer prompt)
  calls        mean LLM calls per task
  latency      mean seconds per task = context building + all LLM calls
  retention    share of tasks whose whole needle is still in the context
"""

import argparse
import csv
import io
import json
import os
from contextlib import redirect_stdout
from statistics import mean
from typing import Dict, List

from repoqa.compute_score import THRESHOLDS, compute_score
from repoqa_ctx.run import read_jsonl
from repoqa_ctx.strategies import NAMES
from repoqa_ctx.tasks import load_dataset

COLUMNS = [
    "strategy",
    "budget",
    "model",
    "mode",
    "n",
    "pass@1",
    "ctx_tokens",
    "ratio",
    "input_tokens",
    "calls",
    "latency_s",
    "retention",
]


def load_runs(root: str) -> List[Dict]:
    runs = []
    for name in sorted(os.listdir(root)):
        run_dir = os.path.join(root, name)
        cfg_path = os.path.join(run_dir, "run_config.json")
        if not os.path.exists(cfg_path):
            continue
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        dry = cfg["identity"]["dry_run"]
        records = read_jsonl(os.path.join(run_dir, "dryrun.jsonl" if dry else "outputs.jsonl"))
        complete = os.path.exists(os.path.join(run_dir, "complete.json"))
        runs.append({"dir": run_dir, "cfg": cfg, "records": records, "complete": complete})
    return runs


def summarize_run(run: Dict, datasets: Dict, split: str) -> Dict:
    ident = run["cfg"]["identity"]
    records = [r for r in run["records"] if split == "all" or r["split"] == split]
    ctx = [r["ctx"] for r in records]
    dry = ident["dry_run"]
    row = {
        "strategy": ident["strategy"],
        "budget": ident["budget"],
        "model": ident["model"] or "-",
        "mode": ("dry-run" if dry else "run") + ("" if run["complete"] else " (partial)"),
        "n": len(records),
        "dir": os.path.basename(run["dir"]),
    }
    if not records:
        return row
    comp = sum(c["ctx_tokens"] for c in ctx)
    row["ctx_tokens"] = mean(c["ctx_tokens"] for c in ctx)
    row["ratio"] = sum(c["orig_ctx_tokens"] for c in ctx) / comp if comp else None
    row["retention"] = mean(c["needle_retained"] for c in ctx)
    if dry:
        row["input_tokens"] = mean(c["prompt_tokens"] for c in ctx)
        row["calls"] = 0
        row["latency_s"] = mean(c["build_s"] for c in ctx)
        return row
    row["input_tokens"] = mean(sum(call["prompt_tokens"] for call in r["calls"]) for r in records)
    row["calls"] = mean(len(r["calls"]) for r in records)
    row["latency_s"] = mean(
        r["ctx"]["build_s"] + sum(call["latency_s"] for call in r["calls"]) for r in records
    )
    path = ident["dataset_path"]
    if path not in datasets:
        datasets[path] = load_dataset(path)
    with redirect_stdout(io.StringIO()):  # compute_score prints a rich table
        scores = compute_score(ident["model"], datasets[path], records, False)
    by_threshold = scores[ident["model"]]["scores"]["all"]
    for t in THRESHOLDS:
        row[f"pass@1@{t}"] = by_threshold[t]["pass@1"]
    row["pass@1"] = by_threshold[0.8]["pass@1"]
    return row


def _fmt(key: str, value) -> str:
    if value is None:
        return "-"
    if key in ("pass@1", "retention"):
        return f"{100 * value:.1f}%"
    if key == "ratio":
        return f"{value:.1f}x"
    if key in ("ctx_tokens", "input_tokens"):
        return f"{value:.0f}"
    if key in ("calls", "latency_s"):
        return f"{value:.2f}"
    return str(value)


def to_markdown(rows: List[Dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(k, row.get(k)) for k in COLUMNS) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=os.path.join("results", "ctx"))
    p.add_argument("--out", help="write Markdown here and a CSV next to it")
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    args = p.parse_args()

    datasets = {}
    rows = [summarize_run(run, datasets, args.split) for run in load_runs(args.root)]
    order = {name: i for i, name in enumerate(NAMES)}
    rows.sort(key=lambda r: (r["model"], order.get(r["strategy"], 99), r["budget"] or 0))
    table = to_markdown(rows)
    print(table)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table)
        csv_cols = COLUMNS + [f"pass@1@{t}" for t in THRESHOLDS] + ["dir"]
        with open(os.path.splitext(args.out)[0] + ".csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
