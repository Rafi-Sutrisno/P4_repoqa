"""Summarize run folders into Markdown tables (threshold 0.8) and a CSV (all thresholds).

    python -m repoqa_ctx.summarize results/ctx --out reports/x/summary.md [--split test]
                                   [--by-language]

Definitions (per run, and per language with --by-language):
  params       strategy parameters that differ from the defaults
  ctx tok      mean code-context tokens sent in the answer prompt (CodeLlama tokenizer)
  ratio        sum of original window tokens / sum of context tokens
  input tok    mean input tokens per task over all LLM calls
  calls        mean LLM calls per task
  latency      mean seconds per task = context building + all LLM calls
  retention    share of tasks whose whole needle is still in the context
Dry-run rows count the select calls that were made (fake LLM) plus the answer call that
would be made.
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
from repoqa_ctx.strategies import NAMES, get_strategy
from repoqa_ctx.tasks import load_dataset

COLUMNS = [
    "strategy",
    "params",
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


def _params_label(strategy: str, params: Dict) -> str:
    try:
        defaults = get_strategy(strategy).DEFAULT_PARAMS
    except ValueError:
        defaults = {}
    diff = {k: v for k, v in params.items() if defaults.get(k) != v}
    return " ".join(f"{k}={v}" for k, v in sorted(diff.items()))


def _scores(run: Dict, records: List[Dict], datasets: Dict) -> Dict:
    ident = run["cfg"]["identity"]
    path = ident["dataset_path"]
    if path not in datasets:
        datasets[path] = load_dataset(path)
    with redirect_stdout(io.StringIO()):  # compute_score prints a rich table
        return compute_score(ident["model"], datasets[path], records, False)[ident["model"]][
            "scores"
        ]


def summarize_run(run: Dict, datasets: Dict, split: str, by_language: bool) -> List[Dict]:
    ident = run["cfg"]["identity"]
    dry = ident["dry_run"]
    records = [r for r in run["records"] if split == "all" or r["split"] == split]
    mode = "dry-run" + (f"/{ident.get('fake_mode')}" if ident.get("fake_mode") else "")
    base = {
        "strategy": ident["strategy"],
        "params": _params_label(ident["strategy"], ident["params"]),
        "budget": ident["budget"],
        "model": ident["model"] or "-",
        "mode": (mode if dry else "run") + ("" if run["complete"] else " (partial)"),
        "dir": os.path.basename(run["dir"]),
    }
    scores = _scores(run, records, datasets) if records and not dry else {}
    groups = [("all", records)]
    if by_language:
        langs = sorted({r["language"] for r in records})
        groups += [(lang, [r for r in records if r["language"] == lang]) for lang in langs]
    rows = []
    for lang, group in groups:
        row = {**base, "lang": lang, "n": len(group)}
        rows.append(row)
        if not group:
            continue
        ctx = [r["ctx"] for r in group]
        comp = sum(c["ctx_tokens"] for c in ctx)
        extra = 1 if dry else 0  # the answer call a dry-run does not make
        row["ctx_tokens"] = mean(c["ctx_tokens"] for c in ctx)
        row["ratio"] = sum(c["orig_ctx_tokens"] for c in ctx) / comp if comp else None
        row["retention"] = mean(c["needle_retained"] for c in ctx)
        row["input_tokens"] = mean(
            sum(call["prompt_tokens"] for call in r["calls"])
            + (r["ctx"]["prompt_tokens"] if dry else 0)
            for r in group
        )
        row["calls"] = mean(len(r["calls"]) + extra for r in group)
        row["latency_s"] = mean(
            r["ctx"]["build_s"] + sum(call["latency_s"] for call in r["calls"]) for r in group
        )
        if scores:
            by_threshold = scores[lang]
            for t in THRESHOLDS:
                row[f"pass@1@{t}"] = by_threshold[t]["pass@1"]
            row["pass@1"] = by_threshold[0.8]["pass@1"]
    return rows


def _fmt(key: str, value) -> str:
    if value is None or value == "":
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


def to_markdown(rows: List[Dict], columns: List[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(k, row.get(k)) for k in columns) + " |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=os.path.join("results", "ctx"))
    p.add_argument("--out", help="write Markdown here and a CSV next to it")
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p.add_argument("--by-language", action="store_true", help="add per-language rows")
    args = p.parse_args()

    datasets = {}
    rows = []
    for run in load_runs(args.root):
        rows += summarize_run(run, datasets, args.split, args.by_language)
    order = {name: i for i, name in enumerate(NAMES)}
    rows.sort(
        key=lambda r: (r["model"], order.get(r["strategy"], 99), r["params"], r["budget"] or 0)
    )
    text = f"Split: {args.split}\n\n" + to_markdown([r for r in rows if r["lang"] == "all"], COLUMNS)
    if args.by_language:
        lang_rows = sorted(
            (r for r in rows if r["lang"] != "all"),
            key=lambda r: (r["lang"], r["model"], order.get(r["strategy"], 99), r["params"], r["budget"] or 0),
        )
        text += "\nPer language:\n\n" + to_markdown(lang_rows, ["lang"] + COLUMNS)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        csv_cols = ["lang"] + COLUMNS + [f"pass@1@{t}" for t in THRESHOLDS] + ["dir"]
        with open(os.path.splitext(args.out)[0] + ".csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
