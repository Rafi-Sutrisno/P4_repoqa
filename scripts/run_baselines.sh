#!/usr/bin/env bash
# Phase 2 baselines on both models: full, no_context, random_line and rag_fn_bm25 (B 1K/2K/4K).
#
#   bash scripts/run_baselines.sh                 # all 600 tasks
#   bash scripts/run_baselines.sh --limit 30      # extra args go to every run
#
# Safe to re-run: --resume auto skips finished runs and continues unfinished ones.
# Run scripts/lab_check.py first.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
elif [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif [ -x .venv/Scripts/python.exe ]; then
  PY=.venv/Scripts/python.exe
else
  PY=python
fi
export PYTHONIOENCODING=utf-8

MODELS=("qwen2.5-coder:7b-instruct" "deepseek-coder:6.7b-instruct")
BUDGETS=(1024 2048 4096)

for model in "${MODELS[@]}"; do
  for strategy in full no_context; do
    "$PY" -m repoqa_ctx.run --strategy "$strategy" --model "$model" --resume auto "$@"
  done
  for budget in "${BUDGETS[@]}"; do
    for strategy in random_line rag_fn_bm25; do
      "$PY" -m repoqa_ctx.run --strategy "$strategy" --budget "$budget" --model "$model" \
        --resume auto "$@"
    done
  done
done

mkdir -p reports
"$PY" -m repoqa_ctx.summarize results/ctx --out "reports/baselines_$(date +%Y%m%d-%H%M%S).md"
