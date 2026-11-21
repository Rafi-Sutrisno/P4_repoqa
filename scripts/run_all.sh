#!/usr/bin/env bash
# All runs on the lab machine, in two rounds:
#   round 1: --limit 30 (5 tasks per language) for every config and both models, to catch
#            errors early
#   round 2: all 600 tasks
# Order: Qwen first, then DeepSeek. Per model, from most to least valuable:
#   full, no_context, rag_fn_bm25, rag_fn_bm25_plus, struct_bm25, index_select, cascade,
#   rag_sw_bm25, random_line   (budgeted strategies at B = 1024, 2048, 4096)
#
#   bash scripts/run_all.sh        # round 1, then round 2
#   bash scripts/run_all.sh 1      # round 1 only
#   bash scripts/run_all.sh 2      # round 2 only
#
# Safe to re-run: --resume auto skips finished runs and continues unfinished ones; every run
# has its own timestamped folder under results/ctx, nothing is overwritten.
# Environment: RESULT_DIR (default results/ctx), OLLAMA_URL (default http://localhost:11434).
# Run scripts/lab_check.py first; afterwards scripts/make_report.sh.
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
ROUNDS="${1:-1 2}"
RESULT_DIR="${RESULT_DIR:-results/ctx}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

run() {
  "$PY" -m repoqa_ctx.run --resume auto --result-dir "$RESULT_DIR" --ollama-url "$OLLAMA_URL" "$@"
}

budgeted() {  # budgeted <model> <strategy> [strategy args JSON] -- <extra args>
  local model=$1 strategy=$2 sargs=$3
  shift 3
  for budget in "${BUDGETS[@]}"; do
    run --strategy "$strategy" --strategy-args "$sargs" --budget "$budget" --model "$model" "$@"
  done
}

all_strategies() {  # all_strategies <model> <extra args...>
  local model=$1
  shift
  run --strategy full --model "$model" "$@"
  run --strategy no_context --model "$model" "$@"
  budgeted "$model" rag_fn_bm25 '{}' "$@"
  budgeted "$model" rag_fn_bm25_plus '{}' "$@"
  budgeted "$model" struct_bm25 '{}' "$@"
  budgeted "$model" index_select '{"index": "sig"}' "$@"
  budgeted "$model" index_select '{"index": "doc"}' "$@"
  budgeted "$model" cascade '{}' "$@"
  budgeted "$model" rag_sw_bm25 '{"window": 256}' "$@"
  budgeted "$model" rag_sw_bm25 '{"window": 512}' "$@"
  budgeted "$model" random_line '{}' "$@"
}

for round in $ROUNDS; do
  for model in "${MODELS[@]}"; do
    echo "=== round $round: $model"
    if [ "$round" = "1" ]; then
      all_strategies "$model" --limit 30
    else
      all_strategies "$model"
    fi
  done
done
