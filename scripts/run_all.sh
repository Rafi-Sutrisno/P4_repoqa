#!/usr/bin/env bash
# All lab runs, by tier:
#   A  600 tasks, B=2048: full, no_context, rag_fn_bm25, rag_fn_bm25_plus, rag_fn_unixcoder,
#      struct_bm25 (dev-selected weights), index_select, cascade, longcodezip
#   B  120 tasks (20 per language), B=2048: every other strategy/variant
#   C  120 tasks: B=1024 and B=4096 for the budgeted tier-A strategies
#   D  DeepSeek: tier A only
# Two rounds: round 1 = every config with --limit 30 (5 per language) to catch errors;
# round 2 = the sizes above. Qwen first, then DeepSeek.
#
#   bash scripts/run_all.sh        # round 1, then round 2
#   bash scripts/run_all.sh 1      # round 1 only
#   bash scripts/run_all.sh 2      # round 2 only
#
# Compressed strategies first run the compress stage in .venv-compress (cached per task, so
# 30 -> 120 -> 600 reuses work). Safe to re-run: --resume auto skips finished runs and
# continues unfinished ones; every run has its own folder, nothing is overwritten.
# Environment: RESULT_DIR (results/ctx), OLLAMA_URL (http://localhost:11434),
#              COMPRESS_DEVICE (auto). Run scripts/lab_check.py and
#              scripts/setup_compress.sh first; afterwards scripts/make_report.sh.
set -euo pipefail
cd "$(dirname "$0")/.."

pick_python() {  # pick_python <venv dir>
  if [ -x "$1/bin/python" ]; then echo "$1/bin/python"
  elif [ -x "$1/Scripts/python.exe" ]; then echo "$1/Scripts/python.exe"
  else echo python; fi
}
PY="${PYTHON:-$(pick_python .venv)}"
CPY="${COMPRESS_PYTHON:-$(pick_python .venv-compress)}"
export PYTHONIOENCODING=utf-8

QWEN="qwen2.5-coder:7b-instruct"
DEEPSEEK="deepseek-coder:6.7b-instruct"
ROUNDS="${1:-1 2}"
RESULT_DIR="${RESULT_DIR:-results/ctx}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
COMPRESS_DEVICE="${COMPRESS_DEVICE:-auto}"
STRUCT_W='{"weights": [3, 1, 1, 3]}'  # reports/struct_weights_dev.md
COMPRESSED=" rag_fn_unixcoder rag_sw_unixcoder longcodezip llmlingua llmlingua2 longllmlingua "

cfg() {  # cfg <model> <strategy> <strategy args JSON> <budget or -> <limit or ->
  local model=$1 strategy=$2 sargs=$3 budget=$4 limit=$5
  local extra=()
  [ "$budget" != "-" ] && extra+=(--budget "$budget")
  [ "$limit" != "-" ] && extra+=(--limit "$limit")
  if [[ "$COMPRESSED" == *" $strategy "* ]]; then
    "$CPY" -m repoqa_ctx.compress --strategy "$strategy" --strategy-args "$sargs" \
      --device "$COMPRESS_DEVICE" ${extra[@]+"${extra[@]}"}
  fi
  "$PY" -m repoqa_ctx.run --resume auto --result-dir "$RESULT_DIR" --ollama-url "$OLLAMA_URL" \
    --model "$model" --strategy "$strategy" --strategy-args "$sargs" ${extra[@]+"${extra[@]}"}
}

tier_a_budgeted() {  # tier_a_budgeted <model> <budget> <limit>
  local m=$1 b=$2 l=$3
  cfg "$m" rag_fn_bm25 '{}' "$b" "$l"
  cfg "$m" rag_fn_bm25_plus '{}' "$b" "$l"
  cfg "$m" rag_fn_unixcoder '{}' "$b" "$l"
  cfg "$m" struct_bm25 "$STRUCT_W" "$b" "$l"
  cfg "$m" index_select '{}' "$b" "$l"
  cfg "$m" cascade '{}' "$b" "$l"
  cfg "$m" longcodezip '{}' "$b" "$l"
}

tier_a() {  # tier_a <model> <limit>
  cfg "$1" full '{}' - "$2"
  cfg "$1" no_context '{}' - "$2"
  tier_a_budgeted "$1" 2048 "$2"
}

tier_b() {  # tier_b <model> <limit>
  local m=$1 l=$2
  cfg "$m" rag_sw_bm25 '{"window": 256}' 2048 "$l"
  cfg "$m" rag_sw_bm25 '{"window": 512}' 2048 "$l"
  cfg "$m" rag_sw_unixcoder '{}' 2048 "$l"
  cfg "$m" struct_bm25 '{}' 2048 "$l"
  cfg "$m" struct_bm25 '{"weights": [3, 1, 1, 3], "skeleton": true}' 2048 "$l"
  cfg "$m" index_select '{"index": "doc"}' 2048 "$l"
  cfg "$m" longllmlingua '{}' 2048 "$l"
  cfg "$m" llmlingua '{}' 2048 "$l"
  cfg "$m" llmlingua2 '{}' 2048 "$l"
  cfg "$m" random_line '{}' 2048 "$l"
  cfg "$m" random_token '{}' 2048 "$l"
}

tier_c() {  # tier_c <model> <limit>
  tier_a_budgeted "$1" 1024 "$2"
  tier_a_budgeted "$1" 4096 "$2"
}

for round in $ROUNDS; do
  if [ "$round" = "1" ]; then
    echo "=== round 1 (--limit 30): Qwen A, B, C; DeepSeek A"
    tier_a "$QWEN" 30
    tier_b "$QWEN" 30
    tier_c "$QWEN" 30
    tier_a "$DEEPSEEK" 30
  else
    echo "=== round 2: Qwen A (600), B (120), C (120); DeepSeek A (600)"
    tier_a "$QWEN" -
    tier_b "$QWEN" 120
    tier_c "$QWEN" 120
    tier_a "$DEEPSEEK" -
  fi
done
