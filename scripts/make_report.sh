#!/usr/bin/env bash
# Summarize every run under results/ctx into reports/<date>_lab/ (to commit and push back):
#   summary.md / summary.csv          test split (main report, see CLAUDE.md)
#   summary_all.md / summary_all.csv  all 600 tasks (comparison with the papers)
# Markdown shows pass@1 at threshold 0.8; the CSVs keep every threshold 0.0-1.0.
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

RESULT_DIR="${RESULT_DIR:-results/ctx}"
OUT="${REPORT_DIR:-reports/$(date +%Y-%m-%d)_lab}"
mkdir -p "$OUT"
"$PY" -m repoqa_ctx.summarize "$RESULT_DIR" --split test --by-language --out "$OUT/summary.md"
"$PY" -m repoqa_ctx.summarize "$RESULT_DIR" --split all --by-language --out "$OUT/summary_all.md"
echo "Wrote $OUT/summary{,_all}.{md,csv}"
