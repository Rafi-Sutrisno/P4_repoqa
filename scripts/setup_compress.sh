#!/usr/bin/env bash
# Set up the compress stage: .venv-compress (separate from .venv) and the official LongCodeZip
# code in third_party/ (read-only, git-ignored), pinned to the commit we read.
#
#   bash scripts/setup_compress.sh            # CUDA torch (lab)
#   TORCH_INDEX=cpu bash scripts/setup_compress.sh   # CPU torch (laptop)
set -euo pipefail
cd "$(dirname "$0")/.."

LCZ_COMMIT=617a30b
TORCH_INDEX="${TORCH_INDEX:-cu124}"  # see https://pytorch.org for the right CUDA build
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
if ! command -v "$PYTHON_BIN" > /dev/null 2>&1; then PYTHON_BIN="py -3.11"; fi

if [ ! -d .venv-compress ]; then
  $PYTHON_BIN -m venv .venv-compress
fi
if [ -x .venv-compress/bin/python ]; then PY=.venv-compress/bin/python; else PY=.venv-compress/Scripts/python.exe; fi
"$PY" -m pip install --upgrade pip
"$PY" -m pip install torch --index-url "https://download.pytorch.org/whl/$TORCH_INDEX"
"$PY" -m pip install -r requirements-compress.txt

mkdir -p third_party
if [ ! -d third_party/LongCodeZip ]; then
  git clone https://github.com/YerbaPage/LongCodeZip.git third_party/LongCodeZip
fi
git -C third_party/LongCodeZip fetch --depth 1 origin "$LCZ_COMMIT" 2> /dev/null || true
git -C third_party/LongCodeZip checkout -q "$LCZ_COMMIT"
echo "LongCodeZip at $(git -C third_party/LongCodeZip rev-parse --short HEAD); compress venv: $PY"
