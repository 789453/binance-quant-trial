#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${QUANT_PROJECT_ROOT:-/root/ontime_strategy}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_ROOT"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-research.txt
QUANT_PROJECT_ROOT="$PROJECT_ROOT" bash scripts/sync_to_freqtrade.sh

echo "Research environment ready: $PROJECT_ROOT/.venv"
echo "Keep Freqtrade's own environment separate under /root/freqtrade."

