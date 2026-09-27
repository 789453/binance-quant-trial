#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${QUANT_PROJECT_ROOT:-/root/ontime_strategy}"
cd "$PROJECT_ROOT"
git pull --ff-only
bash scripts/sync_to_freqtrade.sh
"$PROJECT_ROOT/.venv/bin/python" -m pytest

