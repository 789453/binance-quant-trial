#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${QUANT_PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
FREQTRADE_ROOT="${FREQTRADE_ROOT:-/root/freqtrade}"
USER_DATA="$FREQTRADE_ROOT/user_data"

if [[ ! -d "$USER_DATA" ]]; then
  echo "Freqtrade user_data not found: $USER_DATA" >&2
  exit 1
fi

install -d "$USER_DATA/strategies" "$USER_DATA/data/factor_combo"
for strategy in "$PROJECT_ROOT"/deployment/freqtrade/user_data/strategies/*.py; do
  install -m 0644 "$strategy" "$USER_DATA/strategies/$(basename "$strategy")"
done

example="$PROJECT_ROOT/deployment/freqtrade/user_data/config.factor-combo.example.json"
install -m 0644 "$example" "$USER_DATA/config.factor-combo.example.json"

echo "Strategies and example config synchronized to $USER_DATA"
echo "Existing private configs and databases were not modified."
