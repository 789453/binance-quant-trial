"""Export causal six-factor target history and the latest live target snapshot."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.factor_combo_pipeline import SYMBOLS, add_oriented_signals, build_factor_panel
from research.settings import load_settings
from research.walk_forward import combine_sleeves, factor_sleeve_positions


def atomic_text(text: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(suffix=".tmp", dir=destination.parent)
    os.close(handle)
    temp = Path(name)
    try:
        temp.write_text(text, encoding="utf-8")
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Usually /root/freqtrade/user_data/data/factor_combo")
    parser.add_argument("--force-factors", action="store_true")
    args = parser.parse_args()
    settings = load_settings(args.config)
    panel = build_factor_panel(force=args.force_factors)
    columns = [f"f_{factor_hash[:12]}" for factor_hash in settings["selected_hashes"]]
    oriented = add_oriented_signals(panel, columns)
    sleeves = factor_sleeve_positions(oriented, columns, len(SYMBOLS))
    targets = combine_sleeves(sleeves).rename(columns={"position": "target"})
    targets["signal"] = oriented[columns].clip(-2, 2).mean(axis=1)
    history = targets[["date", "symbol", "signal", "target"]].sort_values(["date", "symbol"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.to_parquet(args.output_dir / "target_history.parquet", index=False, compression="zstd")
    latest_time = history.date.max()
    latest = history.loc[history.date == latest_time]
    payload = {
        "schema_version": 1,
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "signal_bar": latest_time.isoformat(),
        "targets": {row.symbol: {"signal": row.signal, "target": row.target}
                    for row in latest.itertuples()},
    }
    atomic_text(json.dumps(payload, indent=2, ensure_ascii=False), args.output_dir / "latest_targets.json")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
