"""Export research parquet files to Freqtrade's futures OHLCV layout."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd


def atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(suffix=".parquet", dir=destination.parent)
    os.close(handle)
    temp = Path(name)
    try:
        frame.to_parquet(temp, index=False, compression="zstd")
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-data", type=Path, default=root / "data/parquet")
    parser.add_argument("--freqtrade-user-data", type=Path, required=True)
    parser.add_argument("--timeframes", nargs="+", default=["1h"], choices=["5m", "15m", "1h"])
    parser.add_argument("--symbols", nargs="*")
    args = parser.parse_args()
    symbols = args.symbols or sorted(path.name for path in args.research_data.iterdir() if path.is_dir())
    output = args.freqtrade_user_data / "data/binance/futures"
    rows = []
    for symbol in symbols:
        if not symbol.endswith("USDT"):
            raise ValueError(f"Unsupported symbol: {symbol}")
        for timeframe in args.timeframes:
            source = args.research_data / symbol / f"{timeframe}.parquet"
            frame = pd.read_parquet(source, columns=["date", "open", "high", "low", "close", "volume"])
            frame = frame.sort_values("date").drop_duplicates("date", keep="last")
            if frame.date.dt.tz is None or frame.isna().any().any():
                raise ValueError(f"Invalid data in {source}")
            base = symbol[:-4]
            destination = output / f"{base}_USDT_USDT-{timeframe}-futures.parquet"
            atomic_parquet(frame, destination)
            rows.append({"symbol": symbol, "timeframe": timeframe, "rows": len(frame),
                         "start": frame.date.min(), "end": frame.date.max(), "path": destination})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

