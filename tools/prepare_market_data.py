"""Convert Binance Vision monthly/daily futures klines and produce QA reports.

The script writes two deliberately separate datasets:

* ``data/parquet/<SYMBOL>/<timeframe>.parquet`` keeps all source fields plus a few
  generally useful, backward-looking features for research.
* ``freqtrade/user_data/data/binance/futures/*.parquet`` contains only Freqtrade's
  canonical OHLCV columns and naming convention.

It is safe to run repeatedly: each per-symbol file is rebuilt atomically from the
immutable monthly and daily ZIP files.  Only the ``klines`` tree is read; index,
mark-price and other newly downloaded datasets are deliberately ignored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq


SOURCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "ignore",
]
NUMERIC_COLUMNS = SOURCE_COLUMNS[1:]
INTERVALS = {
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "data/binance_usdm/data/futures/um/monthly/klines",
    )
    parser.add_argument(
        "--daily-source",
        type=Path,
        default=root / "data/binance_usdm/data/futures/um/daily/klines",
    )
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--cutoff", default="2026-09-24")
    parser.add_argument("--output", type=Path, default=root / "data/parquet")
    parser.add_argument(
        "--freqtrade-data",
        type=Path,
        default=root / "freqtrade/user_data/data/binance",
    )
    parser.add_argument("--reports", type=Path, default=root / "reports")
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip"])
    return parser.parse_args()


def read_archive(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"{path}: expected exactly one CSV, found {len(csv_names)}")
        with archive.open(csv_names[0]) as stream:
            frame = pd.read_csv(stream)
    # Older Binance archives sometimes have no header.
    if list(frame.columns) != [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]:
        with zipfile.ZipFile(path) as archive, archive.open(csv_names[0]) as stream:
            frame = pd.read_csv(stream, header=None, names=SOURCE_COLUMNS)
    else:
        frame = frame.rename(columns={"count": "trade_count"})
    if list(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(f"{path}: unexpected columns {list(frame.columns)}")
    return frame


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(path: Path, scope: str) -> dict[str, object]:
    checksum_path = path.with_name(path.name + ".CHECKSUM")
    expected = ""
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").split()[0].lower()
    actual = file_sha256(path)
    return {
        "scope": scope,
        "archive": str(path.resolve()),
        "bytes": path.stat().st_size,
        "checksum_exists": checksum_path.exists(),
        "checksum_match": bool(expected) and expected == actual,
        "sha256": actual,
    }


def load_series(
    paths: list[Path], timeframe: str, start: pd.Timestamp, cutoff_exclusive: pd.Timestamp
) -> tuple[pd.DataFrame, dict[str, object]]:
    parts = [read_archive(path) for path in paths]
    frame = pd.concat(parts, ignore_index=True)
    source_rows = len(frame)
    frame["date"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
    in_range = (frame["date"] >= start) & (frame["date"] < cutoff_exclusive)
    out_of_range_rows = int((~in_range).sum())
    frame = frame.loc[in_range].copy()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    duplicate_rows = int(frame.duplicated("date", keep=False).sum())
    frame = frame.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    delta = INTERVALS[timeframe]
    diffs = frame["date"].diff()
    missing_candles = int(((diffs[diffs > delta] / delta) - 1).sum())
    irregular_steps = int(((diffs.notna()) & (diffs % delta != pd.Timedelta(0))).sum())
    null_cells = int(frame[SOURCE_COLUMNS].isna().sum().sum())
    invalid_ohlc = int(
        (
            (frame["high"] < frame[["open", "close", "low"]].max(axis=1))
            | (frame["low"] > frame[["open", "close", "high"]].min(axis=1))
            | (frame[["open", "high", "low", "close"]] <= 0).any(axis=1)
        ).sum()
    )
    negative_volume = int(
        (frame[["volume", "quote_volume", "taker_buy_volume"]] < 0).any(axis=1).sum()
    )
    zero_volume = int((frame["volume"] == 0).sum())

    # Research columns are causal; no forward returns are stored to avoid accidental leakage.
    frame["taker_sell_volume"] = frame["volume"] - frame["taker_buy_volume"]
    frame["taker_buy_ratio"] = np.where(
        frame["volume"] > 0, frame["taker_buy_volume"] / frame["volume"], np.nan
    )
    frame["log_return"] = np.log(frame["close"]).diff()
    frame["range_pct"] = (frame["high"] - frame["low"]) / frame["open"]
    frame["body_pct"] = (frame["close"] - frame["open"]) / frame["open"]
    frame["vwap"] = np.where(frame["volume"] > 0, frame["quote_volume"] / frame["volume"], np.nan)
    for column in ["taker_buy_ratio", "log_return", "range_pct", "body_pct"]:
        frame[column] = frame[column].astype("float32")
    frame["trade_count"] = frame["trade_count"].round().astype("int64")

    quality = {
        "rows_in_archives": source_rows,
        "out_of_range_rows": out_of_range_rows,
        "rows": len(frame),
        "start_utc": frame["date"].iloc[0].isoformat(),
        "end_utc": frame["date"].iloc[-1].isoformat(),
        "duplicate_rows": duplicate_rows,
        "missing_candles": missing_candles,
        "irregular_steps": irregular_steps,
        "null_cells": null_cells,
        "invalid_ohlc_rows": invalid_ohlc,
        "negative_volume_rows": negative_volume,
        "zero_volume_rows": zero_volume,
    }
    return frame, quality


def atomic_parquet(frame: pd.DataFrame, destination: Path, compression: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(suffix=".parquet", dir=destination.parent)
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        frame.to_parquet(temp_path, compression=compression, index=False)
        temp_path.replace(destination)
    finally:
        temp_path.unlink(missing_ok=True)


def parquet_inventory(args: argparse.Namespace, symbols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        for timeframe in INTERVALS:
            paths = [
                ("research", args.output / symbol / f"{timeframe}.parquet"),
                (
                    "freqtrade",
                    args.freqtrade_data
                    / "futures"
                    / f"{symbol[:-4]}_USDT_USDT-{timeframe}-futures.parquet",
                ),
            ]
            for kind, path in paths:
                parquet = pq.ParquetFile(path)
                dates = parquet.read(columns=["date"])["date"]
                rows.append(
                    {
                        "kind": kind,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "path": str(path.resolve()),
                        "bytes": path.stat().st_size,
                        "rows": parquet.metadata.num_rows,
                        "start_utc": pc.min(dates).as_py().isoformat(),
                        "end_utc": pc.max(dates).as_py().isoformat(),
                        "sha256": file_sha256(path),
                    }
                )
    return pd.DataFrame(rows).sort_values(["kind", "symbol", "timeframe"])


def describe_hourly(symbol: str, frame: pd.DataFrame) -> dict[str, object]:
    returns = frame["log_return"].dropna().astype("float64")
    equity = np.exp(returns.cumsum())
    drawdown = equity / equity.cummax() - 1
    elapsed_years = (frame["date"].iloc[-1] - frame["date"].iloc[0]).total_seconds() / (
        365.25 * 24 * 3600
    )
    total_return = frame["close"].iloc[-1] / frame["close"].iloc[0] - 1
    cagr = (frame["close"].iloc[-1] / frame["close"].iloc[0]) ** (1 / elapsed_years) - 1
    return {
        "symbol": symbol,
        "rows": len(frame),
        "start_utc": frame["date"].iloc[0].isoformat(),
        "end_utc": frame["date"].iloc[-1].isoformat(),
        "total_return_pct": float(total_return * 100),
        "cagr_pct": float(cagr * 100),
        "annualized_return_pct": float(returns.mean() * 24 * 365 * 100),
        "annualized_volatility_pct": float(returns.std() * math.sqrt(24 * 365) * 100),
        "return_skew": float(returns.skew()),
        "return_excess_kurtosis": float(returns.kurt()),
        "max_drawdown_pct": float(drawdown.min() * 100),
        "positive_hours_pct": float((returns > 0).mean() * 100),
        "median_hourly_range_pct": float(frame["range_pct"].median() * 100),
        "median_hourly_quote_volume": float(frame["quote_volume"].median()),
        "median_taker_buy_ratio": float(frame["taker_buy_ratio"].median()),
    }


def compare_pair(
    lower: pd.DataFrame, higher: pd.DataFrame, target: str, prefix: str
) -> dict[str, object]:
    lower_indexed = lower.set_index("date")
    higher_indexed = higher.set_index("date")
    aggregate = lower_indexed.resample(target, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    joined = aggregate.join(
        higher_indexed[["open", "high", "low", "close", "volume"]],
        lsuffix="_lower",
        rsuffix="_higher",
        how="inner",
    ).dropna()
    price_abs = (joined["close_lower"] - joined["close_higher"]).abs()
    price_scale = joined[["close_lower", "close_higher"]].abs().max(axis=1).clip(lower=1e-12)
    volume_scale = (
        joined[["volume_lower", "volume_higher"]]
        .abs()
        .max(axis=1)
        .astype("float64")
        .clip(lower=1e-12)
    )
    volume_relative_error = (
        (joined["volume_lower"] - joined["volume_higher"]).abs() / volume_scale
    )
    return {
        f"{prefix}_overlap_rows": len(joined),
        f"{prefix}_close_mismatch_rows": int((price_abs / price_scale > 1e-10).sum()),
        f"{prefix}_volume_mismatch_rows": int((volume_relative_error > 1e-8).sum()),
        f"{prefix}_max_volume_relative_error": (
            float(volume_relative_error.max()) if len(joined) else np.nan
        ),
        f"{prefix}_max_close_abs_error": float(price_abs.max()) if len(joined) else np.nan,
    }


def compare_timeframes(frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    return {
        **compare_pair(frames["5m"], frames["15m"], "15min", "5m_to_15m"),
        **compare_pair(frames["15m"], frames["1h"], "1h", "15m_to_1h"),
    }


def mismatch_details(symbol: str, frames: dict[str, pd.DataFrame]) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for lower_name, higher_name, target in [("5m", "15m", "15min"), ("15m", "1h", "1h")]:
        lower = frames[lower_name].set_index("date")
        higher = frames[higher_name].set_index("date")
        aggregate = lower.resample(target, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        )
        joined = aggregate.join(
            higher[["open", "high", "low", "close", "volume"]],
            lsuffix="_aggregate",
            rsuffix="_native",
            how="inner",
        ).dropna()
        close_scale = joined[["close_aggregate", "close_native"]].abs().max(axis=1).astype("float64").clip(lower=1e-12)
        volume_scale = joined[["volume_aggregate", "volume_native"]].abs().max(axis=1).astype("float64").clip(lower=1e-12)
        close_error = (joined["close_aggregate"] - joined["close_native"]).abs() / close_scale
        volume_error = (joined["volume_aggregate"] - joined["volume_native"]).abs() / volume_scale
        mismatch = joined[(close_error > 1e-10) | (volume_error > 1e-8)]
        for date, row in mismatch.iterrows():
            details.append(
                {
                    "symbol": symbol,
                    "comparison": f"{lower_name}_to_{higher_name}",
                    "date_utc": date.isoformat(),
                    "close_aggregate": row["close_aggregate"],
                    "close_native": row["close_native"],
                    "close_relative_error": close_error.loc[date],
                    "volume_aggregate": row["volume_aggregate"],
                    "volume_native": row["volume_native"],
                    "volume_relative_error": volume_error.loc[date],
                }
            )
    return details


def write_plots(hourly: dict[str, pd.DataFrame], quality: pd.DataFrame, reports: Path) -> None:
    plot_dir = reports / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    closes = pd.concat(
        {symbol: frame.set_index("date")["close"] for symbol, frame in hourly.items()}, axis=1
    ).sort_index()
    normalized = closes / closes.apply(lambda series: series.dropna().iloc[0])
    ax = normalized.plot(figsize=(14, 7), lw=1)
    ax.set(title="USDT perpetual futures: normalized close (start = 1)", ylabel="Normalized price", xlabel="UTC")
    ax.grid(alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "normalized_prices.png", dpi=150)
    plt.close(ax.figure)

    daily = np.log(closes).diff().resample("1D").sum()
    rolling_vol = daily.rolling(30).std() * np.sqrt(365) * 100
    ax = rolling_vol.plot(figsize=(14, 7), lw=1)
    ax.set(title="30-day rolling annualized volatility", ylabel="Volatility (%)", xlabel="UTC")
    ax.grid(alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "rolling_volatility.png", dpi=150)
    plt.close(ax.figure)

    corr = daily.corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for row in range(len(corr)):
        for col in range(len(corr)):
            ax.text(col, row, f"{corr.iloc[row, col]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046)
    ax.set_title("Daily log-return correlation")
    fig.tight_layout()
    fig.savefig(plot_dir / "return_correlation.png", dpi=150)
    plt.close(fig)

    missing = quality.pivot(index="symbol", columns="timeframe", values="missing_candles").fillna(0)
    ax = missing.plot.bar(figsize=(11, 5))
    ax.set(title="Missing candles inside each series", ylabel="Count", xlabel="Symbol")
    ax.grid(axis="y", alpha=0.25)
    ax.figure.tight_layout()
    ax.figure.savefig(plot_dir / "missing_candles.png", dpi=150)
    plt.close(ax.figure)


def main() -> None:
    args = parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Source directory does not exist: {args.source}")
    if not args.daily_source.is_dir():
        raise SystemExit(f"Daily source directory does not exist: {args.daily_source}")
    start = pd.Timestamp(args.start, tz="UTC")
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    cutoff_exclusive = cutoff + pd.Timedelta(days=1)
    args.reports.mkdir(parents=True, exist_ok=True)
    quality_rows: list[dict[str, object]] = []
    descriptive_rows: list[dict[str, object]] = []
    consistency_rows: list[dict[str, object]] = []
    mismatch_rows: list[dict[str, object]] = []
    hourly_frames: dict[str, pd.DataFrame] = {}
    archive_rows: list[dict[str, object]] = []

    symbol_dirs = sorted(path for path in args.source.iterdir() if path.is_dir())
    for symbol_dir in symbol_dirs:
        symbol = symbol_dir.name.upper()
        frames: dict[str, pd.DataFrame] = {}
        for timeframe in INTERVALS:
            monthly_paths = sorted((symbol_dir / timeframe).glob("*.zip"))
            daily_paths = sorted((args.daily_source / symbol / timeframe).rglob("*.zip"))
            paths = monthly_paths + daily_paths
            if not paths:
                continue
            for path in monthly_paths:
                archive_rows.append({"symbol": symbol, "timeframe": timeframe, **verify_archive(path, "monthly")})
            for path in daily_paths:
                archive_rows.append({"symbol": symbol, "timeframe": timeframe, **verify_archive(path, "daily")})
            frame, quality = load_series(paths, timeframe, start, cutoff_exclusive)
            quality_rows.append({"symbol": symbol, "timeframe": timeframe, **quality})
            frames[timeframe] = frame
            atomic_parquet(frame, args.output / symbol / f"{timeframe}.parquet", args.compression)

            freqtrade_frame = frame[["date", "open", "high", "low", "close", "volume"]].copy()
            freqtrade_name = f"{symbol[:-4]}_USDT_USDT-{timeframe}-futures.parquet"
            atomic_parquet(
                freqtrade_frame,
                args.freqtrade_data / "futures" / freqtrade_name,
                args.compression,
            )
        if "1h" in frames:
            hourly_frames[symbol] = frames["1h"]
            descriptive_rows.append(describe_hourly(symbol, frames["1h"]))
        if set(frames) == set(INTERVALS):
            consistency_rows.append({"symbol": symbol, **compare_timeframes(frames)})
            mismatch_rows.extend(mismatch_details(symbol, frames))
        print(f"Prepared {symbol}: {', '.join(frames)}")

    quality_df = pd.DataFrame(quality_rows).sort_values(["symbol", "timeframe"])
    descriptive_df = pd.DataFrame(descriptive_rows).sort_values("symbol")
    consistency_df = pd.DataFrame(consistency_rows).sort_values("symbol")
    archive_df = pd.DataFrame(archive_rows).sort_values(["symbol", "timeframe", "scope", "archive"])
    mismatch_df = pd.DataFrame(mismatch_rows).sort_values(["date_utc", "symbol", "comparison"])
    quality_df.to_csv(args.reports / "data_quality.csv", index=False)
    descriptive_df.to_csv(args.reports / "descriptive_stats.csv", index=False)
    consistency_df.to_csv(args.reports / "timeframe_consistency.csv", index=False)
    archive_df.to_csv(args.reports / "source_archive_integrity.csv", index=False)
    mismatch_df.to_csv(args.reports / "timeframe_mismatch_details.csv", index=False)
    write_plots(hourly_frames, quality_df, args.reports)
    parquet_df = parquet_inventory(args, sorted(hourly_frames))
    parquet_df.to_csv(args.reports / "parquet_inventory.csv", index=False)

    manifest = {
        "source": str(args.source.resolve()),
        "daily_source": str(args.daily_source.resolve()),
        "start_utc": start.isoformat(),
        "cutoff_inclusive_utc": args.cutoff,
        "analysis_output": str(args.output.resolve()),
        "freqtrade_output": str(args.freqtrade_data.resolve()),
        "symbols": sorted(hourly_frames),
        "timeframes": list(INTERVALS),
        "compression": args.compression,
        "quality_totals": {
            key: int(quality_df[key].sum())
            for key in [
                "rows", "duplicate_rows", "missing_candles", "irregular_steps",
                "null_cells", "invalid_ohlc_rows", "negative_volume_rows",
                "zero_volume_rows",
            ]
        },
        "source_archives": {
            "count": len(archive_df),
            "missing_checksums": int((~archive_df["checksum_exists"]).sum()),
            "checksum_failures": int((~archive_df["checksum_match"]).sum()),
        },
        "cross_timeframe_mismatch_rows": len(mismatch_df),
        "parquet_files": len(parquet_df),
    }
    (args.reports / "data_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {len(quality_df)} datasets and reports to {args.reports}")


if __name__ == "__main__":
    main()
