"""Incrementally update research OHLCV parquet files from Binance public API.

Only closed candles are persisted.  Existing rows are preserved and the parquet
is atomically replaced after validation.  No API key is read or required.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "parquet"
SYMBOLS = ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
           "ETHUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "TRXUSDT", "XRPUSDT"]
INTERVAL_MS = {"5m": 300_000, "15m": 900_000, "1h": 3_600_000}
API = "https://fapi.binance.com/fapi/v1/klines"


def request_klines(symbol: str, interval: str, start: int, end: int, proxy: str | None) -> list:
    proxies = {"http": proxy, "https": proxy} if proxy else None
    response = requests.get(API, params={"symbol": symbol, "interval": interval, "startTime": start,
                            "endTime": end, "limit": 1500}, proxies=proxies,
                            headers={"User-Agent": "factor-combo-research/1.0"}, timeout=30)
    response.raise_for_status()
    return response.json()


def convert(rows: list) -> pd.DataFrame:
    names = ["open_time", "open", "high", "low", "close", "volume", "close_time",
             "quote_volume", "trade_count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
    x = pd.DataFrame(rows, columns=names)
    if x.empty:
        return x
    ints = ["open_time", "close_time", "trade_count"]
    floats = [c for c in names if c not in ints and c != "ignore"]
    x[ints] = x[ints].astype("int64")
    x[floats] = x[floats].astype("float64")
    x["ignore"] = pd.to_numeric(x.ignore, errors="coerce").fillna(0).astype("int64")
    x["date"] = pd.to_datetime(x.open_time, unit="ms", utc=True)
    x["taker_sell_volume"] = x.volume - x.taker_buy_volume
    x["taker_buy_ratio"] = np.divide(x.taker_buy_volume, x.volume,
                                      out=np.full(len(x), np.nan), where=x.volume.ne(0)).astype("float32")
    x["log_return"] = np.log(x.close).diff().astype("float32")
    x["range_pct"] = ((x.high - x.low) / x.open).astype("float32")
    x["body_pct"] = ((x.close - x.open) / x.open).astype("float32")
    x["vwap"] = np.divide(x.quote_volume, x.volume,
                           out=x.close.to_numpy(copy=True), where=x.volume.ne(0))
    return x


def validate(x: pd.DataFrame, interval: str) -> None:
    if x.date.duplicated().any() or not x.date.is_monotonic_increasing:
        raise ValueError("duplicate or unordered timestamps")
    if ((x.high < x[["open", "close"]].max(axis=1)) | (x.low > x[["open", "close"]].min(axis=1))).any():
        raise ValueError("invalid OHLC relationship")
    steps = x.open_time.diff().dropna()
    if not steps.eq(INTERVAL_MS[interval]).all():
        bad = int((~steps.eq(INTERVAL_MS[interval])).sum())
        raise ValueError(f"{bad} timestamp gaps found; refusing partial update")


def update_one(symbol: str, interval: str, now_ms: int, proxy: str | None = None) -> dict:
    path = DATA / symbol / f"{interval}.parquet"
    old = pd.read_parquet(path).sort_values("date")
    start = int(old.open_time.max()) + INTERVAL_MS[interval]
    closed_end = (now_ms // INTERVAL_MS[interval]) * INTERVAL_MS[interval] - 1
    batches = []
    cursor = start
    while cursor <= closed_end:
        rows = request_klines(symbol, interval, cursor, closed_end, proxy)
        if not rows:
            break
        batches.extend(rows)
        cursor = int(rows[-1][0]) + INTERVAL_MS[interval]
        time.sleep(0.08)
    if not batches:
        return {"symbol": symbol, "timeframe": interval, "added": 0, "max_date": str(old.date.max())}
    new = convert(batches)
    merged = pd.concat([old, new], ignore_index=True).drop_duplicates("date", keep="last").sort_values("date")
    # Recompute the boundary return that convert() cannot see.
    merged["log_return"] = np.log(merged.close).diff().astype("float32")
    validate(merged, interval)
    temp = path.with_suffix(".parquet.tmp")
    merged.to_parquet(temp, index=False, compression="zstd")
    os.replace(temp, path)
    return {"symbol": symbol, "timeframe": interval, "added": len(merged) - len(old), "max_date": str(merged.date.max())}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", nargs="*", default=SYMBOLS)
    p.add_argument("--timeframes", nargs="*", choices=list(INTERVAL_MS), default=list(INTERVAL_MS))
    p.add_argument("--proxy", help="HTTP or SOCKS proxy, e.g. socks5h://127.0.0.1:7897")
    args = p.parse_args()
    now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    results = [update_one(s, tf, now_ms, args.proxy) for s in args.symbols for tf in args.timeframes]
    report = ROOT / "reports" / "research_data_update.csv"
    pd.DataFrame(results).to_csv(report, index=False)
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
