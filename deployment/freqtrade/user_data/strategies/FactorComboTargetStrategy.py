"""Dry-run bridge consuming targets exported by the research pipeline.

This adapter deliberately fails closed.  It does not calculate the factor library
from Freqtrade's six-column OHLCV because quote volume, trade count and VWAP are
required.  Use tools/export_factor_targets.py to produce target_history.parquet
for backtests and latest_targets.json for dry/live shadow operation.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class FactorComboTargetStrategy(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1h"
    startup_candle_count = 5
    process_only_new_candles = True
    minimal_roi = {"0": 100.0}
    stoploss = -0.08
    use_exit_signal = True
    trailing_stop = False
    position_adjustment_enable = False

    entry_threshold = 0.004
    neutral_threshold = 0.0015
    max_target_age_hours = 3

    order_types = {
        "entry": "market", "exit": "market", "emergency_exit": "market",
        "force_entry": "market", "force_exit": "market", "stoploss": "market",
        "stoploss_on_exchange": True, "stoploss_on_exchange_interval": 60,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    def bot_start(self, **kwargs) -> None:
        user_data = Path(self.config["user_data_dir"])
        self._target_dir = user_data / "data/factor_combo"
        override = os.getenv("FACTOR_TARGET_FILE")
        self._latest_file = Path(override) if override else self._target_dir / "latest_targets.json"

    @staticmethod
    def _symbol(pair: str) -> str:
        return pair.split(":")[0].replace("/", "")

    def _history(self, pair: str) -> DataFrame:
        path = self._target_dir / "target_history.parquet"
        if not path.exists():
            return DataFrame(columns=["date", "factor_signal", "factor_target"])
        symbol = self._symbol(pair)
        frame = pd.read_parquet(path, filters=[[('symbol', '==', symbol)]])
        return frame.rename(columns={"signal": "factor_signal", "target": "factor_target"})[
            ["date", "factor_signal", "factor_target"]]

    def _latest(self, pair: str) -> tuple[datetime | None, float, float]:
        try:
            payload = json.loads(self._latest_file.read_text(encoding="utf-8"))
            record = payload["targets"][self._symbol(pair)]
            bar = datetime.fromisoformat(payload["signal_bar"].replace("Z", "+00:00"))
            return bar, float(record["signal"]), float(record["target"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None, 0.0, 0.0

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        history = self._history(metadata["pair"])
        if not history.empty:
            dataframe = dataframe.merge(history, on="date", how="left")
        else:
            bar, signal, target = self._latest(metadata["pair"])
            dataframe["factor_signal"] = 0.0
            dataframe["factor_target"] = 0.0
            if bar is not None:
                dataframe.loc[dataframe.date == pd.Timestamp(bar), ["factor_signal", "factor_target"]] = (signal, target)
        dataframe[["factor_signal", "factor_target"]] = dataframe[
            ["factor_signal", "factor_target"]].fillna(0.0)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0; dataframe["enter_short"] = 0
        dataframe.loc[dataframe.factor_target >= self.entry_threshold,
                      ["enter_long", "enter_tag"]] = (1, "factor_target_long")
        dataframe.loc[dataframe.factor_target <= -self.entry_threshold,
                      ["enter_short", "enter_tag"]] = (1, "factor_target_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = (dataframe.factor_target <= self.neutral_threshold).astype(int)
        dataframe["exit_short"] = (dataframe.factor_target >= -self.neutral_threshold).astype(int)
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str | None,
                            side: str, **kwargs) -> bool:
        bar, _, target = self._latest(pair)
        if self.config["runmode"].value == "backtest":
            return True
        if bar is None:
            return False
        now = current_time.astimezone(timezone.utc)
        age_hours = (now - bar.astimezone(timezone.utc)).total_seconds() / 3600
        correct_side = target > 0 if side == "long" else target < 0
        return 0 <= age_hours <= self.max_target_age_hours and correct_side

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str, **kwargs) -> float:
        target = 0.0
        try:
            analyzed, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not analyzed.empty:
                target = float(analyzed.iloc[-1]["factor_target"])
        except (KeyError, TypeError, ValueError):
            _, _, target = self._latest(pair)
        scale = min(max(abs(target) * len(self.dp.current_whitelist()), 0.25), 1.0)
        stake = min(proposed_stake * scale, max_stake)
        return max(stake, min_stake) if min_stake is not None else stake

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs) -> float:
        return min(1.0, max_leverage)

