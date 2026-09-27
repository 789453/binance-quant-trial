"""Transparent long/short baseline for 1h USDT perpetual futures.

This is intentionally a research baseline, not a claim of profitability.  Entries
combine a medium-term EMA regime with a 24-hour breakout and trend-strength filter.
Normal exits happen after at least three hours; stale positions are closed after
72 hours.  The emergency stop can, correctly, close sooner.
"""

from datetime import datetime, timedelta

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class SwingTrendFutures(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1h"
    startup_candle_count = 100
    process_only_new_candles = True

    # Disable conventional ROI exits: holding-time and regime exits live in custom_exit.
    minimal_roi = {"0": 100.0}
    # Freqtrade interprets this as leveraged trade risk, not raw underlying movement.
    stoploss = -0.10
    trailing_stop = False
    # This must stay enabled for Freqtrade to call custom_exit().  The vectorised
    # exit columns below remain zero, so only the stateful 3h/72h rules act.
    use_exit_signal = True

    order_types = {
        "entry": "limit",
        "exit": "limit",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "#f2a900"},
            "ema_slow": {"color": "#3b82f6"},
            "breakout_high": {"color": "#22c55e"},
            "breakout_low": {"color": "#ef4444"},
        },
        "subplots": {
            "ADX": {"adx": {"color": "#8b5cf6"}},
            "RSI": {"rsi": {"color": "#64748b"}},
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=24)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=72)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        # shift(1) makes the breakout threshold knowable at candle open.
        dataframe["breakout_high"] = dataframe["high"].rolling(24).max().shift(1)
        dataframe["breakout_low"] = dataframe["low"].rolling(24).min().shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        liquid = dataframe["volume"] > 0
        dataframe.loc[
            liquid
            & (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["adx"] > 20)
            & (dataframe["rsi"].between(52, 75))
            & (dataframe["close"] > dataframe["breakout_high"]),
            ["enter_long", "enter_tag"],
        ] = (1, "trend_breakout_long")
        dataframe.loc[
            liquid
            & (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["adx"] > 20)
            & (dataframe["rsi"].between(25, 48))
            & (dataframe["close"] < dataframe["breakout_low"]),
            ["enter_short", "enter_tag"],
        ] = (1, "trend_breakout_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exits are stateful because this baseline enforces a minimum normal hold.
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        age = current_time - trade.open_date_utc
        if age >= timedelta(hours=72):
            return "max_hold_72h"
        if age < timedelta(hours=3):
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None
        last = dataframe.iloc[-1]
        if trade.is_short and last["ema_fast"] > last["ema_slow"]:
            return "short_regime_flip"
        if not trade.is_short and last["ema_fast"] < last["ema_slow"]:
            return "long_regime_flip"
        return None

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(3.0, max_leverage)

