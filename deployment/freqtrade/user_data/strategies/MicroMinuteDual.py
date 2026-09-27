"""Small-capital BTC/ETH minute strategy with two independent signal modules.

The bot accounting is capped by config to 10 USDT (two 5 USDT slots).  This
strategy additionally locks each pair when cumulative closed PnL in its dedicated
database reaches -2.5 USDT.  It is designed for execution testing, not as a claim
of profitability.
"""

from datetime import UTC, datetime

from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy


class MicroMinuteDual(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1m"
    startup_candle_count = 150
    process_only_new_candles = True

    # Values are leveraged trade returns.  With a 5 USDT stake, -8% is about
    # -0.40 USDT before slippage/fees, allowing several attempts before the cap.
    stoploss = -0.08
    minimal_roi = {"0": 0.025, "15": 0.012, "45": 0.0}
    trailing_stop = False
    use_exit_signal = True

    pair_loss_cap_usdt = 2.5

    order_types = {
        "entry": "market",
        "exit": "market",
        "emergency_exit": "market",
        "force_entry": "market",
        "force_exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": True,
        "stoploss_on_exchange_interval": 60,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    @property
    def protections(self):
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 2},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 30,
                "trade_limit": 2,
                "stop_duration_candles": 10,
                "only_per_pair": True,
                "only_per_side": False,
            },
        ]

    def bot_start(self, **kwargs) -> None:
        self._loss_locked_pairs: set[str] = set()

    @staticmethod
    def _closed_pnl(pair: str) -> float:
        trades = Trade.get_trades_proxy(pair=pair, is_open=False)
        return sum(float(trade.close_profit_abs or 0.0) for trade in trades)

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        if self.config["runmode"].value not in ("live", "dry_run"):
            return
        for pair in self.dp.current_whitelist():
            if pair in self._loss_locked_pairs:
                continue
            if self._closed_pnl(pair) <= -self.pair_loss_cap_usdt:
                self.lock_pair(
                    pair,
                    datetime(2035, 1, 1, tzinfo=UTC),
                    reason="pair_loss_cap_-2.5_USDT",
                    side="*",
                )
                self._loss_locked_pairs.add(pair)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema_mid"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=36)
        dataframe["ema_regime"] = ta.EMA(dataframe, timeperiod=120)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["volume_mean"] = dataframe["volume"].rolling(20).mean()
        dataframe["prior_high"] = dataframe["high"].rolling(20).max().shift(1)
        dataframe["prior_low"] = dataframe["low"].rolling(20).min().shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        liquid = (dataframe["volume"] > 0) & (dataframe["volume_mean"] > 0)
        active_vol = dataframe["atr_pct"].between(0.00025, 0.008)
        reclaim_fast = (dataframe["close"] > dataframe["ema_fast"]) & (
            dataframe["close"].shift(1) <= dataframe["ema_fast"].shift(1)
        )
        reject_fast = (dataframe["close"] < dataframe["ema_fast"]) & (
            dataframe["close"].shift(1) >= dataframe["ema_fast"].shift(1)
        )
        adx_rising = dataframe["adx"] > dataframe["adx"].shift(3)

        # Module 1: short-horizon trend transition with strength/volatility filters.
        dataframe.loc[
            liquid
            & active_vol
            & (dataframe["volume"] > dataframe["volume_mean"])
            & (dataframe["adx"] > 18)
            & adx_rising
            & (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_slow"] > dataframe["ema_regime"])
            & reclaim_fast
            & dataframe["rsi"].between(50, 68),
            ["enter_long", "enter_tag"],
        ] = (1, "trend_long")
        dataframe.loc[
            liquid
            & active_vol
            & (dataframe["volume"] > dataframe["volume_mean"])
            & (dataframe["adx"] > 18)
            & adx_rising
            & (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["ema_slow"] < dataframe["ema_regime"])
            & reject_fast
            & dataframe["rsi"].between(32, 50),
            ["enter_short", "enter_tag"],
        ] = (1, "trend_short")

        # Module 2: false-breakout reversal.  The wick breaches the previous
        # 20-minute range but the candle closes back inside it.
        reversal_volume = dataframe["volume"] > (1.2 * dataframe["volume_mean"])
        dataframe.loc[
            liquid
            & active_vol
            & reversal_volume
            & (dataframe["adx"] < 24)
            & (dataframe["low"] < dataframe["prior_low"] - 0.25 * dataframe["atr"])
            & (dataframe["close"] > dataframe["prior_low"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["rsi"] < 35),
            ["enter_long", "enter_tag"],
        ] = (1, "reversal_long")
        dataframe.loc[
            liquid
            & active_vol
            & reversal_volume
            & (dataframe["adx"] < 24)
            & (dataframe["high"] > dataframe["prior_high"] + 0.25 * dataframe["atr"])
            & (dataframe["close"] < dataframe["prior_high"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["rsi"] > 65),
            ["enter_short", "enter_tag"],
        ] = (1, "reversal_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
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
        closed_pnl = self._closed_pnl(pair)
        estimated_open_pnl = current_profit * float(trade.stake_amount)
        if closed_pnl + estimated_open_pnl <= -self.pair_loss_cap_usdt:
            return "pair_loss_cap"

        age_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        if age_minutes < 2:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None
        last = dataframe.iloc[-1]
        tag = trade.enter_tag or ""

        if tag.startswith("trend_"):
            if (not trade.is_short and last["ema_fast"] < last["ema_mid"]) or (
                trade.is_short and last["ema_fast"] > last["ema_mid"]
            ):
                return "trend_flip"
            if age_minutes >= 45:
                return "trend_timeout"
        elif tag.startswith("reversal_"):
            if (not trade.is_short and current_rate >= last["ema_mid"]) or (
                trade.is_short and current_rate <= last["ema_mid"]
            ):
                return "reversion_complete"
            if age_minutes >= 20:
                return "reversal_timeout"
        return None

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        return self._closed_pnl(pair) > -self.pair_loss_cap_usdt

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return 5.0

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
        requested = 20.0 if pair.startswith("BTC/") else 5.0
        return min(requested, max_leverage)

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "#f59e0b"},
            "ema_mid": {"color": "#8b5cf6"},
            "ema_slow": {"color": "#2563eb"},
            "ema_regime": {"color": "#111827"},
            "prior_high": {"color": "#16a34a"},
            "prior_low": {"color": "#dc2626"},
        },
        "subplots": {
            "ADX": {"adx": {}},
            "RSI": {"rsi": {}},
            "ATR %": {"atr_pct": {}},
        },
    }


class MicroMinuteDualLookahead(MicroMinuteDual):
    """Analysis-only variant so Freqtrade can resize stakes during bias tests."""

    pair_loss_cap_usdt = 1_000_000_000.0

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        return None

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return proposed_stake

