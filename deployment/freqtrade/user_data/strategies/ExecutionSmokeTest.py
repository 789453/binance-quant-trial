"""Execution-only strategy for controlled dry-run/live order round trips.

It never creates autonomous entry or exit signals. Trades can only be opened via
Freqtrade's authenticated force-enter API and are closed immediately by the smoke
test driver. This keeps execution testing separate from research strategies.
"""

from pandas import DataFrame

from freqtrade.strategy import IStrategy


class ExecutionSmokeTest(IStrategy):
    INTERFACE_VERSION = 3
    can_short = True
    timeframe = "1m"
    startup_candle_count = 1
    process_only_new_candles = True

    minimal_roi = {"0": 100.0}
    stoploss = -0.02
    trailing_stop = False
    use_exit_signal = True

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

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return min(2.0, max_leverage)

