import pandas as pd

from research.walk_forward import month_starts


def test_month_starts_are_non_overlapping_monthly_steps():
    starts = month_starts(pd.Timestamp("2025-01-15", tz="UTC"),
                          pd.Timestamp("2025-04-30", tz="UTC"), 1)
    assert starts == [pd.Timestamp("2025-01-01", tz="UTC"),
                      pd.Timestamp("2025-02-01", tz="UTC"),
                      pd.Timestamp("2025-03-01", tz="UTC"),
                      pd.Timestamp("2025-04-01", tz="UTC")]


def test_two_month_step():
    starts = month_starts(pd.Timestamp("2025-01-01", tz="UTC"),
                          pd.Timestamp("2025-06-01", tz="UTC"), 2)
    assert starts[-1] == pd.Timestamp("2025-05-01", tz="UTC")
