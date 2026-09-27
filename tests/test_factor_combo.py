import numpy as np
import pandas as pd

from research.factor_combo_pipeline import ExpressionEvaluator, rolling_z, safe_div


def test_safe_div_preserves_missing_zero_denominator():
    a = pd.Series([1.0, 2.0])
    b = pd.Series([0.0, 2.0])
    out = safe_div(a, b)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1] == 1.0


def test_expression_operator_semantics():
    index = pd.date_range("2025-01-01", periods=10, freq="h", tz="UTC")
    frame = pd.DataFrame({"x": np.arange(10.0), "y": np.arange(10.0)}, index=index)
    evaluator = ExpressionEvaluator(frame)
    delta = evaluator.evaluate("TsDelta($x,3)")
    assert delta.iloc[3] == 3.0
    corr = evaluator.evaluate("TsCorr($x,$y,4)")
    assert corr.iloc[:3].isna().all()
    assert np.allclose(corr.iloc[3:], 1.0)


def test_rolling_z_is_causal_prefix_invariant():
    x = pd.Series(np.sin(np.arange(500) / 13))
    short = rolling_z(x.iloc[:300], 72)
    full = rolling_z(x, 72).iloc[:300]
    pd.testing.assert_series_equal(short, full)
