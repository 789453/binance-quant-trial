"""Causal factor-combination research pipeline for Binance USDT perpetuals.

The implementation follows reference/factor_combo_strategy/
CRYPTO_FINAL20_FACTOR_IMPLEMENTATION_GUIDE.md.  It intentionally keeps research
portfolio construction separate from Freqtrade's order execution lifecycle.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "reference" / "factor_combo_strategy" / "selected_factors.csv"
DATA_ROOT = ROOT / "data" / "parquet"
OUT_ROOT = ROOT / "reports" / "factor_combo"
SYMBOLS = ["ADAUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT",
           "ETHUSDT", "LINKUSDT", "LTCUSDT", "SOLUSDT", "TRXUSDT", "XRPUSDT"]

# Greedy selection in descending original discovery-period Sharpe, capped at 0.75
# absolute pairwise correlation on this project's pre-2025 training data.  This
# preserves strong candidates while preventing near-duplicate expression families.
SELECTED_HASHES = [
    "b73d09ce90a5bc3f0b50fcbb0f6417b5343b0e6e029a42432fbb75f7251f538c",  # liquidity
    "d7d59064d01de84181bc2674d876609e8cdd69f6a15602f3d46d0c290cf8f324",  # cross-scale
    "6a006820d6e191048c19a01a8e7b88c21d1b3de7c6d0e466c618a40493acef99",  # return-vol
    "762b6d12b47c0b5fbc526eebdd63a9065b282e1486a313b13481bb5fbc6ae844",  # short trade state
    "d06ab7f63133c543138a316c00bfb9f73157d7b427fa239dbf47723257e4943a",  # session volume
    "aa53c4da8642b40c14518d9036cce43d8893d4250099a34172f456d84e53030e",  # short illiquidity
]


def safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    return a.div(b.where(b.abs() >= eps))


def rolling_z(x: pd.Series, window: int = 168, min_periods: int | None = None) -> pd.Series:
    min_periods = min_periods or window // 2 + 1
    r = x.rolling(window, min_periods=min_periods)
    std = r.std(ddof=0)
    return (x - r.mean()).div(std.where(std > 1e-12))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build causal base fields for one symbol."""
    x = frame.sort_values("date").set_index("date").copy()
    x["ret_1h"] = x.close.pct_change(fill_method=None)
    x["ret_4h"] = x.close.pct_change(4, fill_method=None)
    x["ret_24h"] = x.close.pct_change(24, fill_method=None)
    x["oc_ret"] = safe_div(x.close - x.open, x.open)
    x["vwap_bias"] = safe_div(x.close - x.vwap, x.vwap)
    x["hl_range"] = safe_div(x.high - x.low, x.open)
    x["realized_vol_4h"] = x.ret_1h.rolling(4, min_periods=3).std(ddof=0)
    x["realized_vol_24h"] = x.ret_1h.rolling(24, min_periods=12).std(ddof=0)
    x["volatility_term_slope"] = safe_div(x.realized_vol_4h, x.realized_vol_24h) - 1
    x["quote_volume_log"] = np.log1p(x.quote_volume.clip(lower=0))
    x["trade_count_log"] = np.log1p(x.trade_count.clip(lower=0))
    x["liquidity_efficiency"] = safe_div(x.ret_1h.abs(), x.trade_count_log)
    x["amihud_log"] = np.log1p(safe_div(x.ret_1h.abs(), x.quote_volume) * 1e9)
    rr = x.trade_count_log.rolling(24, min_periods=12)
    x["trade_count_shock_24h"] = safe_div(x.trade_count_log - rr.mean(), rr.std(ddof=0))

    ny = x.index.tz_convert("America/New_York")
    x["ny_hour"] = ny.hour
    x["ny_date"] = ny.date
    x["session_code"] = (ny.hour // 8).astype(np.int8)
    x["us_overnight_flag"] = (ny.hour < 8).astype(float)
    first_open = x.groupby(["ny_date", "session_code"], sort=False).open.transform("first")
    x["session_cumulative_return"] = safe_div(x.close, first_open) - 1
    for output, source in [
        ("session_volume_surprise", "quote_volume_log"),
        ("session_volatility_surprise", "hl_range"),
        ("session_illiquidity_surprise", "amihud_log"),
    ]:
        history = x.groupby("ny_hour", sort=False)[source].shift(1)
        grouped = history.groupby(x.ny_hour, sort=False)
        mean = grouped.transform(lambda s: s.rolling(60, min_periods=10).mean())
        std = grouped.transform(lambda s: s.rolling(60, min_periods=10).std(ddof=0))
        x[output] = safe_div(x[source] - mean, std)
    return x.replace([np.inf, -np.inf], np.nan)


class ExpressionEvaluator:
    """Restricted AST evaluator for the six documented factor operators."""
    def __init__(self, features: pd.DataFrame):
        self.features = features

    def evaluate(self, expression: str) -> pd.Series:
        tree = ast.parse(expression.replace("$", ""), mode="eval")
        return self._node(tree.body).replace([np.inf, -np.inf], np.nan)

    def _node(self, node: ast.AST):
        if isinstance(node, ast.Name):
            if node.id not in self.features:
                raise ValueError(f"Unknown field: {node.id}")
            return self.features[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError(f"Unsupported expression node: {ast.dump(node)}")
        args = [self._node(a) for a in node.args]
        op = node.func.id
        if op == "Sub": return args[0] - args[1]
        if op == "Mul": return args[0] * args[1]
        if op == "TsDelta": return args[0] - args[0].shift(args[1])
        if op == "TsEMA": return args[0].ewm(span=args[1], min_periods=args[1], adjust=False).mean()
        if op == "TsZScore": return rolling_z(args[0], args[1])
        if op == "TsCorr":
            w = args[2]
            # Requiring w time positions is stricter than pandas' pair-count check.
            out = args[0].rolling(w, min_periods=max(2, w // 2)).corr(args[1])
            out.iloc[: w - 1] = np.nan
            return out
        raise ValueError(f"Unsupported operator: {op}")


def reference_table() -> pd.DataFrame:
    raw = REFERENCE.read_bytes()
    expected = "d04eae5c2a5118062db4069ac3ebe0e6010ba572c152df686f6b140bc8f64c70"
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("selected_factors.csv hash differs from the implementation guide")
    table = pd.read_csv(REFERENCE)
    if len(table) != 20 or not table.expr_hash.is_unique or not set(table.direction).issubset({-1, 1}):
        raise ValueError("Invalid factor registry")
    return table


def build_factor_panel(force: bool = False) -> pd.DataFrame:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cache = OUT_ROOT / "factor_panel.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    registry = reference_table().set_index("expr_hash")
    rows = []
    for symbol in SYMBOLS:
        raw = pd.read_parquet(DATA_ROOT / symbol / "1h.parquet")
        feat = build_features(raw)
        evaluator = ExpressionEvaluator(feat)
        part = pd.DataFrame(index=feat.index)
        part["symbol"] = symbol
        part["asset_return"] = feat.ret_1h
        part["target_4h"] = feat.close.shift(-4).div(feat.close).sub(1)
        for h, row in registry.iterrows():
            part[f"f_{h[:12]}"] = evaluator.evaluate(row.expr)
        rows.append(part.reset_index())
    panel = pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"])
    panel.to_parquet(cache, index=False, compression="zstd")
    return panel


def factor_columns(panel: pd.DataFrame, selected_only: bool = True) -> list[str]:
    hashes = SELECTED_HASHES if selected_only else reference_table().expr_hash.tolist()
    return [f"f_{h[:12]}" for h in hashes]


def add_oriented_signals(panel: pd.DataFrame, factor_cols: list[str]) -> pd.DataFrame:
    directions = reference_table().set_index("expr_hash").direction
    x = panel.copy()
    for col in factor_cols:
        h = next(h for h in directions.index if h.startswith(col[2:]))
        x[col] = x.groupby("symbol", sort=False)[col].transform(rolling_z) * directions[h]
    return x


def prediction_to_positions(panel: pd.DataFrame, prediction: pd.Series, n_assets: int,
                            normalize_prediction: bool = True) -> pd.DataFrame:
    x = panel[["date", "symbol", "asset_return"]].copy()
    x["raw_signal"] = prediction
    signal = (x.groupby("symbol", sort=False).raw_signal.transform(rolling_z)
              if normalize_prediction else x.raw_signal)
    x["signal"] = signal.clip(-2, 2) / 2
    x["target"] = x.signal / n_assets
    x["position"] = x.groupby("symbol", sort=False).target.transform(
        lambda s: s.rolling(4, min_periods=3).mean().shift(2)
    ).fillna(0.0)
    return x


def evaluate_portfolio(name: str, positions: pd.DataFrame, fee_bps: float = 4.0) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = positions.copy()
    x["prev"] = x.groupby("symbol", sort=False).position.shift(1).fillna(0)
    x["turnover"] = (x.position - x.prev).abs()
    x["gross_pnl"] = x.position * x.asset_return.fillna(0)
    x["net_pnl"] = x.gross_pnl - x.turnover * fee_bps / 10_000
    hourly = x.groupby("date", sort=True)[["gross_pnl", "turnover", "net_pnl"]].sum()
    hourly["equity"] = (1 + hourly.net_pnl).cumprod()
    hourly["drawdown"] = hourly.equity.div(hourly.equity.cummax()).sub(1)
    active = x.position.ne(0)
    entries = active & ~x.groupby("symbol", sort=False).position.shift(1).fillna(0).ne(0)
    flips = x.position.mul(x.prev).lt(0)
    trade_events = entries | flips
    trade_rows = []
    for symbol, asset in x.groupby("symbol", sort=False):
        asset = asset.sort_values("date").copy()
        sign = np.sign(asset.position)
        episode = sign.ne(sign.shift()).cumsum()
        for _, trade in asset.loc[sign.ne(0)].groupby(episode[sign.ne(0)]):
            trade_rows.append({
                "strategy": name, "symbol": symbol,
                "entry_time": trade.date.iloc[0], "exit_time": trade.date.iloc[-1],
                "direction": "long" if trade.position.iloc[0] > 0 else "short",
                "holding_hours": len(trade), "net_pnl": trade.net_pnl.sum(),
                "gross_pnl": trade.gross_pnl.sum(), "turnover": trade.turnover.sum(),
            })
    trades = pd.DataFrame(trade_rows)
    pnl = hourly.net_pnl
    years = max((hourly.index.max() - hourly.index.min()).total_seconds() / (365.25 * 86400), 1 / 365.25)
    ann = 365.25 * 24
    metrics = {
        "strategy": name,
        "start": str(hourly.index.min()), "end": str(hourly.index.max()),
        "total_return": hourly.equity.iloc[-1] - 1,
        "cagr": hourly.equity.iloc[-1] ** (1 / years) - 1,
        "sharpe": math.sqrt(ann) * pnl.mean() / pnl.std(ddof=0) if pnl.std(ddof=0) else np.nan,
        "max_drawdown": hourly.drawdown.min(),
        "annualized_volatility": pnl.std(ddof=0) * math.sqrt(ann),
        "mean_hourly_turnover": hourly.turnover.mean(),
        "annualized_turnover": hourly.turnover.mean() * ann,
        "trade_entries_or_flips": int(trade_events.sum()),
        "active_asset_hours": int(active.sum()),
        "positive_hour_ratio": float((pnl > 0).mean()),
        "gross_total_return": float((1 + hourly.gross_pnl).prod() - 1),
        "estimated_cost_drag": float(hourly.gross_pnl.sum() - hourly.net_pnl.sum()),
        "episode_win_rate": float((trades.net_pnl > 0).mean()) if len(trades) else np.nan,
        "median_episode_pnl": float(trades.net_pnl.median()) if len(trades) else np.nan,
        "profit_factor": float(trades.loc[trades.net_pnl > 0, "net_pnl"].sum() /
                               -trades.loc[trades.net_pnl < 0, "net_pnl"].sum())
                         if (trades.net_pnl < 0).any() else np.nan,
    }
    by_asset = x.groupby("symbol").agg(
        net_pnl=("net_pnl", "sum"), turnover=("turnover", "sum"),
        active_hours=("position", lambda s: int(s.ne(0).sum())),
        entries_or_flips=("position", lambda s: int((s.ne(0) & (s.shift().fillna(0).eq(0) | s.mul(s.shift()).lt(0))).sum())),
    ).reset_index()
    return metrics, hourly.reset_index(), by_asset, trades


def fit_predict_lstm_btce(panel: pd.DataFrame, features: list[str], train_end: str,
                          sequence: int = 24) -> tuple[pd.Series | None, dict | None]:
    """Small sequence-model benchmark, deliberately restricted to BTC/ETH."""
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:  # pragma: no cover
        return None, None
    torch.manual_seed(42)
    np.random.seed(42)
    cutoff = pd.Timestamp(train_end, tz="UTC")
    subset = panel.loc[panel.symbol.isin(["BTCUSDT", "ETHUSDT"])].copy()
    train_values = subset.loc[subset.date < cutoff - pd.Timedelta(hours=4), features]
    mean, std = train_values.mean(), train_values.std(ddof=0).replace(0, 1)
    train_x, train_y, test_x, test_index = [], [], [], []
    for _, asset in subset.groupby("symbol", sort=False):
        asset = asset.sort_values("date")
        values = ((asset[features] - mean) / std).clip(-8, 8).to_numpy(np.float32)
        target = (asset.target_4h.clip(-0.20, 0.20) * 100).to_numpy(np.float32)
        dates = asset.date.to_numpy()
        indices = asset.index.to_numpy()
        for i in range(sequence - 1, len(asset)):
            window = values[i - sequence + 1:i + 1]
            if not np.isfinite(window).all() or not np.isfinite(target[i]):
                continue
            date = pd.Timestamp(dates[i])
            if date < cutoff - pd.Timedelta(hours=4):
                train_x.append(window); train_y.append(target[i])
            elif date >= cutoff:
                test_x.append(window); test_index.append(indices[i])
    if not train_x or not test_x:
        return None, None

    class TinyLSTM(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.lstm = nn.LSTM(n_features, 16, batch_first=True)
            self.head = nn.Sequential(nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1))
        def forward(self, value):
            return self.head(self.lstm(value)[0][:, -1]).squeeze(-1)

    model = TinyLSTM(len(features))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.002)
    loss_fn = nn.HuberLoss(delta=0.5)
    dataset = TensorDataset(torch.tensor(np.stack(train_x)), torch.tensor(train_y))
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    started = time.time()
    model.train()
    for _ in range(8):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    chunks = []
    test_tensor = torch.tensor(np.stack(test_x))
    with torch.no_grad():
        for start in range(0, len(test_tensor), 2048):
            chunks.append(model(test_tensor[start:start + 2048]).numpy() / 100)
    values = np.concatenate(chunks)
    prediction = pd.Series(np.nan, index=panel.index, dtype=float)
    prediction.loc[test_index] = values
    truth = panel.loc[test_index, "target_4h"].to_numpy()
    diagnostic = {
        "model": "lstm_btce", "train_rows": len(train_x), "test_rows": len(test_x),
        "test_start": str(panel.loc[test_index, "date"].min()), "test_end": str(panel.loc[test_index, "date"].max()),
        "test_rmse": float(mean_squared_error(truth, values) ** 0.5),
        "test_correlation": float(np.corrcoef(truth, values)[0, 1]), "fit_seconds": time.time() - started,
    }
    return prediction, diagnostic


def fit_predict_models(panel: pd.DataFrame, features: list[str], train_end: str) -> tuple[dict[str, pd.Series], list[dict]]:
    """Frozen, purged train/test comparison. No test-period refitting."""
    usable = panel.dropna(subset=features + ["target_4h"]).copy()
    cutoff = pd.Timestamp(train_end, tz="UTC")
    train = usable.date < cutoff - pd.Timedelta(hours=4)
    test = usable.date >= cutoff
    X_train = usable.loc[train, features + ["symbol"]]
    y_train = usable.loc[train, "target_4h"].clip(-0.20, 0.20)
    X_test = usable.loc[test, features + ["symbol"]]
    y_test = usable.loc[test, "target_4h"].clip(-0.20, 0.20)
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    pre = ColumnTransformer([("num", numeric, features), ("sym", OneHotEncoder(handle_unknown="ignore"), ["symbol"])])
    models = {
        "ridge": Ridge(alpha=20.0),
        "mlp": MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.002, early_stopping=True,
                            max_iter=80, batch_size=2048, random_state=42),
    }
    if LGBMRegressor is not None:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=350, learning_rate=0.025, num_leaves=15, max_depth=5,
            min_child_samples=300, subsample=0.8, colsample_bytree=0.85,
            reg_alpha=0.2, reg_lambda=2.0, random_state=42, verbosity=-1, n_jobs=-1,
        )
    predictions: dict[str, pd.Series] = {}
    diagnostics = []
    for name, model in models.items():
        started = time.time()
        pipe = make_pipeline(pre, model)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        full = pd.Series(np.nan, index=panel.index, dtype=float)
        full.loc[usable.index[test]] = pred
        predictions[name] = full
        diagnostics.append({
            "model": name, "train_rows": int(train.sum()), "test_rows": int(test.sum()),
            "test_start": str(usable.loc[test, "date"].min()),
            "test_end": str(usable.loc[test, "date"].max()),
            "test_rmse": mean_squared_error(y_test, pred) ** 0.5,
            "test_correlation": float(np.corrcoef(y_test, pred)[0, 1]),
            "fit_seconds": time.time() - started,
        })
    return predictions, diagnostics


def run(force: bool = False, train_end: str = "2025-01-01", fee_bps: float = 4.0) -> None:
    panel = build_factor_panel(force=force)
    selected = factor_columns(panel)
    oriented = add_oriented_signals(panel, selected)
    test_start = pd.Timestamp(train_end, tz="UTC")
    eval_panel = oriented.loc[oriented.date >= test_start].copy()

    diag_rows = []
    for col, factor_hash in zip(selected, SELECTED_HASHES):
        for split, mask in [("train", oriented.date < test_start), ("test", oriented.date >= test_start)]:
            sample = oriented.loc[mask, [col, "target_4h"]].dropna()
            diag_rows.append({
                "expr_hash": factor_hash, "split": split, "rows": len(sample),
                "coverage": len(sample) / max(int(mask.sum()), 1),
                "pearson_ic_4h": sample[col].corr(sample.target_4h, method="pearson"),
                "spearman_ic_4h": sample[col].corr(sample.target_4h, method="spearman"),
            })
    pd.DataFrame(diag_rows).to_csv(OUT_ROOT / "factor_diagnostics.csv", index=False)
    train_corr = oriented.loc[oriented.date < test_start, selected].corr()
    max_pair_corr = train_corr.abs().where(~np.eye(len(selected), dtype=bool)).max().max()
    if max_pair_corr > 0.75 + 1e-12:
        raise ValueError(f"Selected factors violate the frozen 0.75 correlation cap: {max_pair_corr:.6f}")
    train_corr.to_csv(OUT_ROOT / "selected_factor_train_correlation.csv")

    predictions, diagnostics = fit_predict_models(oriented, selected, train_end)
    lstm_prediction, lstm_diagnostic = fit_predict_lstm_btce(oriented, selected, train_end)
    if lstm_diagnostic:
        diagnostics.append(lstm_diagnostic)
    strategies: dict[str, pd.Series] = {"group_equal_6": oriented[selected].mean(axis=1)}
    strategies.update(predictions)
    all_metrics, holdout_metrics, all_hourly, all_assets, all_trades = [], [], [], [], []
    for name, pred in strategies.items():
        # Models only predict OOS. Equal-weight is restricted to the identical test interval.
        local_pred = pred.loc[eval_panel.index]
        pos = prediction_to_positions(eval_panel, local_pred, len(SYMBOLS), normalize_prediction=name != "group_equal_6")
        metrics, hourly, assets, trades = evaluate_portfolio(name, pos, fee_bps)
        all_metrics.append(metrics)
        hourly.insert(0, "strategy", name)
        assets.insert(0, "strategy", name)
        all_hourly.append(hourly)
        all_assets.append(assets)
        all_trades.append(trades)
        holdout = pos.loc[pos.date >= pd.Timestamp("2026-01-01", tz="UTC")]
        holdout_metrics.append(evaluate_portfolio(name, holdout, fee_bps)[0])
    if lstm_prediction is not None:
        btce = eval_panel.loc[eval_panel.symbol.isin(["BTCUSDT", "ETHUSDT"])].copy()
        for name, pred in {
            "group_equal_6_btce": oriented[selected].mean(axis=1),
            "lstm_btce": lstm_prediction,
        }.items():
            pos = prediction_to_positions(btce, pred.loc[btce.index], 2,
                                          normalize_prediction=name != "group_equal_6_btce")
            metrics, hourly, assets, trades = evaluate_portfolio(name, pos, fee_bps)
            all_metrics.append(metrics)
            hourly.insert(0, "strategy", name); assets.insert(0, "strategy", name)
            all_hourly.append(hourly); all_assets.append(assets); all_trades.append(trades)
            holdout = pos.loc[pos.date >= pd.Timestamp("2026-01-01", tz="UTC")]
            holdout_metrics.append(evaluate_portfolio(name, holdout, fee_bps)[0])

    metrics_df = pd.DataFrame(all_metrics).sort_values("sharpe", ascending=False)
    metrics_df.to_csv(OUT_ROOT / "strategy_comparison.csv", index=False)
    holdout_df = pd.DataFrame(holdout_metrics).sort_values("sharpe", ascending=False)
    holdout_df.to_csv(OUT_ROOT / "strategy_holdout_2026.csv", index=False)
    pd.concat(all_hourly).to_parquet(OUT_ROOT / "hourly_returns.parquet", index=False, compression="zstd")
    pd.concat(all_assets).to_csv(OUT_ROOT / "by_asset.csv", index=False)
    pd.concat(all_trades, ignore_index=True).to_parquet(OUT_ROOT / "trade_episodes.parquet", index=False, compression="zstd")
    pd.DataFrame(diagnostics).to_csv(OUT_ROOT / "model_diagnostics.csv", index=False)
    selection = reference_table().set_index("expr_hash").loc[SELECTED_HASHES].reset_index()
    selection.to_csv(OUT_ROOT / "selected_combo_factors.csv", index=False)
    manifest = {
        "created_utc": str(pd.Timestamp.now(tz="UTC")), "train_end": train_end,
        "fee_bps_per_unit_turnover": fee_bps, "symbols": SYMBOLS,
        "selected_hashes": SELECTED_HASHES,
        "execution": {"signal_z_window": 168, "clip": [-2, 2], "batches": 4, "position_lag_bars": 2},
        "data_max": str(panel.date.max()), "factor_registry_sha256": hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
    }
    (OUT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(metrics_df, holdout_df, pd.DataFrame(diagnostics), selection, manifest)
    print(metrics_df.to_string(index=False))


def render_report(metrics: pd.DataFrame, holdout: pd.DataFrame, diagnostics: pd.DataFrame,
                  selection: pd.DataFrame, manifest: dict) -> None:
    pct = ["total_return", "cagr", "max_drawdown", "annualized_volatility", "positive_hour_ratio", "gross_total_return", "estimated_cost_drag"]
    shown = metrics.copy()
    holdout_shown = holdout.copy()
    for c in pct: shown[c] = shown[c].map(lambda v: f"{v:.2%}")
    for c in pct: holdout_shown[c] = holdout_shown[c].map(lambda v: f"{v:.2%}")
    chosen = "\n".join(f"- `{r.expr_hash[:12]}` × {int(r.direction):+d}: `{r.expr}`" for r in selection.itertuples())
    text = f"""# 20 因子组合基线：验证与时间留出比较

生成时间：{manifest['created_utc']}  
数据截止：{manifest['data_max']}  
冻结训练截止：{manifest['train_end']}（标签另做 4h purge）

## 结论口径

所有指标均为冻结训练期之后的样本外结果；手续费按单边仓位变化 {manifest['fee_bps_per_unit_turnover']:.1f} bps 计，未含资金费率和额外滑点。仓位使用 168h 因果标准化、4 个重叠批次、`t-2` 执行。结果是研究组合净值，不等同于 Freqtrade 的逐笔订单回测。

{shown.to_markdown(index=False)}

2025 年结果已经用于比较组合结构，因此更严格的未触碰时间留出期为 2026-01-01 以后：

{holdout_shown.to_markdown(index=False)}

## 模型预测诊断

{diagnostics.to_markdown(index=False) if len(diagnostics) else '无'}

## 冻结的 6 因子灰盒组合

{chosen}

六因子按原研究发现期 Sharpe 降序选择，并强制 2025 年以前训练期两两绝对相关不超过 0.75；先固定 CSV 中方向，再做策略层 168h z-score。模型使用完全相同的 6 个输入：Ridge 提供线性下界，LightGBM 提供非线性树模型，MLP 提供小型神经网络对照。

## 审计限制

- 当前结果没有资金费率、盘口冲击、最小下单量与交易所拒单模拟，不能直接用于实盘。
- `trade_entries_or_flips` 是目标仓位从零进入或翻转的事件数；连续调仓另由 turnover 衡量。
- 神经网络仅作为容量上界，不因一次测试胜出就选为生产模型。
- 下一阶段应做滚动重训、分市场状态归因、资金费率接入，并将目标仓位映射为 Freqtrade 可执行的有限状态机。
"""
    (OUT_ROOT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="rebuild factor cache")
    p.add_argument("--train-end", default="2025-01-01")
    p.add_argument("--fee-bps", type=float, default=4.0)
    args = p.parse_args()
    run(args.force, args.train_end, args.fee_bps)


if __name__ == "__main__":
    main()
