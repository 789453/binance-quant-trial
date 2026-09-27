"""Execution-aligned, turnover-aware walk-forward models and holding audit."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from research.factor_combo_pipeline import (
    LGBMRegressor, OUT_ROOT, SELECTED_HASHES, SYMBOLS, add_oriented_signals,
    build_factor_panel, evaluate_portfolio, factor_columns, reference_table,
)
from research.walk_forward import (
    aggregate_decomposition, buy_and_hold, combine_sleeves, factor_contributions,
    factor_sleeve_positions, metrics_from_returns, month_starts, pnl_detail, save_dashboard,
)

V2_ROOT = OUT_ROOT / "walk_forward_v2"
LABEL_END_LAG = 5


def prepare_learning_panel(panel: pd.DataFrame, factors: list[str]) -> tuple[pd.DataFrame, list[str]]:
    x = panel.sort_values(["symbol", "date"]).copy()
    groups = x.groupby("symbol", sort=False)
    model_features: list[str] = []
    for col in factors:
        model_features.append(col)
        for lag in [6, 24]:
            lag_col = f"{col}_lag{lag}"
            delta_col = f"{col}_delta{lag}"
            x[lag_col] = groups[col].shift(lag)
            x[delta_col] = x[col] - x[lag_col]
            model_features.extend([lag_col, delta_col])
    x["ret_4h_lag"] = groups.asset_return.transform(
        lambda s: (1 + s).rolling(4, min_periods=4).apply(np.prod, raw=True) - 1)
    x["ret_24h_lag"] = groups.asset_return.transform(
        lambda s: (1 + s).rolling(24, min_periods=12).apply(np.prod, raw=True) - 1)
    x["vol_24h"] = groups.asset_return.transform(lambda s: s.rolling(24, min_periods=12).std(ddof=0))
    ny_hour = x.date.dt.tz_convert("America/New_York").dt.hour
    x["ny_hour_sin"] = np.sin(2 * np.pi * ny_hour / 24)
    x["ny_hour_cos"] = np.cos(2 * np.pi * ny_hour / 24)
    model_features.extend(["ret_4h_lag", "ret_24h_lag", "vol_24h", "ny_hour_sin", "ny_hour_cos"])

    # A signal observed at t first earns return[t+2], and its four batches cover
    # return[t+2] ... return[t+5].  This aligns the label with portfolio execution.
    future = [groups.asset_return.shift(-k) for k in range(2, 6)]
    raw = np.prod([1 + value for value in future], axis=0) - 1
    x["target_exec_4h"] = raw
    scale = x.vol_24h.clip(lower=1e-5) * 2.0
    x["target_exec_4h_risk"] = (x.target_exec_4h / scale).clip(-6, 6)
    return x, model_features


def preprocessor(features: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), features),
        ("symbol", OneHotEncoder(handle_unknown="ignore"), ["symbol"]),
    ])


def decay_weights(dates: pd.Series, end: pd.Timestamp, half_life_days: float = 90) -> np.ndarray:
    age_days = (end - dates).dt.total_seconds().to_numpy() / 86400
    return np.power(0.5, np.maximum(age_days, 0) / half_life_days)


def fit_ridge(train: pd.DataFrame, features: list[str], fold_start: pd.Timestamp, config: dict):
    val_start = fold_start - pd.DateOffset(months=1)
    fit = train.date < val_start
    val = train.date >= val_start
    best_alpha, best_score = 20.0, -np.inf
    half_life = float(config.get("sample_half_life_days", 90.0))
    for alpha in config.get("alphas", [1.0, 10.0, 50.0, 200.0]):
        pipe = make_pipeline(preprocessor(features), Ridge(alpha=alpha))
        pipe.fit(train.loc[fit, features + ["symbol"]], train.loc[fit, "target_exec_4h_risk"],
                 ridge__sample_weight=decay_weights(train.loc[fit, "date"], val_start, half_life))
        pred = pipe.predict(train.loc[val, features + ["symbol"]])
        score = pd.Series(pred).corr(train.loc[val, "target_exec_4h_risk"].reset_index(drop=True))
        if np.isfinite(score) and score > best_score:
            best_alpha, best_score = alpha, score
    final = make_pipeline(preprocessor(features), Ridge(alpha=best_alpha))
    final.fit(train[features + ["symbol"]], train.target_exec_4h_risk,
              ridge__sample_weight=decay_weights(train.date, fold_start, half_life))
    return final, {"alpha": best_alpha, "validation_ic": best_score}


def fit_lightgbm(train: pd.DataFrame, features: list[str], fold_start: pd.Timestamp, config: dict):
    import lightgbm as lgb
    val_start = fold_start - pd.DateOffset(months=1)
    fit = train.date < val_start
    val = train.date >= val_start
    prep = preprocessor(features)
    x_fit = prep.fit_transform(train.loc[fit, features + ["symbol"]])
    x_val = prep.transform(train.loc[val, features + ["symbol"]])
    params = dict(objective="huber", alpha=0.85, n_estimators=1000, learning_rate=0.025,
                  num_leaves=7, max_depth=3, min_child_samples=400, subsample=0.8,
                  colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=5.0,
                  random_state=42, verbosity=-1, n_jobs=-1)
    params.update({key: value for key, value in config.items() if key in params})
    probe = LGBMRegressor(**params)
    probe.fit(x_fit, train.loc[fit, "target_exec_4h_risk"],
              sample_weight=decay_weights(train.loc[fit, "date"], val_start),
              eval_set=[(x_val, train.loc[val, "target_exec_4h_risk"])],
              callbacks=[lgb.early_stopping(40, verbose=False)])
    best_iteration = int(probe.best_iteration_ or 200)
    final_params = dict(params)
    final_params["n_estimators"] = best_iteration
    model = LGBMRegressor(**final_params)
    x_all = prep.fit_transform(train[features + ["symbol"]])
    model.fit(x_all, train.target_exec_4h_risk,
              sample_weight=decay_weights(train.date, fold_start))
    return prep, model, {"best_iteration": best_iteration}


def walk_forward(panel: pd.DataFrame, features: list[str], train_months: int,
                 step_months: int, start: str | None = None, model_config: dict | None = None):
    model_config = model_config or {}
    train_data = panel.dropna(subset=features + ["target_exec_4h_risk"]).copy()
    predict_data = panel.dropna(subset=features).copy()
    first = train_data.date.min() + pd.DateOffset(months=train_months)
    inferred = pd.Timestamp(year=first.year, month=first.month, day=1, tz="UTC")
    prediction_start = pd.Timestamp(start, tz="UTC") if start else inferred
    folds = month_starts(prediction_start, panel.date.max(), step_months)
    predictions = {name: pd.Series(np.nan, index=panel.index) for name in ["ridge_v2", "lightgbm_v2"]}
    diagnostics = []
    for fold_start in folds:
        fold_end = min(fold_start + pd.DateOffset(months=step_months), panel.date.max() + pd.Timedelta(hours=1))
        train_start = fold_start - pd.DateOffset(months=train_months)
        train_end = fold_start - pd.Timedelta(hours=LABEL_END_LAG)
        train = train_data.loc[train_data.date.ge(train_start) & train_data.date.lt(train_end)]
        test = predict_data.loc[predict_data.date.ge(fold_start) & predict_data.date.lt(fold_end)]
        if len(train) < 5_000 or test.empty:
            continue
        for name in predictions:
            started = time.time()
            if name == "ridge_v2":
                model, params = fit_ridge(train, features, fold_start, model_config.get("ridge", {}))
                pred = model.predict(test[features + ["symbol"]])
            else:
                prep, model, params = fit_lightgbm(train, features, fold_start, model_config.get("lightgbm", {}))
                pred = model.predict(prep.transform(test[features + ["symbol"]]))
            predictions[name].loc[test.index] = pred
            known = test.target_exec_4h_risk.notna().to_numpy()
            actual = test.target_exec_4h_risk.to_numpy()[known]
            scored = pred[known]
            diagnostics.append({
                "model": name, "fold_start": fold_start, "fold_end": fold_end,
                "train_start": train_start, "train_end_exclusive": train_end,
                "train_rows": len(train), "test_rows": len(test),
                "prediction_ic": np.corrcoef(actual, scored)[0, 1],
                "rmse": mean_squared_error(actual, scored) ** 0.5,
                "fit_seconds": time.time() - started, **params,
            })
    return predictions, pd.DataFrame(diagnostics)


def _deadband(series: pd.Series, threshold: float) -> pd.Series:
    values = series.fillna(0).to_numpy()
    held = np.zeros_like(values)
    previous = 0.0
    for i, desired in enumerate(values):
        sign_flip = desired * previous < 0
        close = desired == 0 and previous != 0
        if sign_flip or close or abs(desired - previous) >= threshold:
            previous = desired
        held[i] = previous
    return pd.Series(held, index=series.index)


def model_positions(panel: pd.DataFrame, prediction: pd.Series, n_assets: int,
                    z_window: int = 336, smooth_span: int = 8,
                    neutral_zone: float = 0.20, deadband: float = 0.004) -> pd.DataFrame:
    x = panel[["date", "symbol", "asset_return"]].copy()
    x["raw_signal"] = prediction
    grouped = x.groupby("symbol", sort=False).raw_signal
    mean = grouped.transform(lambda s: s.rolling(z_window, min_periods=168).mean())
    std = grouped.transform(lambda s: s.rolling(z_window, min_periods=168).std(ddof=0)).replace(0, np.nan)
    x["signal"] = ((x.raw_signal - mean) / std).groupby(x.symbol, sort=False).transform(
        lambda s: s.ewm(span=smooth_span, adjust=False, min_periods=3).mean())
    x.loc[x.signal.abs() < neutral_zone, "signal"] = 0.0
    x["target"] = x.signal.clip(-2, 2) / 2 / n_assets
    x["position_pre_deadband"] = x.groupby("symbol", sort=False).target.transform(
        lambda s: s.rolling(4, min_periods=3).mean().shift(2)).fillna(0)
    x["position"] = x.groupby("symbol", sort=False, group_keys=False).position_pre_deadband.apply(
        lambda s: _deadband(s, deadband)).fillna(0)
    return x


def adaptive_prior_blend(base: pd.DataFrame, model: pd.DataFrame, fee_bps: float,
                         lookback_hours: int = 90 * 24, max_model_weight: float = .25) -> pd.DataFrame:
    """Causally admit a model sleeve only when its trailing alpha is positive."""
    hourly = {}
    for name, frame in [("base", base), ("model", model)]:
        z = frame.sort_values(["symbol", "date"]).copy()
        previous = z.groupby("symbol", sort=False).position.shift(1).fillna(0)
        z["net"] = z.position * z.asset_return.fillna(0) - (z.position - previous).abs() * fee_bps / 10_000
        hourly[name] = z.groupby("date").net.sum()
    alpha = hourly["model"] - hourly["base"]
    rolling = alpha.rolling(lookback_hours, min_periods=30 * 24)
    trailing_sharpe = (rolling.mean() / rolling.std(ddof=0) * math.sqrt(365.25 * 24)).shift(1)
    weight = (max_model_weight * (trailing_sharpe / 1.5).clip(0, 1)).fillna(0)
    out = base.copy()
    aligned_weight = weight.reindex(out.date).fillna(0).to_numpy()
    for col in ["raw_signal", "signal", "target", "position"]:
        base_value = base[col].fillna(0).to_numpy()
        model_value = model[col].fillna(0).to_numpy()
        out[col] = (1 - aligned_weight) * base_value + aligned_weight * model_value
    out["model_weight"] = aligned_weight
    return out


def episode_table(positions: pd.DataFrame, strategy: str, fee_bps: float) -> pd.DataFrame:
    detail = pnl_detail(positions, fee_bps)
    rows = []
    for symbol, asset in detail.groupby("symbol", sort=False):
        sign = np.sign(asset.position)
        episode_id = sign.ne(sign.shift()).cumsum()
        for _, episode in asset.loc[sign.ne(0)].groupby(episode_id[sign.ne(0)]):
            rows.append({
                "strategy": strategy, "symbol": symbol,
                "direction": "long" if episode.position.iloc[0] > 0 else "short",
                "entry_time": episode.date.iloc[0], "exit_time": episode.date.iloc[-1],
                "holding_hours": len(episode), "mean_abs_position": episode.position.abs().mean(),
                "max_abs_position": episode.position.abs().max(), "net_pnl": episode.net_pnl.sum(),
                "turnover": episode.turnover.sum(),
            })
    return pd.DataFrame(rows)


def holding_audit(positions: dict[str, pd.DataFrame], fee_bps: float):
    summaries, episodes, exposures, contract_rows = [], [], [], []
    for strategy, pos in positions.items():
        ep = episode_table(pos, strategy, fee_bps)
        episodes.append(ep)
        x = pos.sort_values(["date", "symbol"]).copy()
        x["abs_position"] = x.position.abs()
        for symbol, asset in x.groupby("symbol"):
            asset_ep = ep.loc[ep.symbol == symbol]
            for direction in ["long", "short"]:
                selected = asset_ep.loc[asset_ep.direction == direction]
                contract_rows.append({
                    "strategy": strategy, "symbol": symbol, "direction": direction,
                    "asset_hour_share": float((asset.position.gt(0) if direction == "long" else asset.position.lt(0)).mean()),
                    "episodes": len(selected), "mean_holding_hours": selected.holding_hours.mean(),
                    "median_holding_hours": selected.holding_hours.median(),
                    "p90_holding_hours": selected.holding_hours.quantile(.9),
                    "mean_abs_position": selected.mean_abs_position.mean(),
                    "net_pnl": selected.net_pnl.sum(), "win_rate": (selected.net_pnl > 0).mean(),
                })
        timestamp = x.groupby("date").apply(lambda g: pd.Series({
            "long_count": int((g.position > 0).sum()), "short_count": int((g.position < 0).sum()),
            "flat_count": int((g.position == 0).sum()), "long_gross": g.position.clip(lower=0).sum(),
            "short_gross": -g.position.clip(upper=0).sum(), "net_exposure": g.position.sum(),
            "gross_exposure": g.position.abs().sum(),
            "hhi": (g.position.abs().pow(2).sum() / g.position.abs().sum() ** 2) if g.position.abs().sum() else np.nan,
        }), include_groups=False).reset_index()
        timestamp["effective_assets"] = 1 / timestamp.hhi
        timestamp.insert(0, "strategy", strategy)
        exposures.append(timestamp)
        summaries.append({
            "strategy": strategy, "mean_long_count": timestamp.long_count.mean(),
            "mean_short_count": timestamp.short_count.mean(), "mean_flat_count": timestamp.flat_count.mean(),
            "mean_long_gross": timestamp.long_gross.mean(), "mean_short_gross": timestamp.short_gross.mean(),
            "mean_net_exposure": timestamp.net_exposure.mean(), "mean_gross_exposure": timestamp.gross_exposure.mean(),
            "mean_effective_assets": timestamp.effective_assets.mean(),
            "p10_effective_assets": timestamp.effective_assets.quantile(.1),
            "long_episode_hours": ep.loc[ep.direction == "long", "holding_hours"].mean(),
            "short_episode_hours": ep.loc[ep.direction == "short", "holding_hours"].mean(),
            "long_episode_share": (ep.direction == "long").mean(),
        })
    return pd.DataFrame(summaries), pd.DataFrame(contract_rows), pd.concat(episodes, ignore_index=True), pd.concat(exposures, ignore_index=True)


def _plot_html(fig, title: str, output: Path):
    import plotly.io as pio
    pio.write_html(fig, output, include_plotlyjs="directory", full_html=True,
                   config={"responsive": True, "displaylogo": False}, auto_open=False)


def portfolio_holdings_dashboard(metrics: pd.DataFrame, summary: pd.DataFrame, contract: pd.DataFrame,
                                 exposure: pd.DataFrame, output: Path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    colors = {"equal_6_wf": "#38bdf8", "ridge_v2": "#a78bfa", "lightgbm_v2": "#f59e0b",
              "ridge_adaptive": "#c4b5fd", "lightgbm_adaptive": "#fcd34d"}
    fig = make_subplots(rows=4, cols=2, specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}], [{}, {}]],
                        subplot_titles=("多空资产数量", "Gross / Net Exposure", "有效持仓资产数",
                                        "逐合约多空时间占比", "平均多空持有周期", "净收益与最大回撤", "逐合约多空 PnL"))
    daily = exposure.assign(day=exposure.date.dt.floor("D")).groupby(["strategy", "day"], as_index=False).mean(numeric_only=True)
    for name, g in daily.groupby("strategy"):
        fig.add_trace(go.Scatter(x=g.day, y=g.long_count, name=f"{name} long", line=dict(color=colors[name], width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=g.day, y=-g.short_count, name=f"{name} short", line=dict(color=colors[name], width=1.2, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=g.day, y=g.gross_exposure, name=f"{name} gross", showlegend=False, line=dict(color=colors[name])), row=2, col=1)
        fig.add_trace(go.Scatter(x=g.day, y=g.net_exposure, name=f"{name} net", showlegend=False, line=dict(color=colors[name], dash="dot")), row=2, col=1)
        fig.add_trace(go.Scatter(x=g.day, y=g.effective_assets, name=name, showlegend=False, line=dict(color=colors[name])), row=2, col=2)
    for direction, sign in [("long", 1), ("short", -1)]:
        c = contract.loc[contract.direction == direction]
        for name, g in c.groupby("strategy"):
            fig.add_trace(go.Bar(x=g.symbol, y=g.asset_hour_share * sign, name=f"{name} {direction}",
                                 marker_color=colors[name], opacity=.85 if direction == "long" else .45,
                                 showlegend=False), row=3, col=1)
            fig.add_trace(go.Bar(x=g.symbol, y=g.mean_holding_hours, name=f"{name} {direction}",
                                 marker_color=colors[name], opacity=.85 if direction == "long" else .45,
                                 showlegend=False), row=3, col=2)
            fig.add_trace(go.Bar(x=g.symbol, y=g.net_pnl * sign, name=f"{name} {direction}",
                                 marker_color=colors[name], opacity=.85 if direction == "long" else .45,
                                 showlegend=False), row=4, col=2)
    fig.add_trace(go.Bar(x=metrics.strategy, y=metrics.total_return, name="净收益", marker_color="#38bdf8", showlegend=False), row=4, col=1)
    fig.add_trace(go.Bar(x=metrics.strategy, y=metrics.max_drawdown, name="最大回撤", marker_color="#ef4444", showlegend=False), row=4, col=1)
    fig.update_layout(template="plotly_dark", height=1750, barmode="group", title="Walk-forward 持仓与多空结构审计",
                      legend=dict(orientation="h"), hovermode="x unified")
    fig.update_yaxes(title="资产数量（空头为负）", row=1, col=1)
    fig.update_yaxes(title="组合权重", row=2, col=1)
    fig.update_yaxes(title="1 / HHI", row=2, col=2)
    fig.update_yaxes(title="持仓时间占比", tickformat=".0%", row=3, col=1)
    fig.update_yaxes(title="小时", row=3, col=2)
    fig.update_yaxes(title="收益 / 回撤", tickformat=".0%", row=4, col=1)
    fig.update_yaxes(title="多空收益贡献", row=4, col=2)
    _plot_html(fig, "Holdings audit", output)


def symbol_dashboard(symbol: str, positions: dict[str, pd.DataFrame], hourly_detail: dict[str, pd.DataFrame], output: Path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    price = pd.read_parquet(Path("data/parquet") / symbol / "1h.parquet", columns=["date", "close"])
    start = min(v.date.min() for v in positions.values())
    price = price.loc[price.date >= start]
    price_daily = price.assign(day=price.date.dt.floor("D")).groupby("day", as_index=False).close.last()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=.04,
                        subplot_titles=(f"{symbol} 收盘价", "标准化信号", "实际持仓权重", "逐合约累计 PnL"))
    fig.add_trace(go.Scatter(x=price_daily.day, y=price_daily.close, name="close", line=dict(color="#94a3b8")), row=1, col=1)
    colors = {"equal_6_wf": "#38bdf8", "ridge_v2": "#a78bfa", "lightgbm_v2": "#f59e0b",
              "ridge_adaptive": "#c4b5fd", "lightgbm_adaptive": "#fcd34d"}
    for name, data in positions.items():
        asset = data.loc[data.symbol == symbol].sort_values("date")
        daily = asset.assign(day=asset.date.dt.floor("D")).groupby("day", as_index=False).agg(
            signal=("signal", "last"), position=("position", "last"))
        fig.add_trace(go.Scatter(x=daily.day, y=daily.signal, name=f"{name} signal", line=dict(color=colors[name], width=1.2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=daily.day, y=daily.position, name=f"{name} position", line=dict(color=colors[name], width=1.2),
                                 fill="tozeroy", opacity=.75), row=3, col=1)
        detail = hourly_detail[name]
        pnl = detail.loc[detail.symbol == symbol].set_index("date").net_pnl.cumsum()
        pnl = pnl.groupby(pnl.index.floor("D")).last()
        fig.add_trace(go.Scatter(x=pnl.index, y=pnl, name=f"{name} PnL", line=dict(color=colors[name], width=1.5)), row=4, col=1)
    fig.add_hline(y=0, line_width=1, line_color="#64748b", row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color="#64748b", row=3, col=1)
    fig.update_layout(template="plotly_dark", height=1300, title=f"{symbol} | 信号、仓位与收益", hovermode="x unified")
    fig.update_yaxes(title="价格", row=1, col=1); fig.update_yaxes(title="z / score", row=2, col=1)
    fig.update_yaxes(title="组合权重", tickformat=".2%", row=3, col=1); fig.update_yaxes(title="累计贡献", row=4, col=1)
    _plot_html(fig, symbol, output)


def run(train_months=6, step_months=1, fee_bps=4.0, start=None, settings: dict | None = None):
    settings = settings or {}
    V2_ROOT.mkdir(parents=True, exist_ok=True)
    charts = V2_ROOT / "contracts"
    charts.mkdir(exist_ok=True)
    panel = build_factor_panel(False)
    selected_hashes = settings.get("selected_hashes", SELECTED_HASHES)
    factors = [f"f_{factor_hash[:12]}" for factor_hash in selected_hashes]
    oriented = add_oriented_signals(panel, factors)
    learning, model_features = prepare_learning_panel(oriented, factors)
    predictions, diagnostics = walk_forward(learning, model_features, train_months, step_months, start, settings)
    oos_start = pd.Timestamp(diagnostics.fold_start.min())
    oos = learning.loc[learning.date >= oos_start].copy()
    sleeves = factor_sleeve_positions(oos, factors, len(SYMBOLS))
    equal = combine_sleeves(sleeves)
    equal["raw_signal"] = oos[factors].mean(axis=1)
    equal["signal"] = oos[factors].clip(-2, 2).mean(axis=1)
    equal["target"] = equal.position
    positions = {"equal_6_wf": equal}
    execution = settings.get("execution", {})
    positions.update({name: model_positions(
        oos, pred.loc[oos.index], len(SYMBOLS),
        z_window=int(execution.get("z_window", 336)),
        smooth_span=int(execution.get("smooth_span", 8)),
        neutral_zone=float(execution.get("neutral_zone", .20)),
        deadband=float(execution.get("position_deadband", .004)),
    ) for name, pred in predictions.items()})
    adaptive_days = int(execution.get("adaptive_lookback_days", 90))
    adaptive_cap = float(execution.get("adaptive_max_model_weight", .25))
    positions["ridge_adaptive"] = adaptive_prior_blend(
        equal, positions["ridge_v2"], fee_bps, adaptive_days * 24, adaptive_cap)
    positions["lightgbm_adaptive"] = adaptive_prior_blend(
        equal, positions["lightgbm_v2"], fee_bps, adaptive_days * 24, adaptive_cap)

    metrics_rows, hourly_rows, details = [], [], {}
    for name, pos in positions.items():
        metrics, hourly, _, _ = evaluate_portfolio(name, pos, fee_bps)
        metrics_rows.append(metrics); hourly.insert(0, "strategy", name); hourly_rows.append(hourly)
        details[name] = pnl_detail(pos, fee_bps)
    benchmark_hourly, benchmark_metric = buy_and_hold(learning, oos_start)
    metrics_rows.append(benchmark_metric); benchmark_hourly.insert(0, "strategy", "buy_and_hold"); hourly_rows.append(benchmark_hourly)
    metrics = pd.DataFrame(metrics_rows).sort_values("sharpe", ascending=False)
    hourly = pd.concat(hourly_rows, ignore_index=True)
    summary, contract, episodes, exposure = holding_audit(positions, fee_bps)

    # Per-fold realized returns.
    folds = []
    for fold_start in sorted(pd.to_datetime(diagnostics.fold_start.unique(), utc=True)):
        fold_end = fold_start + pd.DateOffset(months=step_months)
        for name, group in hourly.groupby("strategy"):
            sample = group.loc[group.date.ge(fold_start) & group.date.lt(fold_end)].set_index("date")
            if len(sample):
                m = metrics_from_returns(name, sample.net_pnl, sample.turnover)
                folds.append({"strategy": name, "fold_start": fold_start, "return": m["total_return"],
                              "sharpe": m["sharpe"], "max_drawdown": m["max_drawdown"]})
    fold_metrics = pd.DataFrame(folds)
    factor_table = factor_contributions(oos, sleeves, fee_bps)
    sessions = aggregate_decomposition(details["equal_6_wf"], "ny_session")
    monthly_returns = []
    for name, group in hourly.groupby("strategy"):
        series = group.set_index("date").net_pnl
        compounded = (1 + series).groupby(series.index.strftime("%Y-%m")).prod() - 1
        monthly_returns.extend({"strategy": name, "month": month, "return": value}
                               for month, value in compounded.items())
    monthly_returns = pd.DataFrame(monthly_returns)

    metrics.to_csv(V2_ROOT / "strategy_metrics.csv", index=False)
    diagnostics.to_csv(V2_ROOT / "fold_model_diagnostics.csv", index=False)
    fold_metrics.to_csv(V2_ROOT / "fold_strategy_metrics.csv", index=False)
    summary.to_csv(V2_ROOT / "portfolio_holding_summary.csv", index=False)
    contract.to_csv(V2_ROOT / "contract_long_short_summary.csv", index=False)
    episodes.to_parquet(V2_ROOT / "holding_episodes.parquet", index=False, compression="zstd")
    exposure.to_parquet(V2_ROOT / "portfolio_exposure.parquet", index=False, compression="zstd")
    factor_table.to_csv(V2_ROOT / "factor_contribution.csv", index=False)
    sessions.to_csv(V2_ROOT / "session_decomposition.csv", index=False)
    monthly_returns.to_csv(V2_ROOT / "monthly_returns.csv", index=False)
    hourly.to_parquet(V2_ROOT / "hourly_returns.parquet", index=False, compression="zstd")
    pd.concat([v.assign(strategy=k) for k, v in details.items()]).to_parquet(V2_ROOT / "position_pnl_detail.parquet", index=False, compression="zstd")
    portfolio_holdings_dashboard(metrics.loc[metrics.strategy != "buy_and_hold"], summary, contract, exposure,
                                 V2_ROOT / "holdings_dashboard.html")
    save_dashboard(metrics, hourly, fold_metrics, factor_table, sessions, monthly_returns,
                   V2_ROOT / "model_comparison_dashboard.html", train_months, step_months)
    for symbol in SYMBOLS:
        symbol_dashboard(symbol, positions, details, charts / f"{symbol.lower()}_signals_positions.html")
    manifest = {"created_utc": str(pd.Timestamp.now(tz="UTC")), "train_months": train_months,
                "step_months": step_months, "fee_bps": fee_bps, "folds": diagnostics.fold_start.nunique(),
                "oos_start": str(oos_start), "oos_end": str(oos.date.max()), "model_features": model_features,
                "selected_hashes": selected_hashes,
                "label": "risk-normalized compounded returns t+2 through t+5", "neutral_zone": .20,
                "signal_smoothing_span": 8, "position_deadband": .004,
                "adaptive_prior": "90d trailing alpha Sharpe, model weight 0..25%, shifted one hour"}
    (V2_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metrics.to_string(index=False)); print("\nHOLDINGS\n", summary.to_string(index=False))


def main():
    from research.settings import load_settings
    p = argparse.ArgumentParser()
    p.add_argument("--config"); p.add_argument("--train-months", type=int)
    p.add_argument("--step-months", type=int); p.add_argument("--fee-bps", type=float); p.add_argument("--start")
    a = p.parse_args(); settings = load_settings(a.config); wf = settings["walk_forward"]
    run(a.train_months or int(wf["train_months"]), a.step_months or int(wf["step_months"]),
        a.fee_bps if a.fee_bps is not None else float(wf["fee_bps"]), a.start, settings)


if __name__ == "__main__": main()
