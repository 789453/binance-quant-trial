"""Monthly walk-forward evaluation for the frozen six-factor strategy.

Default protocol: trailing six calendar months of training, four-hour label
purge, and one calendar month of prediction.  Equal-weight has no fitted
parameters but is evaluated over the identical out-of-sample folds.
"""
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from research.factor_combo_pipeline import (
    LGBMRegressor,
    OUT_ROOT,
    SELECTED_HASHES,
    SYMBOLS,
    add_oriented_signals,
    build_factor_panel,
    evaluate_portfolio,
    factor_columns,
    prediction_to_positions,
    reference_table,
)


WF_ROOT = OUT_ROOT / "walk_forward"
PURGE_HOURS = 4


def month_starts(start: pd.Timestamp, end: pd.Timestamp, step_months: int) -> list[pd.Timestamp]:
    start = pd.Timestamp(start).tz_convert("UTC") if pd.Timestamp(start).tzinfo else pd.Timestamp(start, tz="UTC")
    end = pd.Timestamp(end).tz_convert("UTC") if pd.Timestamp(end).tzinfo else pd.Timestamp(end, tz="UTC")
    cursor = pd.Timestamp(year=start.year, month=start.month, day=1, tz="UTC")
    values = []
    while cursor <= end:
        values.append(cursor)
        cursor += pd.DateOffset(months=step_months)
    return values


def make_model(name: str, features: list[str]):
    numeric = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    pre = ColumnTransformer([
        ("num", numeric, features),
        ("symbol", OneHotEncoder(handle_unknown="ignore"), ["symbol"]),
    ])
    if name == "ridge":
        model = Ridge(alpha=20.0)
    elif name == "lightgbm":
        if LGBMRegressor is None:
            raise RuntimeError("lightgbm is not installed")
        model = LGBMRegressor(
            n_estimators=250, learning_rate=0.03, num_leaves=15, max_depth=5,
            min_child_samples=250, subsample=0.8, colsample_bytree=0.85,
            reg_alpha=0.2, reg_lambda=2.0, random_state=42,
            verbosity=-1, n_jobs=-1,
        )
    else:
        raise ValueError(name)
    return make_pipeline(pre, model)


def walk_forward_predictions(panel: pd.DataFrame, features: list[str], train_months: int,
                             step_months: int, start: str | None = None) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    train_data = panel.dropna(subset=features + ["target_4h"]).copy()
    predict_data = panel.dropna(subset=features).copy()
    first_complete = train_data.date.min()
    inferred_point = first_complete + pd.DateOffset(months=train_months)
    inferred_start = pd.Timestamp(year=inferred_point.year, month=inferred_point.month, day=1, tz="UTC")
    prediction_start = pd.Timestamp(start, tz="UTC") if start else inferred_start
    folds = month_starts(prediction_start, panel.date.max(), step_months)
    predictions = {
        "ridge_wf": pd.Series(np.nan, index=panel.index, dtype=float),
        "lightgbm_wf": pd.Series(np.nan, index=panel.index, dtype=float),
    }
    diagnostics: list[dict] = []
    for fold_start in folds:
        fold_end = min(fold_start + pd.DateOffset(months=step_months), panel.date.max() + pd.Timedelta(hours=1))
        train_start = fold_start - pd.DateOffset(months=train_months)
        train_end = fold_start - pd.Timedelta(hours=PURGE_HOURS)
        train_mask = train_data.date.ge(train_start) & train_data.date.lt(train_end)
        test_mask = predict_data.date.ge(fold_start) & predict_data.date.lt(fold_end)
        if train_mask.sum() < 5_000 or test_mask.sum() == 0:
            continue
        X_train = train_data.loc[train_mask, features + ["symbol"]]
        y_train = train_data.loc[train_mask, "target_4h"].clip(-0.20, 0.20)
        X_test = predict_data.loc[test_mask, features + ["symbol"]]
        for name in ["ridge", "lightgbm"]:
            started = time.time()
            model = make_model(name, features)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            destination = f"{name}_wf"
            predictions[destination].loc[predict_data.index[test_mask]] = pred
            known = predict_data.loc[test_mask, "target_4h"].notna().to_numpy()
            truth = predict_data.loc[test_mask, "target_4h"].clip(-0.20, 0.20).to_numpy()[known]
            scored_pred = pred[known]
            diagnostics.append({
                "model": destination, "fold_start": fold_start, "fold_end": fold_end,
                "train_start": train_start, "train_end_exclusive": train_end,
                "train_rows": int(train_mask.sum()), "test_rows": int(test_mask.sum()),
                "rmse": float(np.sqrt(np.mean((truth - scored_pred) ** 2))),
                "prediction_ic": float(np.corrcoef(truth, scored_pred)[0, 1]),
                "fit_seconds": time.time() - started,
            })
    return predictions, pd.DataFrame(diagnostics)


def factor_sleeve_positions(panel: pd.DataFrame, features: list[str], n_assets: int) -> dict[str, pd.DataFrame]:
    sleeves = {}
    base = panel[["date", "symbol", "asset_return"]]
    for col in features:
        x = base.copy()
        x["signal"] = panel[col].clip(-2, 2) / 2
        x["target"] = x.signal / n_assets
        x["position"] = x.groupby("symbol", sort=False).target.transform(
            lambda s: s.rolling(4, min_periods=3).mean().shift(2)
        ).fillna(0.0)
        sleeves[col] = x
    return sleeves


def combine_sleeves(sleeves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    first = next(iter(sleeves.values()))[["date", "symbol", "asset_return"]].copy()
    first["position"] = np.mean([x.position.to_numpy() for x in sleeves.values()], axis=0)
    return first


def pnl_detail(positions: pd.DataFrame, fee_bps: float) -> pd.DataFrame:
    x = positions.sort_values(["symbol", "date"]).copy()
    x["prev_position"] = x.groupby("symbol", sort=False).position.shift(1).fillna(0)
    x["turnover"] = (x.position - x.prev_position).abs()
    x["gross_pnl"] = x.position * x.asset_return.fillna(0)
    x["cost"] = x.turnover * fee_bps / 10_000
    x["net_pnl"] = x.gross_pnl - x.cost
    ny = x.date.dt.tz_convert("America/New_York")
    x["ny_hour"] = ny.dt.hour
    x["ny_session"] = pd.cut(x.ny_hour, [-1, 7, 15, 23], labels=["overnight", "day", "evening"])
    x["direction"] = np.where(x.position > 0, "long", np.where(x.position < 0, "short", "flat"))
    x["month"] = x.date.dt.strftime("%Y-%m")
    return x


def metrics_from_returns(name: str, returns: pd.Series, turnover: pd.Series | None = None) -> dict:
    returns = returns.fillna(0).sort_index()
    equity = (1 + returns).cumprod()
    dd = equity / equity.cummax() - 1
    years = max((returns.index.max() - returns.index.min()).total_seconds() / (365.25 * 86400), 1 / 365.25)
    ann = 365.25 * 24
    return {
        "strategy": name, "start": returns.index.min(), "end": returns.index.max(),
        "total_return": equity.iloc[-1] - 1, "cagr": equity.iloc[-1] ** (1 / years) - 1,
        "sharpe": math.sqrt(ann) * returns.mean() / returns.std(ddof=0) if returns.std(ddof=0) else np.nan,
        "max_drawdown": dd.min(), "annualized_volatility": returns.std(ddof=0) * math.sqrt(ann),
        "positive_hour_ratio": (returns > 0).mean(),
        "mean_hourly_turnover": turnover.mean() if turnover is not None else 0.0,
        "annualized_turnover": turnover.mean() * ann if turnover is not None else 0.0,
    }


def buy_and_hold(panel: pd.DataFrame, start: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    returns = panel.loc[panel.date >= start].pivot(index="date", columns="symbol", values="asset_return").sort_index()
    relatives = (1 + returns.fillna(0)).cumprod()
    equity = relatives.mean(axis=1)
    portfolio_returns = equity.pct_change().fillna(0)
    hourly = pd.DataFrame({"date": portfolio_returns.index, "net_pnl": portfolio_returns.values,
                           "gross_pnl": portfolio_returns.values, "turnover": 0.0})
    hourly["equity"] = (1 + hourly.net_pnl).cumprod()
    hourly["drawdown"] = hourly.equity / hourly.equity.cummax() - 1
    return hourly, metrics_from_returns("buy_and_hold", portfolio_returns)


def aggregate_decomposition(detail: pd.DataFrame, key: str) -> pd.DataFrame:
    return detail.groupby(key, observed=True).agg(
        net_contribution=("net_pnl", "sum"), gross_contribution=("gross_pnl", "sum"),
        cost=("cost", "sum"), turnover=("turnover", "sum"),
        positive_ratio=("net_pnl", lambda s: float((s > 0).mean())),
        observations=("net_pnl", "size"),
    ).reset_index()


def factor_contributions(panel: pd.DataFrame, sleeves: dict[str, pd.DataFrame], fee_bps: float) -> pd.DataFrame:
    registry = reference_table().set_index("expr_hash")
    full = combine_sleeves(sleeves)
    full_metrics = evaluate_portfolio("full", full, fee_bps)[0]
    rows = []
    for col, position in sleeves.items():
        factor_hash = next(h for h in registry.index if h.startswith(col[2:]))
        standalone = evaluate_portfolio(col, position, fee_bps)[0]
        reduced = combine_sleeves({k: v for k, v in sleeves.items() if k != col})
        without = evaluate_portfolio(f"without_{col}", reduced, fee_bps)[0]
        rows.append({
            "expr_hash": factor_hash, "factor": col, "expression": registry.loc[factor_hash, "expr"],
            "standalone_return": standalone["total_return"], "standalone_sharpe": standalone["sharpe"],
            "standalone_max_drawdown": standalone["max_drawdown"],
            "marginal_return": full_metrics["total_return"] - without["total_return"],
            "marginal_sharpe": full_metrics["sharpe"] - without["sharpe"],
        })
    return pd.DataFrame(rows).sort_values("marginal_sharpe", ascending=False)


def save_dashboard(metrics: pd.DataFrame, hourly: pd.DataFrame, fold_metrics: pd.DataFrame,
                   factors: pd.DataFrame, sessions: pd.DataFrame, monthly: pd.DataFrame,
                   output: Path, train_months: int, step_months: int) -> None:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    colors = {"equal_6_wf": "#38bdf8", "ridge_wf": "#a78bfa", "lightgbm_wf": "#f59e0b",
              "ridge_v2": "#7c3aed", "lightgbm_v2": "#ea580c", "ridge_adaptive": "#c4b5fd",
              "lightgbm_adaptive": "#fcd34d", "buy_and_hold": "#94a3b8"}
    fig = make_subplots(rows=4, cols=2, vertical_spacing=0.075, horizontal_spacing=0.08,
                        specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}], [{}, {}]],
                        subplot_titles=("净值曲线", "回撤", "63 天滚动 Sharpe", "月度收益热力图",
                                        "因子贡献", "纽约时段净收益贡献", "Walk-forward 月度 Sharpe"))
    for name, group in hourly.groupby("strategy", sort=False):
        group = group.sort_values("date")
        daily = group.assign(day=group.date.dt.floor("D")).groupby("day", as_index=False).agg(
            equity=("equity", "last"), drawdown=("drawdown", "last"),
            net_pnl=("net_pnl", lambda s: (1 + s).prod() - 1))
        fig.add_trace(go.Scatter(x=daily.day, y=daily.equity, name=name, mode="lines",
                                 line=dict(color=colors.get(name), width=2 if name == "equal_6_wf" else 1.4)), row=1, col=1)
        fig.add_trace(go.Scatter(x=daily.day, y=daily.drawdown, name=name, mode="lines",
                                 showlegend=False, line=dict(color=colors.get(name), width=1.2)), row=2, col=1)
        values = daily.set_index("day").net_pnl
        rolling = values.rolling(63, min_periods=30)
        rs = rolling.mean() / rolling.std(ddof=0) * math.sqrt(365.25)
        fig.add_trace(go.Scatter(x=rs.index, y=rs, name=name, mode="lines", showlegend=False,
                                 line=dict(color=colors.get(name), width=1.2)), row=2, col=2)

    heat = monthly.pivot(index="strategy", columns="month", values="return").reindex(metrics.strategy)
    fig.add_trace(go.Heatmap(z=heat.values, x=heat.columns, y=heat.index,
                             colorscale="RdBu", zmid=0, colorbar=dict(title="月收益", len=.2, y=.42),
                             hovertemplate="%{y}<br>%{x}<br>%{z:.2%}<extra></extra>"), row=3, col=1)
    fig.add_trace(go.Bar(x=factors.factor.str.replace("f_", ""), y=factors.standalone_return,
                         name="单因子收益", marker_color="#38bdf8", showlegend=False,
                         customdata=factors.marginal_sharpe,
                         hovertemplate="%{x}<br>单因子收益 %{y:.2%}<br>边际 Sharpe %{customdata:.3f}<extra></extra>"), row=3, col=2)
    fig.add_trace(go.Bar(x=sessions.ny_session, y=sessions.net_contribution, name="时段贡献",
                         marker_color="#a78bfa", showlegend=False,
                         hovertemplate="%{x}<br>净贡献 %{y:.4f}<extra></extra>"), row=4, col=1)
    fold_plot = fold_metrics.copy()
    for name, group in fold_plot.groupby("strategy"):
        fig.add_trace(go.Bar(x=group.fold_start, y=group.sharpe, name=name, showlegend=False,
                             marker_color=colors.get(name), opacity=.82,
                             hovertemplate=f"{name}<br>%{{x|%Y-%m}}<br>Sharpe %{{y:.2f}}<extra></extra>"), row=4, col=2)
    fig.update_layout(height=1700, template="plotly_dark", barmode="group",
                      title=f"6 因子组合 Walk-forward | 训练 {train_months} 个月 · 步进 {step_months} 个月 · 4 bps",
                      legend=dict(orientation="h", y=1.02, x=0), margin=dict(l=65, r=35, t=110, b=50),
                      hovermode="x unified")
    fig.update_yaxes(title_text="组合净值", row=1, col=1)
    fig.update_yaxes(title_text="回撤", tickformat=".0%", row=2, col=1)
    fig.update_yaxes(title_text="年化 Sharpe", row=2, col=2)
    fig.update_yaxes(title_text="净收益", tickformat=".0%", row=3, col=2)
    fig.update_yaxes(title_text="收益贡献", row=4, col=1)
    fig.update_yaxes(title_text="月度 Sharpe", row=4, col=2)

    metrics_display = metrics.copy()
    for c in ["total_return", "cagr", "max_drawdown", "annualized_volatility", "positive_hour_ratio"]:
        metrics_display[c] = metrics_display[c].map(lambda x: f"{x:.2%}")
    table = metrics_display[["strategy", "total_return", "cagr", "sharpe", "max_drawdown",
                             "annualized_volatility", "annualized_turnover", "positive_hour_ratio"]]
    table.columns = ["策略", "总收益", "CAGR", "Sharpe", "最大回撤", "年化波动", "年化换手", "小时胜率"]
    table_html = table.to_html(index=False, classes="metrics", border=0, float_format=lambda x: f"{x:.3f}")
    chart = pio.to_html(fig, include_plotlyjs=True, full_html=False, config={"responsive": True, "displaylogo": False})
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Factor Combo Walk-forward Dashboard</title>
<style>body{{margin:0;background:#0b1220;color:#e5e7eb;font-family:Inter,"Microsoft YaHei",sans-serif}}main{{max-width:1500px;margin:auto;padding:24px}}h1{{font-size:24px;font-weight:500}}p{{color:#94a3b8}}.panel{{background:#111827;border:1px solid #243044;border-radius:12px;padding:18px;margin:16px 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;text-align:right;border-bottom:1px solid #243044;font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}th{{color:#94a3b8;font-weight:500}}@media(max-width:700px){{main{{padding:10px}}.panel{{padding:8px;overflow-x:auto}}}}</style></head><body><main><h1>6 因子组合 Walk-forward 回测</h1><p>滚动训练 {train_months} 个月，每 {step_months} 个月重训；4h 标签 purge；等权、Ridge、LightGBM 与等权买入持有基准。所有策略曲线均为逐月样本外拼接。</p><section class="panel">{table_html}</section><section class="panel">{chart}</section></main></body></html>"""
    output.write_text(html, encoding="utf-8")


def run(train_months: int = 6, step_months: int = 1, fee_bps: float = 4.0,
        start: str | None = None, force_factors: bool = False) -> None:
    WF_ROOT.mkdir(parents=True, exist_ok=True)
    panel = build_factor_panel(force=force_factors)
    features = factor_columns(panel)
    oriented = add_oriented_signals(panel, features)
    predictions, diagnostics = walk_forward_predictions(oriented, features, train_months, step_months, start)
    valid_starts = diagnostics.fold_start.dropna()
    if valid_starts.empty:
        raise RuntimeError("No valid walk-forward folds")
    oos_start = pd.Timestamp(valid_starts.min())
    oos = oriented.loc[oriented.date >= oos_start].copy()
    sleeves = factor_sleeve_positions(oos, features, len(SYMBOLS))
    equal_positions = combine_sleeves(sleeves)

    positions = {"equal_6_wf": equal_positions}
    for name, pred in predictions.items():
        positions[name] = prediction_to_positions(oos, pred.loc[oos.index], len(SYMBOLS), True)

    metrics_rows, hourly_rows, detail_by_strategy = [], [], {}
    for name, pos in positions.items():
        metrics, hourly, _, _ = evaluate_portfolio(name, pos, fee_bps)
        metrics_rows.append(metrics)
        hourly.insert(0, "strategy", name)
        hourly_rows.append(hourly)
        detail_by_strategy[name] = pnl_detail(pos, fee_bps)
    benchmark_hourly, benchmark_metrics = buy_and_hold(oriented, oos_start)
    metrics_rows.append(benchmark_metrics)
    benchmark_hourly.insert(0, "strategy", "buy_and_hold")
    hourly_rows.append(benchmark_hourly)
    metrics = pd.DataFrame(metrics_rows).sort_values("sharpe", ascending=False)
    hourly = pd.concat(hourly_rows, ignore_index=True)

    # Fold-level realized strategy metrics, distinct from prediction diagnostics.
    fold_rows = []
    all_fold_starts = sorted(pd.to_datetime(diagnostics.fold_start.unique(), utc=True))
    for fold_start in all_fold_starts:
        fold_end = fold_start + pd.DateOffset(months=step_months)
        for name, group in hourly.groupby("strategy"):
            sample = group.loc[(group.date >= fold_start) & (group.date < fold_end)].set_index("date")
            if len(sample):
                fm = metrics_from_returns(name, sample.net_pnl, sample.turnover)
                fold_rows.append({"strategy": name, "fold_start": fold_start, "fold_end": fold_end,
                                  "return": fm["total_return"], "sharpe": fm["sharpe"],
                                  "max_drawdown": fm["max_drawdown"], "turnover": fm["annualized_turnover"]})
    fold_metrics = pd.DataFrame(fold_rows)

    equal_detail = detail_by_strategy["equal_6_wf"]
    sessions = aggregate_decomposition(equal_detail, "ny_session")
    assets = aggregate_decomposition(equal_detail, "symbol")
    directions = aggregate_decomposition(equal_detail, "direction")
    months = aggregate_decomposition(equal_detail, "month")
    monthly_returns = []
    for name, group in hourly.groupby("strategy"):
        series = group.set_index("date").net_pnl
        compounded = (1 + series).groupby(series.index.strftime("%Y-%m")).prod() - 1
        monthly_returns.extend({"strategy": name, "month": month, "return": value}
                               for month, value in compounded.items())
    monthly_returns = pd.DataFrame(monthly_returns)
    factor_table = factor_contributions(oos, sleeves, fee_bps)

    metrics.to_csv(WF_ROOT / "strategy_metrics.csv", index=False)
    diagnostics.to_csv(WF_ROOT / "fold_model_diagnostics.csv", index=False)
    fold_metrics.to_csv(WF_ROOT / "fold_strategy_metrics.csv", index=False)
    factor_table.to_csv(WF_ROOT / "factor_contribution.csv", index=False)
    sessions.to_csv(WF_ROOT / "session_decomposition.csv", index=False)
    assets.to_csv(WF_ROOT / "asset_decomposition.csv", index=False)
    directions.to_csv(WF_ROOT / "direction_decomposition.csv", index=False)
    months.to_csv(WF_ROOT / "month_decomposition.csv", index=False)
    monthly_returns.to_csv(WF_ROOT / "monthly_returns.csv", index=False)
    hourly.to_parquet(WF_ROOT / "hourly_returns.parquet", index=False, compression="zstd")
    pd.concat([v.assign(strategy=k) for k, v in detail_by_strategy.items()]).to_parquet(
        WF_ROOT / "position_pnl_detail.parquet", index=False, compression="zstd")
    manifest = {
        "created_utc": str(pd.Timestamp.now(tz="UTC")), "train_months": train_months,
        "step_months": step_months, "purge_hours": PURGE_HOURS, "fee_bps": fee_bps,
        "oos_start": str(oos_start), "oos_end": str(oos.date.max()), "folds": len(all_fold_starts),
        "selected_hashes": SELECTED_HASHES, "benchmark": "equal-capital passive buy-and-hold of 12 assets",
    }
    (WF_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    save_dashboard(metrics, hourly, fold_metrics, factor_table, sessions, monthly_returns,
                   OUT_ROOT / "walk_forward_dashboard.html", train_months, step_months)
    print(metrics.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-months", type=int, default=6)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--fee-bps", type=float, default=4.0)
    parser.add_argument("--start", help="optional first prediction month, YYYY-MM-DD")
    parser.add_argument("--force-factors", action="store_true")
    args = parser.parse_args()
    if args.train_months < 1 or args.step_months < 1:
        parser.error("train-months and step-months must be positive")
    run(args.train_months, args.step_months, args.fee_bps, args.start, args.force_factors)


if __name__ == "__main__":
    main()
