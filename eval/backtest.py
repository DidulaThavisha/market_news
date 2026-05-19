"""Event-driven backtest for Stage 2+.

Trades each event row in the predictions parquet if confidence clears thresholds.

Entry rule (daily-bar approximation of the plan's intraday rule):
  - If publish_ts_utc is before 09:30 ET on a trading day: enter at that day's open.
  - Otherwise: enter at the next trading session's open.

Exit rule: open of (entry_day + holding_days) trading days. No intraday stops at
this stage — the plan's hybrid σ-stop/+3σ-target exit requires intraday data we
don't have. The 10-day cap dominates risk for the POC.

Sizing: equal-weight per signal (default 1% of starting equity per trade). No
Kelly until model calibration is validated.

Cost model: fixed cost_bps round-trip (default 15 bps = 5 bps commission +
slippage estimate + 5 bps spread + 5 bps adverse selection). Long-only and
short trades pay the same here; borrow cost ignored at this holding horizon.

Trade-gating thresholds (Stage 2 starting points, tune from validation):
  - |P(up) - P(down)| > edge_threshold  (default 0.20)
  - P(material)        > material_threshold (default 0.50)

Outputs:
  - trades.parquet: one row per executed trade
  - equity_curve.parquet: daily equity by exit date
  - dict of summary metrics

Backtest evaluates pure event response: no other positions, no cash management,
no leverage. Equity reflects realized PnL of trades attributed to their exit day.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from data.prices import fetch_daily

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestConfig:
    holding_days: int = 10
    edge_threshold: float = 0.20
    material_threshold: float = 0.50
    position_size: float = 0.01  # 1% of starting equity per trade
    cost_bps_round_trip: float = 15.0
    starting_equity: float = 1_000_000.0


def _decide(row: pd.Series, cfg: BacktestConfig) -> int:
    """+1 long, -1 short, 0 skip."""
    if not {"pred_p_up", "pred_p_down", "pred_p_material"}.issubset(row.index):
        return 0
    edge = float(row["pred_p_up"]) - float(row["pred_p_down"])
    if abs(edge) < cfg.edge_threshold:
        return 0
    if float(row["pred_p_material"]) < cfg.material_threshold:
        return 0
    return 1 if edge > 0 else -1


def _entry_session(publish_ts_utc: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    """First trading session whose 09:30 ET open is at or after publish_ts.

    For events publishing during/after RTH, this is the NEXT session. For
    pre-open events on a trading day, it's the same calendar day.
    """
    et = publish_ts_utc.tz_convert("US/Eastern") if publish_ts_utc.tzinfo else \
         publish_ts_utc.tz_localize("UTC").tz_convert("US/Eastern")
    cutoff = et.replace(hour=9, minute=30, second=0, microsecond=0)
    target_date = pd.Timestamp(et.date()) if et <= cutoff else pd.Timestamp(et.date()) + pd.Timedelta(days=1)
    idx = trading_days.searchsorted(target_date)
    if idx >= len(trading_days):
        return None
    return trading_days[idx]


def _fetch_all_prices(tickers: list[str], start: str, end: str) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        df = fetch_daily(t, start=start, end=end)
        if df.empty:
            continue
        out[t] = df
    return out


def run_backtest(
    predictions: pd.DataFrame,
    cfg: BacktestConfig = BacktestConfig(),
    price_buffer_days: int = 30,
) -> dict[str, Any]:
    """Returns dict with `trades` (DataFrame), `equity_curve` (Series), `summary` (dict)."""
    df = predictions.copy()
    df["publish_ts_utc"] = pd.to_datetime(df["publish_ts_utc"], utc=True)
    df = df.sort_values("publish_ts_utc").reset_index(drop=True)

    tickers = sorted(df["ticker"].unique().tolist())
    start = (df["publish_ts_utc"].min() - pd.Timedelta(days=price_buffer_days)).strftime("%Y-%m-%d")
    end = (df["publish_ts_utc"].max() + pd.Timedelta(days=price_buffer_days + cfg.holding_days * 2)).strftime("%Y-%m-%d")

    print(f"Fetching prices for {len(tickers)} tickers: {start} → {end}")
    prices = _fetch_all_prices(tickers, start, end)
    if not prices:
        raise RuntimeError("no prices fetched")

    cost_frac = cfg.cost_bps_round_trip / 10_000.0

    trades = []
    for _, row in df.iterrows():
        signal = _decide(row, cfg)
        if signal == 0:
            continue
        ticker = row["ticker"]
        px = prices.get(ticker)
        if px is None or px.empty:
            continue
        trading_days = px.index
        entry_day = _entry_session(row["publish_ts_utc"], trading_days)
        if entry_day is None or entry_day not in px.index:
            continue
        entry_loc = px.index.get_loc(entry_day)
        exit_loc = entry_loc + cfg.holding_days
        if exit_loc >= len(px):
            continue  # holding window runs past available data
        exit_day = px.index[exit_loc]

        entry_price = float(px.at[entry_day, "open"])
        exit_price = float(px.at[exit_day, "open"])
        if not (np.isfinite(entry_price) and np.isfinite(exit_price)) or entry_price <= 0:
            continue
        gross_ret = signal * (exit_price / entry_price - 1.0)
        net_ret = gross_ret - cost_frac
        pnl = cfg.starting_equity * cfg.position_size * net_ret

        trades.append({
            "ticker": ticker,
            "sector": row.get("sector"),
            "publish_ts_utc": row["publish_ts_utc"],
            "entry_day": entry_day,
            "exit_day": exit_day,
            "direction_signed": signal,
            "pred_p_up": float(row.get("pred_p_up", np.nan)),
            "pred_p_down": float(row.get("pred_p_down", np.nan)),
            "pred_p_material": float(row.get("pred_p_material", np.nan)),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_return": gross_ret,
            "net_return": net_ret,
            "pnl": pnl,
            "realized_car_0_1": float(row.get("car_0_1", np.nan)),
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {
            "trades": trades_df,
            "equity_curve": pd.Series(dtype=float),
            "summary": {"n_trades": 0, "note": "no trades fired — relax thresholds"},
        }

    # Daily PnL by exit_day, equity curve = starting + cumulative PnL.
    daily_pnl = trades_df.groupby("exit_day")["pnl"].sum().sort_index()
    full_idx = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="B")
    daily_pnl = daily_pnl.reindex(full_idx, fill_value=0.0)
    equity = cfg.starting_equity + daily_pnl.cumsum()

    # Returns of equity for Sharpe (using daily PnL / starting equity as crude return).
    daily_ret = daily_pnl / cfg.starting_equity
    active_days = (daily_ret != 0).sum()
    ann_factor = np.sqrt(TRADING_DAYS_PER_YEAR)

    summary = {
        "n_trades": int(len(trades_df)),
        "n_long": int((trades_df["direction_signed"] > 0).sum()),
        "n_short": int((trades_df["direction_signed"] < 0).sum()),
        "hit_rate": float((trades_df["net_return"] > 0).mean()),
        "avg_gross_return_per_trade": float(trades_df["gross_return"].mean()),
        "avg_net_return_per_trade": float(trades_df["net_return"].mean()),
        "total_pnl": float(trades_df["pnl"].sum()),
        "total_return": float(trades_df["pnl"].sum() / cfg.starting_equity),
        "daily_pnl_mean": float(daily_pnl.mean()),
        "daily_pnl_std": float(daily_pnl.std(ddof=1)) if len(daily_pnl) > 1 else 0.0,
        "sharpe_gross": _sharpe(trades_df["gross_return"].values, ann_factor, active_days, cfg),
        "sharpe_net": _sharpe(trades_df["net_return"].values, ann_factor, active_days, cfg),
        "max_drawdown": _max_drawdown(equity),
        "active_days": int(active_days),
        "n_unique_tickers_traded": int(trades_df["ticker"].nunique()),
        "config": cfg.__dict__,
    }
    return {"trades": trades_df, "equity_curve": equity, "summary": summary}


def _sharpe(returns: np.ndarray, ann_factor: float, active_days: int, cfg: BacktestConfig) -> float | None:
    """Sharpe of per-trade returns, then annualize via avg-trades-per-active-day."""
    if len(returns) < 2:
        return None
    mu = returns.mean()
    sd = returns.std(ddof=1)
    if sd == 0:
        return None
    trades_per_day = len(returns) / max(active_days, 1)
    # Daily PnL ≈ mu * trades_per_day, daily vol ≈ sd * sqrt(trades_per_day).
    return float((mu * trades_per_day) / (sd * np.sqrt(trades_per_day)) * ann_factor)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    return float(dd.min())
