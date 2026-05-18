"""FF4 abnormal returns, CAR windows, idio-vol-normalized direction & materiality labels.

The pipeline:
  1. For each event, locate the "event trading day" t0 = first trading session whose
     09:30 ET open is strictly after the publish timestamp (no look-ahead).
  2. Estimate FF4 factor loadings on rolling window [t0-250d, t0-30d) (skip-30 buffer).
  3. Compute AR_t = (ret_t - RF_t) - (alpha + beta·factors_t) for the event window.
  4. CAR_window = sum of AR over the window's trading days.
  5. Idio-vol sigma_1d = stdev of AR over [t0-60d, t0-30d); window-scaled by sqrt(N).
  6. Direction = down/neutral/up at +/- 0.75*sigma_window; materiality = |CAR| > 1.5*sigma_window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

FACTOR_COLS = ["MKT_RF", "SMB", "HML", "MOM"]
EST_WINDOW = (-250, -30)  # trading days relative to t0; right-exclusive
IDIO_WINDOW = (-60, -30)
DIR_THRESHOLD = 0.75
MATERIAL_THRESHOLD = 1.5


@dataclass(frozen=True)
class EventLabel:
    publish_ts_utc: pd.Timestamp
    t0: pd.Timestamp
    car_window: tuple[int, int]
    car: float
    sigma_window: float
    car_z: float
    direction: str
    material: bool
    factor_r2: float


def event_trading_day(
    publish_ts_utc: pd.Timestamp, trading_days: pd.DatetimeIndex
) -> pd.Timestamp | pd.NaT:
    """First trading day whose 09:30 ET open is strictly after publish_ts_utc."""
    if pd.isna(publish_ts_utc):
        return pd.NaT
    pub_et = publish_ts_utc.tz_convert("US/Eastern")
    pub_date = pd.Timestamp(pub_et.date())
    cutoff = pd.Timestamp("09:30").time()
    if pub_date in trading_days and pub_et.time() < cutoff:
        return pub_date
    future = trading_days[trading_days > pub_date]
    return future[0] if len(future) else pd.NaT


def _window_slice(
    days: pd.DatetimeIndex, anchor: pd.Timestamp, lo: int, hi: int
) -> pd.DatetimeIndex:
    """Return trading days in [anchor + lo, anchor + hi) where lo/hi are trading-day offsets."""
    if anchor not in days:
        return pd.DatetimeIndex([])
    idx = days.get_loc(anchor)
    a = max(0, idx + lo)
    b = min(len(days), idx + hi)
    return days[a:b]


def _factor_window_for_event(
    t0: pd.Timestamp, factors: pd.DataFrame, lo: int, hi: int
) -> pd.DataFrame:
    days = factors.index
    window_days = _window_slice(days, t0, lo, hi)
    return factors.loc[window_days]


def _fit_ff4(stock_returns: pd.Series, factors: pd.DataFrame) -> tuple[pd.Series, float, float]:
    """OLS: (ret - RF) ~ const + factors. Return (params, residual_sigma, R2)."""
    df = factors.join(stock_returns.rename("ret"), how="inner").dropna()
    if len(df) < 30:
        raise ValueError(f"insufficient observations for FF4 fit: {len(df)}")
    y = df["ret"] - df["RF"]
    X = sm.add_constant(df[FACTOR_COLS])
    res = sm.OLS(y, X).fit()
    return res.params, float(np.sqrt(res.mse_resid)), float(res.rsquared)


def _predict_excess(params: pd.Series, factors_row: pd.Series) -> float:
    return float(
        params["const"]
        + sum(params[c] * factors_row[c] for c in FACTOR_COLS)
    )


def label_event(
    publish_ts_utc: pd.Timestamp,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    car_window: tuple[int, int] = (0, 1),
) -> EventLabel | None:
    """Compute one event's label. car_window is inclusive trading-day offsets from t0."""
    trading_days = prices.index
    t0 = event_trading_day(publish_ts_utc, trading_days)
    if pd.isna(t0):
        return None

    est_factors = _factor_window_for_event(t0, factors, *EST_WINDOW)
    est_prices = prices.loc[est_factors.index.intersection(prices.index), "ret"]
    if len(est_prices) < 60:
        return None
    try:
        params, _, r2 = _fit_ff4(est_prices, est_factors)
    except ValueError:
        return None

    idio_factors = _factor_window_for_event(t0, factors, *IDIO_WINDOW)
    idio_prices = prices.loc[idio_factors.index.intersection(prices.index), "ret"]
    idio_joined = idio_factors.join(idio_prices.rename("ret"), how="inner").dropna()
    if len(idio_joined) < 10:
        return None
    pred = idio_joined.apply(lambda r: _predict_excess(params, r), axis=1)
    actual = idio_joined["ret"] - idio_joined["RF"]
    sigma_1d = float((actual - pred).std(ddof=1))

    lo, hi = car_window
    event_days = _window_slice(trading_days, t0, lo, hi + 1)
    if len(event_days) < (hi - lo + 1):
        return None
    ev_factors = factors.loc[event_days]
    ev_prices = prices.loc[event_days, "ret"]
    ev_joined = ev_factors.join(ev_prices.rename("ret"), how="inner").dropna()
    if ev_joined.empty:
        return None
    ev_pred = ev_joined.apply(lambda r: _predict_excess(params, r), axis=1)
    ev_actual = ev_joined["ret"] - ev_joined["RF"]
    ar = ev_actual - ev_pred
    car = float(ar.sum())

    n = len(event_days)
    sigma_window = sigma_1d * np.sqrt(n)
    car_z = car / sigma_window if sigma_window > 0 else 0.0

    if car_z < -DIR_THRESHOLD:
        direction = "down"
    elif car_z > DIR_THRESHOLD:
        direction = "up"
    else:
        direction = "neutral"

    return EventLabel(
        publish_ts_utc=publish_ts_utc,
        t0=t0,
        car_window=car_window,
        car=car,
        sigma_window=sigma_window,
        car_z=car_z,
        direction=direction,
        material=abs(car_z) > MATERIAL_THRESHOLD,
        factor_r2=r2,
    )


def label_events(
    publish_timestamps: pd.Series,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    car_window: tuple[int, int] = (0, 1),
) -> pd.DataFrame:
    rows = []
    for ts in publish_timestamps:
        lbl = label_event(ts, prices, factors, car_window=car_window)
        if lbl is None:
            rows.append({"publish_ts_utc": ts, "t0": pd.NaT, "car": np.nan,
                         "car_z": np.nan, "direction": None, "material": None,
                         "sigma_window": np.nan, "factor_r2": np.nan,
                         "car_window_lo": car_window[0], "car_window_hi": car_window[1]})
            continue
        rows.append({
            "publish_ts_utc": lbl.publish_ts_utc,
            "t0": lbl.t0,
            "car": lbl.car,
            "car_z": lbl.car_z,
            "direction": lbl.direction,
            "material": lbl.material,
            "sigma_window": lbl.sigma_window,
            "factor_r2": lbl.factor_r2,
            "car_window_lo": car_window[0],
            "car_window_hi": car_window[1],
        })
    return pd.DataFrame(rows)
