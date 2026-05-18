"""Daily OHLCV via yfinance, normalized to a consistent schema."""
from __future__ import annotations

import pandas as pd
import yfinance as yf


def fetch_daily(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Adjusted daily OHLCV. Index = trading date (tz-naive, UTC midnight equivalent)."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        actions=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower).rename(columns={"adj close": "close"})
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df["ret"] = df["close"].pct_change()
    return df
