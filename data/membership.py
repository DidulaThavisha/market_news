"""Ticker → GICS sector. Stage-1 hardcoded; Stage-2+ should pull point-in-time
S&P 500 membership from Wikipedia revisions / iShares IVV holdings."""
from __future__ import annotations

# Stage 1 pilot: sector-diverse, liquid large caps. Set chosen in the project plan.
STAGE1_TICKERS = ["JPM", "XOM", "WMT", "MRK", "BA"]

TICKER_SECTOR: dict[str, str] = {
    "JPM": "Financials",
    "XOM": "Energy",
    "WMT": "Consumer Staples",
    "MRK": "Health Care",
    "BA":  "Industrials",
}


def sector_for(ticker: str) -> str:
    return TICKER_SECTOR.get(ticker.upper(), "Unknown")
