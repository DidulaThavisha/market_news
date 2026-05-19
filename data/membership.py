"""Ticker → GICS sector. Stage-1/2 hardcoded; Stage-3+ should pull point-in-time
S&P 500 membership from Wikipedia revisions / iShares IVV holdings.

Stage 2's 25 names are large, liquid, in S&P 500 continuously since 2014, and
sector-diverse across all 11 GICS sectors. Top-7 mega caps (AAPL, MSFT, GOOGL,
GOOG, AMZN, NVDA, META, TSLA) are deliberately excluded per the project plan.
"""
from __future__ import annotations

STAGE1_TICKERS = ["JPM", "XOM", "WMT", "MRK", "BA"]

STAGE2_TICKERS = [
    # Communication Services
    "DIS", "VZ",
    # Consumer Discretionary
    "HD", "MCD", "NKE",
    # Consumer Staples
    "WMT", "KO", "PG",
    # Energy
    "XOM", "CVX",
    # Financials
    "JPM", "BAC", "GS",
    # Health Care
    "MRK", "JNJ", "PFE",
    # Industrials
    "BA", "CAT", "HON",
    # Information Technology
    "ORCL", "IBM", "CSCO",
    # Materials
    "LIN",
    # Real Estate
    "AMT",
    # Utilities
    "NEE",
]

TICKER_SECTOR: dict[str, str] = {
    "DIS": "Communication Services",
    "VZ":  "Communication Services",
    "HD":  "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "WMT": "Consumer Staples",
    "KO":  "Consumer Staples",
    "PG":  "Consumer Staples",
    "XOM": "Energy",
    "CVX": "Energy",
    "JPM": "Financials",
    "BAC": "Financials",
    "GS":  "Financials",
    "MRK": "Health Care",
    "JNJ": "Health Care",
    "PFE": "Health Care",
    "BA":  "Industrials",
    "CAT": "Industrials",
    "HON": "Industrials",
    "ORCL":"Information Technology",
    "IBM": "Information Technology",
    "CSCO":"Information Technology",
    "LIN": "Materials",
    "AMT": "Real Estate",
    "NEE": "Utilities",
}


def sector_for(ticker: str) -> str:
    return TICKER_SECTOR.get(ticker.upper(), "Unknown")
