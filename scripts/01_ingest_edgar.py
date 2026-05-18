"""Stage-0 ingest: pull 8-K filings + body text for one ticker × year, write parquet."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm

from data.edgar_ingest import (
    cik_for_ticker,
    fetch_filing_with_exhibits,
    items_mentioned,
    list_filings,
)

CACHE = Path(__file__).resolve().parents[1] / "cache"


def ingest(ticker: str, start: str, end: str) -> Path:
    cik = cik_for_ticker(ticker)
    all_8k = list_filings(cik, form="8-K")
    mask = (all_8k["filingDate"] >= start) & (all_8k["filingDate"] < end)
    df = all_8k.loc[mask].reset_index(drop=True).copy()
    if df.empty:
        raise SystemExit(f"no 8-Ks for {ticker} in [{start}, {end})")

    bodies, body_items, exhibits_per_filing = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{ticker} bodies"):
        text, exhibits = fetch_filing_with_exhibits(
            cik, row["accessionNumber"], row["primaryDocument"]
        )
        bodies.append(text)
        body_items.append(items_mentioned(text))
        exhibits_per_filing.append(exhibits)

    df["body_text"] = bodies
    df["items_in_body"] = body_items
    df["exhibits"] = exhibits_per_filing
    df["ticker"] = ticker
    df["cik"] = cik

    out = CACHE / f"edgar_8k_{ticker}_{start[:4]}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nwrote {out} ({len(df)} filings)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="JPM")
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--end", default="2020-01-01")
    args = ap.parse_args()
    ingest(args.ticker, args.start, args.end)


if __name__ == "__main__":
    main()
