"""Stage-2 batch: 25 sector-diverse tickers × 2014-2024.

Train period: 2014-2022.  Val: 2023.  Test: 2024.

Stage-1 carry-overs (JPM, XOM, WMT, MRK, BA) will be skipped on rerun thanks
to the idempotent (ticker, year) cache in data/pipeline.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.membership import STAGE2_TICKERS
from data.pipeline import concat_labeled, ingest_and_label

CACHE = ROOT / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=STAGE2_TICKERS)
    ap.add_argument("--start-year", type=int, default=2014)
    ap.add_argument("--end-year", type=int, default=2024, help="inclusive")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    years = range(args.start_year, args.end_year + 1)
    n_ok = 0
    n_skip = 0
    for t in args.tickers:
        for y in years:
            print(f"\n=== {t} {y} ===")
            try:
                ingest_and_label(t, y, overwrite=args.overwrite)
                n_ok += 1
            except Exception as e:
                # Bad ticker / no filings / yfinance hiccup — log and move on so
                # one ticker/year doesn't kill the whole stage.
                print(f"SKIP {t} {y}: {e}")
                n_skip += 1

    combined = concat_labeled(list(args.tickers), years)
    out = CACHE / f"stage2_events_{args.start_year}_{args.end_year}.parquet"
    combined.to_parquet(out, index=False)

    print(f"\nIngest done: {n_ok} ok, {n_skip} skipped.")
    print(f"wrote {out} ({len(combined)} events across {combined['ticker'].nunique()} tickers)")
    by_ticker = combined.groupby("ticker").size().sort_values(ascending=False)
    print("\nEvents per ticker:")
    print(by_ticker.to_string())


if __name__ == "__main__":
    main()
