"""Stage-1 batch: 5 sector-diverse tickers × 2015-2022.

Train period: 2015-2020 (skip COVID Mar-May handled at split time).
Val: 2021.  Test: 2022.

This script is idempotent — re-running skips already-cached (ticker, year) parquets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data.membership import STAGE1_TICKERS
from data.pipeline import concat_labeled, ingest_and_label

CACHE = ROOT / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", default=STAGE1_TICKERS)
    ap.add_argument("--start-year", type=int, default=2015)
    ap.add_argument("--end-year", type=int, default=2022, help="inclusive")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    years = range(args.start_year, args.end_year + 1)
    for t in args.tickers:
        for y in years:
            print(f"\n=== {t} {y} ===")
            ingest_and_label(t, y, overwrite=args.overwrite)

    combined = concat_labeled(list(args.tickers), years)
    out = CACHE / f"stage1_events_{args.start_year}_{args.end_year}.parquet"
    combined.to_parquet(out, index=False)
    print(f"\nwrote {out} ({len(combined)} events across {combined['ticker'].nunique()} tickers)")
    print(combined.groupby("ticker").size().to_string())


if __name__ == "__main__":
    main()
