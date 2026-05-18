"""Stage-0 gate: run the leakage canary on labeled JPM 2019 events."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from eval.leakage_canary import run_canary

CACHE = ROOT / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="JPM")
    ap.add_argument("--year", type=int, default=2019)
    args = ap.parse_args()

    df = pd.read_parquet(CACHE / f"labeled_{args.ticker}_{args.year}.parquet")

    print("=== ALL EVENTS ===")
    print(json.dumps(run_canary(df), indent=2, default=str))

    no_earnings = df[~df["items"].fillna("").str.contains("2.02")].copy()
    print(f"\n=== EARNINGS EXCLUDED ({len(no_earnings)} of {len(df)}) ===")
    print(json.dumps(run_canary(no_earnings), indent=2, default=str))


if __name__ == "__main__":
    main()
