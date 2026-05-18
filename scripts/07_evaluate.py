"""Stage-1 eval: report metrics on the predictions parquet from training/infer.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from eval.metrics import overall_report, stratified_report, year_strat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="Parquet output of training/infer.py")
    args = ap.parse_args()

    df = pd.read_parquet(args.predictions)

    print("=== OVERALL ===")
    print(json.dumps(overall_report(df), indent=2, default=str))

    print("\n=== BY TICKER ===")
    print(json.dumps(stratified_report(df, "ticker"), indent=2, default=str))

    print("\n=== BY YEAR ===")
    print(json.dumps(year_strat(df), indent=2, default=str))


if __name__ == "__main__":
    main()
