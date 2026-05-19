"""Eval: report metrics on the predictions parquet (greedy infer.py or infer_probs.py).

Auto-detects probabilistic outputs (Stage 2+) and includes AUC/Brier/ECE.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from eval.metrics import overall_report, sector_strat, stratified_report, year_strat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="Parquet from infer.py or infer_probs.py")
    args = ap.parse_args()

    df = pd.read_parquet(args.predictions)

    print("=== OVERALL ===")
    print(json.dumps(overall_report(df), indent=2, default=str))

    print("\n=== BY TICKER ===")
    print(json.dumps(stratified_report(df, "ticker"), indent=2, default=str))

    print("\n=== BY YEAR ===")
    print(json.dumps(year_strat(df), indent=2, default=str))

    sector = sector_strat(df)
    if sector is not None:
        print("\n=== BY SECTOR ===")
        print(json.dumps(sector, indent=2, default=str))


if __name__ == "__main__":
    main()
