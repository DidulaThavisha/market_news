"""Run the event-driven backtest against a probabilistic predictions parquet.

Stage 2 must-prove: gross Sharpe > 1.0. Stage 3 must-prove: net Sharpe > 0.5.

The default thresholds (edge=0.20, material=0.50, 1% sizing, 15 bps cost) are
starting points — tune on the val split before quoting test numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from eval.backtest import BacktestConfig, run_backtest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="parquet from training/infer_probs.py")
    ap.add_argument("--out-dir", required=True, help="where to write trades + equity_curve")
    ap.add_argument("--holding-days", type=int, default=10)
    ap.add_argument("--edge-threshold", type=float, default=0.20)
    ap.add_argument("--material-threshold", type=float, default=0.50)
    ap.add_argument("--position-size", type=float, default=0.01)
    ap.add_argument("--cost-bps", type=float, default=15.0)
    args = ap.parse_args()

    df = pd.read_parquet(args.predictions)
    cfg = BacktestConfig(
        holding_days=args.holding_days,
        edge_threshold=args.edge_threshold,
        material_threshold=args.material_threshold,
        position_size=args.position_size,
        cost_bps_round_trip=args.cost_bps,
    )
    print(f"Config: {cfg}")
    print(f"Predictions: {len(df)} events")

    result = run_backtest(df, cfg)
    trades = result["trades"]
    equity = result["equity_curve"]
    summary = result["summary"]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not trades.empty:
        trades.to_parquet(out_dir / "trades.parquet", index=False)
        equity.to_frame("equity").to_parquet(out_dir / "equity_curve.parquet")

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
