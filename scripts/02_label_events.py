"""Stage-0: label JPM 2019 events with FF4 CAR[0,+1] and CAR[-1,0] (canary)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from data.factors import load_ff4
from data.labels import label_events
from data.prices import fetch_daily

CACHE = ROOT / "cache"


def run(ticker: str, year: int) -> Path:
    events = pd.read_parquet(CACHE / f"edgar_8k_{ticker}_{year}.parquet")
    publish = pd.to_datetime(events["acceptanceDateTime"], utc=True)

    prices = fetch_daily(ticker, f"{year - 1}-01-01", f"{year + 1}-06-01")
    factors = load_ff4()

    primary = label_events(publish, prices, factors, car_window=(0, 1))
    canary = label_events(publish, prices, factors, car_window=(-1, 0))

    labeled = events.copy()
    labeled["car_0_1"] = primary["car"].values
    labeled["car_z_0_1"] = primary["car_z"].values
    labeled["direction"] = primary["direction"].values
    labeled["material"] = primary["material"].values
    labeled["sigma_window_0_1"] = primary["sigma_window"].values
    labeled["factor_r2"] = primary["factor_r2"].values
    labeled["t0"] = primary["t0"].values
    labeled["car_minus1_0"] = canary["car"].values
    labeled["car_z_minus1_0"] = canary["car_z"].values

    out = CACHE / f"labeled_{ticker}_{year}.parquet"
    labeled.to_parquet(out, index=False)
    print(f"wrote {out} ({len(labeled)} events)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="JPM")
    ap.add_argument("--year", type=int, default=2019)
    args = ap.parse_args()
    run(args.ticker, args.year)


if __name__ == "__main__":
    main()
