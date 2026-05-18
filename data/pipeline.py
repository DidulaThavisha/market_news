"""Orchestrator: ingest 8-Ks + compute FF4 CAR labels for a (ticker, year)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data.edgar_ingest import (
    cik_for_ticker,
    fetch_filing_with_exhibits,
    items_mentioned,
    list_filings,
)
from data.factors import load_ff4
from data.labels import label_events
from data.prices import fetch_daily

CACHE = Path(__file__).resolve().parents[1] / "cache"


def labeled_path(ticker: str, year: int) -> Path:
    return CACHE / f"labeled_{ticker}_{year}.parquet"


def edgar_path(ticker: str, year: int) -> Path:
    return CACHE / f"edgar_8k_{ticker}_{year}.parquet"


def ingest_edgar(ticker: str, year: int, *, overwrite: bool = False) -> Path:
    out = edgar_path(ticker, year)
    if out.exists() and not overwrite:
        return out

    cik = cik_for_ticker(ticker)
    all_8k = list_filings(cik, form="8-K")
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    df = all_8k[(all_8k["filingDate"] >= start) & (all_8k["filingDate"] < end)].reset_index(drop=True)
    if df.empty:
        df["body_text"] = pd.Series(dtype="string")
        df["items_in_body"] = pd.Series(dtype="object")
        df["exhibits"] = pd.Series(dtype="object")
    else:
        bodies, body_items, exhibits = [], [], []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{ticker} {year}"):
            text, ex = fetch_filing_with_exhibits(
                cik, row["accessionNumber"], row["primaryDocument"]
            )
            bodies.append(text)
            body_items.append(items_mentioned(text))
            exhibits.append(ex)
        df["body_text"] = bodies
        df["items_in_body"] = body_items
        df["exhibits"] = exhibits

    df["ticker"] = ticker
    df["cik"] = cik
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def label(ticker: str, year: int, *, overwrite: bool = False) -> Path:
    out = labeled_path(ticker, year)
    if out.exists() and not overwrite:
        return out

    events = pd.read_parquet(edgar_path(ticker, year))
    if events.empty:
        events.to_parquet(out, index=False)
        return out

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

    labeled.to_parquet(out, index=False)
    return out


def ingest_and_label(ticker: str, year: int, *, overwrite: bool = False) -> Path:
    ingest_edgar(ticker, year, overwrite=overwrite)
    return label(ticker, year, overwrite=overwrite)


def concat_labeled(tickers: list[str], years: range) -> pd.DataFrame:
    """Load and concat all labeled parquets for (ticker × year) grid."""
    frames = []
    for t in tickers:
        for y in years:
            p = labeled_path(t, y)
            if p.exists():
                df = pd.read_parquet(p)
                if not df.empty:
                    frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("acceptanceDateTime")
