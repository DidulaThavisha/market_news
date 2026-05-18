"""Kenneth French daily FF3 + Momentum factors. Public-domain academic data."""
from __future__ import annotations

import io
import zipfile
from functools import lru_cache

import pandas as pd
import requests

FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
MOM_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Momentum_Factor_daily_CSV.zip"
)


def _download_csv_zip(url: str) -> str:
    r = requests.get(url, timeout=60, headers={"User-Agent": "market-news-poc"})
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f:
            return f.read().decode("latin-1")


def _parse_ff_csv(text: str, value_cols: list[str]) -> pd.DataFrame:
    """French CSVs have a header preamble + a numeric daily section + a footer of annuals.

    We find the first row whose first field parses as YYYYMMDD and read until
    the row stops being numeric.
    """
    lines = text.splitlines()
    start = None
    end = None
    for i, ln in enumerate(lines):
        first = ln.split(",", 1)[0].strip()
        if len(first) == 8 and first.isdigit():
            if start is None:
                start = i
            end = i
        elif start is not None:
            break
    if start is None:
        raise ValueError("could not locate daily-data section in French CSV")

    body = "\n".join(lines[start : end + 1])
    df = pd.read_csv(
        io.StringIO(body),
        header=None,
        names=["date", *value_cols],
    )
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in value_cols:
        df[c] = df[c].astype(float) / 100.0
    return df.set_index("date").sort_index()


@lru_cache(maxsize=1)
def load_ff4() -> pd.DataFrame:
    """Return DataFrame indexed by date with columns [MKT_RF, SMB, HML, RF, MOM]."""
    ff3 = _parse_ff_csv(_download_csv_zip(FF3_URL), ["MKT_RF", "SMB", "HML", "RF"])
    mom = _parse_ff_csv(_download_csv_zip(MOM_URL), ["MOM"])
    return ff3.join(mom, how="inner")
