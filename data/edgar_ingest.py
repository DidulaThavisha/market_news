"""SEC EDGAR 8-K ingestion. First-party regulatory filings from listed companies."""
from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests

SEC_USER_AGENT = "MarketNews-POC didula@wso2.com"
SEC_WWW = "https://www.sec.gov"
SEC_DATA = "https://data.sec.gov"
REQ_SLEEP = 0.12  # SEC limit is 10 req/s; stay under it

_session = requests.Session()
_session.headers.update({
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
})


def _get(url: str) -> requests.Response:
    r = _session.get(url, timeout=30)
    time.sleep(REQ_SLEEP)
    r.raise_for_status()
    return r


def cik_for_ticker(ticker: str) -> str:
    """Look up zero-padded CIK from ticker via SEC's company_tickers.json."""
    r = _get(f"{SEC_WWW}/files/company_tickers.json")
    for entry in r.json().values():
        if entry["ticker"].upper() == ticker.upper():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"ticker not in SEC registry: {ticker}")


def _arrays_to_df(arrays: dict) -> pd.DataFrame:
    """Convert SEC's parallel-array filing dict into a DataFrame."""
    n = len(arrays.get("accessionNumber", []))
    if n == 0:
        return pd.DataFrame()
    cols = {
        "accessionNumber": arrays["accessionNumber"],
        "filingDate": pd.to_datetime(arrays["filingDate"], errors="coerce"),
        "acceptanceDateTime": pd.to_datetime(
            arrays["acceptanceDateTime"], utc=True, errors="coerce"
        ),
        "form": arrays["form"],
        "primaryDocument": arrays["primaryDocument"],
        "primaryDocDescription": arrays.get("primaryDocDescription", [""] * n),
        "items": arrays.get("items", [""] * n),
    }
    return pd.DataFrame(cols)


def list_filings(cik: str, form: str | None = "8-K") -> pd.DataFrame:
    """All filings for a CIK across recent + historical shards. Optionally filter by form."""
    r = _get(f"{SEC_DATA}/submissions/CIK{cik}.json")
    j = r.json()
    frames = [_arrays_to_df(j["filings"]["recent"])]
    for shard in j["filings"].get("files", []):
        r2 = _get(f"{SEC_DATA}/submissions/{shard['name']}")
        frames.append(_arrays_to_df(r2.json()))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    if form is not None:
        df = df[df["form"] == form]
    return df.sort_values("filingDate").reset_index(drop=True)


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ITEM_RE = re.compile(r"Item\s+\d+\.\d+", re.IGNORECASE)


def strip_html(html: str) -> str:
    """Cheap HTML → text. Good enough for 8-K body extraction."""
    text = _TAG_RE.sub(" ", html)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def fetch_filing_text(cik: str, accession: str, primary_doc: str, max_chars: int = 8000) -> str:
    """Fetch primary 8-K document; return cleaned text truncated to max_chars."""
    acc_clean = accession.replace("-", "")
    url = f"{SEC_WWW}/Archives/edgar/data/{int(cik)}/{acc_clean}/{primary_doc}"
    r = _get(url)
    text = strip_html(r.text)
    return text[:max_chars]


_EXHIBIT_EXTS = (".htm", ".html", ".txt")


def fetch_filing_with_exhibits(
    cik: str,
    accession: str,
    primary_doc: str,
    max_chars: int = 12000,
) -> tuple[str, list[str]]:
    """Fetch the primary 8-K + any EX-99 press-release exhibits; concat the text.

    Returns (text, list-of-exhibits-fetched). 8-K cover pages are mostly boilerplate;
    the substantive news lives in EX-99.x exhibits.
    """
    acc_clean = accession.replace("-", "")
    base = f"{SEC_WWW}/Archives/edgar/data/{int(cik)}/{acc_clean}"

    parts: list[str] = []
    exhibits: list[str] = []

    parts.append(strip_html(_get(f"{base}/{primary_doc}").text)[:4000])

    idx = _get(f"{base}/index.json").json()
    items = idx.get("directory", {}).get("item", []) or []
    for item in items:
        name = item.get("name", "")
        lower = name.lower()
        if name == primary_doc:
            continue
        if not lower.endswith(_EXHIBIT_EXTS):
            continue
        if "ex99" not in lower and "ex-99" not in lower and "exhibit99" not in lower:
            continue
        try:
            text = strip_html(_get(f"{base}/{name}").text)
        except requests.HTTPError:
            continue
        parts.append(text[:8000])
        exhibits.append(name)

    combined = " ".join(parts).strip()
    return combined[:max_chars], exhibits


def items_mentioned(text: str) -> list[str]:
    """Return unique 8-K Item numbers referenced in the body text."""
    found = {m.group(0).lower().replace("item", "").strip() for m in _ITEM_RE.finditer(text)}
    return sorted(found)
