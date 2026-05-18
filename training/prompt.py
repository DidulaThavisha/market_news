"""Frozen prompt template + feature builders.

Any feature shown to the model must be knowable strictly before publish_ts.
No future prices, no future analyst data, no fundamentals dated after publish.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from data.membership import sector_for

SYSTEM_PROMPT = (
    "You are a financial event reaction predictor. Given a regulatory filing "
    "about a public company, predict the direction and materiality of the "
    "next-day stock-price reaction in factor-adjusted abnormal-return terms."
)

USER_PROMPT_TEMPLATE = """Ticker: {ticker}
Sector: {sector}
Filing date (UTC): {publish_date}
Time relative to market: {time_label}
Same-day 8-K count for ticker: {n_same_day}
Days since previous 8-K: {days_since}
8-K items: {items}
Body: {body}"""

ASSISTANT_TEMPLATE = "Direction: {direction}\nMateriality: {materiality}"

BODY_MAX_CHARS = 1500


_BOILERPLATE_CUES = (
    "ex-99", "ex99", "exhibit 99", "news release", "press release",
    "reports", "announces", "completes", "agrees",
)


def trim_body(body: str, max_chars: int = BODY_MAX_CHARS) -> str:
    """Skip SEC cover boilerplate; return up to max_chars of substantive content."""
    if not body:
        return ""
    lower = body.lower()
    starts = [lower.find(cue) for cue in _BOILERPLATE_CUES]
    starts = [s for s in starts if s >= 0]
    start = min(starts) if starts else 0
    text = body[start : start + max_chars]
    return re.sub(r"\s+", " ", text).strip()


def time_label(publish_ts_utc: pd.Timestamp) -> str:
    """pre_open / intraday / after_close / weekend, computed in US/Eastern."""
    if pd.isna(publish_ts_utc):
        return "unknown"
    et = publish_ts_utc.tz_convert("US/Eastern")
    if et.weekday() >= 5:
        return "weekend"
    t = et.time()
    if t < pd.Timestamp("09:30").time():
        return "pre_open"
    if t < pd.Timestamp("16:00").time():
        return "intraday"
    return "after_close"


@dataclass(frozen=True)
class EventFeatures:
    ticker: str
    sector: str
    publish_date: str
    time_label: str
    n_same_day: int
    days_since: int | str  # str "n/a" when no prior event
    items: str
    body: str
    direction: str | None
    materiality: str | None


def _compute_same_day_and_gap(events: pd.DataFrame) -> pd.DataFrame:
    """Add `n_same_day` and `days_since` columns, per ticker, in chronological order."""
    df = events.sort_values(["ticker", "acceptanceDateTime"]).copy()
    pub = pd.to_datetime(df["acceptanceDateTime"], utc=True)
    df["_pub_date_et"] = pub.dt.tz_convert("US/Eastern").dt.date

    df["n_same_day"] = df.groupby(["ticker", "_pub_date_et"])["_pub_date_et"].transform("size")

    prev_date = df.groupby("ticker")["_pub_date_et"].shift(1)
    gap = (pd.to_datetime(df["_pub_date_et"]) - pd.to_datetime(prev_date)).dt.days
    df["days_since"] = gap.fillna(-1).astype(int)

    df = df.drop(columns=["_pub_date_et"])
    return df


def build_features(events: pd.DataFrame) -> list[EventFeatures]:
    df = _compute_same_day_and_gap(events)
    out: list[EventFeatures] = []
    for _, r in df.iterrows():
        pub = pd.Timestamp(r["acceptanceDateTime"])
        if pub.tzinfo is None:
            pub = pub.tz_localize("UTC")
        ds = int(r["days_since"])
        out.append(EventFeatures(
            ticker=r["ticker"],
            sector=sector_for(r["ticker"]),
            publish_date=pub.tz_convert("US/Eastern").strftime("%Y-%m-%d"),
            time_label=time_label(pub),
            n_same_day=int(r["n_same_day"]),
            days_since="n/a" if ds < 0 else ds,
            items=str(r.get("items") or "").strip() or "n/a",
            body=trim_body(r.get("body_text", "")),
            direction=r.get("direction"),
            materiality="material" if bool(r.get("material")) else "immaterial",
        ))
    return out


def user_prompt(feat: EventFeatures) -> str:
    return USER_PROMPT_TEMPLATE.format(
        ticker=feat.ticker,
        sector=feat.sector,
        publish_date=feat.publish_date,
        time_label=feat.time_label,
        n_same_day=feat.n_same_day,
        days_since=feat.days_since,
        items=feat.items,
        body=feat.body,
    )


def assistant_response(feat: EventFeatures) -> str:
    return ASSISTANT_TEMPLATE.format(
        direction=feat.direction or "neutral",
        materiality=feat.materiality or "immaterial",
    )


def to_messages(feat: EventFeatures, include_response: bool) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(feat)},
    ]
    if include_response:
        msgs.append({"role": "assistant", "content": assistant_response(feat)})
    return msgs


_RESPONSE_RE = re.compile(
    r"Direction:\s*(down|neutral|up)\s*\n?Materiality:\s*(material|immaterial)",
    re.IGNORECASE,
)


def parse_response(text: str) -> tuple[str, str] | None:
    """Parse model output into (direction, materiality). Returns None if malformed."""
    m = _RESPONSE_RE.search(text)
    if not m:
        return None
    return m.group(1).lower(), m.group(2).lower()
