"""Build train/val/test JSONL files from labeled events.

Splits are STRICTLY temporal with a 5-trading-day embargo between segments.
Earnings 8-Ks (Item 2.02) are excluded from training but retained for
post-hoc evaluation in a separate bucket per the project plan.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from training.prompt import EventFeatures, build_features, to_messages

EMBARGO_DAYS = 7  # calendar; covers a 5-trading-day buffer comfortably


@dataclass(frozen=True)
class SplitConfig:
    train_end: str  # exclusive
    val_end: str    # exclusive
    test_end: str   # exclusive
    embargo_days: int = EMBARGO_DAYS


def _drop_earnings(df: pd.DataFrame) -> pd.DataFrame:
    items = df["items"].fillna("").astype(str)
    return df[~items.str.contains(r"\b2\.02\b", regex=True)].reset_index(drop=True)


def split_events(events: pd.DataFrame, cfg: SplitConfig) -> dict[str, pd.DataFrame]:
    df = events.dropna(subset=["car_0_1", "direction"]).copy()
    df["acceptanceDateTime"] = pd.to_datetime(df["acceptanceDateTime"], utc=True)

    embargo = pd.Timedelta(days=cfg.embargo_days)
    train_end = pd.Timestamp(cfg.train_end, tz="UTC")
    val_start = train_end + embargo
    val_end = pd.Timestamp(cfg.val_end, tz="UTC")
    test_start = val_end + embargo
    test_end = pd.Timestamp(cfg.test_end, tz="UTC")

    train = df[df["acceptanceDateTime"] < train_end]
    val = df[(df["acceptanceDateTime"] >= val_start) & (df["acceptanceDateTime"] < val_end)]
    test = df[(df["acceptanceDateTime"] >= test_start) & (df["acceptanceDateTime"] < test_end)]

    return {
        "train": _drop_earnings(train),
        "val": _drop_earnings(val),
        "test": _drop_earnings(test),
        "train_earnings_only": train[~train.index.isin(_drop_earnings(train).index)],
        "val_earnings_only": val[~val.index.isin(_drop_earnings(val).index)],
        "test_earnings_only": test[~test.index.isin(_drop_earnings(test).index)],
    }


def balance_directions(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Oversample minority direction classes (down/up) to match neutral count.

    Use ONLY on train. Val/test must keep the natural base rate so that lift
    over the majority baseline is measured honestly.
    """
    counts = df["direction"].value_counts()
    if counts.empty:
        return df
    target = int(counts.max())
    rng = np.random.default_rng(seed)
    parts = []
    for direction, n in counts.items():
        sub = df[df["direction"] == direction]
        if n < target:
            extra = sub.sample(n=target - int(n), replace=True,
                               random_state=int(rng.integers(0, 2**31)))
            sub = pd.concat([sub, extra], ignore_index=True)
        parts.append(sub)
    return (
        pd.concat(parts, ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )


def _apply_chat_template(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )


def write_jsonl(
    events: pd.DataFrame,
    tokenizer,
    out_path: Path,
    *,
    with_response: bool,
) -> int:
    """Render features → chat template → JSONL rows {text, meta}."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats: list[EventFeatures] = build_features(events)
    n = 0
    with out_path.open("w") as f:
        for ev, feat in zip(events.itertuples(index=False), feats):
            msgs = to_messages(feat, include_response=with_response)
            text = _apply_chat_template(tokenizer, msgs, add_generation_prompt=not with_response)
            meta = {
                "ticker": feat.ticker,
                "sector": feat.sector,
                "publish_ts_utc": pd.Timestamp(ev.acceptanceDateTime).isoformat(),
                "direction": feat.direction,
                "materiality": feat.materiality,
                "car_0_1": float(getattr(ev, "car_0_1", float("nan"))),
                "car_z_0_1": float(getattr(ev, "car_z_0_1", float("nan"))),
                "items": feat.items,
            }
            f.write(json.dumps({"text": text, "meta": meta}) + "\n")
            n += 1
    return n


def write_splits(
    splits: dict[str, pd.DataFrame],
    tokenizer,
    out_dir: Path,
    *,
    balance_train: bool = False,
) -> dict[str, int]:
    """Write train.jsonl (with response), val/test.jsonl (prompt only for inference)."""
    counts: dict[str, int] = {}
    train = splits["train"]
    if balance_train:
        before = train["direction"].value_counts().to_dict()
        train = balance_directions(train)
        after = train["direction"].value_counts().to_dict()
        print(f"train balancing: {before} -> {after}")
    counts["train"] = write_jsonl(
        train, tokenizer, out_dir / "train.jsonl", with_response=True
    )
    for name in ("val", "test"):
        counts[name + "_prompt"] = write_jsonl(
            splits[name], tokenizer, out_dir / f"{name}_prompt.jsonl", with_response=False
        )
        counts[name + "_labeled"] = write_jsonl(
            splits[name], tokenizer, out_dir / f"{name}_labeled.jsonl", with_response=True
        )
    return counts
