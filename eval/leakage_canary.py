"""Leakage canary: text features must NOT predict pre-event abnormal returns.

If a classifier trained on event text predicts the sign of CAR[-1, 0] with
AUC noticeably above 0.5 (out-of-time), the pipeline has look-ahead bias.

We train on the SAME features against:
  - sign(CAR[0, +1])     : may or may not be predictable; we hope it is.
  - sign(CAR[-1, 0])     : MUST NOT be predictable; this is the canary.

Time-ordered split. Returns AUCs and a verdict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score


def _binary_target(car: pd.Series) -> np.ndarray:
    """1 if CAR>0 else 0; NaN-safe."""
    return (car > 0).astype(int).values


def _time_split(n: int, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray]:
    cut = int(n * train_frac)
    idx = np.arange(n)
    return idx[:cut], idx[cut:]


def _auc_via_time_split(text: list[str], y: np.ndarray) -> float | None:
    n = len(text)
    if n < 12:
        return None
    train_idx, test_idx = _time_split(n)
    if len(set(y[train_idx])) < 2 or len(set(y[test_idx])) < 2:
        return None
    vec = TfidfVectorizer(
        max_features=2000, ngram_range=(1, 2), min_df=2, stop_words="english"
    )
    Xtr = vec.fit_transform([text[i] for i in train_idx])
    Xte = vec.transform([text[i] for i in test_idx])
    clf = LogisticRegressionCV(Cs=5, cv=3, max_iter=1000, scoring="roc_auc")
    clf.fit(Xtr, y[train_idx])
    proba = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y[test_idx], proba))


def run_canary(
    labeled_df: pd.DataFrame,
    text_col: str = "body_text",
    time_col: str = "acceptanceDateTime",
) -> dict:
    """Run the canary on a labeled events DataFrame (sorted by publish time)."""
    df = labeled_df.dropna(subset=["car_0_1", "car_minus1_0"]).copy()
    df = df.sort_values(time_col).reset_index(drop=True)
    text = df[text_col].fillna("").astype(str).tolist()

    auc_forward = _auc_via_time_split(text, _binary_target(df["car_0_1"]))
    auc_canary = _auc_via_time_split(text, _binary_target(df["car_minus1_0"]))

    n = len(df)
    verdict = _verdict(n, auc_canary)

    return {
        "n_events": n,
        "auc_forward_CAR_0_1": auc_forward,
        "auc_canary_CAR_minus1_0": auc_canary,
        "verdict": verdict,
    }


def _verdict(n: int, auc_canary: float | None) -> str:
    """Sample-size-aware verdict. Below ~200 events the AUC is too noisy to trust."""
    if auc_canary is None:
        return "INSUFFICIENT_DATA"
    if n < 200:
        return f"INCONCLUSIVE — N={n} too small for meaningful canary (need ~200+ events)"
    if auc_canary > 0.62:
        return "FAIL — text predicts pre-event returns; check for leakage"
    if auc_canary > 0.56:
        return "WARN — borderline; investigate before scaling"
    return "PASS — text does not predict pre-event returns"
