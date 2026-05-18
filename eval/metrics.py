"""Stage-1+ eval metrics computed on the inference parquet from training/infer.py.

Headline metrics:
  - Directional accuracy (3-class) vs majority-class baseline.
  - Spearman rank correlation between the model's directional score and realized CAR.
  - Materiality AUC (gracefully NA if too few positives).
  - Calibration (Brier score on materiality probability proxy).

Stratifications:
  - Per-ticker
  - Per-year (by publish_ts_utc)

Inference here is deterministic (no probabilities) — we proxy direction "score" as
{down: -1, neutral: 0, up: +1} for Spearman, and materiality "score" as 1/0.
Stage 2+ should switch to softmax probabilities for proper calibration.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
)
from scipy.stats import spearmanr

DIRECTION_TO_SCORE = {"down": -1, "neutral": 0, "up": 1}


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def _direction_metrics(df: pd.DataFrame) -> dict[str, Any]:
    valid = df.dropna(subset=["pred_direction", "direction"])
    if valid.empty:
        return {"n": 0}
    y_true = valid["direction"].values
    y_pred = valid["pred_direction"].values

    majority = valid["direction"].value_counts(normalize=True).iloc[0]
    accuracy = float((y_true == y_pred).mean())

    return {
        "n": int(len(valid)),
        "parseable_rate": float(df["pred_direction"].notna().mean()),
        "accuracy": accuracy,
        "majority_baseline": float(majority),
        "lift_over_majority_pp": (accuracy - float(majority)) * 100,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def _materiality_metrics(df: pd.DataFrame) -> dict[str, Any]:
    valid = df.dropna(subset=["pred_materiality", "materiality"])
    if valid.empty:
        return {"n": 0}
    y_true = (valid["materiality"] == "material").astype(int).values
    score = (valid["pred_materiality"] == "material").astype(int).values
    return {
        "n": int(len(valid)),
        "accuracy": float((y_true == score).mean()),
        "auc_proxy": _safe_auc(y_true, score),
        "brier_proxy": float(brier_score_loss(y_true, score)) if len(set(y_true)) >= 2 else None,
        "base_rate_material": float(y_true.mean()),
    }


def _spearman_ic(df: pd.DataFrame) -> dict[str, Any]:
    valid = df.dropna(subset=["pred_direction", "car_0_1"])
    if len(valid) < 5:
        return {"n": int(len(valid))}
    score = valid["pred_direction"].map(DIRECTION_TO_SCORE).astype(float)
    rho, p = spearmanr(score, valid["car_0_1"])
    return {"n": int(len(valid)), "spearman_rho": float(rho), "p_value": float(p)}


def overall_report(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "direction": _direction_metrics(df),
        "materiality": _materiality_metrics(df),
        "ic": _spearman_ic(df),
    }


def stratified_report(df: pd.DataFrame, by: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, sub in df.groupby(by):
        if len(sub) < 10:
            continue
        out[str(key)] = {
            "n": int(len(sub)),
            "direction_accuracy": float(
                (sub["direction"] == sub["pred_direction"]).mean()
            ),
            "spearman": _spearman_ic(sub).get("spearman_rho"),
        }
    return out


def year_strat(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    df = df.copy()
    df["year"] = pd.to_datetime(df["publish_ts_utc"], utc=True).dt.year.astype(str)
    return stratified_report(df, "year")
