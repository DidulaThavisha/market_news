"""Stage 1+ eval metrics computed on the inference parquet.

Auto-detects whether predictions include probabilities (Stage 2+ infer_probs.py)
and upgrades the metrics accordingly:

  Headline (probs available):
    - Spearman IC: rho(P(up) - P(down), CAR[0,+1])
    - Materiality AUC on P(material)
    - Brier on P(material)
    - ECE (10-bin) on P(material)
  Fallback (hard labels only, Stage 1):
    - Spearman IC: rho(score{down/neutral/up: -1/0/+1}, CAR)
    - Materiality AUC/Brier as 1/0 proxies

Stratifications: per-ticker, per-year, per-sector (when sector column present).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

DIRECTION_TO_SCORE = {"down": -1, "neutral": 0, "up": 1}


def _has_probs(df: pd.DataFrame) -> bool:
    return {"pred_p_up", "pred_p_down", "pred_p_material"}.issubset(df.columns)


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float | None:
    if len(set(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, score))


def _ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float | None:
    if len(y_true) < n_bins:
        return None
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi) if hi < 1.0 else (prob >= lo) & (prob <= hi)
        if m.sum() == 0:
            continue
        conf = prob[m].mean()
        acc = y_true[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def _direction_score(df: pd.DataFrame) -> pd.Series:
    """Continuous direction score for Spearman; probs if available, else hard-label proxy."""
    if _has_probs(df):
        return df["pred_p_up"] - df["pred_p_down"]
    return df["pred_direction"].map(DIRECTION_TO_SCORE).astype(float)


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
    if "pred_p_material" in valid.columns:
        score = valid["pred_p_material"].astype(float).values
        proxy = False
    else:
        score = (valid["pred_materiality"] == "material").astype(int).values
        proxy = True
    return {
        "n": int(len(valid)),
        "accuracy": float((y_true == (score > 0.5).astype(int)).mean()),
        "auc": _safe_auc(y_true, score),
        "brier": float(brier_score_loss(y_true, score)) if len(set(y_true)) >= 2 else None,
        "ece_10bin": _ece(y_true, score),
        "base_rate_material": float(y_true.mean()),
        "score_is_proxy": proxy,
    }


def _spearman_ic(df: pd.DataFrame) -> dict[str, Any]:
    valid = df.dropna(subset=["car_0_1"])
    score = _direction_score(valid)
    valid = valid[score.notna()]
    score = score.dropna()
    if len(valid) < 5:
        return {"n": int(len(valid))}
    if float(score.std(ddof=0)) < 1e-12 or float(valid["car_0_1"].std(ddof=0)) < 1e-12:
        return {
            "n": int(len(valid)),
            "spearman_rho": None,
            "p_value": None,
            "note": "constant input — model produces a single score",
            "score_uses_probs": _has_probs(df),
        }
    rho, p = spearmanr(score, valid["car_0_1"])
    return {
        "n": int(len(valid)),
        "spearman_rho": float(rho),
        "p_value": float(p),
        "score_uses_probs": _has_probs(df),
    }


def overall_report(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "direction": _direction_metrics(df),
        "materiality": _materiality_metrics(df),
        "ic": _spearman_ic(df),
    }


def stratified_report(df: pd.DataFrame, by: str, min_n: int = 10) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for key, sub in df.groupby(by):
        if len(sub) < min_n:
            continue
        ic = _spearman_ic(sub)
        out[str(key)] = {
            "n": int(len(sub)),
            "direction_accuracy": float((sub["direction"] == sub["pred_direction"]).mean()),
            "spearman": ic.get("spearman_rho"),
            "materiality_auc": _materiality_metrics(sub).get("auc"),
        }
    return out


def year_strat(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    df = df.copy()
    df["year"] = pd.to_datetime(df["publish_ts_utc"], utc=True).dt.year.astype(str)
    return stratified_report(df, "year")


def sector_strat(df: pd.DataFrame) -> dict[str, dict[str, Any]] | None:
    """Sector stratification — requires `sector` column (Stage 2+ adds this)."""
    if "sector" not in df.columns:
        return None
    return stratified_report(df, "sector")
