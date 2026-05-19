"""Why is the trained model degenerate?

Reads class distributions from the dataset JSONL files and compares them to the
prediction distribution. Flags the most likely failure modes (majority-class
collapse, score variance == 0, base-rate mismatch between train and test).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd


def _counts(jsonl_path: Path) -> tuple[dict, dict, int]:
    dirs: Counter = Counter()
    mats: Counter = Counter()
    n = 0
    with jsonl_path.open() as f:
        for line in f:
            row = json.loads(line)
            m = row.get("meta", {})
            if "direction" in m:
                dirs[m["direction"]] += 1
            if "materiality" in m:
                mats[m["materiality"]] += 1
            n += 1
    return dict(dirs), dict(mats), n


def _pct(counts: dict[str, int], total: int) -> dict[str, str]:
    return {k: f"{v / max(total, 1):.1%}" for k, v in counts.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="cache/dataset_stageN/")
    ap.add_argument("--predictions", required=True, help="predictions parquet")
    args = ap.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_dirs: dict[str, int] = {}
    for name in ("train.jsonl", "val_labeled.jsonl", "test_labeled.jsonl"):
        path = dataset_dir / name
        if not path.exists():
            continue
        d, m, n = _counts(path)
        print(f"\n=== {name} (n={n}) ===")
        print(f"  direction counts: {d}")
        print(f"  direction pct:    {_pct(d, n)}")
        print(f"  materiality cnts: {m}")
        if name == "train.jsonl":
            train_dirs = d

    print("\n=== predictions ===")
    df = pd.read_parquet(args.predictions)
    n = len(df)
    pred = df["pred_direction"].value_counts().to_dict()
    truth = df["direction"].value_counts().to_dict()
    print(f"  n: {n}")
    print(f"  predicted: {pred}  ({_pct(pred, n)})")
    print(f"  truth:     {truth}  ({_pct(truth, n)})")

    if "pred_p_up" in df.columns:
        edges = (df["pred_p_up"] - df["pred_p_down"]).astype(float)
        print("\n=== probabilistic signal sanity ===")
        print(f"  P(up) - P(down): min={edges.min():.3f} max={edges.max():.3f} "
              f"mean={edges.mean():.3f} std={edges.std():.3f}")
        print(f"  P(material):     min={df['pred_p_material'].min():.3f} "
              f"max={df['pred_p_material'].max():.3f} std={df['pred_p_material'].std():.3f}")
        if edges.std() < 1e-3:
            print("  WARN: edge std ~ 0 → model is genuinely producing constant outputs, not just argmax artifacts.")
    else:
        print("\n[no prob columns — re-run with training/infer_probs.py to check for latent signal]")

    print("\n=== DIAGNOSIS ===")
    issues = []
    if len(pred) == 1:
        cls = next(iter(pred))
        share = train_dirs.get(cls, 0) / max(sum(train_dirs.values()), 1)
        issues.append(
            f"Majority-class collapse: model predicts '{cls}' for 100% of test. "
            f"In train this class is {share:.1%}."
        )
    if n < 200:
        issues.append(
            f"Test set is tiny (n={n}). At this size, the eval is high-variance even "
            f"if the model is fine — single-quarter quirks dominate."
        )
    if "pred_p_up" in df.columns:
        edges = (df["pred_p_up"] - df["pred_p_down"]).astype(float)
        if edges.std() < 1e-3:
            issues.append(
                "Probabilistic edge has ~zero variance: model truly hasn't learned to "
                "discriminate, not just an argmax tiebreaker."
            )
    if not issues:
        print("  No obvious red flags from this diagnostic. Investigate training-loss curve next.")
    else:
        for i, msg in enumerate(issues, 1):
            print(f"  {i}. {msg}")

    print("\n=== SUGGESTED FIXES ===")
    print("  1. Rebuild dataset with --balance to oversample minority direction classes.")
    print("  2. Re-run with training/infer_probs.py to see prob distributions before re-training.")
    print("  3. Move to Stage 2 (25 tickers × 11 years → ~4K events) — 500-event scale is borderline for 8B QLoRA.")
    print("  4. Bump --epochs to 4 and/or --lora-r to 32 when re-training.")


if __name__ == "__main__":
    main()
