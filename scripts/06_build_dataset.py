"""Build train/val/test JSONL from the Stage-1 labeled events parquet.

Loads the tokenizer (Qwen3-8B) from HuggingFace to apply its chat template.
Tokenizer load works on CPU/Apple Silicon — no GPU required for this step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from transformers import AutoTokenizer

from training.dataset import SplitConfig, split_events, write_splits

CACHE = ROOT / "cache"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(CACHE / "stage1_events_2015_2022.parquet"))
    ap.add_argument("--out-dir", default=str(CACHE / "dataset_stage1"))
    ap.add_argument("--model", default="unsloth/Qwen3-8B")
    ap.add_argument("--train-end", default="2021-01-01")
    ap.add_argument("--val-end",   default="2022-01-01")
    ap.add_argument("--test-end",  default="2023-01-01")
    args = ap.parse_args()

    df = pd.read_parquet(args.events)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    splits = split_events(df, SplitConfig(args.train_end, args.val_end, args.test_end))
    counts = write_splits(splits, tokenizer, Path(args.out_dir))

    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
