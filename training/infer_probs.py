"""Probabilistic inference for Stage 2+.

Stage 1's infer.py uses greedy decode + regex parse → hard labels only. That
forces the Spearman IC to use a coarse {-1,0,+1} proxy and makes calibration
impossible to measure properly.

This script emits full probability distributions instead:

  P(direction)   softmax over {down, neutral, up}
  P(materiality) softmax over {material, immaterial} | chosen direction

Method (per event):
  1. Take the prompt-only chat sequence (test_prompt.jsonl row, which ends with
     "<|im_start|>assistant\\n").
  2. Append "Direction: ". One forward pass. Look at logits at the LAST position
     — they predict the next token, which is the first token of the direction
     word. Softmax over the three direction first-tokens.
  3. Fill in the argmax direction, append "\\nMateriality: ". One forward pass.
     Softmax over the two materiality first-tokens.

Two forward passes per event. First-token softmax is an approximation when
labels are multi-token, but our six labels tokenize unambiguously under Qwen
BPE — verified at startup.

Output schema (parquet):
  All meta fields from prompt_jsonl, plus:
    pred_direction, pred_p_down, pred_p_neutral, pred_p_up,
    pred_materiality, pred_p_material, pred_p_immaterial
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm


DIRECTIONS = ["down", "neutral", "up"]
MATERIALITY = ["material", "immaterial"]


def _first_token_id(tokenizer, label: str, prefix: str) -> int:
    """First token id of `label` when it follows `prefix`. Handles BPE merging at
    the prefix/label boundary (the token may differ depending on what's before)."""
    prefix_ids = tokenizer(prefix, add_special_tokens=False).input_ids
    full_ids = tokenizer(prefix + label, add_special_tokens=False).input_ids
    if not full_ids[: len(prefix_ids)] == prefix_ids:
        raise SystemExit(
            f"Tokenizer reshuffles prefix when label='{label}' is appended. "
            f"First-token scoring would be wrong; fall back to teacher-forced log-probs."
        )
    return full_ids[len(prefix_ids)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--prompt-jsonl", required=True, help="cache/dataset_stageN/test_prompt.jsonl")
    ap.add_argument("--base-model", default="unsloth/Qwen3-8B")
    ap.add_argument("--out", required=True, help="output parquet")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    args = ap.parse_args()

    from unsloth import FastLanguageModel
    import torch
    from peft import PeftModel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required.")

    adapter_path = Path(args.adapter_dir)
    if not (adapter_path / "adapter_config.json").exists():
        raise SystemExit(f"Missing adapter_config.json in {adapter_path}")

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    print(f"Attaching LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))
    FastLanguageModel.for_inference(model)

    # Direction tokens are scored at the position after "Direction: ". Materiality
    # at the position after "\nMateriality: ". Compute once.
    dir_token_ids = [_first_token_id(tokenizer, lab, "Direction: ") for lab in DIRECTIONS]
    mat_token_ids = [_first_token_id(tokenizer, lab, "\nMateriality: ") for lab in MATERIALITY]
    print(f"direction first-token ids: {dict(zip(DIRECTIONS, dir_token_ids))}")
    print(f"materiality first-token ids: {dict(zip(MATERIALITY, mat_token_ids))}")

    rows = []
    with open(args.prompt_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))

    dir_token_t = torch.tensor(dir_token_ids, device="cuda")
    mat_token_t = torch.tensor(mat_token_ids, device="cuda")

    out_records = []
    for row in tqdm(rows, desc="score"):
        base_text = row["text"]  # ends with "<|im_start|>assistant\n"

        # Direction
        dir_text = base_text + "Direction: "
        ids = tokenizer(dir_text, return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=args.max_seq_length).input_ids.to("cuda")
        with torch.inference_mode():
            last_logits = model(ids).logits[0, -1]
        dir_logits = last_logits[dir_token_t]
        dir_probs = torch.softmax(dir_logits.float(), dim=0).tolist()
        probs_d = dict(zip(DIRECTIONS, dir_probs))
        pred_d = max(probs_d, key=probs_d.get)

        # Materiality conditional on chosen direction
        mat_text = base_text + f"Direction: {pred_d}\nMateriality: "
        ids = tokenizer(mat_text, return_tensors="pt", add_special_tokens=False,
                        truncation=True, max_length=args.max_seq_length).input_ids.to("cuda")
        with torch.inference_mode():
            last_logits = model(ids).logits[0, -1]
        mat_logits = last_logits[mat_token_t]
        mat_probs = torch.softmax(mat_logits.float(), dim=0).tolist()
        probs_m = dict(zip(MATERIALITY, mat_probs))
        pred_m = max(probs_m, key=probs_m.get)

        rec = dict(row["meta"])
        rec.update({
            "pred_direction": pred_d,
            "pred_p_down": probs_d["down"],
            "pred_p_neutral": probs_d["neutral"],
            "pred_p_up": probs_d["up"],
            "pred_materiality": pred_m,
            "pred_p_material": probs_m["material"],
            "pred_p_immaterial": probs_m["immaterial"],
        })
        out_records.append(rec)

    df = pd.DataFrame(out_records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out}  rows={len(df)}")
    print(df["pred_direction"].value_counts().to_string())


if __name__ == "__main__":
    main()
