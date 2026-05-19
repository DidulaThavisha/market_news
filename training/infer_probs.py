"""Probabilistic inference for Stage 2+ (teacher-forced log-probs).

Stage 1's infer.py uses greedy decode + regex parse → hard labels only. That
forces the Spearman IC to use a coarse {-1,0,+1} proxy and makes calibration
impossible to measure properly.

This script emits full probability distributions instead. For each event:

  1. Score each of {"Direction: down", "Direction: neutral", "Direction: up"}
     as a completion of the prompt via one teacher-forced forward pass each.
     Softmax over the three log-probs → P(direction).
  2. Fill in the argmax direction and append "\\nMateriality: ". Score each of
     {"material", "immaterial"} as a completion. Softmax → P(materiality).

Five forward passes per event. Robust to BPE re-merging at the prefix/label
boundary (which is why we don't use a first-token softmax — Qwen's BPE merges
trailing space + label letter together).

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


def teacher_force_logprob(model, tokenizer, prefix_text: str, completion_text: str,
                          max_length: int) -> float:
    """Return log P(completion | prefix) via one forward pass on prefix + completion.

    Handles BPE re-merging at the boundary by finding the longest matching prefix
    between the standalone prefix tokenization and the joint tokenization. Tokens
    that diverge are scored as part of the completion.
    """
    import torch

    full_ids = tokenizer(prefix_text + completion_text, return_tensors="pt",
                         add_special_tokens=False, truncation=True,
                         max_length=max_length).input_ids[0]
    prefix_ids = tokenizer(prefix_text, return_tensors="pt",
                            add_special_tokens=False, truncation=True,
                            max_length=max_length).input_ids[0]

    upto = int(min(prefix_ids.shape[0], full_ids.shape[0]))
    plen = 0
    for i in range(upto):
        if prefix_ids[i].item() != full_ids[i].item():
            break
        plen += 1

    if plen >= int(full_ids.shape[0]):
        return 0.0  # completion produced no new tokens; degenerate

    full_ids_b = full_ids.unsqueeze(0).to("cuda")
    with torch.inference_mode():
        logits = model(full_ids_b).logits[0]  # [seq, vocab]

    n_target = int(full_ids.shape[0]) - plen
    target = full_ids[plen:].to("cuda")
    pred_logits = logits[plen - 1 : plen - 1 + n_target]
    log_probs = torch.log_softmax(pred_logits.float(), dim=-1)
    lp = log_probs[torch.arange(n_target, device="cuda"), target].sum().item()
    return float(lp)


def _softmax_from_logprobs(lps: list[float]) -> list[float]:
    import torch
    t = torch.tensor(lps, dtype=torch.float32)
    return torch.softmax(t, dim=0).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--prompt-jsonl", required=True)
    ap.add_argument("--base-model", default="unsloth/Qwen3-8B")
    ap.add_argument("--out", required=True)
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

    rows = []
    with open(args.prompt_jsonl) as f:
        for line in f:
            rows.append(json.loads(line))

    out_records = []
    for row in tqdm(rows, desc="score"):
        base_text = row["text"]  # ends with "<|im_start|>assistant\n"

        # 3 forward passes for direction.
        dir_lps = [
            teacher_force_logprob(model, tokenizer, base_text,
                                  f"Direction: {lab}", args.max_seq_length)
            for lab in DIRECTIONS
        ]
        dir_probs = _softmax_from_logprobs(dir_lps)
        probs_d = dict(zip(DIRECTIONS, dir_probs))
        pred_d = max(probs_d, key=probs_d.get)

        # 2 forward passes for materiality, conditional on the chosen direction.
        mat_prefix = base_text + f"Direction: {pred_d}\nMateriality: "
        mat_lps = [
            teacher_force_logprob(model, tokenizer, mat_prefix, lab,
                                  args.max_seq_length)
            for lab in MATERIALITY
        ]
        mat_probs = _softmax_from_logprobs(mat_lps)
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
