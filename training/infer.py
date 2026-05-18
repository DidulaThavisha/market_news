"""Batch inference: load LoRA adapter + run on val/test prompts.

REQUIRES CUDA (loads in 4-bit via Unsloth). Run on the same box that trained.

Outputs a parquet with one row per prompt:
  - all `meta` fields from the JSONL (ticker, publish_ts_utc, true labels, car)
  - `raw_response`: the model's generated text
  - `pred_direction`, `pred_materiality`: parsed predictions (None if unparseable)
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

from training.prompt import parse_response


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True, help="LoRA adapter dir (output of train_qwen.py)")
    ap.add_argument("--prompt-jsonl", required=True, help="e.g. cache/dataset_stage1/test_prompt.jsonl")
    ap.add_argument("--base-model", default="unsloth/Qwen3-8B")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out", required=True, help="Output parquet path")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. This script requires an NVIDIA GPU.")

    adapter_path = Path(args.adapter_dir)
    if not (adapter_path / "adapter_config.json").exists():
        raise SystemExit(
            f"No adapter_config.json in {adapter_path}. Did training finish and "
            f"save to {adapter_path}? Expected files: adapter_config.json, "
            f"adapter_model.safetensors."
        )

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
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
    for i in tqdm(range(0, len(rows), args.batch_size), desc="generate"):
        batch = rows[i : i + args.batch_size]
        texts = [r["text"] for r in batch]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=2048).to("cuda")
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        for r, generated, input_ids in zip(batch, out, enc["input_ids"]):
            new_tokens = generated[input_ids.shape[0]:]
            raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
            parsed = parse_response(raw)
            rec = dict(r["meta"])
            rec["raw_response"] = raw
            rec["pred_direction"] = parsed[0] if parsed else None
            rec["pred_materiality"] = parsed[1] if parsed else None
            out_records.append(rec)

    out_df = pd.DataFrame(out_records)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    parseable = out_df["pred_direction"].notna().mean()
    print(f"wrote {args.out}  rows={len(out_df)}  parseable={parseable:.2%}")


if __name__ == "__main__":
    main()
