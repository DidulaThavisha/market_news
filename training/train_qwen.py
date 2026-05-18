"""Unsloth QLoRA fine-tune of Qwen3-8B for direction + materiality prediction.

REQUIRES CUDA. Run on a 24GB+ GPU (4090/A5000/A6000/L40S/H100). Will fail on
Apple Silicon. Install training extras first:

    uv sync --extra training

Usage:

    uv run --extra training python training/train_qwen.py \\
        --data-dir cache/dataset_stage1 \\
        --output-dir outputs/qwen3-8b-lora-stage1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Dir with train.jsonl from script 06")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="unsloth/Qwen3-8B")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-val", action="store_true", help="Evaluate on val_labeled.jsonl each epoch")
    args = ap.parse_args()

    # Imports are deferred so that running with --help on a CPU box doesn't crash.
    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available. This script requires an NVIDIA GPU.")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
    )

    train_files = {"train": str(data_dir / "train.jsonl")}
    if args.use_val:
        val_path = data_dir / "val_labeled.jsonl"
        if val_path.exists():
            train_files["validation"] = str(val_path)
    raw = load_dataset("json", data_files=train_files)

    bf16_ok = torch.cuda.is_bf16_supported()
    targs = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=not bf16_ok,
        bf16=bf16_ok,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        output_dir=str(out_dir),
        save_strategy="epoch",
        eval_strategy="epoch" if "validation" in raw else "no",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=raw["train"],
        eval_dataset=raw.get("validation"),
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
        args=targs,
    )

    # Compute loss only on assistant tokens — prompt is masked.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )

    trainer.train()

    final_dir = out_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"saved LoRA adapter to {final_dir}")


if __name__ == "__main__":
    main()
