# market_news

POC: fine-tune a small LLM to predict whether a news event about a public company
will cause a material next-day price reaction, then drive trades from the signal.

Approach: pull first-party news (SEC EDGAR 8-K filings) for each ticker, label each
event with factor-adjusted abnormal return (FF4 CAR) normalized by per-stock
idiosyncratic vol, fine-tune Qwen3-8B via Unsloth QLoRA to predict direction +
materiality. Backtest event-driven with realistic costs.

The full design — model choice, label engineering, biases, stage gates, exit rules
— is in [`docs/plan.md`](docs/plan.md). Read that before touching the code.

## Status

| Stage | Scope | State |
| --- | --- | --- |
| 0 — Infra | 1 ticker, 1 year (JPM 2019), 31 events | done |
| 1 — Tiny pilot | 5 tickers, ~500 events | in flight (training on Kaggle CUDA) |
| 2 — Mid pilot | 25 tickers, ~4K events | code ready |
| 3 — Full POC | 100 tickers, ~15K events | pending |
| 4 — Full S&P 500 | ~490 tickers, ~75K events | pending |

Each stage has explicit go/kill criteria in the plan.

## Methodology pillars (non-negotiable)

These are the four ways this kind of project usually produces fake alpha. Don't
break any of them.

1. **FF4 abnormal return**, not raw price change. Per-stock idio-vol normalizes
   thresholds so a 2% move in PG isn't conflated with a 2% move in TSLA.
2. **Exact publish timestamps**, timezone-normalized. `t0` is the first trading
   session whose 09:30 ET open is strictly after `publish_ts_utc`.
3. **Earnings excluded from training** (Item 2.02 8-Ks). They have well-known
   pre-announcement drift that swamps everything else.
4. **Strict temporal train/val/test splits** with a 5-day embargo. Never shuffle.
5. **Leakage canary**: text features must not predict pre-event returns
   (`CAR[-1, 0]` AUC ≈ 0.5).

## Data sources

Free, public, commercial-OK only.

| Source | Use | Notes |
| --- | --- | --- |
| SEC EDGAR 8-Ks | News events + body text | Public-domain regulatory filings, first-party from listed companies. 10 req/s rate limit, `User-Agent` header required. |
| yfinance | Daily OHLCV | Adjusted close, daily granularity. |
| Kenneth French Data Library | FF3 + Momentum factors | Public academic data. |

`Zihan1004/FNSPID` was considered and rejected: CC-BY-NC license blocks commercial use.

## Setup

Requires Python 3.12. Uses [uv](https://github.com/astral-sh/uv) for dependency
management.

```bash
uv sync
```

## Pipeline

Each script is independently runnable and persists its output under `cache/`.

### Stage 0 — Infrastructure smoke test (CPU-only, runs locally)

```bash
# 1. Pull 8-Ks (with EX-99 exhibits) for a ticker × year window.
uv run python scripts/01_ingest_edgar.py --ticker JPM --start 2019-01-01 --end 2020-01-01

# 2. Compute FF4 CAR[0,+1] labels + CAR[-1,0] canary labels.
uv run python scripts/02_label_events.py --ticker JPM --year 2019

# 3. Run the leakage canary (sample-size-aware verdict).
uv run python scripts/03_leakage_canary.py --ticker JPM --year 2019

# 4. Print top-10 labeled events for human spot-check.
uv run python scripts/04_spot_check.py
```

### Stage 1 — Pilot data + training set (CPU for data, CUDA for training)

```bash
# 5. Ingest + label the 5-ticker pilot (JPM, XOM, WMT, MRK, BA) across 2015–2022.
#    Idempotent: rerunning skips already-cached (ticker, year) pairs.
uv run python scripts/05_ingest_stage1.py

# 6. Build the prompt dataset: temporal split with 7-day embargo, earnings dropped
#    from train/val/test (kept in *_earnings_only parquet for separate eval).
uv run python scripts/06_build_dataset.py \
    --events cache/stage1_events_2015_2022.parquet \
    --out-dir cache/dataset_stage1
```

After this you should have, under `cache/dataset_stage1/`:
- `train.jsonl` — fully-formatted chat sequences (with assistant response)
- `val_prompt.jsonl` / `test_prompt.jsonl` — prompts only, for inference
- `val_labeled.jsonl` / `test_labeled.jsonl` — prompts + true responses, for reference
- `*_earnings_only.jsonl` — Item 2.02 8-Ks held out for separate evaluation

### Stage 1 — Train + eval on a CUDA box

Install the training extras on the GPU machine (needs CUDA, won't work on Apple
Silicon). Unsloth's exact wheel depends on your CUDA/PyTorch combo; see
[`pyproject.toml`](pyproject.toml) for the fallback if `uv sync` can't resolve it.

```bash
uv sync --extra training

# 7. QLoRA fine-tune Qwen3-8B. ~24GB VRAM minimum.
uv run --extra training python training/train_qwen.py \
    --data-dir cache/dataset_stage1 \
    --output-dir outputs/qwen3-8b-lora-stage1

# 8. Run inference on the held-out test prompts.
uv run --extra training python training/infer.py \
    --adapter-dir outputs/qwen3-8b-lora-stage1/final \
    --prompt-jsonl cache/dataset_stage1/test_prompt.jsonl \
    --out cache/predictions_stage1_test.parquet

# 9. Score: overall, by ticker, by year.
uv run python scripts/07_evaluate.py \
    --predictions cache/predictions_stage1_test.parquet
```

Stage 1 must-prove (from the plan): direction accuracy beats majority baseline by
≥3pp and calibration is monotonic. Otherwise re-examine labels before scaling up.

### Stage 2 — 25-ticker pilot + probabilistic outputs + backtest

```bash
# 10. Ingest + label 25 sector-diverse tickers × 2014–2024. Reuses Stage-1 cache.
uv run python scripts/08_ingest_stage2.py

# 11. Rebuild dataset JSONL on the Stage-2 events file.
uv run python scripts/06_build_dataset.py \
    --events cache/stage2_events_2014_2024.parquet \
    --out-dir cache/dataset_stage2 \
    --train-end 2023-01-01 --val-end 2024-01-01 --test-end 2025-01-01

# CUDA box: train, then probabilistic inference (replaces greedy infer.py for Stage 2+).
uv sync --extra training
uv run --extra training python training/train_qwen.py \
    --data-dir cache/dataset_stage2 \
    --output-dir outputs/qwen3-8b-lora-stage2

uv run --extra training python training/infer_probs.py \
    --adapter-dir outputs/qwen3-8b-lora-stage2/final \
    --prompt-jsonl cache/dataset_stage2/test_prompt.jsonl \
    --out cache/predictions_stage2_test.parquet

# 12. Eval with probs (auto-detected): adds AUC, Brier, ECE, sector strat.
uv run python scripts/07_evaluate.py \
    --predictions cache/predictions_stage2_test.parquet

# 13. Event-driven backtest. Tune thresholds on val first, then quote test numbers.
uv run python scripts/09_backtest.py \
    --predictions cache/predictions_stage2_test.parquet \
    --out-dir cache/backtest_stage2_test
```

Stage 2 must-prove (from the plan): per-sector IC > 0 in ≥7/11 GICS, OOT
Spearman ≥ 0.05, gross Sharpe > 1.0. Backtest defaults (edge=0.20, material=0.50,
1% sizing, 15 bps cost) are starting points — tune on val before quoting test.

## Repo layout

```
data/
  edgar_ingest.py   # SEC EDGAR 8-K filings + EX-99 press-release exhibits
  prices.py         # yfinance daily OHLCV
  factors.py        # Kenneth French FF3 + Momentum
  labels.py         # FF4 abnormal returns, CAR windows, idio-vol thresholds, direction/materiality
  membership.py     # Stage-1 ticker list + sector map (placeholder for full PIT membership)
  pipeline.py       # idempotent ingest + label orchestrator
training/
  prompt.py         # frozen prompt template, parse_response, chat-message builders
  dataset.py        # temporal split with 7-day embargo, earnings carve-out, JSONL writer
  train_qwen.py     # Unsloth QLoRA loop (CUDA-only)
  infer.py          # Stage 1: greedy decode + regex parse → hard labels (CUDA-only)
  infer_probs.py    # Stage 2+: first-token softmax over labels → full prob distributions
eval/
  leakage_canary.py # text-vs-CAR[-1,0] canary; sample-size-aware verdict
  metrics.py        # auto-detects probs; direction acc, Spearman IC, AUC, Brier, ECE
  backtest.py       # event-driven backtest (daily-bar entry/exit, fixed costs)
scripts/            # numbered, run in order
cache/              # ingested + labeled parquet, dataset JSONL, predictions (gitignored)
outputs/            # LoRA adapter checkpoints (gitignored)
docs/
  plan.md           # full design, methodology, stage gates, exit rules
```

## What's not built yet

- Point-in-time S&P 500 membership (needed at Stage 3+ for survivorship-free expansion past the curated 25-name list).
- Earnings calendar tagging beyond Item 2.02 detection (no 8-K filed → still untagged).
- Intraday entry/exit and the plan's σ-stop / +3σ-target rules (Stage 2 backtest uses daily bars and a fixed 10-day cap).
- Kelly sizing + sector/position caps (Stage 2 uses equal-weight 1% per trade).
- Borrow cost on shorts.

See `docs/plan.md` for the full design and what each stage must prove.
