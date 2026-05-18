# POC: Fine-tune an LLM to Predict Market-Moving News (S&P 500)

## Context

You want to build an automated workflow that reads news about a public company and decides whether to trade. Before the trading layer, the central question is: **can a small fine-tuned LLM learn the mapping from news text → next-day abnormal return for a stock?** This plan is the POC that answers that question — end-to-end, but scoped to be cheap to abandon if the signal isn't there.

Your initial plan had three things to fix before building:

1. **"Qwen3.5 9B" doesn't exist.** Real options: Qwen2.5-7B or Qwen3-8B. We'll use Qwen3-8B base.
2. **Raw price change is the wrong label.** It's confounded by market beta and macro moves. We'll use factor-adjusted abnormal return (CAR) normalized by each stock's idiosyncratic volatility.
3. **"Confidence" alone isn't tradeable.** You need direction too — confirmed in our exchange. Two-headed output: direction (down/neutral/up) + materiality (binary).

Plus four methodology pillars that determine whether the result is real or a leakage artifact: point-in-time index membership, exact news timestamps, earnings exclusion, and strict temporal train/val/test splits with embargo.

## Decisions (confirmed with you)

| Decision | Choice |
|---|---|
| GPU | Single 24GB (4090 / A5000) |
| Model output | Direction (3-class) + materiality (binary) |
| News source | Free: FNSPID + SEC EDGAR 8-Ks |
| Earnings days | Excluded from training, evaluated separately |
| Model | **Qwen3-8B base**, QLoRA r=16, max_seq=2048 |
| Scope | Staged: 5 → 25 → 100 tickers (excl. mega-cap top 7) |

## Architecture

```
data/
  membership.py        # point-in-time S&P 500 (Wikipedia revisions + iShares IVV)
  prices.py            # yfinance daily + Kenneth French factors
  news_ingest.py       # FNSPID + EDGAR 8-Ks → unified parquet
  events.py            # dedupe, ticker-link, timestamp normalize (ET/UTC)
  labels.py            # FF4 abnormal return, CAR windows, idio-vol thresholds
  leakage_canary.py    # CAR[-1d,0] predictability test
training/
  prompt.py            # frozen prompt template (no future info)
  dataset.py           # temporal splits with 5-day embargo
  train_qwen.py        # Unsloth QLoRA, two-headed loss
eval/
  metrics.py           # Spearman IC, ECE, sector-stratified
  backtest.py          # event-driven, realistic costs, hybrid exit
```

## Label Engineering

- **Abnormal return**: Fama-French 4-factor (MKT/SMB/HML/MOM). Factor loadings estimated on rolling `[-250d, -30d]` window with 30-day gap before event.
- **Primary label window**: `CAR[0, +1d]` from publish-time-aware entry (pre-open → that day O→C + overnight; intraday → publish→close; after-close → next session O→C).
- **Threshold normalization**: per-stock idiosyncratic vol (60-day rolling stdev of AR).
  - Direction: down if CAR < −0.75·σ, up if CAR > +0.75·σ, else neutral.
  - Materiality: |CAR| > 1.5·σ.
- **Same-day multiple news**: keep all, downweight loss by 1/N(day), expose `same_day_news_count` in prompt.

## Bias Mitigation (non-negotiable)

1. **Survivorship**: point-in-time S&P 500 via Wikipedia revision history + iShares IVV holdings CSVs. Include delisted/acquired/bankrupt tickers active during their tenure.
2. **Look-ahead**: keep exact publish timestamps (FNSPID has them); normalize timezones to UTC; label window must start strictly after `publish_ts`.
3. **Earnings confound**: tag every event with `is_earnings_day` (±1 trading day around earnings via EDGAR 8-K Item 2.02 + yfinance calendar). **Drop these from training**; keep them in a separate eval bucket.
4. **Macro tags**: `is_fomc_day`, `vix_bucket`, `spy_trend` exposed in prompt and used for stratified eval.
5. **Leakage canary**: before any training, verify `CAR[-1d, 0]` has no predictive structure for the same model setup. If it does, the pipeline is leaking.

## Prompt Template (frozen)

```
SYSTEM: You are a financial event reaction predictor.

INPUT:
Ticker: {ticker}
Sector: {gics_sector}
Date: {publish_date}
Time relative to market: {pre_open|intraday|after_close}
Market regime: {vix_bucket}, {spy_trend}
Same-day news count for ticker: {n_same_day}
Days since last material event: {days_since}
Recent 5d return: {ret_5d}
Recent 5d abnormal return: {car_5d}
Headline: {headline}
Body (first 1500 chars): {body_truncated}

OUTPUT:
Direction: {down|neutral|up}
Materiality: {material|immaterial}
```

No analyst targets, no fundamentals unless point-in-time, no future prices.

## Training Setup (Unsloth, 24GB)

```
model:            unsloth/Qwen3-8B (base)
quantization:     4-bit nf4
lora_r:           16
lora_alpha:       32
lora_dropout:     0.05
target_modules:   all linear (q,k,v,o,gate,up,down)
max_seq_length:   2048
learning_rate:    2e-4 (cosine, warmup 3%)
batch_size:       2 × grad_accum 8  (effective 16)
epochs:           2
optimizer:        adamw_8bit
loss:             CE(direction) + BCE(materiality), equal-weight
grad_checkpoint:  on (Unsloth)
```

Direct prediction first. If it works, add STaR-style CoT later (rationales generated conditioned on the known label).

## Evaluation

- **Temporal split** (no shuffle, 5-day embargo between segments):
  - Train: 2014-01 → 2021-12
  - Val:   2022-01 → 2022-12
  - Test:  2023-01 → 2024-12 (touch once)
- **Metrics**:
  - Headline: Spearman ρ between `P(up) − P(down)` and realized CAR on OOT test.
  - Direction: balanced accuracy, macro-F1.
  - Materiality: AUC.
  - Calibration: Brier + ECE.
  - Stratified by sector, by month, by regime (VIX bucket).
- **Backtest** (event-driven, hand-rolled):
  - Entry: publish_ts + 60s (RTH) or next open (otherwise).
  - Exit: hybrid — next material event same ticker OR −2σ stop OR +3σ target OR 10-day hard cap.
  - Costs: 0.5 bps commission, slippage = max(5 bps, 1/√(ADV/order)·k), spread 1–3 bps RTH, borrow 25 bps annual for shorts.
  - Sizing: 0.25× Kelly using calibrated P × expected magnitude, capped at 2%/name, 10%/sector, ±30% net.

## Stage Gates (kill criteria are as important as go criteria)

| Stage | Scope | Must prove | Kill if |
|---|---|---|---|
| 0 — Infra | 1 ticker, 50 events | E2E runs, leakage canary clean | `CAR[-1d,0]` predicts → fix pipeline |
| 1 — Tiny | 5 tickers, ~500 events | Dir-accuracy beats majority by ≥3pp, calibration monotonic | At or below baseline → re-examine labels |
| 2 — Mid | 25 tickers, ~4K events | Per-sector IC > 0 in ≥7/11 GICS; OOT Spearman ≥ 0.05; gross Sharpe > 1.0 | None of the above |
| 3 — Full POC | 100 tickers, ~15K events | Net-of-cost Sharpe > 0.5, stable across 2 OOT years, capacity > $5M | Single-quarter dependence or net Sharpe < 0 |
| 4 — Full SP500 | ~490 tickers, ~75K events | Same metrics hold | Sub-linear capacity scaling fails |

Pilot tickers for Stage 1: JPM, XOM, WMT, MRK, BA (sector-diverse, liquid).

## Top Risks (ranked)

1. **Timestamp leakage** — aggregator vs wire times, TZ confusion. Canary: `CAR[-1d, 0]` predictability.
2. **Earnings confound** — already mitigated by exclusion.
3. **Macro regime dominance** (2020, 2022) — FF4 helps; regime-stratified eval to catch failures.
4. **News duplicates** — dedupe via MinHash/embedding similarity; keep earliest timestamp.
5. **Survivorship** — point-in-time membership.
6. **Alpha decay toward present** — expect 2023–2024 to be harder than 2014–2018.
7. **Ticker memorization** — secondary eval on ticker-disjoint holdout; sometimes mask ticker name in training.
8. **Class imbalance** (most news immaterial) — focal loss or oversample material events.
9. **Costs destroy edge** — first/last 15min excluded for pilot; realistic slippage.

## Verification (end-to-end smoke test before any training)

1. Build FNSPID + EDGAR ingestion for **one ticker (JPM)** and **one year (2019)**.
2. Run `leakage_canary.py` — train a dummy linear classifier on the prompt embeddings predicting `CAR[-1d, 0]`. AUC should be near 0.5.
3. Run the full label pipeline; manually spot-check 10 known-material events (e.g., JPM 2019 trading desk news) for correct direction and materiality.
4. Run 1 epoch on the tiny set with Unsloth to confirm loss decreases and VRAM fits.
5. Only then proceed to Stage 1 (5 tickers).

## Critical Files to Create

- [data/labels.py](data/labels.py) — FF4 CAR, idio-vol thresholds, leakage canary
- [data/news_ingest.py](data/news_ingest.py) — FNSPID + EDGAR ingestion, timestamp normalization
- [data/membership.py](data/membership.py) — point-in-time S&P 500
- [training/train_qwen.py](training/train_qwen.py) — Unsloth QLoRA loop, two-headed loss
- [eval/backtest.py](eval/backtest.py) — event-driven backtest with hybrid exit
- [eval/metrics.py](eval/metrics.py) — Spearman IC, ECE, stratified metrics

## Reuse / Tooling

Everything is new (working directory is empty). External tools to lean on:
- **Unsloth** for QLoRA training loop (handles 4-bit, gradient checkpointing, target modules).
- **HuggingFace `datasets`** for FNSPID.
- **`yfinance`** for OHLCV + earnings calendar.
- **Kenneth French Data Library** for factor returns (CSV download).
- **`statsmodels`** for rolling factor regressions.
- **`vectorbt` or hand-rolled** for backtest (hand-rolled is cleaner for event-driven).

## What I'd Add to Your Original Plan

1. **Factor-adjusted CAR over raw price change** — the single biggest correctness fix.
2. **Idio-vol normalized thresholds** — a 2% move means different things in PG vs TSLA.
3. **Point-in-time S&P 500 + delisted tickers** — without this you train on survivors.
4. **Leakage canary as a Stage-0 gate** — most "alpha" in this kind of work is leakage; build the detector first.
5. **Two-headed output (direction + materiality)** — robust at 15K samples; clean trade gating.
6. **Hybrid exit rule** — your "no time window" stance is honored on entry; exit needs bounded risk via vol-stop + a 10-day fail-safe.
7. **Stage gates with explicit kill criteria** — so you don't sink weeks into a path the data won't support.
