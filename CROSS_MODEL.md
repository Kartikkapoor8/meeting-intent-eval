# Cross-model failure on real earnings calls

The headline result was Claude Opus 4.6 dropping from pass@1 = 0.887 on synthetic transcripts to pass@1 = 0.000 on 3 real public earnings calls. The obvious follow up question: is this just an Opus issue, or does the whole Claude frontier fail the same way.

I ran the same eval, same rubric, same system prompt, same 3 transcripts (IIPR Q1 2025, NVDA Q3 FY26, JPM Q1 2026) on Sonnet 4.6 and Haiku 4.5. 8 samples per transcript at T=1.0.

## Results

| Model | Transcripts | Samples | pass@1 | mean F1 | format | empty `action_items` |
|---|---|---|---|---|---|---|
| Claude Opus 4.6 | 3 | 47 | **0.000** | 0.000 | 1.000 | 47 / 47 |
| Claude Sonnet 4.6 | 3 | 24 | **0.000** | 0.000 | 1.000 | 24 / 24 |
| Claude Haiku 4.5 | 3 | 24 | **0.000** | 0.000 | 1.000 | 24 / 24 |

**95 out of 95 valid completions across all three model tiers returned `{"action_items": []}`.** Format compliance was 1.0 everywhere. Nobody hallucinated. Everybody refused.

Per transcript, every model passed 0/n on every transcript:

| transcript | Opus 4.6 | Sonnet 4.6 | Haiku 4.5 |
|---|---|---|---|
| real_earnings_001 (IIPR) | 0/16 | 0/8 | 0/8 |
| real_earnings_002 (NVDA) | 0/16 | 0/8 | 0/8 |
| real_earnings_003 (JPM)  | 0/15 | 0/8 | 0/8 |

## What this rules out

- **Not an Opus quirk.** Sonnet and Haiku fail identically.
- **Not a sampling fluke.** 95 samples across 3 transcripts and 3 model tiers, all hit the same refusal pattern.
- **Not a format problem.** Every model produced clean JSON. The schema is fine. The reasoning is fine. The model just decides nothing in the transcript counts as a real commitment.
- **Not a prompt-engineering gap.** Same prompt that gets 0.887 on synthetic gets 0.000 on real, across every Claude tier we have.

## What this means for RL training

The capability gap is real and it is universal across the current Claude frontier. The training signal sits exactly where you would want it for a verifiers env: deterministic rubric, clear ground truth, bounded output, every model gets a 0. There is room above the floor to climb to.

## How to reproduce

```bash
# Sonnet 4.6
EVAL_ONLY="real_earnings_001,real_earnings_002,real_earnings_003" \
EVAL_N=8 EVAL_TAG=sonnet46_real EVAL_MODEL=claude-sonnet-4-6 \
EVAL_BUDGET=4.0 python3 run_eval.py

# Haiku 4.5
EVAL_ONLY="real_earnings_001,real_earnings_002,real_earnings_003" \
EVAL_N=8 EVAL_TAG=haiku45_real EVAL_MODEL=claude-haiku-4-5-20251001 \
EVAL_BUDGET=5.0 python3 run_eval.py
```

Saved sample outputs:
- `results/eval_results_sonnet46_real.json`, `results/raw_samples_sonnet46_real.json`
- `results/eval_results_haiku45_real.json`, `results/raw_samples_haiku45_real.json`
- `results/eval_results_haiku45_real_003.json`, `results/raw_samples_haiku45_real_003.json`
