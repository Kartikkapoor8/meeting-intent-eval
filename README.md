# meeting-intent-eval

A Prime Intellect verifiers environment that tests whether a model can pull the *real* action items out of a messy meeting transcript.

The headline result: Claude Opus 4.6 hits **pass@1 = 0.887** on 10 hand-authored synthetic transcripts. We then ran the same model on 3 real public earnings call transcripts. Pass@1 dropped to **0.000**. Same model, same rubric, same prompt. See [REPORT.md](REPORT.md) for the full writeup.

## What's the task

The model gets a meeting transcript. It has to output JSON listing only the real, committed action items. Casual ideas, hedged statements, parking-lot items, and "we should think about that" stuff all count as traps.

```
{"action_items": [{"owner": "Devon", "task": "...", "due": "Friday"}]}
```

## Real vs Synthetic

Same Claude Opus 4.6, same rubric, same system prompt.

| Dataset | Source | Transcripts | Samples each | pass@1 | mean F1 | format compliance |
|---|---|---|---|---|---|---|
| Synthetic | hand-authored project meetings | 10 | 64 | **0.887** | 0.965 | 1.000 |
| Real | public earnings calls (IIPR, NVDA, JPM) | 3 | 16 | **0.000** | 0.000 | 1.000 |

Format compliance is 1.0 in both runs. The model always returns clean JSON. On the real transcripts it returns `{"action_items": []}` on 47 out of 47 valid samples. It refuses to extract anything. The full failure breakdown is in [REPORT.md](REPORT.md).

## How the rubric works

Fully deterministic. No LLM judge.

1. Parse the model's JSON. Bad JSON gets a 0.
2. Match each predicted item to a ground-truth item by canonical (owner, due). Owner is lowercased and reduced to the first name. Due dates normalize: "Wednesday EOD" and "by Wednesday" both become "wednesday"; "before 4pm", "now", "asap" all become "today"; "end of week" becomes "friday".
3. For a match to count, the predicted task field also has to contain at least one anchor keyword for that ground-truth item. Stops the model from guessing right owners and dates with totally wrong tasks.
4. Reward = F1 over matched items. Pass = F1 ≥ 0.99 (effectively all-or-nothing per transcript).

Pass@k is computed with the unbiased estimator from the Codex paper.

## How to run it

```bash
pip install verifiers
export ANTHROPIC_API_KEY=sk-ant-...
python3 run_eval.py
```

Run from the repo root. `pip install verifiers` pulls in `anthropic` and `datasets` automatically.

Defaults: Claude Opus 4.6, 64 samples per transcript, T=1.0, 8 concurrent calls, $18 budget cap. Override with env vars: `EVAL_MODEL`, `EVAL_N`, `EVAL_T`, `EVAL_CONCURRENCY`, `EVAL_BUDGET`. Limit to specific transcripts with `EVAL_ONLY=transcript_id_1,transcript_id_2`.

To run just the 3 real earnings calls (cheaper, ~$8):

```bash
EVAL_ONLY="real_earnings_001,real_earnings_002,real_earnings_003" \
EVAL_N=16 EVAL_TAG=real_only python3 run_eval.py
```

To re-score the saved samples against a modified rubric without spending more API budget:

```bash
python3 rescore.py
```

## Synthetic results breakdown

10 transcripts × 64 samples × Claude Opus 4.6.

| | pass@1 | pass@8 | pass@32 | mean F1 | format compliance |
|---|---|---|---|---|---|
| aggregate | 0.887 | 0.900 | 0.900 | 0.965 | 1.000 |

Two transcripts drove all the failure:
- `eng_standup_003`: 0/64 passed. The model misses a commitment expressed only through dialogue.
- `support_escalation_007`: 56/64 passed. The model occasionally over-extracts a borderline conditional.

The other 8 transcripts: 64/64 each.

## Real results breakdown

3 real public earnings call transcripts × 16 samples × Claude Opus 4.6. Each transcript is at least 30 minutes long, taken from The Motley Fool's free transcripts.

| transcript | source | passing/n | pass@1 |
|---|---|---|---|
| `real_earnings_001` | IIPR Q1 2025 (small cap REIT) | 0/16 | 0.000 |
| `real_earnings_002` | NVDA Q3 FY26 (megacap tech) | 0/16 | 0.000 |
| `real_earnings_003` | JPM Q1 2026 (megacap finance) | 0/15 | 0.000 |

Across all 47 valid completions, the model returned `{"action_items": []}` every time. Format was always valid JSON. The failure mode is over-conservative refusal, not hallucination. Full analysis in [REPORT.md](REPORT.md).

## Repo structure

```
environments/meeting_intent/   # the verifiers env
├── meeting_intent.py          # SingleTurnEnv + Rubric + load_environment
├── pyproject.toml             # ready to publish to the Hub
├── test_rubric.py             # 77 deterministic rubric tests
└── data/                      # 10 synthetic transcripts + 3 real earnings calls
run_eval.py                    # async sampling + scoring + pass@k
rescore.py                     # re-score saved samples
results/                       # model outputs and aggregated numbers
REPORT.md                      # the writeup
```

## License

MIT.
