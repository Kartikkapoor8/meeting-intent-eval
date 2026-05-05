# meeting-intent-eval

A small Prime Intellect verifiers environment that tests whether a model can pull the *real* action items out of a messy meeting transcript.

The interesting result: Claude Opus 4.6 hits **pass@1 = 0.887** but fails one transcript on every single sample. The failed transcript is the only one where the commitment lives in dialogue ("I can pair with you" → "yeah" → "your zoom or mine?" → "yours" → "done") instead of in a single declarative sentence. See [REPORT.md](REPORT.md) for the full writeup.

## What's the task

The model gets a meeting transcript. It has to output JSON listing only the real, committed action items. Casual ideas, hedged statements, parking-lot items, and "we should think about that" stuff all count as traps.

```
{"action_items": [{"owner": "Devon", "task": "...", "due": "Friday"}]}
```

## How the rubric works

Fully deterministic. No LLM judge.

1. Parse the model's JSON. Bad JSON gets a 0.
2. Match each predicted item to a ground-truth item by canonical (owner, due). Owner is lowercased and reduced to the first name. Due dates normalize: "Wednesday EOD" and "by Wednesday" both become "wednesday"; "before 4pm", "now", "asap" all become "today"; "end of week" becomes "friday".
3. For a match to count, the predicted task field also has to contain at least one anchor keyword for that ground-truth item. Stops the model from guessing right owners and dates with totally wrong tasks.
4. Reward = F1 over matched items. Pass = F1 ≥ 0.99 (effectively all-or-nothing per transcript).

Pass@k is computed with the unbiased estimator from the Codex paper.

## How to run it

```bash
pip install verifiers anthropic datasets
export ANTHROPIC_API_KEY=sk-ant-...
python3 run_eval.py
```

Defaults: Claude Opus 4.6, 64 samples per transcript, T=1.0, 8 concurrent calls, $18 budget cap. Override with env vars: `EVAL_MODEL`, `EVAL_N`, `EVAL_T`, `EVAL_CONCURRENCY`, `EVAL_BUDGET`. Limit to specific transcripts with `EVAL_ONLY=transcript_id_1,transcript_id_2`.

To re-score the saved samples against a modified rubric without spending more API budget:

```bash
python3 rescore.py
```

## Results from the run in this repo

| | pass@1 | pass@8 | pass@32 | mean F1 | format compliance |
|---|---|---|---|---|---|
| Claude Opus 4.6, 10 transcripts × 64 samples | 0.887 | 0.900 | 0.900 | 0.965 | 1.000 |

Two transcripts drove all the failure:
- `eng_standup_003`: 0/64 passed. The model misses a commitment expressed only through dialogue.
- `support_escalation_007`: 56/64 passed. The model occasionally over-extracts a borderline conditional.

The other 8 transcripts: 64/64 each. Full per-transcript table and analysis in [REPORT.md](REPORT.md).

## Repo structure

```
environments/meeting_intent/   # the verifiers env
├── meeting_intent.py          # SingleTurnEnv + Rubric + load_environment
├── pyproject.toml             # ready to publish to the Hub
├── test_rubric.py             # 77 deterministic rubric tests
└── data/                      # 10 transcripts + ground truth JSON
run_eval.py                    # async sampling + scoring + pass@k
rescore.py                     # re-score saved samples
results/                       # 640 model outputs + aggregated numbers
REPORT.md                      # the writeup
```

## License

MIT.
