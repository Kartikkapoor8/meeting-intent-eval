# Meeting Intent Eval Results

**Model:** Claude Opus 4.6
**Dataset:** 10 synthetic meeting transcripts, hand-authored
**Method:** 64 samples per transcript at temperature 1.0. Score is F1 over correctly extracted action items. Pass = F1 ≥ 0.99. Pass@k uses the Codex unbiased estimator.
**Cost:** $13.75. **Format:** Prime Intellect verifiers.

## Headline

Pass@1 = **0.887**. Opus 4.6 nails 8 of 10 transcripts on every single sample. Two transcripts fail in interesting ways:

| Transcript | passing/64 | pass@1 | pass@32 |
|---|---|---|---|
| 8 standard transcripts | 64/64 each | 1.000 | 1.000 |
| eng_standup_003 | **0/64** | **0.000** | 0.000 |
| support_escalation_007 | 56/64 | 0.875 | 1.000 |
| **aggregate (10)** | | **0.887** | **0.900** |

The pass@1 to pass@32 gap is tiny (1.3 points) because the eng_standup failure is deterministic. Sampling more does not help.

## The interesting failure

Two action items in eng_standup_003. Opus catches one and misses the other in all 64 samples.

The one it catches:
> Maddie: I'll have the runbook done by Thursday.

Single speaker, explicit task, explicit date. Easy.

The one it misses:
> Chen: I'm kind of stuck.
> Raj: I can pair with you on it after this if you want.
> Chen: yeah, that'd help.
> Raj: cool. Right after standup, your zoom or mine?
> Chen: yours.
> Raj: done.

No one says "I commit to X by Y." The commitment lives in the back-and-forth: hedged offer, acceptance, logistics swap, closing word. The model reads "I can pair... if you want" as optional and stops.

This pattern is everywhere in real meetings. Most commitments do not arrive as clean declarative sentences. They emerge through dialogue.

## The other failure

support_escalation_007 has 3 real action items. Opus gets all 3 right in 56 of 64 samples. In the other 8 it adds a fourth:

> Hana: ok I'll set expectations for end of day and update them tomorrow if needed.

That sentence has a real verb and a real timeframe but the action is contingent. ~12% of the time Opus extracts it as a tracked deliverable. Stochastic over-extraction on judgment calls.

## What this means for the pitch

The 88.7% pass@1 number understates the opportunity. Two reasons.

**One.** The eng_standup failure is 100% reproducible. Sampling does not fix it. Prompting probably does not either (the system prompt already tells the model to handle hedged language). This is the kind of thing you only get from training.

**Two.** The dataset is biased toward easy cases. We hand-authored these and drifted toward explicit declarations because they're easier to ground-truth. Real meetings have multi-turn-confirmation patterns in nearly every transcript, not 1 in 10. Pass@1 on real data should be much lower.

The training opportunities, in priority order:
1. Recognize commitments formed across multiple speakers (offer + accept + logistics).
2. Treat conditional language conservatively unless the conversation closes the conditional.
3. Stop at the dialogue boundary, not at a single utterance.

## What's in the repo

- `environments/meeting_intent/`: the verifiers env, ready to publish to the Hub.
- `run_eval.py`: async sampling against the Anthropic API.
- `rescore.py`: re-score saved samples without spending API budget.
- `results/raw_samples*.json`: all 640 model outputs for reproducibility.
- `results/final_aggregate.json`: the table above as JSON.
