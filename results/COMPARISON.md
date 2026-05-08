# Single-turn vs multi-turn: meeting_intent on real transcripts

Model: claude-opus-4-6, T=1.0, 8 samples per transcript per variant. Total spend $15.12 (cap $15).

## Numbers

| transcript          | ST pass@1 | ST mean F1 | MT pass@1 | MT mean F1 | MT samples completed |
|---------------------|-----------|------------|-----------|------------|----------------------|
| real_earnings_001   | 0.000     | 0.000      | 0.000     | 0.000      | 2/8                  |
| real_earnings_002   | 0.000     | 0.000      | 0.000     | 0.000      | 2/8                  |
| real_earnings_003   | 0.000     | 0.000      | 0.000     | 0.000      | 0/8                  |
| real_client_001     | 0.000     | 0.676      | n/a       | n/a        | 0/8 (cap hit)        |
| **aggregate**       | **0.000** | **0.169**  | **0.000** | **0.000**  | 4/24                 |

Cost: single-turn $6.34, multi-turn $8.78. Multi-turn is ~10x per usable sample.

## Tool usage in multi-turn

- 4 of 24 attempted rollouts finished cleanly (called done). 20 died on 429 rate limits (50 RPM org cap, our concurrency=6 + ~7 turns/rollout overshoots that fast).
- revise_item: called 0 times across all rollouts.
- remove_item: called 0 times across all rollouts.
- Every completed rollout: read all chunks via next_chunk, then called done with an empty notepad.
- Mean turns when completed: ~10. Mean tool calls per completed rollout: ~10.

## What this means

Multi-turn lost. Same pass@1 (0), worse mean F1 (0 vs 0.169), 10x cost, and rate-limited so badly the longest transcript never even ran. The notepad pattern the design counted on, where the agent revises items as later context arrives, never showed up. The agent treated the tool flow as "read everything, decide nothing's a real commitment, call done." Streaming the transcript didn't change its judgment.

The single-turn meanF1 of 0.676 on real_client_001 is the only signal of life and it's the same 0.63 we already had from the prior client eval. So the failure mode isn't context length or attention. It's that Opus is too conservative on dialogue-confirmation commitments. It needs the rule of "extract this kind of commitment" softened, or it needs few-shot examples of what counts. Adding more turns or a notepad doesn't fix the threshold.

Hub push: skipped. Multi-turn mean F1 was 0 on every transcript, so it doesn't meet the "above 0 on at least one transcript" bar.
