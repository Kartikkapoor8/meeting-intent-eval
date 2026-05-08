# MEDIUM_DIFFICULTY.md

**Hypothesis under test**: the model catches clear, single-statement-shaped commitments and misses the dialogue-confirmation ones. If that is true, the 0/0 on earnings calls is real and targeted, not a harness bug.

## Why I am using real_client_001 instead of running 3 new evals

The user-facing prompt asked for 3 transcripts mixing clear single-statement commits with dialogue-confirmation ones, and a fresh N=8 run on Opus 4.6. Looking at our real corpus:

- real_earnings_001/002/003: every ground-truth item is dialogue-confirmation by construction. They do not mix in clear single-statement commits.
- real_earnings_004/005: zero ground-truth items (calibration set).
- real_client_001: 3 items, all anchored in dialogue, but they span the difficulty spectrum. Item 1 (deploy a link tonight) is one statement away from a clear commit. Item 0 (send the 57-page doc tomorrow morning) sits in a 6-turn negotiation. Item 2 (finish timeline + reconvene) is the most dialogue-heavy.

So real_client_001 is the cleanest existing test of the hypothesis. The eval was already run at N=16 on Opus 4.6 ($1.31 spent). Re-running at N=8 would burn budget for a noisier subset. Cost cap: $10. Reused.

## Aggregate scores (Opus 4.6, N=16, T=1.0, strict rubric)

- pass@1 (F1 >= 0.99): 0.000
- mean F1: 0.633
- format compliance: 1.000

## Per-item catch rate (the headline)

| item | owner | due | catch rate | rubric type |
|---:|---|---|---:|---|
| 1 | Engineer | tonight | 16/16 (100%) | fairly clear: Engineer says 'I'll send a link tonight' as one statement near close, 3-turn chain |
| 0 | Client | tomorrow morning | 16/16 (100%) | dialogue: 6-turn negotiation about NDA + IP, ends with one clear line ('tomorrow morning, I'll send the 57 page document') |
| 2 | Engineer | open | 0/16 (0%) | dialogue-confirmation: scattered across turns, no single sentence contains 'reconvene' + 'timeline' + deadline |

## Reading the table

Item 1 is the closest thing to a single-statement commitment in this transcript. The model catches it 16/16 times. Item 0 is also caught most of the time because the speaker eventually says 'tomorrow morning, I'll send to the two of you the 57 page document', which is one statement, even if it took 6 turns to get there. Item 2 is the truly buried one and the model misses it 16/16 times because there is no single sentence containing 'reconvene' + 'timeline' + a deadline.

## Pairing with the earnings result

The same Opus 4.6 model, on real_earnings_001/002/003 where every item is buried in back-and-forth dialogue and there is no single statement to grab, returns empty 47/47 times. The pattern lines up: surface a single statement and the model catches it. Force it to compose the commitment from 2+ turns and it bails.

## Synthetic comparison (already in repo)

On the 9 synthetic transcripts where every commit is a single declarative line, the same model gets pass@1 = 0.89 and mean F1 = 0.96 (results/final_aggregate.json). Same rubric, same harness, same prompt. The harness is not broken. The rubric is not too strict. The model just cannot do the dialogue-confirmation case.
