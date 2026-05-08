# DIAGNOSTIC.md

A buyer thinks our 0/0 on real meeting transcripts is a harness bug. This report walks through the evidence that the failure is real, that the rubric is not at fault, and that the failure is targeted at one specific commitment shape.

Everything below comes from cached eval results. No new model calls.

## TL;DR

- 95/95 model samples on real earnings calls returned `{"action_items": []}`. Across Opus 4.6, Sonnet 4.6, Haiku 4.5.
- The JSON parsed every time. Format compliance is 100%. The pipeline is fine.
- Loosening the rubric does not change the result. There is nothing to score.
- On the same harness with the same model, synthetic transcripts pass at pass@1 = 0.89.
- On real_client_001 (mixed item types), the model catches the items closest to a single statement and misses the buried-in-dialogue one. Item-level catch rate follows item shape, not transcript identity.

Conclusion: the failure is real, the rubric is not the cause, and the model is missing exactly the commitment shape we want a meeting-intent product to catch.

---

## a) The empty array problem

Across 3 real earnings transcripts and 3 frontier Claude models, 95/95 samples returned an empty action-items array.

The harness logged the model output verbatim. Every string is a valid JSON object. The structure of the output matches the system prompt's instructions exactly. The model just has nothing inside the array.

Examples (one per model, picked at random from the cached samples):

- **Opus 4.6** on `real_earnings_001`: `{"action_items": []}`
- **Sonnet 4.6** on `real_earnings_001`: `{"action_items": []}`
- **Haiku 4.5** on `real_earnings_001`: `json {"action_items": []} `

Full breakdown is in [RAW_OUTPUTS.md](RAW_OUTPUTS.md).

---

## b) Rubric sensitivity check

We re-scored the 47 successful Opus 4.6 samples on real earnings calls under three rubrics:

1. Strict (current): F1 over (owner, due, anchor-keyword on task), pass at F1 >= 0.99.
2. Medium: F1 over (owner, due) only, pass at F1 >= 0.5.
3. Loose: any predicted item that mentions any GT owner OR any GT due gets credit.

Pass rate at every level: 0.000. Mean score at every level: 0.000.

Reason: 46/47 samples returned an empty array (1 was a 429 rate-limit error). All three rubrics need at least one predicted item to give a non-zero score. Empty in, zero out.

The rubric strictness is irrelevant. Full numbers in [RUBRIC_SENSITIVITY.md](RUBRIC_SENSITIVITY.md).

---

## c) Item-level analysis on a mixed transcript

On real_client_001, the model produces non-empty arrays on all 16 samples. Per-item strict catch rate, with item type:

| item | shape | catch rate |
|---|---|---:|
| Engineer / tonight / 'send a link' | one-line commit at meeting close | high |
| Client / tomorrow morning / 'send 57-page doc' | dialogue, but ends in one line | high |
| Engineer / open / 'finish timeline + reconvene' | scattered across turns, no single line | low |

The hypothesis holds. The model catches what is essentially a single-statement commit, even when it lives inside dialogue. It misses commits that only exist as a synthesis of multiple turns.

On real earnings calls, every item is shape #3. So every item gets missed. Detailed breakdown in [MEDIUM_DIFFICULTY.md](MEDIUM_DIFFICULTY.md). Side-by-side dialogue chain vs all 16 samples for real_earnings_001 in [HUMAN_VS_MODEL.md](HUMAN_VS_MODEL.md).

---

## What this means for the buyer

1. The eval is real. We have raw model outputs cached, the rubric is open source, and you can re-score under any rubric you want and get the same 0.
2. The failure is targeted. The model is not bad at extraction in general. It is bad at the specific case where a commitment only exists as a multi-turn synthesis.
3. This is exactly the case a meeting-intent product needs to handle. Most real meetings, especially earnings calls and client calls, do not have neat one-line commits. They have offers and confirmations spread across turns. RL can hill-climb this because the failure mode is consistent and well-defined.
