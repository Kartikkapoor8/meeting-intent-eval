# Meeting Intent: One-Page Proof

## TL;DR

Frontier models miss commitments that get made through dialogue, not single statements. Confirmed on a real earnings call: Opus 4.6, Sonnet 4.6, and Haiku 4.5 all returned empty arrays 32 out of 32 times under the default "extract action items" prompt. Calibrated prompt fixes one of four real transcripts on Opus. Failure is real but smaller than the original "universal 0/0" framing claimed.

## The Failure Mode

Single-statement commitments look like "I'll send the doc tomorrow morning." One person, one explicit verb, one date. Models catch these.

Dialogue-confirmation commitments emerge across multiple turns. Person A asks, person B offers something hedged, person A confirms. The commitment is real but no single line contains it.

This matters because most workplace commitments in real meetings look like the second kind. Status syncs, customer calls, demos, earnings Q&A. If a model only catches the first kind, it misses most of the actual work.

## The Transcript

From IIPR Q1 2025 earnings call. Connor Mitchell (Piper Sandler analyst) is pushing Ben Regin (CIO) for a monthly modeling breakdown.

```
Connor Mitchell: Okay. And then any rents as well, I think for some
of these, you may receive January payments, but not February or March?

Ben Regin: In terms of what we actually collected?

Connor Mitchell: Yes. Just thinking about a modeling perspective
going forward, what won't be being received for a quarterly
standpoint or even monthly?

Ben Regin: Yes. We can go into the detail offline, but it was roughly
$4.5 million that we collected from the defaulted tenants during
the quarter.
```

## What Should Be Labeled

```
owner: Ben
task:  Follow up offline with Connor on the monthly breakdown of rent
       collected from defaulted tenants.
due:   offline
```

Evidence quote: "We can go into the detail offline, but it was roughly $4.5 million that we collected from the defaulted tenants during the quarter."

### Why this is a commitment

Connor asked a specific question with a specific use case (modeling). Ben acknowledged the question, gave a partial answer, and explicitly redirected the rest to a follow-up channel ("we can go into the detail offline"). On an earnings call, "offline" means a one-on-one analyst follow-up. That is a real obligation. Ben is on the hook to give Connor a monthly breakdown after the call. A reasonable IR analyst, sell-side analyst, or chief of staff reading this transcript would all extract this same item. The commitment is not arbitrary. It has a named owner (Ben), a named counterparty (Connor), a specific topic (monthly defaulted-tenant rent collection), and a defined channel (offline follow-up).

## What The Model Did

Strict prompt. Same model, same transcript, multiple samples. The exact prompt told the model to reject hedged language and demanded "specific person, specific thing, specific time."

| Model | Samples | Empty arrays | Pass@1 |
|---|---|---|---|
| Claude Opus 4.6 | 16 | 16 | 0.000 |
| Claude Sonnet 4.6 | 8 | 8 | 0.000 |
| Claude Haiku 4.5 | 8 | 8 | 0.000 |
| **Total** | **32** | **32** | **0.000** |

Every single sample returned this:

```json
{"action_items": []}
```

Source files:
- `results/raw_samples_real_only.json` (Opus)
- `results/raw_samples_sonnet46_real.json`
- `results/raw_samples_haiku45_real.json`

Format compliance was 1.0 on every sample. The model is not breaking. It is reading the prompt and deciding the offline follow-up does not count.

## The Contrast

Same transcript, same model (Opus 4.6), calibrated soft-commitment prompt. The prompt tells the model that "follow-ups offered to specific questions" count as commitments and that soft due values like "offline" are valid.

| Prompt | Samples | Empty arrays | Pass@1 |
|---|---|---|---|
| Strict (default) | 16 | 16 | 0.000 |
| Calibrated softcommit | 8 | 0 | **1.000** |

Sample 0 under calibrated:

```json
{
  "action_items": [
    {
      "owner": "Ben Regin",
      "task": "Provide Connor Mitchell with detailed breakdown of rent collected from defaulted tenants during Q1 (roughly $4.5 million)",
      "due": "offline"
    }
  ]
}
```

This is exactly the ground truth. 8 of 8 samples extracted it. Source: `results/ablations/softcommit_full_raw.json`.

For a within-transcript single-vs-dialogue contrast on a longer real meeting, see `real_client_001.txt`. Under the strict prompt, Opus catches "Client: tomorrow morning, I'll send to the two of you the 57 page document" (single declarative line, mean F1 = 0.633) but misses the timeline-and-reconvene commitment that gets built across three Engineer turns plus a Client confirmation. Same model, same prompt, different commitment shape, different result. Source: `results/raw_samples_client_only.json`.

## Why This Is A Real Failure (Not Measurement Error)

What I ruled out:

- **Prompt design.** Tested the calibrated soft-commitment prompt that explicitly tells the model dialogue-confirmation counts. Strict failure persists, calibrated cracks 1 of 4 real transcripts. Half the failure was prompt-label mismatch, half was something else. Source: `results/ABLATIONS.md`.
- **Rubric strictness.** Output is empty under strict, so no rubric setting could match. Even a "any keyword in any field" rubric scores zero on an empty array. The failure is upstream of scoring.
- **Cross-model artifact.** Tested Opus 4.6, Sonnet 4.6, Haiku 4.5 on the same transcript with the same prompt. All three returned empty 8-of-8 or 16-of-16. Pattern is universal across the Anthropic frontier line.
- **Harness bug.** Earlier review flagged a sample-averaging concern (errored samples diluting the mean). Diagnostic confirmed the 0/0 result on `real_earnings_001` is real, not an averaging artifact. Source: `results/DIAGNOSTIC.md`.

Honest current status: under default prompts, the failure is real and reproducible across three frontier models. Under calibrated prompts, one of four real transcripts fully recovers and the other three have a mix of rubric format mismatch and a residual recall ceiling on weak dialogue cues (e.g. on `real_earnings_003`, the "Jeremy revisit consumer deposits next quarter" item was never extracted across 8 calibrated samples). The failure is smaller than the original "0/95 universal" framing but it is not zero.

## What This Means

The pattern: default-trained extraction models treat hedged, distributed, multi-turn commitments as conversational filler instead of as obligations. To fix this at training time you would need labeled data of meeting transcripts where the commitment is explicitly marked across the dialogue chain, not just on the final declarative line.

## Verification Status

**Thesis status: PARTIALLY VERIFIED as of 2026-05-11.** Universal failure under default strict prompts (32/32 cross-model). Calibrated prompt recovers 1 of 4 real transcripts on Opus. See [ABLATIONS.md](ABLATIONS.md) for full ablation methodology.
