# Soft-commitment prompt ablation

## Hypothesis

The strict prompt in `meeting_intent.py` tells the model to reject hedged
language like "maybe", "might", "we'll figure it out", "let's circle back",
and demands a specific person, specific thing, specific time.

The ground truth on real earnings calls includes exactly that kind of
language. "Offline" follow-ups. "Ongoing" commitments. "Next quarter"
promises. "Work to" management language.

Claim from GPT review: the model is not failing the task. It is following
the prompt and rejecting items that the labels say are real. Pass@1 = 0
on real data is partly an artifact of prompt-label mismatch, not a model
limit.

This ablation tests that.

## What changed

Only the system prompt. Same dataset, same rubric, same schema, same
canonicalization, same anchor keywords.

New prompt (in `environments/meeting_intent/meeting_intent_softcommit.py`):

```
You are extracting commitments from a meeting transcript. A commitment
is a follow-up obligation where a specific person has agreed to do a
specific thing. The agreement may be made through dialogue rather than
a single explicit statement.

Soft due values are valid: "offline", "ongoing", "next quarter", "EOD",
"later today", "open", "after the meeting".

Include:
- Hedged but mutual commitments through dialogue
- Follow-ups offered to specific questions ("we can go into that offline")
- Management-style commitments where a leader states the team will work
  toward something

Exclude:
- Generic prepared remarks ("we'll provide updates as we progress")
- Vague aspirations ("hopefully we can")
- Statements where no specific party is taking ownership
```

Run config: claude-opus-4-6, T=1.0, N=8 per transcript.

## Results

Strict baseline: `results/eval_results_real_only.json` (N=16) and
`results/eval_results_client_only.json` (N=16). SoftCommit: this run
(N=8). Smaller N for soft is the cost cap, not the design.

| Transcript | Strict pass@1 | SoftCommit pass@1 | Strict mean F1 | SoftCommit mean F1 |
|---|---|---|---|---|
| real_earnings_001 | 0.000 (0/16) | **1.000 (8/8)** | 0.000 | **1.000** |
| real_earnings_002 | 0.000 (0/16) | 0.000 (0/8) | 0.000 | 0.000 |
| real_earnings_003 | 0.000 (0/15) | 0.000 (0/8) | 0.000 | 0.000 |
| real_client_001 | 0.000 (0/16) | 0.000 (0/8) | 0.633 | 0.540 |
| **aggregate** | **0.000** | **0.250** | **0.158** | **0.385** |

Smoke test (real_earnings_001 only, 8 samples) hit 8/8 pass@1, so we
proceeded to the full run.

## Raw output samples

### real_earnings_001 (the one transcript that flipped)

Strict baseline returned `{"action_items": []}` on every sample (16/16
empty).

SoftCommit, sample 0:
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

This matches the ground truth exactly. Owner canonicalizes "Ben Regin" to
"ben". Due "offline" is a literal match. Task contains anchor keywords
"rent", "collected", "modeling".

### real_earnings_002 (still failing under softcommit)

Strict: `{"action_items": []}` every time.

SoftCommit, sample 0:
```json
{
  "action_items": [
    {
      "owner": "NVIDIA (Colette Kress / Jensen Huang)",
      "task": "Work to hold gross margins in the mid-seventies for fiscal year 2027 ...",
      "due": "fiscal year 2027"
    },
    ...
  ]
}
```

The model now extracts the right item. The semantics are right. The task
text contains the anchor keywords ("margin", "mid-seven", "seventies").
But the rubric does not match because:

1. Owner canonicalizes to "nvidia" (first token), not "colette".
2. Due "fiscal year 2027" canonicalizes to "fiscal year 2027" (no
   weekday or today/tomorrow match). Ground-truth due "next year"
   canonicalizes to "year". They do not match.

Sample 5 actually got the owner right ("Colette Kress" -> "colette")
but used "ongoing" for due, and ground truth wants "next year". Still
miss.

### real_earnings_003 (still failing)

Strict: `{"action_items": []}` every time.

SoftCommit, sample 0:
```json
{
  "action_items": [
    {
      "owner": "Jeremy Barnum",
      "task": "Remove the 'cash returns to investors' line item ...",
      "due": "open"
    },
    {
      "owner": "JPMorgan (Jamie Dimon / Jeremy Barnum)",
      "task": "Submit focused comment letter on Basel III endgame and G-SIB ...",
      "due": "during comment period"
    }
  ]
}
```

Two failure modes here:

1. The model invented a Jeremy item ("remove cash returns line") that is
   not in the ground truth. Wrong topic.
2. Owner "JPMorgan (Jamie Dimon / Jeremy Barnum)" canonicalizes to
   "jpmorgan", not "jamie". Due "during comment period" does not match
   ground-truth "ongoing".
3. The model never extracted the "Jeremy next quarter consumer deposit"
   item across any of the 8 samples.

### real_client_001 (softcommit slightly worse)

Strict, sample 0 got 2 of 3 items perfectly:
```json
{"action_items": [
  {"owner": "Engineer", "task": "Deploy a link to the App demo for Client to explore on his own", "due": "tonight"},
  {"owner": "Client", "task": "Send the 57-page document with 22 story treatments to Engineer and Partner", "due": "tomorrow morning"}
]}
```
That is precision 1.0, recall 0.667, F1 = 0.8.

SoftCommit, sample 0:
```json
{"action_items": [
  {"owner": "Client", "task": "Send the 57-page document ...", "due": "tomorrow morning"},
  {"owner": "Client", "task": "Prepare and send information about ... Band C ...", "due": "after the meeting"},
  {"owner": "Engineer", "task": "Deploy a link to ...", "due": "later tonight"},
  ...
]}
```

The softcommit prompt makes the model extract more items. Some of those
extra items are not in the ground truth, which drops precision. So mean
F1 goes from 0.633 (strict) to 0.540 (softcommit). The permissive
prompt helps recall but hurts precision on a transcript where the
strict prompt was already finding the easy wins.

## Interpretation

Pass@1 moved from 0.000 to 0.250 in aggregate. One of four real
transcripts cracks open completely (0/16 to 8/8). On the other three,
the model now extracts the right items most of the time but the rubric
still says zero.

So the prompt-label mismatch was real but it explains only part of the
failure. About a quarter of the missed pass@1 was the prompt telling the
model to reject correct items. The rest is something else.

What "something else" looks like, from the raw outputs:

1. **Owner format mismatch.** On corporate calls the model picks
   institutional owners ("NVIDIA", "JPMorgan (Jamie Dimon / Jeremy
   Barnum)") because that is how the dialogue actually reads. The
   rubric canonicalizes to the first token, which gives "nvidia",
   "jpmorgan". Ground truth uses single first names ("Colette",
   "Jamie"). The semantics are right but the format is wrong.

2. **Due format mismatch.** The model uses "fiscal year 2027",
   "during comment period", "after the meeting", "later tonight". The
   rubric only canonicalizes to {today, tomorrow, weekday names} or to
   the literal lowercased string. Ground truth uses "next year",
   "ongoing", "open", "tonight". So even a faithful summary fails the
   string match.

3. **Recall ceiling.** real_earnings_003 has three ground-truth items.
   Across 8 samples the model never extracted the "Jeremy / next
   quarter / consumer deposits" item. The transcript probably does not
   have a strong enough dialogue cue for the model to flag that
   exchange, even with a permissive prompt.

4. **Precision tax on permissive prompts.** On real_client_001 the
   permissive prompt extracts dialogue items the strict prompt was
   correctly ignoring (e.g., "think about how to direct University MIS
   students... by next Tuesday"). Mean F1 drops 0.10 because the extra
   items are not in the ground truth.

## Bottom line

The original 0/0 result on real data was not the whole story. About a
quarter of it was the strict prompt telling the model to reject items
that the labels said were real. The fix is real and replicable.

The remaining 75% of the gap is not prompt design. It is owner-format
mismatch, due-format mismatch, and a real recall ceiling on
dialogue-confirmation commitments where the cue is weak. Fixing those
needs either looser canonicalization in the rubric (multi-token owners,
broader due-string normalization) or a different evaluation contract
that scores semantic match instead of string match.

For the meeting_intent benchmark this means the headline pass@1 number
is sensitive to prompt phrasing. Any future scoring should re-run with
a calibrated prompt, not the strict one.

## Cost

Smoke: $0.73. Full: $6.90. Total: $7.63. Cap was $10.
