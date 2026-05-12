# meeting_intent prompt ablations v2

aggressive prompt ablations on the 3 transcripts that stayed at 0/8 under the
calibrated softcommit prompt (real_earnings_002, real_earnings_003,
real_client_001). question: is this a real model recall ceiling, or is the
prompt still wrong?

setup: opus 4.6, t=1.0, 8 samples per (variant, transcript). pass@1
threshold f1 >= 0.99. rubric is the same f1+anchor scorer used everywhere
else.

## summary table

| transcript          | v1 baseline | v2 dialogue | v3 few-shot | v4 oracle | v5 direct |
|---------------------|-------------|-------------|-------------|-----------|-----------|
| real_earnings_002   | 0/8         | 0/8         | 0/8         | **8/8**   | not run   |
| real_earnings_003   | 0/8         | 0/8         | 0/8         | 0/8       | not run   |
| real_client_001     | 0/8         | 0/8         | not run     | 0/8       | not run   |

cost cap hit after v2 + v3 (v3 only got through 2 of 3 transcripts on the
full-transcript runs). got user approval to extend $5 to run v4 oracle.
total spend $10.76. v5 was not run.

## what each variant did

**v1 (baseline calibrated softcommit prompt).** existing data from
`softcommit_full.json`. mean f1 was 0.0 on the two earnings transcripts and
0.54 on real_client_001. so v1 was not at "nothing comes out". on
real_client_001 the model was already producing partial credit.

**v2 (dialogue-explicit addendum).** added the "look for ask + offer
patterns" paragraph. zero pass@1 on all three. mean f1 stayed at 0.0 on
both earnings transcripts. on real_client_001 caught_any=8/8 with mean f1
0.547. essentially the same as v1.

**v3 (v2 + few-shot positive/negative examples).** zero pass@1 on the two
earnings transcripts that ran. budget hit before real_client_001 finished.
no movement vs v2.

**v4 (oracle snippet, only the dialogue chain + a few context lines).**
this is where the picture changed.
- real_earnings_002: 8/8. pass@1 flipped from 0.0 to 1.0. model nails the
  colette mid-70s gross margin commitment when handed the 3-paragraph
  snippet.
- real_earnings_003: 0/8 but caught_any=8/8 and mean f1 0.333. model
  consistently catches 1 of 3 ground-truth items per sample.
- real_client_001: 0/8 but caught_any=8/8 and mean f1 0.719. model
  consistently catches 2-3 of 3 ground-truth items but over-predicts with
  spurious "send follow-up email" items.

**v5 (direct "find it" prompt).** not run. cost cap.

## what the raw outputs say

i looked at the actual model outputs to see what's happening.

**real_earnings_002.** on the full transcript (v2/v3), opus does name the
colette commitment but writes the due as "fiscal year 2027". the rubric
canonicalizes "next year" to "year" but leaves "fiscal year 2027" alone, so
the due fails to match. on the oracle snippet (v4), the model uses "next
year" instead, which canonicalizes to "year" and matches. **same recognition,
different phrasing**. not a model failure. not really a prompt failure
either. it's a rubric/canonicalization mismatch.

**real_earnings_003.** even on the oracle snippet, opus consistently:
- catches the tax-season -> next-quarter commitment (jeremy) ✓
- misreads the "very well i'll add that to list" commitment. ground truth
  says jeremy is committing to add **mayo's pushback on artificial
  tangible-book-value targets** to the list. opus instead writes "remove
  cash returns to investors metric from materials, as requested by jamie
  dimon". opus reads "add that to list" as referring to dimon's preceding
  sentence about removing the cash-returns metric. genuinely ambiguous in
  the transcript. this is a real recognition gap.
- catches the g-sib arbitrage commitment but writes the owner as
  "james dimon", which canonicalize_owner returns as "james". ground
  truth owner is "jamie", which canonicalizes to "jamie". james ≠ jamie.
  another rubric/canonicalization miss, not a model miss.

**real_client_001.** model finds all 3 ground-truth commitments. mean f1
on v4 was 0.719. it over-predicts with a spurious "send a follow-up email"
item. it also writes engineer's reconvene due as "after the meeting" which
canonicalizes to "after meeting", not "open". so the recall is fine, the
precision is not, and the spurious item drags pass@1 to zero. precision
problem, not recall.

## verdict

**C (PARTIAL), trending toward rubric failure rather than model failure.**

what the data actually shows:

- real_earnings_002 is not a model failure at all. v4 oracle gets 8/8 and
  the v2/v3 misses are because opus phrased the due as "fiscal year 2027"
  instead of "next year". prompt fixes alone did not move the needle
  because they don't change the phrasing of the date. rubric fix
  (treat "fiscal year YYYY" as "next year") would flip v1 to passing.
- real_earnings_003 is half-recognition, half-canonicalization. opus
  consistently catches the tax-season commitment. it consistently
  misreads the "add to list" commitment because the dialogue is genuinely
  ambiguous. it catches the g-sib commitment but the owner canonicalizer
  rejects "james" vs "jamie". 1/3 real recognition gap, 1/3 rubric, 1/3
  catches.
- real_client_001 is mostly over-prediction. model finds all 3 but adds
  spurious follow-up items. precision drag. prompt could in principle fix
  this with a "do not add items that were not explicitly committed"
  instruction. rubric could in principle fix this with looser due
  canonicalization for "after the meeting" -> "open".

so the original meeting_intent thesis ("model fails to extract
dialogue-confirmation commitments from real meeting transcripts") is
**partially false**. on real_earnings_002 the model can extract it fine.
on the other two there is some real recognition gap but most of the
measured failure is rubric mismatch.

## what to do next

if the goal is to ship a real benchmark:

1. fix the canonicalizer. add "fiscal year YYYY" -> "year". add "jamie" / "james" aliasing for james dimon. add "after the meeting" -> "open" or extend the soft-due set.
2. re-score v1 with the fixed canonicalizer. expectation: real_earnings_002 flips to passing, real_client_001 moves up. real_earnings_003 stays at 0/8 or near-zero because of the real ambiguity in the "add to list" commitment.
3. if you want a real model-failure result, build it from the residual after canonicalization fixes. it is smaller than the current "0/8 on 3 transcripts" headline suggests.
4. run v5 on real_earnings_003 specifically. if direct prompting still can't get the model to interpret "add to list" as referring to mayo's pushback rather than dimon's remove-cash-returns line, that is the actual model failure to write up.

## headline finding

most of the "model can't extract these commitments" result is rubric
canonicalization rejecting correct extractions, not the model failing to
find them.

## artifacts

- `results/ablations/v2_dialogue_explicit.json` (+ `_raw.json`)
- `results/ablations/v3_few_shot.json` (+ `_raw.json`)
- `results/ablations/v4_oracle_snippet.json` (+ `_raw.json`)
- `results/ablations/run_log_v2.txt`
- harness scripts: `run_ablations_v2.py`, `run_ablations_v4_only.py`

variants v3 (real_client_001) and v5 (all) not run because of $10 cost cap
hit during v3.
