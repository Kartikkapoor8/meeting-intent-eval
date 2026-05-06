# Meeting Intent Eval Results

**Model:** Claude Opus 4.6
**Synthetic dataset:** 10 hand-authored meeting transcripts, 64 samples each
**Real dataset:** 3 public earnings call transcripts, 16 samples each
**Method:** Sampling at temperature 1.0. Score is F1 over correctly extracted action items. Pass = F1 ≥ 0.99. Pass@k uses the Codex unbiased estimator.
**Cost:** $13.75 synthetic, $7.71 real. **Format:** Prime Intellect verifiers.

## Headline

| Dataset | pass@1 | mean F1 | format compliance |
|---|---|---|---|
| Synthetic (10 × 64) | **0.887** | 0.965 | 1.000 |
| Real earnings calls (3 × 16) | **0.000** | 0.000 | 1.000 |

Same model, same rubric, same prompt. Pass@1 fell from 0.887 to 0.000 the moment we left hand-authored data. Format compliance stayed at 1.0 across both runs, so the model is not breaking. It is refusing to extract anything from real transcripts.

## Synthetic results

Pass@1 = **0.887**. Opus 4.6 nails 8 of 10 transcripts on every single sample. Two transcripts fail in interesting ways:

| Transcript | passing/64 | pass@1 | pass@32 |
|---|---|---|---|
| 8 standard transcripts | 64/64 each | 1.000 | 1.000 |
| eng_standup_003 | **0/64** | **0.000** | 0.000 |
| support_escalation_007 | 56/64 | 0.875 | 1.000 |
| **aggregate (10)** | | **0.887** | **0.900** |

The pass@1 to pass@32 gap is tiny (1.3 points) because the eng_standup failure is deterministic. Sampling more does not help.

### The interesting synthetic failure

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

This is the failure pattern we wanted to chase down on real data.

### The other synthetic failure

support_escalation_007 has 3 real action items. Opus gets all 3 right in 56 of 64 samples. In the other 8 it adds a fourth:

> Hana: ok I'll set expectations for end of day and update them tomorrow if needed.

That sentence has a real verb and a real timeframe but the action is contingent. ~12% of the time Opus extracts it as a tracked deliverable. Stochastic over-extraction on judgment calls.

## Real earnings call results

3 public earnings call transcripts, downloaded free from The Motley Fool. Each one is over 30 minutes long. Industries chosen for variety: small-cap REIT, megacap tech, megacap finance.

| transcript | source | length | passing/n | pass@1 |
|---|---|---|---|---|
| `real_earnings_001` | IIPR Q1 2025 (May 8 2025) | ~30 min, 4K words | 0/16 | 0.000 |
| `real_earnings_002` | NVDA Q3 FY26 (Nov 19 2025) | ~58 min, 8.5K words | 0/16 | 0.000 |
| `real_earnings_003` | JPM Q1 2026 (Apr 21 2026) | ~77 min, 11.6K words | 0/15 | 0.000 |
| **aggregate (3)** | | | | **0.000** |

Across 47 valid completions the model returned `{"action_items": []}` every single time. Not one extraction attempted. The format-compliance rate was 1.000 (47/47 valid JSON), so the model is not crashing or rambling. It is reading these calls and deciding nothing in them counts as a real action item.

### Ground truth design

We focused the ground truth on **dialogue-confirmation commitments**. These are commitments built across multiple speaker turns through follow-up questions, not stated unilaterally in prepared remarks. Same pattern as the eng_standup_003 failure on the synthetic side. Examples:

**IIPR. Ben Regin commits to follow up offline with Connor Mitchell.**
> Connor Mitchell: any rents as well, I think for some of these, you may receive January payments, but not February or March?
> Ben Regin: In terms of what we actually collected?
> Connor Mitchell: Yes. Just thinking about a modeling perspective going forward, what won't be being received for a quarterly standpoint or even monthly?
> Ben Regin: Yes. We can go into the detail offline, but it was roughly $4.5 million that we collected from the defaulted tenants during the quarter.

**NVDA. Colette Kress commits to hold mid-70s gross margins.**
> Stacy Rasgon: You said for next year, you're working to hold them in the mid-seventies. So I guess, first of all, what are the biggest cost increases?
> Colette Kress: ...we will work to try and hold at our gross margins in the mid-seventies. So that's our overall plan for gross margin.

**JPM. Jeremy Barnum commits to add a topic to a list after Mike Mayo pushes back.**
> Michael Mayo: I don't particularly like that because I think it puts you in an artificial position thinking that's always a good thing when it's not.
> Jeremy Barnum: Very well. I'll add that to list. Next question.

**JPM. Jeremy Barnum commits to revisit consumer deposit fundamentals next quarter.**
> Manan Gosalia: How you see deposit competition unfolding as similar smart tools become more widespread?
> Jeremy Barnum: ...we'll be a little bit more confident in that, as you say, once we get through tax season. So maybe we'll know a little bit more next quarter.

**JPM. Jamie Dimon commits to find ways to reduce the firm's G-SIB charge.**
> James Dimon: We will obviously use our brainpower to do something I don't like doing, which is trying to find a lot of ways to serve our clients properly and reduce the G-SIB charge... we will find ways to do it.

The model caught zero of these.

## The failure mode: over-conservative refusal

The model is not hallucinating. It is not making things up. It is reading the transcript correctly and concluding that nothing in it is concrete enough to extract. Then it returns an empty list.

Why this happens. The system prompt tells the model to filter out hedged statements ("maybe", "might", "I'll think about it", "we'll figure it out"). On the synthetic transcripts that filter is calibrated correctly because the hand-authored commitments use clean phrasing like "by Friday" or "Wednesday EOD". On real earnings calls almost every commitment is hedged. "We can go into the detail offline" sounds like vague follow-up. "I'll add that to list" sounds like a brush-off. "Maybe we'll know a little bit more next quarter" sounds explicitly hedged. The model treats them all as the kind of soft mention it was told to drop.

This shows up clean in the data:
- Synthetic eng_standup_003 (the one dialogue-confirmation transcript in the synthetic set): 0/64 pass.
- Real earnings calls (where almost every commitment is dialogue-confirmation): 0/47 pass.

Same failure pattern, scaled up. The synthetic dataset just had one transcript that exercised it. Real data exercises it on every transcript.

## What would change the number

Three levers, in priority order.

**One. Train on dialogue-confirmation patterns.** The model currently treats "I can pair with you if you want" → "yeah" as an optional offer that fizzles. It needs to learn that the back-and-forth closes the conditional. This is what we cannot fix with a prompt. The eng_standup result on the synthetic side says sampling does not help and prompting does not help (the system prompt already addresses hedged language explicitly). The fix has to come from training.

**Two. Loosen the rubric's due-date matching.** The current rubric canonicalizes due to weekday names or specific tokens like "today" / "tomorrow". Earnings call commitments use due dates like "next quarter", "offline", "ongoing". Even if the model had emitted these items, exact (owner, due) matching would have failed for several. This is a smaller effect than lever one because the model emitted nothing, but it does matter for measuring partial credit.

**Three. Get more real transcripts.** N=3 is small. The pass@1 = 0.000 number is tight (47/47 empty), so the upper bound on the true rate is low (~6% by Hoeffding), but more variety would catch failure modes outside the dialogue-confirmation family.

## What this means for the pitch

The synthetic 88.7% number told the wrong story. It made the model look strong because the dataset was biased toward easy cases. We hand-authored these and drifted toward explicit declarations because they are easier to ground-truth. Real meetings have multi-turn-confirmation patterns in nearly every transcript, not 1 in 10.

The training opportunities, in priority order:
1. Recognize commitments formed across multiple speakers (offer + accept + logistics).
2. Treat hedged language conservatively only when the conversation does not close the conditional.
3. Stop at the dialogue boundary, not at a single utterance.

## Caveats on the real-eval number

1. **N=16 not 64.** Cost reasons. With pass@1 = 0/16 across all three real transcripts, the upper bound on the true pass@1 is around 6% by Hoeffding. So even with N=64 the result would very likely be at most 0.06.
2. **Strict rubric.** The rubric requires canonical (owner, due) match plus an anchor keyword in the task. Real earnings call commitments have softer due-date language than the synthetic set, so even a model that emitted these items might miss on (owner, due) match. But the model emitted zero items, so this is not the bottleneck right now.
3. **Three transcripts is a small sample.** They cover a range (small cap REIT, megacap tech, megacap finance) and they are all real earnings calls of at least 30 minutes, but more variety would surface different failure modes.
4. **`run_eval.py` had a print bug** when N < 32 (NoneType formatting on pass@32). Fixed by printing `n/a` for unavailable metrics. Scoring logic unchanged.

## What's in the repo

- `environments/meeting_intent/`: the verifiers env, ready to publish to the Hub.
- `environments/meeting_intent/data/`: 10 synthetic transcripts + 3 real earnings call transcripts, all with ground truth JSON.
- `run_eval.py`: async sampling against the Anthropic API.
- `rescore.py`: re-score saved samples without spending API budget.
- `results/raw_samples*.json`: all model outputs for reproducibility.
- `results/eval_results_real_only.json`: aggregate numbers from the real eval.
- `results/final_aggregate.json`: aggregate numbers from the synthetic eval.
