# Kartik tab update — paste into shared Google Doc

- Status: meeting_intent thesis **PARTIALLY VERIFIED** as of 2026-05-11
- Proof: one-page proof at https://github.com/Kartikkapoor8/meeting-intent-eval/blob/claude/brave-volhard/results/ONE_PAGER.md (will move to main once merged)
- Headline finding: under default "extract action items" prompts, Opus 4.6 + Sonnet 4.6 + Haiku 4.5 all return empty arrays 32 of 32 times on a real earnings-call dialogue commitment (Ben Regin / Connor Mitchell offline follow-up, IIPR Q1 2025)
- Honest update: original framing was "universal 0/0 failure on real transcripts." Calibrated soft-commitment prompt fixes 1 of 4 real transcripts on Opus (8/8 pass on real_earnings_001). Other 3 transcripts still fail, split between rubric format mismatch and a residual recall ceiling on weak dialogue cues. Failure is real but smaller than the original claim
- Next: pick one of three follow-ups — (a) loosen rubric canonicalization for owner/due format mismatches and re-score, (b) collect 5 to 10 more real dialogue-commitment transcripts to test the recall-ceiling hypothesis, (c) write the training-data brief on what dialogue-commitment supervision would look like
