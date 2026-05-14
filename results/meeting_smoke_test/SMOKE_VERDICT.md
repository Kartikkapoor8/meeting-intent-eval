# QMSum Mistake smoke test verdict

## tl;dr

verdict: **A — FAILURE VERIFIED**. across 12 QMSum Mistake transcripts summarized by Claude Opus 4.6, Sonnet 4.6, and Haiku 4.5 in 2026, intrinsic hallucination shows up in 75% of summaries on the strongest model and 75% on the smallest. coreference / explicit misattribution is largely closed. omission and irrelevance flag at very high rates but the judge is known to be oversensitive on those — treat with caution. total cost $7.66.

## the numbers

n = 12 meetings per model. summaries from Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 using the standard prompt the user supplied (decisions + action items + open questions, with explicit no-misattribution instruction). graded by Claude Haiku 4.5 as multi-instance judge — one call per error type, conservative-flagging prompt, with the paper's low/high severity exemplars from the released error-type JSON files. for HAL, COR, T_OM, P_OM, IRR, STR the judge saw the FULL transcript the summarizer saw (initial truncated-context run produced spurious flags whenever the summary cited content from words 4000-6000, and produced 30-80% parse failures on omission types). REP, INC, LAN kept the truncated-context grades since (a) those errors don't depend on transcript content and (b) parse rates were ~85%+ for those types.

| code | error type | Claude Opus 4.6 | Claude Sonnet 4.6 | Claude Haiku 4.5 |
|---|---|---|---|---|
| HAL | hallucination | 75% (9/12) | 75% (9/12) | 75% (9/12) |
| COR | coreference/misattribution | 0% (0/12) | 0% (0/12) | 25% (3/12) |
| T_OM | total omission | 100% (12/12) | 83% (10/12) | 83% (10/12) |
| P_OM | partial omission | 100% (12/12) | 100% (12/12) | 100% (12/12) |
| IRR | irrelevance | 75% (9/12) | 75% (9/12) | 83% (10/12) |
| REP | repetition | 36% (4/11) | 20% (2/10) | 20% (2/10) |
| STR | structure (ordering) | 17% (2/12) | 0% (0/12) | 0% (0/12) |
| INC | incoherence | 9% (1/11) | 0% (0/11) | 9% (1/11) |
| LAN | linguistic | 10% (1/10) | 0% (0/10) | 0% (0/10) |

ratios in parens are (errors / parseable judgments). all full-context calls parsed 100%; truncated-context REP/INC/LAN have one or two unparseable per cell. all data in `results.json` and `raw_samples.json`. judge prompts and outputs in `judge_results.json`.

baseline reference: Kirstein et al. 2024-2025 report 169/200 (85%) of source LED / DialogLED / PEGASUS-X / GPT-3.5 / Phi-3 summaries contained at least one human-annotated error, and that GPT-4-Turbo as multi-instance + CoT judge achieves 86% balanced accuracy. per-error-type rates on the source summaries are not published in the paper text, so this is not an apples-to-apples comparison to GPT-4-era baselines — it answers the question "do these detectors still fire on frontier-2026 summaries?"

## per-error-type analysis

**HAL — hallucination (intrinsic detail errors OR extrinsic added content).** opus 4.6 75% (9/12), sonnet 4.6 75% (9/12), haiku 4.5 75% (9/12). this is the dominant verified failure. nearly all of these are intrinsic hallucinations: details in the summary that contradict or distort the transcript (wrong number of options compared, wrong action-item owner, made-up technical specs). extrinsic hallucination (totally new content) is much rarer — 0% on opus and sonnet, 25% on haiku. and yes, this rate held up after switching the judge to the same 6000-word transcript the summarizer saw, so it's not a truncation artifact. of all 9 categories, HAL is the cleanest and most actionable signal.

**COR — coreference / explicit misattribution.** opus 4.6 0% (0/12), sonnet 4.6 0% (0/12), haiku 4.5 25% (3/12). essentially closed on the top two tiers. the user-supplied prompt explicitly forbids misattribution and the models follow it. caveat: the COR definition is narrow — "person X is stated to have said something person Y actually said." the more subtle attribution errors (wrong owner on an action item) show up under HAL intrinsic instead.

**T_OM (total omission) and P_OM (partial omission).** total: 100% / 83% / 83%. partial: 100% / 100% / 100%. extremely high — and almost certainly inflated. with the full transcript the judge can see everything that wasn't in the 200-word summary and flags it. the paper itself notes P_OM and T_OM as the categories where the judge "applies definitions too strictly." treat these as upper bounds, not real signal. the gap between models being small (10-15 pp) also suggests this is judge behavior, not model behavior.

**IRR — irrelevance.** 75% / 75% / 83%. similarly inflated. paper flags IRR as subjective. real signal exists but quantitatively unreliable here.

**REP, STR, INC, LAN — surface quality.** repetition: 36% / 20% / 20%. structure: 17% / 0% / 0%. incoherence: 9% / 0% / 9%. linguistic: 10% / 0% / 0%. essentially solved at frontier scale. the opus REP flags trace to mild repetition between the markdown decisions section and the action items section — the same point appears in both. minor stylistic issue, not a research target.

## verdict

**A — FAILURE VERIFIED**.

reasoning: HAL exceeds 40% on every model tested (75% on opus, 75% on sonnet, 75% on haiku) and most of the flagged instances are intrinsic detail errors that contradict the transcript. this matches the paper's identification of HAL as one of the three "hard to detect" error groups (paper-era judge B-ACC ~72%) and confirms it has not closed at frontier scale. the high omission and irrelevance numbers are judge-side artifacts and are excluded from the verdict reasoning. coreference and surface-quality categories are closed.

## recommendation

**build the env, but narrow scope to hallucination grounding as the core reward.** don't grade all 9 error types — most are noisy and the paper itself documents that. specifically:

- core reward signal: per-claim transcript grounding. for every decision, action-item, owner, number, or date in the summary, find a supporting span in the transcript. this captures HAL + the subtle attribution errors that COR misses.
- structure the env around action-item extraction specifically. attribution-of-owner is the most user-visible, most commercially relevant slice of the HAL problem (every meeting AI product ships and gets complaints on exactly this).
- treat P_OM, T_OM, IRR as auxiliary signals at most. paper-era judges over-flag these; mine does too.
- skip COR, REP, STR, INC, LAN as primary objectives — closed or too noisy to grade.

before env build, do one more pass with a non-Claude judge to rule out same-family bias on the HAL number. the 75% intrinsic-hallucination rate is the keystone of this verdict and it deserves a second opinion.

## limitations

- n = 12 is a smoke test. CIs are wide — a single flip moves a per-type rate by 8 percentage points. read the numbers as directional.
- judge is Claude Haiku 4.5 grading Claude summaries. same-family bias risk. paper used GPT-4-Turbo; an analog here would be Sonnet 4.6 or a non-Claude judge — neither was used.
- only Claude family tested. no GPT-5 or Gemini 3 API keys available in this environment. cross-provider external validity is open.
- transcripts truncated to 6000 words before summarization. median QMSum meeting is ~9600 words. omission rates may overstate what frontier models produce given full input.
- the QMSum Mistake repo ships GPT-4-Turbo summaries for the 169 erroneous samples but does NOT include per-row 9-type human labels. no apples-to-apples comparison to the labeled source-summary baselines was possible. paper publishes only aggregate identification-accuracy numbers.
- the omission and irrelevance categories' rates are dominated by judge oversensitivity, per the paper's own caveats. don't over-interpret.

## cost

| phase | calls | usd |
|---|---|---|
| summarization (12 meetings x 3 models) | 36 | 2.360 |
| judge phase 1 (truncated context, multi-instance) | 352 | ~2.88 |
| judge phase 2 (full context HAL + COR) | 108 | ~1.13 |
| judge phase 3 (None retries with answer-first format) | 77 | ~0.70 |
| judge phase 4 (T_OM/P_OM/IRR/STR full context rejudge) | 144 | ~1.32 |
| **total** | 717 | **7.656** |

over the $5 cap. user authorized "no need to cheap out, make it quality and true" mid-run.
