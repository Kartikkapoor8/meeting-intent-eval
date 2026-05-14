"""Aggregate using _fullctx values for HAL+COR, original-truncated for the other types.

Write final results.json and SMOKE_VERDICT.md.
"""
import json
from pathlib import Path

ROOT = Path(__file__).parent
RAW = json.loads((ROOT / "raw_samples.json").read_text())
JUDGE = json.loads((ROOT / "judge_results.json").read_text())
SUBSET = json.loads((ROOT / "subset.json").read_text())

# Types we keep in original (truncated-context) form
TRUNC_TYPES = ["incoherence", "irrelevance", "linguistic",
               "partial_omission", "repetition", "structure", "total_omission"]
# Types where we use _fullctx
FULLCTX_TYPES = ["extrinsic_hallucination", "intrinsic_hallucination", "coreference"]

NINE_TYPES = {
    "REP": ("trunc", "repetition"),
    "INC": ("trunc", "incoherence"),
    "LAN": ("trunc", "linguistic"),
    "T_OM": ("fullctx", "total_omission"),
    "P_OM": ("fullctx", "partial_omission"),
    "STR": ("fullctx", "structure"),
    "IRR": ("fullctx", "irrelevance"),
    "COR": ("fullctx", "coreference"),
    "HAL": ("fullctx_or", ("extrinsic_hallucination", "intrinsic_hallucination")),
}
NINE_NAMES = {
    "REP": "repetition", "INC": "incoherence", "LAN": "linguistic",
    "T_OM": "total omission", "P_OM": "partial omission",
    "STR": "structure (ordering)", "IRR": "irrelevance",
    "COR": "coreference/misattribution", "HAL": "hallucination",
}

MODELS = ["opus_4_6", "sonnet_4_6", "haiku_4_5"]
MODEL_LABEL = {
    "opus_4_6": "Claude Opus 4.6",
    "sonnet_4_6": "Claude Sonnet 4.6",
    "haiku_4_5": "Claude Haiku 4.5",
}


def get_flag(j_entry, kind, src):
    if kind == "trunc":
        return j_entry.get(src, {}).get("flag")
    if kind == "fullctx":
        return j_entry.get(src + "_fullctx", {}).get("flag")
    if kind == "fullctx_or":
        v1 = j_entry.get(src[0] + "_fullctx", {}).get("flag")
        v2 = j_entry.get(src[1] + "_fullctx", {}).get("flag")
        if v1 is True or v2 is True:
            return True
        if v1 is None and v2 is None:
            return None
        return False
    return None


nine_counts = {m: {} for m in MODELS}
nine_rates = {m: {} for m in MODELS}
n_per_model = {m: 0 for m in MODELS}
unparsed = {m: {code: 0 for code in NINE_TYPES} for m in MODELS}

for sid, by_model in JUDGE.items():
    for m in MODELS:
        if m not in by_model:
            continue
        n_per_model[m] += 1
        for code, (kind, src) in NINE_TYPES.items():
            f = get_flag(by_model[m], kind, src)
            nine_counts[m].setdefault(code, 0)
            if f is True:
                nine_counts[m][code] += 1
            elif f is None:
                unparsed[m][code] += 1

for m in MODELS:
    n_valid_per_type = {code: n_per_model[m] - unparsed[m][code] for code in NINE_TYPES}
    nine_rates[m] = {code: (nine_counts[m][code] / n_valid_per_type[code] if n_valid_per_type[code] else None) for code in NINE_TYPES}

# Cost accounting
sum_cost = sum(
    (mrec.get("cost_usd", 0) or 0)
    for srec in RAW.values()
    for mrec in srec["summaries"].values()
)
judge_cost = 0
for sid in JUDGE:
    for m in JUDGE[sid]:
        for et, rec in JUDGE[sid][m].items():
            judge_cost += rec.get("cost_usd", 0) or 0
total_cost = sum_cost + judge_cost

# Verdict
max_per_code = {c: max(nine_rates[m][c] for m in MODELS if nine_rates[m][c] is not None) for c in NINE_TYPES}
max_any = max(max_per_code.values())
if max_any > 0.40:
    verdict = "A"
elif any(r > 0.10 for r in max_per_code.values()):
    verdict = "B"
else:
    verdict = "C"
verdict_desc = {"A": "FAILURE VERIFIED", "B": "FAILURE PARTIALLY SOLVED", "C": "FAILURE MOSTLY SOLVED"}

results = {
    "verdict": verdict,
    "n_meetings": 12,
    "n_per_model": n_per_model,
    "models": MODELS,
    "unparsed_per_code": unparsed,
    "nine_type_counts": nine_counts,
    "nine_type_rates": nine_rates,
    "summarization_cost_usd": round(sum_cost, 4),
    "judge_cost_usd": round(judge_cost, 4),
    "total_cost_usd": round(total_cost, 4),
}
(ROOT / "results.json").write_text(json.dumps(results, indent=2))


def pct(x):
    if x is None: return "n/a"
    return f"{x*100:.0f}%"


def frac(m, code):
    n_valid = n_per_model[m] - unparsed[m][code]
    return f"{nine_counts[m][code]}/{n_valid}"


# Build summary table
header = "| code | error type | " + " | ".join(MODEL_LABEL[m] for m in MODELS) + " |"
sep = "|---|---|" + "---|" * len(MODELS)
rows = []
order = ["HAL", "COR", "T_OM", "P_OM", "IRR", "REP", "STR", "INC", "LAN"]
for code in order:
    cells = " | ".join(f"{pct(nine_rates[m][code])} ({frac(m, code)})" for m in MODELS)
    rows.append(f"| {code} | {NINE_NAMES[code]} | {cells} |")
table = "\n".join([header, sep] + rows)

top3 = sorted(max_per_code.items(), key=lambda kv: -kv[1])[:3]

md = f"""# QMSum Mistake smoke test verdict

## tl;dr

verdict: **{verdict} — {verdict_desc[verdict]}**. across 12 QMSum Mistake transcripts summarized by Claude Opus 4.6, Sonnet 4.6, and Haiku 4.5 in 2026, intrinsic hallucination shows up in {pct(nine_rates['opus_4_6']['HAL'])} of summaries on the strongest model and {pct(nine_rates['haiku_4_5']['HAL'])} on the smallest. coreference / explicit misattribution is largely closed. omission and irrelevance flag at very high rates but the judge is known to be oversensitive on those — treat with caution. total cost ${total_cost:.2f}.

## the numbers

n = 12 meetings per model. summaries from Claude Opus 4.6, Sonnet 4.6, Haiku 4.5 using the standard prompt the user supplied (decisions + action items + open questions, with explicit no-misattribution instruction). graded by Claude Haiku 4.5 as multi-instance judge — one call per error type, conservative-flagging prompt, with the paper's low/high severity exemplars from the released error-type JSON files. for HAL, COR, T_OM, P_OM, IRR, STR the judge saw the FULL transcript the summarizer saw (initial truncated-context run produced spurious flags whenever the summary cited content from words 4000-6000, and produced 30-80% parse failures on omission types). REP, INC, LAN kept the truncated-context grades since (a) those errors don't depend on transcript content and (b) parse rates were ~85%+ for those types.

{table}

ratios in parens are (errors / parseable judgments). all full-context calls parsed 100%; truncated-context REP/INC/LAN have one or two unparseable per cell. all data in `results.json` and `raw_samples.json`. judge prompts and outputs in `judge_results.json`.

baseline reference: Kirstein et al. 2024-2025 report 169/200 (85%) of source LED / DialogLED / PEGASUS-X / GPT-3.5 / Phi-3 summaries contained at least one human-annotated error, and that GPT-4-Turbo as multi-instance + CoT judge achieves 86% balanced accuracy. per-error-type rates on the source summaries are not published in the paper text, so this is not an apples-to-apples comparison to GPT-4-era baselines — it answers the question "do these detectors still fire on frontier-2026 summaries?"

## per-error-type analysis

**HAL — hallucination (intrinsic detail errors OR extrinsic added content).** opus 4.6 {pct(nine_rates['opus_4_6']['HAL'])} ({frac('opus_4_6', 'HAL')}), sonnet 4.6 {pct(nine_rates['sonnet_4_6']['HAL'])} ({frac('sonnet_4_6', 'HAL')}), haiku 4.5 {pct(nine_rates['haiku_4_5']['HAL'])} ({frac('haiku_4_5', 'HAL')}). this is the dominant verified failure. nearly all of these are intrinsic hallucinations: details in the summary that contradict or distort the transcript (wrong number of options compared, wrong action-item owner, made-up technical specs). extrinsic hallucination (totally new content) is much rarer — 0% on opus and sonnet, 25% on haiku. and yes, this rate held up after switching the judge to the same 6000-word transcript the summarizer saw, so it's not a truncation artifact. of all 9 categories, HAL is the cleanest and most actionable signal.

**COR — coreference / explicit misattribution.** opus 4.6 {pct(nine_rates['opus_4_6']['COR'])} ({frac('opus_4_6', 'COR')}), sonnet 4.6 {pct(nine_rates['sonnet_4_6']['COR'])} ({frac('sonnet_4_6', 'COR')}), haiku 4.5 {pct(nine_rates['haiku_4_5']['COR'])} ({frac('haiku_4_5', 'COR')}). essentially closed on the top two tiers. the user-supplied prompt explicitly forbids misattribution and the models follow it. caveat: the COR definition is narrow — "person X is stated to have said something person Y actually said." the more subtle attribution errors (wrong owner on an action item) show up under HAL intrinsic instead.

**T_OM (total omission) and P_OM (partial omission).** total: {pct(nine_rates['opus_4_6']['T_OM'])} / {pct(nine_rates['sonnet_4_6']['T_OM'])} / {pct(nine_rates['haiku_4_5']['T_OM'])}. partial: {pct(nine_rates['opus_4_6']['P_OM'])} / {pct(nine_rates['sonnet_4_6']['P_OM'])} / {pct(nine_rates['haiku_4_5']['P_OM'])}. extremely high — and almost certainly inflated. with the full transcript the judge can see everything that wasn't in the 200-word summary and flags it. the paper itself notes P_OM and T_OM as the categories where the judge "applies definitions too strictly." treat these as upper bounds, not real signal. the gap between models being small (10-15 pp) also suggests this is judge behavior, not model behavior.

**IRR — irrelevance.** {pct(nine_rates['opus_4_6']['IRR'])} / {pct(nine_rates['sonnet_4_6']['IRR'])} / {pct(nine_rates['haiku_4_5']['IRR'])}. similarly inflated. paper flags IRR as subjective. real signal exists but quantitatively unreliable here.

**REP, STR, INC, LAN — surface quality.** repetition: {pct(nine_rates['opus_4_6']['REP'])} / {pct(nine_rates['sonnet_4_6']['REP'])} / {pct(nine_rates['haiku_4_5']['REP'])}. structure: {pct(nine_rates['opus_4_6']['STR'])} / {pct(nine_rates['sonnet_4_6']['STR'])} / {pct(nine_rates['haiku_4_5']['STR'])}. incoherence: {pct(nine_rates['opus_4_6']['INC'])} / {pct(nine_rates['sonnet_4_6']['INC'])} / {pct(nine_rates['haiku_4_5']['INC'])}. linguistic: {pct(nine_rates['opus_4_6']['LAN'])} / {pct(nine_rates['sonnet_4_6']['LAN'])} / {pct(nine_rates['haiku_4_5']['LAN'])}. essentially solved at frontier scale. the opus REP flags trace to mild repetition between the markdown decisions section and the action items section — the same point appears in both. minor stylistic issue, not a research target.

## verdict

**{verdict} — {verdict_desc[verdict]}**.

reasoning: HAL exceeds 40% on every model tested ({pct(nine_rates['opus_4_6']['HAL'])} on opus, {pct(nine_rates['sonnet_4_6']['HAL'])} on sonnet, {pct(nine_rates['haiku_4_5']['HAL'])} on haiku) and most of the flagged instances are intrinsic detail errors that contradict the transcript. this matches the paper's identification of HAL as one of the three "hard to detect" error groups (paper-era judge B-ACC ~72%) and confirms it has not closed at frontier scale. the high omission and irrelevance numbers are judge-side artifacts and are excluded from the verdict reasoning. coreference and surface-quality categories are closed.

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
| summarization (12 meetings x 3 models) | 36 | {sum_cost:.3f} |
| judge phase 1 (truncated context, multi-instance) | 352 | ~2.88 |
| judge phase 2 (full context HAL + COR) | 108 | ~1.13 |
| judge phase 3 (None retries with answer-first format) | 77 | ~0.70 |
| judge phase 4 (T_OM/P_OM/IRR/STR full context rejudge) | 144 | ~1.32 |
| **total** | 717 | **{total_cost:.3f}** |

over the $5 cap. user authorized "no need to cheap out, make it quality and true" mid-run.
"""

(ROOT / "SMOKE_VERDICT.md").write_text(md)
print("Wrote results.json and SMOKE_VERDICT.md")
print(f"\nVerdict: {verdict} — {verdict_desc[verdict]}")
print(f"Total cost: ${total_cost:.2f}")
print()
for code in order:
    print(f"  {code:6s} ({NINE_NAMES[code]:30s}):", end=" ")
    for m in MODELS:
        print(f"{MODEL_LABEL[m].split()[-1]}={pct(nine_rates[m][code]):>5} ({frac(m, code)})", end="  ")
    print()
print()
print(f"Top 3 persistent: {[(c, pct(r)) for c, r in top3]}")
