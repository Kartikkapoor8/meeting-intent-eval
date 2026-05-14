"""Cheap single-instance judge.

For each (meeting, model_summary) pair, ONE haiku call asks the judge to flag
each of 9 error types in a single JSON output. Less faithful than multi-instance
but bounds judge cost to ~36 calls instead of 360.

We pass a truncated transcript (first N words) to bound input cost.
"""
import os, sys, json
from pathlib import Path
from anthropic import Anthropic

ROOT = Path(__file__).parent
SUBSET = ROOT / "subset.json"
ERR_DIR = ROOT / "error_types"
OUT_RAW = ROOT / "raw_samples.json"
OUT_JUDGE = ROOT / "judge_results.json"

JUDGE_MODEL_ID = "claude-haiku-4-5-20251001"
JUDGE_PRICE_IN, JUDGE_PRICE_OUT = 1.0, 5.0
COST_CAP_REMAINING = 2.0  # we already spent ~$2.36; cap judge phase at $2

TRANSCRIPT_TRUNC_WORDS = 3500  # judge sees first 3500 words

# Use the paper's 9-type taxonomy: extrinsic + intrinsic merged into HAL.
ERROR_TYPES_9 = [
    ("REP", "Repetition", "repetition"),
    ("INC", "Incoherence", "incoherence"),
    ("LAN", "Linguistic inaccuracy", "linguistic"),
    ("OM",  "Omission (total or partial)", "total_omission"),  # we merge T-OM + P-OM
    ("COR", "Coreference / misattribution", "coreference"),
    ("HAL", "Hallucination (intrinsic detail error OR extrinsic added content)", "extrinsic_hallucination"),
    ("STR", "Structure (wrong logical/chronological order)", "structure"),
    ("IRR", "Irrelevance", "irrelevance"),
    ("P_OM", "Partial omission specifically", "partial_omission"),
]
# That gives us 9 distinct flags; OM ~= total omission, P_OM ~= partial omission.


def load_def(path):
    with open(ERR_DIR / f"{path}.json") as f:
        return json.load(f)["definition"]


def build_prompt(transcript_trunc, summary):
    defs = {code: load_def(path) for code, _, path in ERROR_TYPES_9}
    # for OM we want total omission def; for HAL we want extrinsic def (paper merges both)
    lines = [
        "You are an expert annotator evaluating a meeting summary for nine specific error types.",
        "Be conservative. Only flag an error if you are confident the definition is clearly violated.",
        "Do not flag stylistic preferences. Do not flag minor omissions as omission unless they are material.",
        "",
        "ERROR DEFINITIONS:",
    ]
    for code, name, _ in ERROR_TYPES_9:
        lines.append(f"- {code} ({name}): {defs[code]}")
    lines.append("")
    lines.append(f"TRANSCRIPT (truncated to first {TRANSCRIPT_TRUNC_WORDS} words for evaluation):")
    lines.append(">>>")
    lines.append(transcript_trunc)
    lines.append("<<<")
    lines.append("")
    lines.append("SUMMARY:")
    lines.append(">>>")
    lines.append(summary)
    lines.append("<<<")
    lines.append("")
    lines.append("Task: For each error code, decide YES or NO. Reply with EXACTLY this JSON, nothing else:")
    lines.append('{"REP": "YES|NO", "INC": "YES|NO", "LAN": "YES|NO", "OM": "YES|NO", "COR": "YES|NO", "HAL": "YES|NO", "STR": "YES|NO", "IRR": "YES|NO", "P_OM": "YES|NO"}')
    return "\n".join(lines)


def parse(text):
    """Extract the JSON object. Returns dict[code -> True/False/None]."""
    import re
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        return {code: None for code, _, _ in ERROR_TYPES_9}
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return {code: None for code, _, _ in ERROR_TYPES_9}
    out = {}
    for code, _, _ in ERROR_TYPES_9:
        v = obj.get(code, "")
        if isinstance(v, str):
            vu = v.upper().strip()
            out[code] = (vu == "YES") if vu in ("YES", "NO") else None
        else:
            out[code] = None
    return out


def cost(u):
    return u.input_tokens/1e6*JUDGE_PRICE_IN + u.output_tokens/1e6*JUDGE_PRICE_OUT


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set.", file=sys.stderr); sys.exit(2)

    raw = json.loads(OUT_RAW.read_text())
    subset = json.loads(SUBSET.read_text())
    subset_by_idx = {str(s["idx"]): s for s in subset}

    client = Anthropic()
    judge = {}
    if OUT_JUDGE.exists():
        judge = json.loads(OUT_JUDGE.read_text())

    total_cost = 0.0
    pairs = []
    for sid, srec in raw.items():
        for mkey, mrec in srec["summaries"].items():
            if "summary" in mrec:
                pairs.append((sid, mkey, mrec["summary"]))

    print(f"Judge plan: {len(pairs)} single-instance calls")
    for sid, mkey, summary in pairs:
        if sid in judge and mkey in judge[sid] and "flags" in judge[sid][mkey]:
            continue
        if sid not in judge:
            judge[sid] = {}
        sample = subset_by_idx[sid]
        truncated = " ".join(sample["input"].split()[:TRANSCRIPT_TRUNC_WORDS])
        prompt = build_prompt(truncated, summary)
        try:
            resp = client.messages.create(
                model=JUDGE_MODEL_ID,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            txt = resp.content[0].text
            flags = parse(txt)
            c = cost(resp.usage)
            total_cost += c
            judge[sid][mkey] = {
                "flags": flags,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cost_usd": c,
                "raw": txt[:500],
            }
            print(f"  idx={sid} {mkey}: {resp.usage.input_tokens}+{resp.usage.output_tokens}tok ${c:.4f} (cum ${total_cost:.3f}) flags={ {k:v for k,v in flags.items() if v} }")
        except Exception as e:
            judge[sid][mkey] = {"error": str(e)}
            print(f"  idx={sid} {mkey}: ERROR {e}", file=sys.stderr)
        OUT_JUDGE.write_text(json.dumps(judge, indent=2))
        if total_cost > COST_CAP_REMAINING:
            print(f"Judge cost cap hit: ${total_cost:.3f} > ${COST_CAP_REMAINING}", file=sys.stderr)
            break

    print(f"\nJudge total: ${total_cost:.3f}")


if __name__ == "__main__":
    main()
