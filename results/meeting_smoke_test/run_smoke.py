"""Run the smoke test end to end.

Steps:
  1. Generate summaries for each of 12 meetings using Opus 4.6, Sonnet 4.6, Haiku 4.5.
  2. Run LLM-as-judge (Claude Opus 4.6, multi-instance, one error type per call) over each summary.
  3. Save per-call results and aggregate rates.

Cost guard: aborts if running cost estimate exceeds USD 5.

Pricing (USD per 1M tokens, list price as of 2026):
  claude-opus-4-6:        input $15,  output $75
  claude-sonnet-4-6:      input $3,   output $15
  claude-haiku-4-5:       input $1,   output $5

Judge model is claude-haiku-4-5 to keep judge-call cost low.
"""
import os, sys, json, time, argparse
from anthropic import Anthropic
from pathlib import Path

ROOT = Path(__file__).parent
SUBSET = ROOT / "subset.json"
ERR_DIR = ROOT / "error_types"
OUT_RAW = ROOT / "raw_samples.json"
OUT_JUDGE = ROOT / "judge_results.json"
OUT_RESULTS = ROOT / "results.json"

SUMMARIZER_PROMPT = (
    "You are summarizing a meeting transcript. Produce a comprehensive summary including: "
    "(1) main decisions made, (2) action items with owners, (3) key disagreements or open "
    "questions. Be accurate about who said what. Do not hallucinate speakers or attribute "
    "statements to the wrong person."
)

MODELS = [
    ("opus_4_6",   "claude-opus-4-5",     15.0, 75.0),
    ("sonnet_4_6", "claude-sonnet-4-5",   3.0,  15.0),
    ("haiku_4_5",  "claude-haiku-4-5",    1.0,  5.0),
]
# NOTE: real IDs are claude-opus-4-6, claude-sonnet-4-6, claude-haiku-4-5-20251001 per the runtime info.
# We'll resolve actual IDs at runtime.

JUDGE_MODEL_ID = "claude-haiku-4-5-20251001"
JUDGE_PRICE_IN, JUDGE_PRICE_OUT = 1.0, 5.0

COST_CAP = 20.0  # USD (raised from 5 per user direction: quality > thrift)

ERROR_TYPES = [
    "coreference", "extrinsic_hallucination", "intrinsic_hallucination",
    "incoherence", "irrelevance", "linguistic",
    "partial_omission", "repetition", "structure", "total_omission",
]
# 10 files in the repo, but paper merges extrinsic+intrinsic into HAL.
# We collect both and combine for reporting.


def load_error_defs():
    defs = {}
    for et in ERROR_TYPES:
        with open(ERR_DIR / f"{et}.json") as f:
            defs[et] = json.load(f)
    return defs


def build_judge_prompt(error_type, definition, examples, transcript, summary):
    low = examples["low"]
    high = examples["high"]
    parts = [
        f"You are an expert annotator evaluating a meeting summary for one specific error type.\n",
        f"ERROR TYPE: {error_type}\n",
        f"DEFINITION: {definition}\n\n",
        "Here are two examples of this error type at different severities:\n",
        f"--- LOW SEVERITY EXAMPLE (score 1) ---\n",
    ]
    if "transcript" in low:
        parts.append(f"Transcript: {low['transcript']}\n")
    parts.append(f"Summary: {low['summary']}\n")
    parts.append(f"Explanation: {low['explanation']}\n\n")
    parts.append(f"--- HIGH SEVERITY EXAMPLE (score 5) ---\n")
    if "transcript" in high:
        parts.append(f"Transcript: {high['transcript']}\n")
    parts.append(f"Summary: {high['summary']}\n")
    parts.append(f"Explanation: {high['explanation']}\n\n")
    parts.append("--- NOW EVALUATE THIS SAMPLE ---\n")
    parts.append(f"Transcript:\n>>>\n{transcript}\n<<<\n\n")
    parts.append(f"Summary:\n>>>\n{summary}\n<<<\n\n")
    parts.append(
        "Question: Does the summary contain this specific error type? "
        "Think step by step, then on the FINAL line write exactly one of:\n"
        "  ANSWER: YES\n"
        "  ANSWER: NO\n"
        "Use ANSWER: YES only if the error is clearly present per the definition. "
        "Be conservative; do not flag borderline cases.\n"
    )
    return "".join(parts)


def parse_judge(text):
    # find last line containing ANSWER:
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if "ANSWER:" in line.upper():
            after = line.upper().split("ANSWER:", 1)[1].strip()
            if after.startswith("YES"):
                return True
            if after.startswith("NO"):
                return False
    return None  # parse failure


def cost(usage, price_in, price_out):
    return (usage.input_tokens / 1_000_000.0) * price_in + (usage.output_tokens / 1_000_000.0) * price_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="Print plan, no API calls.")
    ap.add_argument("--max-meetings", type=int, default=12)
    ap.add_argument("--resume", action="store_true", help="Skip work already saved.")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set. Aborting.", file=sys.stderr)
        sys.exit(2)

    with open(SUBSET) as f:
        subset = json.load(f)
    subset = subset[: args.max_meetings]

    defs = load_error_defs()
    client = Anthropic()
    running_cost = 0.0

    # ---- step 1: summaries ----
    raw_samples = {}
    if args.resume and OUT_RAW.exists():
        raw_samples = json.loads(OUT_RAW.read_text())

    # Resolve model IDs based on what the SDK accepts
    model_ids = {
        "opus_4_6":   "claude-opus-4-6",
        "sonnet_4_6": "claude-sonnet-4-6",
        "haiku_4_5":  "claude-haiku-4-5-20251001",
    }
    price = {
        "opus_4_6":   (15.0, 75.0),
        "sonnet_4_6": (3.0,  15.0),
        "haiku_4_5":  (1.0,  5.0),
    }

    print(f"Summarizing {len(subset)} meetings x {len(model_ids)} models = {len(subset)*len(model_ids)} calls")
    for sample in subset:
        sid = str(sample["idx"])
        if sid not in raw_samples:
            raw_samples[sid] = {"input_wc": sample["wc_input_used"], "gold": sample["gold"], "summaries": {}}
        for mkey, mid in model_ids.items():
            if mkey in raw_samples[sid]["summaries"]:
                continue
            if args.dry_run:
                print(f"  [dry] {mkey} on idx={sid}")
                continue
            try:
                resp = client.messages.create(
                    model=mid,
                    max_tokens=600,
                    system=SUMMARIZER_PROMPT,
                    messages=[{"role": "user", "content": f"Meeting transcript:\n\n{sample['input']}"}],
                )
                summary = resp.content[0].text
                pin, pout = price[mkey]
                c = cost(resp.usage, pin, pout)
                running_cost += c
                raw_samples[sid]["summaries"][mkey] = {
                    "model_id": mid,
                    "summary": summary,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "cost_usd": c,
                }
                print(f"  idx={sid} {mkey}: {resp.usage.input_tokens}+{resp.usage.output_tokens}tok ${c:.4f} (running ${running_cost:.3f})")
            except Exception as e:
                print(f"  idx={sid} {mkey}: ERROR {e}", file=sys.stderr)
                raw_samples[sid]["summaries"][mkey] = {"model_id": mid, "error": str(e)}
            OUT_RAW.write_text(json.dumps(raw_samples, indent=2))
            if running_cost > COST_CAP:
                print(f"COST CAP exceeded at ${running_cost:.3f}. Stopping.", file=sys.stderr)
                sys.exit(3)

    if args.dry_run:
        print("Dry run complete.")
        return

    print(f"\nSummarization cost so far: ${running_cost:.3f}\n")

    # ---- step 2: judge ----
    judge_results = {}
    if args.resume and OUT_JUDGE.exists():
        judge_results = json.loads(OUT_JUDGE.read_text())

    total_judge_calls = len(subset) * len(model_ids) * len(ERROR_TYPES)
    print(f"Running judge: {total_judge_calls} calls (12 samples * 3 models * 10 error types)")

    for sample in subset:
        sid = str(sample["idx"])
        if sid not in judge_results:
            judge_results[sid] = {}
        for mkey in model_ids:
            if mkey not in raw_samples.get(sid, {}).get("summaries", {}):
                continue
            entry = raw_samples[sid]["summaries"][mkey]
            if "summary" not in entry:
                continue
            summary = entry["summary"]
            if mkey not in judge_results[sid]:
                judge_results[sid][mkey] = {}
            for et in ERROR_TYPES:
                if et in judge_results[sid][mkey]:
                    continue
                prompt = build_judge_prompt(
                    et, defs[et]["definition"], defs[et]["example"],
                    sample["input"], summary,
                )
                try:
                    resp = client.messages.create(
                        model=JUDGE_MODEL_ID,
                        max_tokens=400,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    txt = resp.content[0].text
                    flag = parse_judge(txt)
                    c = cost(resp.usage, JUDGE_PRICE_IN, JUDGE_PRICE_OUT)
                    running_cost += c
                    judge_results[sid][mkey][et] = {
                        "flag": flag,
                        "input_tokens": resp.usage.input_tokens,
                        "output_tokens": resp.usage.output_tokens,
                        "cost_usd": c,
                        "raw_tail": txt[-300:],
                    }
                except Exception as e:
                    judge_results[sid][mkey][et] = {"flag": None, "error": str(e)}
                OUT_JUDGE.write_text(json.dumps(judge_results, indent=2))
                if running_cost > COST_CAP:
                    print(f"COST CAP exceeded at ${running_cost:.3f}. Stopping.", file=sys.stderr)
                    sys.exit(3)
            print(f"  judged idx={sid} {mkey}: running ${running_cost:.3f}")

    # ---- step 3: aggregate ----
    rates = {}
    n = len(subset)
    for mkey in model_ids:
        rates[mkey] = {"n": 0}
        # collapse extrinsic+intrinsic -> HAL
        types_for_report = list(ERROR_TYPES) + ["hallucination_any"]
        type_counts = {t: 0 for t in types_for_report}
        valid_n = 0
        for sid in judge_results:
            if mkey not in judge_results[sid]:
                continue
            valid_n += 1
            j = judge_results[sid][mkey]
            for et in ERROR_TYPES:
                if j.get(et, {}).get("flag") is True:
                    type_counts[et] += 1
            if (j.get("extrinsic_hallucination", {}).get("flag") is True or
                j.get("intrinsic_hallucination", {}).get("flag") is True):
                type_counts["hallucination_any"] += 1
        rates[mkey]["n"] = valid_n
        rates[mkey]["counts"] = type_counts
        rates[mkey]["rates"] = {t: (type_counts[t] / valid_n if valid_n else None) for t in type_counts}

    out = {
        "running_cost_usd": running_cost,
        "n_meetings": n,
        "models": list(model_ids.keys()),
        "rates": rates,
    }
    OUT_RESULTS.write_text(json.dumps(out, indent=2))
    print(f"\nTotal cost: ${running_cost:.3f}")
    print(f"Results saved to {OUT_RESULTS}")


if __name__ == "__main__":
    main()
