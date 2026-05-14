"""Resume judging with retries. Re-runs any entry that lacks a 'flag' value.

Treats errored entries (no cost_usd, no flag) as needing retry.
Uses Haiku 4.5 multi-instance, one error type per call.
Per-call timeout 90s. Up to 3 retries with backoff.
"""
import os, sys, json, time
from pathlib import Path
from anthropic import Anthropic

ROOT = Path(__file__).parent
SUBSET = ROOT / "subset.json"
ERR_DIR = ROOT / "error_types"
OUT_RAW = ROOT / "raw_samples.json"
OUT_JUDGE = ROOT / "judge_results.json"

JUDGE_MODEL_ID = "claude-haiku-4-5-20251001"
JUDGE_PRICE_IN, JUDGE_PRICE_OUT = 1.0, 5.0

ERROR_TYPES = [
    "coreference", "extrinsic_hallucination", "intrinsic_hallucination",
    "incoherence", "irrelevance", "linguistic",
    "partial_omission", "repetition", "structure", "total_omission",
]

# Truncate transcript fed to judge to keep calls under timeout
JUDGE_TRANSCRIPT_TRUNC = 4000  # words


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
    parts.append(f"Transcript (truncated to first {JUDGE_TRANSCRIPT_TRUNC} words):\n>>>\n{transcript}\n<<<\n\n")
    parts.append(f"Summary:\n>>>\n{summary}\n<<<\n\n")
    parts.append(
        "Question: Does the summary contain this specific error type? "
        "Think briefly, then on the FINAL line write exactly one of:\n"
        "  ANSWER: YES\n"
        "  ANSWER: NO\n"
        "Use ANSWER: YES only if the error is clearly present per the definition. "
        "Be conservative; do not flag borderline cases.\n"
    )
    return "".join(parts)


def parse_judge(text):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if "ANSWER:" in line.upper():
            after = line.upper().split("ANSWER:", 1)[1].strip()
            if after.startswith("YES"):
                return True
            if after.startswith("NO"):
                return False
    return None


def cost(usage):
    return usage.input_tokens/1e6*JUDGE_PRICE_IN + usage.output_tokens/1e6*JUDGE_PRICE_OUT


def needs_redo(entry):
    if not entry:
        return True
    if "flag" not in entry:
        return True
    if entry["flag"] is None:
        # parse failure or error; redo
        return True
    return False


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set.", file=sys.stderr); sys.exit(2)

    raw = json.loads(OUT_RAW.read_text())
    subset = json.loads(SUBSET.read_text())
    subset_by_idx = {str(s["idx"]): s for s in subset}
    defs = load_error_defs()

    judge = json.loads(OUT_JUDGE.read_text()) if OUT_JUDGE.exists() else {}

    # Build TODO list
    todo = []
    for sid, srec in raw.items():
        for mkey, mrec in srec["summaries"].items():
            if "summary" not in mrec:
                continue
            jrec = judge.setdefault(sid, {}).setdefault(mkey, {})
            for et in ERROR_TYPES:
                if needs_redo(jrec.get(et)):
                    todo.append((sid, mkey, et))

    print(f"TODO: {len(todo)} judge calls (out of 360 total)")

    client = Anthropic(timeout=90.0, max_retries=2)
    total_cost = 0.0
    n_done = 0
    n_failed = 0
    failures = []

    for sid, mkey, et in todo:
        sample = subset_by_idx[sid]
        summary = raw[sid]["summaries"][mkey]["summary"]
        trunc = " ".join(sample["input"].split()[:JUDGE_TRANSCRIPT_TRUNC])
        prompt = build_judge_prompt(et, defs[et]["definition"], defs[et]["example"], trunc, summary)
        success = False
        last_err = None
        for attempt in range(3):
            try:
                resp = client.messages.create(
                    model=JUDGE_MODEL_ID,
                    max_tokens=400,
                    messages=[{"role": "user", "content": prompt}],
                )
                txt = resp.content[0].text
                flag = parse_judge(txt)
                c = cost(resp.usage)
                total_cost += c
                judge[sid][mkey][et] = {
                    "flag": flag,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "cost_usd": c,
                    "raw_tail": txt[-200:],
                    "attempt": attempt,
                }
                success = True
                break
            except Exception as e:
                last_err = str(e)
                time.sleep(2 ** attempt)  # 1, 2, 4 s
        if not success:
            judge[sid][mkey][et] = {"flag": None, "error": last_err}
            n_failed += 1
            failures.append((sid, mkey, et))
        n_done += 1
        if n_done % 5 == 0 or not success:
            OUT_JUDGE.write_text(json.dumps(judge, indent=2))
            print(f"  [{n_done}/{len(todo)}] cost ${total_cost:.3f} (failed so far: {n_failed})", flush=True)

    OUT_JUDGE.write_text(json.dumps(judge, indent=2))
    print(f"\nResume done. Made {n_done} calls, failed {n_failed}, cost ${total_cost:.3f}")
    if failures:
        print("Failures:")
        for f in failures[:20]:
            print("  ", f)


if __name__ == "__main__":
    main()
