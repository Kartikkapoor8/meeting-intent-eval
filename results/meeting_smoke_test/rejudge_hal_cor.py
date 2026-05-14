"""Re-run HAL (intrinsic + extrinsic) and COR judges with the FULL transcript
that the summarizer saw, to eliminate truncation-induced false positives."""
import os, sys, json, time
from pathlib import Path
from anthropic import Anthropic

ROOT = Path(__file__).parent
SUBSET = json.loads((ROOT / "subset.json").read_text())
ERR_DIR = ROOT / "error_types"
RAW = json.loads((ROOT / "raw_samples.json").read_text())
JUDGE_PATH = ROOT / "judge_results.json"

JUDGE_MODEL_ID = "claude-haiku-4-5-20251001"
JUDGE_PRICE_IN, JUDGE_PRICE_OUT = 1.0, 5.0
RETARGET_TYPES = ["extrinsic_hallucination", "intrinsic_hallucination", "coreference"]


def load_def(et):
    return json.loads((ERR_DIR / f"{et}.json").read_text())


def build_prompt(error_type, definition, examples, transcript, summary):
    low = examples["low"]; high = examples["high"]
    parts = [
        f"You are an expert annotator evaluating a meeting summary for one specific error type.\n",
        f"ERROR TYPE: {error_type}\n",
        f"DEFINITION: {definition}\n\n",
        "--- LOW SEVERITY EXAMPLE (score 1) ---\n",
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
    parts.append(f"Transcript (full, same context the summarizer saw):\n>>>\n{transcript}\n<<<\n\n")
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


def parse(text):
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if "ANSWER:" in line.upper():
            after = line.upper().split("ANSWER:", 1)[1].strip()
            if after.startswith("YES"): return True
            if after.startswith("NO"): return False
    return None


def cost(u):
    return u.input_tokens/1e6*JUDGE_PRICE_IN + u.output_tokens/1e6*JUDGE_PRICE_OUT


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("missing key", file=sys.stderr); sys.exit(2)
    subset_by_idx = {str(s["idx"]): s for s in SUBSET}
    defs = {et: load_def(et) for et in RETARGET_TYPES}
    judge = json.loads(JUDGE_PATH.read_text())

    todo = []
    for sid, srec in RAW.items():
        for mkey, mrec in srec["summaries"].items():
            if "summary" not in mrec: continue
            for et in RETARGET_TYPES:
                todo.append((sid, mkey, et))

    print(f"Re-judging {len(todo)} (HAL + COR) calls with full transcript")
    client = Anthropic(timeout=120.0, max_retries=2)
    total_cost = 0.0
    flip_count = 0
    for sid, mkey, et in todo:
        summary = RAW[sid]["summaries"][mkey]["summary"]
        # Use full transcript that the summarizer saw
        transcript = subset_by_idx[sid]["input"]
        prompt = build_prompt(et, defs[et]["definition"], defs[et]["example"], transcript, summary)
        old_flag = judge.get(sid, {}).get(mkey, {}).get(et, {}).get("flag")
        try:
            for attempt in range(3):
                try:
                    resp = client.messages.create(
                        model=JUDGE_MODEL_ID,
                        max_tokens=400,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    break
                except Exception as e:
                    if attempt == 2: raise
                    time.sleep(2 ** attempt)
            txt = resp.content[0].text
            new_flag = parse(txt)
            c = cost(resp.usage)
            total_cost += c
            judge[sid][mkey][et + "_fullctx"] = {
                "flag": new_flag,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cost_usd": c,
                "raw_tail": txt[-300:],
            }
            if old_flag != new_flag:
                flip_count += 1
                print(f"  FLIP {sid}/{mkey}/{et}: {old_flag} -> {new_flag}")
        except Exception as e:
            judge[sid][mkey][et + "_fullctx"] = {"flag": None, "error": str(e)}
            print(f"  ERR {sid}/{mkey}/{et}: {e}", file=sys.stderr)
        JUDGE_PATH.write_text(json.dumps(judge, indent=2))

    print(f"\nRe-judge done. Cost ${total_cost:.3f}. Flips: {flip_count}/{len(todo)}")


if __name__ == "__main__":
    main()
