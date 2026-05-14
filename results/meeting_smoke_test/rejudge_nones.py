"""Retry only entries where the _fullctx flag is None (parse failure or error).

Uses answer-first prompt + max_tokens=800 to avoid truncation.
"""
import os, sys, json, time
from pathlib import Path
from anthropic import Anthropic

ROOT = Path(__file__).parent
SUBSET = {str(s["idx"]): s for s in json.loads((ROOT / "subset.json").read_text())}
RAW = json.loads((ROOT / "raw_samples.json").read_text())
JUDGE_PATH = ROOT / "judge_results.json"
ERR_DIR = ROOT / "error_types"

JUDGE_MODEL_ID = "claude-haiku-4-5-20251001"
JUDGE_PRICE_IN, JUDGE_PRICE_OUT = 1.0, 5.0

RETARGET_TYPES = ["extrinsic_hallucination", "intrinsic_hallucination", "coreference"]


def load_def(et):
    return json.loads((ERR_DIR / f"{et}.json").read_text())


def build_prompt(error_type, definition, examples, transcript, summary):
    low = examples["low"]; high = examples["high"]
    parts = [
        f"You evaluate a meeting summary for one specific error type.\n",
        f"ERROR TYPE: {error_type}\n",
        f"DEFINITION: {definition}\n\n",
        "--- LOW SEVERITY EXAMPLE (score 1) ---\n",
    ]
    if "transcript" in low: parts.append(f"Transcript: {low['transcript']}\n")
    parts.append(f"Summary: {low['summary']}\nExplanation: {low['explanation']}\n\n")
    parts.append(f"--- HIGH SEVERITY EXAMPLE (score 5) ---\n")
    if "transcript" in high: parts.append(f"Transcript: {high['transcript']}\n")
    parts.append(f"Summary: {high['summary']}\nExplanation: {high['explanation']}\n\n")
    parts.append("--- SAMPLE TO EVALUATE ---\n")
    parts.append(f"Transcript (full):\n>>>\n{transcript}\n<<<\n\n")
    parts.append(f"Summary:\n>>>\n{summary}\n<<<\n\n")
    parts.append(
        "STRICT FORMAT: your reply MUST start with one of these two tokens on the very first line, "
        "with no preamble:\n  ANSWER: YES\n  ANSWER: NO\n"
        "Then on subsequent lines write a one-sentence justification. "
        "Use YES only if the error is clearly present per the definition.\n"
    )
    return "".join(parts)


def parse(text):
    s = text.strip()
    if not s: return None
    # First, try the strict ANSWER: prefix
    for line in s.splitlines():
        line = line.strip()
        if "ANSWER:" in line.upper():
            after = line.upper().split("ANSWER:", 1)[1].strip()
            if after.startswith("YES"): return True
            if after.startswith("NO"): return False
    # Fallback: starts with YES/NO
    first = s.splitlines()[0].strip().upper()
    if first.startswith("YES"): return True
    if first.startswith("NO"): return False
    return None


def cost(u): return u.input_tokens/1e6*JUDGE_PRICE_IN + u.output_tokens/1e6*JUDGE_PRICE_OUT


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("missing key", file=sys.stderr); sys.exit(2)
    defs = {et: load_def(et) for et in RETARGET_TYPES}
    judge = json.loads(JUDGE_PATH.read_text())

    todo = []
    for sid in judge:
        for mk in judge[sid]:
            for et in RETARGET_TYPES:
                key = et + "_fullctx"
                v = judge[sid][mk].get(key)
                if not v or v.get("flag") is None:
                    todo.append((sid, mk, et))

    print(f"Retrying {len(todo)} None _fullctx entries")
    client = Anthropic(timeout=180.0, max_retries=2)
    total = 0.0
    for sid, mk, et in todo:
        summary = RAW[sid]["summaries"][mk]["summary"]
        transcript = SUBSET[sid]["input"]
        prompt = build_prompt(et, defs[et]["definition"], defs[et]["example"], transcript, summary)
        try:
            for attempt in range(3):
                try:
                    resp = client.messages.create(
                        model=JUDGE_MODEL_ID,
                        max_tokens=800,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    break
                except Exception as e:
                    if attempt == 2: raise
                    time.sleep(2 ** attempt)
            txt = resp.content[0].text
            flag = parse(txt)
            c = cost(resp.usage)
            total += c
            judge[sid][mk][et + "_fullctx"] = {
                "flag": flag,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cost_usd": c,
                "raw_tail": txt[-300:],
                "raw_head": txt[:150],
                "retry": True,
            }
            print(f"  {sid}/{mk}/{et}: flag={flag} tokens={resp.usage.output_tokens}")
        except Exception as e:
            judge[sid][mk][et + "_fullctx"] = {"flag": None, "error": str(e)}
        JUDGE_PATH.write_text(json.dumps(judge, indent=2))
    print(f"\nRetry cost: ${total:.3f}")


if __name__ == "__main__":
    main()
