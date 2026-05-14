"""Rejudge total_omission and partial_omission with full transcript and
answer-first format. The original truncated-context run produced too many
parse failures (verbose judge responses hit max_tokens before writing ANSWER).
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
TARGETS = ["total_omission", "partial_omission", "irrelevance", "structure"]


def load_def(et): return json.loads((ERR_DIR / f"{et}.json").read_text())


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
        "STRICT FORMAT: reply MUST begin on the very first line with one of:\n"
        "  ANSWER: YES\n  ANSWER: NO\n"
        "Then one sentence of justification. Use YES only when the error is clearly present per the definition. "
        "For omission errors, the bar is that material content (decisions, action items, key disagreements) is missing — "
        "not stylistic compression.\n"
    )
    return "".join(parts)


def parse(text):
    s = text.strip()
    if not s: return None
    for line in s.splitlines():
        line = line.strip()
        if "ANSWER:" in line.upper():
            after = line.upper().split("ANSWER:", 1)[1].strip()
            if after.startswith("YES"): return True
            if after.startswith("NO"): return False
    first = s.splitlines()[0].strip().upper()
    if first.startswith("YES"): return True
    if first.startswith("NO"): return False
    return None


def cost(u): return u.input_tokens/1e6 + u.output_tokens/1e6*5


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("missing key"); sys.exit(2)
    defs = {et: load_def(et) for et in TARGETS}
    judge = json.loads(JUDGE_PATH.read_text())

    todo = []
    for sid in RAW:
        for mk in RAW[sid]["summaries"]:
            if "summary" not in RAW[sid]["summaries"][mk]: continue
            for et in TARGETS:
                todo.append((sid, mk, et))

    print(f"Rejudging {len(todo)} (T_OM/P_OM/IRR/STR) with full context, answer-first")
    client = Anthropic(timeout=180.0, max_retries=2)
    total = 0.0
    flips = 0
    for sid, mk, et in todo:
        summary = RAW[sid]["summaries"][mk]["summary"]
        transcript = SUBSET[sid]["input"]
        prompt = build_prompt(et, defs[et]["definition"], defs[et]["example"], transcript, summary)
        old = judge.get(sid, {}).get(mk, {}).get(et, {}).get("flag")
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
            flag = parse(txt)
            c = cost(resp.usage); total += c
            judge[sid][mk][et + "_fullctx"] = {
                "flag": flag,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
                "cost_usd": c,
                "raw_head": txt[:120],
            }
            if flag != old:
                flips += 1
        except Exception as e:
            judge[sid][mk][et + "_fullctx"] = {"flag": None, "error": str(e)}
        JUDGE_PATH.write_text(json.dumps(judge, indent=2))
    print(f"\nDone. cost ${total:.3f}. flips vs original {flips}/{len(todo)}")


if __name__ == "__main__":
    main()
