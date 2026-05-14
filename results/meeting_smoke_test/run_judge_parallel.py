"""Parallel multi-instance judge.

Same protocol as run_smoke.py judge phase (one Haiku 4.5 call per error type,
CoT prompt, conservative threshold). Runs concurrent requests for speed.

Picks up from existing judge_results.json (resumes).
"""
import os, sys, json, asyncio
from pathlib import Path
from anthropic import AsyncAnthropic

ROOT = Path(__file__).parent
SUBSET = ROOT / "subset.json"
ERR_DIR = ROOT / "error_types"
OUT_RAW = ROOT / "raw_samples.json"
OUT_JUDGE = ROOT / "judge_results.json"

JUDGE_MODEL_ID = "claude-haiku-4-5-20251001"
JUDGE_PRICE_IN, JUDGE_PRICE_OUT = 1.0, 5.0
COST_CAP_TOTAL = 20.0
CONCURRENCY = 8

ERROR_TYPES = [
    "coreference", "extrinsic_hallucination", "intrinsic_hallucination",
    "incoherence", "irrelevance", "linguistic",
    "partial_omission", "repetition", "structure", "total_omission",
]


def load_error_defs():
    defs = {}
    for et in ERROR_TYPES:
        with open(ERR_DIR / f"{et}.json") as f:
            defs[et] = json.load(f)
    return defs


def build_judge_prompt(error_type, definition, examples, transcript, summary):
    low = examples["low"]; high = examples["high"]
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
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if "ANSWER:" in line.upper():
            after = line.upper().split("ANSWER:", 1)[1].strip()
            if after.startswith("YES"): return True
            if after.startswith("NO"): return False
    return None


def cost(u):
    return u.input_tokens/1e6*JUDGE_PRICE_IN + u.output_tokens/1e6*JUDGE_PRICE_OUT


async def judge_one(client, sem, sample, mkey, summary, et, defs, results, results_lock):
    sid = str(sample["idx"])
    async with sem:
        if results.get(sid, {}).get(mkey, {}).get(et, {}).get("flag") is not None:
            return None
        try:
            resp = await client.messages.create(
                model=JUDGE_MODEL_ID,
                max_tokens=400,
                messages=[{"role":"user","content":
                    build_judge_prompt(et, defs[et]["definition"], defs[et]["example"],
                                       sample["input"], summary)}],
            )
            txt = resp.content[0].text
            flag = parse_judge(txt)
            c = cost(resp.usage)
            async with results_lock:
                results.setdefault(sid, {}).setdefault(mkey, {})[et] = {
                    "flag": flag,
                    "input_tokens": resp.usage.input_tokens,
                    "output_tokens": resp.usage.output_tokens,
                    "cost_usd": c,
                    "raw_tail": txt[-300:],
                }
            return c
        except Exception as e:
            async with results_lock:
                results.setdefault(sid, {}).setdefault(mkey, {})[et] = {"flag": None, "error": str(e)}
            print(f"  ERR idx={sid} {mkey} {et}: {e}", file=sys.stderr)
            return 0.0


async def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set."); sys.exit(2)
    raw = json.loads(OUT_RAW.read_text())
    subset = {str(s["idx"]): s for s in json.loads(SUBSET.read_text())}
    defs = load_error_defs()
    results = json.loads(OUT_JUDGE.read_text()) if OUT_JUDGE.exists() else {}

    # Build tasks
    tasks_list = []
    for sid, srec in raw.items():
        sample = subset[sid]
        for mkey, mrec in srec["summaries"].items():
            if "summary" not in mrec: continue
            for et in ERROR_TYPES:
                existing = results.get(sid, {}).get(mkey, {}).get(et)
                if existing and existing.get("flag") is not None:
                    continue
                tasks_list.append((sample, mkey, mrec["summary"], et))

    print(f"Judge tasks remaining: {len(tasks_list)}")
    if not tasks_list:
        return

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    results_lock = asyncio.Lock()
    save_counter = [0]

    async def runner():
        coros = [judge_one(client, sem, s, m, summ, et, defs, results, results_lock)
                 for s, m, summ, et in tasks_list]
        done = 0
        total_cost = 0.0
        for fut in asyncio.as_completed(coros):
            c = await fut
            done += 1
            if c is not None:
                total_cost += c
            if done % 10 == 0:
                async with results_lock:
                    OUT_JUDGE.write_text(json.dumps(results, indent=2))
                print(f"  {done}/{len(tasks_list)} done, judge cost so far ${total_cost:.3f}")
                if total_cost > COST_CAP_TOTAL:
                    print(f"COST CAP HIT at judge=${total_cost:.3f}")
                    return
        async with results_lock:
            OUT_JUDGE.write_text(json.dumps(results, indent=2))
        print(f"Done. Judge total cost: ${total_cost:.3f}")

    await runner()

if __name__ == "__main__":
    asyncio.run(main())
