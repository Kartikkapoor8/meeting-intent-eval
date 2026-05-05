"""Run meeting_intent against Claude Opus and compute pass@k."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ENV_DIR = Path(__file__).parent / "environments" / "meeting_intent"
sys.path.insert(0, str(ENV_DIR))

from meeting_intent import f1_reward, format_reward, load_environment  # noqa: E402

from anthropic import AsyncAnthropic  # noqa: E402

MODEL = os.environ.get("EVAL_MODEL", "claude-opus-4-6")
N_SAMPLES = int(os.environ.get("EVAL_N", "64"))
TEMPERATURE = float(os.environ.get("EVAL_T", "1.0"))
MAX_TOKENS = 1024
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "8"))
BUDGET_USD = float(os.environ.get("EVAL_BUDGET", "18.0"))
PASS_THRESHOLD = 0.99
ONLY = set(t.strip() for t in os.environ.get("EVAL_ONLY", "").split(",") if t.strip())
RESULTS_TAG = os.environ.get("EVAL_TAG", "")

# Opus 4.x list price
PRICE_IN = 15.0 / 1_000_000
PRICE_OUT = 75.0 / 1_000_000


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


class Cost:
    def __init__(self, cap):
        self.cap = cap
        self.in_tok = 0
        self.out_tok = 0

    @property
    def usd(self):
        return self.in_tok * PRICE_IN + self.out_tok * PRICE_OUT

    def add(self, i, o):
        self.in_tok += i
        self.out_tok += o

    def over(self):
        return self.usd > self.cap


async def sample_one(client, system, user, sem, cost, attempt=0):
    async with sem:
        if cost.over():
            return {"text": "", "in_tok": 0, "out_tok": 0, "error": "budget_cap"}
        try:
            resp = await client.messages.create(
                model=MODEL,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
        except Exception as e:
            if attempt < 2:
                await asyncio.sleep(1.5 * (attempt + 1))
                return await sample_one(client, system, user, sem, cost, attempt + 1)
            return {"text": "", "in_tok": 0, "out_tok": 0, "error": str(e)[:200]}
        text = "".join(getattr(b, "text", "") for b in resp.content)
        i_tok = resp.usage.input_tokens
        o_tok = resp.usage.output_tokens
        cost.add(i_tok, o_tok)
        return {"text": text, "in_tok": i_tok, "out_tok": o_tok}


async def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    env = load_environment()
    print(f"Model: {MODEL}")
    print(f"Transcripts: {len(env.dataset)}")
    print(f"Samples per transcript: {N_SAMPLES} at T={TEMPERATURE}")
    print(f"Concurrency: {CONCURRENCY}, budget cap: ${BUDGET_USD}")
    print()

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    cost = Cost(BUDGET_USD)

    results = []
    t_start = time.time()

    for row in env.dataset:
        tid = row["info"]["transcript_id"]
        if ONLY and tid not in ONLY:
            continue
        prompt = row["prompt"]
        system = next((m["content"] for m in prompt if m["role"] == "system"), "")
        user = next(m["content"] for m in prompt if m["role"] == "user")

        print(f"[{tid}] sampling {N_SAMPLES}...", flush=True)
        coros = [sample_one(client, system, user, sem, cost) for _ in range(N_SAMPLES)]
        samples = await asyncio.gather(*coros)

        scores, fmt_oks = [], []
        for s in samples:
            scores.append(f1_reward(completion=s["text"], info=row["info"]))
            fmt_oks.append(format_reward(completion=s["text"]))

        n = sum(1 for s in samples if not s.get("error"))
        c = sum(1 for s in scores[:n] if s >= PASS_THRESHOLD)
        n_eff = max(n, 1)

        row_result = {
            "transcript_id": tid,
            "n_attempted": len(samples),
            "n_successful": n,
            "n_passing": c,
            "pass_at_1": pass_at_k(n, c, 1) if n >= 1 else 0.0,
            "pass_at_8": pass_at_k(n, c, 8) if n >= 8 else None,
            "pass_at_32": pass_at_k(n, c, 32) if n >= 32 else None,
            "mean_f1": sum(scores) / n_eff,
            "format_compliance": sum(fmt_oks) / n_eff,
            "scores": scores,
            "samples": samples,
        }
        results.append(row_result)
        p1, p8, p32 = row_result["pass_at_1"], row_result["pass_at_8"], row_result["pass_at_32"]
        print(
            f"  passing={c}/{n}  "
            f"pass@1={p1:.3f}  pass@8={p8:.3f}  pass@32={p32:.3f}  "
            f"meanF1={row_result['mean_f1']:.3f}  fmt={row_result['format_compliance']:.3f}  "
            f"spent=${cost.usd:.2f}",
            flush=True,
        )
        if cost.over():
            print("BUDGET CAP REACHED, stopping early.")
            break

    elapsed = time.time() - t_start
    n_rows = len(results)

    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "model": MODEL,
        "n_samples_per_transcript": N_SAMPLES,
        "temperature": TEMPERATURE,
        "pass_threshold": PASS_THRESHOLD,
        "n_transcripts": n_rows,
        "elapsed_sec": round(elapsed, 1),
        "cost_usd": round(cost.usd, 4),
        "input_tokens": cost.in_tok,
        "output_tokens": cost.out_tok,
        "aggregate": {
            "pass_at_1": avg("pass_at_1"),
            "pass_at_8": avg("pass_at_8"),
            "pass_at_32": avg("pass_at_32"),
            "mean_f1": avg("mean_f1"),
            "format_compliance": avg("format_compliance"),
        },
        "per_transcript": [
            {k: v for k, v in r.items() if k not in ("samples",)}
            for r in results
        ],
    }

    out = Path("results")
    out.mkdir(exist_ok=True)
    suffix = f"_{RESULTS_TAG}" if RESULTS_TAG else ""
    with open(out / f"eval_results{suffix}.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out / f"raw_samples{suffix}.json", "w") as f:
        json.dump({r["transcript_id"]: r["samples"] for r in results}, f, indent=2)

    print()
    print("=" * 60)
    print(f"RESULTS  model={MODEL}  transcripts={n_rows}  n={N_SAMPLES}  T={TEMPERATURE}")
    print("=" * 60)
    print(f"pass@1            {summary['aggregate']['pass_at_1']:.3f}")
    print(f"pass@8            {summary['aggregate']['pass_at_8']:.3f}")
    print(f"pass@32           {summary['aggregate']['pass_at_32']:.3f}")
    print(f"mean F1           {summary['aggregate']['mean_f1']:.3f}")
    print(f"format compliance {summary['aggregate']['format_compliance']:.3f}")
    print(f"cost              ${summary['cost_usd']:.2f}")
    print(f"elapsed           {summary['elapsed_sec']}s")
    print()
    print("Saved: results/eval_results.json, results/raw_samples.json")


if __name__ == "__main__":
    asyncio.run(main())
