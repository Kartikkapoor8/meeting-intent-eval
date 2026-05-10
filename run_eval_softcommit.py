"""Run the softcommit ablation against Claude Opus and compute pass@k.

Mirrors run_eval.py but loads load_environment_softcommit. Errored samples
are filtered before scoring (the bug fix from Phase 1, applied here too).
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ENV_DIR = Path(__file__).parent / "environments" / "meeting_intent"
sys.path.insert(0, str(ENV_DIR))

from meeting_intent_softcommit import (  # noqa: E402
    f1_reward,
    format_reward,
    load_environment_softcommit,
)

from anthropic import AsyncAnthropic  # noqa: E402

MODEL = os.environ.get("EVAL_MODEL", "claude-opus-4-6")
N_SAMPLES = int(os.environ.get("EVAL_N", "8"))
TEMPERATURE = float(os.environ.get("EVAL_T", "1.0"))
MAX_TOKENS = 1024
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "4"))
BUDGET_USD = float(os.environ.get("EVAL_BUDGET", "10.0"))
PASS_THRESHOLD = 0.99
ONLY = set(t.strip() for t in os.environ.get("EVAL_ONLY", "").split(",") if t.strip())
OUT_PATH = os.environ.get("EVAL_OUT", "results/ablations/softcommit.json")

PRICE_IN = 15.0 / 1_000_000
PRICE_OUT = 75.0 / 1_000_000


def _fmt(v):
    return f"{v:.3f}" if v is not None else "n/a"


def pass_at_k(n, c, k):
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

    env = load_environment_softcommit()
    print(f"Model: {MODEL}")
    print(f"Variant: SOFTCOMMIT")
    print(f"Filter ONLY: {sorted(ONLY) if ONLY else '(all)'}")
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

        successful = [s for s in samples if not s.get("error")]
        errored = [s for s in samples if s.get("error")]
        n_attempted = len(samples)
        n_successful = len(successful)
        n_errored = len(errored)

        scores, fmt_oks = [], []
        for s in successful:
            scores.append(f1_reward(completion=s["text"], info=row["info"]))
            fmt_oks.append(format_reward(completion=s["text"]))

        n = n_successful
        c = sum(1 for s in scores if s >= PASS_THRESHOLD)
        n_eff = max(n, 1)

        if n_errored:
            err_kinds = {}
            for s in errored:
                k = s.get("error", "unknown")[:60]
                err_kinds[k] = err_kinds.get(k, 0) + 1
            print(f"  errored: {n_errored}/{n_attempted}  succeeded: {n_successful}/{n_attempted}", flush=True)
            for k, cnt in err_kinds.items():
                print(f"    [{cnt}x] {k}", flush=True)

        row_result = {
            "transcript_id": tid,
            "n_attempted": n_attempted,
            "n_successful": n_successful,
            "n_errored": n_errored,
            "n_passing": c,
            "pass_at_1": pass_at_k(n, c, 1) if n >= 1 else 0.0,
            "pass_at_8": pass_at_k(n, c, 8) if n >= 8 else None,
            "mean_f1": sum(scores) / n_eff,
            "format_compliance": sum(fmt_oks) / n_eff,
            "scores": scores,
            "samples": samples,
        }
        results.append(row_result)
        p1 = row_result["pass_at_1"]
        p8 = row_result["pass_at_8"]
        print(
            f"  passing={c}/{n}  pass@1={_fmt(p1)}  pass@8={_fmt(p8)}  "
            f"meanF1={row_result['mean_f1']:.3f}  fmt={row_result['format_compliance']:.3f}  "
            f"spent=${cost.usd:.2f}",
            flush=True,
        )
        if cost.over():
            print("BUDGET CAP REACHED, stopping early.")
            break

    elapsed = time.time() - t_start

    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "model": MODEL,
        "variant": "softcommit",
        "n_samples_per_transcript": N_SAMPLES,
        "temperature": TEMPERATURE,
        "pass_threshold": PASS_THRESHOLD,
        "n_transcripts": len(results),
        "elapsed_sec": round(elapsed, 1),
        "cost_usd": round(cost.usd, 4),
        "input_tokens": cost.in_tok,
        "output_tokens": cost.out_tok,
        "aggregate": {
            "pass_at_1": avg("pass_at_1"),
            "pass_at_8": avg("pass_at_8"),
            "mean_f1": avg("mean_f1"),
            "format_compliance": avg("format_compliance"),
        },
        "per_transcript": [
            {k: v for k, v in r.items() if k != "samples"} for r in results
        ],
    }

    raw = {r["transcript_id"]: r["samples"] for r in results}

    out_path = Path(OUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = out_path.with_name(out_path.stem + "_raw.json")

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(raw_path, "w") as f:
        json.dump(raw, f, indent=2)

    print()
    print("=" * 60)
    print(f"SOFTCOMMIT RESULTS  model={MODEL}  transcripts={len(results)}  n={N_SAMPLES}  T={TEMPERATURE}")
    print("=" * 60)
    print(f"pass@1            {_fmt(summary['aggregate']['pass_at_1'])}")
    print(f"pass@8            {_fmt(summary['aggregate']['pass_at_8'])}")
    print(f"mean F1           {_fmt(summary['aggregate']['mean_f1'])}")
    print(f"format compliance {_fmt(summary['aggregate']['format_compliance'])}")
    print(f"cost              ${summary['cost_usd']:.2f}")
    print(f"elapsed           {summary['elapsed_sec']}s")
    print()
    print(f"Saved: {out_path}, {raw_path}")


if __name__ == "__main__":
    asyncio.run(main())
