"""Run only V4 (oracle snippet) on the 3 failing transcripts."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
ENV_DIR = ROOT / "environments" / "meeting_intent"
sys.path.insert(0, str(ENV_DIR))

import run_ablations_v2 as core  # noqa: E402

from anthropic import AsyncAnthropic  # noqa: E402


async def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {core.MODEL}  variant: V4 only")
    print(f"Samples per transcript: {core.N_SAMPLES} at T={core.TEMPERATURE}")

    ground_truths = {}
    for tid in core.TRANSCRIPTS:
        gt = json.loads((core.DATA / f"{tid}_ground_truth.json").read_text())
        ground_truths[tid] = gt["action_items"]

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(core.CONCURRENCY)
    cost = core.Cost(5.0)  # fresh $5 cap
    t_start = time.time()

    out_dir = ROOT / "results" / "ablations"

    per = await core.run_variant("v4", client, cost, sem, ground_truths)

    summary = {
        "model": core.MODEL,
        "variant": "v4",
        "n_samples_per_transcript": core.N_SAMPLES,
        "temperature": core.TEMPERATURE,
        "pass_threshold": core.PASS_THRESHOLD,
        "transcripts": [r["transcript_id"] for r in per],
        "elapsed_sec": round(time.time() - t_start, 1),
        "cost_usd": round(cost.usd, 4),
        "per_transcript": [
            {k: v for k, v in r.items() if k != "samples"} for r in per
        ],
    }
    raw = {r["transcript_id"]: r["samples"] for r in per}
    with open(out_dir / "v4_oracle_snippet.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(out_dir / "v4_oracle_snippet_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    print()
    print(f"V4 cost ${cost.usd:.2f}  elapsed {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
