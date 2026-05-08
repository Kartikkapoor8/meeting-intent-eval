"""Compare single-turn and multi-turn meeting_intent on the same 4 real transcripts.

Both runs hit the Anthropic Messages API directly. The single-turn loop matches
run_eval.py. The multi-turn loop drives the StatefulToolEnv tool functions
locally so we can use Anthropic's native tool-use format and track per-rollout
tool stats.

Shared $15 budget cap across both runs. Single-turn first, then multi-turn with
whatever's left.
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

from meeting_intent import (  # noqa: E402
    ANCHORS,
    DATA_DIR,
    f1_reward,
    format_reward,
    load_environment,
    score_extraction,
)
from meeting_intent_multiturn import (  # noqa: E402
    SYSTEM_PROMPT as MT_SYSTEM_PROMPT,
    add_item,
    build_chunks,
    done,
    load_environment_multiturn,
    next_chunk,
    remove_item,
    revise_item,
)

from anthropic import AsyncAnthropic  # noqa: E402

MODEL = "claude-opus-4-6"
N_SAMPLES = 8
TEMPERATURE = 1.0
MAX_TOKENS = 1024
MAX_TURNS_MT = 30
BUDGET_USD = 15.0
CONCURRENCY = 6
PASS_THRESHOLD = 0.99

PRICE_IN = 15.0 / 1_000_000
PRICE_OUT = 75.0 / 1_000_000

TRANSCRIPTS = [
    "real_earnings_001",
    "real_earnings_002",
    "real_earnings_003",
    "real_client_001",
]

RESULTS_DIR = Path("results")


# --- shared infra ---------------------------------------------------------


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


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


# --- single-turn rollout ---------------------------------------------------


async def single_turn_rollout(client, system, user, sem, cost, attempt=0):
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
                return await single_turn_rollout(client, system, user, sem, cost, attempt + 1)
            return {"text": "", "in_tok": 0, "out_tok": 0, "error": str(e)[:200]}
        text = "".join(getattr(b, "text", "") for b in resp.content)
        cost.add(resp.usage.input_tokens, resp.usage.output_tokens)
        return {"text": text, "in_tok": resp.usage.input_tokens, "out_tok": resp.usage.output_tokens}


async def run_single_turn(client, sem, cost):
    env = load_environment()
    rows_by_id = {r["info"]["transcript_id"]: r for r in env.dataset}
    results = []
    t_start = time.time()

    for tid in TRANSCRIPTS:
        if tid not in rows_by_id:
            print(f"[ST] missing transcript {tid}, skipping", flush=True)
            continue
        row = rows_by_id[tid]
        prompt = row["prompt"]
        system = next((m["content"] for m in prompt if m["role"] == "system"), "")
        user = next(m["content"] for m in prompt if m["role"] == "user")

        print(f"[ST {tid}] sampling {N_SAMPLES}...", flush=True)
        coros = [single_turn_rollout(client, system, user, sem, cost) for _ in range(N_SAMPLES)]
        samples = await asyncio.gather(*coros)

        scores, fmt_oks = [], []
        for s in samples:
            scores.append(f1_reward(completion=s["text"], info=row["info"]))
            fmt_oks.append(format_reward(completion=s["text"]))

        n = sum(1 for s in samples if not s.get("error"))
        c = sum(1 for s in scores[:n] if s >= PASS_THRESHOLD)
        n_eff = max(n, 1)
        rr = {
            "transcript_id": tid,
            "n_attempted": len(samples),
            "n_successful": n,
            "n_passing": c,
            "pass_at_1": pass_at_k(n, c, 1) if n >= 1 else 0.0,
            "pass_at_8": pass_at_k(n, c, 8) if n >= 8 else None,
            "mean_f1": sum(scores) / n_eff,
            "format_compliance": sum(fmt_oks) / n_eff,
            "scores": scores,
            "samples": samples,
        }
        results.append(rr)
        print(
            f"  passing={c}/{n}  pass@1={rr['pass_at_1']:.3f}  "
            f"meanF1={rr['mean_f1']:.3f}  fmt={rr['format_compliance']:.3f}  "
            f"spent=${cost.usd:.2f}",
            flush=True,
        )
        if cost.over():
            print("[ST] BUDGET CAP, stopping single-turn early.")
            break

    elapsed = time.time() - t_start
    return _summarize(results, elapsed, cost, variant="singleturn")


# --- multi-turn rollout ----------------------------------------------------


# Anthropic tool defs derived from the verifiers env (state is hidden).
def build_anthropic_tools():
    env = load_environment_multiturn()
    out = []
    for td in env.tool_defs:
        out.append({"name": td.name, "description": td.description, "input_schema": td.parameters})
    return out


def call_local_tool(name: str, args: dict, state: dict) -> str:
    """Run one of the env's tool functions against the local state dict."""
    if name == "next_chunk":
        return next_chunk(state)
    if name == "add_item":
        return add_item(args.get("owner", ""), args.get("task", ""), args.get("due", ""), state)
    if name == "revise_item":
        return revise_item(
            args.get("idx", -1),
            state,
            owner=args.get("owner"),
            task=args.get("task"),
            due=args.get("due"),
        )
    if name == "remove_item":
        return remove_item(args.get("idx", -1), state)
    if name == "done":
        return done(state)
    return f"error: unknown tool {name}"


async def multi_turn_rollout(client, system, transcript, info, tools, sem, cost):
    """Run one multi-turn rollout. Returns dict with score and per-rollout stats."""
    chunks = build_chunks(transcript)
    state: dict = {"chunks": chunks, "chunk_idx": 0, "notepad": []}

    user_seed = (
        "You are about to review a meeting transcript chunk by chunk. "
        "Start by calling next_chunk(), then add/revise/remove items as evidence "
        "accumulates, and call done() when next_chunk() returns END."
    )
    messages: list[dict] = [{"role": "user", "content": user_seed}]
    turns = 0
    tool_calls: list[str] = []
    error = None

    while turns < MAX_TURNS_MT:
        if cost.over():
            error = "budget_cap"
            break
        try:
            async with sem:
                resp = await client.messages.create(
                    model=MODEL,
                    system=system,
                    messages=messages,
                    tools=tools,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                )
        except Exception as e:
            error = str(e)[:200]
            break
        cost.add(resp.usage.input_tokens, resp.usage.output_tokens)
        turns += 1

        # Append assistant content as-is so subsequent turns are valid.
        assistant_blocks = [_block_to_dict(b) for b in resp.content]
        messages.append({"role": "assistant", "content": assistant_blocks})

        if resp.stop_reason != "tool_use":
            # Agent stopped without tools (likely emitted plain text). End rollout.
            break

        # Execute every tool_use block, build user/tool_result reply.
        tool_results = []
        for block in resp.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            tool_calls.append(block.name)
            result_text = call_local_tool(block.name, dict(block.input or {}), state)
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result_text,
                }
            )
            if state.get("final_env_response") is not None:
                break

        messages.append({"role": "user", "content": tool_results})

        if state.get("final_env_response") is not None:
            break

    score = score_extraction(state["notepad"], info["action_items"], ANCHORS[info["transcript_id"]])
    return {
        "score": score,
        "turns": turns,
        "tool_calls": tool_calls,
        "n_tool_calls": len(tool_calls),
        "called_done": state.get("final_env_response") is not None,
        "called_revise": "revise_item" in tool_calls,
        "called_remove": "remove_item" in tool_calls,
        "notepad": state["notepad"],
        "error": error,
    }


def _block_to_dict(block):
    """Convert an Anthropic content block to a serializable dict for replay."""
    t = getattr(block, "type", None)
    if t == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if t == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": dict(block.input or {}),
        }
    if t == "thinking":
        return {"type": "thinking", "thinking": getattr(block, "thinking", "")}
    # Fallback for unknown block types
    return {"type": t, "raw": str(block)[:500]}


async def run_multi_turn(client, sem, cost):
    env = load_environment_multiturn()
    rows_by_id = {r["info"]["transcript_id"]: r for r in env.dataset}
    tools = build_anthropic_tools()
    results = []
    t_start = time.time()

    for tid in TRANSCRIPTS:
        if tid not in rows_by_id:
            print(f"[MT] missing transcript {tid}, skipping", flush=True)
            continue
        row = rows_by_id[tid]
        info = row["info"]
        transcript = info["transcript"]

        if cost.over():
            print("[MT] BUDGET CAP before starting transcript, skipping rest.")
            break

        print(f"[MT {tid}] sampling {N_SAMPLES}...", flush=True)
        coros = [
            multi_turn_rollout(client, MT_SYSTEM_PROMPT, transcript, info, tools, sem, cost)
            for _ in range(N_SAMPLES)
        ]
        samples = await asyncio.gather(*coros)

        scores = [s["score"] for s in samples if s.get("error") is None]
        n = len(scores)
        c = sum(1 for s in scores if s >= PASS_THRESHOLD)
        n_eff = max(n, 1)
        # Format compliance for multi-turn = called done() (the structured "I'm finished" signal)
        fmt = sum(1 for s in samples if s.get("called_done")) / max(len(samples), 1)

        rr = {
            "transcript_id": tid,
            "n_attempted": len(samples),
            "n_successful": n,
            "n_passing": c,
            "pass_at_1": pass_at_k(n, c, 1) if n >= 1 else 0.0,
            "pass_at_8": pass_at_k(n, c, 8) if n >= 8 else None,
            "mean_f1": sum(scores) / n_eff,
            "format_compliance": fmt,
            "scores": scores,
            "mean_turns": sum(s["turns"] for s in samples) / max(len(samples), 1),
            "mean_tool_calls": sum(s["n_tool_calls"] for s in samples) / max(len(samples), 1),
            "n_called_done": sum(1 for s in samples if s.get("called_done")),
            "n_called_revise": sum(1 for s in samples if s.get("called_revise")),
            "n_called_remove": sum(1 for s in samples if s.get("called_remove")),
            "samples": samples,
        }
        results.append(rr)
        print(
            f"  passing={c}/{n}  pass@1={rr['pass_at_1']:.3f}  "
            f"meanF1={rr['mean_f1']:.3f}  done={rr['n_called_done']}/{len(samples)}  "
            f"meanTurns={rr['mean_turns']:.1f}  meanCalls={rr['mean_tool_calls']:.1f}  "
            f"revise={rr['n_called_revise']}  remove={rr['n_called_remove']}  "
            f"spent=${cost.usd:.2f}",
            flush=True,
        )
        if cost.over():
            print("[MT] BUDGET CAP, stopping multi-turn early.")
            break

    elapsed = time.time() - t_start
    return _summarize(results, elapsed, cost, variant="multiturn")


# --- summary --------------------------------------------------------------


def _summarize(results, elapsed, cost, variant):
    def avg(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "variant": variant,
        "model": MODEL,
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
            {k: v for k, v in r.items() if k != "samples"}
            for r in results
        ],
        "raw": results,
    }
    return summary


def save(summary, name):
    RESULTS_DIR.mkdir(exist_ok=True)
    public = {k: v for k, v in summary.items() if k != "raw"}
    with open(RESULTS_DIR / f"eval_comparison_{name}.json", "w") as f:
        json.dump(public, f, indent=2)
    raw_dump = {r["transcript_id"]: r["samples"] for r in summary["raw"]}
    with open(RESULTS_DIR / f"raw_samples_comparison_{name}.json", "w") as f:
        json.dump(raw_dump, f, indent=2, default=str)
    print(f"Saved results/eval_comparison_{name}.json + raw_samples_comparison_{name}.json")


async def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    cost = Cost(BUDGET_USD)

    print(f"Model: {MODEL}  budget cap: ${BUDGET_USD}  concurrency: {CONCURRENCY}")
    print(f"Transcripts: {TRANSCRIPTS}")
    print(f"Samples per transcript per variant: {N_SAMPLES}")
    print()

    print("=" * 60, flush=True)
    print("PHASE 1: SINGLE-TURN", flush=True)
    print("=" * 60, flush=True)
    st_summary = await run_single_turn(client, sem, cost)
    save(st_summary, "singleturn")
    print(f"Single-turn done. Spent ${cost.usd:.2f} so far.", flush=True)

    print()
    print("=" * 60, flush=True)
    print("PHASE 2: MULTI-TURN", flush=True)
    print("=" * 60, flush=True)
    mt_summary = await run_multi_turn(client, sem, cost)
    save(mt_summary, "multiturn")

    print()
    print("=" * 60)
    print(f"FINAL SPEND: ${cost.usd:.2f}")
    print(f"  input tokens:  {cost.in_tok:,}")
    print(f"  output tokens: {cost.out_tok:,}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
