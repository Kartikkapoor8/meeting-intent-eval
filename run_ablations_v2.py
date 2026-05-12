"""Aggressive prompt ablations on the 3 currently-failing transcripts.

V2: dialogue-explicit prompt
V3: V2 + few-shot examples
V4: V3 + oracle snippet (only relevant evidence lines, not full transcript)
V5: direct "find it" prompt

Each variant: 8 samples per transcript, Opus 4.6, T=1.0.
3 transcripts x 4 variants x 8 = 96 calls total. Cost cap $10.
"""
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

from meeting_intent import (  # noqa: E402
    ANCHORS,
    canonicalize_due,
    canonicalize_owner,
    f1_reward,
    format_reward,
    parse_completion,
    score_extraction,
)

from anthropic import AsyncAnthropic  # noqa: E402

MODEL = os.environ.get("EVAL_MODEL", "claude-opus-4-6")
N_SAMPLES = int(os.environ.get("EVAL_N", "8"))
TEMPERATURE = float(os.environ.get("EVAL_T", "1.0"))
MAX_TOKENS = 1024
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "4"))
BUDGET_USD = float(os.environ.get("EVAL_BUDGET", "10.0"))
PASS_THRESHOLD = 0.99

PRICE_IN = 15.0 / 1_000_000
PRICE_OUT = 75.0 / 1_000_000

TRANSCRIPTS = ["real_earnings_002", "real_earnings_003", "real_client_001"]
DATA = ENV_DIR / "data"


# ---------- prompts ----------

V1_PROMPT = """You are extracting commitments from a meeting transcript. A commitment is a follow-up obligation where a specific person has agreed to do a specific thing. The agreement may be made through dialogue rather than a single explicit statement. Both parties may need to confirm through back-and-forth for the commitment to count.

Soft due values are valid: "offline", "ongoing", "next quarter", "EOD", "later today", "open", "after the meeting".

Include:
- Hedged but mutual commitments through dialogue (e.g., one party requests, another accepts even if the acceptance is soft)
- Follow-ups offered to specific questions (e.g., "we can go into that offline" when responding to an analyst question)
- Management-style commitments where a leader states the team will work toward something

Exclude:
- Generic prepared remarks like "we'll provide updates as we progress"
- Vague aspirations like "hopefully we can"
- Statements where no specific party is taking ownership

Return JSON with the same schema as before:
{"action_items": [{"owner": "...", "task": "...", "due": "..."}]}
"""

DIALOGUE_ADDENDUM = """

Pay specific attention to commitments that emerge through dialogue between two or more speakers. When one party requests something and another party agrees (even if the agreement is hedged, indirect, or implied through context), that is a valid commitment. Look for patterns where:
- One speaker asks for information or follow-up
- Another speaker offers to provide it
- The offer is the commitment, even without explicit 'I will do X by Y' phrasing
"""

FEW_SHOT_BLOCK = """

Examples of valid commitments (these ARE commitments):
- Analyst: "Can you break down the monthly numbers?" CEO: "We can go into the detail offline."
  -> Owner: CEO, Task: provide monthly breakdown offline, Due: offline
- Manager: "We need to figure out the timeline." Lead: "Let me work on that and circle back next week."
  -> Owner: Lead, Task: figure out timeline and report back, Due: next week
- Investor: "Will gross margins hold next year?" CFO: "We're going to work to try to maintain them in the mid-seventies."
  -> Owner: CFO, Task: maintain gross margins in mid-seventies, Due: next year

Examples that are NOT commitments:
- "We'll provide updates as we progress through the year." -> Generic prepared remark, no specific obligation.
- "Hopefully we can announce some new investments soon." -> Hedged aspiration, no specific party taking ownership.
"""

V2_PROMPT = V1_PROMPT + DIALOGUE_ADDENDUM
V3_PROMPT = V1_PROMPT + DIALOGUE_ADDENDUM + FEW_SHOT_BLOCK
V4_PROMPT = V3_PROMPT  # same prompt, different user-message body (snippet instead of full)

V5_PROMPT = """This transcript contains at least one commitment made through dialogue between participants. Your task is to identify it. The commitment may be soft, hedged, or implied. Look carefully at exchanges where one party requests something and another responds.

Output the commitment in JSON format:
{"action_items": [{"owner": "...", "task": "...", "due": "..."}]}

If after careful reading you cannot identify any commitment, output {"action_items": []} -- but only after genuine effort.

Soft due values are valid: "offline", "ongoing", "next quarter", "EOD", "later today", "open", "next year", "after the meeting".
"""


# ---------- oracle snippets ----------

# real_earnings_002: Stacy Rasgon -> Colette mid-70s GM commitment
EARN_002_SNIPPET = """Stacy Rasgon: Questions. Colette, I had some questions on margins. You said for next year, you're working to hold them in the mid-seventies. So I guess, first of all, what are the biggest cost increases? Is it just memory, or is it something else? What are you doing to work toward that? Is it how much is, like, you know, cost optimizations versus pre buys versus pricing And then also, how should we think about OpEx growth next year given the revenues seem likely to grow materially? From where we're running right now?

Colette Kress: Stacy. Let me see if I can start with remembering where we were with the current fiscal year that we're in. Remember earlier this year, we indicated that through cost improvements and mix that we would exit the year in our gross margins in the mid-seventies. We achieved that. So now it's time for us to communicate where are we And getting ready to also execute that in Q4. working right now in terms of next year. Next year, there are input prices. That are well known in the industries that we need to work through. And our systems are by no means very easy to work with. There are tremendous amount of components, many different parts of it as we think about that. So we're taking all of that into account, but we do believe if we look at working again on cost improvements, cycle time, and mix, that we will work to try and hold at our gross margins in the mid-seventies. So that's our overall plan for gross margin.

Jensen Huang: I think that's spot on. I think the only thing that would add is remember that we plan, we forecast, we plan, and we negotiate with our supply chain well in advance.
"""

# real_earnings_003: three commitments stitched together
EARN_003_SNIPPET = """[Dialogue chain 1 - tax season]
Michael Mayo: And then separately, Jeremy, you mentioned no change in the core NII despite being asset sensitive. And in terms of the deposit growth, you had some really amazing deposit growth and then you kind of hit an air pocket for a little while in this quarter, consumer deposits were up 2%. I guess taxes probably helped that out. Is this the start to getting back on that higher deposit growth path or not yet?

Jeremy Barnum: Well, I think air pocket is a little bit of a strong word, but fair enough. I recognize the dynamic that you're describing. And I think it's a little bit too early to sort of say, like, yay, like we're back with like super robust consumer deposit growth, partially because of your point actually about tax. I think you're right, that probably is contributing a little bit right now. But at a high level, we talked about at Company Update, our consumer deposit growth expectations being low to mid-single digits. And I think that is still the belief, and I think we'll be a little bit more confident in that, as you say, once we get through tax season. So maybe we'll know a little bit more next quarter.

[Dialogue chain 2 - add to list]
James Dimon: ... I want to remove that little thing that says cash returns to investors, which is a dividends and stock buyback. I don't particularly like that because I think it puts you in an artificial position thinking that's always a good thing when it's not.

Jeremy Barnum: Very well. I'll add that to list. Next question.

[Dialogue chain 3 - G-SIB arbitrage]
James Mitchell: Okay. And just a follow-up on the balance sheet growth in markets. It has been strong, I think, up over 20% year-over-year. Would you saying when you think about the impact of the G-SIB surcharge on JPMorgan specifically, does that start to impinge your ability to grow that as much as you want? How is that factoring into your capital decision in the Markets business?

Jeremy Barnum: I think the short answer is yes. And that's a big part of the reason that we spent -- the time that we spent today talking about the problem for the surcharge. It disproportionately accrues to the Markets business and disproportionately accrues to the relatively low risk density type of stuff that the client base really needs and wants these days. And that's why we think it's important that regulators think very carefully about what they're actually trying to achieve here.

James Dimon: I'll add one other thing. We will obviously use our brainpower to do something I don't like doing, which is trying to find a lot of ways to serve our clients properly and reduce the G-SIB charge, which is usually called arbitrage. So I'm not sure the outcome is great for the system, but we will find ways to do it.
"""

# real_client_001: three commitments stitched
CLIENT_001_SNIPPET = """[Dialogue chain 1 - 57 page document]
Client: ... what are your thoughts, Partner? Was that too much right now? Or should Engineer take a look at it and say, shit, in 20 minutes, this could really bring up a bunch of more dots on the on the map. What are your thoughts, Partner Engineer?

Partner: Yeah, I think my take is we should. Well, if you signed an NDA, right, like the whole purpose of that is to trust at that point ... we want to get that going in the places where things are going to be built out. So if it's helpful for Engineer to do that, if it's helpful for the University to do that, I think how we just protect all this stuff is with NDAs.

Client: Well, they would never. Yeah, I'm happy to send it to Engineer. Engineer, I could send it to you ... So I'll create on my side, a document file, and share it with you and Partner in case Partner's got nothing better to do but to Read 57 pages.

Client (a bit later): Well, tomorrow morning, I'll send to the two of you the 57 page document.

[Dialogue chain 2 - deploy link tonight]
Engineer: Any last questions about the App itself? What I'm going to do is deploy a link. And later tonight, it might be late because I'm come home a little bit late. Don't tell my mom. What I'll do is I'll send a link to you and then you'll be able to do on the App on your own and you could kind of check out the features by yourself and like log in and see how it works.

Client: I'm beyond pleased. Partner, thank you, Engineer.

[Dialogue chain 3 - timeline and reconvene]
Engineer (at close): What I'll do is I'll send a forward, like another just email. And then from here, what I see potentially in the future is just you can send me over that doc. I can add, I can finish up the timeline. I can add a couple more points and we can get it a little bit more functional and we can reconvene for a much shorter meeting, maybe 10, 20 minutes.

Client: That all sounds good to me, my friend.
"""

SNIPPETS = {
    "real_earnings_002": EARN_002_SNIPPET,
    "real_earnings_003": EARN_003_SNIPPET,
    "real_client_001": CLIENT_001_SNIPPET,
}


# ---------- helpers ----------

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


def make_user_message(tid: str, variant: str) -> str:
    if variant == "v4":
        body = SNIPPETS[tid]
        header = (
            "Below is a focused excerpt from a meeting transcript containing the most "
            "relevant exchanges. Extract ONLY the real, committed action items per the "
            "rules. Output JSON only.\n\n"
        )
    else:
        body = (DATA / f"{tid}.txt").read_text()
        header = (
            "Below is a meeting transcript. Extract ONLY the real, "
            "committed action items per the rules. Output JSON only.\n\n"
        )
    return f"{header}<transcript>\n{body}\n</transcript>"


def system_for(variant: str) -> str:
    return {
        "v2": V2_PROMPT,
        "v3": V3_PROMPT,
        "v4": V4_PROMPT,
        "v5": V5_PROMPT,
    }[variant]


def non_empty_count(samples):
    n = 0
    for s in samples:
        if s.get("error"):
            continue
        pred = parse_completion(s["text"])
        if pred is not None and len(pred) > 0:
            n += 1
    return n


def caught_any_gt(samples, gt, anchors):
    """Soft check: did at least one sample produce a prediction that matched
    ANY ground truth item (same owner+due AND task anchor hit) regardless of
    other false positives."""
    caught = 0
    for s in samples:
        if s.get("error"):
            continue
        pred = parse_completion(s["text"]) or []
        for p in pred:
            p_owner = canonicalize_owner(p.get("owner"))
            p_due = canonicalize_due(p.get("due"))
            p_task = (p.get("task") or "").lower() if isinstance(p.get("task"), str) else ""
            for i, g in enumerate(gt):
                g_owner = canonicalize_owner(g.get("owner"))
                g_due = canonicalize_due(g.get("due"))
                if p_owner != g_owner or p_due != g_due:
                    continue
                if not any(kw.lower() in p_task for kw in anchors[i]):
                    continue
                caught += 1
                break
            else:
                continue
            break
    return caught


async def run_variant(variant: str, client, cost, sem, ground_truths):
    print(f"\n=== Variant {variant.upper()} ===")
    per_transcript = []
    for tid in TRANSCRIPTS:
        if cost.over():
            print("BUDGET CAP - skipping remaining")
            break
        system = system_for(variant)
        user = make_user_message(tid, variant)
        gt = ground_truths[tid]
        anchors = ANCHORS[tid]

        print(f"[{variant} | {tid}] sampling {N_SAMPLES}...", flush=True)
        coros = [sample_one(client, system, user, sem, cost) for _ in range(N_SAMPLES)]
        samples = await asyncio.gather(*coros)

        info = {"transcript_id": tid, "action_items": gt}
        scores, fmt_oks = [], []
        successful = [s for s in samples if not s.get("error")]
        for s in successful:
            scores.append(f1_reward(completion=s["text"], info=info))
            fmt_oks.append(format_reward(completion=s["text"]))
        n = len(successful)
        c = sum(1 for sc in scores if sc >= PASS_THRESHOLD)

        ne = non_empty_count(samples)
        caught = caught_any_gt(samples, gt, anchors)

        row = {
            "transcript_id": tid,
            "n_attempted": len(samples),
            "n_successful": n,
            "n_errored": len(samples) - n,
            "n_passing": c,
            "n_non_empty": ne,
            "n_caught_any_gt": caught,
            "pass_at_1": pass_at_k(n, c, 1) if n >= 1 else 0.0,
            "pass_at_8": pass_at_k(n, c, 8) if n >= 8 else None,
            "mean_f1": (sum(scores) / max(n, 1)) if scores else 0.0,
            "format_compliance": (sum(fmt_oks) / max(n, 1)) if fmt_oks else 0.0,
            "scores": scores,
            "samples": samples,
        }
        per_transcript.append(row)
        print(
            f"  passing={c}/{n}  pass@1={row['pass_at_1']:.3f}  "
            f"non_empty={ne}/{n}  caught_any={caught}  "
            f"meanF1={row['mean_f1']:.3f}  spent=${cost.usd:.2f}",
            flush=True,
        )
    return per_transcript


async def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print(f"Model: {MODEL}")
    print(f"Transcripts: {TRANSCRIPTS}")
    print(f"Samples per (variant, transcript): {N_SAMPLES} at T={TEMPERATURE}")
    print(f"Concurrency: {CONCURRENCY}, budget cap: ${BUDGET_USD}")

    ground_truths = {}
    for tid in TRANSCRIPTS:
        gt = json.loads((DATA / f"{tid}_ground_truth.json").read_text())
        ground_truths[tid] = gt["action_items"]

    client = AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENCY)
    cost = Cost(BUDGET_USD)
    t_start = time.time()

    out_dir = ROOT / "results" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = ["v2", "v3", "v4", "v5"]
    out_files = {
        "v2": "v2_dialogue_explicit.json",
        "v3": "v3_few_shot.json",
        "v4": "v4_oracle_snippet.json",
        "v5": "v5_direct_prompt.json",
    }

    all_results = {}
    for variant in variants:
        per = await run_variant(variant, client, cost, sem, ground_truths)
        elapsed = time.time() - t_start
        summary = {
            "model": MODEL,
            "variant": variant,
            "n_samples_per_transcript": N_SAMPLES,
            "temperature": TEMPERATURE,
            "pass_threshold": PASS_THRESHOLD,
            "transcripts": [r["transcript_id"] for r in per],
            "elapsed_sec_cumulative": round(elapsed, 1),
            "cost_usd_cumulative": round(cost.usd, 4),
            "per_transcript": [
                {k: v for k, v in r.items() if k != "samples"} for r in per
            ],
        }
        raw = {r["transcript_id"]: r["samples"] for r in per}
        with open(out_dir / out_files[variant], "w") as f:
            json.dump(summary, f, indent=2)
        with open(out_dir / out_files[variant].replace(".json", "_raw.json"), "w") as f:
            json.dump(raw, f, indent=2)
        all_results[variant] = summary
        if cost.over():
            print(f"BUDGET CAP reached after variant {variant}, stopping.")
            break

    with open(out_dir / "ablations_v2_all.json", "w") as f:
        json.dump({"cost_usd": round(cost.usd, 4),
                   "elapsed_sec": round(time.time() - t_start, 1),
                   "variants": all_results}, f, indent=2)

    print()
    print("=" * 60)
    print("DONE")
    print(f"total cost ${cost.usd:.2f}  elapsed {time.time() - t_start:.1f}s")
    for variant in variants:
        if variant not in all_results:
            continue
        per = all_results[variant]["per_transcript"]
        line = f"  {variant}:"
        for r in per:
            line += f" {r['transcript_id']}={r['n_passing']}/{r['n_successful']}"
        print(line)


if __name__ == "__main__":
    asyncio.run(main())
