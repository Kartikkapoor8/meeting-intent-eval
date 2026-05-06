"""
meeting_intent — extract real action items from messy meeting transcripts.

A Prime Intellect verifiers environment. The model is given a casual meeting
transcript and must output structured JSON listing only the real, committed
action items — not casual mentions, hedged ideas, or "we should think about
that" parking-lot items.

The rubric is fully deterministic (no model in the loop):
  1. Parse the model's JSON output.
  2. For each predicted action item, canonicalize (owner, due) and require
     the task description to contain at least one anchor keyword for that
     ground-truth item.
  3. Reward = F1 over correctly-matched items, in [0, 1].
  4. Pass = reward >= 0.99 (effectively all-or-nothing per transcript).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset


DATA_DIR = Path(__file__).parent / "data"


SYSTEM_PROMPT = """You are reviewing a transcript from a casual, informal meeting. Your job is to extract ONLY the real action items — concrete commitments where a specific person committed to do a specific thing by a specific time.

Do NOT include:
- Casual ideas, suggestions, or things people said "we should" do "someday" / "eventually" / "at some point"
- Hedged statements ("maybe", "might", "I'll think about it", "I'll see what they say", "probably")
- Items explicitly deferred or sent to the parking lot
- Items where no specific owner was assigned ("someone should...")
- Items where no specific deadline was committed
- Vague follow-ups ("let's circle back", "we'll figure it out")

Output a SINGLE JSON object, with no surrounding prose, no explanation, no markdown code fences. Use exactly this shape:

{"action_items": [{"owner": "<single first name>", "task": "<short description>", "due": "<deadline as stated>"}]}

If there are no real action items, output {"action_items": []}.
"""


# Anchor keywords per ground-truth item (one OR-group per item, parallel to the
# action_items array in each transcript's ground_truth.json). For a predicted
# item to count as a semantic match, its `task` field must contain at least one
# keyword from this group (case-insensitive substring match) — preventing
# coincidental (owner, due) collisions from passing the rubric.
ANCHORS: dict[str, list[list[str]]] = {
    "brainstorm_001": [
        ["rank", "scor", "spike"],                   # Devon: search ranking spike
        ["mock", "figma", "hover", "filter panel"],  # Priya: filter panel mocks
    ],
    "design_crit_002": [
        ["mixpanel", "drop", "funnel", "metric"],         # Tom: drop-off data
        ["mock", "step", "v4", "revis", "redesign"],      # Sam: revise step 2 mocks
    ],
    "eng_standup_003": [
        ["pair", "auth", "token", "401", "bug"],          # Raj: pair on auth bug
        ["runbook", "migration", "doc"],                  # Maddie: runbook
    ],
    "customer_call_004": [
        ["soc", "report"],                                # Jess: SOC 2 report
        ["questionnaire"],                                # Mark: security questionnaire
        ["sandbox", "ingestion", "credentials", "environment"],  # Erin: sandbox env
    ],
    "exec_sync_005": [
        ["burn", "projection", "forecast", "q3"],         # Brent: burn projection
        ["comp", "framework", "leveling", "pre-read", "band"],  # Wes: comp framework
    ],
    "pipeline_review_006": [
        ["template", "pricing"],                                  # Lin: template
        ["quote", "acme", "pric"],                                # Marco: Acme quote
        ["security", "questionnaire", "soc", "review"],           # Dan: Volt security review
    ],
    "support_escalation_007": [
        ["log", "import", "investigat", "backup", "tenant"],      # Reza: investigate
        ["call", "cto", "schedule", "calendar", "set up", "meeting"],  # Hana: schedule call
        ["one-pager", "one pager", "writ", "draft", "explanation", "summar", "document"],  # Reza: one-pager
    ],
    "hiring_debrief_008": [
        ["leveling", "comp", "rec", "level rec", "l4"],           # Vik: leveling rec
        ["backchannel", "ping", "ref", "contact", "former colleague", "volta"],  # Sara: backchannel
        ["reject", "kp"],                                         # Mei: rejection email
    ],
    "brainstorm_no_commits_009": [],  # No real action items — empty list is correct
    "prod_triage_010": [
        ["latency", "p99", "api", "investig", "spike"],           # Raul: latency
        ["auth", "investig", "service"],                          # Min: auth investigation
        ["status page", "status", "post", "investigating", "update"],  # Raul: status page
    ],
    # Real public earnings-call transcripts (workplace-Q&A format).
    # Ground truth focuses on dialogue-confirmation commitments — softer than the
    # synthetic set, used to expose the failure pattern on real data.
    "real_earnings_001": [
        ["offline", "rent", "collect", "follow up", "follow-up", "modeling"],  # Ben: offline w/ Connor
    ],
    "real_earnings_002": [
        ["margin", "mid-seven", "mid-70", "gross", "seventies"],  # Colette: hold mid-70s GM
    ],
    "real_earnings_003": [
        ["mayo", "book value", "tangible", "round-trip", "list", "artificial"],  # Jeremy: add to list
        ["deposit", "tax season", "consumer", "next quarter"],                    # Jeremy: revisit next quarter
        ["g-sib", "gsib", "charge", "brainpower", "arbitrage", "capital"],         # Jamie: reduce G-SIB
    ],
}


WEEKDAYS = {
    "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
}
DAY_ABBR = {
    "mon": "monday", "tue": "tuesday", "tues": "tuesday",
    "wed": "wednesday", "weds": "wednesday",
    "thu": "thursday", "thur": "thursday", "thurs": "thursday",
    "fri": "friday", "sat": "saturday", "sun": "sunday",
}


def canonicalize_owner(s: Any) -> str:
    """Lowercase, strip non-alpha, take first token. 'Devon Smith' -> 'devon'."""
    if not isinstance(s, str):
        return ""
    cleaned = re.sub(r"[^a-z\s]", " ", s.lower()).strip()
    if not cleaned:
        return ""
    return cleaned.split()[0]


def canonicalize_due(s: Any) -> str:
    """Normalize a due-date string to one of {today, tomorrow, monday..sunday} or raw lowercase."""
    if not isinstance(s, str):
        return ""
    t = s.lower().strip()

    # Same-day variants (handle before stripping qualifiers like "this")
    if "tonight" in t:
        return "today"
    if any(p in t for p in ("this morning", "this afternoon", "this evening")):
        return "today"

    # Same-day implicit time-of-day expressions ("now", "asap", "before 4pm", "by 5:30")
    if t in ("now", "right now", "asap", "immediately"):
        return "today"
    if re.search(r"\b\d+\s*(am|pm)\b", t) or re.search(r"\b\d+:\d+\b", t) or re.search(r"\bnoon\b", t):
        return "today"

    # End-of-week variants → friday
    if re.search(r"\bend\s+of\s+(the\s+)?week\b", t) or re.search(r"\beow\b", t):
        return "friday"

    # Strip prepositions / qualifiers
    t = re.sub(r"\b(by|on|before|this|next|coming|the)\b", " ", t)
    # Strip end-of-day variants — treat "wednesday EOD" as "wednesday"
    t = re.sub(r"\b(end of day|eod|cob|close of business|end of business)\b", " ", t)
    # Strip punctuation
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    if "today" in t:
        return "today"
    if "tomorrow" in t:
        return "tomorrow"
    for tok in t.split():
        if tok in WEEKDAYS:
            return tok
        if tok in DAY_ABBR:
            return DAY_ABBR[tok]
    return t


def parse_completion(text: Any) -> list[dict] | None:
    """Extract the action_items list from a model response. Returns None on parse failure."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    # Strip a single fenced code block if present
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL)
    if fence:
        s = fence.group(1).strip()
    # Direct parse first
    obj: Any
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        # Fall back to first {...} block
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    items = obj.get("action_items")
    if not isinstance(items, list):
        return None
    # Filter to dict items only
    return [it for it in items if isinstance(it, dict)]


def score_extraction(
    predicted: list[dict],
    ground_truth: list[dict],
    anchors: list[list[str]],
) -> float:
    """F1 over correctly-matched action items (owner, due) + anchor-keyword check on task."""
    n_gt = len(ground_truth)
    n_pred = len(predicted)
    if n_gt == 0 and n_pred == 0:
        return 1.0
    if n_gt == 0 or n_pred == 0:
        return 0.0

    used_gt: set[int] = set()
    matched = 0
    for p in predicted:
        p_owner = canonicalize_owner(p.get("owner"))
        p_due = canonicalize_due(p.get("due"))
        p_task = (p.get("task") or "").lower() if isinstance(p.get("task"), str) else ""
        for i, g in enumerate(ground_truth):
            if i in used_gt:
                continue
            g_owner = canonicalize_owner(g.get("owner"))
            g_due = canonicalize_due(g.get("due"))
            if p_owner != g_owner or p_due != g_due:
                continue
            if not any(kw.lower() in p_task for kw in anchors[i]):
                continue
            matched += 1
            used_gt.add(i)
            break

    if matched == 0:
        return 0.0
    precision = matched / n_pred
    recall = matched / n_gt
    return 2 * precision * recall / (precision + recall)


def _completion_text(completion: Any) -> str:
    """Extract assistant text from either a plain string or a list of chat messages."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for msg in reversed(completion):
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and isinstance(p.get("text", ""), str)
                )
    return ""


def f1_reward(completion: Any, info: dict, **kwargs) -> float:
    """Main reward signal: F1 over correctly-extracted action items."""
    pred = parse_completion(_completion_text(completion))
    if pred is None:
        return 0.0
    transcript_id = info["transcript_id"]
    gt = info["action_items"]
    anchors = ANCHORS[transcript_id]
    return score_extraction(pred, gt, anchors)


def format_reward(completion: Any, **kwargs) -> float:
    """Tracked-only metric: 1.0 if completion parses to expected JSON shape, else 0.0."""
    return 1.0 if parse_completion(_completion_text(completion)) is not None else 0.0


def load_environment(**kwargs) -> vf.Environment:
    """Load the meeting_intent SingleTurnEnv from local data files."""
    rows: list[dict] = []
    for txt_path in sorted(DATA_DIR.glob("*.txt")):
        transcript_id = txt_path.stem
        gt_path = DATA_DIR / f"{transcript_id}_ground_truth.json"
        gt = json.loads(gt_path.read_text())
        transcript = txt_path.read_text()
        rows.append(
            {
                "question": (
                    "Below is a meeting transcript. Extract ONLY the real, "
                    "committed action items per the rules. Output JSON only.\n\n"
                    f"<transcript>\n{transcript}\n</transcript>"
                ),
                "answer": json.dumps(gt["action_items"]),
                "info": {
                    "transcript_id": transcript_id,
                    "action_items": gt["action_items"],
                },
                "task": "meeting_intent",
            }
        )

    dataset = Dataset.from_list(rows)
    rubric = vf.Rubric(funcs=[f1_reward, format_reward], weights=[1.0, 0.0])

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        **kwargs,
    )
