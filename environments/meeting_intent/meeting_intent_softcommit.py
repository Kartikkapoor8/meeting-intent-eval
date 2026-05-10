"""
meeting_intent_softcommit — soft-commitment prompt ablation of meeting_intent.

Same dataset, same rubric, same schema as meeting_intent. The only thing that
changes is the system prompt. The original prompt explicitly excludes hedged
language ("maybe", "might", "let's circle back", "we'll figure it out") and
demands "specific person, specific thing, specific time."

The ground truth on real earnings calls includes exactly those things:
"offline" follow-ups, "ongoing" commitments, "next quarter" promises, and
management-style "we'll work to" language. So the model is being told to
reject items that the labels say are correct.

This variant rewrites the prompt to be permissive about soft due values and
dialogue-confirmation commitments, while still excluding generic prepared
remarks. If pass@1 moves materially under this prompt, the strict result was
inflated by prompt-label mismatch. If pass@1 stays at zero, the failure is
not about prompt design.
"""

from __future__ import annotations

import json

import verifiers as vf
from datasets import Dataset

from meeting_intent import (
    ANCHORS,  # noqa: F401  (re-exported for tests / parity checks)
    DATA_DIR,
    canonicalize_due,  # noqa: F401
    canonicalize_owner,  # noqa: F401
    f1_reward,
    format_reward,
    load_environment,  # noqa: F401  (kept importable for parity)
    parse_completion,  # noqa: F401
    score_extraction,  # noqa: F401
)


SOFTCOMMIT_SYSTEM_PROMPT = """You are extracting commitments from a meeting transcript. A commitment is a follow-up obligation where a specific person has agreed to do a specific thing. The agreement may be made through dialogue rather than a single explicit statement. Both parties may need to confirm through back-and-forth for the commitment to count.

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


def load_environment_softcommit(**kwargs) -> vf.Environment:
    """Load meeting_intent with the soft-commitment system prompt swapped in.

    Identical dataset rows, identical rubric, identical schema. The only
    change is the system_prompt fed to the model. Apples-to-apples ablation.
    """
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
                "task": "meeting_intent_softcommit",
            }
        )

    dataset = Dataset.from_list(rows)
    rubric = vf.Rubric(funcs=[f1_reward, format_reward], weights=[1.0, 0.0])

    return vf.SingleTurnEnv(
        dataset=dataset,
        system_prompt=SOFTCOMMIT_SYSTEM_PROMPT,
        rubric=rubric,
        **kwargs,
    )
