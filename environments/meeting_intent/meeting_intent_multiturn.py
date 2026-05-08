"""
meeting_intent_multiturn — multi-turn variant of meeting_intent.

The agent sees the transcript chunk by chunk via tool calls. Instead of a
single-shot JSON dump, it builds up a notepad of action items as evidence
accumulates. Tools:

  - next_chunk()                     get the next transcript window
  - add_item(owner, task, due)       append to the notepad
  - revise_item(idx, owner?, task?, due?)  edit a notepad entry
  - remove_item(idx)                 delete a notepad entry
  - done()                           finish the rollout

The rubric is the same F1 scorer used by the single-turn env, applied to the
final notepad.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import verifiers as vf
from datasets import Dataset

from meeting_intent import (
    ANCHORS,
    DATA_DIR,
    parse_completion,  # noqa: F401  (kept for downstream callers)
    score_extraction,
)


# Speaker-turn regex. Matches a line that starts with a speaker label and a
# colon, e.g. "Devon:", "Alan Gold:", "Tom Catherwood:". Tolerates 1-4 word
# names. Bracketed scene markers like "[Brainstorm]" are not speaker turns and
# are kept attached to the next real turn.
SPEAKER_TURN_RE = re.compile(r"^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z\.]+){0,3}):\s")

# Chunking limits (per spec: 8-12 turns OR ~500 words, whichever comes first).
CHUNK_TARGET_TURNS = 10
CHUNK_MAX_TURNS = 12
CHUNK_MAX_WORDS = 500


SYSTEM_PROMPT = """You are reviewing a meeting transcript chunk by chunk to extract real action items. You have a notepad you build up with tool calls as you read.

Your tools:
  next_chunk()                              get the next chunk of the transcript, or "END" when finished
  add_item(owner, task, due)                add an action item to the notepad
  revise_item(idx, owner?, task?, due?)     edit a notepad entry by index, only changing fields you pass
  remove_item(idx)                          delete a notepad entry by index
  done()                                    finish and submit the notepad as your final answer

Workflow:
  1. Call next_chunk() to read the first chunk.
  2. After each chunk, add any new committed action items to the notepad with add_item.
  3. If a later chunk contradicts or refines an earlier item, use revise_item or remove_item.
  4. Keep calling next_chunk() until it returns "END".
  5. Then call done().

What counts as a real action item: a concrete commitment where a specific person committed to do a specific thing by a specific time.

Do NOT add to the notepad:
  - Casual ideas or suggestions ("we should", "someday", "eventually")
  - Hedged statements ("maybe", "might", "I'll think about it", "probably")
  - Items deferred or sent to the parking lot
  - Items where no specific owner was assigned
  - Items where no specific deadline was committed
  - Vague follow-ups ("let's circle back", "we'll figure it out")

Action item shape:
  owner: a single first name
  task: a short description
  due:  a deadline as stated (e.g. "Friday", "tomorrow", "EOD Wednesday")

If the meeting has no real action items, leave the notepad empty and call done().
"""


# --- Chunking ---------------------------------------------------------------


def split_into_speaker_turns(transcript: str) -> list[str]:
    """Split a transcript into speaker-turn strings.

    A turn starts at a line matching SPEAKER_TURN_RE and runs until the next
    such line. Header lines before the first speaker turn are kept as a
    preamble turn so no content is lost.
    """
    lines = transcript.splitlines()
    turns: list[str] = []
    buf: list[str] = []
    for line in lines:
        if SPEAKER_TURN_RE.match(line):
            if buf:
                turns.append("\n".join(buf).rstrip())
            buf = [line]
        else:
            buf.append(line)
    if buf:
        turns.append("\n".join(buf).rstrip())
    # Drop empty turns (e.g. trailing whitespace blocks).
    return [t for t in turns if t.strip()]


def chunk_speaker_turns(turns: list[str]) -> list[str]:
    """Group speaker turns into windows.

    A window closes when either the turn count reaches CHUNK_TARGET_TURNS and
    we're at a sentence-ish boundary, or hard-caps at CHUNK_MAX_TURNS, or
    word count exceeds CHUNK_MAX_WORDS.
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for turn in turns:
        cur.append(turn)
        cur_words += len(turn.split())
        n = len(cur)
        if n >= CHUNK_MAX_TURNS or cur_words >= CHUNK_MAX_WORDS or n >= CHUNK_TARGET_TURNS:
            chunks.append("\n\n".join(cur))
            cur = []
            cur_words = 0
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def build_chunks(transcript: str) -> list[str]:
    """Top-level helper: transcript text -> list of chunk strings."""
    turns = split_into_speaker_turns(transcript)
    if not turns:
        # Fallback: paragraph split on blank lines.
        paras = [p.strip() for p in transcript.split("\n\n") if p.strip()]
        return chunk_speaker_turns(paras) if paras else [transcript]
    return chunk_speaker_turns(turns)


# --- Tools ------------------------------------------------------------------
#
# Each tool takes `state` as a hidden arg, injected by update_tool_args. The
# agent sees only the user-visible parameters in the function signature.


def next_chunk(state: dict) -> str:
    """Return the next transcript chunk, or "END" when the transcript is exhausted.

    Args:
        (no agent-visible args)

    Returns:
        The next chunk of the transcript as a string, or the literal string
        "END" once all chunks have been returned.
    """
    chunks = state.get("chunks", [])
    idx = state.get("chunk_idx", 0)
    if idx >= len(chunks):
        return "END"
    chunk = chunks[idx]
    state["chunk_idx"] = idx + 1
    return f"[chunk {idx + 1}/{len(chunks)}]\n{chunk}"


def add_item(owner: str, task: str, due: str, state: dict) -> str:
    """Append an action item to the notepad.

    Args:
        owner: Single first name of the person responsible.
        task: Short description of the action item.
        due: Deadline as stated in the transcript.

    Returns:
        Confirmation string showing the new index and full notepad size.
    """
    notepad = state.setdefault("notepad", [])
    item = {"owner": owner, "task": task, "due": due}
    notepad.append(item)
    return f"added at idx {len(notepad) - 1}; notepad now has {len(notepad)} item(s)"


def revise_item(
    idx: int,
    state: dict,
    owner: str | None = None,
    task: str | None = None,
    due: str | None = None,
) -> str:
    """Edit a notepad entry. Only fields you pass are changed.

    Args:
        idx: 0-based index of the notepad entry to edit.
        owner: New owner, or omit to leave unchanged.
        task: New task description, or omit to leave unchanged.
        due: New deadline, or omit to leave unchanged.

    Returns:
        Confirmation string with the updated entry, or an error if idx is bad.
    """
    notepad = state.setdefault("notepad", [])
    if not isinstance(idx, int) or idx < 0 or idx >= len(notepad):
        return f"error: idx {idx} out of range (notepad has {len(notepad)} items)"
    entry = notepad[idx]
    if owner is not None:
        entry["owner"] = owner
    if task is not None:
        entry["task"] = task
    if due is not None:
        entry["due"] = due
    return f"revised idx {idx}: {entry}"


def remove_item(idx: int, state: dict) -> str:
    """Delete a notepad entry by index.

    Args:
        idx: 0-based index of the entry to remove.

    Returns:
        Confirmation string, or an error if idx is bad.
    """
    notepad = state.setdefault("notepad", [])
    if not isinstance(idx, int) or idx < 0 or idx >= len(notepad):
        return f"error: idx {idx} out of range (notepad has {len(notepad)} items)"
    removed = notepad.pop(idx)
    return f"removed idx {idx} ({removed}); notepad now has {len(notepad)} item(s)"


def done(state: dict) -> str:
    """Finish the rollout. The notepad becomes the final answer.

    Args:
        (no agent-visible args)

    Returns:
        Confirmation string with the final notepad size.
    """
    state["final_env_response"] = "done"
    notepad = state.get("notepad", [])
    return f"done. final notepad has {len(notepad)} item(s)."


# --- Reward ------------------------------------------------------------------


def f1_reward_from_state(state: Any = None, info: Any = None, **kwargs) -> float:
    """F1 over the final notepad. Reuses the existing scorer.

    Falls back to 0.0 if state or info is missing.
    """
    if not isinstance(state, dict) or not isinstance(info, dict):
        return 0.0
    notepad = state.get("notepad", [])
    if not isinstance(notepad, list):
        return 0.0
    transcript_id = info.get("transcript_id")
    if transcript_id not in ANCHORS:
        return 0.0
    gt = info.get("action_items", [])
    anchors = ANCHORS[transcript_id]
    pred = [it for it in notepad if isinstance(it, dict)]
    return score_extraction(pred, gt, anchors)


def done_called_reward(state: Any = None, **kwargs) -> float:
    """Tracked-only metric: 1.0 if the agent explicitly called done()."""
    if not isinstance(state, dict):
        return 0.0
    return 1.0 if state.get("final_env_response") is not None else 0.0


# --- Environment -------------------------------------------------------------


class MeetingIntentMultiTurnEnv(vf.StatefulToolEnv):
    """StatefulToolEnv that streams a meeting transcript chunk by chunk.

    State shape (set by setup_state):
        chunks:        list[str]       pre-chunked transcript windows
        chunk_idx:     int             pointer into chunks
        notepad:       list[dict]      running action item list
        info:          dict            transcript_id + ground truth (carried through)
    """

    async def setup_state(self, state: vf.State, **kwargs) -> vf.State:
        # `info` is populated by the rollout from the dataset row's "info" field.
        info = state.get("info") or {}
        transcript = info.get("transcript", "")
        state["chunks"] = build_chunks(transcript) if transcript else []
        state["chunk_idx"] = 0
        state["notepad"] = []
        return state

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs,
    ) -> dict:
        # All five tools take a hidden `state` arg. Inject it here so the
        # agent never sees `state` in the schema.
        if tool_name in ("next_chunk", "add_item", "revise_item", "remove_item", "done"):
            tool_args = dict(tool_args)
            tool_args["state"] = state
        return tool_args


def load_environment_multiturn(**kwargs) -> vf.Environment:
    """Load the multi-turn variant of meeting_intent."""
    rows: list[dict] = []
    for txt_path in sorted(DATA_DIR.glob("*.txt")):
        transcript_id = txt_path.stem
        gt_path = DATA_DIR / f"{transcript_id}_ground_truth.json"
        gt = json.loads(gt_path.read_text())
        transcript = txt_path.read_text()
        rows.append(
            {
                "question": (
                    "You are about to review a meeting transcript chunk by chunk. "
                    "Start by calling next_chunk(), then add/revise/remove items as "
                    "evidence accumulates, and call done() when next_chunk() returns END."
                ),
                "answer": json.dumps(gt["action_items"]),
                "info": {
                    "transcript_id": transcript_id,
                    "action_items": gt["action_items"],
                    "transcript": transcript,
                },
                "task": "meeting_intent_multiturn",
            }
        )

    dataset = Dataset.from_list(rows)
    rubric = vf.Rubric(
        funcs=[f1_reward_from_state, done_called_reward],
        weights=[1.0, 0.0],
    )

    max_turns = kwargs.pop("max_turns", 50)
    timeout_seconds = kwargs.pop("timeout_seconds", 300)

    env = MeetingIntentMultiTurnEnv(
        tools=[],  # added below with hidden state arg
        max_turns=max_turns,
        timeout_seconds=timeout_seconds,
        dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        **kwargs,
    )
    env.add_tool(next_chunk, args_to_skip=["state"])
    env.add_tool(add_item, args_to_skip=["state"])
    env.add_tool(revise_item, args_to_skip=["state"])
    env.add_tool(remove_item, args_to_skip=["state"])
    env.add_tool(done, args_to_skip=["state"])
    return env
