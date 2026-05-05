"""Smoke tests for the meeting_intent rubric. Run: python3 test_rubric.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from meeting_intent import (
    ANCHORS,
    DATA_DIR,
    canonicalize_due,
    canonicalize_owner,
    f1_reward,
    format_reward,
    parse_completion,
    score_extraction,
)


def color(s, code):
    return f"\033[{code}m{s}\033[0m"


def assert_eq(name, got, expected):
    ok = got == expected
    tag = color("PASS", 32) if ok else color("FAIL", 31)
    print(f"  [{tag}] {name}: got={got!r} expected={expected!r}")
    return ok


def assert_close(name, got, expected, tol=1e-6):
    ok = abs(got - expected) <= tol
    tag = color("PASS", 32) if ok else color("FAIL", 31)
    print(f"  [{tag}] {name}: got={got:.4f} expected={expected:.4f}")
    return ok


def section(title):
    print()
    print(color(f"=== {title} ===", 36))


passed, failed = 0, 0
def track(ok):
    global passed, failed
    passed += int(ok); failed += int(not ok)


# --- Canonicalization ---
section("canonicalize_owner")
for raw, exp in [
    ("Devon", "devon"),
    ("DEVON", "devon"),
    ("  Maddie  ", "maddie"),
    ("Devon Smith", "devon"),
    ("Raj/Chen", "raj"),
    ("Raj-Chen pair", "raj"),
    ("", ""),
    (None, ""),
]:
    track(assert_eq(f"owner({raw!r})", canonicalize_owner(raw), exp))

section("canonicalize_due")
for raw, exp in [
    ("Friday", "friday"),
    ("by Friday", "friday"),
    ("Wednesday EOD", "wednesday"),
    ("Wed", "wednesday"),
    ("next Wednesday", "wednesday"),
    ("tomorrow EOD", "tomorrow"),
    ("end of day today", "today"),
    ("today EOD", "today"),
    ("Monday.", "monday"),
    ("Tues", "tuesday"),
    ("EOD Friday", "friday"),
    ("tonight", "today"),
    ("tonight EOD", "today"),
    ("this afternoon", "today"),
    ("this evening", "today"),
    ("end of week", "friday"),
    ("end of the week", "friday"),
    ("by end of week", "friday"),
    ("EOW", "friday"),
    ("tomorrow afternoon", "tomorrow"),
    ("tomorrow morning", "tomorrow"),
    ("now", "today"),
    ("right now", "today"),
    ("asap", "today"),
    ("before 4pm", "today"),
    ("by 5:30pm", "today"),
    ("around noon", "today"),
    ("", ""),
]:
    track(assert_eq(f"due({raw!r})", canonicalize_due(raw), exp))


# --- Parse completion ---
section("parse_completion")
ok = parse_completion('{"action_items": [{"owner":"a","task":"b","due":"c"}]}')
track(assert_eq("plain JSON", ok, [{"owner":"a","task":"b","due":"c"}]))
ok = parse_completion('```json\n{"action_items": []}\n```')
track(assert_eq("fenced JSON", ok, []))
ok = parse_completion('Sure! Here is the JSON:\n{"action_items": [{"owner":"x","task":"y","due":"z"}]}\nThanks.')
track(assert_eq("prose-wrapped JSON", ok, [{"owner":"x","task":"y","due":"z"}]))
track(assert_eq("garbage", parse_completion("not json at all"), None))
track(assert_eq("missing key", parse_completion('{"foo": []}'), None))


# --- Per-transcript end-to-end on perfect / wrong / empty completions ---
def perfect_completion_for(transcript_id):
    gt_path = DATA_DIR / f"{transcript_id}_ground_truth.json"
    gt = json.loads(gt_path.read_text())
    items = [{"owner": x["owner"], "task": x["task"], "due": x["due"]} for x in gt["action_items"]]
    return json.dumps({"action_items": items})


def info_for(transcript_id):
    gt_path = DATA_DIR / f"{transcript_id}_ground_truth.json"
    gt = json.loads(gt_path.read_text())
    return {"transcript_id": transcript_id, "action_items": gt["action_items"]}


section("PERFECT completion -> 1.0 on every transcript")
for tid in sorted(ANCHORS.keys()):
    completion = perfect_completion_for(tid)
    info = info_for(tid)
    score = f1_reward(completion=completion, info=info)
    track(assert_close(f"f1[{tid}]", score, 1.0))
    track(assert_close(f"format[{tid}]", format_reward(completion=completion), 1.0))


section("EMPTY action_items -> correct (1.0 if GT empty, 0.0 if GT has items)")
for tid in sorted(ANCHORS.keys()):
    completion = '{"action_items": []}'
    info = info_for(tid)
    score = f1_reward(completion=completion, info=info)
    expected = 1.0 if not info["action_items"] else 0.0
    track(assert_close(f"f1[{tid}]", score, expected))


section("MALFORMED JSON -> 0.0 reward, 0.0 format")
for tid in sorted(ANCHORS.keys())[:1]:
    info = info_for(tid)
    track(assert_close(f"f1[{tid}]", f1_reward(completion="lol nope", info=info), 0.0))
    track(assert_close(f"format[{tid}]", format_reward(completion="lol nope"), 0.0))


section("OVER-EXTRACTION (correct items + 2 trap items) -> precision drops")
# brainstorm_001: 2 GT items. Add 2 trap items pulled from non_commitments.
tid = "brainstorm_001"
info = info_for(tid)
over = json.dumps({
    "action_items": [
        {"owner": "Devon", "task": "spike new search ranking", "due": "Friday"},
        {"owner": "Priya", "task": "send updated mocks with hover states", "due": "Wednesday EOD"},
        {"owner": "Priya", "task": "redesign empty state for zero-result filters", "due": "soon"},
        {"owner": "Maya", "task": "loop in Jen from support", "due": "this week"},
    ],
})
score = f1_reward(completion=over, info=info)
# matched=2, n_pred=4, n_gt=2 -> p=0.5 r=1 F1 = 2/3
track(assert_close("over-extract F1", score, 2/3))


section("UNDER-EXTRACTION (only 1 of 2 correct) -> recall drops")
under = json.dumps({
    "action_items": [
        {"owner": "Devon", "task": "spike a new ranking function", "due": "Friday"},
    ],
})
score = f1_reward(completion=under, info=info_for("brainstorm_001"))
# matched=1, n_pred=1, n_gt=2 -> p=1 r=0.5 F1 = 2/3
track(assert_close("under-extract F1", score, 2/3))


section("WRONG ANCHORS (right owner+due but task is wrong topic) -> 0 match")
wrong_topic = json.dumps({
    "action_items": [
        {"owner": "Devon", "task": "buy more snacks for the office", "due": "Friday"},
        {"owner": "Priya", "task": "buy more snacks for the office", "due": "Wednesday"},
    ],
})
score = f1_reward(completion=wrong_topic, info=info_for("brainstorm_001"))
track(assert_close("wrong-topic F1 = 0", score, 0.0))


section("DUE DATE PHRASING TOLERANCE")
flex = json.dumps({
    "action_items": [
        {"owner": "Devon", "task": "ranking spike behind a flag", "due": "Fri"},
        {"owner": "Priya", "task": "filter panel mocks with hover states", "due": "EOD Wednesday"},
    ],
})
score = f1_reward(completion=flex, info=info_for("brainstorm_001"))
track(assert_close("Fri / EOD Wednesday F1 = 1.0", score, 1.0))


# Final summary
print()
total = passed + failed
status = color(f"{passed}/{total} passed", 32 if failed == 0 else 31)
print(f"=== SUMMARY === {status}")
sys.exit(0 if failed == 0 else 1)
