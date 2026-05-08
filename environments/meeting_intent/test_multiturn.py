"""Smoke tests for the multi-turn variant. Run: python3 test_multiturn.py

Covers:
  - chunking (speaker-turn split, window sizes)
  - tool mutation of state (add / revise / remove / next_chunk / done)
  - done() terminates the env (final_env_response is set)
  - the final notepad in state flows through the rubric
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from meeting_intent import ANCHORS, DATA_DIR
from meeting_intent_multiturn import (
    CHUNK_MAX_TURNS,
    CHUNK_MAX_WORDS,
    add_item,
    build_chunks,
    done,
    done_called_reward,
    f1_reward_from_state,
    load_environment_multiturn,
    next_chunk,
    remove_item,
    revise_item,
    split_into_speaker_turns,
)


def color(s, code):
    return f"\033[{code}m{s}\033[0m"


passed, failed = 0, 0


def assert_eq(name, got, expected):
    global passed, failed
    ok = got == expected
    tag = color("PASS", 32) if ok else color("FAIL", 31)
    print(f"  [{tag}] {name}: got={got!r} expected={expected!r}")
    passed += int(ok)
    failed += int(not ok)


def assert_true(name, got):
    global passed, failed
    ok = bool(got)
    tag = color("PASS", 32) if ok else color("FAIL", 31)
    print(f"  [{tag}] {name}: got={got!r}")
    passed += int(ok)
    failed += int(not ok)


def assert_close(name, got, expected, tol=1e-6):
    global passed, failed
    ok = abs(got - expected) <= tol
    tag = color("PASS", 32) if ok else color("FAIL", 31)
    print(f"  [{tag}] {name}: got={got:.4f} expected={expected:.4f}")
    passed += int(ok)
    failed += int(not ok)


def section(title):
    print()
    print(color(f"=== {title} ===", 36))


# --- Chunking ---
section("split_into_speaker_turns")
sample = """[Header line]
Maya: hello there.
Devon: hi.

Priya: how are you?
Maya: ok ok."""
turns = split_into_speaker_turns(sample)
# Header is kept as its own preamble turn (so content isn't dropped); then 4 speaker turns.
assert_eq("turn count", len(turns), 5)
assert_true("first turn is the header", turns[0].startswith("[Header line]"))
assert_true("second turn is Maya: hello", turns[1].startswith("Maya:"))
assert_true("last turn is Maya: ok ok.", turns[-1].startswith("Maya: ok ok"))


section("build_chunks on real transcript")
brainstorm_txt = (DATA_DIR / "brainstorm_001.txt").read_text()
chunks = build_chunks(brainstorm_txt)
assert_true("at least one chunk", len(chunks) >= 1)
assert_true("each chunk under hard caps", all(
    len(c.split()) <= CHUNK_MAX_WORDS + 200 and  # words can overshoot a bit on the last turn
    c.count("\n\n") + 1 <= CHUNK_MAX_TURNS + 1
    for c in chunks
))
# Concatenated chunks should cover most of the original (allow whitespace differences).
joined = "\n\n".join(chunks)
assert_true("chunk content non-trivial", len(joined) > 0.5 * len(brainstorm_txt))


section("build_chunks on long real earnings call")
earnings_txt = (DATA_DIR / "real_earnings_001.txt").read_text()
echunks = build_chunks(earnings_txt)
assert_true("multiple chunks for long transcript", len(echunks) >= 2)


# --- Tool mutation of state ---
section("next_chunk() mutates chunk_idx and returns chunks then END")
state = {"chunks": ["a", "b", "c"], "chunk_idx": 0, "notepad": []}
out0 = next_chunk(state)
out1 = next_chunk(state)
out2 = next_chunk(state)
out3 = next_chunk(state)
assert_true("chunk 0 contains 'a'", "a" in out0)
assert_true("chunk 1 contains 'b'", "b" in out1)
assert_true("chunk 2 contains 'c'", "c" in out2)
assert_eq("after exhaust returns END", out3, "END")
assert_eq("chunk_idx advanced past end", state["chunk_idx"], 3)


section("add_item appends to notepad")
state = {"chunks": [], "chunk_idx": 0, "notepad": []}
add_item("Devon", "spike search ranking", "Friday", state)
add_item("Priya", "filter panel mocks", "Wednesday EOD", state)
assert_eq("notepad length after 2 adds", len(state["notepad"]), 2)
assert_eq("first owner", state["notepad"][0]["owner"], "Devon")
assert_eq("second due", state["notepad"][1]["due"], "Wednesday EOD")


section("revise_item only changes provided fields")
state = {"notepad": [{"owner": "Devon", "task": "old task", "due": "Friday"}]}
revise_item(0, state, task="new task")
assert_eq("task updated", state["notepad"][0]["task"], "new task")
assert_eq("owner unchanged", state["notepad"][0]["owner"], "Devon")
assert_eq("due unchanged", state["notepad"][0]["due"], "Friday")
revise_item(0, state, owner="Sam", due="Monday")
assert_eq("owner now Sam", state["notepad"][0]["owner"], "Sam")
assert_eq("due now Monday", state["notepad"][0]["due"], "Monday")
# Out-of-range idx returns an error string and does not crash.
err = revise_item(99, state, task="x")
assert_true("revise_item out-of-range returns error", "out of range" in err)


section("remove_item drops by index")
state = {"notepad": [
    {"owner": "A", "task": "1", "due": "Mon"},
    {"owner": "B", "task": "2", "due": "Tue"},
    {"owner": "C", "task": "3", "due": "Wed"},
]}
remove_item(1, state)
assert_eq("notepad length after remove", len(state["notepad"]), 2)
assert_eq("first remaining is A", state["notepad"][0]["owner"], "A")
assert_eq("second remaining is C", state["notepad"][1]["owner"], "C")
err = remove_item(99, state)
assert_true("remove_item out-of-range returns error", "out of range" in err)


section("done() sets final_env_response")
state = {"notepad": [{"owner": "A", "task": "x", "due": "Mon"}]}
assert_eq("final_env_response unset before done", state.get("final_env_response"), None)
done(state)
assert_true("final_env_response set after done", state.get("final_env_response") is not None)


# --- Final notepad flows through rubric ---
section("f1_reward_from_state on a perfect notepad")
tid = "brainstorm_001"
gt = json.loads((DATA_DIR / f"{tid}_ground_truth.json").read_text())["action_items"]
state = {
    "notepad": [{"owner": x["owner"], "task": x["task"], "due": x["due"]} for x in gt],
    "final_env_response": "done",
}
info = {"transcript_id": tid, "action_items": gt}
score = f1_reward_from_state(state=state, info=info)
assert_close(f"f1[{tid}] perfect", score, 1.0)


section("f1_reward_from_state on empty notepad")
empty_state = {"notepad": [], "final_env_response": "done"}
# Empty GT -> 1.0
empty_tid = "brainstorm_no_commits_009"
empty_info = {"transcript_id": empty_tid, "action_items": []}
score = f1_reward_from_state(state=empty_state, info=empty_info)
assert_close(f"empty notepad on empty-GT transcript [{empty_tid}]", score, 1.0)
# Non-empty GT -> 0.0
score = f1_reward_from_state(state=empty_state, info=info)
assert_close(f"empty notepad on non-empty-GT transcript [{tid}]", score, 0.0)


section("f1_reward_from_state on partial notepad (under-extraction)")
under_state = {
    "notepad": [{"owner": "Devon", "task": "ranking spike", "due": "Friday"}],
    "final_env_response": "done",
}
score = f1_reward_from_state(state=under_state, info=info)
# 1 of 2 matched: p=1, r=0.5, F1 = 2/3
assert_close("under-extract F1 = 2/3", score, 2 / 3)


section("done_called_reward")
assert_close("done not called -> 0", done_called_reward(state={"notepad": []}), 0.0)
assert_close("done called -> 1", done_called_reward(state={"final_env_response": "done"}), 1.0)


# --- Env construction sanity ---
section("load_environment_multiturn builds an env")
env = load_environment_multiturn()
assert_true("env has the 5 tools", len(env.tools) == 5)
tool_names = sorted(env.tool_map.keys())
assert_eq("tool names", tool_names, ["add_item", "done", "next_chunk", "remove_item", "revise_item"])
# Confirm `state` is hidden from each tool's schema.
for td in env.tool_defs:
    props = td.parameters.get("properties", {}) if isinstance(td.parameters, dict) else {}
    assert_true(f"state hidden in {td.name} schema", "state" not in props)


# --- Summary ---
print()
total = passed + failed
status = color(f"{passed}/{total} passed", 32 if failed == 0 else 31)
print(f"=== SUMMARY === {status}")
sys.exit(0 if failed == 0 else 1)
