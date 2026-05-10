"""Unit tests for the soft-commitment ablation variant.

We want this to be apples-to-apples with meeting_intent. The only thing that
should differ is the system prompt. Run: python3 test_softcommit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import meeting_intent
import meeting_intent_softcommit
from meeting_intent_softcommit import (
    SOFTCOMMIT_SYSTEM_PROMPT,
    load_environment_softcommit,
)


def color(s, code):
    return f"\033[{code}m{s}\033[0m"


passed, failed = 0, 0


def check(name, cond, detail=""):
    global passed, failed
    tag = color("PASS", 32) if cond else color("FAIL", 31)
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{tag}] {name}{suffix}")
    if cond:
        passed += 1
    else:
        failed += 1


def section(title):
    print()
    print(color(f"=== {title} ===", 36))


# --- Prompt content ---
section("system prompt content")
check(
    "prompt mentions soft due values",
    '"offline"' in SOFTCOMMIT_SYSTEM_PROMPT and '"ongoing"' in SOFTCOMMIT_SYSTEM_PROMPT,
)
check(
    "prompt allows dialogue-confirmation commitments",
    "dialogue" in SOFTCOMMIT_SYSTEM_PROMPT.lower(),
)
check(
    "prompt mentions JSON schema",
    '{"action_items":' in SOFTCOMMIT_SYSTEM_PROMPT,
)
check(
    "prompt is different from strict prompt",
    SOFTCOMMIT_SYSTEM_PROMPT != meeting_intent.SYSTEM_PROMPT,
)
check(
    "prompt does NOT carry over the strict 'do not include hedged' clause",
    "Hedged statements" not in SOFTCOMMIT_SYSTEM_PROMPT
    or "hedged but mutual" in SOFTCOMMIT_SYSTEM_PROMPT.lower(),
)


# --- Schema parity ---
section("schema parity with strict variant")
strict_env = meeting_intent.load_environment()
soft_env = load_environment_softcommit()

check(
    "soft env has same number of rows as strict env",
    len(soft_env.dataset) == len(strict_env.dataset),
    detail=f"strict={len(strict_env.dataset)} soft={len(soft_env.dataset)}",
)


def transcript_ids(ds):
    return sorted(row["info"]["transcript_id"] for row in ds)


check(
    "soft env covers the same transcripts as strict env",
    transcript_ids(soft_env.dataset) == transcript_ids(strict_env.dataset),
)


# --- Rubric parity (apples-to-apples) ---
section("rubric parity")
strict_funcs = [f.__name__ for f in strict_env.rubric.funcs]
soft_funcs = [f.__name__ for f in soft_env.rubric.funcs]
check(
    "rubric reward functions are identical",
    strict_funcs == soft_funcs,
    detail=f"strict={strict_funcs} soft={soft_funcs}",
)
check(
    "rubric weights are identical",
    list(strict_env.rubric.weights) == list(soft_env.rubric.weights),
)


# --- System prompt actually wired into the env ---
section("env wiring")
# verifiers SingleTurnEnv stores the system prompt; row prompts should start with it.
sample_row = soft_env.dataset[0]
prompt_msgs = sample_row["prompt"]
sys_msg = next((m for m in prompt_msgs if m["role"] == "system"), None)
check(
    "soft env first-row system prompt is the SOFTCOMMIT prompt",
    sys_msg is not None and sys_msg["content"] == SOFTCOMMIT_SYSTEM_PROMPT,
)


# --- Strict env still works ---
section("strict env unchanged")
strict_sample = strict_env.dataset[0]
strict_sys = next((m for m in strict_sample["prompt"] if m["role"] == "system"), None)
check(
    "strict env still uses the original SYSTEM_PROMPT",
    strict_sys is not None and strict_sys["content"] == meeting_intent.SYSTEM_PROMPT,
)


# --- load_environment_softcommit returns valid Environment ---
section("environment object validity")
check(
    "soft env exposes a dataset attribute",
    hasattr(soft_env, "dataset"),
)
check(
    "soft env exposes a rubric attribute",
    hasattr(soft_env, "rubric"),
)


print()
total = passed + failed
status = color(f"{passed}/{total} passed", 32 if failed == 0 else 31)
print(f"=== SUMMARY === {status}")
sys.exit(0 if failed == 0 else 1)
