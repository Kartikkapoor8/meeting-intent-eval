"""Re-score saved samples against the current rubric and compute the final aggregate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "environments" / "meeting_intent"))

from meeting_intent import f1_reward, format_reward  # noqa: E402

RAW_FILES = [
    "results/raw_samples.json",         # original 5 easy
    "results/raw_samples_hard3.json",   # 3 medium
    "results/raw_samples_adv2.json",    # 2 adversarial
]
DATA_DIR = Path("environments/meeting_intent/data")
PASS_THRESHOLD = 0.99


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    p = 1.0
    for i in range(k):
        p *= (n - c - i) / (n - i)
    return 1.0 - p


def info_for(transcript_id: str) -> dict:
    gt = json.loads((DATA_DIR / f"{transcript_id}_ground_truth.json").read_text())
    return {"transcript_id": transcript_id, "action_items": gt["action_items"]}


all_samples: dict[str, list] = {}
for path in RAW_FILES:
    p = Path(path)
    if not p.exists():
        continue
    data = json.loads(p.read_text())
    for tid, samples in data.items():
        all_samples.setdefault(tid, []).extend(samples)

results = []
for tid in sorted(all_samples.keys()):
    samples = all_samples[tid]
    info = info_for(tid)
    scores = [f1_reward(completion=s["text"], info=info) for s in samples]
    fmt_oks = [format_reward(completion=s["text"]) for s in samples]
    n = len(samples)
    c = sum(1 for x in scores if x >= PASS_THRESHOLD)
    results.append(
        {
            "transcript_id": tid,
            "n": n,
            "n_passing": c,
            "pass_at_1": pass_at_k(n, c, 1),
            "pass_at_8": pass_at_k(n, c, 8),
            "pass_at_32": pass_at_k(n, c, 32),
            "mean_f1": sum(scores) / n,
            "format_compliance": sum(fmt_oks) / n,
        }
    )

n_rows = len(results)
agg = {
    "pass_at_1": sum(r["pass_at_1"] for r in results) / n_rows,
    "pass_at_8": sum(r["pass_at_8"] for r in results) / n_rows,
    "pass_at_32": sum(r["pass_at_32"] for r in results) / n_rows,
    "mean_f1": sum(r["mean_f1"] for r in results) / n_rows,
    "format_compliance": sum(r["format_compliance"] for r in results) / n_rows,
}

print(f"{'transcript':<32}  {'n':>4}  {'pass':>5}  {'p@1':>5}  {'p@8':>5}  {'p@32':>5}  {'meanF1':>6}")
print("-" * 75)
for r in results:
    print(
        f"{r['transcript_id']:<32}  {r['n']:>4}  {r['n_passing']:>5}  "
        f"{r['pass_at_1']:.3f}  {r['pass_at_8']:.3f}  {r['pass_at_32']:.3f}  {r['mean_f1']:.4f}"
    )
print("-" * 75)
print(
    f"{'AGGREGATE (' + str(n_rows) + ' transcripts)':<32}  {'':>4}  {'':>5}  "
    f"{agg['pass_at_1']:.3f}  {agg['pass_at_8']:.3f}  {agg['pass_at_32']:.3f}  {agg['mean_f1']:.4f}"
)
print()
print(f"format compliance: {agg['format_compliance']:.3f}")

final = {
    "model": "claude-opus-4-6",
    "n_samples_per_transcript": 64,
    "temperature": 1.0,
    "pass_threshold": PASS_THRESHOLD,
    "n_transcripts": n_rows,
    "aggregate": agg,
    "per_transcript": results,
}
Path("results/final_aggregate.json").write_text(json.dumps(final, indent=2))
print()
print("Saved: results/final_aggregate.json")
