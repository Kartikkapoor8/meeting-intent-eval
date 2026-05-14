"""Pick 12 diverse meetings from QMSum Mistake source CSV.

Diversity by transcript length bucket. The CSV has 169 erroneous samples drawn from
QMSum (AMI/ICSI/parliament). We don't have the per-row 9-type error labels in the
public repo, so we stratify by length.

Outputs subset.json with 12 entries: {idx, input, gold, gpt4turbo_predicted, wc_input, wc_gold}.
"""
import csv, json, sys, statistics, random
csv.field_size_limit(sys.maxsize)

SRC = "/Users/admin/RLTEST/.claude/worktrees/great-goldwasser/results/meeting_smoke_test/qmsum_mistake_source.csv"
OUT = "/Users/admin/RLTEST/.claude/worktrees/great-goldwasser/results/meeting_smoke_test/subset.json"
N = 12
SEED = 7
TRUNC_WORDS = 6000  # truncate very long transcripts to keep cost bounded

random.seed(SEED)

with open(SRC) as f:
    rows = list(csv.DictReader(f))

# Bucket by transcript length
for i, r in enumerate(rows):
    r["_idx"] = i
    r["_wc_input"] = len(r["Input"].split())
    r["_wc_gold"] = len(r["Gold"].split())

rows.sort(key=lambda r: r["_wc_input"])
n = len(rows)
# 4 buckets: short, mid-short, mid-long, long. Sample 3 from each.
buckets = [rows[i*n//4:(i+1)*n//4] for i in range(4)]
picked = []
for b in buckets:
    picked.extend(random.sample(b, 3))

# Truncate
def trunc(text, max_w):
    w = text.split()
    if len(w) <= max_w:
        return text, len(w), False
    return " ".join(w[:max_w]), max_w, True

subset = []
for r in picked:
    truncated_input, wc, was_trunc = trunc(r["Input"], TRUNC_WORDS)
    subset.append({
        "idx": r["_idx"],
        "wc_input_full": r["_wc_input"],
        "wc_input_used": wc,
        "truncated": was_trunc,
        "wc_gold": r["_wc_gold"],
        "input": truncated_input,
        "gold": r["Gold"],
        "gpt4turbo_predicted": r["Predicted"],
    })

with open(OUT, "w") as f:
    json.dump(subset, f, indent=2)

print(f"Wrote {len(subset)} samples to {OUT}")
print(f"Input word counts (used): min={min(s['wc_input_used'] for s in subset)}, "
      f"median={int(statistics.median(s['wc_input_used'] for s in subset))}, "
      f"max={max(s['wc_input_used'] for s in subset)}")
print(f"Truncated: {sum(1 for s in subset if s['truncated'])}/{len(subset)}")
