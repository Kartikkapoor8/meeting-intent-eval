"""score raw samples against ground truth, write results.json"""
import json, os, re
from collections import defaultdict

HERE = os.path.dirname(__file__)
GT = {e["image"]: e for e in json.load(open(os.path.join(HERE, "ground_truth.json")))}
RAW = json.load(open(os.path.join(HERE, "raw_samples.json")))

CLOCK_TOL_MIN = 1
NUMERIC_TOL_FRAC = 0.05  # 5% of scale_max

# unit equivalences (lowercase, model often differs from GT)
UNIT_OK = {
    "mph": {"mph", "miles per hour", "mi/h"},
    "kg": {"kg", "kilogram", "kilograms", "kgs"},
    "rpm": {"rpm", "r/min", "rev/min", "revolutions per minute"},
    "mmhg": {"mmhg", "mm hg", "mm of mercury", "torr"},
    "oz": {"oz", "ounce", "ounces"},
}

def extract_json(text):
    """parse json from model output, handling ``` fences and stray text"""
    t = text.strip()
    # strip code fences
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```\s*$", "", t)
    # find first { ... } block
    m = re.search(r"\{[\s\S]*\}", t)
    if not m: return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def norm_unit(u):
    if u is None: return ""
    return str(u).strip().lower()

def units_match(pred_unit, gt_unit):
    p = norm_unit(pred_unit)
    g = norm_unit(gt_unit)
    if not g or g == "time": return True
    if p == g: return True
    aliases = UNIT_OK.get(g, {g})
    return p in aliases

def parse_clock(reading):
    """return (h,m) or None"""
    if isinstance(reading, str):
        m = re.search(r"(\d{1,2})[:hH](\d{2})", reading)
        if m:
            return int(m.group(1)) % 24, int(m.group(2))
    return None

def score_clock(pred, gt_str, gt_alt=None):
    """pass if within 1 minute"""
    p = parse_clock(pred.get("reading"))
    if p is None: return False, "couldn't parse reading"
    gh, gm = parse_clock(gt_str)
    ph, pm = p
    # match modulo 12 hours (clock can be AM/PM)
    pmins_pre = ph * 60 + pm
    cands = [(gh, gm)]
    if gt_alt:
        for a in gt_alt:
            c = parse_clock(a)
            if c: cands.append(c)
    # try also +12 on hours since 12h dial
    def candidate_minutes(c):
        return [(c[0] % 12) * 60 + c[1], ((c[0] % 12) + 12) * 60 + c[1]]
    p_mins_options = [(ph % 12) * 60 + pm, ((ph % 12) + 12) * 60 + pm]
    for c in cands:
        for cm in candidate_minutes(c):
            for pm_o in p_mins_options:
                diff = abs(cm - pm_o)
                diff = min(diff, 24*60 - diff)
                if diff <= CLOCK_TOL_MIN:
                    return True, f"diff={diff}min vs {c[0]:02d}:{c[1]:02d}"
    return False, f"pred={ph:02d}:{pm:02d} vs gt={gh:02d}:{gm:02d}"

def score_numeric(pred, gt_val, scale_max, gt_unit):
    try:
        r = pred.get("reading")
        if isinstance(r, str):
            # try to extract number from string
            mm = re.search(r"-?\d+(?:\.\d+)?", r.replace(",", ""))
            if not mm: return False, "no number in reading"
            r = float(mm.group(0))
        else:
            r = float(r)
    except (TypeError, ValueError):
        return False, "reading not numeric"
    tol = NUMERIC_TOL_FRAC * scale_max
    diff = abs(r - gt_val)
    unit_ok = units_match(pred.get("unit"), gt_unit)
    if diff <= tol and unit_ok:
        return True, f"diff={diff:.2f} tol={tol:.2f}"
    why = []
    if diff > tol: why.append(f"diff={diff:.2f}>{tol:.2f}")
    if not unit_ok: why.append(f"unit_mismatch:{pred.get('unit')}!={gt_unit}")
    return False, ",".join(why)

def confidence_label(c):
    return str(c or "").strip().lower()

results = {"per_image": [], "per_type": defaultdict(lambda: {"n": 0, "pass1": 0, "pass8": 0, "samples_correct": 0, "samples_total": 0}),
           "overall": {}, "confidence": {"correct": [], "wrong": []}}

samples_correct = 0
samples_total = 0
n_pass1 = 0
n_pass8 = 0

for img, samples in RAW.items():
    gt = GT[img]
    is_clock = gt["instrument_type"] == "analog_clock"
    n = len(samples)
    correct = 0
    diags = []
    for s in samples:
        pred = extract_json(s["text"])
        if pred is None:
            diags.append("PARSE FAIL")
            continue
        if is_clock:
            ok, why = score_clock(pred, gt["ground_truth_reading"], gt.get("ground_truth_alt"))
        else:
            ok, why = score_numeric(pred, gt["ground_truth_reading"], gt["scale_max"], gt["unit"])
        diags.append(("OK " if ok else "X  ") + why + " | " + json.dumps({k: pred.get(k) for k in ("reading", "unit", "confidence")}))
        if ok: correct += 1
        # confidence tracking
        c = confidence_label(pred.get("confidence"))
        if c:
            (results["confidence"]["correct" if ok else "wrong"]).append(c)
    pass1 = (samples[0] is not None and diags[0].startswith("OK "))
    pass8 = correct > 0
    results["per_image"].append({
        "image": img, "type": gt["instrument_type"],
        "gt": gt["ground_truth_reading"], "unit": gt.get("unit"),
        "n_correct": correct, "n_total": n,
        "pass1": pass1, "pass8": pass8,
        "sample_diagnostics": diags,
    })
    t = gt["instrument_type"]
    results["per_type"][t]["n"] += 1
    results["per_type"][t]["pass1"] += int(pass1)
    results["per_type"][t]["pass8"] += int(pass8)
    results["per_type"][t]["samples_correct"] += correct
    results["per_type"][t]["samples_total"] += n
    samples_correct += correct
    samples_total += n
    n_pass1 += int(pass1)
    n_pass8 += int(pass8)

results["overall"] = {
    "n_images": len(RAW),
    "pass1": n_pass1, "pass1_pct": round(100 * n_pass1 / len(RAW), 1),
    "pass8": n_pass8, "pass8_pct": round(100 * n_pass8 / len(RAW), 1),
    "samples_correct": samples_correct, "samples_total": samples_total,
    "samples_pct": round(100 * samples_correct / samples_total, 1),
}
# turn per_type defaultdict into dict with pct
pt = {}
for k, v in results["per_type"].items():
    pt[k] = {**v,
             "pass1_pct": round(100 * v["pass1"] / v["n"], 1),
             "pass8_pct": round(100 * v["pass8"] / v["n"], 1),
             "samples_pct": round(100 * v["samples_correct"] / v["samples_total"], 1)}
results["per_type"] = pt

# confidence summary
def freq(lst):
    out = defaultdict(int)
    for x in lst: out[x] += 1
    return dict(out)
results["confidence"] = {
    "correct": freq(results["confidence"]["correct"]),
    "wrong": freq(results["confidence"]["wrong"]),
}

with open(os.path.join(HERE, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(f"overall pass@1: {results['overall']['pass1_pct']}%  ({n_pass1}/{len(RAW)})")
print(f"overall pass@8: {results['overall']['pass8_pct']}%  ({n_pass8}/{len(RAW)})")
print(f"overall sample acc: {results['overall']['samples_pct']}%  ({samples_correct}/{samples_total})")
print("\nper type:")
for t, v in results["per_type"].items():
    print(f"  {t:25s} n={v['n']} pass@1={v['pass1_pct']}% pass@8={v['pass8_pct']}% sample={v['samples_pct']}%")
print("\nconfidence on correct:", results["confidence"]["correct"])
print("confidence on wrong  :", results["confidence"]["wrong"])
print("\nper image breakdown:")
for r in results["per_image"]:
    print(f"  {r['image']:40s} gt={str(r['gt']):8s} ({r['unit']}) -> {r['n_correct']}/{r['n_total']} pass@1={r['pass1']}")
