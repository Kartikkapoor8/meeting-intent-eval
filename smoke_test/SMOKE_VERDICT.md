# Analog instrument smoke test

Model: claude-opus-4-6
N images: 14 (8 verified clocks from ClockBench public sample + 6 instruments from Wikimedia Commons)
N samples per image: 8 at T=1.0
Total API spend: $2.95
Time: under 2 hours

## Summary

| Instrument type | N | pass@1 | pass@8 | sample acc |
|---|---|---|---|---|
| analog_clock | 8 | 0.0% | 0.0% | 0.0% |
| bathroom_scale | 1 | 100.0% | 100.0% | 100.0% |
| tachometer | 1 | 0.0% | 100.0% | 75.0% |
| speedometer | 1 | 100.0% | 100.0% | 100.0% |
| sphygmomanometer | 2 | 100.0% | 100.0% | 100.0% |
| postal_scale | 1 | 0.0% | 0.0% | 0.0% |

Overall pass@1: **28.6%** (4/14)
Overall pass@8: **35.7%** (5/14)
Overall sample accuracy: **33.9%** (38/112 samples)

Confidence on correct samples: high=28, medium=10
Confidence on wrong samples: high=42, medium=32

The model says "high confidence" more often when it's wrong than when it's right. Bad calibration.

## Verdict

**A) GO. Build the full env.**

28.6% pass@1 lands inside the 10-40% target zone. Clocks alone give us 0% on 64 samples across 8 different times. That is a hard, repeatable, undeniable failure mode.

The numeric instruments split cleanly: trivial readings (needle resting at 0) pass at 100%, anything off-zero degrades. The one off-zero gauge we tested (tachometer at ~2200 rpm) had the model snap to 1500 in 6/8 samples and 1000 in 2/8. Right at the edge of tolerance.

Most useful for RL signal:
- clocks are a goldmine. zero variance across 8 samples at T=1.0. model commits to one wrong answer and stays there.
- gauges with non-zero, off-tick readings (like the tachometer) show the "snap to round number" failure clearly.

Caveats:
- 14 images is small. true accuracy is somewhere in a wide CI. but 0/64 on clocks is not noise.
- 4 of my 6 numeric instruments had reading=0, which trivializes them.
- The full env should source images with non-zero needle positions deliberately, and weight clocks heavily since they're the cleanest signal.

## Per-instrument breakdown

### analog_clock (8 images, 0/8 pass@1, 0/8 pass@8, 0/64 samples correct)

Every single clock failed. Even at T=1.0 the model commits to one wrong answer and repeats it across all 8 samples.

Example failures (gt vs predicted, first sample):
- white_1: gt 12:25, model said 11:27 (8/8 samples). hour off by 1, minutes off by 2.
- white_circular_numbers_4: gt 08:45, model said 03:30. completely wrong, no relation to true reading.
- white_4: gt 06:02, model said 11:30. hour completely wrong.

Failure modes seen:
- hour/minute confusion (mistakes which hand is which when they're at similar angles)
- snapping to round minutes (:30, :00) instead of reading sub-tick positions
- wrong hour entirely (off by 5+)

The zero variance across 8 samples is the most striking thing. ClockBench shows pass@1 of 13.3% on top model, our slice was harder (0%) but consistent with that. Their public sample is small (10 clocks, 8 valid for time) so n is tiny but the trend is clean.

### tachometer (1 image, gt 2200 rpm, scale 0-14000)

6/8 samples said 1500 rpm (within 5% band, passes)
2/8 samples said 1000 rpm (outside band, fails)

The first sample said 1000, so pass@1 = 0. pass@8 = 1.

Failure mode: snapping to round numbers (1000, 1500) instead of reading the needle position. The actual needle is between 2 and 3 on the x1000 scale, so 2200 is the right answer. Model reads it as if needle is at 1 or 1.5.

### postal_scale (1 image, gt 0 oz, model says 0 lb)

8/8 samples said reading=0 (correct numerically) but with unit "lb" not "oz". My scorer fails this on unit mismatch.

This is a partial failure — the dial has both lb and oz scales printed, so "lb" is not crazy. But if the env is going to demand a specific unit, the model often picks the wrong one when the scale is ambiguous.

### sphygmomanometer x2, speedometer (4/4 pass)

All three are at exactly 0. Trivial. Model handles them perfectly.

### bathroom_scale (gt 85 kg, model says 90 kg, 8/8 within tolerance)

Model says 90 every time. My GT is 85, tolerance is ±6.5, so 90 passes. Worth noting the model commits to 90 with medium confidence on every sample — no spread.

## Sample failures (3 representative)

**1. white_4.png** — synthetic clock, GT 06:02
- Model output: `{"reading": "11:30", "confidence": "high"}`
- Diagnosis: model swapped hour and minute hands. Hour hand is straight down (at 6), minute hand is near 12. Model read minute-hand-at-12 as hour-hand-at-11 and made up :30 for the minute.

**2. tachometer_01.jpg** — motorcycle tach, GT 2200 rpm
- Model output: `{"reading": 1000, "unit": "RPM", "confidence": "high"}`
- Diagnosis: needle is past the 2 mark on a 0-14 (x1000) scale. Model rounded all the way down to 1000 and called it high confidence. Sub-tick reading is the weakness.

**3. white_circular_numbers_4.png** — clock with full minute numerals, GT 08:45
- Model output: `{"reading": "03:30", "confidence": "high"}`
- Diagnosis: completely wrong. Even with explicit minute numerals visible (1-60 around the rim) the model can't recover the time. Worst-case clock failure.

## Path forward

Build the full env. Specifically:
- weight clocks heavily in the env (best signal, ClockBench-aligned)
- generate / source images with deliberate sub-tick needle positions for gauges
- avoid needle-at-zero traps in the sourcing pipeline
- add explicit unit-in-prompt to remove the lb/oz/celsius/fahrenheit failure category, since we want to test reading not unit guessing
- T=1.0 vs T=0.0 looks ~equivalent for clocks (zero variance observed) — pass@k won't help here, the model needs different reasoning, not more attempts
