# RUBRIC_SENSITIVITY.md

Took the 47 successful Opus 4.6 samples on real_earnings_001/002/003 and scored them under three rubric strictness levels. The point: rubric strictness does not matter when the model returned an empty array.

## Rubric definitions

- **Strict**: F1 over (owner, due, anchor-keyword on task). Threshold to pass: F1 >= 0.99. This is the rubric used in the original eval.
- **Medium**: F1 over (owner, due) only. No keyword check on task. Threshold to pass: F1 >= 0.5.
- **Loose**: any predicted item whose canonical owner OR canonical due matches any ground-truth value gets partial credit. Score = fraction of predicted items that hit. Threshold to pass: any score > 0.

## Pass rates and mean scores

| rubric | pass count | pass rate | mean score |
|---|---:|---:|---:|
| strict (F1 >= 0.99 over owner+due+anchor) | 0/47 | 0.000 | 0.000 |
| medium (F1 >= 0.5 over owner+due) | 0/47 | 0.000 | 0.000 |
| loose (any owner-or-due match counts) | 0/47 | 0.000 | 0.000 |

## Why pass rate is identical across rubrics

Of the 47 samples: 47 returned empty arrays, 0 returned non-empty, 0 were malformed.

Every rubric I implemented requires at least one predicted item to score above zero. When the array is empty there is nothing to score, so the strict and the loose rubric give the same answer: 0.0. The rubric is irrelevant. The failure is upstream of scoring.

## What this rules out

- It is not the anchor-keyword check rejecting otherwise-correct items. There are no items.
- It is not the owner canonicalization stripping last names. There are no owners.
- It is not the due-date normalization losing 'tonight' or 'EOD'. There are no due dates.
- It is not the F1 threshold being too high. F1 is 0.0 by construction when the array is empty.

If the buyer wants the rubric loosened to give credit for empty arrays, that is not a rubric, that is rewarding the model for refusing to do the task.
