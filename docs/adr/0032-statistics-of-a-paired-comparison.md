# ADR-0032: Statistics of a paired comparison — the engine is the unit resampled, repeats pool per engine, and the arithmetic a verdict rests on lives in the Evaluation domain

- Status: proposed (the first curve is read through it; the harness's statistics work
  finalises it)
- Date: 2026-09-19
- Full text before condensation: commit `5f14447`

## Context

The preregistration judges a cell of the label-efficiency curve by the reduction in validation
RMSE against the arm trained from scratch: a 95 % interval from a bootstrap over the 18 validation
engines (10,000 resamples, same engines on both sides), one uncorrected endpoint, eleven secondary
cells under Holm, and a practical floor. No code computed it. Pretraining's `UnitBootstrap`
(ADR-0020) speaks of masks and tokens and may not be imported. The first reading is made by a
report script before the harness exists.

## Decision

- **The arithmetic a verdict rests on is Evaluation domain code now** (`evaluation/domain/statistics/`),
  not a script's. The harness extends it.
- **The engine is the resampled unit, both sides at once.** `PairedUnitErrors` holds per-unit
  errors of control and candidate in one order and refuses a pair over different units or window
  counts. A resample picks engines with replacement and adds sums and counts before the root.
- **Repeats pool per engine**: squared errors of every seed are summed per engine, so a cell's
  error is the RMSE over its repeats. The spread between repeats is reported beside the interval
  (it feeds the floor), never folded into it — five seeds are not an interval.
- **Percentile interval, two-sided bootstrap p-value, Holm step-down, one practical floor.**
  `PairedUnitBootstrap` is seeded; the observed reduction counts as one more resample, so no
  p-value is zero. `HolmCorrection` treats members that have not run as not rejected, so a partial
  grid is read more strictly. `PracticalFloor` = max(a share of the control's error, the control's
  spread over repeats).
- **The verdict is domain code** (`ComparisonRules` → `ComparisonVerdict`): the endpoint is
  confirmed only if it takes the minimum share off, its interval excludes zero, *and* it clears the
  floor. A secondary cell is judged by the family's correction and is never "confirmed".
- **Plain Python, no numpy**: 180,000 additions.
- **Task readings beside the endpoint** (`RemainingLifeMetrics`): error below the ceiling, error on
  each engine's last window, share within a fifth of the label, the asymmetric score as a mean per
  window. Reported, never thresholded. On validation engines that run to failure the last-window
  error is not comparable with published test-set numbers.

## Consequences

- Known-answer tests: a true reduction has its interval above zero, no difference leaves zero
  inside, Holm on a textbook family and on a partial one, both parts of the floor, every verdict
  branch.
- `scripts/label_curve_report.py` composes the domain over stored cells, refuses shards not run as
  one grid (one commit, one backbone, one plan per mode) and holds no formula.
- Pretraining keeps its own bootstrap in its own words.

## Alternatives considered

- *A bootstrap in the script*: a decision on arithmetic outside the rules.
- *Sharing Pretraining's bootstrap*: shared idea, not shared type.
- *Resampling seeds, or seeds and engines*: five seeds are not an interval; two-level bootstrap is an
  ablation for later.
- *Differencing two unpaired intervals*: discards the pairing.

## Revisit when

- The harness's statistics work: two-level bootstrap, Benjamini–Hochberg, variance over seeds;
  then this status is settled.
- A task with one label per unit → the per-unit error is one squared error.
- A metric that does not add over units → paired errors carry their own combination rule.
