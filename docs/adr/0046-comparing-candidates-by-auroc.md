# ADR-0046: Comparing candidates by AUROC — one orientation for every measure, strata in the resampling, thresholds in the measure's unit

- Status: proposed (accepted once the interval's coverage is measured on known answers)
- Date: 2026-09-27

## Context

The paired comparison (ADR-0032) is built on root mean squared error: a reduction is the control's
error minus the candidate's, the practical floor and the endpoint's minimum are shares of the
control's error, and repeats pool by adding squared errors per unit. A binary task is read by the
area under the ROC curve (AUROC), which is higher-is-better, does not split into sums per unit
and is undefined on a resample holding one outcome. Its literature states differences in AUROC
points, not as shares.

## Decision

- **Every measure is an error, lower is better.** `ErrorMeasure` (`RMSE`, `AUROC_SHORTFALL`) is
  part of the campaign's design, registered before the grid runs, defaulting to the one the task's
  kind calls for and refused where it does not apply. The AUROC error is one minus the area: the
  probability that a random positive and negative are ordered the wrong way, ties counting half.
  A reduction is then exactly the gain in area, and the floor, the rule of one standard error and
  the verdict taxonomy keep their meaning without a sign in each. Reports, the API and the
  published event state the area itself and the Brier score beside it.
- **Thresholds are stated relative or absolute.** `ComparisonRules` carries a `ThresholdKind` for
  the endpoint's minimum and the floor's fixed part together. A share of one minus the area is a
  share of the distance to perfect ranking, which asks twice as much against a weak control as
  against a strong one; an absolute gain is what the literature reports. The key names in the
  campaign file and the stored design say which kind is meant; designs stored before read as
  relative.
- **Resampling in strata.** The bootstrap resamples units with replacement, both sides on the
  same draw, within strata: all units as one stratum for a squared error (the draw is bit for bit
  the earlier one), and for a ranking the units holding a positive apart from the rest, so every
  resample keeps both outcomes in the side's proportion.
- **Repeats pool by the mean of their areas**, each over the same resampled units. Ranking five
  models' answers together would compare one model's positives with another's negatives.
- **One sweep per resample.** A ranking is sorted once per repeat; a resample only changes how
  often each unit counts, and the area follows in one pass over tie groups with those weights.
  Pure Python, as the rest of the statistics.
- **Selection by the same error.** A selection campaign minimises one minus the area over the
  inner holdout; the Brier score is reported and not selected on.

## Consequences

- Cost, measured on 1,600 stays, five repeats and 10,000 resamples: about 19 s per comparison,
  about 11 min for a verdict of 35 comparisons, read once per revision (`VerdictMemo`). NumPy in
  the domain is the lever if this binds; it would make NumPy a dependency of every process.
- The two-level bootstrap (ADR-0032) stays a squared-error ablation.
- The percentile interval's coverage for AUROC with few positives is not yet measured. A generator
  with known areas (`KnownRanking`, binormal answers with a shared unit effect) is in place; the
  coverage is measured before the first binary campaign is read, and a bias-corrected interval
  is registered before the run if the percentile one falls short.
- Cells without kept answers cannot be read by area and are refused by name.

## Alternatives considered

- **A metric with a direction.** Natural names, but a sign in every consumer of the reduction,
  and relative thresholds would still need the distance to a perfect score.
- **DeLong's test.** Exact for one pair of models; it does not pool repeats, and would give the
  two measures two different procedures.
- **Pooling repeats by ranking all their answers together.** Biased downwards when the repeats'
  scores are not on one scale.
