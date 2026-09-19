# ADR-0032: Statistics of a paired comparison — the engine is the unit resampled, repeats pool per engine, and the arithmetic a verdict rests on lives in the Evaluation domain

- Status: proposed (the first curve is read through it; the harness's statistics work
  finalises it)
- Date: 2026-09-19

## Context

The label-efficiency curve is a grid of cells — a transfer mode at a budget of labels under a
seed — and the preregistration says how a cell is judged: the reduction in validation RMSE
against the arm trained from scratch, its 95 % interval from a bootstrap over the 18 validation
engines, ten thousand resamples, the same engines on both sides; one endpoint carrying no
correction and eleven secondary cells under a Holm correction; a practical floor below which a
distinguishable difference is reported as nil. What did not exist was code that computes any of
it. The pretraining context has a bootstrap over units for its own diagnostics
(`UnitBootstrap`, ADR-0020), but a context may import another's published contracts and the
shared packages only, and that bootstrap speaks of masks and tokens.

The first reading of the curve is a preliminary result, made by a report script before the
evaluation harness exists; the harness will repeat the comparison and publish that reading. The
question was where the arithmetic that decides a verdict should live in the meantime: in the
script that is retired with the harness, or in the context that will own the verdict.

## Decision

**The arithmetic a verdict rests on is the Evaluation domain's, now.** A script composes and
renders; it does not decide. The rule that logic a verdict depends on belongs under the
architecture rules, the types and the coverage rather than in a script is the repository's own,
and a preliminary result read through a script's private bootstrap would be a number nobody
could hold to the registered rules later. So `evaluation/domain/statistics/` holds the minimum
the registered rules need, and the harness's statistics work extends it rather than writing it
again.

**The engine is the unit resampled, on both sides at once.** `PairedUnitErrors` holds the
control's and the candidate's error per unit in one order, and refuses a pair over different
units or different counts of windows: windows of one engine overlap and an engine is one
realisation of the process, so the engine is the level at which the errors are independent, and
a comparison paired on engines removes the variation between engines from the difference. A
resample picks engines with replacement and scores each side by adding the picked sums and
counts before the root is taken, so the interval is over the error the endpoint is stated in.

**Repeats of a cell pool per engine before the pair is made.** The five seeds of a cell are one
candidate over all of its answers: the squared errors of every repeat are added per engine, so
a cell's error is the root of the mean squared error over its repeats, and the interval over
engines is read from that pooled pair. The spread of the error between repeats is reported
beside the interval — it is what the practical floor's measured part reads — and never folded
into it, since five seeds do not make an interval and the preregistration declares the engines
as the binding constraint.

**Percentile interval, two-sided bootstrap p-value, Holm step-down, one practical floor.**
`PairedUnitBootstrap` is seeded, so a comparison always yields the same interval; its p-value is
twice the smaller share of resamples on either side of zero, the test the percentile interval
inverts, and it is what the Holm correction ranks. The share counts the observed reduction as
one more resample on its side, so no p-value is zero and the smallest one, two in ten thousand
and one, says how many resamples were drawn rather than claiming a certainty. `HolmCorrection`
holds the smallest p-value of a family to the level over the family's size and stops at the
first failure; given a family larger than the p-values it is handed, it treats the members that
have not run as never rejected, so a partial grid is read more strictly than the whole one and
never less. `PracticalFloor` is the larger of a share of the control's error and the spread of
that error over the control's repeats, as registered. `PairedDifference` carries the reduction,
its share, the interval and the p-value, and answers the two questions the rules ask: whether
the claim is confirmed at a stated minimum share, and whether zero is excluded.

**The verdict is the domain's as well, not the script's.** `ComparisonRules` holds the four
registered numbers — the minimum share, the floor's share, the level and the size of the
secondary family — and turns a difference and a floor into a `ComparisonVerdict`. The endpoint
is confirmed only when it takes the share off, keeps its whole interval above zero *and* clears
the floor: the registration calls a reduction under the floor practically nil, and a
confirmation the same document calls nil would be a contradiction, so the floor binds the
endpoint as it already binds the synthetic control. A secondary cell is judged by the family's
word and never by its own interval, so its verdict and its rejection cannot disagree; and no
secondary cell is ever confirmed. Every branch of the verdict is a domain test.

**Plain Python, no array library.** Eighteen engines and ten thousand resamples are a hundred
and eighty thousand additions; the domain stays free of numpy, and the code reads as the
procedure it states.

**The readings beside the endpoint are the task's.** `RemainingLifeMetrics` computes, from the
same predictions, the error below the ceiling, the error on the last window of each engine, the
share of answers within a fifth of the label and the benchmark's asymmetric score. They are
reported and never thresholded, as registered; they live beside the outcome because they are the
remaining-life task's readings and not a statistic of any comparison. The asymmetric score is a
mean per window rather than the benchmark's sum: the benchmark sums one answer per engine at a
cut-off, which a sum over hundreds of overlapping validation windows does not reproduce, while
a mean does not scale with how many windows a side holds and so reads across the validation
and the test side. The last-window error is the benchmark's protocol only on a test side cut
short of failure; on the validation side, whose engines run to failure, an engine's last window
ends within four cycles of it, so the reading there is the error at the end of life and is not
comparable with published numbers — the note says so where it prints it.

## Consequences

- `evaluation/domain/statistics/`: `PairedUnitErrors`, `PairedUnitBootstrap`,
  `BootstrapInterval`, `PairedDifference`, `PracticalFloor`, `HolmCorrection`,
  `ComparisonRules`, `ComparisonVerdict`; `evaluation/domain/transfer/remaining_life_metrics.py`.
  Tests with a known answer: a true reduction is found with its whole interval above zero, no
  true difference leaves zero inside, the Holm step-down on a textbook family and over a family
  larger than what ran, the floor from both of its parts, every branch of the verdict.
- `AdaptationOutcome` reports how many labelled windows and how many units it learnt from, so a
  budget can be shown beside the engines that stood behind it.
- `scripts/label_curve_report.py` composes the domain over the stored cells — which cells make
  the family, which is the endpoint, what the registered numbers are — refuses shards that were
  not run as one grid (one commit, one backbone, one plan per mode), and renders the note's
  tables and its one-sentence conclusion, which says when the grid is incomplete and the
  reading therefore not the registered one; `scripts/label_curve_figures.py` draws from the
  files. Neither holds a formula or a rule of judgement.
- The pretraining context keeps its own `UnitBootstrap`: it summarises tallies of masked tokens
  by kind of mask and is not a paired comparison of candidates. Two bootstraps over units exist
  in two contexts, each in that context's words, which is what a bounded context means.

## Alternatives considered

- **A bootstrap in the report script, the harness writing the real one later.** The first
  intention for a preliminary reading. Rejected: the number read from it would decide whether
  the next stage starts, and a decision taken on arithmetic outside the rules is the kind of
  number the preregistration exists to prevent; and the harness would write the same sixty
  lines again.
- **Moving the pretraining bootstrap to the shared packages and reusing it.** Rejected: it
  summarises mask kinds over token tallies; what is shared with this one is the idea of
  resampling units, not a type.
- **Resampling seeds, or seeds and engines together.** Rejected for the first reading as the
  preregistration rejects it: five seeds are not an interval. A two-level bootstrap is named
  there as an ablation of the statistical method itself, for the harness.
- **An interval over each arm's own error, differenced.** Rejected: it discards the pairing, and
  the engines' own spread would swamp a difference the pairing shows.

## Revisit when

- The harness's statistics work arrives. Then the two-level bootstrap over seeds and engines,
  the Benjamini–Hochberg alternative to Holm, and the report of variance over seeds as a
  component are weighed, and this record's status is settled.
- A task arrives whose unit carries one label rather than a window's worth. Then the paired
  error per unit is one squared error, and the pooling over repeats is a mean over answers to
  one question rather than a sum over windows.
- A comparison needs a metric that does not combine by addition over units — the asymmetric
  score does, the alpha-lambda share does over counts, a rank-based one would not. Then the
  paired errors carry the metric's own combination rule rather than sums and counts.
