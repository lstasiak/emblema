# ADR-0020: Judging a masked-reconstruction run — rules in the domain, an interval per comparison, a noise floor, and a probe for the diagnostic the control cannot answer

- Status: accepted
- Date: 2026-09-15
- Full text before condensation: commit `5f14447`

## Context

ADR-0019 defined the objective and a baseline per mask kind, not how a run is *judged*. The first
leg showed the gaps: it called single-token masks trivial from one pair of means (0.0167 vs
0.0135) with nothing to say whether that survives another draw of units; a baseline at the noise
floor can be "beaten" by nothing, and a model below the floor has leaked; beating interpolation and
ridge separately is not beating both together. The losses were still falling, and the spectral
diagnostic was mute because 99.4 % of the control's energy sits at one cycle per window.

## Decision

- **Rules live in `pretraining/domain/assessment`**, under the architecture rules, type checker and
  coverage. Each rule returns a `Check` (measured, expected, status, reading); `decide` folds them
  into an `Outcome` by precedence: implementation fault → baseline at the floor → not beaten
  without proof of convergence → not beaten after convergence. Thresholds are constants with their
  reasoning, fixed before the runs they judge.
- **Every comparison has a bootstrap interval over validation units**, the level at which data are
  independent; a kind is `learnt` only when the whole interval is above zero. Errors are ratios of
  sums, so per-unit tallies are summed; statistics sit behind the `MaskKindSummariser` port. Masks
  are drawn once, so intervals hold variation between units, not draws or seeds.
- **A noise floor where the corpus knows one** (generated corpora): below 0.8 × floor is a leak;
  a baseline within 1.5 × floor leaves nothing to teach, so the strategy changes, not the training;
  the linear baseline decides only where the floor leaves room.
- **A second baseline: one ridge over both sources** — other channels' nearest values and the
  channel's own interpolated line. Both ridges are fitted under masks the strategy draws, not on
  intact windows. Penalty fixed at 1.0.
- **Convergence is read off two runs**: a stored run of the same configuration with at most half
  the epochs must give the same verdicts and a final loss at most 5 % higher. A decaying schedule
  flattens any curve, so one curve proves nothing.
- **The learning-rate schedule is a domain value** (linear warmup, cosine decay to a floor).
- **A spectral probe corpus**: the dense control layout with factor periods compressed from 24–300
  to 6–32 steps, kept in the report's module, not the control registry.

## Consequences

- Stored runs are re-read from what they measured, never from their conclusions; replaying 23
  stored runs reproduced every printed assessment byte for byte.
- At the published tier every mask kind is `learnt` on both control layouts and the probe, against
  both baselines; single tokens clear zero by +0.0041 on the dense layout
  (`docs/verification/masked-reconstruction.md`).
- The convergence rule warns on all three corpora (doubling still lowers the loss by 28–43 %). The
  warning is kept, not tuned away.
- **Nothing guards against a too-weak baseline**: the fixed penalty is not swept, because every
  recorded number would change for baselines beaten five- to fortyfold.
- Percentile intervals over 20–40 units are approximate.
- Pretraining's `Verdict`, `Decision`, `Interval` are not Evaluation's types of the same name.
- `MaskKindSummariser` has one adapter (`UnitBootstrap`): a fake of statistics passes no content
  test.

## Alternatives considered

- *Report the excess only*: produced the one reading the next leg overturned.
- *Paired test over per-unit means*: a different quantity from a ratio of sums.
- *Bootstrap over windows*: overlapping windows make the interval too narrow.
- *Convergence from one curve*: a decaying schedule passes the runs meant to be caught.
- *Probe in the control registry*, or *widening the control's band*: a control's dials must not turn.
- *Rules in the report script*: outside the rules, the type checker and coverage.

## Revisit when

- A real corpus with no noise floor → estimate one and defend it, or report which verdicts rest on
  less.
- More than one pretraining seed → intervals across seeds.
- A second summariser (paired test, BCa, permutation) → the single-adapter exception lapses.

## Sources

Efron and Tibshirani (1993), *An Introduction to the Bootstrap*; Bouckaert and Frank (2004), PAKDD;
Demšar (2006), JMLR.
