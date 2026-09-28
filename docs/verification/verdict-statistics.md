# Statistics of the verdict

Measurements of the procedure itself, on data with a known answer, rather than of any model.

## 2026-09-25 — how well the registered interval keeps its word

**Question.** Every verdict rests on a percentile bootstrap over the validation units, the same
units on both sides, 10,000 resamples, a 95 % interval. A percentile interval over a few dozen
units is known to run short of its level. By how much, at the sizes this project reads?

**Conditions.** Commit of this note; MacBook Pro M1 Pro, CPU, pure Python. The known-answer
generator (`emblema.evaluation.adapters.synthetic.known_answer.KnownAnswer`): a control at RMSE
30 and a candidate at 25 (a true reduction of 5) or at 30 (no difference); 30 windows per unit;
each unit's difficulty multiplies both sides by a factor drawn in [0.7, 1.3]; each side's error on
each unit is then scattered by a further ±10 %. Per unit count, 400 datasets for each rate, a
2,000-resample interval per dataset, seed 1.

    uv run scripts/bootstrap_calibration_report.py --out data/report/verdict-statistics

| Units | Coverage of the true reduction | Zero excluded (two-sided) | Whole interval above zero |
| --- | --- | --- | --- |
| 18 | 89.8 % | 10.2 % | 3.8 % |
| 21 | 91.8 % | 9.5 % | 5.0 % |
| 40 | 92.8 % | 7.5 % | 3.8 % |
| 80 | 93.5 % | 4.2 % | 1.5 % |

Nominal: 95 %, 5 %, 2.5 %. The standard error of a rate over 400 datasets is about 1.5 points at
10 % and 1 point at 5 %.

**Conclusions.**

1. At the 21 validation engines of the turbofan task the interval covers about 92 % rather than
   95 %, and a true zero is excluded about twice as often as the level says.
2. The endpoint's rule asks for the whole interval above zero. Under no true difference that
   happens about 5 % of the time at 21 units, twice the nominal 2.5 %.
3. The shortfall closes with the units: at 80 units the rates are within a point or two of
   nominal. It is a property of the percentile interval at small samples, not of the pairing or
   the pooling.
4. The rule stands as registered: the confirmed endpoint (+12.3 %, interval [+1.49, +3.30]) is
   far from the boundary this shortfall moves. Every reading made under the rule carries this
   note beside it.

**Limitations.** The generator's scatter is a guess at the shape of real per-engine errors, not a
fit to them; the rates would differ under heavier tails. A bias-corrected interval (BCa) would
close part of the gap and is the natural change if a future endpoint sits near the boundary; it
would be a criterion change, registered before the run that reads it.

## 2026-09-28 — declared before the run: the interval over stays on the intensive-care task

**Question.** A comparison on the intensive-care task is a paired bootstrap of the difference in
the area under the ROC curve, over stays drawn in two strata (died, survived), five repeats of a
side pooled by the mean of their areas. Does the percentile interval keep its 95 % at the size of
that task's validation side, and what does a seed that moves a model's whole area do to it?

**Conditions.** Commit of this section; MacBook Pro M1 Pro, CPU, pure Python, eight processes.
The known-answer generator (`emblema.evaluation.adapters.synthetic.known_ranking.KnownRanking`):
3,994 stays, 568 positive, one window each, as set B holds them; binormal answers with half the
variance within an outcome shared by the stay across both sides and every repeat; a control at an
area of 0.70 or 0.85 and a candidate 0.02 above it (coverage) or level with it (false positives).
Four regimes: one repeat; five repeats; five repeats whose area strays from the side's by a
standard deviation of 0.01 or of 0.03, drawn independently per side and repeat — the worst case,
since in a campaign both sides of a repeat learn from the same draw of labels and its luck is
partly shared. Per setting, 400 datasets for each rate, a 1,000-resample interval per dataset,
resampling seed 1.

    uv run scripts/ranking_calibration_report.py --out data/report/verdict-statistics/ranking --workers 8

**Predictions.**

1. Without a spread of repeats, at both levels and both repeat counts: coverage between 93 and
   97 %, zero excluded two-sided between 3 and 7 %, the whole interval above zero at most 3.5 %.
   At almost 4,000 stays the percentile interval has no small-sample shortfall to show.
2. With a spread of 0.01 the interval, which resamples stays and not seeds, is too narrow for a
   difference that moves with the seed: coverage between 75 and 90 %.
3. With a spread of 0.03 coverage falls below 60 % and a true zero is excluded two-sided more
   than a fifth of the time.

**Reading, declared beforehand.** If, without a spread of repeats, every setting covers at least
93 % and lies wholly above a true zero at most 3.5 % of the time — two standard errors of a rate
over 400 datasets from nominal — the percentile interval stays registered for the task. Otherwise
the bias-corrected interval is implemented and registered before the grid. The rows under a spread of repeats change no rule by themselves: they are what
the least gain and the fixed part of the floor are chosen against in the registration before the
grid, since the floor, not the interval, is what stands between a seed's luck and a confirmation.
