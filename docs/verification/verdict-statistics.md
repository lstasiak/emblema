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
