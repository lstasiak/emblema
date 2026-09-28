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
the bias-corrected interval is implemented and registered before the grid. The rows under a
spread of repeats change no rule by themselves: they are what the least gain and the fixed part
of the floor are chosen against in the registration before the grid, since the floor, not the
interval, is what stands between a seed's luck and a confirmation.

## 2026-09-28 — the interval over stays measured

Under the declaration above, at `6693c552`, on the MacBook Pro M1 Pro, eight processes, 68 minutes.

| Control's area | Repeats | Spread of a repeat's area | Coverage of the true gain | Zero excluded (two-sided) | Whole interval above zero |
| --- | --- | --- | --- | --- | --- |
| 0.70 | 1 | 0 | 96.0 % | 4.5 % | 2.2 % |
| 0.70 | 5 | 0 | 95.5 % | 4.5 % | 2.0 % |
| 0.70 | 5 | 0.01 | 79.8 % | 20.2 % | 9.5 % |
| 0.70 | 5 | 0.03 | 41.2 % | 58.8 % | 27.5 % |
| 0.85 | 1 | 0 | 96.5 % | 3.2 % | 1.2 % |
| 0.85 | 5 | 0 | 96.8 % | 3.2 % | 1.2 % |
| 0.85 | 5 | 0.01 | 70.2 % | 28.0 % | 13.5 % |
| 0.85 | 5 | 0.03 | 31.0 % | 69.0 % | 33.2 % |

Nominal: 95 %, 5 %, 2.5 %; a rate's standard error over 400 datasets is about a point at 5 %.

**Conclusions.**

1. The first prediction held. Without a spread of repeats every setting covers 95.5–96.8 % and
   lies wholly above a true zero at most 2.2 % of the time. By the reading declared beforehand
   the percentile interval stays registered for the task; no bias-corrected interval is needed.
2. The second prediction failed at the higher level: a spread of 0.01 lowers coverage to 79.8 %
   at an area of 0.70, inside the band, but to 70.2 % at 0.85, below it. The higher the area, the
   narrower the interval over stays, so the same seed-to-seed spread is a larger share of the
   difference's variance.
3. The third prediction held: under a spread of 0.03 the interval covers a third of the time and
   excludes a true zero in most datasets.
4. Whether the seeds or the stays limit a comparison depends on how far the seeds spread, which
   is not known for this task yet. Here the difference's standard deviation over stays alone is
   about 0.004–0.006, so seeds dominate once a side's area strays by more than about 0.007–0.009
   between seeds; at 0.01 a true zero already lies wholly above zero four to five times as often
   as the level says. What stands between a seed's luck and a finding is the practical floor,
   which every comparison, the family's included, must clear: the larger of its fixed part and
   the control's spread over seeds.

**Limitations.** The level held under the generator's defaults only: binormal answers that never
tie, half their variance shared by the stay, areas of 0.70 and 0.85; the section below asks what
happens away from them. The spread is drawn independently per side, the worst case: both sides
of a repeat learn from one draw of labels, so part of a seed's luck is shared and cancels in the
difference. How far seeds spread on this task is unknown until the grid runs, and the floor
reads the control's spread only; a contender that spreads more than the control is exposed
beyond it. The fixed part of the floor is therefore chosen, in the registration before the grid,
against the rows above rather than against the interval's width. The seconds a comparison took
were measured beside seven other processes; on one process a comparison under the registered
10,000 resamples and five repeats takes about 51 s.

## 2026-09-28 — declared before the run: the level beyond the generator's defaults, and why a seed's spread costs what it does

**Question.** The level held under the generator's defaults only; a tree at fifty labels answers
in a handful of values, two networks from one backbone agree more than half, and a model at fifty
labels may rank barely above chance. Does the level hold away from those defaults? Beside it, the
explanation of the conclusions above — that coverage under a spread is what a normal interval
over stays keeps against the stays and the seeds together — is reproduced from committed code.

**Conditions.** Commit of this section; the machine and generator above. The decomposition: at
each spread of the registered suite and its level, the standard deviation over 300 datasets with
no true difference of the point difference, with the spread and without it, seeded apart from the
calibration; the coverage predicted is that of a normal 95 % interval as wide as the stays alone
make it. The robustness suite: five repeats, no spread, one assumption moved at a time — an area
of 0.60; the stay's share of an answer's variance at 0.2 and at 0.9; answers cut into 20 and into
5 equally likely levels at the population's quantiles, the true gain then the difference of the
areas those levels can reach, known exactly. A gain of 0.02 before the cut, or none; 200 datasets
for each rate, a 1,000-resample interval per dataset.

    uv run scripts/ranking_calibration_report.py --out data/report/verdict-statistics/ranking --decompose --datasets 300 --workers 8
    uv run scripts/ranking_calibration_report.py --out data/report/verdict-statistics/robustness --suite robustness --datasets 200 --workers 8

**Prediction.** Every setting of the robustness suite covers the true gain at least 92 % of the
time and lies wholly above a true zero at most 4.5 % of the time — two standard errors of a rate
over 200 datasets from nominal. Ties, counted half in the area and resampled with their stays, do
not move the interval off its level.

**Not a prediction.** The decomposition was first computed by a throwaway script on the same
seeds while the conclusions above were written, and it agreed with them: predicted coverage 79.9,
40.6, 71.5 and 31.9 % against 79.8, 41.2, 70.2 and 31.0 % measured. It is run again here from
committed code so the note cites what can be reproduced; it is a check made after the result.

**Reading, declared beforehand.** If the prediction holds, the percentile interval stays
registered for every kind of candidate the grid runs. If a setting fails it, the assumption it
moved is named, and the interval for the comparisons it describes is revisited before the grid
rather than after.

## 2026-09-28 — the level beyond the generator's defaults measured

Under the declaration above, at `e6c68914`, on the MacBook Pro M1 Pro, eight processes: the
decomposition in about a minute, the robustness suite in 21 minutes.

The decomposition, reproduced from committed code on the seeds the throwaway script used, gives
the numbers quoted in the declaration to the last digit:

| Control's area | Spread of a repeat's area | SD over stays | SD over seeds (expected) | Coverage predicted | Coverage measured |
| --- | --- | --- | --- | --- | --- |
| 0.70 | 0.01 | 0.0055 | 0.0064 (0.0063) | 79.9 % | 79.8 % |
| 0.70 | 0.03 | 0.0055 | 0.0194 (0.0190) | 40.6 % | 41.2 % |
| 0.85 | 0.01 | 0.0042 | 0.0064 (0.0063) | 71.5 % | 70.2 % |
| 0.85 | 0.03 | 0.0042 | 0.0194 (0.0190) | 31.9 % | 31.0 % |

The robustness suite, five repeats, no spread, 200 datasets per rate:

| Control's area | Shared | Levels | True gain | Coverage of the true gain | Zero excluded (two-sided) | Whole interval above zero |
| --- | --- | --- | --- | --- | --- | --- |
| 0.60 | 0.5 | — | 0.0200 | 95.0 % | 4.5 % | 1.5 % |
| 0.80 | 0.2 | — | 0.0200 | 96.5 % | 4.0 % | 1.5 % |
| 0.80 | 0.9 | — | 0.0200 | 95.0 % | 5.5 % | 2.0 % |
| 0.80 | 0.5 | 20 | 0.0199 | 96.5 % | 4.0 % | 1.0 % |
| 0.80 | 0.5 | 5 | 0.0185 | 96.0 % | 4.5 % | 2.0 % |

Nominal: 95 %, 5 %, 2.5 %; a rate's standard error over 200 datasets is about 1.5 points.

**Conclusions.**

1. The prediction held: every setting covers 95.0–96.5 % and lies wholly above a true zero at
   most 2.0 % of the time. By the reading declared beforehand the percentile interval stays
   registered for every kind of candidate the grid runs — a model near chance, two sides that
   share little or almost all of their answers, and answers in as few as five values.
2. Ties cost area, not calibration: answers in five levels reach 0.0185 of a stated gain of
   0.02, and the interval covers that gain as it covers an untied one.
3. The coverage lost under a spread of repeats is accounted for, to within 1.3 points, by the
   variance over stays and the variance over seeds added; nothing else in the procedure is
   needed to explain it.

**Limitations.** Every setting is still binormal within an outcome, with one window per stay as
the task has; heavier tails or several windows per unit are not covered. The robustness suite
moved one assumption at a time and at one level of area; it did not combine a low area with ties.
How far the grid's seeds actually spread is the open quantity, read from the selections on the
tuning side before the numbers of the protocol are registered.
