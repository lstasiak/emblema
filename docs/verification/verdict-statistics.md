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

## 2026-09-29 — the spread over seeds read from the selections, and declared on a fixed fifth before it runs

**Question.** How far does a cell's area move between seeds on the intensive-care task? The
least gain the endpoint must reach and the floor's fixed part are registered against it, before
the first cell of the grid. The selections' repeats were read first; what they cannot answer is
put to a campaign declared here before it runs.

**Conditions.** The eleven network selections, 44 orders placed at `54986419` and run on Colab
G4 sessions, read from the local registry by `scripts/campaign_pairs_report.py` and
`scripts/seed_spread_report.py` of the commit that adds this section; a thousand resamples for
the standard error over units. Each repeat holds out a fifth of the tuning stays, 800 stays with
108 deaths, chosen by the repeat's seed: repeat 1 of every selection scored the same 800 stays,
and repeats 1 and 2 share 152 of them.

    uv run scripts/campaign_pairs_report.py --campaign ID [--campaign ID ...] --out DIR
    uv run scripts/seed_spread_report.py --out DIR

Pooled over the 60 variants of each budget, three repeats each (area; SD over the repeats; SE of
one area over its 800 stays, in two strata; the SD left once each repeat's shift shared by every
variant is taken out):

| Budget | Median SD | RMS SD | RMS SE over stays | SD without the repeat's shift |
|---|---|---|---|---|
| 50 | 0.0506 | 0.0571 | 0.0283 | 0.0342 |
| 200 | 0.0310 | 0.0343 | 0.0267 | 0.0297 |
| 1,000 | 0.0128 | 0.0149 | 0.0232 | 0.0148 |
| all | 0.0125 | 0.0145 | 0.0220 | 0.0140 |

**Conclusions from the selections.**

1. A repeat changes the seed, the draw of labels and the stays scored at once. The grid holds
   the stays scored fixed, so a selection's spread is not the spread the floor reads.
2. At 1,000 and every stay the repeats spread less than one area's own error over its stays.
   Three fifths, shared by every selection, cannot separate a seed from the stays it was scored
   on: the seed's own part there is not measured.
3. At 50 and 200 the repeats spread more than the stays alone explain. The spread without the
   repeat's shift, 0.034 and 0.030, still holds each variant's own answer to its fifth, so it
   bounds a seed's spread on fixed stays from above rather than measuring it.

**The run declared.** `campaigns/seed-spread-physionet2012.toml`: the grid's control, the
network trained from nothing, and its endpoint, full fine-tuning under the mixed backbone, each
at the variant its selection chose per budget; budgets of 50, 200, 1,000 and every stay; seeds 1
to 10; every seed scored on one fifth of the tuning stays fixed by a seed no selection used,
`division_seed = 101`, so repeats differ in the draw of labels and the model's seed only, as the
grid's do. The validation side is not read. A campaign divided this way chooses nothing; the
harness refuses to read a selection from it. Eighty cells, ordered to the ml pool and run on one
G4 under CUDA MPS, four processes.

**Predictions.**

1. At 200, the SD of the area over the ten seeds lies between 0.010 and 0.030 for the network
   from nothing and between 0.005 and 0.025 for full fine-tuning.
2. At 1,000 and every stay, both arms spread by at most 0.015.
3. At 50, the network from nothing spreads by at least 0.020.

**Reading, declared beforehand.** Neither reading consults any difference between the two arms.

- *Least gain at the endpoint.* 0.02 in area is a judgement: the smallest gain at 200 labels
  worth reporting on this task, not a quantity measured here. It stands if neither arm spreads
  by more than 0.02 at 200. Otherwise it is raised to the larger arm's SD at 200, rounded up to
  0.005, since a gain one seed's luck reaches is not the least that counts.
- *Fixed part of the floor.* The smallest SD of the control over the four budgets, rounded down
  to 0.005 and not below 0.01. The measured part, the control's spread over the grid's own
  seeds, still binds wherever it is larger; the fixed part keeps a grid whose seeds happen to be
  quiet from lowering the floor below the spread measured here.

**Limitations.** A fifth of the tuning side, 800 stays, is scored rather than the 3,994 of the
validation side. The part of a seed's spread that comes from how its answers meet particular
stays shrinks with more stays, so a spread read here errs high, towards a stricter floor. Ten
seeds read an SD to about a quarter of itself. The arms' variants were chosen on fifths that
overlap this one by about a fifth, which may raise the level of the areas and not their spread.

## 2026-09-29 — the spread over seeds on a fixed fifth measured

Under the declaration above: the campaign `596849cd…` of
`campaigns/seed-spread-physionet2012.toml`, four orders placed at `c9a04936` and run on one Colab
G4 under CUDA MPS, four processes, the longest order in 56 minutes; read by the same two
scripts. All 80 cells scored the same 800 stays, 127 deaths; the full fine-tuning arm started
from the mixed backbone's weights `sha256:869ed545…`. A correction to the section above: its 108
deaths hold for the selections' first fifth only; the three fifths hold 108, 103 and 115 deaths
in 800, 799 and 799 stays.

| Budget | Arm | Mean area | SD over 10 seeds | SE over stays |
|---|---|---|---|---|
| 50 | from nothing | 0.623 | 0.0526 | 0.0271 |
| 50 | full fine-tuning | 0.676 | 0.0444 | 0.0245 |
| 200 | from nothing | 0.651 | 0.0406 | 0.0261 |
| 200 | full fine-tuning | 0.692 | 0.0400 | 0.0251 |
| 1,000 | from nothing | 0.744 | 0.0352 | 0.0239 |
| 1,000 | full fine-tuning | 0.736 | 0.0227 | 0.0239 |
| all | from nothing | 0.782 | 0.0109 | 0.0212 |
| all | full fine-tuning | 0.764 | 0.0186 | 0.0220 |

**Conclusions.**

1. The first prediction failed: at 200 both arms spread by about 0.040, above both bands.
2. The second failed at 1,000 (0.035 and 0.023) and for full fine-tuning at every stay (0.019);
   it held for the network from nothing at every stay (0.011).
3. The third held: at 50 the network from nothing spreads by 0.053.
4. The selections understated the spread: by about a quarter at 200 (0.031 against 0.040) and
   by half or more at 1,000 (0.013 against 0.023–0.035). Every selection's repeat learnt and
   scored on the same division as every other's, so the sixty variants of a budget read three
   divisions, not sixty; ten seeds on one fifth read ten.
5. By the reading declared beforehand: the larger arm's SD at 200 is 0.0406, above 0.02, so the
   endpoint's least gain is raised to **0.045**; the control's smallest SD, 0.0109 at every stay,
   rounds down to 0.010, so the floor's fixed part is **0.01**.

**Limitations.** The areas are read on a fifth of the tuning side and are not the grid's; the
difference between the arms is not read here and decides nothing. The spread is of one seed's
area; a grid pools its seeds per stay, and its floor reads the control's spread over its own
seeds beside the fixed part.

## 2026-10-08 — declared before the run: the seeds and the stays a comparison at 50 stays needs

**Question.** The next comparisons on the intensive-care task are expected to differ by about
0.02 in area. They include a readout of other layers, longer training of the larger shape, robust
normalisation, the weight of the task's data in the mixture, pretraining hyperparameters, a second
objective and patches. Each is read by the replacement rule on the fifth of the tuning side held
out by seed 101 (800 stays) over seeds 1 to 10. How often does that rule replace a configuration
whose true gain is 0.02? Which numbers of seeds and stays scored make it do so four times in five?

**What is run.** Nothing is trained. The run resamples answers already exported:

- `data/report/l1/yardstick/predictions.csv` and `data/report/l5/yardstick/predictions.csv` hold
  the curve's yardstick on the validation side: 3,994 stays, seeds 1 to 10, nine backbones, at
  20, 50 and 200 stays. These cells have been read and are read here again only for their
  spread. No comparison of the claim is made.
- `data/report/l2/pairs/predictions.csv` holds the pretext campaigns on the fifth itself: 800
  stays, the same seeds, four backbones. This checks the spread at the size where the rule runs.

**Three kinds of comparison**, each a stand-in for upcoming readings:

| Kind | Pairs | Stands in for |
|---|---|---|
| A: two readouts of one backbone | `frozen_ridge` against `frozen_probe@learning_rate=0.03` under each backbone | another layer or several layers read under the same weights |
| B: one readout under two backbones | `frozen_ridge` under every pair of backbones of the same shape | longer training, normalisation, mixture weights, hyperparameters, objective |
| C: two networks trained from labels | the network from nothing against full fine-tuning; the network from nothing in one campaign against itself in another | normalisation and patches read without pretraining; the second pair is a null pair |

**Method.**

1. A design is a number of seeds K ∈ {5, 10, 20, 30} and a number of stays scored
   N ∈ {800, 1,000, 1,333, 2,000}. These are the held-out shares one in five, four, three and
   two, which the campaign file already offers through `one_in`.
2. *Resampling the stays.* For each pair and budget, 2,000 draws. Each draw takes N stays from the
   3,994 with replacement, in the two strata of the outcome at the population's prevalence.
   For every seed it computes both sides' areas and their difference, and the standard error of
   the difference pooled over the seeds. That standard error comes from placement values
   (DeLong), the normal approximation of the stratified bootstrap. It is first checked
   against the 10,000-resample intervals of `scripts/campaign_pairs_report.py` on the L-2 pairs,
   and it is used only if the two agree to within 10 %.
3. *Three parts of the variance* follow from the draws, per pair, budget and N:
   - the seed's own effect on the difference: the spread between seeds within a draw, less
     the interaction;
   - the shared effect of which stays were scored, common to every seed of a draw;
   - the interaction between seed and stay.
4. *The rule, simulated.* A seed's difference is δ, the true gain, plus the seed's effect, the
   shared effect and the interaction, each drawn from a normal distribution with the variance
   estimated in step 3; 20,000 simulated readings per design. The rule is applied as registered:
   - the 95 % interval of the difference pooled over seeds lies above zero;
   - the mean over seeds exceeds twice its standard error;
   - for kind B only, the mean exceeds the difference between the two pretraining seeds of the
     mixture, held at 0.004 (validation side) and at 0.008 (fifth).
5. *The model checked.* The resampled draws cannot draw a new seed: they hold the ten that ran.
   At K = 10 the model is therefore run once more with the seeds' effects held at the ten
   measured. It is then compared with applying the rule directly to the resampled draws, shifted
   so that the expected difference is δ. The model is used if the two powers agree to within
   0.05 for kind B at 50 stays at every N and δ. Otherwise only the direct reading at K = 10 is
   reported, and no other K is answered.
6. Power is the share of simulated readings in which the rule replaces, at
   δ ∈ {0, 0.01, 0.02, 0.03, 0.04}. At δ = 0 it is the rate of false replacement. Each kind's
   power is the median over its pairs; the pairs' range is printed beside it.

**Reading, declared beforehand.**

- For each kind at 50 stays, the chosen design is the one with the fewest seeds whose power at
  δ = 0.02 reaches 0.8. Ties go to fewer stays held out. The number of stays scored costs almost
  nothing; each seed costs one more cell.
- If no design in the grid reaches 0.8 for a kind, it keeps 10 seeds and the largest share. The
  gain that design detects four times in five is then stated as the smallest gain its readings
  can resolve.
- The chosen design enters each later declaration of the stage. The pretext rule in
  `docs/preregistration.md` names the fifth and seeds 1 to 10. If the design changes it, the
  change is a configuration row in the register, made before any reading under the new design.
- 20 and 200 stays are reported and choose nothing. FD001 (16 engines on the fifth) is not
  simulated: its readings are descriptive.

**Predictions.** Their sources are the intervals of the L-2 pairs on the fifth
(`pretext-variants.md`, 2026-10-06) and of the curve's pairs on the validation side
(`intensive-care-curve.md`, 2026-10-05). At 50 stays the half-widths on the fifth are 0.015 to
0.021, a standard error of 0.008 to 0.011. On the validation side they are 0.007 to 0.009, a
standard error of 0.0036 to 0.0046.

1. *The design in force is underpowered.* For kind B at K = 10 and N = 800, power at δ = 0.02 lies
   between 0.25 and 0.65 (δ / SE from 1.8 to 2.5, less the interval's 1.96).
2. *The stays limit it, not the seeds.* For kind B at N = 800, going from 10 to 30 seeds narrows
   the interval by less than 20 %. Power at δ = 0.02 stays below 0.8 at every K.
3. *Half the tuning side fixes it.* For kind B at N = 2,000, power at δ = 0.02 lies between 0.6
   and 0.95 at K = 10 and reaches 0.8 at K ≤ 20. The interval's standard error scales as about
   1/√N, between 0.005 and 0.007.
4. *A readout is cheaper to read than a backbone.* Kind A's seed-to-seed spread of the difference
   at 50 stays lies below kind B's: both sides share the same draw of labels and the same states.
5. *The rule keeps its size.* The rate of false replacement at δ = 0 is at most 0.03 in every
   design, and the null pair of kind C falls within the same bound.

**Limitations.**

- The stays are resampled from the validation side. The fifth is a different sample of the same
  population; the check on the L-2 pairs bounds the difference at N = 800 only.
- A true gain is modelled as a constant shift of every seed's difference. A real variant can also
  change the spread.
- Only one pair of pretraining seeds exists. Its difference enters as a fixed threshold, and the
  spread it carries between backbones is not simulated: no number of seeds or stays removes it.
- Seeds other than the ten that ran are drawn by the model, whose normal parts are checked only
  against those ten.
