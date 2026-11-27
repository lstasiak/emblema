# The published forecaster fine-tuned for a turbine or a farm it never read

On a turbine read as new, Chronos-2 zero-shot with the hour's operating conditions misses about
twice what the farm's boosted trees miss (`wind-fleet-cold-start.md`). This note fine-tunes it on
turbines it may read and asks how much of that gap closes: for a new turbine of a known farm, and
for a farm none of whose turbines it read. It is a pilot on validation years. Its rules were
written before each run in a file beside the run's files (`data/report/l60a/RULE.md`), not
committed, so it sets the shape of the next reading and claims nothing.

## 2026-11-24 — measured: how wide and how long to fine-tune, on a year before the reading's

**Question.** How much does fine-tuning lower Chronos-2's error on a new turbine, and does the
gain still grow with the adapters' rank, the learning rate and the steps, or with every weight
tuned?

**Conditions.**

| | |
|---|---|
| Instrument | `e2f3202e` on Colab's card. The Mac's grid ran the same scripts on the working tree before they were committed; the card repeats its best point. |
| Cells | Penmanshiel, turbines 01 and 08 read as new from 10 starts in 2019, days 2, 7 and 14 pooled: 1,439 windows. The form is `chronos2_nbm`: 17 temperatures, six operating signals known in the forecast hour. |
| Fit | The farm's other eight unheld turbines, hours before 2019. A series is 30 days of a turbine and the hour after; a draw reads the 47 to 720 hours from its start. A step reads 128 rows, six such series. Low-rank adapters on the library's own modules, a constant rate, so the adapters kept on the way are the fits of fewer steps. Seed 1. |
| Machines | Mac (M1 Pro, MPS, fp32): three fits to 2,000 steps. Colab (RTX PRO 6000 Blackwell): six fits to 8,000 steps, three at once on the card, 16 to 19 minutes each. `chronos-forecasting` 2.3.2, `peft` 0.18.1; answers in fp32. Validation numbers. |
| Measure | As in `wind-fleet-cold-start.md`: the geometric mean over windows of the answer's squared error over the bar's, 95 % interval over whole starts. The zero-shot answers are the Mac's. |
| Rule | The lowest ratio against zero-shot on 47 hours; within 2 % the cheaper point. The chosen point goes on even if the curve still falls. |

**The ratio against zero-shot on 47 hours**, by steps:

| Fit | 500 | 1,000 | 2,000 | 4,000 | 8,000 |
|---|---|---|---|---|---|
| Mac, rank 8, rate 10⁻⁵ | 0.942 | 0.887 | 0.848 | | |
| Mac, rank 64, rate 10⁻⁵ | 0.870 | 0.824 | 0.771 | | |
| Mac, rank 8, rate 10⁻⁴ | 0.843 | 0.798 | 0.734 | | |
| Card, rank 8, rate 10⁻⁴ | | 0.782 | 0.722 | 0.673 | **0.634** |
| Card, rank 64, rate 10⁻⁴ | | 0.782 | 0.713 | 0.695 | 0.671 |
| Card, rank 8, rate 3·10⁻⁴ | | 0.772 | 0.725 | 0.746 | 0.674 |
| Card, rank 64, rate 3·10⁻⁴ | | 0.986 | 1.015 | 0.878 | 0.926 |
| Card, every weight, rate 10⁻⁶ | | 0.805 | 0.761 | 0.708 | 0.664 |
| Card, every weight, rate 10⁻⁵ | | 0.761 | 0.710 | 0.663 | 0.676 |

The chosen point, rank 8 at 10⁻⁴ and 8,000 steps:

| | Ratio [95 %] |
|---|---|
| Against zero-shot, 47 hours | **0.634 [0.605; 0.663]** |
| Against zero-shot, every hour since the start | 0.521 [0.491; 0.550] |
| Against the farm's trees fitted on the same turbines and years | 0.897 [0.754; 1.045] |
| Zero-shot against those trees | 1.415 [1.168; 1.675] |

**Conclusions.**

- Fine-tuning removes about a third of the zero-shot error on these cells.
- The curve still falls at 8,000 steps (0.673 to 0.634), so the steps are not saturated.
- Tuning every weight is level with the adapters, not better (0.663 and 0.664 at best). At a rate
  of 10⁻⁴ rank 64 adds nothing to rank 8 (0.671 against 0.634); at 10⁻⁵ it read 9 % lower at 2,000
  steps, no more than the higher rate gives rank 8. At 3·10⁻⁴ it is unstable.
- The two machines agree: 0.722 on the card against 0.734 on the Mac at the same point.

**Limitations.** One fold of one farm (the turbines left out together), two turbines, ten starts. The point is the best of 33, and
its interval overlaps the next ones'. A plausibility threshold on the trainer's loss failed as
first written (one log line against another reads different series) and was restated as the mean
of five lines.

## 2026-11-25 — measured: fine-tuning closes most of the gap to the farm's trees on a new turbine, carries to another farm, and gains nothing from satellite series

**Question.** On the first month of 2020 to 2022, does the fine-tuned forecaster reach the farm's
trees on a new turbine? Fine-tuned on the other farm alone, does it lead the best answer a farm
has with no history of its own? Does satellite telemetry beside the turbines add to that?

**Conditions.**

| | |
|---|---|
| Instrument | `e2f3202e`; Colab (RTX PRO 6000 Blackwell), 17 fits and 48 answers in 2.6 hours, three at once on the card; versions as above. Validation numbers. |
| Cells | Those of `wind-fleet-cold-start.md`, rebuilt: Kelmarsh 2,907 windows over 31 starts, Penmanshiel 7,078 over 30; days 2, 7 and 14 pooled. |
| Fits | The chosen setting; every series ends before 2020. New turbine: the farm in folds, Kelmarsh four of one turbine, Penmanshiel five of two; a fold's fit reads the other turbines and answers only its own. New farm: every unheld turbine of the other farm, seeds 1 and 2. With satellites: as many 30-day series of the two ESA missions as of turbines, their channels in groups of at most 17 with no covariate. |
| Steps | 8,000 per 246 series, as in the section above: Kelmarsh fold 4,400, Penmanshiel fold 11,200, Kelmarsh from Penmanshiel 14,000, Penmanshiel from Kelmarsh 5,900. The fits with satellites run 1.70 times as long, so they draw the turbines as often. The seed-1 fits also keep their adapters at 8,000 steps, or run on to them, read beside; seed 2 has only the main steps. |
| Bars | The farm's trees fitted before 2020 with the turbine's fold left out, 1.06 and 1.07 against the trees of `wind-fleet-cold-start.md`. For a new farm: the best answer from the turbine's own data, the other farm's trees added; the cell's own regression leads it. |
| Rules | New turbine: an upper edge of at most 1.0 on both farms is parity; a point under 1.3 leaves a gap that more data or steps may close; above, fine-tuning does not reach a fleet. New farm: an upper edge of at most 0.80 against the bar and under 1.0 against zero-shot, in both seeds on both farms, makes a candidate claim; a point under 1.0 a descriptive gain. Satellites: an upper edge under 0.95 everywhere would justify a corpus of several domains. |

A first draft with 8,000 steps for every fit, one seed and equal steps for the fits with
satellites was corrected before anything ran.

**A new turbine**, on 47 hours unless said:

| | Kelmarsh | Penmanshiel |
|---|---|---|
| Zero-shot against the trees | 1.746 [1.608; 1.881] | 1.965 [1.848; 2.094] |
| Fine-tuned against zero-shot | 0.666 [0.645; 0.688] | 0.633 [0.614; 0.652] |
| **Fine-tuned against the trees** | **1.163 [1.082; 1.246]** | **1.244 [1.183; 1.311]** |
| The fit of 8,000 steps against the trees | 1.055 [0.981; 1.129] | 1.274 [1.210; 1.346] |
| Fine-tuned against the trees of `wind-fleet-cold-start.md` | 1.230 [1.152; 1.312] | 1.333 [1.267; 1.407] |
| Every hour since the start, against the trees | 0.807 [0.762; 0.851] | 0.883 [0.847; 0.920] |
| The same against the best of the classical bar with the farm, on day 2 | 0.951 [0.872; 1.038] | 1.066 [0.999; 1.135] |
| on day 7 | 0.822 [0.737; 0.913] | 0.926 [0.852; 1.011] |
| on day 14 | 0.794 [0.741; 0.854] | 0.862 [0.808; 0.920] |

The rule reads a gap that remains. The trees of `wind-fleet-cold-start.md` are fitted on the hours
before 1 January of each start's year, so for later starts they read 2020 and 2021, which the
forecaster does not; against them Penmanshiel's point passes 1.3. The rows by day were read
after the run. The best of their bar is those trees; they read 47 hours of the turbine, and the
forecaster up to 30 days.

Per temperature, on the cells: the forecaster misses 1.8 to 2.4 times what the trees miss on the
stator, the three pitch motors and the transformer on both farms, and 1.9 times on Kelmarsh's
front bearing. It misses 0.5 to 0.8 times as much on the processor, the top box and the
transformer cell on both. On Penmanshiel its error is even across nine of ten turbines; of the
four where the trees' error is largest it leads on two (0.63 and 0.79), is level on one (0.97)
and trails on one (1.76).

**A new farm**, fine-tuned on the other farm alone:

| | Kelmarsh, seed 1 | seed 2 | Penmanshiel, seed 1 | seed 2 |
|---|---|---|---|---|
| Against the bar without the farm | 0.651 [0.622; 0.681] | 0.639 [0.608; 0.669] | 0.761 [0.739; 0.784] | 0.790 [0.762; **0.817**] |
| Against zero-shot | 0.658 [0.635; 0.680] | 0.646 [0.623; 0.668] | 0.717 [0.697; 0.738] | 0.744 [0.722; 0.767] |

One upper edge passes 0.80, so the rule reads a descriptive gain. Seed 2 against seed 1 reads
0.982 [0.961; 1.001] and 1.038 [1.019; 1.057]: the noise of a fit, which the intervals over starts
do not hold. The fit on the farm's own turbines reads 1.012 [0.986; 1.039] against the other
farm's on Kelmarsh (135 series against 430) and 0.882 [0.865; 0.899] on Penmanshiel (344 against
180).

**The satellites beside the turbines**, against the same fit without them:

| | Kelmarsh | Penmanshiel |
|---|---|---|
| Seed 1 | 1.046 [1.024; 1.068] | 1.040 [1.023; 1.060] |
| Seed 2 | 1.063 [1.040; 1.088] | 0.984 [0.963; 1.006] |
| Across seeds | 1.065 and 1.044 | 1.003 and 1.021 |
| At equal steps, seed 1 | 1.059 [1.041; 1.077] | 1.103 [1.080; 1.126] |

No upper edge is under 0.95: the corpus is the fleet alone.

**The steps were not saturated.** The longer fit leads the shorter by 0.907 [0.895; 0.918] on
Kelmarsh's folds, 0.924 on Kelmarsh from Penmanshiel and 0.966 [0.955; 0.978] on Penmanshiel
from Kelmarsh; on Penmanshiel's folds by 0.976, within a fit's noise.

**Checked.** Every fit reached its steps with a falling loss; no fit read a turbine it answers;
the ratios recomputed from the answers' files without the reading script agree to three digits;
a simulation of the library's batching gives 1.703 for the satellites' steps.

**Conclusions.**

- On 47 hours the fine-tuned forecaster misses 16 and 24 % more than the farm's trees fitted on
  the same years, where zero-shot missed 75 and 97 % more. Parity is not reached.
- The gap is largest on the stator, the pitch motors and the transformer, where the hour's load
  plausibly matters most. Longer fits read lower in three of four comparisons.
- From day 7 on Kelmarsh and day 14 on Penmanshiel, handed every hour since the start, it leads
  the classical bar. No classical answer there reads both the farm and more than 47 hours of the
  turbine, so this is a hypothesis for a declared reading, not a result.
- Most of the gain carries to a farm never read (0.64 to 0.79 against the best answer without
  it). Whether the amount of data or the farm matters more is not separated here.
- Satellite series beside the turbines add nothing, and on Kelmarsh cost about 5 %.

**Limitations.** Two farms of one maker; one cut at 2020 for every start. Rules written before
the runs but not committed. Scaling the steps down with the series cost the smaller fits: about
9 % on Kelmarsh's folds, and likely the 0.80 edge on Penmanshiel, where both seeds were read at
5,900 steps (seed 1 at 8,000 reads 0.736 [0.711; 0.761] against the bar). The fits draw contexts of 47 to 720
hours and the main reading hands 47. One seed for the folds. Turbines and years held unread are
untouched.

## 2026-11-25 — declared before the run: classical answers that read a new turbine's own hours, and trees tuned as the forecaster's recipe will be

**Question.** Handed every hour since a turbine's start, the pilot's forecaster led the farm's
trees, which read 47 hours. Does it still lead once classical answers read the farm and those
hours too, on a new turbine and on a new farm? And what do the trees gain from tuning of the
kind the forecaster's recipe is about to get? This section declares one reading of the pilot's
own answers against new bars. No fit is run for it, and it claims nothing: it decides what the
next fits are for.

**Found since the pilot.** Penmanshiel's blade angle is unread in 2016 and 2017. Of its target
hours before 2020, 52 % lack one of the six operating signals; Kelmarsh's lack under 1 %. The
cells and the trees take only hours with all six, while the pilot's fits read every hour: of
its 430 series of Penmanshiel about 208 are in the cells' form. Its comparison of a farm's own
fit with the other farm's therefore mixes form with amount.

**Conditions.**

| | |
|---|---|
| Instrument | the commit that adds this section |
| Cells | the pilot's, the first month of 2020 to 2022: Kelmarsh 2,907 windows over 31 starts, Penmanshiel 7,078 over 30. Every tree ends before 2020. The measure and its interval are the pilot's. |
| Machine | Mac (M1 Pro), processor only. Validation numbers. |
| Tried first | Every command below on the 2019 cells of Penmanshiel already read: the plain trees equal the pilot's, each correction answers the same cells. Twenty-four deliberate breakages of the new code, each caught by a test. |

**The classical answers.** Each is fitted on the farm's other turbines before 2020, the new
turbine's fold left out. Their settings were fixed before any was read on these cells.

| Answer | What it is |
|---|---|
| trees | the pilot's: 300 rounds at a rate of 0.1, at most 100,000 rows, each with all six signals |
| slow | 1,000 rounds at 0.03 |
| gaps | every row, with those that lack a signal, the gap left missing |
| both | every row with all six signals, of the farm's other turbines and of the other farm's |
| offset | the trees, plus the mean of their residuals on the new turbine's own hours so far, shrunk as if a day of hours with no residual had come first |
| ridge | the trees, plus a ridge of that residual on the trees' answer, the residual of the hour before and the six signals, every coefficient shrunk towards no correction. Of eleven penalties it takes, hour by hour, the one that answered the hours before best |
| boosted | the trees, plus small trees (100 rounds, depth 3) fitted once a day to the residuals on the trees' rows, once the trees have answered two days of the turbine's hours |

A residual is a reading less the trees' answer for an own hour that could have been a cell. A
correction reads only hours before its cell's; the small trees those before its cell's day.

**The bars.** A bar is read at the member that reads lowest on the table's cells, per farm,
which favours the bar.

- *Tuned*, against answers from 47 hours: trees, slow, gaps, both.
- *With the turbine's history*, against answers from every hour since the start: those four and
  the three corrections.
- *For a new farm*, against answers from every hour since the start: the pilot's bar without
  the farm, the other farm's trees (every unheld turbine of it before 2020, the pilot's
  settings) and the same three corrections of them.

If slow, gaps or both reads more than 2 % under the trees on a farm (the pooled point), the
three corrections are also computed on it for that farm and join the second bar; with more
than one such member, on the lowest.

**The reading.** Each farm apart, days 2, 7 and 14 pooled; the tables by day are read beside
and decide nothing. The pilot fitted each arm more than once: for a new turbine at its main
steps and at 8,000, for a new farm in two seeds and at 8,000 steps. A row is read on the fit
that reads lowest against the bar on that farm, which favours the forecaster.

| The forecaster's answer from every hour since the start, against the bar that reads those hours | What follows |
|---|---|
| New turbine: upper edge under 1.0 on both farms | it stays a candidate for the held-out reading |
| New turbine: point under 1.0 on both farms, an upper edge at 1.0 or above | open; the final recipe's reading decides |
| New turbine: point at 1.0 or above on either farm | the pilot's lead was the bar's handicap |
| New farm: the same three outcomes | the same, for the new farm |

*Plausibility, or stop and find the cause:* the plain trees of this run equal the pilot's on
every cell; each correction answers every cell the trees answer; against the plain trees and
the pilot's own bars, the pilot's answers read what the pilot's tables hold.

The answers from 47 hours against the tuned bar are read beside: the pilot already read a gap
there. The next fits, a screen of the recipe's steps, draws and corpus, are declared in a
section of their own after this reading. If neither row group leaves a candidate or an open
question, that section first says what a fit is still for.

**Predictions.** The sources are the cells tried: one fold of Penmanshiel in 2019, where the
trees are fitted on years in which 73 % of hours lack a signal, so they are weaker than here.

| | Prediction | From |
|---|---|---|
| The best correction against the trees, new turbine | 0.65 to 0.85 on both farms | the ridge on the cells tried: 0.656 [0.588; 0.720]; stronger trees should leave less to correct |
| The forecaster against the bar with the turbine's history | Kelmarsh 0.95 to 1.24, Penmanshiel 1.04 to 1.36: the third row | the pilot's 0.807 [0.762; 0.851] and 0.883 [0.847; 0.920] against the trees, divided by the row above; on the cells tried a shorter fit read 1.169 [1.012; 1.300] |
| The best correction against the other farm's trees | 0.40 to 0.60 | the cells tried: 0.424 [0.403; 0.441] |
| The forecaster against the new farm's bar | 0.67 to 0.85: a candidate | on the cells tried the zero-shot model read 1.033 [0.990; 1.112] against that bar; the pilot's new-farm fits read 0.646 to 0.744 against the zero-shot model on 47 hours, at most 0.767 at an upper edge |
| Slow, gaps and both against the trees | 0.95 to 1.04 each | the cells tried: 0.994, 0.963 and 1.034 |

**Limitations.** The corrections' settings are fixed, not tuned. The forecaster is not given
the same correction of its own residuals: that needs its answer for every own hour, which the
pilot's files do not hold. The bar's member is chosen on the cells read, and so is the pilot's
fit; the first favours the bar, the second the forecaster. Before this section was committed,
a reviewer of the code ran the trees and the ridge on four starts of two Kelmarsh turbines in
these years, to compare a ridge clipped to its fitted range with the one declared. One number
came of it, the difference between the two (1 to 2 % of the squared error); the ridge stays as
declared, and no answer of the forecaster was read.

## 2026-11-27 — measured: the forecaster against trees that read a turbine's own hours

**Question.** As declared above (2026-11-25): does the pilot's forecaster, handed every hour
since a turbine's start, still lead once the trees read those hours too, on a new turbine and on
a new farm?

**Conditions.** Declared at `35a22c6f7`. The pilot's answers, the cells of the first month of
2020 to 2022, every tree fitted before 2020, on the Mac's processor (M1 Pro). Validation numbers.
The plausibility rows held: the plain trees equal the pilot's on every cell, each correction
answers the cells the trees answer, and the pilot's rows against its own bars repeat its tables.
Both on Kelmarsh and gaps on Penmanshiel read more than 2 % under the trees, so their
corrections joined the bar with history, as declared.

**The declared rows.** Each farm's lowest fit, days 2, 7 and 14 pooled.

| | Kelmarsh | Penmanshiel | Row |
|---|---|---|---|
| New turbine, from the start, against the bar with history | 0.927 [0.891; 0.965] | 1.148 [1.111; 1.183] | the pilot's lead was the bar's handicap |
| New farm, from the start, against the new farm's bar | 0.532 [0.516; 0.558] | 0.577 [0.555; 0.604] | stays a candidate |

**The classical answers** against the trees (the other farm's trees for the last row).

| | Kelmarsh | Penmanshiel |
|---|---|---|
| slow / gaps / both | 0.982 / 0.991 / 0.964 | 0.989 / 0.970 / 1.024 |
| offset / ridge / boosted | 0.883 / 0.803 / 0.902 | 0.886 / 0.791 / 0.887 |
| the other farm's trees, ridge | 0.666 [0.615; 0.714] | 0.416 [0.396; 0.437] |

On a new turbine the forecaster's ratio of mean squared errors is 1.075 on Kelmarsh and 1.126
on Penmanshiel. By day it falls with the turbine's hours: 1.021, 0.903, 0.866 on Kelmarsh and
1.272, 1.121, 1.064 on Penmanshiel (days 2, 7, 14). Penmanshiel's turbines read 0.94 to 1.33.

**Found after the reading, and fixed.**

| Defect | Effect | Fix |
|---|---|---|
| Kelmarsh's blade angle, an operating condition every answer reads, cut at 3.7° by the robust score's bound: 11 % of its hours at the bound | pitching above rated wind read as blades at rest | both farms published again with the robust deviation floored (ADR-0086); the other signals keep their hourly means to the bit |
| The other farm's trees read the answered farm's hours in their own farm's units | the new farm's bar weaker than it is | the hours scored as the trees' farm scores them |

Read again on the fixed data, the classical side changes only where these reach. On Kelmarsh
the ridge reads 0.805 and both 0.938. The other farm's trees with the ridge read 0.390
[0.359; 0.427] on Kelmarsh and 0.253 [0.232; 0.275] on Penmanshiel against those trees alone.
The zero-shot model from the start reads 1.179 and 1.199 against the new farm's bar, where it
read 0.945 and 1.028: that bar is now 17–25 % stronger. Every row of Penmanshiel that reads
nothing of Kelmarsh repeats to the digit, and its new-turbine row stands at 1.148. The pilot's
fits that read Kelmarsh cannot be answered again: the card kept no adapters. So the new-farm
row and Kelmarsh's new-turbine row are read again on new fits (the next section).

**Where the forecaster trails, read after the reading (post hoc).** On a new turbine against the
bar with history, each cell's error by the size of the hour's change in its target:

| Fifth of the hour's change | 1 (calmest) | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Kelmarsh | 0.84 | 0.90 | 1.03 | 0.96 | 1.18 |
| Penmanshiel | 0.95 | 0.99 | 1.09 | 1.12 | 1.17 |

The largest fifth holds 52 % and 57 % of the bar's loss. Neither side misses a level: the mean
signed error is near nought on every day. Per temperature the forecaster trails on the six that
follow load and pitch: stator (1.43, 1.52), transformer (1.92, 1.50), three pitch motors
(1.26–1.33, 1.63–1.66) and hub (1.49, 1.40). It leads on the controller's and the top box's
(0.62–0.80). The pilot's lead from the start came from knowing the turbine's level, which the
ridge gives the trees within a day; what remains is the response to the hour's load.

**Predictions against outcomes.**

| Prediction | Outcome | Why missed |
|---|---|---|
| best correction 0.65–0.85 | 0.803, 0.791 | — |
| new turbine: Kelmarsh 0.95–1.24, Penmanshiel 1.04–1.36 | 0.927, 1.148 | Kelmarsh: derived from the main fit, while the row reads the lowest (0.745 / 0.803 = 0.928) |
| the other farm's correction 0.40–0.60 | 0.666, 0.416 | Kelmarsh: carried over from a trial on Penmanshiel alone, and the units' defect |
| new farm 0.67–0.85 | 0.532, 0.577 | a gain on 47 hours carried over to the rung from the start; the bar's defect |
| tuned members 0.95–1.04 | 0.964–1.024 | — |

**Conclusions.**

1. On a new turbine of a known farm, the pilot's forecaster no longer leads once the trees are
   corrected by the turbine's own hours: it trails on Penmanshiel, and on Kelmarsh leads by the
   declared measure but trails in the mean squared error. Its lead from the start was the bar's
   handicap.
2. The gap lies in the hours of largest change and the temperatures that follow load, not in
   level.
3. The new farm's lead does not stand as read: its bar was weakened by the units' defect, and its
   answers read the cut blade angle.
4. Two data defects reached every reading of the farms' conditions; both are fixed at the
   publication and in the trees.

**Limitations.** The bar's member and the forecaster's fit were each chosen on these cells. The
forecaster was not given the corrections the bar was. The post hoc split was read on the first
reading's answers, before the fixes. Before this reading was declared, a reviewer of the code
ran the trees and the ridge on four starts of Kelmarsh (disclosed above).

## 2026-11-27 — declared before the run: a screen of the forecaster's recipe on the fixed publications

**Question.** On a new turbine of a known farm the forecaster trails the trees corrected by the
turbine's own hours (the section above). This screen fits one change of the recipe at a time:
more steps, complete hours, a larger adapter, draws of 47 hours, the other farm beside. On fold
0 it chooses the recipe, its steps and its switches, and reads where the forecaster still
trails. It also fits the new farm's answers again on the fixed publications. Whether the gap is
the recipe's is decided on every fold, in a run declared later. This section claims nothing.

**Conditions.**

| | |
|---|---|
| Instrument | the commit that adds this section; publications Kelmarsh `f147f8b1…`, Penmanshiel `f12c42c9…`. The rules below are applied by reading scripts kept with the run's files. |
| Cells | the first month of 2020 to 2022 rebuilt on those publications (Kelmarsh 2,907 windows over 31 starts, Penmanshiel 7,078 over 30). A new turbine is read on fold 0 (Kelmarsh 1; Penmanshiel 01 and 08), a new farm on every turbine. Every fit and tree ends before 2020. The measure and its interval are the pilot's. |
| Machines | tier M: two sessions on a Colab card (RTX PRO 6000), four compiled fits at once (1.7 times as fast as uncompiled, timed on this card); the reading on the Mac (M1 Pro). Validation numbers. |
| Tried first | a fit of 20 steps of each new arm on the Mac, every cell answered from a kept adapter; the reading run whole on stand-in answers of known ratios, which it returns |

**The fits.** Seeds 1 and 2; adapters of rank 8 at a constant rate of 10⁻⁴; batch 128; draws of
47 to 720 hours; 64,000 steps, the adapters kept every 8,000 and each answered.

| Arm | Read on | What it changes |
|---|---|---|
| s | a new turbine | nothing: the pilot's recipe, longer |
| sf | a new turbine (Penmanshiel) | spans cut from hours with all six signals (Kelmarsh lacks under 1 %) |
| r64 | a new turbine | adapters of rank 64 |
| c47 | a new turbine | every draw exactly 47 hours |
| b | a new turbine | every unheld turbine of the other farm beside |
| o | a new farm | the other farm's unheld turbines alone |
| of | a new farm (Kelmarsh) | o from hours with all six signals |

**Rules.** An arm's value on a farm and rung is the mean over its seeds of the log ratio against
the zero-shot model on the same cells.

- *Plausibility, or stop and find the cause.* On Penmanshiel, whose data did not change, s in
  seed 1 at 8,000 steps reads within 0.03 of the pilot's fit of 8,000 steps against the trees on
  fold 0, from every hour since the start: the same recipe, compiled. Every fit reaches its
  steps with its adapters kept, finite losses, and its last five logged losses under its first
  five (checked on the records the card returns). Every cell is answered, or the reading
  refuses.
- *Plateau.* On each farm and rung, the fewest steps within 2 % of the arm's best kept adapter.
  Under a constant rate a curve can rise 2–3 % between kept adapters (the pilot's dose of rank 8
  at 3·10⁻⁴: 0.725, then 0.746), so a plateau is read against the best, not the previous one.
  The longest fit reading more than 2 % above the best is named as harm.
- *A switch* is adopted where it leads its base by 3 % or more with both seeds on the same side
  in a test, and trails by 3 % or more, both seeds on the same side, in none. Seed 2 against
  seed 1 read 0.982 [0.961; 1.001] and 1.038 [1.019; 1.057] in the pilot, so 3 % with both
  seeds agreeing is above one seed's noise. Each arm is read at its own plateau, which can lie
  up to 2 % above its best: the cheaper point is chosen on purpose.
- *Tests*, both rungs unless named: sf against s on Penmanshiel; r64 against s and b against s
  on both farms; c47 against s on both farms, on 47 hours alone; of against o on Kelmarsh.
- *The recipe.* For a new turbine, s with each adopted switch among sf, r64 and b; its steps are
  the largest plateau of those arms over both farms and rungs. If c47 is adopted, the rung from
  47 hours is answered by c47, at its largest plateau over both farms. Switches adopted together
  are first fitted together in the final recipe's run; here each is read apart. For a new farm,
  of on Kelmarsh if adopted, else o; its steps are the largest plateau over both farms and
  rungs. An answer is the mean of the two seeds.

**Readings,** each farm apart, days 2, 7 and 14 pooled, all descriptive.

- The curves, the switches and the recipe, by the rules.
- Each arm at its plateau: from every hour since the start against the bar with history; from
  47 hours against the tuned trees; a new farm's arms against the new farm's bar.
- Where the gap lies, for the recipe's arms: the ratio of mean squared errors per temperature,
  for the six that follow load and pitch against the rest, and per fifth of the hour's change,
  as in the section above.

The forecaster's answers on its own hours, corrected as the trees' are, and the new farm's
verdict on them, are declared in a section of their own before that run.

**Predictions.** Where an interval was read before, the range comes from it; other rows name
what they rest on.

| | Prediction | From |
|---|---|---|
| s's plateau, each farm and rung | 16,000 to 48,000 steps | the pilot's losses still falling at 8,000 steps; 4,400 to 8,000 steps took 7.8 % off on Kelmarsh, 8,000 to 11,200 1.6 % on Penmanshiel |
| sf against s, each rung | 0.90 to 1.00: adopted or level | judgement: the pitch motors trail by 1.63–1.66 on Penmanshiel, whose fits read half their hours without the blade angle, and by 1.26–1.33 on Kelmarsh |
| r64 against s | 0.97 to 1.03: not adopted | rank 64 against 8 at 10⁻⁴ in the pilot's dose, one seed: 1.00, 0.99, 1.03, 1.06 from 1,000 to 8,000 steps |
| b against s | 0.95 to 1.03: not adopted | a farm's fit against the other farm's in the pilot: 1.012 [0.986; 1.039] and 0.882 [0.865; 0.899]; the trees with the other farm beside: 0.938 and 0.992 |
| c47 against s, 47 hours | 0.90 to 1.02 | no earlier reading |
| A new turbine, s at its plateau against the bar with history | Kelmarsh 0.82 to 0.97, Penmanshiel 1.02 to 1.18 | each farm's lowest pilot fit on every fold, 0.927 [0.891; 0.965] and 1.148 [1.111; 1.183], less up to 8 % from steps; Kelmarsh's read on the cut blade angle; fold 0 alone may lie outside (Penmanshiel's turbines 0.94–1.33) |
| A new farm, o at its plateau against the new farm's bar | 0.64 to 0.71 | the lowest of the pilot's new-farm fits, 0.516–0.558 and 0.555–0.604, times the bar's strengthening read on the zero-shot model (1.25, 1.17) |

**Limitations.** Fold 0 holds one turbine of Kelmarsh and two of Penmanshiel, so the switches are
read with the width of those cells. The plateaus, the switches and the new farm's arm are chosen
on the cells these readings read, which favours the forecaster as the bar's best member favours
the bar. Compiled fits are not bit-identical to the pilot's; the plausibility row bounds the
difference. The new farm's cells were read before. Both farms are of one maker.
