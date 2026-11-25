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
