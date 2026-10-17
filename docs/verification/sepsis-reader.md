# The sepsis task as a reader

Whether the sepsis task of PhysioNet 2019 reads a difference between backbones finer than the
intensive-care task of 2012 does. The intensive-care reader resolves 0.02 to 0.03 in area at
500 stays on half its tuning side (`verdict-statistics.md`, 2026-10-13); the gains self-supervised
pretraining reports in the literature at this share of labels are 0.01 to 0.02. The sepsis task
holds about five times the stays and twice the positives on the same share of its tuning side.
The rules in force are in `docs/preregistration.md`, the sepsis task's own section and the row
"The reader of the stage"; this note changes no rule and no verdict.

## 2026-10-16 — declared before the run: how far the probe's reading moves on the sepsis task

**Question.** On half the tuning side of the sepsis task, how far does the closed-form probe's
area move between stays and between seeds at 400 and 4,000 patients and at every patient, and
does any of those budgets resolve a gain of 0.01 four times in five by the registered rule?

**The data.**

- *The task.* `physionet2019-sepsis` (`03c3a159-…`): the first day of a stay, whether its label
  turns after it; 19,522 stays on the tuning side, 5.1 % positive. Half of them, held out by
  seed 101 as for the intensive-care reader, are scored: about 9,760 stays, about 500 septic,
  against the intensive-care half's 2,000 stays and 289 deaths.
- *The campaign.* `campaigns/reader-probes-physionet2019.toml`: `frozen_ridge@layer=concat`
  and `untrained_ridge@layer=concat` at 400, 4,000 and every patient, seeds 1 to 10. Sixty cells
  a backbone, on this machine's accelerator. Defined under the point of about 10⁹ values
  (`backbone-scale-1e9-m`, whose vocabulary holds 2019's channels), the four pretrainings of the
  intensive-care corpora (`backbone-clinical-m`, `-seed2`, `-seed3`, `-seed4`) and PhysioNet
  2019 alone (`backbone-physionet2019-m`). The mixture of four and the backbones of the stays
  cannot be read: their vocabularies end before 2019's channels.
- *The order of the runs.* The point of about 10⁹ values first, which answers the question
  alone; the four intensive-care pretrainings next, which give the error between backbones s on
  this task; 2019 alone descriptively. Each run is about two hours.
- *The trial.* Under the 10⁹ point, 2,000 validation windows of 2019 are read in 28 s with the
  model loaded (`scripts/state_spread_report.py --corpus physionet2019`); a cell at every patient
  reads about 19,500 windows, so about three minutes, and sixty cells about two hours. The
  intensive-care cell took 37 s over 4,000 windows of two days.
- *The reading.* `scripts/campaign_pairs_report.py` for the pairs, probe less its untrained
  encoder, paired over stays and seed by seed; `scripts/comparison_power_report.py` for the
  power at gains of 0.01 and 0.02 under the registered rule, as the intensive-care reader was
  read.

**Predictions.** The intensive-care half gives the paired interval of probe less untrained
encoder a half-width of 0.021 at 500 stays and 0.023 at every stay, and the probe's spread over
seeds 0.014 to 0.018 at 500, 0.010 at 1,000 and zero at every stay. The interval over stays
scales with the positives first: from 289 to about 500 it narrows to about three quarters by
the standard error of an area in two strata.

1. *The interval.* At every patient the paired interval's half-width lies between 0.014 and
   0.020.
2. *The seeds.* The probe's spread over seeds is 0.02 or more at 400 patients, where a draw
   holds about twenty septic stays, and 0.010 or less at 4,000; zero at every patient, by
   construction.
3. *The resolution.* No budget resolves a gain of 0.01 four times in five by the registered
   rule; every patient resolves 0.02.

**What follows.**

- If prediction 3 holds: the sepsis task is a second task, not a finer instrument. The
  positive control of transfer planned next is designed for a resolution of 0.02, with at least
  four pretrainings a side or a reading pooled over both tasks, and the intensive-care reader
  stays the reader of the stage.
- If it fails, a budget resolving 0.01: that budget on the sepsis task becomes the reader of
  the next declarations, by a rule row, and the four intensive-care pretrainings give its s.
- A cell that refuses, or a campaign that does not run whole, is named and its backbone left
  out; a run short of the 10⁹ point stops the reading.

**Limitations.**

- The reading is of one backbone's pairs at first; s between pretrainings waits for the four.
- Half the tuning side is scored so that readings pair with the intensive-care reader's by seed
  and division; the validation side stays unread, as it has since 2026-10-08.
- Hospital A of 2019 is the centre the stays of 2012 come from; the task over hospital B alone
  is not read here.

## 2026-10-16 — M1 Pro, MPS: how far the probe's reading moves on the sepsis task, measured

**Question.** The one declared above: how far the probe's area moves between stays and
between seeds on half the tuning side of the sepsis task at 400 and 4,000 patients and at every
patient, and whether any budget resolves 0.01 four times in five by the registered rule.

**Conditions.** Commit `e4824b7d`, the declaration; six campaigns defined and ordered from it
(`data/report/l21/define.sh`, `orders.sh`). The campaign under the point of about 10⁹ values
(`e986265f-…`) ran on this machine, MPS, fp32, torch 2.14.0: 60 cells in 83 min, 32 s a cell at
400 patients, 48 s at 4,000 and 74 s at every patient, all accepted (`run.sh`). The other five
campaigns are ordered and not yet run. Pairs by `scripts/campaign_pairs_report.py`
(`pairs.sh`, 9,761 stays, 10,000 resamples), power by `scripts/comparison_power_report.py`
(`power.sh`); CSV under `data/report/l21/pairs/` and `power/`.

**Area on the 9,761 stays held out**, mean over seeds 1 to 10 (the seeds' standard deviation),
under the point of about 10⁹ values.

| Budget | probe | probe at initialisation | probe less initialisation, paired over stays | seed by seed |
|---|---|---|---|---|
| 400 | 0.582 (0.030) | 0.647 (0.018) | **−0.065 [−0.080; −0.051]** | −0.065 ± 0.012 |
| 4,000 | 0.680 (0.008) | 0.728 (0.005) | **−0.047 [−0.064; −0.030]** | −0.047 ± 0.003 |
| all | 0.701 (0) | 0.743 (0.004) | **−0.042 [−0.060; −0.024]** | −0.042 ± 0.001 |

The half-width of the paired interval: 0.015 at 400, 0.017 at 4,000, 0.018 at every patient.
At every patient the probe's seeds draw the same labels, so its spread is zero; the encoder at
its initialisation is drawn under each seed, so its spread there is the draw's.

**Power of the registered rule** for a gain δ read as the probe less its untrained encoder,
from the measured components (the stays' part, the seeds' part and their interaction), 10
seeds, 9,761 stays; the intensive-care reader at 500 stays beside it from
`verdict-statistics.md`, 2026-10-13.

| Budget | δ = 0.01 | δ = 0.015 | δ = 0.02 | δ = 0.03 |
|---|---|---|---|---|
| 400 | 0.04 | 0.11 | 0.28 | 0.73 |
| 4,000 | 0.23 | 0.44 | 0.68 | 0.94 |
| all | 0.19 | 0.39 | 0.60 | 0.93 |
| intensive care, 500 stays, kind A | 0.08 | 0.18 | 0.35 | 0.73 |

**Predictions.**

1. *The interval*: held. At every patient the half-width is 0.018, within 0.014 to 0.020.
2. *The seeds*: held. The probe spreads by 0.030 at 400 patients, 0.008 at 4,000, zero at every
   patient.
3. *The resolution*: held in its first part, failed in its second. No budget resolves 0.01 four
   times in five (0.23 at best, at 4,000); every patient resolves 0.02 three times in five
   (0.60), not four, and 4,000 patients 0.68. Both resolve 0.03.

**Read after the run, beside the declared readings.** Under the point of about 10⁹ values the
probe reads the sepsis task worse than the same encoder at its initialisation at every budget,
by 0.042 to 0.065, every interval below zero and every seed agreeing. On the intensive-care task
the initialisation leads only from 1,000 stays up (by 0.01 to 0.04); on sepsis it leads already
at 400 patients. One backbone, descriptive.

**Conclusions.**

1. For a reading under one backbone the sepsis task is a finer instrument by about twice the
   power, not a fine one: at 4,000 patients the rule sees 0.02 two times in three (0.68 against
   0.35 at 500 stays of the intensive-care task) and 0.03 nineteen times in twenty (0.94 against
   0.73), and 0.01 at best one time in four. The interval narrows by a quarter only, from 0.023
   to 0.018, because the positives set its width, as the declaration said; the gain in power
   comes from the seeds' part, which 4,000 labels shrink to 0.008.
2. Whether it is finer between backbones waits for the error between pretrainings s on this
   task, which the four intensive-care pretrainings give. Until then the reader of the stage
   stays the intensive-care probe at 500 stays, and the positive control of transfer planned
   next is designed for a resolution of 0.03, or 0.02 under a reading pooled over both tasks.
3. Under this backbone pretraining removes what a linear head would use on the sepsis task at
   every budget. The pretrained weights read the first day of a stay worse than a draw does.
4. The run costs what the trial said: 60 cells in 83 min against about two hours predicted,
   under the trial's 28 s for 2,000 windows.

**Limitations.**

- One backbone. The error between pretrainings s on this task waits for the four intensive-care
  pretrainings, ordered and not run; their reading follows in a section of its own.
- The power at 400 patients is read from a seed spread of 0.030 with about twenty septic labels
  a draw; the budget is below any reading the stage would make.
- The task over hospital B alone is not read.

## 2026-10-17 — M1 Pro, MPS: the four intensive-care pretrainings and 2019 alone, measured

**Question.** The part the section above left open: the error between pretrainings s on the
sepsis task, from the four intensive-care pretrainings, and whether the registered rule between
two backbones resolves 0.01 or 0.02 at any budget here; PhysioNet 2019 alone descriptively.

**Conditions.** Commit `e4824b7d`, the five campaigns ordered from it with the one above. On
this machine, MPS, fp32, one after another: the four intensive-care pretrainings
(`backbone-clinical-m`, `-seed2`, `-seed3`, `-seed4`) in 68, 77, 90 and 106 minutes, the last
two beside another reading on the accelerator, and 2019 alone (`backbone-physionet2019-m`) in
81 minutes; 60 cells each, every cell accepted. Pairs by `scripts/campaign_pairs_report.py`
over one export of the five campaigns above (`pairs.sh`, 9,761 stays) and one of 2019 alone
(`pairs-p2019.sh`); s by the registered formula per budget over the six pairs of intensive-care
pretrainings (`data/report/l16/s.py`, `power/s.txt`); power by
`scripts/comparison_power_report.py` with that s for the pairs between pretrainings
(`power.sh --seed-gap B …`). CSV under `data/report/l21/pairs/`, `pairs-p2019/` and `power/`.

**Area on the 9,761 stays held out**, mean over seeds 1 to 10 (the seeds' standard
deviation); the probe at the encoder's initialisation and under 10⁹ from the section above.

| Budget | at initialisation | 10⁹ | intensive care s1 / s2 / s3 / s4 | 2019 alone |
|---|---|---|---|---|
| 400 | 0.647 (0.018) | 0.582 (0.030) | 0.645 (0.030) / 0.643 (0.017) / 0.643 (0.028) / 0.659 (0.014) | 0.651 (0.018) |
| 4,000 | 0.728 (0.005) | 0.680 (0.008) | 0.733 (0.004) / 0.730 (0.004) / 0.712 (0.007) / 0.718 (0.005) | 0.725 (0.003) |
| all | 0.743 (0.004) | 0.701 (0) | 0.749 / 0.747 / 0.729 / 0.724 (0) | 0.736 (0) |

**The pairs**, paired over stays (95 % interval), seed by seed (mean ± standard error).

| Kind | Budget | Reading |
|---|---|---|
| A, probe less initialisation, intensive care | 400 | −0.004 to +0.013; s4 +0.013 [+0.004; +0.021] ± 0.006, the others across zero |
| A | 4,000 | −0.016 to +0.006; s3 −0.016 [−0.028; −0.004] ± 0.003, the others across zero |
| A | all | −0.019 to +0.006; s3 −0.014 [−0.026; −0.001], s4 −0.019 [−0.033; −0.004], s1 and s2 across zero |
| A, 2019 alone | 400 / 4,000 / all | +0.005 [−0.003; +0.013] ± 0.008 / −0.003 [−0.013; +0.008] ± 0.002 / −0.007 [−0.019; +0.005] |
| B, seed j less seed i, intensive care | 400 / 4,000 / all | **s = 0.0096 / 0.0131 / 0.0159**, pooled 0.0131 |
| B, the widest pair | 400 / 4,000 / all | +0.017 [+0.009; +0.025] (s4 less s2) / −0.022 [−0.033; −0.010] (s3 less s1) / −0.024 [−0.039; −0.009] (s4 less s1) |

Half-widths of the paired interval: kind A 0.007 to 0.015 at 400, 0.009 to 0.017 at 4,000
and 0.011 to 0.018 at every patient (0.018 is 10⁹; the intensive-care pretrainings 0.011 to
0.014); kind B 0.006 to 0.008, 0.009 to 0.011 and 0.012 to 0.015.

**Power at ten seeds**, read directly off the draws (medians over pairs, kind A over the five
backbones), the rule as registered with s at the budget for kind B; the intensive-care reader at
500 stays from `verdict-statistics.md`, 2026-10-13.

| Kind | Budget | seed SE | interval SE | δ = 0.01 | 0.02 | 0.03 |
|---|---|---|---|---|---|---|
| A, pretraining's share | 400 | 0.008 | 0.005 | 0.16 | 0.78 | 1.00 |
| A | 4,000 | 0.002 | 0.005 | **0.43** | **0.97** | 1.00 |
| A | all | 0.001 | 0.007 | 0.33 | 0.87 | 1.00 |
| A, intensive care, 500 stays | | 0.012 | 0.010 | 0.08 | 0.35 | 0.73 |
| B, between pretrainings, s = 0.0096 | 400 | 0.006 | 0.003 | 0.00 | 0.28 | 0.99 |
| B, s = 0.0131 | 4,000 | 0.003 | 0.005 | 0.00 | 0.10 | 0.74 |
| B, s = 0.0159 | all | 0 | 0.007 | 0.00 | 0.05 | 0.39 |
| B, intensive care, 500 stays, s = 0.0057 | | 0.005 | 0.009 | 0.19 | 0.58 | 0.91 |

Kind A under 10⁹ alone read 0.23 / 0.68 at 4,000 above; the four intensive-care pretrainings
carry smaller seed and interaction parts, and the median over five moves to 0.43 / 0.97.

**Predictions, re-read over the five backbones.**

1. *The interval.* Holds under 10⁹ (0.018) and fails narrower under the intensive-care
   pretrainings (0.011 to 0.014 against 0.014 to 0.020): the 500 septic stays narrow the
   interval more than the two-strata scaling from 289 deaths gave.
2. *The seeds.* Holds at 4,000 (0.003 to 0.008) and at every patient (zero); at 400 the
   spread is 0.014 to 0.030, below 0.02 under two of the five backbones and under 2019 alone.
3. *The resolution.* As above: no budget resolves 0.01 four times in five (0.43 at best,
   kind A at 4,000). Every patient resolves 0.02 within a backbone (kind A, 0.87) and not
   between pretrainings (kind B, 0.05): s between the four intensive-care pretrainings is
   0.016 there, nearly three times the 0.0057 at 500 stays on the intensive-care task.

**Decision, by the reading declared.**

- *The sepsis task is a second task, not a finer instrument between backbones.* By the
  registered rule two backbones resolve a gain of 0.02 four times in five at no budget here
  (0.28, 0.10, 0.05) against 0.58 at 500 stays of the intensive-care task. The intensive-care
  reader stays the reader of the stage; no rule row changes.
- *Within a backbone the task reads finer than the stays at 500.* The probe less its
  initialisation at 4,000 patients resolves 0.02 in 0.97 and 0.01 in 0.43, against 0.35 and
  0.08 at 500 stays. A declaration about the pretraining's share under one backbone can be
  read here at 4,000 patients.
- The positive control of transfer planned next needs a resolution of 0.02 between backbones,
  and neither task alone gives it with four pretrainings a side: by the stage's threshold for
  n pretrainings a side, twice √(SE² + (s²/2)(1/n_c + 1/n_r)), four against four sit at about
  0.021 on the sepsis task at every patient (SE 0.007, s 0.016) and 0.019 on the intensive-care
  task at 500 stays (SE 0.009, s 0.0057), so a gain of 0.02 is read about one time in two on
  either. The control's declaration computes the power of a reading pooled over both tasks
  before it runs.

**Conclusions.**

1. On the sepsis task no pretraining helps the probe. 10⁹ reads worse than the initialisation
   by 0.04 to 0.07 at every budget; the four intensive-care pretrainings lie within ±0.02 of
   it, one below at 4,000 and two at every patient; 2019 alone, the task's own corpus, lies
   within ±0.01. On the intensive-care task the same probes gained 0.02 to 0.04 at 200 stays
   and lost from 1,000 up; here the loss starts at 400 patients.
2. The pretrainings differ more on this task than on the stays: s rises from 0.0057 to 0.016,
   and the spread of the four at every patient (0.724 to 0.749) is the size of the gain the
   stage looks for. A rule between two backbones is bound by that spread, not by the stays.
3. The campaigns cost 68 to 106 minutes each on this machine; the pairs over five campaigns
   2.5 hours in one process.

**Limitations.**

- s comes from four pretrainings of one recipe, as on the intensive-care task; 0.013 pooled
  against 0.014 registered.
- The probe is the reader; a network fine-tuned on the sepsis task is not read, and the
  validation side stays unread.
- The task over hospital B alone is not read.
