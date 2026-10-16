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
