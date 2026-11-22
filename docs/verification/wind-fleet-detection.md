# Faults on wind turbines: do detectors with no label rank them above quiet stretches?

The product's alarms on satellite telemetry are departures of each channel from its own recent
hours, with no model and no label (`telemetry-channel-local.md`). This note reads the same
detectors on a second kind of system, wind turbines with labelled faults, beside the comparator
that industry would reach for, a model of normal behaviour. The data is CARE to Compare
(`wind-fleet-sources.md`): 95 excerpts of 36 turbines. An excerpt is a year to fit on (the
fitted year) and a window after it that is scored (the scored window), holding one event: the
lead-up to a fault, or a stretch without one.

## 2026-11-22 — measured: a pilot of the detectors on the fitted years

**Question.** Before any scored window is read: which form of the model of normal behaviour
answers the temperatures better, do residuals taken out of fold stand for those of later hours,
and does the statistic rank placebo events at chance?

**Conditions.** Commit `ac14b709` and the working tree that became the instrument; Mac (M1 Pro),
CPU; `data/report/l62c/pilot.py` and `pilot_placebo.py`. The fitted years of the 19 turbines
read, without the hours lying in a scored window of the same turbine (matched by a hash of the
row's averages, whatever its stamp). The last 61 days of each year stand in for a scored
window; scale, floor and models are fitted on the hours before them. Limits were written before
the run. The departures on an hour's readings ran on farms A and B only (21 excerpts).

| Check | Limit | Read |
|---|---|---|
| Normal behaviour, form | the lower squared error out of fold | trees 1.71, ridge 4.74: trees |
| Residuals of the stand-in window over those out of fold; temperatures answered at all | 0.8 to 1.25; at least 90 % | trees 1.13, ridge 1.00; 99.9 % |
| Hours of the stand-in window above the earlier hours' 98th percentile, median | 1 to 6 % | departures 0.9 to 2.1 % (two of twelve under 1 %), trees 3.4 % |
| Placebo: events of the true lengths and labels laid at random in the year, area | 0.5 ± 0.04 | trees 0.49 and 0.51; the answer with no model 0.49; departures 0.49 to 0.60, four of twelve outside; the event's length alone 0.57, outside by construction |

**Conclusions.**

- The model of normal behaviour is boosted trees; its out-of-fold residuals run about a tenth
  below those of later hours.
- The placebo limit was too tight for the 14 to 43 excerpts a placebo ranks: shuffled labels
  spread its mean area by 0.014 to 0.043, and one series of seventeen lies beyond two such
  spreads (the largest departure of the readings over 24 hours, chance 0.01).
- Eleven of the twelve departures stand above 0.5, mean 0.53, with no consistent tie to the
  event's length (rank correlation −0.32 to +0.38). A lean that follows the labels is not
  removed by shuffling them; it is a limit of the reading below.

## 2026-11-22 — declared before the run: the channel-local departures and a model of normal behaviour on the read half of the fault benchmark

**Question.** Does the family of channel-local departures, taken from the telemetry unchanged,
rank the excerpts with a fault above those without? Does a model of normal behaviour?

**Conditions.**

| | |
|---|---|
| Commit | the one that adds this section; the instrument at `a7db1abb` |
| Machine | Mac (M1 Pro), CPU; validation numbers |
| Data | CARE to Compare, version 6 (Zenodo 15846963), read from its files. Read: 19 turbines, 52 excerpts, 24 with a fault and 28 without (farm A 6 and 7, B 4 and 5, C 14 and 16). Held unread: 17 turbines, 43 excerpts (`docs/preregistration.md`); the scripts refuse their files. |
| Hours | An excerpt is laid on its clock hours. An hour is whole with six rows, all of normal operation (status 0 or 2). Detectors read whole hours only: 99.7 %, 97.1 % and 94.7 % of the hours with a normal row in farms A, B and C. A row with half its averages at exactly nought is a dropped row. A channel is read in an hour only with all six readings. |
| Channel-local departures | The package's `ChannelDepartures` on hourly windows of six readings, the twelve detectors of the telemetry's verdict: largest or mean over channels, the hour's summary or its readings, against the 5, 23 or 71 hours before. Channels: every ten-minute average except angles and counters (42, 56 and 226). Each is scored in robust deviations from its median over the fitted year, clipped at ±100; in standard deviations where the robust spread is nought (up to 42 of farm C's 238 averages). An hour is scored where it is whole and half its history holds a reading. |
| Normal behaviour | Per component temperature (24, 24 and 74): boosted trees (XGBoost, rate 0.1, 300 rounds) from the hour's wind speed, power, rotor speed, pitch and outside temperatures and the mean power of the 3, 6 and 24 hours before. Never the temperature's own past. Fitted on the excerpt's year; its residuals there out of fold, weeks dealt to six folds in turn; the scored window answered by the whole year's model. An hour's score is the largest, or the mean, over temperatures of the absolute residual in the robust deviations of the year's. Two detectors. |
| Statistic | Per excerpt and detector: the largest median of the hourly score over 12 hours running inside the event, at least 9 of them scored (12 hours is the span the benchmark's counter needs). The statistic is one minus the share of the fitted year's blocks whose largest such median is as high; a block is any run of consecutive hours of the event's length. Events no block reaches tie at the top. An event a detector cannot read takes the lowest statistic. |
| Measure | The area under the curve between excerpts with a fault and without, pairs inside a farm only, for the best detector of a family. Its chance is the share of 10,000 shuffles of the labels inside each farm (seed 1) whose best reaches it. One excerpt without a fault (87, farm B) lies inside the lead-up of excerpt 27 and is ranked by nothing: 51 excerpts, 24 and 27. |
| Reading | `data/report/l62c/run.sh` (`care_departure_scores.py`, `care_normal_behaviour_scores.py --form trees`), then `care_detection_reading.py read`. |

**Rules.** Two verdicts, each a family's best against chance. The smaller chance is held to
1.25 % and the other to 2.5 % (Holm, one-sided). A verdict can confirm and cannot close: on 19
turbines a family that fails has not been shown to read nothing.

| Outcome | What follows |
|---|---|
| The departures pass, their best above the answer with no model | the telemetry's detectors carry to a second kind of system with no label and no tuning; the read half is a detection task for a model of new systems, the departures its bar |
| The departures pass, their best not above it | the pass is read as stops, not as faults seen early |
| Normal behaviour passes and the departures do not | a system driven by its operating conditions is read next through the residuals of a model of normal behaviour; a forecaster that knows the conditions is read as that model, against the trees |
| Neither passes | unresolved; the areas stand as descriptions and detection for new systems stays on the satellites |

Descriptive, with no verdict:

- every detector's own area: inside each farm, with its own uncorrected chance and an
  approximate interval over turbines drawn again; each verdict's best with excerpt 87 ranked;
- the difference of a named pair, the mean residual against the mean departure of the hour's
  readings over 24 hours, its chance from swapping the two inside each turbine;
- the answer with no model, the share of an hour's rows not of normal operation; the event's
  length alone;
- the departures over the temperatures only, and scaled on a turbine's first 7, 30 and 90 days
  (by the hour's summary, six detectors each); the model of normal behaviour fitted on the
  farm's other read turbines, its residuals still scaled on the excerpt's own year;
- the benchmark's own score and its four parts over all 52 excerpts as labelled, with the alarm
  at the level that 0.5, 1, 2, 5 and 10 % of the fitted year's scored hours exceed; no level is
  chosen;
- should a verdict pass, its family read again with the hours of each fitted year that lie in
  a sibling excerpt's fault lead-up left out of the year's blocks.

**Seen before this declaration.**

- The tables of events: labels, lengths and descriptions. The event's length alone ranks at
  0.57: 23 of 24 faults last less than 9 or more than 16 days, 23 of 28 quiet events between.
  By their descriptions at least five of farm C's 14 read faults leave no trace a sensor would
  show. The detectors and channels were fixed by rule, not by the descriptions.
- The wind speed of all 95 excerpts (2026-11-20, `wind-fleet-sources.md`); no value was joined
  to a label.
- The fitted years of the read turbines (the pilot above and `data/report/l62c/audit.py`). In
  45 ordered pairs of one turbine's excerpts the scored window of one lies in the other's year,
  8 under shifted stamps. The years are fitted as given, so the earlier note's "no scored window
  lies in another excerpt's training year" holds only between the two halves.
- No sensor of a scored window but the wind speed.

**Predictions.** No earlier interval of ours exists on turbines; the ranges rest on mechanism
and on two published readings.

| | Predicted | Source |
|---|---|---|
| Departures, best of twelve | area 0.50 to 0.68; does not pass | a history of at most 71 hours against lead-ups of 2 to 66 days; a conformal detector found 5 of 18 faults on held turbines (Liu, Chen and Wang, WES Discussions, wes-2026-145) |
| Normal behaviour, best of two | 0.55 to 0.75; unresolved, or passes narrowly | the trees explain 0.94 of the temperatures' variance; the benchmark's authors separate faults with an autoencoder of normal behaviour (Gück and others, *Data* 9(12):138, 2024) |
| The answer with no model | 0.50 to 0.70 | stops inside lead-ups, by the events' descriptions |
| The benchmark's score at 2 %, the named pair's two detectors | 0.40 to 0.56 | a coin scores 0.5 and the authors' autoencoder 0.66 (Gück and others); the held-turbine reading above 0.506 [0.378; 0.632]; a nearly silent alarm scores about 0.4 by its accuracy alone |

**Power** (`care_detection_reading.py power` on the 51 excerpts' layout; bars from 20,000 draws,
powers from 2,000).

| Family | Area its best must reach at 1.25 % / 2.5 % | Power at 1.25 %, true area 0.75 | true area 0.80 |
|---|---|---|---|
| Departures, twelve | 0.75 to 0.77 / 0.73 to 0.75 | 0.41 to 0.91 | 0.71 to 0.99 |
| Normal behaviour, two | 0.71 to 0.72 / 0.69 to 0.70 | 0.68 to 0.85 | 0.89 to 0.96 |

The lower power is where one detector of the family carries the fault, the higher where all do.
With no skill the best of twelve stands at about 0.59 and the best of two at 0.53. With no fault
the shuffle passes a family of twelve at its level where excerpts are independent (1.25 % and
2.95 % at the two levels, within the draws' noise) and less often where a turbine's excerpts
share a part of their score (0.4 to 0.6 % and 0.9 to 1.5 %).

**Trial, before this declaration.** The exact commands ran on six excerpts: per farm, the first
with a fault and the first without (`data/report/l62c/trial.sh`). The reading shuffled labels
inside each farm, wrote no excerpt's statistic and printed only the sizes of its tables; its
files were deleted unopened.

**Limitations (declared).**

- Half the benchmark, 19 turbines: every number is a validation number. The held half waits for
  one reading of a later detector.
- Shuffling labels inside a farm takes the quiet events for exchangeable with the faults. 24 of
  the 52 events run beside another read turbine's in real time, and a turbine's excerpts share
  rows.
- 16 of the 24 faults and 11 of the 28 quiet excerpts hold a sibling's event in their fitted
  year, and 15 years hold a fault's lead-up, 9 of them a quiet excerpt's. A lead-up in the year
  raises its blocks and lowers the excerpt's statistic, which leans towards a larger area.
- The events are read as annotated; a lead-up's true start is unknown.
- The benchmark's score reads the alarm's level more than the ranking: its accuracy is two
  fifths of it and gates the rest.
