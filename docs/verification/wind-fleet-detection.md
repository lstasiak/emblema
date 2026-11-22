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

## 2026-11-22 — measured: the first reading, by an instrument that tied an event's score to its stops

**Question.** As declared above. The reading ran once. Its instrument scored an event lower the
less of it was in normal operation, which follows the label, so its numbers say nothing of the
detectors and are kept as the record of what was read.

**Conditions.** Declaration `5b60406e`, instrument `a7db1abb`; Mac (M1 Pro), CPU; validation
numbers. `data/report/l62c/run.sh` in parts of six excerpts, then `read.sh`. Six workers on farm
C's excerpts exhausted the memory; from the fifth part four ran. Departures 4 h 14 min, normal
behaviour 28 min.

| | Best detector | Area | Chance | Verdict |
|---|---|---|---|---|
| Departures, twelve | mean of the hour's summary, 6 hours | 0.486 | 0.95 | not confirmed |
| Normal behaviour, two | largest residual | 0.463 | 0.78 | not confirmed |

| Beside | Area | In farms A, B, C |
|---|---|---|
| The answer with no model | 0.775, interval 0.64 to 0.91, own chance 0.001 | 0.98, 0.22, 0.78 |
| The event's length alone | 0.574 | 0.30, 0.75, 0.61 |
| The twelve departures, each | 0.30 to 0.49 | 0.02 to 0.17, 0.50 to 1.00, 0.30 to 0.54 |
| Normal behaviour fitted on the farm's other turbines | 0.50 and 0.55 | 0.10, 0.69 and 0.75, 0.56 and 0.62 |
| The benchmark's score at 2 % | departures 0 to 0.47; normal behaviour 0.52 and 0.57 | |

The named pair differed by 0.000 (chance 1.0). No verdict passed, so the check without sibling
faults' hours did not run.

**Why the areas lie below a half** (`data/report/l62c/posthoc_coverage.py`, `pilot_gaps.py`).

- The detectors read whole hours of normal operation only. A fault's event held 55 % such hours
  on average, a quiet one 93 %.
- In farm A five of the six read faults hold no row of normal operation in their event, by
  the score files' row masks, so no hour a detector could score. The
  farm's status was derived from its failure logs; the publisher's guidance for the benchmark
  (`docs/care2compare_faq.rst` in its code) is that there the labels "should mainly be used for
  filtering training data. For prediction-time evaluation, they should largely be ignored."
  All five events no detector could read were these faults, and the rule ranked them lowest.
- The statistic set an event with gaps beside whole blocks of its year. The largest of fewer
  medians is smaller, whatever a detector saw: the share of an event's hours that were scored
  ranks the faults at 0.15 by itself.
- The fitted years show it with no event read. Placebo events with the true events' gaps rank
  at 0.33 to 0.39 by the first statistic, 2.2 to 3.6 spreads of shuffled labels below a half.

**Seen after the reading, on its labels** (`posthoc_read_only.py`). Among the 46 excerpts a
detector could read, the first statistic gave 0.35 to 0.63. The median of an event's running
medians instead of their largest gave the departures 0.46 to 0.68, normal behaviour 0.68 and
0.70, and the model fitted on the farm's other turbines 0.76 and 0.77; the upper quartile
0.41 to 0.69. These were chosen among after the labels were read and confirm nothing.

**Conclusions.**

- The first reading is void as a reading of the detectors: its masks and its statistic tied an
  event's score to how much of it was in normal operation.
- The fault is the instrument's and is repaired at its source; the reading is repeated below.
- The status alone ranks the faults at 0.77: in farm A by construction (0.98), since its status
  marks the lead-up; at 0.22 in B and 0.78 in C, where it is the operator's.

## 2026-11-23 — declared before the run: the corrected reading on the same read half

**Question.** As on 2026-11-22, by an instrument whose reading of an event does not hang on its
stops: do the departures, a model of normal behaviour, and that model fitted with none of the
turbine's own rows rank the excerpts with a fault above those without?

**What changes, and why.** Measured on the fitted years of the read turbines, with no score of
a scored window read (`data/report/l62c/fitside_stops.py`, `fitside_decay.py`, `pilot_gaps.py`,
`fitside_settled.py`).

| | First reading | Corrected | Why |
|---|---|---|---|
| Farm A's scored window | read by its status | every row read; the fitted year still by its status | the publisher's guidance; and the rows of status 3 and 4 there are a running turbine (without power in wind above 5 m/s 2 to 3 % of the time, as with status 0), where in farms B and C the rows not of normal operation are mostly a stopped one (93 to 99 % in B; in C 99 % for status 3, 60 % for 4, 40 % for 5) |
| Hours read | every scored hour | settled hours: the 24 before all scored, in the year and the event alike | 1 to 3 hours after a stop of six hours or more a score stands 1.1 to 2.6 times a settled hour's, 13 to 24 hours after 1.0 to 1.4, later 1.0; settled hours are 52 % of a year's |
| Blocks of the year | whole | read at the hours, from their start, where the event holds a running median; left out with fewer than half of them | an event with gaps is set beside blocks with the same gaps |
| What is compared | the largest running median | their median, the event's level | a fault that grows over days lifts the level through its lead-up; seen to rank higher after the first reading |
| An event a detector cannot read | ranked lowest | left out, counted | an event with no 12 hours running of which 9 are settled, or whose year holds no block to set beside it, gives an early detector nothing to judge its turbine by, and such events are mostly faults. An excerpt leaves a family's ranking when any of its detectors cannot read it, so the verdicts may rank different excerpts |
| Verdicts | departures; normal behaviour | those and normal behaviour fitted on the farm's other read turbines | a turbine with no history is the product's question; made a verdict after it was seen to rank highest |
| Answers with no model | the share of an hour's rows not of normal operation | that, and the share of an event's rows not read; both in farms B and C, where the status says what the turbine did | in farm A the status is the label |

Everything else is as declared on 2026-11-22: data, held turbines, whole hours, detectors,
measure, shuffles, the excerpt left out of rankings, the benchmark's score, which in farm A
counts every scored row.

The checks below read six of the sixteen detectors of the verdicts: three departures, both of
normal behaviour, one of the farm's model. The last row counts the scored windows' settled hours
and reads no label.

| Check on the fitted years | Limit, written before | Read |
|---|---|---|
| Placebo events with the true events' gaps, read through them, before the settled rule and with an unreadable placebo ranked lowest: area | within two spreads of shuffled labels of a half | level 0.44 to 0.50, peak 0.44 to 0.52; all within, the level below a half in all six |
| Placebo events with the gaps the year has, settled hours, on the 27 excerpts whose every placement was read: area | the same | 0.47 to 0.56; all within (−0.8 to +1.7 spreads) |
| That placebo's statistic: rank correlation with the block's share not settled | within ±0.10 | −0.04 to +0.04 for the departures and the farm's model; normal behaviour +0.10 and +0.13, one outside |
| A year's blocks: rank correlation of the level with the share not settled | within ±0.10 | departures −0.06 to +0.02; normal behaviour and the farm's model +0.12 to +0.14, outside |
| True events with a settled span | | farm B 8 of 8, farm C 27 of 30 |

**Conditions.** Commit: the one that adds this section; the instrument at `c08aab6a`. Mac
(M1 Pro), CPU; validation numbers. Farms B
and C keep the first run's score files, which the correction does not touch: one excerpt of
farm B scored again equals its file to the bit. Farm A's 13 excerpts are scored anew
(`data/report/l62c/rerun.sh`), then `care_detection_reading.py read` with two named pairs:
normal behaviour's mean residual against the departures' mean over channels of the hour's
readings beside the 23 hours before, and the farm's model against the excerpt's own, both by
the mean residual.

**Rules.** Three verdicts, each a family's best against chance; the smallest chance is held to
0.83 %, the next to 1.25 %, the largest to 2.5 %, and a failure stops the verdicts ranked after
it (Holm, one-sided; `confirmed` in `verdicts.csv`). A pass is a hypothesis for the held
turbines, since the level was chosen on these labels; a failure closes nothing.

Two checks stand beside a pass, by point values of the reading's tables:

- It is read as faults seen early only if its best detector's area in farms B and C
  (`operators_status` in `verdicts.csv`) exceeds the areas there of both answers with no model
  (`operators_status` of `availability` and `event_unread_share` in `areas.csv`).
- It is read as the events' length if its best detector's area by the last seven days
  (`last_week`) is below 0.70 for a family of two or 0.75 for the twelve.

Also beside every verdict: its best detector's areas with every scored hour read, by the events'
peaks, with the unreadable events ranked lowest, and with excerpt 87 ranked. Should a verdict
pass, its family is read again with the hours of each fitted year that hold a row of a sibling
excerpt's fault lead-up left out of the year's blocks; that script is written after the reading.

A row below follows only from a pass that clears both checks; otherwise the pass is reported and
nothing follows. The rows add.

| Outcome | What follows |
|---|---|
| The farm's model passes | a model that never read the turbine ranks its faults; it is the bar a model of new systems is read against on the held turbines |
| Normal behaviour passes | a system driven by its operating conditions is read through the residuals of a model of normal behaviour |
| The departures pass | the telemetry's detectors carry to a second kind of system |
| None passes | unresolved; detection for new systems stays on the satellites |

**Seen before this declaration.** Everything listed on 2026-11-22; the whole first reading;
the areas quoted above under the median, the upper quartile and the largest median on 46
excerpts, with each detector's own chance (the farm's model's below 0.005) and the areas in
farms B and C; the share of each event's hours that were scored, by label and farm; and, from
the trial, the count of hours scored in two of farm A's excerpts. The level and the third verdict
were chosen with those in view.

**Predictions.** From what was seen after the first reading, by another statistic on 46
excerpts, so wide.

| | Predicted | Source |
|---|---|---|
| Departures, best of twelve | 0.45 to 0.70; does not pass | 0.46 to 0.68 after the first reading |
| Normal behaviour, best of two | 0.55 to 0.75; unresolved | 0.68 and 0.70 |
| The farm's model, best of two | 0.60 to 0.80; unresolved, or passes | 0.76 and 0.77 |
| The answers with no model, farms B and C | 0.55 to 0.75 | the first reading: 0.22 and 0.78 by the status; the share of an event's hours not scored ranked the faults at 0.85 over all farms |

**Power** (`care_detection_reading.py power`, the layout of 51 excerpts; bars from 20,000 draws,
powers from 2,000). A family of two must reach 0.73, 0.72 and 0.69 to 0.70 at the three levels, a family
of twelve 0.76 to 0.78, 0.75 to 0.77 and 0.73 to 0.75. At the strictest level a true area of 0.75
passes with power 0.61 to 0.78 for two detectors and 0.38 to 0.87 for twelve; at 0.80, 0.84 to
0.93 and 0.62 to 0.98. With no skill the best of two stands at about 0.53 and the best of twelve
at 0.57 to 0.61 (`fitside_decay.py`). Events left out lower the power below these.

**Trial.** The exact commands ran on two excerpts of farm A and one of farm B
(`data/report/l62c/trial_corrected.sh`). Farm B's files equal the first run's to the bit; the
departures' fitted years of farm A's two are unchanged, and the hours scored in the scored window of its excerpt with a
fault rose from 0 to 381 of 385. The reading ran over those and the first run's files of farms B
and C with labels shuffled inside each farm, wrote no excerpt's statistic and printed only the
sizes of its tables; its files were deleted unopened.

**Limitations (declared).** Those of 2026-11-22, and:

- The statistic and the third verdict were chosen after the labels were read.
- Over the fitted years a block's level under a model of normal behaviour follows its share of
  unsettled hours at +0.12 to +0.14 beyond the day after a gap: stretches with many stops read
  rougher. That cannot be told from a fault's own trace; the answers with no model stand beside.
- In farm A no answer with no model exists: its status is the label.
- Events left out for want of a settled span are mostly faults; what a detector does there is
  not read.
