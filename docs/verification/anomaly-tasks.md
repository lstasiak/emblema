# The anomaly-detection tasks

Whether a pretrained encoder's frozen features set the annotated events of telemetry apart from
its normal windows better than the same encoder's at its initialisation and than the summaries
of the channels, with no label spent. The task and its measure are
[ADR-0067](../adr/0067-a-task-of-events-read-without-a-label.md); the raw satellite reader
extends [ADR-0025](../adr/0025-reading-the-satellite-telemetry.md).

## 2026-10-27 — declared before the run: the instrument, its power, and the first reading

**Questions.** (1) Does the detection campaign run end to end over each of the four tasks —
the two satellite missions, the testbed and the OPS-SAT fragments — from a declared file to a
verdict, with the three detectors answering every validation window and nothing refused?
(2) What gain in event-weighted area does the registered rule confirm with power 0.8 on each
task's validation side, given the events it holds? (3) Under the backbones pretrained over cells
on the intensive-care stays, how far do the frozen encoder's states lie from the same encoder at
its initialisation and from the summaries, on corpora none of them ever read?

**Conditions.**

- *Commit*: the one that adds this section. *Machine*: M1 Pro; the encoders' states on MPS,
  fp32, the distances and the summaries on the CPU; tier M's backbones.
- *Corpora*: published to the remote bucket continuing the vocabulary of the 2019 stays
  (`bf7de2fa…`), in the order the next screen's composition needs (`data/report/l33/chain.txt`):
  the eICU demo, then the raw satellite telemetry (`esa_ad_raw`: 48 target channels of
  Mission1 from three packets and one group, the eleven lightweight channels of Mission2, every
  reading at its own instant, windows of one hour every hour, the months of each mission's
  second half held out), SKAB (windows of 100 s every 10 s, the 34 experiments held out, the
  anomaly-free recording the training side) and OPSSAT-AD (one window of 1,340 s per fragment,
  the authors' test part held out). The Water Quality Portal follows with the next screen.
- *Tasks*: `esa-mission1-events`, `esa-mission2-events`, `skab-anomalies`,
  `opssat-anomalies`, as the register's section of 2026-10-27 states them: the corpus's training
  side is the fitting side, one held-out unit in three is frozen by seed 1, the rest validate.
- *Campaign*: `campaigns/detection-events.toml` under each backbone of
  `data/report/l33/backbones.txt` — the base recipe of the cell screen at two seeds
  (`e9bf831a…`, `b23be234…`) as the screen, the tail recipe at two (`0b2a37e7…`, `4751c1aa…`)
  beside — `untrained_knn` the control, `frozen_knn` the endpoint, `summaries_knn` beside;
  five neighbours, the alarm at the fitting side's 2 %; seeds 1 to 3. Defined and ordered by
  `data/report/l33/define.sh`, run by `run.sh`, read by `scripts/campaign_pairs_report.py`.
- *Power*: `scripts/detection_power_report.py`, the validation structure of each task from the
  raw annotations (`data/report/l33/power/<task>/structure.csv`), 100 worlds of scores per
  level and gain, 500 resamples per world, the least gain tried 0.02, so that the table says
  where the rule fails as well as where it holds.

**The validation sides** (from the raw annotations, under the registered split):

| Task | Units | Windows | Events | Positive windows | Chance level |
|---|---|---|---|---|---|
| `esa-mission1-events` | 56 months | 41,088 | 62 (38 anomalies) | 4,759 | 0.002 |
| `esa-mission2-events` | 14 months | 10,224 | 236 (1 anomaly) | 475 | 0.024 |
| `skab-anomalies` | 22 experiments | 2,191 | 22 | 1,060 | 0.019 |
| `opssat-anomalies` | 352 fragments | 352 | 83 | 83 | 0.236 |

**Predictions, from earlier intervals.**

- Mechanics: every cell answers; the frozen and summary detectors repeat exactly over seeds and
  the control varies with its draw of weights (`encoder-candidates.md`: a fresh encoder's
  spread is the seed's).
- Zero-shot without the channels' identity, the pretext sits on the prior: −0.12 to −0.21
  against the baselines on every corpus the backbone never read (`zero-shot-pretext.md`).
  The rows of every channel of these corpora are drawn at random under both encoder arms, so
  the frozen encoder's gain over its initialisation is predicted **within ±T on every task**,
  the machinery alone moving it; the reading that bears on the claim comes with the next
  screen, whose base pretrains on the first half of the satellites' months.
- The summaries against the encoders: on the stays the summaries' head leads the frozen probe
  by 0.05 to 0.09 in area (`head-and-representation.md`); here the summaries are predicted to
  lead both encoder arms by more than T on the satellites and the testbed. On the fragments
  the benchmark's best unsupervised detector over handcrafted features reads an area of 0.78
  (the universal detector of Ruszczak et al. 2025); `summaries_knn` is predicted within
  [0.4; 0.8], the
  wide lower edge for a detector of five neighbours against the benchmark's tuned ones.

**Thresholds (T) and levels of outcome.** T is the smallest gain the rule confirms with power
0.8 in the simulation at the lower level, whatever the task: **0.05 in event-weighted area**
(the table below: at a gain of 0.02 the rule confirms a fifth of the worlds or fewer at the
lower level on every task, at 0.05 at least 0.99 of them). The least gain that counts is
therefore 0.05
on each of the four tasks and the floor's fixed part 0.01, registered from this section before
any reading; the campaign file states them. A gain within ±0.05 is "less than about 0.05", not
"none". Levels: a gain at or above 0.05 is real and at the level of the summaries' lead over
the probe on the stays, the one earlier reading of what a representation is worth here; a loss
beyond 0.05 is harm. None of it is read against the claim on this first use.

**Power** (share of 100 worlds the rule confirms; level = the control's expected area):

| Task | Level | Gain 0.02 | Gain 0.05 | Gain 0.10 |
|---|---|---|---|---|
| `opssat-anomalies` | 0.3 | 0.03 | 1.00 | 1.00 |
| `opssat-anomalies` | 0.5 | 0.90 | 1.00 | 1.00 |
| `skab-anomalies` | 0.1 | 0.13 | 0.99 | 1.00 |
| `skab-anomalies` | 0.3 | 0.99 | 1.00 | 1.00 |
| `esa-mission2-events` | 0.1 | 0.15 | 1.00 | 1.00 |
| `esa-mission2-events` | 0.3 | 1.00 | 1.00 | 1.00 |
| `esa-mission1-events` | 0.1 | 0.00 | 1.00 | 1.00 |
| `esa-mission1-events` | 0.3 | 0.18 | 1.00 | 1.00 |

**Limitations declared.** The simulation's scores are a model, not a detector's: every normal
window a standard normal draw, every event's windows that draw plus a shift shared by the event,
both sides on one noise, so the power is the sampling of units and events alone. The months of
a mission's half are whole months, so the split differs from the benchmark's exact half by less
than a month on either mission. The testbed's fitting side was recorded a month before most of
its experiments, so drift raises false alarms under every arm alike. The benchmark's timing and
range metrics are not computed at a window's resolution. On the fragments the frozen window of
1,340 s holds a fragment of 0.6 to 22 minutes in its beginning, so a short fragment's cells are
mostly empty and its duration is part of what every detector reads. The alarm's share of 2 %
is one for every task, so on the fragments, whose fitting side is a fifth anomalous by the
authors' count, the readings beside the area are taken at an alarm the benchmark would not
set; the area itself is alarm-free. A run of flagged windows that crosses a window the task
does not score, or an hour with no reading, counts as two false alarms in the event-wise score.

## 2026-10-28 — measured: the first reading under the cell screen's backbones

**Question.** Under the four backbones of the cell screen, does the nearest-neighbour detector
over the frozen encoder's states set the annotated events apart better than the same detector
over the encoder at its initialisation, and than over the channels' summaries, on four tasks
none of the backbones read? On this first use the reading is of the instrument, not of the claim.

**Conditions.**

- *Commit*: `46ebac47` (every order carries it; `campaign_run` refused any other tree).
  *Machine*: M1 Pro; the encoders on MPS, fp32; the distances and the summaries on the CPU;
  tier M backbones. Validation side only; preliminary.
- *Campaigns*: 16, one per task and backbone (`base1`, `base2`, `tail1`, `tail2`, as declared),
  each three detectors × three seeds at every window, defined by `data/report/l33/define.sh`, run
  by `run.sh` under `caffeinate` (2026-10-28 00:40 to 05:09; per backbone M1 about 55 min, M2
  6 min, SKAB and OPS-SAT under a minute; no cell retried), read by
  `SET=all bash data/report/l33/read.sh` (`scripts/campaign_report.py` per campaign,
  `scripts/campaign_pairs_report.py --resamples 10000 --seed 1 --level 0.95 --floor-area 0.01`
  over the two backbones of a recipe pooled), and `data/report/l33/beside.py all` for the
  readings beside the area.
- *Encoder reading*: every block side by side, mean over the window, as the campaign declares.
  A first run read the last block alone through a wiring error; it was stopped, kept as a
  diagnostic and is not reported.
- *Validation sides as scored*: M1 41,088 windows, 4,759 positive, 62 events; M2 10,223 (the
  mission's last hour has no reading), 475, 236; SKAB 2,350, 1,091, 22; OPS-SAT 352, 83, 83.

**Event-weighted PR-AUC**, mean over the three seeds (base1 / base2 / tail1 / tail2; the
untrained encoder and the summaries do not depend on the pretrained weights):

| Task | Chance | Untrained encoder (control) | Frozen encoder (endpoint) | Summaries |
|---|---|---|---|---|
| `esa-mission1-events` | 0.002 | 0.032 | 0.058 / 0.072 / 0.058 / 0.067 | 0.021 |
| `esa-mission2-events` | 0.024 | 0.488 | 0.527 / 0.593 / 0.414 / 0.307 | 0.532 |
| `skab-anomalies` | 0.017 | 0.024 | 0.024 / 0.022 / 0.023 / 0.022 | 0.212 |
| `opssat-anomalies` | 0.236 | 0.482 | 0.420 / 0.406 / 0.420 / 0.414 | 0.396 |

**Per-campaign verdict** on the endpoint (frozen against untrained; least gain 0.05):

| Task | base1 | base2 | tail1 | tail2 |
|---|---|---|---|---|
| `esa-mission1-events` | +0.026, distinguishable, below 0.05 | +0.040, distinguishable, below 0.05 | +0.026, distinguishable, below 0.05 | +0.035, distinguishable, below 0.05 |
| `esa-mission2-events` | +0.039, indistinguishable | +0.105, indistinguishable | −0.074, indistinguishable | −0.182, indistinguishable |
| `skab-anomalies` | −0.001, indistinguishable | −0.002, indistinguishable | −0.001, indistinguishable | −0.002, indistinguishable |
| `opssat-anomalies` | −0.062, worse | −0.076, worse | −0.062, worse | −0.067, worse |

**The declared pairs**, two backbones of a recipe pooled, gain in area with the 95 % interval
(the floor 0.010 to 0.011, on M2 0.148 where the control's spread over seeds enters it):

| Task | Recipe | Frozen − untrained | Summaries − untrained | Frozen − summaries |
|---|---|---|---|---|
| `esa-mission1-events` | base | +0.033 [+0.005; +0.090] | −0.011 [−0.037; +0.006] | +0.044 [+0.008; +0.115] |
| | tail | +0.030 [+0.004; +0.086] | −0.011 [−0.037; +0.006] | +0.041 [+0.006; +0.113] |
| `esa-mission2-events` | base | +0.072 [−0.068; +0.152] | +0.044 [−0.162; +0.145] | +0.028 [−0.039; +0.141] |
| | tail | −0.128 [−0.310; +0.073] | +0.044 [−0.162; +0.145] | −0.172 [−0.267; +0.016] |
| `skab-anomalies` | base | −0.001 [−0.007; +0.026] | **+0.188** [+0.057; +0.340] | **−0.189** [−0.335; −0.050] |
| | tail | −0.001 [−0.006; +0.023] | **+0.188** [+0.057; +0.340] | **−0.189** [−0.336; −0.052] |
| `opssat-anomalies` | base | **−0.069** [−0.110; −0.026] | **−0.086** [−0.160; −0.002] | +0.017 [−0.063; +0.094] |
| | tail | **−0.065** [−0.101; −0.027] | **−0.086** [−0.160; −0.002] | +0.021 [−0.058; +0.096] |

Bold: past the least gain of 0.05 with the interval clear of zero.

**Readings beside the area**, mean over seeds and the four backbones (the frozen encoder's
range over backbones in brackets): share of events found with at most 1 % of normal windows
flagged / corrected event-wise F0.5 at the detector's alarm.

| Task | Untrained encoder | Frozen encoder | Summaries |
|---|---|---|---|
| `esa-mission1-events` | 0.19 / 0.007 | 0.20 (0.18–0.22) / 0.005 | 0.13 / 0.018 |
| `esa-mission2-events` | 0.68 / 0.83 | 0.63 (0.40–0.82) / 0.70 | 0.72 / 0.84 |
| `skab-anomalies` | 0.21 / 0.057 | 0.20 (0.18–0.23) / 0.044 | 0.45 / 0.040 |
| `opssat-anomalies` | 0.16 / 0.41 | 0.04 (0.02–0.06) / 0.27 | 0.01 / 0.20 |

**Predictions against the reading.**

| Declared | Read | Held |
|---|---|---|
| Every cell answers | 144 of 144 cells, no refusal, no retry | yes |
| Frozen and summary detectors repeat over seeds | summaries exactly; frozen exactly on SKAB only, elsewhere a spread up to 0.03 (M1, OPS-SAT) and 0.36 (M2) | in part |
| Frozen − untrained within ±0.05 on every task | M1 and SKAB yes; OPS-SAT −0.065 to −0.069; M2 −0.18 to +0.11, indistinguishable | in part |
| Summaries lead both encoder arms by more than 0.05 on the satellites and the testbed | SKAB by 0.19; M1 0.01 to 0.05 behind; M2 within the noise | SKAB only |
| `summaries_knn` on OPS-SAT within [0.4; 0.8] | 0.396 | no, by 0.004 |

**Conclusions.**

1. The instrument runs end to end on all four tasks under every backbone: 16 campaigns, 144
   cells, verdicts out of the registry. The two arms that do not depend on the pretrained
   weights read the same area in all four campaigns of a task.
2. No task reaches the registered gain over the initialisation. On M1 the frozen encoder gains
   0.026 to 0.040 under every backbone, distinguishable, below 0.05. On OPS-SAT it loses 0.06 to
   0.08 under every backbone. On M2 the control's spread over seeds (0.31 to 0.64) leaves every
   comparison indistinguishable. On SKAB it moves nothing.
3. The frozen arm varies over seeds where the vocabulary grows. SKAB's channels have rows in
   the checkpoint (identifiers 127–134 of the vocabulary the backbones were built over, never
   trained, since the backbones read only the 2012 and 2019 stays). The raw satellite channels
   (377–435) and the fragments' channels (436–444) get rows drawn per seed. The declared prediction overlooked this; the declaration's
   own statement that the rows are random implies it.
4. The summaries lead only on SKAB, and there both encoder arms read near chance (0.022 to
   0.024 against 0.017). Within one experiment, the encoder arms score its event windows above its
   normal windows in 15 to 18 of 22 experiments, the summaries in all 22 (base campaigns, every
   seed). Across experiments the shift from the fitting recording dominates: 93 to 98 % of all
   windows lie above the alarm under every arm, so the pooled ranking of the encoder arms sits
   near chance. M1 shifts the same way between the mission's halves: the alarm set on the first
   half flags 92 to 97 % of the second half's normal windows under the encoder arms and 64 %
   under the summaries (base1, seed 1, `flagged.py` in the run's directory), which is why its
   areas stay low; on M2 and OPS-SAT it flags 0 to 3 %.
5. On OPS-SAT a count beats every detector. Scoring each validation fragment by its number of
   readings alone, with no model, reads 0.508, above the untrained encoder (0.482), the frozen
   one (0.41 to 0.42) and the summaries (0.396); its duration reads 0.418 (a diagnostic after
   the run, `opssat_shortcut.py` in the run's directory; the best of nine of the authors'
   features, chosen on this side, reads 0.63). Under the window of 1,340 s that holds a whole
   fragment, the task rewards how much of the window is filled, and the encoder arms read
   mostly that.
6. Nothing here is read against the claim. Every channel row of these corpora is untrained
   under both arms, so the comparison measures the encoder's machinery over random identities.
   The reading that bears on the claim needs backbones pretrained on the satellites' first
   halves (the next screen).

**Limitations.**

- Validation side, tier M backbones on one machine; the test sides are untouched.
- The power simulation used the testbed's structure from the raw annotations (2,191 windows,
  1,060 positive); the published corpus scores 2,350 and 1,091: five or six more windows at the
  end of every experiment, 11 to 16 in four. The structure was cut from the annotations, not
  from the published block, and the simulation was not rerun at the scored counts; the other
  three tasks match it to the window (M2 one fewer).
- The power simulation modelled the sampling of units and events, not the spread of a drawn
  encoder over seeds. On M2 that spread (0.31 to 0.64 for the control) sets the floor at 0.148,
  three times the least gain. The floor is the control's standard deviation over its repeats,
  not its standard error, so more seeds do not lower it: M2 resolves no gain below about 0.15
  while a drawn encoder spreads so.
- On M2 one anomaly sits among 236 events, so its PR-AUC over anomalies alone (about 0.00 under
  every arm) says nothing; on M1 it reads 0.023 to 0.035 under every arm.
- On M2 the 18-hour channel never has two readings in an hour, so some of its summaries are
  undefined on the whole fitting side and are held constant.
- The OPS-SAT fitting side carries the authors' anomalous fragments (about a fifth), so the
  detectors fit on contaminated data, as the benchmark's unsupervised ones do. The benchmark's
  0.78 comes from tuned detectors over handcrafted features and is not reproduced here.

## 2026-10-28 — declared before the run: each unit as its own reference, and the counts of readings

**Questions.** (1) On the first mission and the testbed, where the alarm set on the fitting side
flagged 64 to 98 % of the normal windows, does reading each distance against its own unit (its
month, its experiment) let the events stand out, and does the frozen encoder then gain over its
initialisation? (2) On every task, how much of what the detectors find does the sampling alone
tell, read by a detector over the counts of readings per channel?

**Conditions.**

- *Commit*: the one that adds this section. *Machine*, tier, backbones, tasks and the fitting
  and validation sides as in the measured section above.
- *Campaigns* (register, 2026-10-28): `campaigns/detection-events-per-unit.toml` over
  `esa-mission1-events` (under `base1` and `base2` only: its cells take six minutes each) and
  `skab-anomalies` (all four backbones); `campaigns/detection-events-counts.toml` over
  `esa-mission2-events` and `opssat-anomalies` (all four). Seeds 1 to 3, the least gain 0.05 and
  the floor's fixed part 0.01 unchanged. Defined and ordered by `data/report/l33/second/define.sh`
  from `plan.txt`, run by `run.sh`, read by `read.sh` and `beside.py`.
- *Reference per unit*: a window's distance less the median of its unit's distances, over their
  median absolute deviation times 1.4826 (one where it is zero); the alarm is the 98th
  percentile of the fitting windows read so within their own units. No label is read. A unit's
  windows are read together, so the detection is retrospective within a unit.
- *Counts*: `counts_knn` reads one number per channel, the readings the window holds, with the
  same five neighbours and alarm.
- *Trial*: one campaign of each file (`base1-skab-anomalies`, `base1-opssat-anomalies`), run and
  accepted before this section was committed (15 and 12 cells, 71 s and 49 s), its measures not
  read.

**Predictions, from the first reading's intervals.**

| | Predicted | Source |
|---|---|---|
| Mechanics | Every cell answers. On M2 and OPS-SAT the three detectors of the first reading return its areas to the third decimal (untrained 0.488 / 0.482, summaries 0.532 / 0.396). | first reading |
| Alarm, M1 and SKAB | Normal windows above the alarm fall from 0.64–0.98 to at most 0.10 under every arm | the construction: every unit read as the fitting units were |
| SKAB, frozen − untrained | Within ±0.05 | first reading −0.001 [−0.007; +0.026]; rows of SKAB's channels untrained in both arms |
| SKAB, summaries − both encoder arms | More than +0.05 | within one experiment the summaries ranked the event above the normal windows in 22 of 22 experiments, the encoder arms in 15 to 18 |
| M1, frozen − untrained | +0.005 to +0.09, below 0.05 confirmed | first reading +0.033 [+0.005; +0.090] |
| OPS-SAT, counts | `counts_knn` within [0.40; 0.60], within 0.05 of the untrained encoder (0.482) or above it | a fragment's count of readings alone read 0.508, its duration 0.418 |
| M1, M2, counts | No prediction; descriptive | no earlier reading |

**Levels and what each outcome would mean.** As registered: a gain at or above 0.05 is real; a
loss beyond 0.05 is harm. If the per-unit reference brings the alarm to the predicted share and
lifts the areas of M1 and SKAB, it becomes the reading of those tasks in the next screen; if
not, drift is not what holds them down and the next screen reads them as first registered. If
`counts_knn` matches or beats the encoder arms on a task, the task's events are told by the
sampling, and a representation is held to that bar there.

**Limitations declared.** A unit is read whole, so an event that fills most of its month (some
of the first mission's last up to forty days) is read against itself and can vanish. The
testbed's events cover about 46 % of each experiment: the order within an experiment is kept,
but its event windows are read over a spread their own share widens, about two spreads above
the median where an experiment with few events would put them nine, so pooled with the other
experiments they lose standing, and an experiment more than half event would read its events
as the norm. A unit without a spread of its own, a lone window or windows alike, is read over
the fitting side's spread; units none of which holds two windows are refused. The per-unit areas of the encoder and summary arms could be
computed from the first reading's export by the same arithmetic; they are not, before this
section is committed, and after the run they are recomputed so as a check that the code reads
what the declaration says. M2 resolves no gain below its floor of about 0.15 (above).
