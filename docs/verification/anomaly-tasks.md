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
