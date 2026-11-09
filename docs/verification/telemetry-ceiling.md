# The ceiling without labels on the first mission

What the events found at 1 % of normal windows can confirm, which validation events of
`esa-mission1-events` anything finds with no label, and whether the events nothing finds carry a
signal finer than the hour.

## 2026-11-09 — measured: the least gain in the events found at 1 % of normal windows

**Question.** What gain in the share of events found does the rule confirm with power 0.8 on the
first mission? The share has been reported beside the area with no least gain.

**Conditions.**

| | |
|---|---|
| Code | `scripts/detection_power_report.py` at the commit adding this section |
| Machine | MacBook Pro M1 Pro, CPU; no training; validation windows only |
| Rule | gain at least the floor (0.016) and the paired 95 % interval over months above zero; 1,000 resamples |
| Structure | `structure --task esa-mission1-events`: 41,088 windows, 56 months, 62 events, 4,759 event windows |
| Models | `shared` (the area's: a shift per event, the candidate adds to every event); `jittered` (a draw per event on the candidate's lift, so events are also lost); `onset` (the same on each event's first window alone); 1,000 worlds, shifts by bisection over 150 worlds |
| Width | the jitter that matches the observed median half-width of the interval, 0.122, over the seven recall pairs of `telemetry-history-pretext.md`: 1.5 (`jittered`), 4 (`onset`) |
| Real pairs | `injected`: eight pairs from the export of the forecasting pretext's campaigns, `channel_local` against the forecast's and the interpolation's errors, 39,960 windows; missed events chosen at random have their best window raised to a score that found events reach, until the gain is the one asked for; 100 draws a pair |

```
detection_power_report.py jitter --level {0.4,0.6} --jitters 1 1.5 2 2.5 3 4 --worlds 100 --resamples 500 --shift-worlds 150
detection_power_report.py recall --level {0.4,0.6} --models {jittered,onset} --jitter {1.5,4} --gains 0 0.12 0.14 0.16 0.18 0.2 0.22 --minimum 0.016 --worlds 1000 --resamples 1000 --shift-worlds 150
detection_power_report.py injected --pairs pairs.tsv --gains 0 0.08 0.1 0.12 0.14 0.16 0.18 0.2 0.22 --minimum 0.016 --worlds 100 --resamples 1000
```

**Power of the rule at a true gain.** Models at control levels 0.4 / 0.6.

| Gain | ≈ events | Real pairs, mean (range) | `jittered` | `onset` |
|---|---|---|---|---|
| 0 | 0 | 0.001 (0.00–0.01) | 0.02 / 0.02 | 0.03 / 0.03 |
| 0.12 | 7 | 0.16 (0.00–0.78) | 0.45 / 0.45 | 0.55 / 0.47 |
| 0.14 | 9 | 0.59 (0.00–1.00) | 0.59 / 0.57 | 0.67 / 0.56 |
| 0.16 | 10 | 0.74 (0.03–1.00) | 0.67 / 0.72 | 0.77 / 0.67 |
| **0.18** | **11** | **0.82 (0.19–1.00)** | **0.79 / 0.84** | **0.83 / 0.81** |
| 0.20 | 12 | 0.89 (0.38–1.00) | 0.87 / 0.93 | 0.91 / 0.90 |
| 0.22 | 14 | 0.96 (0.77–1.00) | 0.91 / 0.95 | 0.96 / 0.93 |

The models' half-widths are 0.118–0.129, except `onset` at level 0.4, 0.089–0.118, which no jitter
widens to the observed width. The real pairs' are 0.127–0.138. The realised gain at a nominal 0.18
is 0.178–0.186. The area's model gives 0.018 at no gain against the observed 0.122 and is not used.

**Conclusions.**

- The least gain is 0.18, about 11 of the 62 events; the floor's fixed part is 0.016, one event.
- A normal approximation from the observed width, 1.43 × 0.122 = 0.17, agrees.
- At 0.16 no method reaches power 0.8, and at 0.18 all do within their Monte Carlo error (±0.025).
- The rule holds its level: at no gain it confirms at most 0.03 of the worlds.

**Limitations.**

- Power depends on the pair: 0.19 to 1.00 at 0.18 across eight real pairs. For two detectors that
  disagree widely, a gain below the least gain closes nothing.
- The injection raises one window of an event and models neither a detector that gains whole
  classes nor one that trades events.
- The real pairs all share one control, `channel_local`.

## 2026-11-09 — declared before the run: the ceiling without labels and the finer cells

**Question.** (1) Of the first mission's validation events that the channels' departures from
their recent hours miss, how many does any other detector with no label find, beyond chance?
(2) Do the events nothing found stand out at ten minutes, on the channels they are annotated on,
where the hour does not show them?

**What each outcome changes.** (1) opens or keeps deferred a pretext term that discriminates
synthetic perturbations per channel. (2) puts ten-minute cells in the queue or closes them. Both
restate the product claim: a detector with no label above the benchmark's comparators, and the
model as a reader of unknown classes.

**What was seen before this declaration.** No result per event. Seen: the count of events found
by each of the six departure arms, the bins arms among them unpublished; the column of the 11
events for those arms was not opened. The ten-minute departures are therefore not read blind as a
detector, which is why (2) is a different reading, on annotated channels. The recall pairs of the
forecasting pretext are published.

**Conditions.**

| | |
|---|---|
| Code | `scripts/telemetry_ceiling_report.py` at the commit adding this section |
| Machine | MacBook Pro M1 Pro, CPU; answers already on disk, no fit, no platform; validation numbers |
| Windows | every detector's common set: 39,384 of 41,088, less the 1,704 with truncated history; 61 events; event 150 has no window left and is reported aside |
| Later fitting months | 26,124 windows, 29 events, read separately; the 70 events taught on the fitting side are never scored |
| Found | a window above the alarm at 1 % of normal windows; for a detector with repeats, in at least half of them |
| Detector | a candidate under one pretraining; 95 on the validation months, 347 repeats |
| Families | departures: `channel_local` and the six `local` arms (on the later fitting months the twelve `local` arms, no `channel_local`). Other, with no label: `*_knn` without `events=`/`head=`, the forecast, interpolation and linear answers' errors, `*-whitened`, `*-farthest`. Taught: `*-direction`, `*-trees`, `taught-*`, fine-tuning with every event; split by whether the event's class occurs on the fitting side |
| Selection months | the fine-tuned readers from the initialisation only, since a pretraining read those months |
| Chance | 1,000 rotations of the scores within months, one offset per month shared by every detector and repeat, so detectors that agree stay in agreement |
| Fusion | the largest rank over `channel_local`, the cells' farthest feature, the forecast's largest error, the interpolation's and the states' whitened distance, first repeat, read at a combined 1 %, against `channel_local` by the registered rule, 10,000 resamples |
| Finer cells | each of the 11 events on its annotated channels the corpus reads; the departure of the hour and of its ten-minute bins, 6 hours of context, above the alarm at 1 % of that channel's normal windows, and the largest departure over that alarm against the same quantity under 1,000 rotations of the months, shared by the channels (chance of merely passing an alarm grows with an event's windows and channels whatever its signal) |
| Beside | `channel_local`'s recall with its scores read against each month and without; events by benchmark category, locality and whether their class occurs on the fitting side |

```
telemetry_ceiling_report.py measure --side refit --out DIR/refit --rotations 1000
telemetry_ceiling_report.py measure --side select --out DIR/select --rotations 1000
telemetry_ceiling_report.py resolution --out DIR --rotations 1000
telemetry_ceiling_report.py read --dir DIR
```

**Trial, with the targets rotated within months.** The whole path ran: 39,384 windows, 61 events,
95 detectors, 347 repeats; 26,124, 29, 92, 220 on the later fitting months; 11 events. Other
detectors found 13 events against 9.6 expected and 13.5 at the rotations' 95th percentile, so the
rule stayed closed, as it must. The rows of the departures match the raw files: an hourly mean of
each of the 48 channels correlates 0.9996 or more with its own row and less with any other.
Ten published counts of events found were reproduced exactly.

**Rule.** The perturbation pretext reopens only if all three hold: the events found by other
detectors among those the departures miss exceed the rotations' 95th percentile; that count less
its expectation under the rotations is at least 11, the events the least gain of 0.18 asks of one
detector over 61; and the fusion beats `channel_local` alone by the registered rule (gain at least
0.016, interval above zero). The ten-minute cells enter the queue if at least 4 of the 11 events
are found at ten minutes, their largest departure over the alarm beaten by at most 0.05 of the
rotations, and are not found so at the hour, on at least 2 different sets of channels, since 8 of
the 11 touch channels 14, 21 or 29.

**Power.** The first rule cannot open if the departures miss fewer than 11 events: then no
detector can pass the least gain over the departures on these events, and the reason to keep the
term deferred is the instrument's ceiling on this mission, not an absence of signal. The 11 events
were chosen for being unfound, so the second reading describes them and does not generalise.

**Predictions**, from earlier intervals where one exists.

| Quantity | Range | Source |
|---|---|---|
| Events the departures find | 42 to 52 of 61 | the arms' published 37 to 41, and the union of six arms being at most the sum of what they disagree on |
| Excess of the other detectors | 0 to 5 | none directly; the forecast against `channel_local` reads −0.112 to +0.085 in recall, 7 events at the upper edge; the rotations' expectation is about 10 |
| Local events found with no label | 17 to 25 of 33 | 12 to 17 for one arm in the forecasting pretext's note, more for a union |
| Fusion against `channel_local` | −0.05 to +0.10 | that interval's range, a fusion being no worse than its best member by much |
| Of the 11, at the hour on annotated channels | 1 to 5 | the departures found 3 of them across all channels |
| Of the 11, gained at ten minutes | 0 to 3 | a short local event gains from a finer cell, the extreme of more bins gains noise too |

Outcome read as: the perturbation pretext stays deferred, the finer cells are not queued. A count
of events is the measure here, so the levels are the rule's: reopened or not, queued or not.
