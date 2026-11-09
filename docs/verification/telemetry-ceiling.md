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
