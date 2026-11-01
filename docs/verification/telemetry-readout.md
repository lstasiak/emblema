# The telemetry leg read under its instruments

The first mission's events read by the detectors the leg was reformulated with: the whitened
distance over the encoder's states (ADR-0069), the pretext's error per channel and for the
window's tail (ADR-0070, ADR-0071), the bars of the field over the cells themselves (ADR-0072),
and detectors taught a few of the fitting side's events (ADR-0073). The screen runs on the two
backbones the hypotheses came from, so it confirms nothing: it selects what the confirmation on
fresh pretrainings reads.

## 2026-11-01 — declared before the run: the first mission under four instruments, and a grid of two encoders by three counts of taught events

**Questions.** (1) Under the whitened distance read against each month, do the pretrained
encoder's states beat their initialisation, and every bar of the field under the same
instrument? (2) Does the pretext's error read per channel, or for the window's tail, beat a
ridge over the same visible cells under the same knobs, and the raw pretext's error? (3) Taught
three, five or ten of the fitting side's events, does the pretrained encoder gain more from the
teaching than its initialisation does (the interaction), does teaching lift it above its untaught
self, and where does it stand against trees taught the same events over the summaries? Beside,
descriptively: the nearest-neighbour instrument on the same cells, so the change of instrument
is measured; the combination of the whitened distance and the events' direction; the cells
taught; the second mission under every zero-label detector; the scored events split by whether
their class was among the taught.

**What this screen can say.** The backbones are `base1` and `base2`, the two the hypotheses of
ADR-0069 to ADR-0073 were read off, so no reading here confirms a claim. The screen selects, by
the rule below, which readouts the confirmation on eight fresh pretrainings reads, and the
prediction of that confirmation is taken from the lower edge of the interval found here.

**Conditions.**

| | |
| --- | --- |
| Code | `98d19b5f`, which declares the campaigns' files and from which the trial ran; this section and the register's rows follow it |
| Tasks | `esa-mission1-events` `8a6b1c18…` (every fitting window, 61,363; 41,088 scored over 62 events), `esa-mission2-events` `09368ab9…`; tier M |
| Backbones | `base1` `0a4dc540…`, `base2` `48175bbf…` (`composition-screen.md`) |
| Campaigns, per backbone | `telemetry-mission1-taught` (the states under the whitened distance and the nearest neighbours, untaught and taught 3, 5 and 10 events, the combination at 5; seeds 1–10), `telemetry-mission1-pretext` (the pretext's error raw, per channel, for the tail and for the tail per channel, each with its ridge; seeds 1–3), `telemetry-mission2` (the states and the pretext's error as above, against the fitting side; seeds 1–3) |
| Campaigns, once | `telemetry-mission1-bars` (the summaries under both distances, the counts, the cells under five distances, the trees taught 3, 5 and 10 events; seeds 1–10), `telemetry-mission2-bars` (seeds 1–3) |
| Platform | Colab G4, fp32, lanes under CUDA MPS; the trial on the M1 Pro, MPS, fp32, one point per family |
| Worker | `data/report/t42c/worker-env.sh`, the backbone's weights per campaign |
| Reading | `data/report/l35/screen/read.sh`: `campaign_pairs_report.py` (pairs and interactions, 10,000 resamples, seed 1), `taught_class_report.py` |

**Rules.**

- *Measure.* The register's event-weighted PR-AUC; the pairs by the registered rule (least gain
  0.05, floor 0.01, interval over months in two strata); the events found at 1 % false alarms
  and the benchmark's corrected event-wise F0.5 beside, the latter descriptive for the taught
  detectors, whose alarm is set over scores of rows the fit saw.
- *Interaction.* The taught pair's reduction less the untaught pair's under one draw of the
  months, paired by seed; its least gain 0.03, from the power simulated over the first
  mission's validation events (`detection_power_report.py interaction`, below); the three counts
  are one family under Holm.
- *Selection for the confirmation.* A readout goes to the confirmation when its gain over its
  control has an interval, over the mean of both backbones, with its lower edge above zero, and
  a point estimate above zero under each backbone alone, and it beats its bar under both: for
  the states, the initialisation under the same instrument and every bar of the field under it;
  for the pretext's error, the ridge under the same knobs and the raw pretext's error; for a
  taught count, the interaction with the pretrained encoder lifted above its untaught self and
  above the trees taught the same events. Every readout that qualifies goes, as a declared family
  under Holm; where none does, the confirmation reads the leg descriptively and the leg turns to
  the product reading, the best detector against the trees and the bars.
- *Descriptive.* The second mission; the combination; the cells taught; the trees against each
  month; the class split; the nearest-neighbour cells of the taught campaign beside the whitened.

**Predictions.**

| Reading | Predicted | Source |
| --- | --- | --- |
| M1 states, whitened, less initialisation | within ±0.03 | `anomaly-signal-in-states.md`: every scorer lifts every encoder alike on M1 |
| M1 states, whitened, PR-AUC | in [0.04, 0.15] | no earlier reading of the whitened distance against each month; the global whitened distance read 0.02–0.03, the nearest neighbours against each month 0.08–0.11 |
| M1 states, whitened, less summaries whitened | in [+0.01, +0.16], if the whitening lifts both alike | `composition-screen.md`: +0.074 [+0.011; +0.158] under the nearest neighbours |
| M1 pretext per channel less ridge per channel | in [−0.05, +0.02] | raw: −0.026 [−0.050; +0.019] |
| M1 pretext for the tail | no prediction | no earlier reading on M1 |
| M1 taught 3, each encoder, less untaught | below zero | heterogeneous events; one class holds 28 of the fitting half's 99 and appears once among the scored |
| M1 taught 10, each encoder, less untaught | above zero | the direction collects the signal spread over 7–9 components |
| M1 interaction at every count | in [−0.03, +0.05] | no earlier reading; the mechanism of the geometry note |
| M1 trees taught 10 less summaries untaught | at least +0.05 | ADBench: tree ensembles with 1 % labels match dedicated methods |
| M1 trees against taught states | no prediction | the summaries read 0.02 untaught on M1 |
| M2 states, whitened, less initialisation | in [−0.55, −0.28] | `anomaly-signal-in-states.md`: 0.12–0.33 against 0.60–0.67 |

**Power of the interaction's rule** (share of 60 worlds the rule confirms at a least gain of
0.02 and the interval above zero; both sides lifted by 0.05 by the teaching, the candidate by the
gain more; one noise, one draw of the months, 500 resamples; `detection_power_report.py
interaction`, `data/report/l35/power/`). The least gain of the interaction is 0.03, the smallest
simulated gain the rule confirms with power 0.8 at the lower level; the floor's fixed part 0.01.

| Level | Interaction 0.02 | 0.03 | 0.04 | 0.05 | 0.10 |
| --- | --- | --- | --- | --- | --- |
| 0.1 | 0.05 | 0.93 | 0.98 | 1.00 | 1.00 |
| 0.2 | 0.53 | — | — | 1.00 | 1.00 |

The model shares one noise across the four sides, so it says nothing of a detector's own seed:
the real interaction carries the draw of the events and of the initialisation too, and its
interval over months will be wider than the simulation's.

**Trial.** The five campaign files, each cut to one point per family under seed 1 and `base1`,
defined, ordered, run and accepted through the campaign's own commands on the M1 Pro, over the
second mission, since the first mission's costs were measured on real cells before (ADR-0070 to
ADR-0073): 18 cells in about 30 minutes from `98d19b5f`. The reading plan ran whole on them, to
files left unopened: 16 pairs, one interaction and four class rows. Its first two runs failed
and were fixed before this section, on a spread over one seed and on candidates whose names hold
commas; no number was read.

**Limitations declared.** Two pretrainings, the ones the hypotheses came from. One seed draws
both the events taught and the untrained encoder's initialisation, so the paired difference
between encoders mixes the two draws. The power simulation models the sampling of months and
events, not a detector's seed. Validation side, tier M.
