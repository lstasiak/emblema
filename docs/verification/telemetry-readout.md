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

## 2026-11-02 — measured: no readout qualifies; under every instrument the pretrained states match their initialisation, the pretext's error trails a ridge, and a few events teach either encoder alike

**Conditions.**

| | |
| --- | --- |
| Code | campaigns declared at `98d19b5f`; cells run from `06406c8d`, which only gives the ml processes scikit-learn |
| Platform | Colab G4, fp32, 24 lanes under CUDA MPS; 152 pieces, every one exit 0; 35 lane-hours; a cell 3 min (states) to 27 min (ridge per channel) |
| Registry | 514 cells accepted; a first-mission campaign of ten seeds closes with a verdict of eleven pairs, 2.5 h in pure Python |
| Reading | `read.sh` as declared, each pair in a process of its own over the predictions of its sides (on the trial's campaigns byte for byte the sequential tables); 10,000 resamples, seed 1 |
| Units | first mission: 56 months, 41,088 windows, 4,759 in 62 events (20 events of four windows or fewer, one of 571); chance under event weights 0.002; second mission 14 units, 10,223 windows |

**Family A: the states under the whitened distance, first mission.** Gain = the candidate's
event-weighted PR-AUC less the control's, mean of both backbones; interval over months.

| Pair | Control | Candidate | Gain [interval] | p |
| --- | --- | --- | --- | --- |
| pretrained less initialisation, whitened | 0.080 | 0.090 (base1 0.100, base2 0.080) | +0.009 [−0.009; +0.026] | 0.38 |
| pretrained less initialisation, nearest neighbours | 0.089 | 0.096 (0.086, 0.105) | +0.007 [−0.010; +0.024] | 0.66 |
| whitened less nearest neighbours, pretrained | 0.096 | 0.090 | −0.006 [−0.024; +0.011] | 0.90 |
| whitened less nearest neighbours, initialisation | 0.089 | 0.080 | −0.008 [−0.035; +0.020] | 0.69 |

The bars of the field under the same reference, and the pretrained whitened states against each:

| Bar | Area | States less bar [interval] | p |
| --- | --- | --- | --- |
| summaries, whitened | 0.018 | +0.072 [+0.015; +0.156] | 0.010 |
| summaries, nearest neighbours | 0.021 | +0.069 [+0.011; +0.153] | 0.019 |
| counts | 0.006 | +0.084 [+0.022; +0.174] | 0.004 |
| cells, nearest neighbours | 0.029 | +0.060 [+0.006; +0.135] | 0.028 |
| cells, whitened | 0.017 | +0.073 [+0.016; +0.157] | 0.007 |
| cells, centre | 0.029 | +0.061 [+0.004; +0.131] | 0.020 |
| cells, subspace | 0.056 | +0.034 [+0.003; +0.077] | 0.024 |
| **cells, farthest** | **0.154** | −0.065 [−0.172; +0.036] | 0.20 |

**Family B: the pretext's error, first mission** (three seeds; the ridge does not read the
backbone, so its cells are the same under both).

| Knobs | Pretext (base1, base2) | Ridge | Pretext less ridge [interval] | Pretext less raw pretext |
| --- | --- | --- | --- | --- |
| raw | 0.127 (0.114, 0.139) | 0.152 | −0.026 [−0.050; +0.019] | — |
| per channel | 0.021 (0.021, 0.021) | 0.142 | −0.121 [−0.194; −0.058] | −0.106 [−0.193; −0.049] |
| tail | 0.060 (0.058, 0.062) | 0.057 | +0.003 [−0.001; +0.024] | −0.067 [−0.158; −0.005] |
| per channel and tail | 0.017 | 0.010 | +0.007 [+0.001; +0.026] | −0.110 [−0.198; −0.050] |

Per channel the raw pretext's 37 events found at 1 % false alarms fall to 16 (the ridge 29):
the short events caught at 0.2 % of the normal windows fall to 1.2 %, as the floor on a quiet
channel's scale lifts that channel's ordinary error over them. One code path serves both
detectors; the knob, not a fault, does it.

**Family D: a few of the fitting side's events taught, first mission** (ten seeds, each
drawing the events and the initialisation; sd over seeds 0.05 to 0.09).

| Events | Pretrained taught less untaught | Initialisation taught less untaught | Interaction [interval], per seed | Pretrained less initialisation, taught |
| --- | --- | --- | --- | --- |
| 3 | +0.012 [−0.054; +0.066] | +0.029 [−0.034; +0.076] | −0.017 [−0.039; +0.006], −0.017 ± 0.015 | −0.008 [−0.029; +0.011] |
| 5 | +0.035 [−0.043; +0.103] | +0.048 [−0.020; +0.109] | −0.013 [−0.044; +0.017], −0.013 ± 0.008 | −0.003 [−0.034; +0.023] |
| 10 | +0.041 [−0.036; +0.112] | +0.032 [−0.036; +0.098] | +0.010 [−0.023; +0.036], +0.010 ± 0.013 | +0.019 [−0.016; +0.043] |

Per backbone at five events: base1 0.170 against 0.100 untaught, base2 0.080 against 0.080.
The combination of the whitened distance and the direction reads −0.035 [−0.104; +0.038]
against the direction alone. By class, at ten events the scored events of a taught class gain
+0.08 to +0.11 under either encoder and the other classes lose −0.06 to −0.08. The trees
taught the same events over the summaries read 0.002, 0.008 and 0.010 at 3, 5 and 10 events,
below the untaught summaries (0.021); the taught states beat them by +0.10 to +0.12 (lower
edges +0.06 to +0.07). Refitted on a real cell, the trees separate the taught rows in sample at
AUROC 0.999 and the scored events at 0.56 and 0.46: they learn the taught events, which the
scored side's classes do not repeat.

**Second mission, descriptive** (three seeds; areas, mean of both backbones unless given).

| Detector | Area |
| --- | --- |
| pretrained states whitened / nearest neighbours | 0.224 (0.123, 0.325) / 0.056; less initialisation −0.412 [−0.495; −0.273] / −0.432 [−0.594; −0.312] |
| initialisation whitened / nearest neighbours | 0.636 / 0.488 |
| pretext raw / per channel / tail / both | 0.489 / 0.685 / 0.314 / 0.426; ridge 0.496 / 0.722 / 0.206 / 0.403 |
| summaries plain / whitened; cells plain / whitened / farthest; counts | 0.532 / 0.622; 0.427 / 0.593 / 0.618; 0.024 |

**Beside** (first mission, events found at 1 % false alarms / the benchmark's F0.5 at the
cell's alarm): pretrained states whitened 0.16 / 0.64, their initialisation 0.20 / 0.62;
taught five 0.55 and 0.49; pretext raw 0.53 / 0.22, ridge raw 0.52 / 0.30; cells farthest
0.42 / 0.68; summaries 0.08 / 0.22; trees taught ten 0.14.

**Predictions against the reading.**

| Reading | Predicted | Measured |
| --- | --- | --- |
| M1 states whitened less initialisation; their PR-AUC; less summaries whitened | ±0.03; [0.04, 0.15]; [+0.01, +0.16] | +0.009; 0.090; +0.072 (all held) |
| M1 pretext per channel less ridge per channel | [−0.05, +0.02] | −0.121 (failed) |
| M1 taught 3 / 10 less untaught | below / above zero | +0.012 and +0.029 / +0.041 and +0.032 (3 failed within noise; 10 held at the point) |
| M1 interaction, every count | [−0.03, +0.05] | −0.017, −0.013, +0.010 (held) |
| M1 trees taught 10 less summaries untaught | at least +0.05 | −0.013 (failed) |
| M2 states whitened less initialisation | [−0.55, −0.28] | −0.412 (held) |

**Selection.** No readout qualifies: the states' gain over their initialisation has its lower
edge below zero and a point of −0.001 under `base2`; no pretext knob beats both the ridge and
the raw error; no interaction reaches 0.03 and none has its lower edge above zero. By the
declared rule the confirmation reads the leg descriptively, and the leg turns to the product
reading: the best detector against the trees and the bars.

**Conclusions.**

1. Whitened or not, taught or not, the pretrained states read as their initialisation does on
   the first mission and far below it on the second: the third screen to find it, under a third
   instrument, and the same under PR-AUC, recall at 1 % and F0.5.
2. The pretext's error does not read past a ridge over the same cells. Per channel it loses
   the short events on the first mission and gains on the second; the tail loses on both.
3. A few taught events lift either encoder by 0.03 to 0.05, on the taught classes alone; the
   pretrained encoder learns no more from them than its initialisation. The trees learn the
   taught events and nothing beyond.
4. Product reading, zero label: on the first mission the cells' farthest neighbour (0.154) and
   the raw ridge (0.152) lead, the pretext 0.127, the states 0.090; on the second the ridge per
   channel (0.722). With five to ten events a direction over the states, pretrained or not,
   beats the trees taught the same events by 0.10 to 0.12.
5. The verdict's bootstrap and the reading's each cost hours on a first-mission campaign in
   pure Python.

**Limitations.** Two pretrainings, the ones the hypotheses came from; the initialisation and
the ridge are one control each. One seed draws both the taught events and the initialisation.
The event weights hinge the measure on twenty events of four windows or fewer, hence intervals
of ±0.07 for the taught detectors. The measures disagree on the order of detectors (F0.5 ranks
the states above the pretext, the area below); the comparison with the initialisation is the
same under every one. The per-channel knob and the trees went to the platform on predictions
from the literature without a pilot on the fitting side; both failed. Validation side, tier M.

**What follows.** The leg keeps no claim of pretraining over frozen states; the confirmation
reads it as the product table above. The pretraining question moves to fine-tuning with a few
taught events, pretrained against random initialisation, piloted on the host first.

## 2026-11-06 — the cells' farthest bar read 0.154 by a near-constant feature's spread

The bar `cells_knn@distance=farthest,reference=unit`, 0.154 above, was refitted on the same
mission by the context report ([telemetry-context.md](telemetry-context.md)) with 39 fitting
windows fewer, 61,324 against 61,363, and read 0.017 on the same 41,088 windows. Within every
month the two rankings are the same (Spearman 0.999, each month's area equal to the third
decimal); they differ in how the months line up in one ranking. One feature, the last
reading's offset in one bin of one channel, has a spread of 0.0096 on the smaller fitting side
and none on the larger, where the standardisation read it over one: so its departures grew a
hundredfold, and in the months 2009-07 to 2009-12 it was the farthest feature of half the
windows. Under one alignment for both, each month's scores over their mean and spread, both
read 0.21; over their rank within the month, 0.076. The direction and the whitened distance
over the states have continuous scores and read within 0.03 of their recorded areas under the
same alignment. The standardisation now floors a feature's spread at a tenth of the typical
one, so the bar is recomputed, with the same fit, in the channel-local note that follows.
