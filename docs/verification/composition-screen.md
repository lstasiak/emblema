# Screen over the composition

Whether the small encoder over cells, pretrained without labels on a composition of intensive-care
stays and raw satellite telemetry, fine-tunes to a lower error at 20, 50 and 200 labelled stays
than the best network from nothing in the same shape at each budget; whether adding the Water
Quality Portal changes that; and whether its states find the first mission's events without a
label. The rules are the register's, "Pretraining in the tasks' regime" and "The anomaly-detection
tasks" ([preregistration](../preregistration.md)); the detectors scored by the pretext's error are
[ADR-0068](../adr/0068-a-detector-scored-by-the-pretext-error.md).

## 2026-10-30 — declared before the run: four pretrainings over the composition, read on three legs

**Questions.** (1) Does fine-tuning a backbone pretrained over the composition gain over the
network from nothing at 20, 50 and 200 stays of the half, each arm at the setting its rule chose
at that budget? (2) Does the arm with the portal replace the base by the three registered
readings? (3) On the first mission, do the encoder's states beat both their initialisation and
the summaries, and does the pretext's error beat a ridge over the same visible cells? Beside them,
descriptively: 100 stays, the rate in force, the sepsis reader, the stays the pretraining read,
the other three telemetry tasks.

**Conditions.**

| | |
| --- | --- |
| Code | `378ba707`, which adds this section and places the campaigns' orders; pretrainings ordered from `7c46829f` |
| Tasks | in-hospital death `a6a9c653…` and sepsis `03c3a159…` on the half of the tuning side held out by seed 101, seeds 1 to 10; `esa-mission1-events` (the leg's endpoint), `esa-mission2-events`, `skab-anomalies`, `opssat-anomalies`, every fitting window, seeds 1 to 3; tier M |
| Pretrainings | the small shape (2 blocks × 64, 16 heads) over six-bin cells; *base*: half of the 2012 and of the 2019 training stays, each drawn by the run's seed, the eICU demo's stays of 48 hours and the first half of each mission's months of raw ESA telemetry, read 6, 2, 10 and 1 times an epoch; *wqp*: the same plus the portal's 24,095 locations once an epoch; seeds 1 and 2 each; 17 epochs, read off a run at seed 0 by the rule written before it (`scripts/pretraining_budget_report.py`, `5d8d1ab2`) |
| Platform | the pretrainings on one Colab G4 (RTX PRO 6000) in four lanes under CUDA MPS, fp32, 6,803 to 7,218 s; the campaigns on one Colab G4 in 24 lanes under CUDA MPS, fp32 |
| Campaigns, per backbone | `composition-physionet2012` (the network from nothing at the rate in force; fine-tuning at 0.00001, 0.00003, 0.0001, 0.000333, 0.001 and 0.003, and with the encoder at a tenth or a hundredth of the rate once the head is solved; 20, 50, 100, 200); `cell-pretraining-probes-physionet2012` (500); `reader-probes-physionet2019` (400, 4,000, every patient); `composition-mission1` (the states and their initialisation, the pretext's error); `detection-events-per-unit` on SKAB, `detection-events-counts` on the second mission and OPS-SAT |
| Campaigns, once | `composition-scratch-physionet2012` (the network from nothing at the six rates); `composition-mission1-bars` (summaries, counts, the ridge and the interpolation, under `base1`'s masks, which every backbone draws by one rule) |
| Worker | `data/report/t42c/worker-env.sh` (the rate in force 0.001), the backbone's weights per campaign |
| Classical bar | trees per channel 0.621 / 0.628 / 0.736 at 20 / 50 / 200 on the same half and seeds (`gap-to-the-trees.md`) |

| Recipe | Seed 1 | Seed 2 |
| --- | --- | --- |
| base | `1408e0b6` (`0a4dc540…`) | `89e4d665` (`48175bbf…`) |
| wqp | `25968c1c` (`41c6b588…`) | `ce8a19fc` (`984a8904…`) |

**What was seen before this section.** The four pretrainings were accepted and none was read on a
task. Every one kept its last epoch; the portal's arm reads the clinical pretext 0.003 to 0.018
of the trivial cost above the base. Under seed 2 the ESA validation holds a reading thousands of
deviations out: its loss is about 12,219 against 2.9 under seed 1, its ratio to the trivial
predictor 1.00 at every epoch, so the kept epoch is unchanged but seed 2 says nothing of ESA.
Under the budget run's backbone the pretext answered ESA's cells within ten deviations about three
times closer than interpolation. The trial ran every campaign at seed 1 under that backbone and
printed only the cells accepted: 9 and 6 a budget for fine-tuning and the network from nothing,
2 for the probe, 2 a budget for the sepsis reader, 3 and 5 on the first mission, 5, 4 and 4 on
SKAB, OPS-SAT and the second mission; the reading ran on it to files left unopened. The first
mission's cells scored by an error took 46 to 73 minutes on the M1 Pro; the reading of a draw was
then batched, and the trial's `pretext_error` cell, run again, answers its 41,088 windows
identically in 3,046 s against 4,386.

**Reading.** Pairs by `scripts/campaign_pairs_report.py` (10,000 resamples); a side of several
pretrainings is their mean.

- *Settings.* At each budget apart by the registered rule (`scripts/rate_choice_report.py`, 2,000
  resamples, seed 1): fine-tuning over its eight settings and all four backbones together, so the
  arms' contrast reads both at one setting; the network from nothing over its six rates. Where
  several clear the bar the largest gain is chosen; an edge of the grid is named.
- *Gains.* Fine-tuning at its chosen setting less the network from nothing at its chosen rate,
  over four pretrainings and per recipe (n = 2), against T = 2·√(SE² + (s²/2)/n), each given its
  level (`scripts/pretraining_screen_report.py`); s from the pairs of seeds within a recipe.
  Descriptive: a screen confirms nothing. The rate in force and 100 stays beside.
- *Choice of the recipe.* The portal less the base: fine-tuning at 20, 50 and 200 against
  2·√(SE² + s²/2); the probe at 500 against 2·√(SE² + s²), s the larger of 0.014 and the probe's
  own spread between seeds; the first mission's `frozen_knn@reference=unit` against 0.05 with the
  interval clear of zero. The portal replaces the base when it is worse by more than the threshold
  on none and better on at least one.
- *The telemetry leg.* On the first mission, per recipe: the states less their initialisation and
  less the summaries, each counted at 0.05 with the interval clear of zero, both needed; the
  pretext's error less the ridge and less the summaries the same way; the interpolation beside.
  The other three tasks as in their first reading (`anomaly-tasks.md`).
- *Null pairs.* The network from nothing at the rate in force under `base1`'s campaign against the
  other four, at every budget, within ±0.02, or the reading stops.
- *Probe, sepsis, transduction.* The probe at 500 and the sepsis reader at 4,000 less the encoder
  at its initialisation, per recipe; `scripts/transduction_report.py` on the stays each backbone
  read against the rest, fine-tuning at its chosen setting on 2012, the sepsis reader on 2019.

**Predictions** (from earlier intervals, the lower edge where a gain is claimed; sources in
`cell-pretraining.md` unless named).

| Reading | Prediction | Source |
| --- | --- | --- |
| Spread s | in [0.014, 0.025] | own spread 0.013 over cells; 0.025 at 50 over raw tokens (`few-label-fine-tuning.md`) |
| Gain over four, chosen settings, 20 | noise, inside [−0.005, +0.010] | +0.003 [−0.005; +0.010] at both arms' chosen rates |
| Gain over four, chosen settings, 50 | noise, inside [−0.001, +0.015] | +0.014 [+0.006; +0.022] there, less the 0.007 the network from nothing gains from its best rate at 50 |
| Gain over four, chosen settings, 200 | noise, inside [−0.015, +0.007] | −0.004 [−0.015; +0.007] |
| Gain at the rate in force, 200 | inside [+0.010, +0.031] | +0.021 there |
| Settings | the network from nothing at 0.0001 or below at 200; fine-tuning's choice within 0.003 of the rate in force at every budget; the encoder's shares: none declared | the network rose from 0.718 to 0.742 down to 0.0001; fine-tuning moved 0.003 or less across rates; the shares were read only under backbones whose probe gained nothing |
| Curve | the gain at the chosen settings changes sign between 50 and 200 | as over cells |
| Portal less base | within the threshold on all three readings, so the base stays | grids less base within ±T at every budget and +0.007 on the probe; the portal fails the tasks' regime (`task-regime-sources.md`) |
| Probe at 500 | above its threshold for each recipe, inside [+0.044, +0.091] | +0.060 ± 0.008 for the base over cells, up to +0.091 for its variants |
| First mission, states less initialisation | inside [−0.011, +0.037], below 0.05: the endpoint not met | +0.007 [−0.011; +0.037] under backbones that never read ESA (`anomaly-tasks.md`) |
| First mission, states less summaries | inside [+0.011, +0.148] | +0.063 [+0.011; +0.148] there |
| First mission, pretext less ridge | none declared; the pretext above the summaries | no detector scored by an error was read; the summaries' 0.021 is ten times chance |
| Sepsis reader, 4,000 | above initialisation, inside [0.000, +0.091] | no backbone over cells was read on sepsis; the 2012 probe gained +0.060 to +0.091 |
| Transduction, 2012 and 2019 | the gap inside [−0.025, +0.009] at every budget | −0.007 / −0.003 / −0.010 at 20 / 50 / 200 on 2012, intervals reaching −0.025 and +0.009 |
| Null pairs | within ±0.002 | ±0.0015 within one run over cells |
| Other telemetry tasks | SKAB states less initialisation within ±0.05, the summaries ahead there; OPS-SAT states below initialisation | `anomaly-tasks.md`, 2026-10-28 |

**What follows.** The recipe the rule chooses is pretrained eight times more and confirmed on set
B at 30 seeds, at the setting each arm chose at each budget, by a declaration of its own; if it
is the portal's, the register's row widening the claim's data comes first. The shape of 4.8M
parameters waits for that confirmation, under the condition the register names. The telemetry leg
is read again on the eight fresh pretrainings by the same rule.

**Limitations.** Two pretrainings an arm give a coarse spread, bounded below by the floor. The
budget is a bound from above: the run that set it decayed its rate over 24 epochs, these over 17.
Of ESA's readings, 0.06 % of the first mission's and 0.05 % of the second's lie beyond ten
deviations on the months the pretraining read, at most 226; on the held-out months the detection
tasks validate on, 0.14 % and 0.05 %, up to about 26,000 on the first mission and one reading of
5·10¹¹ on the second. Such readings dominate ESA's validation loss in the pretraining, which reads
these months among the held-out ones, under seed 2 entirely, so the kept epoch answers the
clinical corpora more than ESA. The network from nothing is not reproduced across runs; it is read
against fine-tuning within this run only. Validation side, tier M.

## 2026-10-31 — Colab G4, fp32: fine-tuning gains 0.016 at 20 and 50 stays, the portal stays out, and the satellites' states lose the readings' size

**Question.** As declared above: the gain of fine-tuning over the best network from nothing at
each budget, whether the portal replaces the base, and whether the first mission's events are
found without a label by the encoder's states and by the pretext's error.

**Conditions.**

| | |
| --- | --- |
| Code | `378ba707` for every cell (each order carries it); the reading at `3902a4e0`, whose scripts are those of `378ba707` |
| Machine | one Colab G4 (RTX PRO 6000), 24 lanes under CUDA MPS, fp32; 219 pieces of the 30 orders, none retried, 4,184 s |
| Cells | 2,207, accepted lane by lane (`data/report/l32b5a/accept-campaigns.sh`) |
| Reading | `data/report/l32b5a/read.sh`, every step as declared, exit 0 |

**Areas** under the four pretrainings, at each arm's chosen setting (AUROC, mean over seeds):

| Stays | Fine-tuning (setting) | Network from nothing (rate) | Trees per channel |
| --- | --- | --- | --- |
| 20 | 0.631 (0.0001) | 0.615 (0.001) | 0.621 |
| 50 | 0.670 (0.001) | 0.654 (0.001) | 0.628 |
| 100 | 0.711 (0.001) | 0.695 (0.000333) | |
| 200 | 0.745 (0.001) | 0.741 (0.0001) | 0.736 |

The encoder at a tenth or a hundredth of the rate never clears the bar (−0.003 to −0.021).

**Gains of fine-tuning over the network from nothing** (`pretraining_screen_report.py`;
s = 0.014, the floor; the pairs' own spread 0.009):

| Stays | Four pretrainings | Base (n = 2) | Portal (n = 2) | Level, four |
| --- | --- | --- | --- | --- |
| 20 | +0.016 [+0.007; +0.025] | +0.025 [+0.015; +0.036] | +0.007 [−0.002; +0.016] | noise (T 0.030) |
| 50 | +0.016 [+0.007; +0.025] | +0.025 [+0.015; +0.034] | +0.007 [−0.003; +0.016] | real and small (T 0.015) |
| 200 | +0.004 [−0.007; +0.015] | +0.011 [−0.001; +0.022] | −0.003 [−0.014; +0.008] | noise (T 0.015) |
| 200, rate in force | +0.022 [+0.012; +0.032] | +0.031 [+0.020; +0.043] | +0.013 [+0.002; +0.024] | at the least gain |

**Choice of the recipe.** The portal less the base: fine-tuning −0.018, −0.018 and −0.014 at 20,
50 and 200 stays, each within ±T (0.025, 0.027, 0.021); the probe at 500 −0.014 (T 0.020); the
first mission's states +0.005 [−0.004; +0.016]. Worse past T on none, better on none: **the base
stays**.

**Representation.** The probe at 500 stays: base 0.762 against 0.673 untrained, +0.090
[+0.067; +0.113]; portal +0.061 [+0.039; +0.083]. The sepsis reader at 4,000 patients, base:
+0.009 [−0.002; +0.019]; at every patient +0.013 [+0.002; +0.025].

**The first mission** (event-weighted PR-AUC; reduction of 1 − PR-AUC, 10,000 resamples):

| Pair | Base | Portal |
| --- | --- | --- |
| States less their initialisation (endpoint) | +0.018 [−0.008; +0.045] | +0.011 [−0.011; +0.036] |
| States less the summaries | **+0.074** [+0.011; +0.158] | **+0.080** [+0.013; +0.165] |
| Pretext's error less the ridge | −0.026 [−0.050; +0.019] | −0.017 [−0.048; +0.065] |
| Pretext's error less the summaries | **+0.106** [+0.048; +0.194] | **+0.114** [+0.059; +0.225] |
| Pretext's error less the interpolation | +0.040 [+0.012; +0.102] | +0.048 [+0.015; +0.143] |

PR-AUC: the ridge 0.152, the pretext's error 0.127 / 0.135, the states 0.096 / 0.101, their
initialisation 0.077, the interpolation 0.087, the summaries 0.021. Neither leg's condition
holds: the states do not pass their initialisation, the pretext's error does not pass the ridge.

**The other telemetry tasks.** SKAB: states less initialisation +0.005 / −0.021, the summaries
ahead by 0.13 to 0.15. OPS-SAT: the states below their initialisation, −0.052 [−0.091; −0.014]
and −0.029 [−0.071; +0.006]. The second mission: the states far below their initialisation,
−0.432 [−0.594; −0.312] and −0.393 [−0.545; −0.238] (PR-AUC 0.06 / 0.10 against 0.49), where
backbones that never read ESA lay between 0.31 and 0.59 (`anomaly-tasks.md`).

**Predictions.**

| Reading | Predicted | Observed | Held |
| --- | --- | --- | --- |
| Spread s | [0.014, 0.025] | 0.014 (own 0.009) | yes |
| Gain, four, 20 | noise, in [−0.005, +0.010] | noise, +0.016 | level yes, range no (above) |
| Gain, four, 50 | noise, in [−0.001, +0.015] | real and small, +0.016 | no (above) |
| Gain, four, 200 | noise, in [−0.015, +0.007] | noise, +0.004 | yes |
| Gain at the rate in force, 200 | in [+0.010, +0.031] | +0.022 | yes |
| Settings | network from nothing at 0.0001 or below at 200; fine-tuning within 0.003 of the rate in force | 0.0001; within 0.0016 | yes |
| Curve | the gain changes sign between 50 and 200 | +0.016 to +0.004, no change | no |
| Portal less base | within T on all three; the base stays | within T; the base stays | yes |
| Probe at 500 | above its threshold for each recipe, in [+0.044, +0.091] | +0.090, +0.061 | yes |
| M1 states less initialisation | in [−0.011, +0.037], below 0.05 | +0.018, +0.011 | yes |
| M1 states less summaries | in [+0.011, +0.148] | +0.074, +0.080 | yes |
| M1 pretext above the summaries | above | +0.106, +0.114 | yes |
| Sepsis reader, 4,000 | above initialisation, in [0.000, +0.091] | +0.009 [−0.002; +0.019] | no (not clear of zero) |
| Transduction | gap in [−0.025, +0.009] | −0.010 to −0.005, intervals [−0.023; +0.009] | yes |
| Null pairs | within ±0.002 | base 0.000; portal +0.013, −0.008, +0.004 | no for the portal |
| SKAB; OPS-SAT | states within ±0.05, summaries ahead; states below | as predicted | yes |

The null pairs of the portal's campaigns differ because the network from nothing is drawn over
the larger of the task's and the backbone's vocabularies: 508 channels under the portal, 435
under the base, so its initial weights differ. Every gain reads the network from nothing in one
campaign (`48ef3714`, 435 channels), so no gain moves. The draw alone moves that network by up to
0.013 at 20 stays, the size of the floor on s.

**After the results (not declared; window-level AUROC, each seed).** The events of the first
mission split by whether their window holds a reading beyond ten deviations (13 % of them;
none of the normal windows on the second mission):

| Detector, base | M1 beyond 10 | M1 within 10 | M2 beyond 10 | M2 within 10 |
| --- | --- | --- | --- | --- |
| Initialisation, seeds 1–3 | 0.77–0.83 | 0.49–0.51 | 0.95–0.99 | 0.75–0.79 |
| States | 0.73–0.75 | 0.50–0.52 | 0.67–0.75 | 0.55–0.59 |
| Pretext's error | 0.81–0.83 | 0.54–0.55 | | |
| Ridge | 0.82–0.83 | 0.58–0.59 | | |
| Summaries | 0.83 | 0.57 | 0.93 | 0.70 |

The largest |z| of a window, ranked alone, reads 0.60 on M1 and 0.85 on M2. Scripts and tables:
`data/report/l32b5a/posthoc/`.

**Conclusions.**

1. Fine-tuning over the base gains about 0.016 at 20 and 50 stays over the best network from
   nothing, with intervals clear of zero, and nothing at 200; above the predictions, at the
   literature's level, from a screen: a confirmation on fresh pretrainings, predicted from the
   lower edge, decides it.
2. The portal joins nowhere: the base stays the recipe.
3. The representation stands above its initialisation on the stays (probe +0.090) and only
   faintly on sepsis.
4. On the first mission neither the states nor the pretext's error meet their condition. Outside
   the 13 % of events with a reading beyond ten deviations every detector is near chance (0.59 at
   best), so the task resolves mainly whether a reading's size is seen.
5. After the results: pretraining over the satellites lowers the states' sensitivity to large
   readings (0.05 to 0.07 on M1, 0.2 to 0.3 on M2, below a random encoder of the same shape).
   The mechanism is not shown; a test of the states under scaled inputs comes first.

**Limitations.** Two pretrainings a recipe; the base was chosen as the better, so its own
+0.025 is optimistic. The analysis after the results uses another measure than the register's,
on the same windows that chose it, and its third subgroup on M2 (56 windows, 45 of one class) is
too small to read. The trees come from another run on the same half and seeds. Validation side,
tier M.
