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
| Code | the commit that adds this section; pretrainings ordered from `7c46829f` |
| Tasks | in-hospital death `a6a9c653…` and sepsis `03c3a159…` on the half of the tuning side held out by seed 101, seeds 1 to 10; `esa-mission1-events` (the leg's endpoint), `esa-mission2-events`, `skab-anomalies`, `opssat-anomalies`, every fitting window, seeds 1 to 3; tier M |
| Pretrainings | the small shape (2 blocks × 64, 16 heads) over six-bin cells; *base*: half of the 2012 and of the 2019 training stays, each drawn by the run's seed, the eICU demo's stays of 48 hours and the first half of each mission's months of raw ESA telemetry, read 6, 2, 10 and 1 times an epoch; *wqp*: the same plus the portal's 24,095 locations once an epoch; seeds 1 and 2 each; 17 epochs, read off a run at seed 0 by the rule written before it (`scripts/pretraining_budget_report.py`, `5d8d1ab2`) |
| Platform | the pretrainings on one Colab G4 (RTX PRO 6000) in four lanes under CUDA MPS, fp32, 6,803 to 7,218 s; the campaigns on Kaggle T4 cards, fp32 |
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
0.42 % of ESA's raw readings lie beyond ten deviations, up to 26,358; they dominate its validation
loss, under seed 2 entirely, so the kept epoch answers the clinical corpora more than ESA. The
network from nothing is not reproduced across runs; it is read against fine-tuning within this run
only. Validation side, tier M.
