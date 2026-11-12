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

## 2026-10-31 — Colab G4, fp32: fine-tuning gains 0.016 at 50 stays, the portal stays out, and the first mission's events are not found

**Question.** As declared above: the gain of fine-tuning over the network from nothing, each at
the setting its rule chose at each budget; whether the portal replaces the base; and whether the
first mission's events are found without a label by the encoder's states and the pretext's error.

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

The encoder at a tenth or a hundredth of the rate never clears the bar (−0.021 to +0.001).

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

Bold: past the least gain of 0.05 with the interval clear of zero.

PR-AUC: the ridge 0.152, the pretext's error 0.127 / 0.135, the states 0.096 / 0.101, their
initialisation 0.077 / 0.090, the interpolation 0.087, the summaries 0.021. Neither leg's condition
holds: the states do not pass their initialisation, the pretext's error does not pass the ridge.

**The other telemetry tasks.** SKAB: states less initialisation +0.005 / −0.021, the summaries
ahead by 0.13 to 0.15. OPS-SAT: the states below their initialisation, −0.052 [−0.091; −0.014]
and −0.029 [−0.071; +0.006]. The second mission: the states far below their initialisation,
−0.432 [−0.594; −0.312] and −0.393 [−0.545; −0.238] (PR-AUC 0.06 / 0.10 against 0.49); the
states under backbones that never read ESA read 0.31 to 0.59 there (`anomaly-tasks.md`).

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
| Transduction, 2012 | gap in [−0.025, +0.009] | −0.010 to −0.005, intervals [−0.023; +0.009] | yes |
| Transduction, sepsis | the same | −0.004, +0.004, +0.004; intervals up to +0.021 | points yes, intervals wider |
| Null pairs | within ±0.002 | base 0.000; portal +0.013, −0.008, +0.004 | no for the portal |
| SKAB; OPS-SAT | states within ±0.05, summaries ahead; states below | as predicted | yes |

The null pairs of the portal's campaigns differ because the network from nothing is drawn over
the larger of the task's and the backbone's vocabularies: 508 channels under the portal, 435
under the base, so its initial weights differ. Every gain reads the network from nothing in one
campaign (`48ef3714`, 435 channels), so no gain moves. The draw alone moves that network by up to
0.013 at 20 stays, the size of the floor on s.

**After the results (not declared; window-level AUROC, each seed).** Events split by whether
their window holds a reading beyond ten deviations: 13 % of the event windows on M1 and 47 % on
M2 do; no normal window on M2 does.

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

1. Over the four pretrainings fine-tuning gains 0.016 at 50 stays (real and small), the same at
   20 within the noise threshold, and nothing at 200; above the predictions. A screen confirms
   nothing. At the lower edge (0.007) the registered confirmation's power is below 0.25
   (`cell-pretraining.md`, 2026-10-29), so a miss there would not show the gain absent.
2. The portal joins nowhere: the base stays the recipe.
3. The representation stands above its initialisation on the stays (probe +0.090) and only
   faintly on sepsis.
4. On the first mission neither the states nor the pretext's error meet their condition.
5. After the results: on M1 every detector is near chance on the 87 % of events without a
   reading beyond ten deviations, and the pretrained states trail a random encoder only on the
   rest (by 0.05 to 0.07); on M2 they trail it everywhere (by 0.2 to 0.3 beyond, about 0.2
   within). Whether pretraining loses a reading's size is not shown; a test of the states under
   scaled inputs comes first.

**Limitations.** Two pretrainings a recipe; the base's own +0.025 rests on two pretrainings and
is the upper of two recipes, so the confirmation predicts from the four's lower edge. The
analysis after the results uses another measure than the register's, on the windows whose
results prompted it. The trees come from another run on the same half and seeds. Validation
side, tier M.

## 2026-11-12 — declared before the run: two pretrainings of the base over the input the stage settled, read on three legs

**Questions.** (1) On the stays cleaned and under the robust score (ADR-0078, ADR-0079) and
the telemetry under the robust score held to 100 (ADR-0080), does fine-tuning a backbone
pretrained over the composition gain over the network from nothing at 20, 50 and 200 stays of
the half, each arm at the setting its rule chose, 20 and 50 one family under Holm? (2) Does the
robust score lift the network from nothing on the 2019 stays as on 2012 (`clinical-input.md`,
2026-11-12), read on the sepsis task at 400 and 4,000 patients? (3) Does the pretraining read
the telemetry under its new score: a finite validation loss below the trivial predictor's under
both seeds, the pretext answering hidden cells of the second mission's channels 18 to 20 better
than at initialisation and than interpolation; and do the first mission's states and pretext
error read as on the standard publication? Beside, descriptively: 100 stays, the rate in force,
the probe, the sepsis reader, transduction, the second mission, the testbed and the fragments.
A screen confirms nothing: it sizes the gain the confirmation is declared from and checks the
new input's mechanics before sixteen pretrainings read it.

**Conditions.**

| | |
| --- | --- |
| Code | the commit adding this section and the Holm family to `pretraining_screen_report.py`; the campaigns' orders placed from it; the pretrainings ordered from `b5ea3b62`, which adds the 2019 campaign file |
| Input | 2012 `d2d86d29…`, 2019 `79364c26…` and the eICU demo `5f24d626…` cleaned and under the robust score held to 10; the raw ESA telemetry `490c1547…` under the robust score held to 100; the chain of vocabularies unchanged in its names |
| Tasks | in-hospital death on the 2012 publication above (`c8584b61…`), sepsis on the 2019 publication above (`dc3955ac…`) and on the one as read (`03c3a159…`), on the half of the tuning side held out by seed 101, seeds 1 to 10; `esa-mission1-events` (`60bbbcf4…`) and `esa-mission2-events` (`c064e171…`) on the robust telemetry, `skab-anomalies` and `opssat-anomalies` as before, every fitting window, seeds 1 to 3; tier M |
| Pretrainings | `backbone-cells-composition-m` at seeds 1 and 2, the recipe of 2026-10-30 unchanged, over the input above: `d61a6bcf` and `88c3e34a` |
| Platform | the pretrainings on Kaggle, one T4 each (the shape is bound by the host, so a G4 would idle); the campaigns on one Colab G4 in 24 lanes under CUDA MPS, fp32 |
| Campaigns, per backbone | as on 2026-10-30: `composition-physionet2012`, `cell-pretraining-probes-physionet2012`, `reader-probes-physionet2019`, `composition-mission1`, `detection-events-per-unit` on SKAB, `detection-events-counts` on the second mission and OPS-SAT |
| Campaigns, once | `composition-scratch-physionet2012`, `composition-mission1-bars` under the first pretraining's masks; `clinical-input-physionet2019`, the network from nothing at five rates, on each of the two 2019 publications |
| Worker | `data/report/t42c/worker-env.sh` (the rate in force 0.001), the backbone's weights per campaign |
| Classical bar | the trees per channel of `clinical-input.md` (2026-11-12), 0.608 / 0.641 / 0.738 at 20 / 50 / 200 stays on 2,000 stays over 30 seeds: another design than this half, so beside; the trees on this half come with the confirmation |

**What was seen before this section.** The first pretraining's order ran in a clean clone on
the Mac's CPU into the second epoch (loss 0.32 to 0.15, no NaN). The trial ran every campaign
at seed 1 and one budget (20 stays, 400 patients, 500 for the probe) under the 2026-10-30
screen's `base1`, whose vocabulary names the same channels, on one Colab G4: 50 pieces, none
retried, 1,863 s. Cells accepted: 17 on the 2012 stays, 12 on the 2019 patients, 8 on the first
mission, 4, 5 and 4 on the second, the testbed and the fragments. The reading ran whole on them
to files left unopened, after two corrections it showed: the budgets each export holds, and the
identifier the transduction reads.

**Reading.** Pairs by `scripts/campaign_pairs_report.py` (10,000 resamples); a side of two
pretrainings is their mean. `data/report/l32b6/read.sh`, every step; a budget is read where
its cells were exported.

- *Settings.* At each budget apart by the registered rule (`scripts/rate_choice_report.py`,
  2,000 resamples, seed 1): fine-tuning over its eight settings and both backbones together;
  the network from nothing over its six rates; on the 2019 stays each publication's network
  over its five rates at 400 and 4,000 apart.
- *Gains.* Fine-tuning at its chosen setting less the network from nothing at its chosen rate
  over the two pretrainings, against T = 2·√(SE² + s²/4), each given its level
  (`scripts/pretraining_screen_report.py`); s the larger of 0.014 and the pair of seeds' own
  spread. The gains at 20 and 50 stays are one family: a gain is a gain only where Holm over
  the pairs' bootstrap p-values rejects it too (`--family 20 50`); 200 stays is read alone.
  The rate in force and 100 stays beside.
- *The 2019 stays.* The network from nothing on the robust publication less on the one as read,
  each at the rate it chose, at 400 and 4,000 patients, paired by patient and seed; the rate in
  force beside. Descriptive: harm at a budget reopens the 2019 input before the confirmation.
- *The telemetry's mechanics.* From the pretrainings' logs, the ESA validation loss at the kept
  epoch under each seed, finite, its ratio to the trivial predictor below one; after the
  acceptance, `scripts/cell_pretext_audit.py` with `--model trained` and `--model
  initialisation` on the second mission's windows: the pretext's error on the hidden cells of
  channels 18 to 20 below the initialisation's and interpolation's. A seed that fails either
  stops the telemetry leg until it is explained.
- *The first mission.* As on 2026-10-30: the states less their initialisation and the
  summaries, the pretext's error less the ridge and the summaries, each at 0.05 with the
  interval clear of zero; the interpolation beside. The other three tasks as first read.
- *Null pairs.* The network from nothing at the rate in force under the first pretraining's
  campaign against the second's and its own, at every budget, within ±0.02, or the reading stops.
- *Probe, sepsis, transduction.* The probe at 500 and the sepsis reader at 400, 4,000 and every
  patient less the encoder at its initialisation; the transduction as on 2026-10-30.

**Size and power.** The largest real gain in reach is fine-tuning's +0.025 at 50 stays on the
raw input (2026-10-31, n = 2), of which the normalisation alone has since given +0.012; what
remains is near the 0.010 the confirmation is sized for. With n = 2 and s at the floor, T is
about 0.017 at 50 stays and 0.03 at 20: the screen cannot close a direction at 0.010 (power
below 0.25) and is not asked to.

**Predictions** (from earlier intervals, the lower edge where a gain is claimed; sources
named).

| Reading | Prediction | Source |
| --- | --- | --- |
| Spread s | in [0.014, 0.025] | own spread 0.009 on 2026-10-31; 0.025 over raw tokens (`few-label-fine-tuning.md`) |
| Gain, chosen settings, 20 | noise, inside [−0.015, +0.010] | the base's +0.025 [+0.015; +0.036] on the raw input (2026-10-31) less the +0.028 [+0.021; +0.036] the robust score gave the network from nothing (`clinical-input.md`) |
| Gain, chosen settings, 50 | noise or real and small, inside [−0.005, +0.020] | +0.025 [+0.015; +0.034] less +0.012 [+0.005; +0.018] |
| Gain, chosen settings, 200 | noise, inside [−0.015, +0.010] | +0.011 [−0.001; +0.022] less +0.004 [−0.002; +0.009] |
| Gain at the rate in force, 200 | inside [+0.010, +0.040] | +0.031 [+0.020; +0.043] less +0.002 |
| Network from nothing, areas | within 0.01 of 0.632 / 0.669 / 0.748 at 20 / 50 / 200 | `clinical-input.md`, 30 seeds on 2,000 stays |
| Settings | the network from nothing at 0.001 at 20 and 50, 0.0001 at 200; fine-tuning within 0.003 of the rate in force | `clinical-input.md`; 2026-10-31 |
| 2019, robust less as read, network from nothing | 400: gain or noise, inside [−0.010, +0.030]; 4,000: noise, inside [−0.010, +0.015]; harm at neither | 2012 at 20 / 50 / 200: +0.028, +0.012, +0.004 |
| ESA validation loss | finite under both seeds, within a factor of three of each other, ratio to the trivial predictor below one | seed 1 read 2.9, seed 2 about 12,219 on the standard score from readings thousands of deviations out, which the bound of 100 removes (`telemetry-input.md`) |
| Second mission's hidden cells, channels 18 to 20 | the trained pretext's error below initialisation's and interpolation's | the budget run's pretext answered ESA's cells three times closer than interpolation (2026-10-30) |
| Probe at 500 | above its threshold, inside [+0.044, +0.113] | +0.090 [+0.067; +0.113] for the base; +0.060 ± 0.008 over cells |
| First mission: states less initialisation; less summaries; pretext less ridge; less interpolation | inside [−0.010, +0.045], the endpoint not met; [+0.011, +0.158]; [−0.050, +0.020]; [+0.012, +0.102] | the base's four intervals of 2026-10-31 |
| Second mission, states less initialisation | below zero, inside [−0.35, +0.05], narrower than before | the states trailed a random encoder by 0.2 to 0.3 where the encoder read channels 18 to 20 as constant (2026-10-31) |
| Sepsis reader, 4,000 | inside [−0.002, +0.025] | +0.009 [−0.002; +0.019] at 4,000, +0.013 [+0.002; +0.025] at every patient |
| Transduction, 2012 and 2019 | the gap inside [−0.025, +0.009] at every budget | 2026-10-31 |
| Null pairs | within ±0.002 | ±0.0015 within one run over cells |
| Other telemetry tasks | SKAB states less initialisation within ±0.05, the summaries ahead; OPS-SAT states below initialisation | 2026-10-31 |

**What follows.** The spread and the gains read here go into the power model of the
confirmation, sixteen fresh pretrainings at 200 seeds on set B, by a declaration of its own;
the telemetry leg is read on the sixteen by the rule in force. A harm on the 2019 stays, or a
seed whose telemetry fails its mechanics, comes before that declaration.

**Limitations.** Two pretrainings; the spread is bounded below by the floor. The 2019 reading
is descriptive. The telemetry is read on the robust publication for the first time, so the
comparison with 2026-10-31 is across publications, not paired. Validation side, tier M.
