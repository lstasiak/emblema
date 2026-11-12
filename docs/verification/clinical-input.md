# How the clinical values are prepared for the network

Readings of the intensive-care stays are standardised per channel before a network sees them.
On PhysioNet 2012 a handful of impossible values (a pH of 735, temperatures below zero) and the
heavy tails of the laboratory channels inflate the standard deviation, so the clinical range of
some channels reaches the network compressed near zero. This note reads whether cleaning those
values (ADR-0078) and normalising robustly or by quantile (ADR-0079) changes what the network
from scratch learns, against the trees on the same input.

## 2026-11-11 — declared before the run: four preparations of the stays, read by the network from scratch and the trees

**Question.** Does the small network from scratch on six-bin cells read the in-hospital-death
task better when the stays are cleaned by the literature's bounds, and better again when they
are normalised robustly or by quantile, at 20, 50, 200 and every stay of the half? Which
preparation do the pretrainings that follow use?

**Conditions.**

| | |
| --- | --- |
| Publications | PhysioNet 2012 by the command of the mixture's chain (one window per stay, set B held out, the vocabulary continued from the ESA-AD manifest `773503…`): *raw* `717bc832…` (the publication in use; published again from this code, it yields the same manifest), *cleaned* `bf7c1a0f…` (`corpora/cleaning/physionet2012.toml`, the standard score), *robust* `d2d86d29…`, *quantile* `a578b287…` (cleaned, then normalised as named). Units, split, windows and vocabulary are identical across the four; the cleaned three hold 681 fewer tokens |
| Tasks | in-hospital death defined on each: raw `a6a9c653…`, cleaned `271dc30e…`, robust `c8584b61…`, quantile `39310271…` |
| Campaigns | `clinical-input-low-physionet2012` (20, 50, 200 stays) and `clinical-input-all-physionet2012` (every stay, each arm stopping on a fifth of its labels) under each task: the small shape at rates 0.003, 0.001, 0.000333, 0.0001 and 0.00003, and the trees per channel at the setting in force and at `learning_rate=0.1,rounds=300` |
| Half and seeds | the half of the tuning side held out by seed 101, seeds 2 to 31; cells pair by stay and seed across the four tasks |
| Worker | `data/report/t42c/worker-env.sh` under the composition's six-bin backbone `0a4dc540…`, read only for the size of its vocabulary, so the network starts from the same weights under every task |
| Platform | the networks on Colab G4 in lanes under CUDA MPS, fp32; the trees as the general pool on the Mac's CPU |
| Tier | M; validation side of the tuning set; set B is not read |

**What the cleaning did.** Over sets A and B the bounds removed 15 pH readings, 205 temperatures,
234 weights, 201 urine outputs and a few others, and clipped 691 urine outputs, 198 mean arterial
pressures and fewer elsewhere (each manifest counts them). On the training side the clinical range
of pH (7.2–7.5) spans 0.04 of a unit under the raw standard score and 3.9 once cleaned; of ALT
(20–200), 0.15 raw and cleaned, 4.3 robust and 1.4 by quantile; of the heart rate (60–120), 3.3 to
3.4 under all four.

**What was seen before this section.** A trial ran both campaigns at seed 1, at 50 and every
stay, under the raw and the quantile tasks, on the Mac (MPS for the networks, CPU for the trees):
28 cells accepted, about a minute a network cell. The reading plan then ran whole on it, and its
last step printed the level of the quantile arm against the raw one at that seed: noise at 50
stays, a gain at every stay. The predictions, the rule and its thresholds below were written
before that print (the file at 20:31, the print at 20:39) and were not changed. Seed 1 is left out
of the run, which reads seeds 2 to 31. That does not make every stay unread: at every stay each
seed learns from the same labels and scores the same stays, so seeds share most of their noise
there, while at 50 stays the stays' share is 0.0006 against a seed's 0.0157 (`data/report/l48/
power/`). The choice is therefore read at 20, 50 and 200 stays, and every stay is read beside it,
for every arm alike.

**Reading** (`data/report/l48/read.sh`, run whole on the trial before this section was
committed).

- *Rates.* Each arm keeps, at each budget apart, the rate the registered rule chooses over its
  five (`scripts/rate_choice_report.py`, 2,000 resamples): the rate in force, 0.001, unless
  another's mean area exceeds it by more than one standard error. Every arm meets the same grid,
  because a value of another scale may want another rate. The grid of the screen over the
  composition also held 0.00001, which no budget chose there.
- *Gains.* Each arm's network at its rate less the raw network at its rate, at 20, 50 and 200
  stays, paired over stays and seeds (`scripts/campaign_pairs_report.py`, 10,000 resamples). The
  three budgets of an arm are one family under Holm: a *gain* where Holm rejects the budget's
  paired p-value at 0.05 and the mean over seeds is positive and exceeds twice its standard error;
  *harm* where the interval lies below zero and the loss exceeds twice that error, read at each
  budget alone so as to stop an arm more readily; *noise* otherwise.
- *The rule.* An arm stays where it gains at one budget or more and harms at none. Of the arms
  that stay, the simplest is kept (cleaned, then robust, then quantile), and a more complex one
  replaces it only where its own pair with it gains at one budget and harms at none, by the same
  test. If no arm stays, the cleaned publication is kept unless it harms at a budget, and the raw
  one then: its bounds correct errors of the source and were fixed before any reading. The kept
  input is pretrained twice and screened against the composition's base before any confirmation
  reads it.
- *Beside, descriptively.* Every stay for every arm; each pair at the rate in force, 0.001, where
  either side chose another, since a gain read at chosen rates may come partly from the choice;
  the trees on each input less the trees on the raw one; each network at its rate less the trees
  on its own input, at every stay also less the trees at the tuned setting.

**Power.** Simulated over pairs of the network's rates in the screen over the composition, the
closest pairs measured of one network under two settings (`scripts/comparison_power_report.py`;
`data/report/l48/power/`), by its normal model of seeds and stays, one budget alone and without
Holm. With 30 seeds the rule finds a gain of 0.01 with power 0.90 at 50 stays and 0.69 at 200,
and of 0.02 with power 1.00 at both; at 10 seeds the model gives 0.49 and 0.34, while applying
the rule to the ten seeds that ran gives 0.13 at 200 stays, so the model is the optimistic
reading there. At no gain the rule fires at 0.03 to 0.04 a budget; Holm holds an arm's three
budgets to 0.05 together. Every stay was not simulated.

**Predictions** (from earlier intervals; the lower edge where a gain is claimed).

| Reading | Prediction | Source |
| --- | --- | --- |
| Cleaned less raw, network | noise at every budget, inside [−0.017, +0.016] | no interval measured a cleaning; the clip at three on the cells read −0.017 and +0.003 at 50 and 200 stays, points within ±T (`encoder-candidates.md`, 2026-10-24); +0.016 is the lower edge below |
| Robust or quantile less raw, network | a gain at one budget or more, inside [0, +0.016]; harm at none | no interval measured a normalisation; STraTS with a linear value embedding regained +0.034 [+0.016; +0.053] once its readings were clipped (`intensive-care-curve.md`, 2026-10-02), and its lower edge bounds the gain from above since a clip on the cells gained nothing; argued, so a gain under 0.01 may read as noise |
| Trees, cleaned or robust less raw | within ±0.005 | argued: both are affine within a channel's bounds, and cleaning changes the window summaries of few stays |
| Trees, quantile less raw | within ±0.01 | argued: a nonlinear map moves the windows' means, not the order of single readings |
| Network less trees at every stay, raw | −0.008, inside [−0.023, +0.007] | `encoder-candidates.md`, 2026-10-24 |
| Rates | 0.0001 or below at 200 stays for every arm | the network from scratch rose from 0.718 to 0.742 down to 0.0001 there (`cell-pretraining.md`, 2026-10-29) |

**What follows.** The kept input becomes the publication of 2012, 2019 and the eICU demo that
the pretrainings read; ADR-0079 is accepted with the normalisation kept, or rejected if the
cleaned standard score is kept.

**Limitations.** One task and one corpus; 2019 and the eICU demo are republished under the
input this reading keeps, not read here. The power comes from pairs of rates, whose seeds may
agree more than those of two inputs. Choosing the best of five rates lifts every arm alike, so
levels are optimistic. The choice lifts an arm only where a rate other than the one in force
clears its bar, so differences between arms may also carry it; the pairs at the rate in force
show how much. The bounds keep zero weights (255 readings in 12 stays) and remove 234 weights
above 250 kg in 6 stays that may be real (`data/report/l48/audit/weights.md`); neither was tuned.
Validation side, tier M.

## 2026-11-12 — Colab G4 and the Mac's CPU: the robust score is kept; it lifts the network from scratch at 20 and 50 stays

**Question.** As declared above: which of the four preparations of the stays' values the network
from scratch reads best at 20, 50 and 200 stays, and so which one the pretrainings read.

**Conditions.**

| | |
| --- | --- |
| Code | `d3d65254` for every cell (each order carries it) and for the reading |
| Networks | one Colab G4 (RTX PRO 6000), 24 lanes under CUDA MPS, fp32; 32 pieces, 2,400 cells, 4,005 s. Two earlier starts of the same cell answered no cell: the cell had not fetched the stays' outcomes, so every piece stopped within 8 s |
| Trees | the Mac's CPU (M1 Pro), 8 orders, 960 cells, 3 h 42 min |
| Reading | `data/report/l48/read.sh` on the run's eight campaigns, every step as declared, exit 0; every campaign complete, 2,000 stays scored a cell |

**Areas** (AUROC, mean over 30 seeds; the network at the rate it chose, the rate in brackets where
not 0.001):

| Stays | Raw | Cleaned | Robust | Quantile | Trees, raw / robust | Trees tuned, raw |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 0.604 | 0.607 (0.003) | **0.632** | 0.608 | 0.608 / 0.601 | 0.616 |
| 50 | 0.657 (0.003) | 0.661 | **0.669** | 0.656 | 0.641 / 0.644 | 0.642 |
| 200 | 0.744 (0.0001) | 0.748 (0.0001) | 0.748 (0.0001) | 0.745 (0.0001) | 0.738 / 0.738 | 0.741 |
| every stay | 0.827 (0.000333) | 0.828 | 0.832 | 0.835 (0.000333) | 0.832 / 0.836 | 0.839 |

Two choices lie at the edge of the grid: 0.003 for the cleaned network at 20 stays and for the
raw one at 50.

**Gains of each preparation over the raw one**, the network at each side's chosen rate (paired
over 2,000 stays and 30 seeds; Holm over an arm's three budgets):

| Pair | 20 | 50 | 200 | Level |
| --- | --- | --- | --- | --- |
| Cleaned less raw | +0.003 [+0.001; +0.006] | +0.004 [0.000; +0.007] | +0.003 [+0.001; +0.006] | noise at each: Holm rejects none (least adjusted p 0.051) |
| Robust less raw | **+0.028** [+0.021; +0.036] | **+0.012** [+0.005; +0.018] | +0.004 [−0.002; +0.009] | gain at 20 and 50, harm at none |
| Quantile less raw | +0.004 [−0.001; +0.009] | −0.001 [−0.007; +0.005] | +0.001 [−0.004; +0.007] | noise at each |
| Quantile less robust | −0.024 | −0.013 | −0.002 | harm at 20 and 50 |

Robust is the only arm that stays, and so it is kept. It gains on 26 of 30 seeds at 20 stays and
20 of 30 at 50. Robust less cleaned gains +0.025 at 20 stays, so the gain comes from the
normalisation, not from the cleaning.

**Beside, descriptively.**

| Reading | 20 | 50 | 200 | Every stay |
| --- | --- | --- | --- | --- |
| Robust less raw, both at the rate in force | +0.028 | +0.014 | +0.002 | +0.009 [+0.003; +0.015] |
| Robust less raw, at their rates | | | | +0.005 [−0.001; +0.010] |
| Quantile less raw, at their rates | | | | +0.009 [+0.003; +0.014] |
| Trees, robust less raw | −0.007 [−0.010; −0.004] | +0.002 | −0.001 | +0.004 |
| Network on robust less the best trees of the four inputs, at the setting in force | +0.023 (23 of 30 seeds) | +0.025 (24) | +0.010 (18) | |
| Network less the tuned trees, raw / robust | | | | −0.012 / −0.007 |

The screen over the composition (`composition-screen.md`, 2026-10-31) measured fine-tuning at
0.631 and 0.670 at 20 and 50 stays, on the raw publication, at seeds 1 to 10; the network from
scratch on the robust one reads 0.632 and 0.669 here. Seeds and campaigns differ, so this is no
paired reading, but at 50 stays the screen's gain of +0.016 is about as large as what the
normalisation alone gives (+0.012).

**Predictions.**

| Reading | Predicted | Observed | Held |
| --- | --- | --- | --- |
| Cleaned less raw, network | noise, in [−0.017, +0.016] | noise, +0.003 to +0.004 | yes |
| Robust less raw, network | gain at one budget or more, in [0, +0.016]; harm at none | gain at 20 and 50, +0.028 and +0.012 | level yes, range no at 20 (above) |
| Quantile less raw, network | as robust | noise at each budget | no |
| Trees, cleaned or robust less raw | within ±0.005 | within, except robust at 20 stays, −0.007 | no at one budget |
| Trees, quantile less raw | within ±0.01 | −0.006 to +0.003 | yes |
| Network less trees, every stay, raw | −0.008, in [−0.023, +0.007] | −0.005 [−0.021; +0.011] | yes |
| Rates | 0.0001 or below at 200 for every arm | 0.0001 for every arm | yes |

**Conclusions.**

- The pretrainings read the stays cleaned by the literature's bounds and normalised by the robust
  score; ADR-0079 is accepted with it for the intensive-care corpora.
- The robust score lifts the network from scratch where labels are few: +0.028 at 20 stays and
  +0.012 at 50. At 200 the gain is within the noise; at every stay it is +0.005 to +0.009.
- Cleaning alone moves the network by less than 0.005 at the chosen rates, and the quantile
  score does not move it at 20 to 200 stays. At every stay the quantile score is within the noise
  of the robust one (+0.004 [−0.001; +0.008]).
- At 20 and 50 stays the network on the robust input leads the best tree model, tuned or not,
  on any input, by 0.015 and 0.023. At every stay it stays below the tuned trees, within the noise.
- A gain of pretraining measured against the network on the raw input is not a gain against this
  control: at 50 stays the screen's +0.016 is about as large as the normalisation's own gain,
  read across campaigns, unpaired.

**Limitations.** One task and one corpus: 2019 and the eICU demo are published robust without a
reading of their own. Robust both scales and bounds at ±10; which of the two gives the gain was
not read. The trees lose 0.007 on the robust input at 20 stays for no reason measured, so they
are no fairer a bar there than on the cleaned one. Two arms chose a rate at the edge of the grid.
The comparison with the screen is across campaigns and seeds. Validation side, tier M.
