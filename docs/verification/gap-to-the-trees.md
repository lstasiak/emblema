# The network trained from nothing against the trees on the intensive-care stays

## 2026-10-22 — measured: the power of a non-inferiority rule against the trees on set B

**Question.** An encoder trained from nothing is to be judged against the trees by one
confirmation on set B, learning from every stay of set A. How likely is a rule of
non-inferiority to pass a network that truly matches the trees, for each margin, number of seeds
and number of scored stays?

**Conditions.**

| | |
|---|---|
| Code | `302fb316`; no new run |
| Machine | MacBook Pro M1 Pro, CPU |
| Noise taken from | campaign `87cbc0e7…` (`campaigns/curve-physionet2012-mixed5.toml`): learnt from every stay of set A, scored on set B (3,994 stays, 568 deaths), five seeds; `boosted_trees_per_channel` against `from_scratch` at every stay (`intensive-care-curve.md`, 2026-09-30) |
| Rule simulated | the paired interval of the network's area less the trees' lies above −δ, and that difference plus δ exceeds twice its standard error; the rule of `scripts/comparison_power_report.py` applied to the difference shifted by δ, at a true difference of zero |
| Draws | stays resampled within the two outcomes; 3,994 stays as set B, 8,000 as every stay of sets A and B scored once |

    uv run scripts/campaign_pairs_report.py --out data/report/l32b1/power-b \
      --campaign 87cbc0e7-806d-4062-bb05-3779ca53100f
    uv run scripts/comparison_power_report.py \
      --predictions data/report/l32b1/power-b/predictions.csv --out data/report/l32b1/power-b/sim \
      --budgets all --stays 3994 8000 --seeds 5 10 20 --gains 0.005 0.01 0.015 0.02 0.025 0.03 \
      --pair NI trees-vs-scratch 87cbc0e7-… boosted_trees_per_channel 87cbc0e7-… from_scratch

The simulation reads the spread of the difference, not its mean.

**Components of the difference** (standard deviations in area):

| Stays | Seed | Stays | Interaction |
|---|---|---|---|
| 3,994 | 0.0108 | 0.0047 | 0.0058 |
| 8,000 | 0.0107 | 0.0034 | 0.0041 |

**Power at a true difference of zero** (normal model of the components; at five seeds the model
agrees with the draws within 0.07):

| Stays | Seeds | δ = 0.005 | δ = 0.01 | δ = 0.015 | δ = 0.02 | δ = 0.025 |
|---|---|---|---|---|---|---|
| 3,994 | 5 | 0.18 | 0.40 | 0.67 | 0.86 | 0.96 |
| 3,994 | 10 | 0.21 | 0.51 | **0.80** | 0.95 | 0.99 |
| 3,994 | 20 | 0.20 | 0.54 | 0.84 | 0.97 | 1.00 |
| 8,000 | 10 | 0.29 | 0.67 | 0.92 | 0.99 | 1.00 |
| 8,000 | 20 | 0.32 | 0.76 | 0.97 | 1.00 | 1.00 |

**Conclusions.**

1. A margin of 0.01, near the top of the gains the literature reports from pretraining at every
   label (0.004 to 0.013 in area), has power 0.51 on set B at ten seeds. It stays below 0.80 on
   every stay that can be scored: 0.76 at 8,000 stays and twenty seeds. Set C is frozen until the
   final run.
2. More seeds help little: past ten, the stays and not the seeds carry the error.
3. On set B at ten seeds, 0.80 is reached at δ = 0.015. A network that passes is then no more
   than 0.015 below the trees by the 95 % interval, a one-sided bound at 97.5 %.
4. The margin of the confirmation is therefore 0.015, registered on 2026-10-22.
5. The same table, read with no margin, is the power of the superiority test that follows a
   passed confirmation: 0.21 at a true lead of 0.005 and 0.51 at 0.01 (set B, ten seeds). A lead
   of the size the literature reports over trees would most likely go unseen.

**Limitations.** The network of `87cbc0e7` ran without the stop; on the held-out fifth the stop
cut the spread over seeds from 0.018 to 0.006 (`intensive-care-curve.md`, 2026-10-02). The power
above is therefore likely conservative for a network under the stop, though not certainly so. Scoring 8,000 stays assumes a second
side as noisy as set B.

## 2026-10-23 — declared before the run: the gap to the trees taken apart

**Question.** On half the tuning stays the network trained from nothing lies 0.083 in area below
the trees per channel at 200 stays and 0.038 at 500. How much of that gap is what a network makes
of the trees' own table, and how much is what the encoder fails to build from the readings? Does
the encoder add anything once the table is handed to the head? And where do the trees stand at
20, 50, 1,000 and every stay of the half?

**Why this reading.** A new encoder can only close what it fails to build. A logistic head over
the table lies within 0.006 of the trees at 200 stays and 0.018 below them at 500
(`head-and-representation.md`, 2026-10-23), so most of the gap at 200 looks like representation.
A network handed the table under the network's own regime, stop, draws and seeds (ADR-0061) reads
the two parts on the same stays.

**Conditions.**

| | |
|---|---|
| Code | `87aac08e` (the knob at `fe5911eb`); campaigns `campaigns/gap-*-physionet2012.toml` |
| Task | in-hospital death, `a6a9c653…`; half of the tuning side by seed 101 (1,997 stays learnt from, 2,000 scored); seeds 1 to 10; tier M |
| Worker | `data/report/t42c/worker-env.sh` under the mixture of four at its first seed (`0d81c01e…`), whose 4.8M shape the network from nothing borrows |
| Recipes | registered per budget: `from_scratch` at 20 and 50, `@learning_rate=0.003` at 200, `@patience=10,stop_share=0.2` at 500, `@learning_rate=0.000333,patience=10,stop_share=0.2` at 1,000 and every stay; the statistics read under the same recipe, and at 500 and every stay also at three times and a third of its rate |
| Classical | trees per channel and MiniRocket at their settings in force; at every stay the trees `@learning_rate=0.1,rounds=300` and MiniRocket `@grid_resolution=2`, as chosen on the tuning side |
| Machines | classical cells on the M1 Pro CPU; network cells on one Colab G4 under CUDA MPS, fp32 |
| Reused cells | trees and MiniRocket at 200 and 500 from `6a388747…` (M1 Pro CPU); the network on its tokens at 20 and 50 from `reader-low` (`d4e39bae…`) and at 1,000 from `reader-1000` (`723cc57d…`), Kaggle T4 |

**Readings** (each the paired difference in area, second over first, over the same stays and
seeds):

| Name | Pair | Budgets |
|---|---|---|
| D1 | statistics alone (`summaries=only`) → trees per channel | 20, 50, 200, 500, 1,000, every stay |
| D2 | network on its tokens → statistics alone | 200, 500, every stay; 20, 50, 1,000 against the reused cells |
| D3 | statistics alone → statistics beside the encoder (`summaries=beside`) | 200, 500, every stay |
| Bar | network on its tokens → trees, and → MiniRocket | every budget |

At every stay the trees' variant chosen there is the bar; the trees at their setting in force are
read beside it. D1 and D2 are read with the statistics under the network's recipe; the two other
rates are read beside them and change no conclusion.

**Rule.** Paired bootstrap over stays in the two outcomes, 10,000 resamples, 95 %; a difference is
carried when its interval excludes zero and its mean exceeds twice its standard error over seeds.
No term for the spread between pretrainings: nothing here is pretrained. D1, D2 and D3 at 500
stays and every stay are one family under Holm (six); the other budgets are read descriptively.
The cells of the network on its tokens at 200, 500 and every stay run again on the G4; they are
compared with the registered T4 cells (`65c6a55e`, `8b477e7f`, `71d94bd4`) to the device's
precision and within 0.02.

**Predictions** (the edge of an earlier interval named as its source):

1. At every stay of the half the trees read between 0.787 (500 stays, `head-and-representation.md`,
   2026-10-20) and 0.861 (6,400 stays, `intensive-care-curve.md`, 2026-10-03), and lead the network
   on its tokens by at least 0.02, the lower edge at 500 stays (+0.038 [+0.024; +0.052]).
2. The statistics alone read no more than 0.01 below the logistic head over them (0.730 / 0.769 at
   200 / 500). D1 lies within the logistic head's own interval against the trees: [−0.007; +0.019]
   at 200 and [+0.004; +0.033] at 500.
3. D2 is carried at 200 stays and lies at no less than +0.059, the lower edge of the logistic head
   over the network (+0.077 [+0.059; +0.094]); at 500 it lies at no less than +0.002 (+0.020
   [+0.002; +0.037]) and is not carried after Holm.
4. D3's interval holds zero at 500 and every stay, and lies at or below zero at 200: states beside
   the summaries added nothing to them under a closed-form head (−0.014 [−0.029; +0.003] at 200,
   −0.010 [−0.023; +0.004] at 500).

**What follows from each outcome.**

- *Representation gap* — D2 carried at 500 or every stay, and D1 within 0.015 at every stay: the
  encoder fails to build what the table holds; the level of the statistics alone is the target of
  the next encoder's candidates.
- *A network over the table falls short* — D1 above 0.015 at every stay and carried: no network
  over these ten statistics reaches the trees. That is not a bound on every architecture, since an
  encoder may read more than the table holds, but the next candidates then need it; the reading is
  talked through before any is built.
- *The encoder adds something* — D3 carried above zero: the encoder holds what the table does not,
  and a pretext has something to build on.
- *Checks before reading* — the statistics alone at least 0.03 below the logistic head means the
  branch is suspect, and the G4 rerun of the network more than 0.02 from the T4 cells means the
  device enters the pairs; either stops the reading until it is explained.

**Limitations.** The branch draws its head wider before the encoder, so under `beside` the encoder
starts from another draw than on its tokens alone; D3 does not compare those two. The recipe of
the statistics alone was chosen for the encoder; its two other rates are descriptive. A trial ran
the exact path at 20 stays (classical and the statistics alone) and at 500 (all three networks),
seed 1, on the M1 Pro (MPS for the networks), printing counts of cells only: six cells of 2,000
stays each, the network cells at 500 in 104 and 113 s and the statistics alone in 4 to 13 s; the
stored descriptions name `summaries` where it is turned and nothing new where it is not.

## 2026-10-23 — M1 Pro CPU and Colab G4, fp32: the gap to the trees taken apart

**Question.** The one declared above: how much of the gap between the network trained from
nothing and the trees per channel is what a network makes of the trees' own table (D1), how much
is what the encoder fails to build from the readings (D2), and whether the encoder adds anything
once the table is handed to the head (D3).

**Conditions.** Campaigns defined and ordered at `82f6bab5`: `f7611c9f…` (20, 50), `9458a4d9…`
(200), `d1741c48…` (500), `a5192869…` (1,000), `105c9081…` (every stay). Classical cells on the
M1 Pro CPU (90 cells, 45 minutes); network cells on one Colab G4 under CUDA MPS, four lanes (160
cells, the longest lane 23 minutes). Reused cells as declared, except that the network at 1,000
stays (`723cc57d…`) borrows the shape of the mixture's second pretraining (`94f00d72…`), not the
first: the same architecture and parameter count, so the same initialisation under a seed. Pairs by
`scripts/campaign_pairs_report.py`, `data/report/l32b1/pairs.sh`; CSV under
`data/report/l32b1/pairs/`.

**Checks before reading.** Both held. The statistics alone read 0.706 and 0.754 at 200 and 500
stays, 0.024 and 0.015 below the logistic head over them, within the declared 0.03. The network on
its tokens rerun on the G4 reads within 0.006 of the T4 cells in its mean at 200, 500 and every
stay (−0.005, −0.001, +0.006). Single seeds differ by up to 0.098: training is not repeatable
across devices to their precision, only in its mean over seeds.

**Area under the ROC curve**, mean over seeds 1 to 10:

| Reading | 20 | 50 | 200 | 500 | 1,000 | every stay |
|---|---|---|---|---|---|---|
| Trees per channel | 0.621 | 0.628 | **0.736** | **0.787** | **0.818** | **0.839** |
| Statistics alone (`summaries=only`) | **0.644** | **0.674** | 0.706 | 0.754 | 0.788 | 0.810 |
| Statistics beside the encoder (`summaries=beside`) | | | 0.718 | 0.761 | | 0.817 |
| Network on its tokens | 0.599 | 0.625 | 0.647 | 0.749 | 0.789 | 0.796 |
| MiniRocket | 0.589 | 0.655 | 0.713 | 0.756 | 0.784 | 0.809 |

At every stay the trees are the variant chosen there (`learning_rate=0.1,rounds=300`); at their
setting in force they read 0.832. A logistic head over the summaries at every learnt stay of the
half, fitted once afterwards and not declared, reads 0.810.

**The pairs** (gain of the second over the first in area, 95 % interval over stays, seed by seed
mean ± SE; **bold** = carried and, at 500 and every stay, through Holm over six):

| Pair | 20 | 50 | 200 | 500 | 1,000 | every stay |
|---|---|---|---|---|---|---|
| D1 statistics alone → trees | −0.023 [−0.038; −0.009] ± 0.017 | **−0.046** [−0.061; −0.031] ± 0.009 | **+0.030** [+0.016; +0.044] ± 0.005 | **+0.033** [+0.019; +0.048] ± 0.007 | **+0.030** [+0.014; +0.047] ± 0.005 | **+0.030** [+0.011; +0.049] ± 0.002 |
| D2 network on its tokens → statistics alone | **+0.045** [+0.024; +0.066] ± 0.016 | **+0.049** [+0.027; +0.069] ± 0.013 | **+0.059** [+0.041; +0.076] ± 0.011 | +0.004 [−0.014; +0.022] ± 0.011 | −0.001 [−0.019; +0.016] ± 0.006 | +0.014 [−0.006; +0.032] ± 0.005 |
| D3 statistics alone → beside the encoder | | | **+0.012** [+0.004; +0.020] ± 0.003 | +0.007 [+0.001; +0.014] ± 0.006 | | **+0.007** [+0.002; +0.012] ± 0.002 |
| Network on its tokens → trees | +0.022 [+0.006; +0.039] | +0.003 [−0.014; +0.019] | **+0.089** [+0.073; +0.103] | **+0.038** [+0.024; +0.051] | **+0.029** [+0.014; +0.044] | **+0.043** [+0.025; +0.061] |
| Network on its tokens → MiniRocket | −0.010 [−0.030; +0.010] | +0.030 [+0.007; +0.053] | **+0.066** [+0.046; +0.085] | +0.007 [−0.011; +0.024] | −0.005 [−0.023; +0.012] | +0.013 [−0.006; +0.031] |

The statistics alone at three times and a third of the recipe's rate lie 0.026 to 0.039 below the
trees at 500 and every stay; no rate closes D1.

**Predictions.** 1 held: the trees read 0.839 at every stay of the half and lead the network on
its tokens by 0.043. 2 failed: the statistics alone read 0.014 and 0.005 below the logistic head
less 0.01, and D1 at 200 stays (+0.030) lies above the logistic head's interval. 3 held: D2 is
+0.059 at 200 stays, at the lower edge declared, and +0.004 at 500, not carried. 4 failed: D3 is
above zero at every budget it was read at, carried at 200 and every stay.

**Conclusions.**

1. **Below 500 stays the gap is representation.** At 20, 50 and 200 stays the network on its
   tokens loses 0.045 to 0.059 to a network handed the trees' table (D2). The same network over
   the table reads above the trees at 20 and 50 stays (by 0.023 and 0.046), where the claim of
   the project lives; only the lead at 50 is carried, and the trees there run a setting chosen at
   larger budgets.
2. **From 500 stays the encoder builds what the table holds.** D2 is 0.004, −0.001 and +0.014 at
   500, 1,000 and every stay, none carried. What remains to the trees there, 0.029 to 0.043, is
   what the trees make of the same table that a smooth model, as trained here, does not: a
   logistic head and a network over it both read 0.810 at every stay, 0.029 below the trees (D1).
   That is an upper bound on the table's cost to a smooth model, not a settled one (below).
3. **The encoder adds a little beyond the table** under end-to-end training: +0.007 to +0.012
   (D3). The probe over frozen states added nothing to the summaries (`head-and-representation.md`,
   2026-10-20); a trained encoder does.
4. By the declared rule this is the outcome *a network over the table falls short*: D1 is above
   0.015 at every stay and carried. An encoder that only recovers the table will not reach the
   registered parity at every stay; it needs to read more than ten statistics per channel hold.
5. MiniRocket is within reach of the network at every budget but 200; the trees are the bar.

**Limitations.** The statistics alone and beside the encoder ran under the network's recipe:
from 500 stays they learn from the four fifths of the draw the stop leaves, with no penalty on a
layer 256 wide, while the trees learn from all of it and the logistic head chooses its penalty.
Three rates do not close D1, but a head over the table trained on the whole draw under a penalty
could. The network on its
tokens at 20, 50 and 1,000 stays comes from T4 cells against G4 cells on the other side of D2.
The logistic head at every stay was fitted after the run and is descriptive.

## 2026-10-23 — declared before the run: is the table's distance to the trees its training?

**Question.** The network over the trees' table read 0.030 to 0.033 below the trees from 200
stays up (D1, above). From 500 stays it learnt from the four fifths of the draw its stop leaves,
with no penalty, while the trees learn from all of it and the logistic head over the same table
chooses its penalty; that head lies 0.006 and 0.018 below the trees at 200 and 500. Is D1 the
network's training, or what a smooth model makes of the table? The answer decides whether the
next encoder needs more than the table to reach the trees at every stay. And are the trees at
20 and 50 stays the bar they could be?

**Conditions.** As the section above, at `2d6c4b27`; every cell on the M1 Pro (the network over
the table on MPS, fp32; the trees on the CPU).

| Campaign | Budgets | Candidates |
|---|---|---|
| `gap-table-200` | 200 | the setting in force (no stop, every drawn stay) at a weight decay of 0.1 and of 1 |
| `gap-table-500` | 500 | no stop, every drawn stay, at the schedule's rate, decay 0, 0.1 and 1 |
| `gap-table-all` | every stay | no stop, every drawn stay, at 0.000333, decay 0, 0.1 and 1 |
| `gap-trees-low` | 20, 50 | trees `max_depth=2,rounds=50`, `learning_rate=0.1,max_depth=3`, `max_depth=3,min_leaf_weight=5` |

A trial ran one point of each family at seed 1 (500 stays and 20), printing counts of cells
only: three network cells in 56 s, three tree cells in 12 s.

**Readings.** D1′ at 200, 500 and every stay: the best of the network's settings over the table
(the new ones and the one read above) → the trees. The best of several chosen on the stays it is
scored on reads high; that is the direction against the conclusion it can support. At 20 and 50:
the best of the trees' four settings → the network over the table under its setting in force, and
→ the network on its tokens. Rule as above (interval and twice the standard error); no family, all
descriptive except D1′ at every stay, which is the one reading a decision rests on.

**Predictions.**

1. At 200 and 500 stays the best network over the table reads within 0.01 of the logistic head
   (0.730, 0.769; `head-and-representation.md`, 2026-10-23), and D1′ lies within that head's
   interval against the trees: [−0.007; +0.019] at 200, [+0.004; +0.033] at 500.
2. At every stay D1′ is carried and at least +0.02: the logistic head over the table, on every
   drawn stay with its chosen penalty, reads 0.810, 0.029 below the trees (one fit, descriptive).
3. At 50 stays the best trees still read below the network over the table, by no more than the
   upper edge of D1 there (0.031); at 20 the two cannot be told apart.

**What follows.** If D1′ at every stay is not carried, the table's distance to the trees is the
network's training, and an encoder that rebuilds the table under a better recipe can meet the
registered parity. If it is carried, a smooth reading of the table falls short of the trees there,
and parity at every stay needs an encoder that reads more than the table holds.
