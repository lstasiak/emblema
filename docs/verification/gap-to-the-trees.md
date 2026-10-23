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
