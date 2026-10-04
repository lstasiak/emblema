# Label-efficiency curve on the intensive-care task, preliminary and on the validation side

Purpose: read the label-efficiency claim on a second task, in-hospital death after a stay in
intensive care (PhysioNet/CinC Challenge 2012), by the rules registered before any of its numbers
existed (`docs/preregistration.md`, the intensive-care task). Sections are dated and appended.

**Every number here is validation, not test, and the result is preliminary.** The frozen side,
set C, is not opened.

## 2026-09-30 — both grids, Colab G4 under CUDA MPS and the M1 Pro

**Question.** Under the mixed backbone, does full fine-tuning at 200 labelled stays raise the
area under the ROC curve of the network trained from nothing by at least the registered least
gain, 0.045, with the whole interval above zero and above the floor? And what do the other ways
of using a backbone, the classical baselines and the patch model do at 50, 200, 1,000 and every
stay, under that backbone and under the backbone of the stays alone?

**Conditions.**

| | |
|---|---|
| Campaigns | `87cbc0e7…`, `campaigns/curve-physionet2012-mixed5.toml`, backbone `sha256:869ed545…`; `9129eabc…`, `campaigns/curve-physionet2012-stays.toml`, backbone `sha256:c5878c03…` |
| Code | orders placed at `933425fb` |
| Variants | each candidate at its selection's choice per budget, on the tuning side (`campaigns/selection-*-physionet2012.toml`, run 2026-09-29) |
| Sides | learnt from set A's stays, 50 to 3,997; scored on set B, 3,994 stays, 568 deaths, every cell |
| Seeds | 1 to 5 per cell; tier M |
| Rules | least gain 0.045, floor the larger of 0.01 and the control's SD over seeds, paired bootstrap over stays in two strata, 10,000 resamples, Holm over 35 and 15 |
| Machines | networks: one Colab G4, CUDA MPS, four processes, the longest order in 106 minutes; classical: MacBook Pro M1 Pro, CPU, 61 minutes |

    uv run scripts/campaign_report.py --campaign ID --out DIR --everything 3997
    uv run scripts/campaign_curve_figures.py DIR --figure PATH

![Mixed backbone](figures/intensive-care-curve-mixed5.png)

![Backbone of the stays alone](figures/intensive-care-curve-stays.png)

**The endpoint**, at 200 stays:

| Backbone | From nothing | Full fine-tuning | Gain [95 % interval] | Floor | Verdict |
|---|---|---|---|---|---|
| mixed (the claim) | 0.681 | 0.704 | +0.024 [+0.012; +0.036] | 0.040 | distinguishable, practically nil: **not confirmed** |
| stays alone | 0.693 | 0.706 | +0.013 [+0.001; +0.025] | 0.041 | distinguishable, practically nil |

**Mean area over five seeds**, mixed backbone (the ways of using the stays-alone backbone in
brackets; its control is its own selection's variant):

| Candidate | 50 | 200 | 1,000 | all |
|---|---|---|---|---|
| from nothing (control) | 0.65 (0.65) | 0.68 (0.69) | 0.78 (0.78) | 0.81 (0.81) |
| frozen probe | 0.64 (0.67) | 0.66 (0.75) | 0.76 (0.82) | 0.79 (0.84) |
| probe in closed form | 0.65 (0.66) | 0.70 (0.72) | 0.75 (0.79) | 0.78 (0.83) |
| LoRA | 0.64 (0.69) | 0.69 (0.71) | 0.77 (0.81) | 0.79 (0.83) |
| full fine-tuning | 0.67 (0.71) | 0.70 (0.71) | 0.77 (0.79) | 0.81 (0.80) |
| trees per channel | 0.66 | 0.76 | 0.83 | 0.86 |
| trees over the spectrum | 0.56 | 0.60 | 0.66 | 0.68 |
| trees across channels | 0.56 | 0.61 | 0.67 | 0.70 |
| MiniRocket | 0.71 | 0.76 | 0.82 | 0.85 |
| patch model | 0.69 | 0.72 | 0.79 | 0.83 |

Under the mixed backbone 9 of 32 secondary comparisons are distinguishable, none of them a way of
using the backbone; under the stays alone 6 of 12, with the frozen probe above the floor at 200,
1,000 and every stay, LoRA at 1,000 and every stay, and the closed-form probe at every stay.

**Conclusions.**

1. The endpoint is not confirmed under either backbone. Full fine-tuning's gain at 200 is
   distinguishable from zero but below both the floor and the least gain. Under the mixed
   backbone it clears no floor at any budget; under the stays alone it clears the floor at 50
   and 1,000, with gains of 0.061 and 0.015, and not at every stay.
2. Under the mixed backbone no way of using it beats the network from nothing at any budget; the
   closed-form probe is worse at 1,000 and every stay. Under the stays alone the frozen probe
   gains 0.055, 0.047 and 0.031 in area at 200, 1,000 and every stay. The mixture, which learnt
   the stays about as well as eight epochs on them alone (`manual-handoff.md`, 2026-09-29), is
   the weaker backbone for this task. This comparison was named a diagnostic in the registration
   but given no prediction beforehand, so it is descriptive only.
3. From 200 stays up a classical baseline has the highest mean area of any candidate:
   MiniRocket at 200 (0.765), the trees per channel at 1,000 and every stay (0.829 and 0.856).
   The nearest network is the stays-alone probe, 0.824 at 1,000, above MiniRocket's 0.818. At
   50 the stays-alone full fine-tuning and MiniRocket are level, 0.712 and 0.711. The two
   campaigns are not paired against each other, so these are means, not tests. The patch
   model's mean is above every pretrained arm's at 50 and 200 under the mixed backbone.
4. One seed's area moves by about 0.04 at 200 stays; the floor at 200 is therefore 0.040 and
   0.041, above the gain of every pretrained arm but the stays-alone probe.

**Limitations.**

- Learnt from set A's 3,997 stays and scored on set B. Published benchmarks on this corpus
  (P12) usually learn from about 9,600 of 11,988 stays with other splits, and report 0.85–0.86
  for their best networks; the levels here are not comparable with them.
- Two possible handicaps of the networks, not measured: the selections of the network from
  nothing and of full fine-tuning chose the edge of their rate grid at several budgets (÷3 at
  1,000 and every stay, ×3 at 200 under the mixed backbone), so the best rate may lie beyond it;
  and the floor of 2,000 optimiser steps, 160 epochs at 200 stays and 640 at 50 with no early
  stop, was confirmed on the turbofans only. Both touch the control and the endpoint alike.
- The two campaigns' controls differ, each at its own selection's variant, so a cell of one is
  not the same cell of the other.
- Networks ran on a G4, the classical baselines on the M1 Pro's processor; every cell of one
  budget of one pool ran on one kind of machine.

## 2026-09-30 — the cost of laying stays on a grid

**Question.** MiniRocket and the patch model can only read a regular grid, so an irregular stay
is resampled for them. Does that cost them area against the network that reads the raw readings,
the one trained from nothing?

**Conditions.**

| | |
|---|---|
| Campaign | `87cbc0e7…`, mixed backbone, the section above; no new run |
| Grid | one step an hour, 49 steps over 48 hours and a minute (MiniRocket at every stay: two steps an hour, its selection's choice); a step with no reading carries the channel's last one forward, a channel not yet seen carries zero, the corpus mean; every channel doubled by a mask of the steps really observed; statics held at every step |
| Patch model | 192 wide, 4 blocks, patches of 8 steps at a stride of 4, dropout 0.2, mean pooling; rate 0.001, 0.003 at 1,000 |
| Network on raw readings | the backbone's shape, 256 wide, 6 blocks, dropout 0; mean pooling; rate 0.001 at 50, 0.003 at 200, 0.000333 at 1,000 and every stay |
| Schedule | both networks from nothing: 30 epochs of 16 stays and at least 2,000 steps |
| Rules | the registered secondary comparisons, paired over set B's 3,994 stays, Holm over 35 |

**Gain in area over the network on raw readings**, mean over five seeds:

| Stays | Raw readings | Patch model [95 % interval] | MiniRocket [95 % interval] |
|---|---|---|---|
| 50 | 0.652 | 0.694: +0.042 [+0.027; +0.057] | 0.711: +0.060 [+0.044; +0.075] |
| 200 | 0.681 | 0.722: +0.041 [+0.028; +0.055] | 0.765: +0.084 [+0.072; +0.097] |
| 1,000 | 0.775 | 0.791: +0.016 [+0.003; +0.028], n.s. | 0.818: +0.043 [+0.029; +0.057] |
| all | 0.809 | 0.826: +0.017 [+0.004; +0.029], n.s. | 0.850: +0.041 [+0.029; +0.053] |

n.s.: not distinguishable under the family's correction; every other cell is.

**Conclusions.**

1. Measured as the gap to the network on raw readings, the grid costs the methods that need it
   nothing: both lead that network at every budget, and six of the eight gains are
   distinguishable.
2. The patch model is the closer comparison: a transformer trained from nothing on the same
   schedule. It leads by about 0.04 at 50 and 200 stays and by under 0.02, not distinguishable,
   from 1,000 up.
3. The raw readings do not give the network an advantage at these budgets. Whether they cost it
   one is not settled here (limitations).

**Limitations.**

- The patch model differs from the network on raw readings in more than its input: it is
  narrower and shallower, reads patches of eight hours rather than single readings, and is
  regularised by dropout 0.2 against none. The gap is the cost of the whole design, not of the
  grid alone. Only the same network fed the gridded readings as tokens would isolate the grid.
- The mask tells the grid methods where a reading was made. The network on raw readings learns
  the same from which readings its tokens hold, so the mask gives the grid no information the
  other side lacks.
- One backbone's campaign; the stays-alone campaign holds no grid method.

## 2026-10-01 — declared before the run: the task at the size published results learn from

**Question.** The network trained from nothing reaches 0.81 in area at every stay of set A
(3,997 stays), while the trees per channel reach 0.856. The best published networks for in-hospital
death reach 0.83–0.86 learning from about 2,600 to 7,700 stays. Is the network's deficit a matter of
how many stays it learns from, or of the network and how it is trained?

**What the published results learn from**, read in each paper:

| Source | Stays learnt from | Scored on | Networks' area |
|---|---|---|---|
| Shukla and Marlin, ICLR 2021 (mTAN), Table 2 | about 2,560, set A | 20 % of set A | 0.76–0.86; mTAND-Enc 0.854 |
| Tipirneni and Reddy, TKDD 2022 (STraTS), Tables 4–5 | about 3,200: half of the 80 % of sets B and C it learns from | set A | 0.80–0.84; without its pretraining 0.835 |
| Horn et al., ICML 2020 (SeFT), Table 1 | about 7,700: 64 % of all three sets, by its appendix | about 2,400 | 0.79–0.86; GRU-D and Transformer 0.863 |
| Johnson and Mark, AMIA 2017, on the challenge's winner | set A, 4,000 | set C | trees in a Bayesian ensemble, 0.860 |

Each of the three papers on networks keeps the model of the best validation epoch. STraTS, the recurrent baselines it
runs and SeFT use dropout of 0.2 or more, and STraTS is 32 to 50 wide with two blocks, far
smaller than this project's backbone shape (256 wide, six blocks, no dropout).

**Design.** `campaigns/reproduction-physionet2012.toml`, campaign `af0cae4a…`, tier M. A version of
the corpus published from sets A and B with one stay in five held out by seed 1 (manifest
`sha256:d5d0947a…`): 6,400 stays to learn from, 1,600 to score on, normalisation fitted on the
6,400; stays, windows, tokens and channels are those of the version the grids read (manifest
`sha256:cef44de2…`), only the division and the statistics differ. Set C stays frozen. Three
candidates at every stay, at the variants their selections chose at every stay in the registered
grids: the network from nothing at a rate of 0.000333, the trees per channel, MiniRocket at two
steps an hour. Five seeds; the network's cells through an order on a Colab GPU, the classical
cells on the M1 Pro's processor. Read as means and paired intervals by
`scripts/campaign_report.py`; no verdict is drawn. Each candidate's description, its compute
budget included, is the one the mixed backbone's grid recorded. A first definition, `0cb127f5…`,
was made without the grids' worker settings and recorded MiniRocket's published penalties
rather than the registered fifteen; it was never ordered.

**Predictions.**

1. The trees per channel reach 0.86 or more, and MiniRocket within 0.01 of them.
2. If the network's deficit is the number of stays, it reaches 0.84 or more at 6,400, the level
   of published networks at 3,200 to 7,700 stays, and its gap to the trees is under 0.02.
3. If it is the network or its training, it stays below 0.83 and the gap to the trees stays near
   0.04, as at 3,997 stays.

The second outcome is expected less: mTAN reaches 0.85 from fewer stays than set A holds.

**Reading.** The second outcome points the remaining diagnostics at the size of the labelled
side; the third points them at the network: dropout, the gridded input and the floor of steps,
each declared before its own run. A result between the two is reported as such. No verdict of
the registered grids changes.

## 2026-10-01 — Colab G4 and the M1 Pro: the task at the size published results learn from

**Question.** The one declared above: at 6,400 stays from sets A and B, does the network trained
from nothing reach the level of published networks, so that its deficit at 3,997 stays is a matter
of size, or does it stay where it was?

**Conditions.**

| | |
|---|---|
| Campaign | `af0cae4a…`, `campaigns/reproduction-physionet2012.toml`; task over manifest `sha256:d5d0947a…` |
| Code | orders placed at `a6a777fd` |
| Sides | learnt from 6,400 stays of sets A and B; scored on the 1,599 of the other 1,600 that hold a window, 213 deaths |
| Seeds | 1 to 5; tier M; no verdict drawn |
| Machines | network: one Colab G4, 3.6 min a cell; classical: MacBook Pro M1 Pro, CPU, 7 s a cell for the trees, 23 min for MiniRocket |

    uv run scripts/campaign_report.py --campaign af0cae4a-1026-45a4-8d53-57a9e17f9e44 \
        --out DIR --everything 6400

**Area under the ROC curve**, mean ± SD over five seeds, and the paired gain over the network
(95 % interval, 10,000 resamples over stays):

| Candidate | Area | Gain over the network |
|---|---|---|
| network from nothing, rate 0.000333 | 0.799 ± 0.009 | — |
| trees per channel | 0.861 ± 0.000 | +0.062 [+0.043; +0.081] |
| MiniRocket, two steps an hour | 0.824 ± 0.003 | +0.025 [+0.006; +0.046] |

**The same candidates at 3,997 stays** (the mixed backbone's grid above, scored on set B; not
paired with this run): network 0.809, trees 0.856, MiniRocket 0.850.

**Conclusions.**

1. The third outcome declared holds: the network stays below 0.83 and trails the trees by 0.062,
   more than at 3,997 stays. The second, a deficit of size, is rejected.
2. Published networks reach 0.83–0.86 learning from 3,200 to 7,700 stays; this network reaches
   0.80 at 6,400. Its deficit lies in the network or how it is trained, which the remaining
   diagnostics take up: dropout, the gridded input and the floor of steps.
3. The trees per channel reach the level of the challenge's winner (0.860) from 4,000 stays and
   gain nothing measurable from 6,400; neither does the network.
4. The first prediction holds for the trees, not for MiniRocket: at 0.824 it is 0.037 below them
   and 0.026 below its own area at 3,997 stays, far beyond the spread of its seeds. Not explained
   here; its variant was selected on the smaller labelled side.

**Limitations.**

- Learnt and scored on a division of sets A and B that the registered grids do not use; 3,209 of
  the 6,400 stays are set B's, the registered grids' validation side. No model fitted here is used
  by them.
- The comparisons with 3,997 stays are across scored sides and not paired; a difference of 0.01
  between them is within what another scored side moves an area by.
- Each candidate runs at the variant selected at every stay of set A; nothing was selected again
  at 6,400 stays.
- 1,599 stays scored and 213 deaths: the paired intervals above are about ±0.02 wide.

## 2026-10-01 — declared before the run: dropout and the gridded input

**Question.** At 3,997 and 6,400 stays the network trained from nothing reaches about 0.80 in area,
where published networks reach 0.83–0.86 from 3,200 to 7,700 stays (the two sections above). Those
networks drop a fifth of their activations or more; this one drops none. The patch model, which
led it by 0.041 at 200 stays on the validation side, both drops a fifth and reads its readings on
a grid of one step an hour. Does the same network gain from either knob, and how much of the gap
does each close?

**Design.** Two campaigns on the registered task, each scored on the fifth of the tuning stays held
out by seed 101 for every seed, about 800 stays, the fifth the spread over seeds was read on; the
validation side is not read.

| | `campaigns/grid-cost-all-physionet2012.toml` | `campaigns/grid-cost-200-physionet2012.toml` |
|---|---|---|
| Campaign | `f8c5225b…` | `4e28a197…` |
| Stays learnt from | every tuning stay outside the fifth | 200 |
| Network's rate | 0.000333 | 0.003 |
| Seeds | 1 to 5 | 1 to 10 |

Five candidates in each: the network from nothing at the rate the registered grid ran it at that
budget; the same network fed the readings laid on one step an hour, a token per channel and step
from the channel's first reading on, its gap the time since a real reading; the same network under
a dropout of 0.2; the same network under both; the patch model at its registered setting. Each is
described in its campaign exactly as the registered grid describes it, the turned knobs aside. 50
stays are left out: one seed moves the area there by 0.057 on this fifth, so even ten seeds would
read a difference of two candidates only to about 0.03. A first definition, `d4d5e7a4…`, ran every
network at 0.001 and was never ordered: at every stay that rate is not the one the network's
selections chose, and a gain under dropout could then be a correction of the rate.

**Reading.** Each candidate against the network from nothing, by `scripts/campaign_pairs_report.py`:
the gain in area paired over the stays, two strata, 10,000 resamples, 95 % interval; and beside
it the gain seed by seed, each seed's two answers drawn from the same labels, with its mean and
standard error over the seeds. A knob *gains* where the interval lies above zero and the mean over
seeds exceeds twice its standard error; the interval alone cannot see how far another seed moves
either side. At 200 stays a knob *closes the patch model's lead* where it gains and its gain is at
least half the patch model's.

**Predictions.**

1. At every stay, the dropout gains at least 0.015, half the gap between this network and the
   published ones at a similar size.
2. At every stay and at 200, the grid alone moves the area by less than 0.01 either way: published
   networks reach the same level on hourly aggregates and on raw readings.
3. At both budgets, the two knobs together land within 0.01 of the dropout alone.
4. At 200 stays the patch model gains 0.02 or more, and the dropout closes its lead; at every stay
   the patch model's gain is under 0.02, as on the validation side.

**What follows.** If the dropout gains, the networks of the next backbones learn under a dropout a
selection chooses, and the gap that remains is read against the shape. If the grid gains, the
representation of irregular readings is reopened before any backbone is retrained. If neither
does, a narrower network from nothing is the next candidate. No verdict of the registered grids
changes.

## 2026-10-01 — Colab G4: dropout and the gridded input

**Question.** The one declared above: does the network from nothing gain from a dropout of 0.2,
from readings laid on one step an hour, or from both, at every stay and at 200, and does either
close the patch model's lead at 200?

**Conditions.**

| | |
|---|---|
| Campaigns | `f8c5225b…` (every stay, seeds 1–5) and `4e28a197…` (200 stays, seeds 1–10), both selections |
| Code | orders placed at `477bbe2b` |
| Scored | the fifth of the tuning stays held out by seed 101: 800 stays, 127 deaths, the same for every cell |
| Machine | one Colab G4, two orders at once under CUDA MPS: 79 min for the 50 cells at 200, 103 min for the 25 at every stay; tier M; no verdict drawn |

    uv run scripts/campaign_pairs_report.py --campaign f8c5225b-872f-4c7a-97a7-5b0861827486 \
        --campaign 4e28a197-7eee-4355-b272-4a786ac0f2c8 --out DIR
    uv run scripts/campaign_pairs_report.py --out DIR \
        --pair CONTROL_CAMPAIGN CONTROL BUDGET CANDIDATE_CAMPAIGN CANDIDATE BUDGET [...]

**Area under the ROC curve**, mean over seeds (seed-by-seed values in the CSV):

| Candidate | Every stay | 200 stays |
|---|---|---|
| network from nothing | 0.786 | 0.652 |
| … under a dropout of 0.2 | 0.782 | 0.654 |
| … on the hourly grid | 0.797 | 0.664 |
| … under both | 0.798 | 0.665 |
| patch model | 0.796 | 0.696 |

**Gain in area over the network from nothing**: paired over stays (95 % interval), and seed by
seed (mean ± standard error). *Gains* marks where both conditions declared hold.

| Candidate | Budget | Paired gain | Seed by seed | Gains |
|---|---|---|---|---|
| dropout 0.2 | every stay | −0.005 [−0.022; +0.012] | −0.005 ± 0.005 | no |
| dropout 0.2 | 200 | +0.002 [−0.017; +0.020] | +0.002 ± 0.017 | no |
| hourly grid | every stay | +0.010 [−0.014; +0.036] | +0.010 ± 0.008 | no |
| hourly grid | 200 | +0.012 [−0.010; +0.033] | +0.012 ± 0.010 | no |
| both | every stay | +0.012 [−0.012; +0.036] | +0.012 ± 0.004 | no |
| both | 200 | +0.013 [−0.008; +0.033] | +0.013 ± 0.013 | no |
| patch model | every stay | +0.010 [−0.019; +0.038] | +0.010 ± 0.010 | no |
| patch model | 200 | +0.043 [+0.018; +0.069] | +0.043 ± 0.017 | yes |

Against the network under dropout, the grid added on top gains +0.017 [−0.007; +0.041] at every
stay and +0.011 [−0.009; +0.032] at 200, and the patch model at 200 gains +0.042 [+0.017; +0.067].

**Conclusions.**

1. Prediction 1 fails: the dropout gains nothing at either budget. Its estimate at every stay is
   slightly below zero, and its interval excludes the declared gain of 0.015. This network's gap
   to the published ones is not the dropout they use.
2. Prediction 2 is not decided. The grid's estimate is +0.010 to +0.017 in all four comparisons
   that add it, at the declared limit of 0.01 or above it, but no single comparison gains. With
   the grid, the network ties the patch model at every stay (0.797 against 0.796). The grid's
   effect, if it has one, is about 0.01, too small to explain a gap of 0.04–0.06.
3. Prediction 3 fails narrowly, and only because the dropout adds nothing: both knobs together
   land on the grid alone (within 0.001), not on the dropout alone.
4. Prediction 4 holds for the patch model and fails for the dropout. The patch model's lead at
   200 is reproduced (+0.043, against +0.041 on the validation side), and at every stay it is
   under 0.02. Neither knob closes the lead: the gridded network still trails the patch model at
   200 by about 0.03. Dropout and the grid together account for at most a third of the lead,
   which lies in how the patch model reads a window.
5. As declared for a result where neither knob gains, a narrower network from nothing is the next
   candidate. The learning rate's edge and the floor of steps follow first because they are
   cheaper. No verdict of the registered grids changes.

**Limitations.**

- 800 stays with 127 deaths. At every stay the paired interval is about ±0.025, wider than the
  spread of the seeds, so a gain of 0.01 cannot be confirmed there by the first condition,
  whatever the number of seeds.
- One dropout, 0.2, applied wherever the encoder learns; the rate was not selected again under it.
- Learnt from about 3,200 tuning stays at every stay, the size of the published fixed split, not
  from the registered 3,997.

## 2026-10-01 — declared before the run: a published network on the same stays

**Question.** Is the network's deficit in the data it is given or in the network? Neither the
size of the learning side, nor dropout, nor the grid explains why the network from nothing
reaches 0.786 where STraTS (Tipirneni and Reddy, TKDD 2022) reports 0.835 from about 3,200
stays. STraTS reads stays as triplets of time, variable and value, much as this project's tokens
do. Its official code, run on the same stays, separates the two.

**Design.** STraTS's PyTorch code at commit `e936cda`
(https://github.com/sindhura97/STraTS), in the setting its run script gives this corpus
without self-supervision: 64 wide, 2 blocks, 16 heads, dropout 0.2, rate 5e-4, batch 16, a
positive class weighted by the ratio of the classes, early stop on the sum of the areas under
the ROC and the precision-recall curves, patience 10, at most 50 epochs. Its dataset, model and
evaluator are imported unchanged. Its training loop is restated step for step in
`scripts/strats_reference_run.py`.

| | |
|---|---|
| Learnt from | 2,560 of the 3,200 tuning stays outside the scored fifth, three of them without a reading; the other 640 drive the early stop |
| Scored | the fifth held out by seed 101: 800 stays, 127 deaths, the stays every campaign above scored |
| Preparation (1) | its own, from the challenge's files: negative values dropped, repeated rows dropped, the ward as four variables |
| Preparation (2) | this project's tokens of the same stays (`scripts/strats_reference_data.py`) |
| Seeds | 1 to 5 for each |

Before the run, the two preparations were compared reading by reading. They hold the same
readings, apart from 455 readings at 48:00 that the published window of 48 hours leaves out and
24 negative temperatures that STraTS drops, of 1.75 million. After the tokens' standardisation is
undone, every value both hold agrees to within 1e-3. Its network standardises each variable again on its learning
side, so (1) and (2) differ by those readings alone.

**Reading.** Each preparation against the network from nothing at every stay (`f8c5225b…`,
0.786) and (2) against (1), by `scripts/campaign_pairs_report.py` after
`scripts/strats_reference_answers.py` has put STraTS's answers beside the campaign's. The rule is
the one used above: a side *gains* where the paired interval over stays lies above zero and the
mean over seeds exceeds twice its standard error.

**Predictions.**

1. On its own preparation STraTS reaches at least 0.83 and gains over the network from nothing.
2. On this project's tokens it lands within 0.01 of its own preparation.

**What follows.** If both hold, this project's data are as good as the published preparation of
them, and the deficit lies in the network or in how it is trained. Candidates are, in order: how
the static features enter, how the states are pooled, how a value is embedded, the early stop,
and the weighting of the classes. They are tried on the network from nothing before any backbone
is retrained. If (2) falls short of (1) by 0.02 or more, the preparation loses something, and
the data are revisited first. If (1) stays below 0.81, these stays are harder than the published
split, and STraTS's area here replaces 0.835 as the reference.

## 2026-10-01 — Colab G4: a published network on the same stays

**Question.** Does STraTS, run on the stays the network was scored on, reach the published area,
and does it lose anything when fed this project's tokens instead of its own preparation? The
design and predictions are the section above.

**Conditions.** Commit `9af2e1c0`, STraTS at `e936cda`. One NVIDIA RTX PRO 6000 Blackwell (Colab
G4), torch 2.11.0+cu130, CUDA, fp32, five runs at a time; tier M. The split file hashes the same
on Colab and on the Mac (`787648be5e3adeb3`). Every run learnt from 2,557 stays, stopped on 640,
weighted the positive class by 6.61 and stopped ten epochs after its best one: 4.5 to 10
minutes of training a run, none of them at the cap of 50 epochs. Commands:

    python scripts/strats_reference_data.py \
        --manifest durable/sha256/cef44de2... sha256:cef44de2... \
        --workspace data/workspace --corpora data/raw --out DATA
    python scripts/strats_reference_run.py --strats STRATS --seed N \
        --data DATA/PREPARATION.pkl --out RUN --device cuda
    uv run scripts/strats_reference_answers.py --into DIR --side strats-e936cda PREPARATION \
        --purpose selection --runs RUN_1 ... RUN_5
    uv run scripts/campaign_pairs_report.py --out DIR --pair f8c5225b... \
        from_scratch@learning_rate=0.000333 all strats-e936cda PREPARATION all

**Area under the ROC curve on the 800 scored stays**, mean over seeds 1 to 5, and the range of
the seeds.

| Candidate | Mean | Seeds |
|---|---|---|
| network from nothing (`f8c5225b…`) | 0.786 | 0.770–0.797 |
| STraTS, its own preparation (1) | 0.822 | 0.820–0.828 |
| STraTS, this project's tokens (2) | 0.827 | 0.814–0.842 |

**Gain in area**, paired over stays (95 % interval), and seed by seed (mean ± standard error).

| Control | Candidate | Paired gain | Seed by seed | Gains |
|---|---|---|---|---|
| network from nothing | STraTS (1) | +0.036 [+0.011; +0.062] | +0.036 ± 0.004 | yes |
| network from nothing | STraTS (2) | +0.040 [+0.016; +0.066] | +0.040 ± 0.007 | yes |
| STraTS (1) | STraTS (2) | +0.004 [−0.004; +0.013] | +0.004 ± 0.003 | no |

**Conclusions.**

1. Prediction 1 fails on its threshold and holds on its gain. STraTS on its own preparation
   reaches 0.822, short of 0.83, and gains over the network from nothing by 0.036. It learnt from
   2,557 stays where the published run learnt from about 3,200. It stays above 0.81, so 0.835
   stays the published reference, and 0.82 is the reference on these stays.
2. Prediction 2 holds. On this project's tokens STraTS lands 0.004 above its own preparation,
   within the declared 0.01, and its interval excludes a loss of 0.01. The tokens carry what the
   published preparation carries.
3. Every seed of STraTS, on either preparation, scores above every seed of the network. Trained
   on the same tokens, with fewer stays to learn from, a published network gains about 0.04 over
   this one on the same stays.
4. Only the threshold of prediction 1 fails, on fewer stays than the published run, so the
   branch declared for both predictions holding is followed: the deficit lies in the network or
   how it is trained, not in the data. Next, the network from nothing tries, one at a time: how
   the static features enter, how the states are pooled, how a value is embedded, the early stop
   and the weighting of the classes. No verdict of the registered grids changes.

**Limitations.**

- 800 stays with 127 deaths: the paired interval is about ±0.025, so (2) against (1) is read
  only to about 0.01.
- One setting of STraTS, the one its run script gives this corpus; it was not tuned here.
- STraTS stopped on 640 of the tuning stays; the network learnt from all 3,200 without a stop.
  Their difference in what they learnt from favours the network.

## 2026-10-02 — declared before the run: the static features set apart

**Question.** Does the network from nothing gain when its static features are read apart from
its readings, and does a learnt pooling add to that? On the same stays, STraTS gains about 0.04
over it from the same tokens (the section above). Of what STraTS does differently, the static
features are tried first. STraTS reads a stay's demographics on a path of their own, beside its
pooled readings. This network pools its static features, four at most, with some four hundred
readings, so each weighs as one reading among them.

**Design.** Campaign `e9e0aac5…` (`campaigns/statics-apart-all-physionet2012.toml`) has the same
fifth, seeds, budget, schedule and description of the control as `f8c5225b…`.

| Candidate | Pooling | Static features |
|---|---|---|
| network from nothing (control) | mean | among the readings |
| … static features apart | mean | apart |
| … attention | learnt attention | among the readings |
| … both | learnt attention | apart |

*Apart* means the encoder is unchanged. Its states of the readings are pooled by the scheme, and
its states of the static features are averaged on their own. The head reads the two side by side
(`StaticsApartPooling`), 512 wide instead of 256. The attention pooling starts as the mean and
learns which states to weigh. The selections of this network never chose it, at any budget,
under either backbone (`campaigns/selection-scratch-physionet2012.toml`), with the static
features among the readings.

**Reading.** As above: each candidate against the control, and *both* against the static
features apart, by `scripts/campaign_pairs_report.py`. A candidate *gains* where the paired
interval over stays lies above zero and the mean over seeds exceeds twice its standard error.
Each candidate is also paired with STraTS on this project's tokens, to measure what is left of
its lead. The control reruns `f8c5225b…`'s cells on the same code path, so its area is a check:
it lands within 0.01 of 0.786.

**Predictions.**

1. The static features apart gain over the control. On this fifth that takes a gain of about
   0.025 or more.
2. The attention alone does not gain.
3. Both together land within 0.01 of the static features apart.

**What follows.** If the static features apart gain, the placement is a candidate for the whole
matrix: it is a knob of the head, so a pretrained backbone takes it without being trained again.
The next element is how a value is embedded. If they do not gain, they explain at most a part of
the gap smaller than this fifth can confirm, and the value's embedding follows at once. Neither
outcome changes a verdict of the registered grids.

**Limitations.**

- The static features are averaged after the encoder. The value of every token enters through
  one projection shared by all channels, so age and height are told apart only by what the
  encoder makes of their channels. STraTS feeds the raw demographics to a network of their own.
- At every stay the paired interval is about ±0.025, so a gain smaller than that is not
  confirmed, however consistent the seeds. A smaller one shows as an estimate whose interval
  holds zero, as the grid's did above.

## 2026-10-02 — declared before the run: the mixture without SMD

**Question.** Does SMD teach the mixed backbone anything the intensive-care task uses? In the
mixture of five, SMD takes 40 % of the steps. At 1,900 tokens a window it takes 59 % of the
tokens and about three quarters of the attention. Its validation stays at 0.35 to 0.40 of the
trivial predictor's loss from the first epoch to the eighth (`manual-handoff.md`, 2026-09-29).
In its training halves, 18 % of the channel series are constant throughout, and a channel is
constant across 26 % of the windows it is read in. The stays get 3 % of the steps.

**Design.** Two backbones, each against the mixture of five (`869ed545…`):

| Backbone | Experiment | Corpora | Epochs | Steps | Steps on the stays |
|---|---|---|---|---|---|
| mixture of five | `backbone-mixed5-m` | all five | 8 | 36,950 | 1,000 |
| A, without SMD | `backbone-mixed5-nosmd-m` | without SMD | 8 | 22,340 | 1,000 |
| B, without SMD, the mixture's steps | `backbone-mixed5-nosmd-m-13` | without SMD | 13 | 36,310 | 1,625 |

Everything else is the mixture's: manifests, vocabulary, shape, masks, objective, rate and seed.
B's warm-up keeps the mixture's share of the run. A asks whether SMD's steps teach the other
corpora anything at equal exposure. B asks whether those steps are better spent on the rest,
which then is seen more often; adding steps adds no data. Each backbone is trained once, on a
Kaggle T4.

**Reading.**

- *Pretext.* Each corpus's validation loss as a share of the trivial predictor's, at the kept
  epoch, against the mixture's.
- *Task.* Campaign `campaigns/mixture-transfer-200-physionet2012.toml`, defined once under each
  backbone (`4d17e289…` under the mixture). The frozen probe and full fine-tuning at 200 stays,
  at the variants the mixture's selections chose, seeds 1 to 10, on the fifth held out by seed
  101. Each backbone is paired with the mixture by `scripts/campaign_pairs_report.py`, and B with
  A, by the rule used above.
- One seed moves the area at 200 stays by about 0.04 on this fifth, so the seeds read a
  difference of two backbones to about 0.013. Each backbone is one pretraining run, whose own
  spread is not known. A difference is acted on only where it gains by both conditions and by
  0.03 or more.
- Check: the mixture's fine-tuning repeats the cells of `596849cd…` (0.692), so it lands within
  0.01 of that.

**Predictions.**

1. Pretext: A's loss on the stays lies within 0.02 of the mixture's 0.487. B's lies at least
   0.02 below A's, as the stays alone fell from 0.478 to 0.430 between 8 and 16 epochs.
2. Task, A against the mixture: neither the probe nor fine-tuning differs. SMD teaches the
   stays nothing.
3. Task, B against A: the probe gains and fine-tuning does not. The stays-alone backbone's probe
   led the mixture's by 0.09 at 200 stays, and B sees the stays 1.6 times as often.

**What follows.** If A holds the mixture's areas, SMD leaves the mixture of the next backbones,
or returns republished without its constant channels: three quarters of the arithmetic, for
nothing the task uses. If the mixture gains over A, SMD helps the transfer despite its own
plateau, and it stays, cleaned. If B gains over A, how often a corpus is seen matters, and the
next mixtures need a weight per corpus, which they do not have now (a corpus weighs its share
of the windows). No verdict of the registered grids changes.

**Limitations.**

- One pretraining run per backbone: a difference under about 0.03 cannot be told from the
  run's own spread.
- One task. C-MAPSS is read under another version of the corpus (per condition) than the one
  the mixture learnt from, so the turbofan task is left to the next mixtures.
- SMD's channel rows stay in the table of A and B, untrained, so all three backbones have the
  same parameters. The intensive-care task does not read those rows.

## 2026-10-02 — Colab G4: the static features set apart

**Question.** Does setting the static features apart, or pooling under attention, gain the
network from nothing anything? The design and predictions are in the section declared on
2026-10-02 for the static features.

**Conditions.** Commit `4691e274`, campaign `e9e0aac5…`, order `e61fac82…`. One NVIDIA RTX PRO
6000 Blackwell (Colab G4), torch 2.11.0+cu130, CUDA, one process; tier M. 20 cells in 2,005 s,
about 100 s a cell. Read with:

    uv run scripts/campaign_pairs_report.py --out DIR \
        --campaign e9e0aac5... --campaign f8c5225b...
    uv run scripts/strats_reference_answers.py --into DIR --side strats-e936cda tokens ...
    uv run scripts/campaign_pairs_report.py --out DIR --pair CONTROL ... CANDIDATE ...

**Area under the ROC curve on the 800 scored stays**, seeds 1 to 5:

| Candidate | Mean | Seeds |
|---|---|---|
| network from nothing (control) | 0.782 | 0.754–0.804 |
| … static features apart | 0.769 | 0.740–0.803 |
| … attention | 0.786 | 0.769–0.798 |
| … both | 0.765 | 0.749–0.785 |
| STraTS on this project's tokens | 0.827 | 0.814–0.842 |

**Gain in area**, paired over stays (95 % interval), and seed by seed (mean ± standard error).

| Control | Candidate | Paired gain | Seed by seed | Gains |
|---|---|---|---|---|
| network from nothing | static features apart | −0.013 [−0.033; +0.008] | −0.013 ± 0.011 | no |
| network from nothing | attention | +0.005 [−0.008; +0.017] | +0.005 ± 0.012 | no |
| network from nothing | both | −0.017 [−0.036; +0.003] | −0.017 ± 0.010 | no |
| static features apart | both | −0.004 [−0.021; +0.013] | −0.004 ± 0.015 | no |
| static features apart | STraTS | +0.058 [+0.033; +0.084] | +0.058 ± 0.010 | yes |
| both | STraTS | +0.062 [+0.037; +0.087] | +0.062 ± 0.009 | yes |
| control of `f8c5225b…` | control here | −0.004 [−0.020; +0.011] | −0.004 ± 0.009 | no |

**Conclusions.**

1. Prediction 1 fails. Set apart, the static features gain nothing; the estimate is below zero,
   and its interval excludes a gain of 0.01. Set apart this way, the static features close none
   of the gap to STraTS.
2. Prediction 2 holds: the attention alone gains nothing, as its selections said.
3. Prediction 3 holds: both together land 0.004 from the static features apart.
4. The check holds on the mean: the control lands at 0.782 against 0.786. Seed by seed it does
   not repeat its earlier cells: seed 4 moves from 0.788 to 0.754 on the same code, seeds and
   kind of card. Training on this card is not deterministic from run to run, so the spread of
   the seeds includes that noise.
5. As declared, the value's embedding is the next element. STraTS's lead over the network stays
   at 0.04 to 0.045, by which run of the control it is read against. No verdict of the registered grids changes.

**Limitations.**

- The static features apart are a mean of their states after the encoder, which shares one
  projection of the value across all channels. A network of their own fed the raw values, as
  STraTS has, is not what was tried.
- 800 stays: a difference under about 0.025 is not confirmed either way.

## 2026-10-02 — declared before the run: STraTS turned towards this network, part by part

**Question.** Which parts of STraTS carry its lead of about 0.04 over the network from nothing
on the same stays and tokens? Moving its parts into this network one at a time costs a campaign
each, and the first one, the static features apart, gained nothing (the section above). Here
STraTS is turned towards this network instead, one part at a time and then all of them, in
`scripts/strats_reference_run.py --ablate`, so each part's share of the lead is read where the
lead exists.

**Design.** STraTS at `e936cda` on this project's tokens (preparation (2) above), the same
division, seeds 1 to 5 for each variant; the clone is not edited.

| Variant | Changed towards this network |
|---|---|
| `baseline` | nothing; run again in the same session, since training on the card is not deterministic |
| `unweighted` | no weight on the positive class |
| `fixed-epochs` | no early stop: 30 epochs, the last weights |
| `our-schedule` | as `fixed-epochs`, under this network's rate: peak 0.000333, a tenth warming up, cosine to a hundredth, no clipping |
| `no-dropout` | no dropout anywhere, the dropout of whole variables included |
| `our-size` | 256 wide, 6 blocks, 4 heads |
| `linear-value` | a value embedded by one linear map instead of CVE |
| `statics-among` | the static features as triplets among the readings; no path for demographics |
| `mean-pooling` | the mean over the observed triplets instead of the learnt attention |
| `ours` | all of the above |

Not turned, so `ours` still differs from this network in: time embedded by CVE rather than
fixed Fourier features; no gap feature; residuals averaged rather than pre-normalised; a
feed-forward width of twice the width rather than four times; at most 880 readings a stay,
which cuts 13 of 3,997 stays; 2,557 stays learnt from rather than about 3,200, since 640 drive
STraTS's validation.

**Reading.** Each variant against `baseline` of the same session, by
`scripts/strats_reference_answers.py` and `scripts/campaign_pairs_report.py`. A variant *loses*
where the paired interval over stays lies below zero and the mean over seeds is below minus
twice its standard error. A part is acted on only where its variant loses 0.025 or more. `ours`
is also paired with the network from nothing (`e9e0aac5…`, 0.782).

**Predictions.**

1. The largest losses come from the training regime (`fixed-epochs`, `our-schedule`) and the
   size (`our-size`).
2. `unweighted`, `statics-among`, `mean-pooling` and `linear-value` each lose less than 0.025.
3. `ours` lands within 0.02 of the network from nothing.
4. `baseline` lands within 0.01 of the earlier 0.827.

**What follows.** A part whose variant loses 0.025 or more is tried in this network, alone,
under one campaign. If `ours` stays well above the network from nothing, the lead lies in what
is not turned, and the next ablation turns those. No verdict of the registered grids changes.

**Limitations.**

- A part's loss is read inside STraTS. In this network it may act otherwise, which is why a
  part is confirmed by a campaign of this network before anything changes.
- Five seeds and 800 stays: a loss under about 0.025 is not told from none.

## 2026-10-02 — Colab G4: STraTS turned towards this network, part by part

**Question.** Which parts of STraTS carry its lead over the network from nothing? The design and
predictions are in the section declared on 2026-10-02 for STraTS turned part by part.

**Conditions.** Commit `88ec177b`, STraTS at `e936cda`, this project's tokens; the split file
hashes as on the Mac (`787648be5e3adeb3`). One NVIDIA RTX PRO 6000 Blackwell (Colab G4), torch
2.11.0+cu130, CUDA, fp32, five runs at a time; tier M; 50 runs of 4 to 11 minutes of training.
Every run's record states the parts it turned, its rate, shape, dropout, class weight and number of
parameters, and each matches its variant: the linear value has 400 parameters fewer than CVE, the
mean 4,224 fewer than the learnt attention. `baseline` repeats the earlier runs exactly, step for
step: STraTS trains deterministically on this card.

**Area under the ROC curve on the 800 scored stays**, mean and range over seeds 1 to 5, and the
loss against `baseline` paired over stays (95 % interval) and seed by seed (mean ± standard error).

| Variant | Area | Seeds | Paired loss | Seed by seed | Loses |
|---|---|---|---|---|---|
| `baseline` | 0.827 | 0.814–0.842 | | | |
| `unweighted` | 0.826 | 0.818–0.839 | −0.000 [−0.007; +0.006] | −0.000 ± 0.003 | no |
| `fixed-epochs` | 0.820 | 0.797–0.831 | −0.007 [−0.017; +0.003] | −0.007 ± 0.009 | no |
| `our-schedule` | 0.816 | 0.811–0.825 | −0.011 [−0.020; −0.001] | −0.011 ± 0.005 | yes, under 0.025 |
| `no-dropout` | 0.824 | 0.816–0.835 | −0.002 [−0.013; +0.008] | −0.002 ± 0.003 | no |
| `our-size` | 0.721 | 0.705–0.742 | −0.105 [−0.148; −0.065] | −0.105 ± 0.007 | yes, see below |
| `linear-value` | 0.787 | 0.783–0.792 | −0.039 [−0.063; −0.018] | −0.039 ± 0.004 | **yes** |
| `statics-among` | 0.817 | 0.801–0.826 | −0.010 [−0.020; +0.000] | −0.010 ± 0.006 | no |
| `mean-pooling` | 0.822 | 0.813–0.839 | −0.004 [−0.008; −0.001] | −0.004 ± 0.002 | yes, under 0.025 |
| `ours` | 0.571 | 0.439–0.619 | −0.255 [−0.299; −0.212] | −0.255 ± 0.033 | yes, see below |

`our-schedule` against `fixed-epochs`: −0.004 [−0.013; +0.005]. `ours` against the network from
nothing (`e9e0aac5…`, 0.782): −0.210 [−0.257; −0.166].

**`our-size` does not train.** Its validation area moves between 0.58 and 0.75 from the first
epoch to the last, over the five seeds, and its best validation sum is 1.02 against the
baseline's 1.43. Not
overfitting, but no learning at all: STraTS has no normalisation inside its blocks and averages
each residual with its input, and six such blocks 256 wide do not train at its rate. The loss
measures STraTS's blocks at this network's size, not the size; `ours`, which carries it, says
nothing either.

**Conclusions.**

1. Prediction 1 fails. The training regime costs little: no class weight, no early stop, this
   network's schedule and no dropout each lose under 0.025, the most 0.011. The size could not
   be read (above).
2. Prediction 2 fails for one part. The value embedded by one linear map instead of CVE loses
   0.039, by both conditions, and lands at 0.787, where this network from nothing stands
   (0.782–0.786). The static features among the readings and the mean instead of the attention
   lose 0.010 and 0.004, under the threshold, as the campaign `e9e0aac5…` found from the other
   side.
3. Prediction 3 fails: `ours` lands at 0.571, because its blocks do not train at this size.
4. Prediction 4 holds: `baseline` repeats 0.827 exactly.
5. As declared, the value's embedding is tried in this network next, alone, under one campaign.
   This network embeds a value and its gap by one linear map shared by every channel; CVE passes
   the value through a narrow hidden layer and a tanh first. The size is tried in this network
   too, since its own blocks are normalised and the question STraTS could not answer remains.
   No verdict of the registered grids changes.

**Limitations.**

- A part's loss is read inside STraTS. The value embedding's 0.039 is a lead to test in this
  network, not a measure of what it will gain there.
- Five seeds and 800 stays: losses under about 0.025 are not told from none, even where the
  paired interval excludes zero.

## 2026-10-02 — declared before the run: a nonlinear value and a small shape

**Question.** Does the network from nothing gain from a value embedded through a narrow hidden
layer and a tanh, from a much smaller shape, or from both? Turned towards this network, STraTS
lost 0.039 to a value embedded by one linear map and landed where this network stands (the
section above). The size could not be read there; it can here, because this network's blocks
are normalised.

**Design.** Campaign `bb75a701…` (`campaigns/value-shape-all-physionet2012.toml`) has the same
fifth, seeds, budget, schedule and description of the control as `e9e0aac5…`.

| Candidate | Value embedding | Shape | Parameters |
|---|---|---|---|
| network from nothing (control) | one linear map | 256 wide, 4 heads, 6 blocks, feed-forward 1,024 | 4.8 million |
| … nonlinear value | 2 → 16 → tanh → 256 | as the control | 4.8 million |
| … small shape | one linear map | 64 wide, 16 heads, 2 blocks, feed-forward 128 (STraTS's) | 77,000 |
| … both | 2 → 8 → tanh → 64 | as the small shape | 78,000 |

The nonlinear value (`NonlinearValueEmbedding`) takes a token's value and its gap, as the
linear map does. CVE in STraTS takes the value alone and embeds time by a second CVE; this
network keeps its Fourier time features.

**Reading.** As above: each candidate against the control, *both* against each change alone,
and each against STraTS on this project's tokens, by `scripts/campaign_pairs_report.py`. A
candidate *gains* where the paired interval over stays lies above zero and the mean over seeds
exceeds twice its standard error; on this fifth that takes about 0.025. The control's area lands
within 0.01 of 0.782.

**Predictions.**

1. The nonlinear value gains over the control.
2. The small shape does not gain: a dropout of 0.2 gained nothing (the section on dropout and
   the grid), so this network is not held back by fitting its learning side too closely.
3. Both together land within 0.01 of the nonlinear value alone.

**What follows.** If the nonlinear value gains, it is a change to the encoder, not to the head,
so a backbone takes it only by being pretrained again; it becomes a candidate for the next
backbones, before the transfer matrix. If the small shape holds the control's area, a smaller
encoder becomes a candidate too, at a fraction of the arithmetic. If the nonlinear value gains
nothing, STraTS's lead does not carry over through this part, and what this network does not
share with STraTS is read next: time embedded by CVE, and the residual without normalisation.
No verdict of the registered grids changes.

**Limitations.**

- The nonlinear value is one form of the idea, its hidden width the square root of the width,
  as in CVE. Another form could gain where this one does not.
- 800 stays: a difference under about 0.025 is not confirmed either way.

## 2026-10-02 — Colab G4: a nonlinear value and a small shape

**Question.** Does the network from nothing gain from a nonlinear value embedding, from
STraTS's small shape, or from both? The design and predictions are in the section declared on
2026-10-02 for the nonlinear value and the small shape.

**Conditions.** Commit `b8a7a09d`, campaign `bb75a701…`, order `0db49a68…`. One Colab G4, CUDA,
one process; tier M. 20 cells in 1,493 s: about 100 s a cell in the control's shape, 47 s in the
small one. Read with:

    uv run scripts/campaign_pairs_report.py --out DIR \
        --campaign bb75a701... --campaign e9e0aac5...
    uv run scripts/strats_reference_answers.py --into DIR --side strats-e936cda tokens ...
    uv run scripts/campaign_pairs_report.py --out DIR --pair CONTROL ... CANDIDATE ...

The order declares `value_embedding: nonlinear` and the four counts of the shape in each
variant's method, and every cell was checked against that method before it ran. The cells of
the small shape ran in half the time.

**Area under the ROC curve on the 800 scored stays**, seeds 1 to 5:

| Candidate | Parameters | Mean | Seeds |
|---|---|---|---|
| network from nothing (control) | 4.8 million | 0.782 | 0.764–0.792 |
| … nonlinear value | 4.8 million | 0.779 | 0.764–0.801 |
| … small shape | 77,000 | 0.803 | 0.783–0.821 |
| … both | 78,000 | 0.783 | 0.766–0.795 |
| STraTS on this project's tokens | — | 0.827 | 0.814–0.842 |

**Gain in area**, paired over stays (95 % interval), and seed by seed (mean ± standard error).

| Control | Candidate | Paired gain | Seed by seed | Gains |
|---|---|---|---|---|
| network from nothing | nonlinear value | −0.003 [−0.024; +0.016] | −0.003 ± 0.010 | no |
| network from nothing | small shape | +0.021 [+0.003; +0.040] | +0.021 ± 0.010 | yes, narrowly |
| network from nothing | both | +0.001 [−0.019; +0.021] | +0.001 ± 0.007 | no |
| nonlinear value | both | +0.004 [−0.018; +0.027] | +0.004 ± 0.007 | no |
| small shape | both | −0.020 [−0.034; −0.006] | −0.020 ± 0.008 | loses |
| network from nothing | STraTS | +0.045 [+0.021; +0.070] | +0.045 ± 0.009 | yes |
| nonlinear value | STraTS | +0.048 [+0.022; +0.074] | +0.048 ± 0.007 | yes |
| small shape | STraTS | +0.023 [+0.003; +0.044] | +0.023 ± 0.006 | yes |
| both | STraTS | +0.044 [+0.023; +0.065] | +0.044 ± 0.006 | yes |
| control of `e9e0aac5…` | control here | +0.000 [−0.014; +0.014] | +0.000 ± 0.006 | no |

**Conclusions.**

1. Prediction 1 fails. The nonlinear value gains nothing: −0.003, an interval that excludes a
   gain of 0.02. The part that carries 0.039 of STraTS's lead inside STraTS does not carry it
   into this network.
2. Prediction 2 fails. The small shape gains 0.021: the interval lies above zero and the mean
   exceeds twice its standard error, 0.019. With about 60 times fewer parameters it closes about
   half of the gap to STraTS; 0.023 remains, and that remainder is confirmed.
3. Prediction 3 holds: both land 0.004 from the nonlinear value alone. Inside the small shape,
   though, the nonlinear value loses 0.020 of the small shape's gain, and that loss is
   confirmed too.
4. The check holds: the control lands at 0.782, as in `e9e0aac5…`.
5. As declared for a nonlinear value that gains nothing, what this network does not share with
   STraTS is read next: time embedded by CVE, and the residual without normalisation. The small
   shape is a candidate for the encoder of the next backbones. No verdict of the registered
   grids changes.

**Limitations.**

- The small shape gains by the rule, but only just: 0.021 against a bar of 0.019. Training on
  this card is not deterministic from run to run, and with five seeds a bar of twice the
  standard error is a loose one.
- The shape changes four counts at once. Which of width, depth, heads and feed-forward width
  carries the gain is not read here.
- Every candidate learns at 0.000333, the rate chosen for the control's shape; the small shape's
  own rate was not tuned.
- The nonlinear value is one form of the idea: the value and the gap embedded together, through
  a hidden layer as wide as the square root of the width (8 in the small shape). STraTS embeds
  the value alone. Another form could gain where this one does not.
- 800 stays: a difference under about 0.025 is not confirmed either way.

## 2026-10-02 — Colab G4: the mixture without SMD

**Question.** Does SMD teach the mixed backbone anything the intensive-care task uses, and are
its steps better spent on the other corpora? The design and predictions are in the section
declared on 2026-10-02 for the mixture without SMD.

**Conditions.** Backbones trained on a Kaggle T4 at `b6569a82` (tables in `manual-handoff.md`,
2026-10-02): A, `backbone-mixed5-nosmd-m`, keeps its eighth epoch (weights `994d1642…`); B,
`backbone-mixed5-nosmd-m-13`, keeps its twelfth of thirteen (`b4d3ad42…`). Campaigns
`4d17e289…` (mixture of five), `f7dae909…` (A) and `d5c27be6…` (B), orders placed at
`b8a7a09d`, three orders of 20 cells on one Colab G4 under CUDA MPS, 747 to 760 s an order.
Both candidates at the variants the mixture's selections chose at 200 stays: the frozen probe
at a rate of 0.03 over the mean of the states, full fine-tuning at a peak of 0.001 over the
same pooling, 2,000 steps at least in batches of 16, so 154 passes over the 200 stays, the last
weights kept. Scored on the fifth held out by seed 101, 800 stays, seeds 1 to 10. Read with:

    uv run scripts/campaign_pairs_report.py --out DIR --campaign 4d17e289... \
        --campaign f7dae909... --campaign d5c27be6... --campaign 596849cd...
    uv run scripts/campaign_pairs_report.py --out DIR --pair CONTROL ... CANDIDATE ...

**Pretext**, the kept epoch's validation loss as a share of the trivial predictor's:

| Backbone | Kept epoch | C-MAPSS | SKAB | ESA-AD | Stays | Mean of these four |
|---|---|---|---|---|---|---|
| mixture of five | 8 | 0.026 | 0.260 | 0.277 | 0.487 | 0.263 |
| A, without SMD | 8 | 0.006 | 0.266 | 0.294 | 0.489 | 0.264 |
| B, without SMD, 13 passes | 12 | 0.006 | 0.267 | 0.286 | 0.480 | 0.260 |

The mixture of five also ends at 0.364 on SMD, which A and B do not read.

**Area under the ROC curve at 200 stays**, mean over seeds 1 to 10 and their range:

| Backbone | Frozen probe | Full fine-tuning |
|---|---|---|
| mixture of five | 0.668 (0.599–0.717) | 0.686 (0.637–0.765) |
| A, without SMD | 0.650 (0.620–0.672) | 0.693 (0.657–0.726) |
| B, without SMD, 13 passes | 0.644 (0.609–0.691) | 0.636 (0.587–0.688) |
| network from nothing, `596849cd…` | — | 0.651 (0.591–0.698) |

The check holds: the mixture's fine-tuning lands at 0.686 against 0.692 in `596849cd…`.

**Gain in area**, paired over stays (95 % interval), and seed by seed (mean ± standard error).
A side *gains* or *loses* where the interval excludes zero and the mean over seeds exceeds twice
its standard error; a difference is acted on where it also reaches 0.03.

| Control | Candidate | Arm | Paired gain | Seed by seed | Reads |
|---|---|---|---|---|---|
| mixture of five | A | probe | −0.018 [−0.045; +0.009] | −0.018 ± 0.010 | no |
| mixture of five | A | fine-tuning | +0.007 [−0.014; +0.027] | +0.007 ± 0.011 | no |
| mixture of five | B | probe | −0.024 [−0.046; −0.001] | −0.024 ± 0.010 | loses, under 0.03 |
| mixture of five | B | fine-tuning | −0.050 [−0.074; −0.027] | −0.050 ± 0.009 | **loses** |
| A | B | probe | −0.006 [−0.028; +0.016] | −0.006 ± 0.006 | no |
| A | B | fine-tuning | −0.057 [−0.079; −0.035] | −0.057 ± 0.008 | **loses** |

Fine-tuning against the probe of the same backbone: the mixture +0.019 [−0.005; +0.044], A
+0.043 [+0.014; +0.074], B −0.008 [−0.030; +0.015].

**Conclusions.**

1. Prediction 1 holds for A and fails for B. A's loss on the stays lies 0.002 from the
   mixture's; B's lies 0.009 below A's, not the declared 0.02. Without SMD, C-MAPSS returns
   to 0.006, the level of the mixture of four, so the "four times higher" C-MAPSS of the
   mixture of five came with SMD's share of the run, not with the stays.
2. Prediction 2 holds. A holds the mixture's areas under both arms: the probe's estimate is
   −0.018, inside an interval that reaches +0.009, and fine-tuning's +0.007. SMD, three quarters
   of the mixture's attention, teaches the intensive-care task nothing measurable. As declared,
   SMD leaves the mixture of the next backbones unless it returns without its constant channels.
3. Prediction 3 fails. B's probe does not gain over A's (−0.006), and B's fine-tuning loses
   0.057 to A's, by both conditions and well past 0.03. B saw the stays 1.6 times as often and
   ends with the better pretext loss on every corpus but SKAB, yet fine-tuned it lands below its
   own probe (0.636 against 0.644) and below the network from nothing (0.651). Fine-tuning
   takes from A what the probe cannot show (+0.043) and from B nothing at all.
4. The pretext loss did not order the backbones on the task, for the second time: the turbofan
   backbone of 16 epochs did not replace the one of 8 either (`label-efficiency-curve.md`,
   2026-09-22). A backbone is chosen on the task, by the probe and by fine-tuning, never by its
   pretext loss. The fine-tuning recipe at 200 stays, 154 passes at a peak of 0.001 with the
   last weights kept, is the suspect: it leaves the better-trained backbone worse than nothing.
   What it does to B at a tenth of the rate, and under a floor of 500 steps, is read next; both
   knobs exist. No verdict of the registered grids changes.

**Limitations.**

- One pretraining run per backbone. A's and B's difference under fine-tuning is beyond what the
  seeds of the task explain, but a pretraining run's own spread has not been measured, and a
  run of 13 epochs is also a longer cosine decay, not only more steps.
- One task and one fifth of 800 stays with 127 deaths. C-MAPSS is read under another version of
  the corpus than these backbones learnt from, so the turbofan task is not read here.
- SMD's channel rows stay in A's and B's tables untrained, which the intensive-care task does
  not read.

## 2026-10-02 — declared before the run: the rate and the floor of full fine-tuning

**Question.** Does the recipe of full fine-tuning at 200 stays, not the backbone, decide what a
backbone transfers? The mixture without SMD at thirteen passes (B) ends with the better pretext
loss and fine-tunes to 0.057 below its eight-pass twin (A), below its own frozen probe and below
the network from nothing (the section above). The recipe in force is a peak of 0.001 over 2,000
steps at least, 154 passes over the 200 stays with the last weights kept and no stop. Here the
same two backbones fine-tune at a third and a tenth of that rate, and under a floor of 500
steps, 39 passes.

**Design.** Two files, each defined once under A (`994d1642…`) and once under B
(`b4d3ad42…`), the same fifth, seeds and budget as the mixture campaigns:

| File | Floor of steps | Candidates |
|---|---|---|
| `campaigns/fine-tuning-rate-200-physionet2012.toml` | 2,000, the registered one | full fine-tuning at 0.001 (control), 0.000333, 0.0001 |
| `campaigns/fine-tuning-floor-200-physionet2012.toml` | 500 | the same three |

Everything else stays the mixture campaigns' recipe: the mean of the states pooled, warm-up a
tenth of the run, cosine decay to a hundredth, no weight decay, batches of 16, the last weights.
Four campaigns of 30 cells, seeds 1 to 10, on a Colab GPU: `810ad80d…` (rate, A),
`cddace8e…` (rate, B), `5d47a256…` (floor, A), `60b7e023…` (floor, B).

**Reading.** By `scripts/campaign_pairs_report.py`, the rule used above: a side *gains* where
the paired interval over stays lies above zero and the mean over seeds exceeds twice its
standard error; a difference is acted on where it also reaches 0.03. Pairs: each rate and each
floor against the control of its own backbone; B against A at every setting; the control of
each new campaign at the floor of 2,000 against the fine-tuning cell of the mixture campaign of
the same backbone (`f7dae909…`, 0.693; `d5c27be6…`, 0.636), which it repeats.

**Predictions.**

1. The checks hold: each control at the floor of 2,000 lands within 0.01 of its mixture
   campaign's fine-tuning.
2. B gains from a lower rate: at 0.0001 and the floor of 2,000 it gains 0.03 or more over its
   control and lands at or above its probe (0.644).
3. A does not: at every rate and floor A lands within 0.02 of its control (0.693), so the
   recipe in force is near A's best and the registered selections, made under the mixture, are
   not far off for a backbone of its kind.
4. At B's best setting B lands within 0.02 of A at the same setting: the longer pretraining is
   not worse, it was fine-tuned wrongly.
5. The floor of 500 at 0.001 moves B up and A down, each by less than the rate does.

**What follows.** If 2 and 4 hold, the fine-tuning cells of the registered grids at 200 stays
measured the recipe as much as the backbone, and the rate of fine-tuning joins the selection of
every campaign of the next backbones, chosen per backbone. If the floor matters more than the
rate, the one floor every arm shares has to be re-read for fine-tuning, which is a change to the
campaign's budget, not to a knob. If B stays below A at every setting, the longer pretraining
itself transfers worse, and the next backbones keep the mixture's eight passes. No verdict of
the registered grids changes.

**Limitations.**

- Three rates and two floors, one backbone of each length; the surface is read at six points.
- One pretraining run per backbone, as above.
- 800 stays, 127 deaths, ten seeds: a difference under about 0.03 is not told from none.

## 2026-10-02 — declared before the run: the small shape again, on another fifth and at 200 stays

**Question.** Does the small shape's gain hold where it has not been read: on a fifth no
campaign has scored, over ten seeds, and at 200 stays? On the fifth held out by seed 101 it
gained 0.021 at every stay against a bar of 0.019, over five seeds, as one of about ten
comparisons this diagnosis has made on that fifth (the section on the nonlinear value and the
small shape). And which of its four counts carries the gain?

**Design.** Two campaigns, four shapes each, every candidate at the rate the registered grid
ran the network from nothing at that budget, so the candidates differ in shape only:

| Candidate | Width | Heads | Blocks | Feed-forward | Parameters |
|---|---|---|---|---|---|
| network from nothing (control) | 256 | 4 | 6 | 1,024 | 4.8 million |
| small shape (STraTS's) | 64 | 16 | 2 | 128 | 77,000 |
| shallow: the control's width in two blocks | 256 | 4 | 2 | 1,024 | 1.6 million |
| narrow: the small width in six blocks | 64 | 16 | 6 | 128 | 210,000 |

| Campaign | File | Fifth | Rate | Seeds |
|---|---|---|---|---|
| `afe91236…` | `campaigns/small-shape-all-physionet2012.toml` | seed 202, about 800 stays | 0.000333 | 1–10 |
| `a9ab91cc…` | `campaigns/small-shape-200-physionet2012.toml` | seed 101, the one above | 0.003 | 1–10 |

At 200 stays the campaign pairs with `4e28a197…` (the grid and the dropout), whose network
from nothing (0.652) and patch model (0.696) were scored on the same fifth and seeds. On a Kaggle
T4, one campaign a device.

**Reading.** As above, by `scripts/campaign_pairs_report.py`: each shape against the control
of its campaign; the shallow and the narrow against the small shape; at 200, the small shape
against the patch model of `4e28a197…`, and the control against that campaign's control, which
it repeats. A candidate *gains* where the paired interval over stays lies above zero and the
mean over seeds exceeds twice its standard error. The small shape's gain at every stay is
*confirmed* where it gains on the new fifth by 0.015 or more.

**Predictions.**

1. The check holds: at 200 the control lands within 0.02 of 0.652.
2. At every stay on the new fifth, the small shape gains 0.015 or more over the control.
3. At 200 stays the small shape gains 0.02 or more over the control and closes at least half of
   the patch model's lead of 0.043.
4. Both half-turned shapes gain over the control at every stay, and the narrow lands closer to
   the small shape than the shallow does: what the control pays for is its parameters, not its
   depth.

**What follows.** If 2 holds, the small shape is the encoder the next backbones are pretrained
in, and a pretraining at that shape over the stays is the next pretraining ordered. If 3 holds
too, the shape, not the input, explains most of what the patch model had over the network at
200 stays. If 2 fails, the gain of 2026-10-02 was the fifth's and the control's shape stands
for the next backbones. Whichever half-turned shape lands nearer the small one says which count
to vary next, with the small shape's own rate. No verdict of the registered grids changes.

**Limitations.**

- Each candidate learns at the control's rate at its budget; a shape's own rate is not chosen.
- The two half-turned shapes read two of fifteen ways of turning four counts.
- A fifth of 800 stays: at every stay a gain under about 0.015 is not told from none even over
  ten seeds; at 200 stays, under about 0.03.

## 2026-10-02 — declared before the run: STraTS without its size and its value, and under clipped values

**Question.** Two questions left by the ablation of 2026-10-02 (STraTS turned part by part) and
by the campaign `bb75a701…`. First: turned towards this network in everything but its size and
its value embedding, where does STraTS land? That reads at once what the parts never turned (time
embedded by CVE, the gap feature, a feed-forward width of twice the width, residuals averaged
rather than normalised) carry together. Second: is what the value's embedding through a tanh
buys STraTS the bound it puts on readings that are off by a unit or a decimal point? STraTS
has no normalisation inside its blocks, so a value embedded by one linear map carries a pH of
735 (88 standard deviations) into the residual stream as it is, while a tanh cannot pass more
than its weights. This network's blocks are normalised, which would explain why the nonlinear
value gained it nothing.

**Design.** STraTS at `e936cda` on this project's tokens, the division and seeds 1 to 5 of the
ablation, `scripts/strats_reference_run.py`; the clone is not edited.

| Variant | `--ablate` |
|---|---|
| `baseline` | nothing; run again in the same session |
| `minus-size-value` | every part but `our-size` and `linear-value`: `unweighted fixed-epochs our-schedule no-dropout statics-among mean-pooling` |
| `baseline-clipped` | `clipped-values`: every standardised value clipped to ±5 |
| `linear-clipped` | `linear-value clipped-values` |

Clipping is not a part of this network, which clips nothing; `ours` does not include it. The
runner logs how many readings each run clips.

**Reading.** Each variant against `baseline` of the same session, and `linear-clipped` against
`linear-value` of the earlier session (0.787), by `scripts/strats_reference_answers.py` and
`scripts/campaign_pairs_report.py`, the rule of the ablation: a variant *loses* or *gains* where
the paired interval over stays excludes zero and the mean over seeds exceeds twice its standard
error in that direction.

**Predictions.**

1. `baseline` repeats 0.827.
2. `minus-size-value` loses no more than 0.03 to `baseline` and lands within 0.015 of this
   network's small shape on the same fifth (0.803): the training regime and the head cost
   about 0.02 together, and the parts never turned cost nothing measurable.
3. `baseline-clipped` lands within 0.01 of `baseline`: with a tanh the bound changes nothing.
4. `linear-clipped` recovers at least half of what `linear-value` lost: it gains 0.02 or more
   over `linear-value` and lands at 0.807 or above.

**What follows.** If 2 holds, STraTS with this network's regime and head is this network's
small shape, and what separates the two is the value's embedding and nothing else that was not
turned; the search for the remaining 0.023 then stays with the value and the readings' tails.
If 4 holds, the embedding's lead is a bound on the tails, not a nonlinearity: the next
candidate for this network is the readings' preparation, clipped or robustly standardised, which
is a new version of the corpus, rather than another embedding. If 4 fails and 3 holds, the tanh
buys STraTS a nonlinearity its unnormalised blocks cannot make, and this network, whose blocks
can, has nothing to take from it. No verdict of the registered grids changes.

**Limitations.**

- One bound, five standard deviations; a tighter one could act where this one does not.
- Clipping bounds a tail; it does not undo what the tail did to the scale. STraTS
  standardises each variable by a standard deviation the tail inflates (pH: 8.2 against 0.08
  without the 11 readings beyond its range), so the clinical range of such a variable stays
  compressed near zero under both variants. A robust standardisation is the next variant if
  clipping gains nothing.
- Five seeds and 800 stays: a difference under about 0.025 is not told from none.
- The parts never turned are read together; if `minus-size-value` loses more than 0.03, which
  of them carries it is not read here.

## 2026-10-02 — Colab G4: the rate and the floor of full fine-tuning

**Question.** Does the recipe of full fine-tuning at 200 stays, not the backbone, decide what
backbone B transfers? The design and predictions are in the section declared on 2026-10-02 for
the rate and the floor of full fine-tuning.

**Conditions.** Orders placed at `dcad779c`; four orders of 30 cells on one Colab G4 under CUDA
MPS, four at once: 778–785 s for the floor of 500, 1,906–1,907 s for the floor of 2,000; mean
utilisation 96 %. Scored on the fifth held out by seed 101, 800 stays, seeds 1 to 10. Read with:

    uv run scripts/campaign_pairs_report.py --out DIR --campaign 810ad80d... \
        --campaign cddace8e... --campaign 5d47a256... --campaign 60b7e023... \
        --campaign f7dae909... --campaign d5c27be6...
    uv run scripts/campaign_pairs_report.py --out DIR --pair CONTROL ... CANDIDATE ...

**Area under the ROC curve at 200 stays**, mean over seeds 1 to 10 (seed-by-seed values in the
CSV); the frozen probe and the network from nothing from the campaigns above, for reference.

| Backbone | Floor | Peak 0.001 | Peak 0.000333 | Peak 0.0001 | Probe | From nothing |
|---|---|---|---|---|---|---|
| A, without SMD, 8 passes | 2,000 | 0.689 | 0.688 | 0.682 | 0.650 | 0.651 |
| A | 500 | 0.690 | 0.702 | 0.685 | | |
| B, without SMD, 13 passes | 2,000 | 0.654 | 0.643 | 0.622 | 0.644 | |
| B | 500 | 0.663 | 0.639 | 0.618 | | |

**Gain in area**, paired over stays (95 % interval) and seed by seed (mean ± standard error).
A side *gains* or *loses* where the interval excludes zero and the mean over seeds exceeds twice
its standard error; a difference is acted on where it also reaches 0.03.

| Control | Candidate | Paired gain | Seed by seed | Reads |
|---|---|---|---|---|
| A at 0.001, floor 2,000 | A at 0.000333 | −0.001 [−0.016; +0.013] | −0.001 ± 0.008 | no |
| A at 0.001, floor 2,000 | A at 0.0001 | −0.008 [−0.024; +0.009] | −0.008 ± 0.008 | no |
| A at 0.001, floor 2,000 | A at 0.001, floor 500 | +0.000 [−0.011; +0.013] | +0.000 ± 0.008 | no |
| A at 0.001, floor 2,000 | A at 0.000333, floor 500 | +0.012 [−0.003; +0.028] | +0.012 ± 0.009 | no |
| A at 0.001, floor 2,000 | A at 0.0001, floor 500 | −0.004 [−0.023; +0.014] | −0.004 ± 0.008 | no |
| B at 0.001, floor 2,000 | B at 0.000333 | −0.011 [−0.028; +0.006] | −0.011 ± 0.007 | no |
| B at 0.001, floor 2,000 | B at 0.0001 | −0.032 [−0.051; −0.012] | −0.032 ± 0.009 | **loses** |
| B at 0.001, floor 2,000 | B at 0.001, floor 500 | +0.009 [−0.007; +0.025] | +0.009 ± 0.013 | no |
| B at 0.001, floor 2,000 | B at 0.000333, floor 500 | −0.015 [−0.031; +0.001] | −0.015 ± 0.009 | no |
| B at 0.001, floor 2,000 | B at 0.0001, floor 500 | −0.036 [−0.055; −0.016] | −0.036 ± 0.009 | **loses** |
| A at 0.001, floor 2,000 | B, the same | −0.036 [−0.059; −0.013] | −0.036 ± 0.016 | loses |
| A at 0.000333, floor 2,000 | B, the same | −0.045 [−0.070; −0.020] | −0.045 ± 0.017 | loses |
| A at 0.0001, floor 2,000 | B, the same | −0.060 [−0.086; −0.034] | −0.060 ± 0.010 | loses |
| A at 0.001, floor 500 | B, the same | −0.027 [−0.050; −0.005] | −0.027 ± 0.008 | loses, under 0.03 |
| A at 0.000333, floor 500 | B, the same | −0.063 [−0.089; −0.037] | −0.063 ± 0.009 | loses |
| A at 0.0001, floor 500 | B, the same | −0.067 [−0.094; −0.041] | −0.067 ± 0.009 | loses |
| A of `f7dae909…` (0.693) | A at 0.001, floor 2,000 | −0.003 [−0.014; +0.007] | −0.003 ± 0.012 | check holds |
| B of `d5c27be6…` (0.636) | B at 0.001, floor 2,000 | +0.018 [+0.002; +0.033] | +0.018 ± 0.011 | check fails on the mean |

**Conclusions.**

1. Prediction 1 holds for A and fails for B: A's control repeats its mixture campaign's cell
   within 0.003; B's lands 0.018 above its own, on the same code, order kind and card. The
   card's training is not deterministic from run to run, and 30 cells of B moved its mean by
   about what one seed moves it. B's level is read to about 0.02.
2. Prediction 2 fails, in the other direction. B does not gain from a lower rate: at a tenth of
   the rate it loses 0.032 at the floor of 2,000 and 0.036 at 500, by both conditions; at a
   third it loses 0.011 and 0.015, not confirmed. The less fine-tuning moves B's weights, the
   worse B does.
3. Prediction 3 holds. A lands within 0.012 of its control at every rate and floor; the recipe
   in force is near A's best, and no rate or floor read here would have changed A's cell.
4. Prediction 4 fails. At its best setting (0.001, floor 500) B lands 0.027 below A at the same
   setting, and 0.036 to 0.067 below A at the others. B is below A at every one of the six
   points, and at every point B's fine-tuning lands at or below the network from nothing
   (0.651), while A's lands 0.03 to 0.05 above it.
5. Prediction 5 holds in the letter and says little: the floor of 500 moves B by +0.009 and A
   by 0.000 at 0.001, both inside noise.
6. As declared for B below A at every setting: the longer pretraining itself transfers worse
   under fine-tuning, and the next backbones keep the mixture's eight passes. The recipe is not
   what limited fine-tuning at 200 stays, so the fine-tuning rate does not join the selections
   on this account. What B holds that fine-tuning cannot use, when its probe reads the same as
   A's, is not answered here. No verdict of the registered grids changes.

**Post hoc, not declared: the weights themselves.** The Frobenius norm of each module's
weights, A against B (`data/report/t42d/fine-tuning/weight-norms.txt`,
`data/report/t42d/weight_norms.py`): B's attention weights are 1.20 to 1.31 times A's in every
block and its feed-forward weights 1.25 to 1.57 times, growing with depth; the embeddings, the
time encoding and the value projection are within 3 %, and the LayerNorm gains differ by under
10 % except the first block's attention norm, whose mean gain is 0.16 in B against 0.33 in A.
The mixture of five, trained for about as many steps as B, has norms like B's and fine-tunes
like A, so the size of the weights alone does not explain B. Larger weights under pre-norm
blocks make a given rate move the function less, which is consistent with B wanting a higher
rate, not a lower one; a rate of 0.003 for B is the one cheap reading this leaves open.

**Limitations.**

- One pretraining run per backbone, and B's own level moved by 0.018 between two runs of the
  same cells.
- Three rates and two floors; a rate above 0.001 was not read.
- The weights' norms are a measurement made after the result, to be declared before it is read
  for anything.

## 2026-10-02 — Colab G4: STraTS without its size and its value, and under clipped values

**Question.** Where does STraTS land turned towards this network in everything but its size and
its value embedding, and is what the value's embedding through a tanh buys it the bound it puts
on readings that are off by a unit or a decimal point? The design and predictions are in the
section declared on 2026-10-02 for STraTS without its size and its value.

**Conditions.** Commit `dcad779c`, STraTS at `e936cda`, this project's tokens, the division
and seeds of the ablation (the split file hashes as on the Mac). One Colab G4, CUDA, fp32, five
runs at a time under CUDA MPS; 20 runs of 3.5 to 9 minutes. Each run's record names its parts;
the clipped runs clip 2,789 or 2,790 of the 1.75 million readings (0.16 %) to ±5 standard
deviations, the count differing by one between the sides each run's standardisation is fitted
on. Archive `durable/sha256/d791aa24…`, unpacked under `data/report/t42d/strats-ft/runs/`;
read by `data/report/t42d/strats-ft-pairs.sh`.

**Area under the ROC curve on the 800 scored stays**, seeds 1 to 5:

| Variant | Mean | Seeds |
|---|---|---|
| `baseline` | 0.827 | 0.814–0.842 |
| `minus-size-value`: every part but the size and the value | 0.783 | 0.769–0.794 |
| `baseline-clipped` | 0.827 | 0.815–0.842 |
| `linear-clipped` | 0.821 | 0.809–0.826 |
| `linear-value` (session of 2026-10-02, part by part) | 0.787 | 0.783–0.792 |
| this network's small shape (`bb75a701…`) | 0.803 | 0.783–0.821 |
| this network from nothing (`bb75a701…`) | 0.782 | 0.764–0.792 |

**Gain in area**, paired over stays (95 % interval), and seed by seed (mean ± standard error).

| Control | Candidate | Paired gain | Seed by seed | Reads |
|---|---|---|---|---|
| `baseline` | `minus-size-value` | −0.043 [−0.063; −0.025] | −0.043 ± 0.009 | loses |
| `baseline` | `baseline-clipped` | +0.001 [−0.000; +0.001] | +0.001 ± 0.000 | within 0.01 |
| `baseline` | `linear-clipped` | −0.006 [−0.020; +0.008] | −0.006 ± 0.004 | no |
| `linear-value` | `linear-clipped` | **+0.034 [+0.016; +0.053]** | +0.034 ± 0.004 | **gains** |
| small shape | `minus-size-value` | −0.020 [−0.042; +0.002] | −0.020 ± 0.011 | no |
| `baseline` of the earlier session | `baseline` here | +0.000 [+0.000; +0.000] | 0.000 ± 0.000 | repeats exactly |

**Conclusions.**

1. Prediction 1 holds: `baseline` repeats 0.827 to the stay.
2. Prediction 2 fails on both counts. STraTS under this network's training regime and head loses
   0.043, beyond the declared 0.03, and lands at 0.783, where this network from nothing stands
   (0.782) and 0.020 below this network's small shape, not within 0.015. The parts cost more
   together than one at a time (their single losses summed to 0.027). What was never turned
   (time by CVE, the gap, the feed-forward width, the residual's normalisation) does not carry
   STraTS's lead: with the regime and head turned, STraTS is this network, value embedding
   aside.
3. Prediction 3 holds: with CVE the bound changes nothing, +0.001.
4. **Prediction 4 holds, and more than asked.** With a linear value embedding and the readings
   clipped to ±5 standard deviations, STraTS recovers 0.034 of the 0.039 the linear map lost,
   and lands 0.006 from its baseline, within noise. The 0.16 % of readings beyond the bound,
   which the tanh of CVE cannot pass, are what the linear map could not take.
5. As declared for 4 holding: the value embedding's lead over this network is a bound on the
   readings' tails, not a nonlinearity, which is why the nonlinear value gained this network
   nothing. The next candidate for this network is the readings' preparation, clipped or
   standardised robustly, and the first reading of it is this network from nothing, in both
   shapes, on readings clipped at the input. For the next backbones that is a change to the
   corpus's preparation, declared in its own section. No verdict of the registered grids
   changes.

**Limitations.**

- Clipping is read inside STraTS. This network's blocks are normalised, so an outlier token
  may cost it less, or otherwise; it is read in this network next.
- One bound, ±5 standard deviations after a standardisation the tails inflate; clinical
  pipelines bound by physiological ranges instead, which this did not try.
- Five seeds and 800 stays: a difference under about 0.025 is not told from none.

## 2026-10-02 — declared before the run: the network from nothing on bounded values

**Question.** Does this network gain when every token's value is bounded at its input? Inside
STraTS, the value embedding's lead of 0.039 over a linear map is a bound on the readings'
tails: with the readings clipped to ±5 standard deviations, the linear map recovers 0.034 of it
(the section above). This network embeds a value by a linear map and its blocks are normalised,
which may absorb an outlier token or may not; its nonlinear value, which also bounds, gained it
nothing in its own shape and lost 0.020 in the small one, so the answer is not known.

**Design.** Campaign `40a666f9…` (`campaigns/value-clip-all-physionet2012.toml`), the fifth
held out by seed 101, every candidate at 0.000333, seeds 1 to 10:

| Candidate | Shape | Values at the input |
|---|---|---|
| network from nothing (control) | 256 wide, 4 heads, 6 blocks, feed-forward 1,024 | as they are |
| … bounded | the same | clipped to ±5 |
| small shape | 64, 16, 2, 128 | as they are |
| small shape, bounded | the same | clipped to ±5 |

The bound is the plan's knob `value_clip`, applied by a module without weights in front of the
encoder (`ClippedValues`); the gap feature is not bounded. Ten seeds this time, because a mean
over ten seeds moved by 0.018 between two runs of the same cells on this card; the campaigns
of five seeds (`bb75a701…`) and STraTS's runs are therefore compared by their means, not paired.
One order on a Colab GPU.

**Reading.** By `scripts/campaign_pairs_report.py`, within the campaign: each bounded candidate
against its unbounded shape, the small shape against the control, and the two bounded ones
against each other. A candidate *gains* where the paired interval over stays lies above zero
and the mean over seeds exceeds twice its standard error. Reference levels on this fifth:
STraTS 0.827, STraTS with a linear value on clipped readings 0.821, the small shape 0.803, the
control 0.782.

**Predictions.**

1. The check holds: the control lands within 0.01 of 0.782 and the small shape within 0.015 of
   0.803.
2. The bound gains the control's shape 0.02 or more.
3. The bound gains the small shape 0.015 or more, and the small shape bounded lands at 0.815 or
   above, within 0.01 of STraTS with a linear value on clipped readings.
4. The two gains add: the small shape bounded lands above the control bounded by 0.01 or more.

**What follows.** If 2 or 3 holds, the readings' tails cost this network too, and the next
backbones are pretrained on a corpus whose preparation bounds them: a new version of the
intensive-care corpus, by physiological ranges as the clinical pipelines do or by a bound after
a robust standardisation, with the window of 48 hours and a minute, under an ADR that replaces
that part of ADR-0031; the other corpora get the same reading of their tails first. If neither
holds, the tails cost STraTS alone, for want of normalisation in its blocks, and the shape is
decided on by the campaigns on Kaggle. No verdict of the registered grids changes.

**Limitations.**

- One bound, ±5 after the corpus's standardisation, whose standard deviation the tails inflate
  (pH: 8.2 against 0.08 without 11 readings); the clinical range stays compressed under both
  candidates, so a gain here is a lower bound on what a robust preparation could give.
- A bound at the input of a network trained from nothing; a backbone pretrained on unbounded
  values is another question.
- 800 stays: a difference under about 0.02 is not confirmed by the paired interval even over
  ten seeds.

## 2026-10-02 — Colab G4: the network from nothing on bounded values

**Question.** Does this network gain when every token's value is bounded at its input, in its
own shape or in the small one? The design and predictions are in the section declared on
2026-10-02 for the network from nothing on bounded values.

**Conditions.** Commit `7be57e0d`, campaign `40a666f9…`, order `1d788275…`. One Colab G4, CUDA,
one process; tier M; 40 cells in 2,972 s. Scored on the fifth held out by seed 101, 800 stays,
seeds 1 to 10. Read with `data/report/t42d/value-clip-pairs.sh`; CSV under
`data/report/t42d/value-clip/`.

**Area under the ROC curve on the 800 scored stays**, mean over seeds 1 to 10, and the mean
over seeds 1 to 5 beside the earlier campaign's:

| Candidate | Seeds 1–10 | Seeds 1–5 | Seeds 1–5 in `bb75a701…` |
|---|---|---|---|
| network from nothing (control) | 0.790 | 0.792 | 0.782 |
| … bounded at ±5 | 0.789 | 0.790 | — |
| small shape | 0.794 | 0.803 | 0.803 |
| … bounded at ±5 | 0.795 | 0.803 | — |

The small shape's cells under seeds 1 to 5 repeat the earlier campaign's to the third decimal;
the control's do not (0.792 against 0.782). Under seeds 6 to 10 the small shape lands at 0.785
and the control at 0.789.

**Gain in area**, paired over stays (95 % interval), and seed by seed (mean ± standard error).

| Control | Candidate | Paired gain | Seed by seed | Reads |
|---|---|---|---|---|
| control | control bounded | −0.001 [−0.011; +0.010] | −0.001 ± 0.003 | no |
| small shape | small shape bounded | +0.001 [−0.002; +0.003] | +0.001 ± 0.001 | no |
| control | small shape | +0.004 [−0.011; +0.019] | +0.004 ± 0.007 | no |
| control bounded | small shape bounded | +0.005 [−0.010; +0.020] | +0.005 ± 0.007 | no |
| control | small shape bounded | +0.004 [−0.011; +0.020] | +0.004 ± 0.007 | no |

**Conclusions.**

1. Prediction 1 holds: the control lands 0.008 from 0.782 and the small shape 0.009 from
   0.803, both inside the declared bounds, though the control's seeds moved by up to 0.026.
2. Predictions 2, 3 and 4 fail. The bound gains neither shape anything: −0.001 and +0.001,
   with intervals that exclude a gain of 0.01. Bounding the small shape's values changes its
   seed-by-seed answers by 0.001 ± 0.001: the 0.16 % of readings beyond ±5 reach this network
   and cost it nothing.
3. As declared for neither prediction holding: the readings' tails cost STraTS alone, for want
   of normalisation in its blocks, and this network's blocks absorb them. The corpus's
   preparation is not reopened on this account; the window of 48 hours and a minute remains the
   one change the next publication makes.
4. Not declared, but read in passing: over ten seeds the small shape's gain on this fifth is
   +0.004, not the +0.021 five seeds gave. Its first five seeds repeat exactly; its next five
   land at 0.785 against the control's 0.789. The gain of 2026-10-02 was those five seeds'. The
   shape is decided by the campaigns on Kaggle, on another fifth and at 200 stays, as declared
   there.
5. STraTS's lead of about 0.035 over this network stands, and neither of its two parts carries
   over one at a time: its value embedding's part is a bound this network does not need, and
   its training regime and head, which cost STraTS 0.043 when turned together, cost this
   network nothing when turned one at a time (dropout, the static features apart, attention
   pooling). What is left untried in this network is that regime whole: the early stop on a
   validation side, the class weight and the variable dropout together. No verdict of the
   registered grids changes.

**Limitations.**

- One bound; a robust standardisation that undoes the compression of the clinical range was not
  tried, and nothing here says it would gain nothing.
- The control's cells are not deterministic on this card, the small shape's are; a comparison
  of the two across campaigns carries the control's drift.
- 800 stays: a difference under about 0.02 is not confirmed either way.

## 2026-10-03 — Kaggle T4: the small shape again, on another fifth and at 200 stays

**Question.** Does the small shape's gain hold on a fifth no campaign had scored, and at 200
stays, and which of its counts carries it? The design and predictions are in the section
declared on 2026-10-02 for the small shape again.

**Conditions.** Orders placed at `dcad779c`; two orders on one Kaggle session, a Tesla T4
each: `afe91236…` (every stay, fifth of seed 202) in 32,406 s, `a9ab91cc…` (200 stays, fifth
of seed 101) in 10,739 s; 40 cells each, seeds 1 to 10; tier M. Read with
`data/report/t42d/small-shape-pairs.sh`; CSV under `data/report/t42d/small-shape/`.

**Area under the ROC curve**, mean over seeds 1 to 10 (seed by seed in the CSV):

| Candidate | Parameters | Every stay, fifth 202 | 200 stays, fifth 101 |
|---|---|---|---|
| network from nothing (control) | 4.8 million | 0.798 | 0.660 |
| small shape: 64 wide, 2 blocks | 77,000 | 0.803 | 0.686 |
| shallow: 256 wide, 2 blocks | 1.6 million | 0.787 | 0.674 |
| narrow: 64 wide, 6 blocks | 210,000 | 0.789 | 0.653 |
| patch model (`4e28a197…`, same fifth and seeds) | | | 0.696 |

**Gain in area**, paired over stays (95 % interval), and seed by seed (mean ± standard error).
A candidate *gains* where the interval lies above zero and the mean over seeds exceeds twice
its standard error.

| Budget | Control | Candidate | Paired gain | Seed by seed | Reads |
|---|---|---|---|---|---|
| every stay | control | small | +0.005 [−0.009; +0.020] | +0.005 ± 0.006 | no |
| every stay | control | shallow | −0.011 [−0.022; −0.000] | −0.011 ± 0.005 | loses, narrowly |
| every stay | control | narrow | −0.009 [−0.024; +0.007] | −0.009 ± 0.006 | no |
| every stay | small | shallow | −0.016 [−0.031; −0.002] | −0.016 ± 0.005 | loses |
| every stay | small | narrow | −0.014 [−0.024; −0.004] | −0.014 ± 0.006 | loses |
| 200 | control | small | +0.026 [+0.009; +0.044] | +0.026 ± 0.017 | interval yes, seeds no |
| 200 | control | shallow | +0.014 [−0.003; +0.031] | +0.014 ± 0.016 | no |
| 200 | control | narrow | −0.007 [−0.030; +0.015] | −0.007 ± 0.017 | no |
| 200 | small | shallow | −0.012 [−0.032; +0.007] | −0.012 ± 0.014 | no |
| 200 | small | narrow | −0.034 [−0.050; −0.017] | −0.034 ± 0.008 | loses |
| 200 | control of `4e28a197…` (0.652) | control here | +0.008 [−0.010; +0.026] | +0.008 ± 0.019 | check holds |
| 200 | patch model (0.696) | small | −0.009 [−0.035; +0.015] | −0.009 ± 0.016 | no |

**Conclusions.**

1. Prediction 1 holds: the control at 200 lands 0.008 from the earlier campaign's.
2. Prediction 2 fails. On the new fifth at every stay the small shape gains 0.005, an interval
   that excludes the declared 0.015 only just and a mean under one standard error. With the
   reading of the same day on the first fifth over ten seeds (+0.004), the small shape's gain
   at every stay is not confirmed: the +0.021 of 2026-10-02 belonged to five seeds.
3. Prediction 3 holds in its first part and not in its second. At 200 stays the small shape
   gains 0.026 by the paired interval, above the declared 0.02, and lands 0.009 from the patch
   model, closing most of its lead of 0.043; but seed by seed the gain is 1.5 standard errors,
   under the bar of two, so by the rule it does not gain. The spread over seeds at 200 stays
   (0.045) is three times the one at every stay.
4. Prediction 4 fails, and the shapes point the other way. At every stay both half-turned
   shapes lose to the small one by 0.014 to 0.016, by both conditions, and sit at or below the
   control. At 200 the shallow shape, the control's width in two blocks, lands 0.014 above the
   control and 0.012 below the small shape, while the narrow shape, the small width in six
   blocks, lands below the control and 0.034 below the small shape, by both conditions. Depth
   is what costs at 200 stays, not the count of parameters; the width matters little; and the
   small shape needs both of its counts to hold at every stay.
5. For the next backbones: the small shape is not confirmed to gain at every stay, where the
   backbones are compared, and gains by the interval at 200 stays, where the thesis is read. A
   two-block encoder is a candidate for the low-budget regime, not a settled choice; the
   control's shape stands until a pretrained small shape is read against a pretrained large one
   on the task. No verdict of the registered grids changes.

**Limitations.**

- Each shape learns at the control's rate at its budget; a shape's own rate could move the
  small ones more than the control.
- At 200 stays ten seeds read a difference only to about 0.017 seed by seed, so a gain of
  0.026 is neither confirmed nor excluded by the second condition.
- Two half-turned shapes read two of fifteen ways of turning four counts.

## 2026-10-03 — declared before the run: the published regime, whole

**Question.** Does this network gain from the training regime of the published network taken
whole? Turned towards this network's regime one part at a time, STraTS lost at most 0.011 to any
part; turned in all of them, 0.043, and landed where this network stands (the sections of
2026-10-02). This network tried the parts it had one at a time (dropout, the static features
apart, attention pooling) and gained nothing, and neither its value embedding nor the readings'
tails carry STraTS's lead into it. The parts together are the one measured lever left.

**Design.** Campaign `e1959b4f…` (`campaigns/published-regime-all-physionet2012.toml`), the
fifth held out by seed 101, seeds 1 to 10, every candidate capped at the control's thirty
epochs, under ADR-0047:

| Candidate | Shape | Regime |
|---|---|---|
| network from nothing (control) | 256 wide, 6 blocks | as every campaign: 0.000333 with warm-up and cosine decay, every epoch, last weights |
| … with the stop alone | the same | a fifth of the labelled units held out by unit, patience 10, best epoch's weights |
| … under the published regime | the same | the stop; the positive outcome weighted by the ratio of the classes; a fifth of each window's channels withheld per step; dropout 0.2; a constant rate of 0.0005 |
| the small shape under the published regime | 64 wide, 16 heads, 2 blocks, feed-forward 128 | the same |

The stop scores the held-out units after each epoch by the sum of the areas under the ROC and
precision-recall curves, as STraTS does; the learning side is 2,560 of the 3,200 tuning stays,
as STraTS's was. Not transplanted: STraTS's gradient clipping at 0.3, its cap of 50 epochs and
its path for the demographics. One order on a Colab GPU.

**Reading.** By `scripts/campaign_pairs_report.py`, within the campaign: the stop against the
control; the regime against the control and against the stop; the small shape under the regime
against the regime in the control's shape. A candidate *gains* where the paired interval over
stays lies above zero and the mean over seeds exceeds twice its standard error. Reference
levels on this fifth, by their means: STraTS 0.827; STraTS under this network's regime and head
0.783; this network's control 0.782–0.790 over the earlier campaigns; its small shape 0.794–0.803.

**Predictions.**

1. The check holds: the control lands within 0.01 of 0.790, the mean of its last reading over
   ten seeds.
2. The stop alone gains less than 0.01: on 2,560 stays the cosine decay already ends the run
   where it would stop.
3. The regime whole gains less than 0.02 in the control's shape. Its parts, where this network
   has read them, gained nothing, and the interaction STraTS showed is expected to be STraTS's,
   whose blocks have no normalisation to steady them.
4. The small shape under the regime lands below 0.815: the lead of 0.827 is not reached by
   transplanting the regime into this network's blocks.

**What follows.** If 3 or 4 fails and the regime gains 0.02 or more, the lever is found: the
regime becomes a protocol variant for the budget of every stay, ADR-0047 is accepted with that
reading, the preregistration records the change, and the next backbones' campaigns read their
arms under it at that budget. If the predictions hold, nothing measured separates this network
from the published one except what it does not need, and the diagnosis closes: the remaining
0.03 at every stay is left as the published network's, the shape is read on pretrained backbones
as declared on 2026-10-03, and the next ticket starts from eight-pass backbones without SMD.
No verdict of the registered grids changes.

**Limitations.**

- One stop share and one patience; one class weight; one channel dropout rate.
- The stop's learning side is a fifth smaller than the control's, so a gain under the regime is
  a gain against fewer labels, and a loss may be the labels'.
- 800 stays and ten seeds: a difference under about 0.02 is not confirmed by the paired
  interval.

## 2026-10-03 — Colab G4: the published regime, whole

**Question.** Does this network gain from the published network's training regime taken whole,
and from its stop alone? The design and predictions are in the section declared on 2026-10-03
for the published regime.

**Conditions.** Commit `1f489909`, campaign `e1959b4f…`, order `bdfdabf2…`. One Colab G4, CUDA,
one process; tier M; 40 cells in 2,502 s. Scored on the fifth held out by seed 101, 800 stays,
seeds 1 to 10. Read with `data/report/t42d/published-regime-pairs.sh`; CSV under
`data/report/t42d/published-regime/`.

**Area under the ROC curve on the 800 scored stays**, mean over seeds 1 to 10, with the seeds'
standard deviation and the seconds a cell took:

| Candidate | Mean | SD over seeds | Seconds a cell |
|---|---|---|---|
| network from nothing (control; 30 epochs, last weights) | 0.783 | 0.018 | 98 |
| … with the stop alone | 0.810 | 0.006 | 51 |
| … under the published regime | 0.799 | 0.012 | 55 |
| the small shape under the published regime | 0.818 | 0.011 | 37 |
| STraTS on the same stays, for reference | 0.827 | 0.010 | |

A cell under the stop learns from 2,560 stays and scores 640 after each epoch; its seconds say
the runs ended at about half the control's epochs.

**Gain in area**, paired over stays (95 % interval), and seed by seed (mean ± standard error).

| Control | Candidate | Paired gain | Seed by seed | Reads |
|---|---|---|---|---|
| control | stop alone | **+0.027 [+0.009; +0.044]** | +0.027 ± 0.005 | **gains** |
| control | regime, whole | +0.016 [−0.006; +0.038] | +0.016 ± 0.008 | no |
| stop alone | regime, whole | −0.011 [−0.023; +0.001] | −0.011 ± 0.004 | no, narrowly |
| regime, whole | small shape under it | +0.019 [+0.004; +0.033] | +0.019 ± 0.006 | gains |
| control | small shape under the regime | **+0.034 [+0.015; +0.054]** | +0.034 ± 0.008 | **gains** |

**Conclusions.**

1. Prediction 1 holds: the control lands at 0.783, 0.007 from 0.790.
2. Prediction 2 fails, and it is the finding. The stop alone gains 0.027 by both conditions,
   from fewer labels (2,560 against 3,200), and cuts the spread over seeds from 0.018 to 0.006.
   Thirty epochs of cosine decay with the last weights kept were not where this network should
   have stopped: kept at its best epoch on held-out stays, the network from nothing lands at
   0.810.
3. Prediction 3 holds in the letter and misleads: the regime whole gains 0.016 in the control's
   shape, under 0.02, because the parts beside the stop (the class weight, the channels
   withheld, dropout, the constant rate) cost 0.011 against the stop alone. The lever is the
   stop; the rest of the regime is STraTS's, not this network's.
4. Prediction 4 fails. The small shape under the regime lands at 0.818, above 0.815 and 0.009
   from STraTS, with an interval that includes STraTS's level. With the stop, the small shape
   gains over the control's shape (+0.019), which it did not do without one over ten seeds.
5. As declared for the regime gaining 0.02 or more: the lever is found. ADR-0047 is accepted
   with this reading, the stop enters the protocol for the budget of every stay
   (`docs/preregistration.md`, 2026-10-03), and the next backbones' campaigns read their arms
   under it there. The gap between this network and the published one at every stay is closed to
   within noise by two things the diagnosis measured: the best epoch kept on held-out labels, and
   two blocks instead of six. No verdict of the registered grids changes: they were read under
   the regime registered at the time, and the stop is a change for campaigns defined from now on.

**Limitations.**

- The stop alone was read in the control's shape; the small shape was read under the regime
  whole. The small shape with the stop alone is the one cell this leaves unread.
- One share and one patience; the stop's held-out fifth is 640 stays here and would be 40 at
  200 stays, where it was not read and is not registered.
- The regime's other parts were read together; which of them costs the 0.011 is not read.
- 800 stays and ten seeds: a difference under about 0.02 is not confirmed by the paired
  interval.

## 2026-10-03 — declared before the run: the stop alone in the small shape, and the stop at 200 and 1,000 stays

**Question.** Two things the reading of the published regime left unread. Does the small shape
gain under the stop alone, the recipe registered at every stay, or did it need the regime's
other parts? And does the stop gain at 200 and 1,000 stays, where a fifth held out is 40 and
200 stays, the budgets where the thesis is read and where the registered rule does not reach?

**Design.** Three campaigns, every one scored on the fifth held out by seed 101 over seeds 1 to
10, the validation side unread, under ADR-0047; the stop holds a fifth of the labelled stays
out by unit with a patience of ten epochs, the schedule's epochs its cap.

| Campaign | Budget | Rate | Cap | Candidates |
|---|---|---|---|---|
| `f7e06744…` (`campaigns/small-shape-stop-all-physionet2012.toml`) | every stay | 0.000333 | 30 epochs | the stop alone in the control's shape, run again; the small shape under the stop alone |
| `4230d83b…` (`campaigns/stop-200-physionet2012.toml`) | 200 | 0.003 | 154 epochs | the control; the stop alone; the small shape; the small shape under the stop alone |
| `53e8d389…` (`campaigns/stop-1000-physionet2012.toml`) | 1,000 | 0.000333 | 32 epochs | the same four |

The rates are the ones the earlier campaigns ran the network at each budget, so within a
campaign the candidates differ in shape and in the stop and in nothing else. The small shape is
64 wide, 16 heads, 2 blocks, feed-forward 128. At every stay the stop in the control's shape
runs again because a cell of that shape does not repeat to the stay on a GPU; at 200 the four
cells pair with `a9ab91cc…` (the small shape at 200: control 0.660, small 0.686, the patch model
0.696 on the same fifth and seeds). Three orders on one Colab GPU.

**Reading.** By `scripts/campaign_pairs_report.py`, within each campaign: at every stay, the
small shape under the stop against the stop in the control's shape, and against the small shape
under the regime whole (0.818, campaign `e1959b4f…`, deterministic cells); at 200 and 1,000,
the stop against the control in each shape, and the small shape against the control's shape
under the stop. A candidate *gains* where the paired interval over stays lies above zero and
the mean over seeds exceeds twice its standard error.

**Predictions.**

1. The checks hold: the stop in the control's shape lands within 0.01 of 0.810 at every stay,
   and the control at 200 within 0.015 of 0.660.
2. At every stay the small shape under the stop alone lands at or above the small shape under
   the regime whole, 0.818 to 0.830: the regime's other parts cost the control's shape 0.011
   and are not expected to help the small one. It gains over the stop in the control's shape by
   0.01 to 0.02, seed by seed and not by the interval.
3. At 1,000 stays the stop gains in both shapes by both conditions, 0.02 or more: the run is
   32 epochs of cosine decay over 1,000 stays, and the held-out 200 stays score an epoch to
   about 0.05.
4. At 200 stays the stop gains in the control's shape by the paired interval, 0.02 to 0.04,
   and the spread over seeds does not fall: 154 epochs at 0.003 over 200 stays overfit, but 40
   held-out stays with five or six deaths score an epoch to about 0.11, so the epoch kept is
   noisy. The small shape under the stop lands at or above the patch model's 0.696.

**What follows.** If 2 holds, the stop alone is the recipe of every shape at every stay, and
the two-block backbone's arms read under it there; if the small shape under the stop lands
below the regime whole by more than twice the standard error, the regime's other parts are its
recipe and the arms of the two-block backbone declare them. If 3 holds, the registered rule
extends to 1,000 stays, by a register row, before the next backbones' campaigns. If 4 holds in
both conditions the rule extends to 200 as well; if it holds by the interval only, the stop at
200 stays a declared variant beside the fixed epochs there, read again on the pretrained arms
under more seeds before it is registered; if the stop loses at 200, the fixed epochs stand at
200 and 50, and the rule's lower edge is 1,000. No verdict of the registered grids changes.

**Limitations.**

- One share and one patience at every budget; at 200 stays a larger share would leave fewer
  stays to learn from, and a longer patience would spend the cap.
- At 1,000 stays nothing of this network was read on this fifth before; the check there is the
  control's level against the registered grid's, read under another division.
- Ten seeds read a difference to about 0.013 at 200 stays and about 0.007 at every stay.

## 2026-10-03 — Colab G4: the stop alone in the small shape, and the stop at 200 and 1,000 stays

**Question.** Does the small shape gain under the stop alone, and does the stop gain at 200 and
1,000 stays? The design and predictions are in the section declared on 2026-10-03 for the stop
alone in the small shape.

**Conditions.** Commit `3c30f410`; campaigns `f7e06744…` (every stay), `4230d83b…` (200),
`53e8d389…` (1,000); orders `1391bd94…`, `571e8098…`, `d6b5e0a1…`. One Colab G4, CUDA, three
processes under CUDA MPS; tier M; 100 cells in 1,738 s. Scored on the fifth held out by seed
101, 800 stays, seeds 1 to 10. Read with `data/report/t42d/stop-pairs.sh`; CSV under
`data/report/t42d/stop/`.

**Area under the ROC curve on the 800 scored stays**, mean over seeds 1 to 10, the seeds'
standard deviation, and the seconds a cell took (three cells shared the GPU).

| Budget | Candidate | Mean | SD over seeds | Seconds a cell |
|---|---|---|---|---|
| every stay | the stop alone, control's shape (run again) | 0.810 | 0.009 | 120 |
| every stay | the small shape under the stop alone | 0.805 | 0.011 | 43 |
| every stay | the small shape under the regime whole (`e1959b4f…`, for reference) | 0.818 | 0.011 | 37 |
| 1,000 | control (32 epochs, last weights) | 0.750 | 0.032 | 76 |
| 1,000 | the stop alone | 0.769 | 0.025 | 47 |
| 1,000 | the small shape | 0.763 | 0.020 | 26 |
| 1,000 | the small shape under the stop | 0.777 | 0.019 | 17 |
| 200 | control (154 epochs, last weights) | 0.656 | 0.023 | 73 |
| 200 | the stop alone | 0.668 | 0.043 | 9 |
| 200 | the small shape | 0.690 | 0.046 | 26 |
| 200 | the small shape under the stop | 0.652 | 0.085 | 4 |

**Gain in area**, paired over stays (95 % interval), and seed by seed (mean ± standard error).
A candidate *gains* where the interval lies above zero and the mean over seeds exceeds twice
its standard error.

| Budget | Control | Candidate | Paired gain | Seed by seed | Reads |
|---|---|---|---|---|---|
| every stay | stop alone of `e1959b4f…` (0.810) | stop alone here | −0.000 [−0.005; +0.004] | −0.000 ± 0.003 | check holds |
| every stay | stop alone | small shape under the stop | −0.005 [−0.015; +0.005] | −0.005 ± 0.005 | no |
| every stay | small shape under the regime (0.818) | small shape under the stop | **−0.013 [−0.025; −0.002]** | −0.013 ± 0.005 | **loses** |
| 1,000 | control | stop alone | +0.018 [+0.003; +0.033] | +0.018 ± 0.013 | interval yes, seeds no |
| 1,000 | small shape | small shape under the stop | +0.013 [+0.002; +0.024] | +0.013 ± 0.007 | interval yes, seeds no |
| 1,000 | stop alone | small shape under the stop | +0.008 [−0.004; +0.019] | +0.008 ± 0.005 | no |
| 1,000 | control | small shape | +0.013 [−0.003; +0.028] | +0.013 ± 0.011 | no |
| 200 | control of `a9ab91cc…` (0.660) | control here | −0.004 [−0.021; +0.013] | −0.004 ± 0.015 | check holds |
| 200 | small shape of `a9ab91cc…` (0.686) | small shape here | +0.004 [+0.002; +0.006] | +0.004 ± 0.003 | a T4 and a G4 differ by 0.004 |
| 200 | control | stop alone | +0.012 [−0.008; +0.031] | +0.012 ± 0.013 | no |
| 200 | small shape | small shape under the stop | **−0.038 [−0.059; −0.018]** | −0.038 ± 0.022 | **loses by the interval** |
| 200 | stop alone | small shape under the stop | −0.016 [−0.032; +0.002] | −0.016 ± 0.025 | no |

**Conclusions.**

1. Prediction 1 holds: the stop in the control's shape repeats to 0.000 at every stay and the
   control at 200 lands 0.004 from its earlier reading.
2. Prediction 2 fails. Under the stop alone the small shape lands at 0.805, 0.005 under the
   control's shape and 0.013 under itself under the regime whole, by both conditions. The
   regime's other parts (the class weight, the channels withheld, dropout, the constant rate)
   cost the control's shape 0.011 and gain the small shape 0.013: they regularise a network of
   77,000 parameters and burden one of 4.8 million. As declared, the regime whole is the recipe
   of the two-block shape, and the two-block backbone's arms declare it; the stop alone is the
   recipe of the control's shape.
3. Prediction 3 fails in its second condition. At 1,000 stays the stop gains 0.018 in the
   control's shape and 0.013 in the small one by the paired interval, under 0.02 and at 1.4 and
   1.9 standard errors seed by seed. The registered rule does not extend to 1,000 stays; the
   stop there is a declared variant beside the fixed epochs, to be read on the pretrained arms
   under more seeds before it is registered, as declared for the same case at 200.
4. Prediction 4 fails the other way. At 200 stays the stop gains nothing in the control's shape
   (+0.012, 1.0 standard error) and loses 0.038 in the small shape by the interval, with a
   spread over seeds of 0.085 and three seeds near 0.5. The seconds say why: the stopped cells
   took 4 and 9 s against 26 and 73 s for the whole 154 epochs, so the stop fired at about the
   fifteenth epoch. Under the stop the learning side is 160 stays, ten steps an epoch, and the
   patience of ten epochs is 100 steps, inside the warm-up of 154. The stop kept weights from
   the warm-up, chosen by the noise of 40 held-out stays (an epoch's area there is read to about
   0.11). The fixed epochs stand at 200 and 50; the registered rule's lower edge stays the budget
   of every stay.
5. A patience counted in epochs is 1,600 steps at every stay, 500 at 1,000 and 100 at 200. The
   stop as registered is a recipe for the budget where it was read, not a rule of training that
   scales with the budget. A stop whose patience is counted in steps, or begins after the
   warm-up, is a variant this reading did not run; whether it enters the next ticket's first
   declaration at 200 and 1,000 is a decision for that ticket. No verdict of the registered
   grids changes.

**Limitations.**

- The epochs a stopped cell trained are inferred from its seconds; the runtime does not record
  them.
- One share and one patience at every budget, both the published network's; at 200 stays a
  larger share would leave fewer stays to learn from.
- At 1,000 stays nothing of this network was read on this fifth before; the control's 0.750 has
  no earlier reading to check against.
- The small shape's cells repeat to the stay on one GPU and differ by 0.004 between a T4 and a
  G4, so a difference of that size between campaigns run on different cards is the card's.

## 2026-10-04 — declared before the run: the recipe at 20, 50 and 200 stays and at 50 windows, and the seeds' spread under it

**Question.** The curve over the scale of pretraining (`docs/preregistration.md`, "The scale of
pretraining"; ADR-0049) reads every point under one recipe of adaptation, chosen on the tuning
side before the first point is read on the validation side. Does solving the head before the
first step, or a stop whose patience is counted in steps past the warm-up and whose held-out
stays hold both outcomes (ADR-0050), replace each arm's setting in force at the curve's budgets?
And how far does one seed move the area of the network from nothing and of the probe solved in
closed form under that recipe, which sets the least gain and the floor's fixed part?

**Design.** Four campaigns under the mixture of four (`backbone-mixed4-m`, weights
`sha256:0d81c01e…`), scored on the fifth held out by seed 101 over seeds 1 to 10, the validation
side unread. The intensive-care task is `a6a9c653…`, FD001 `be30c0ac…`, both over the
publications the backbone was pretrained on.

| Campaign file | Budget | Each arm from | Knobs turned |
|---|---|---|---|
| `campaigns/recipe-20-physionet2012.toml` | 20 stays | its setting at 50 | the head solved first |
| `campaigns/recipe-50-physionet2012.toml` | 50 stays | its setting at 50 | the head solved first; the stop; both |
| `campaigns/recipe-200-physionet2012.toml` | 200 stays | its setting at 200 | the head solved first; the stop; both |
| `campaigns/recipe-50-fd001.toml` | 50 windows | its setting at 50 | the head solved first; the stop by engine; both |

Each campaign also holds the probe solved in closed form (`frozen_ridge`) and the same probe
over the encoder at the control's initialisation (`untrained_ridge`), the first and zeroth
points of the curve. The stop holds a fifth of the labelled units out, within each outcome on
the intensive-care task, and waits 500 steps past the warm-up: a stopped run at 50 or 200 is
capped at about 1,500 steps, its warm-up about 150, and 500 steps is the patience the stop gained
under at 1,000 stays (2026-10-03, ten epochs of 50 steps). Twenty stays hold two or three deaths, so no stop runs
there: it would score one death and leave the solved head one to learn from. The settings in
force are the selections' choices on the earlier publication, `selection-scratch-*` and
`selection-fine-tuning-*`, since a selection on another task cannot be named as a tuned choice.

**Reading, declared beforehand.** By `scripts/campaign_pairs_report.py`, within each campaign
and arm, every variant against the arm's setting in force, paired over the scored units and
pooled over the seeds.

- A variant *replaces* the setting where its paired 95 % interval lies above zero (a gain in area
  on the intensive-care task, a reduction in RMSE on FD001) and its mean over seeds exceeds twice
  its standard error, the rule of 2026-10-03. Among the variants that replace it, the one with
  the largest mean gain is chosen; where none does, the setting in force stands. Each arm is
  chosen on its own: the comparison holds the knobs on offer equal, and the control is not held
  below its best.
- The choice per arm and budget is registered as variants before the first point of the curve
  is read on the validation side.
- *Least gain*, at each of 20, 50 and 200 stays: the larger of the two standard deviations over
  the ten seeds, of the network from nothing at its chosen variant and of `frozen_ridge`, rounded
  up to 0.005. *Floor's fixed part*: the smallest standard deviation of the network from nothing
  at its chosen variant over the three budgets, rounded down to 0.005 and not below 0.01. The
  reading of 2026-09-29, under the recipe. FD001 keeps its registered relative rule.
- `untrained_ridge` against `frozen_ridge` and against the network from nothing is reported as
  measured and chooses nothing.

**Predictions.**

1. The checks hold: the network from nothing at its setting lands within 0.02 of 0.623 at 50
   stays and within 0.015 of 0.66 at 200 (the same fifth, the earlier publication of the same
   stays).
2. The head solved first replaces full fine-tuning's setting at 50 stays, by 0.01 to 0.03, and
   on FD001 at 50 windows; at 200 stays it gains under 0.01 and does not replace. It replaces
   nothing for the network from nothing at any budget: a head solved over states of an encoder
   at its initialisation is forgotten in the first hundred steps.
3. The stop replaces the network from nothing's setting at 200 stays, by 0.01 to 0.03: the
   stop that fired inside the warm-up there gained +0.012 without replacing, and a patience
   that begins after the warm-up removes that cause. At 50 stays it replaces nothing: eleven
   held-out stays with two deaths read an epoch to about 0.2.
4. Both knobs together replace no arm by more than the better of the two alone.
5. `untrained_ridge` lands below the network from nothing at 50 stays, between 0.55 and 0.62,
   and `frozen_ridge` above `untrained_ridge` by at least 0.03.
6. At 50 stays one seed moves the network from nothing by 0.04 to 0.06 and `frozen_ridge` by
   0.02 to 0.04, so the least gain at 50 lies between 0.04 and 0.06.

**What follows.** The chosen variants and the thresholds enter `docs/preregistration.md` by one
row before any campaign of the curve reads the validation side. If the stop replaces at 200 or
50 stays, it is the recipe there for the curve only; the registered grids' recipe is unchanged.
If the head solved first replaces full fine-tuning's setting, the trained probe is read beside
it under its setting in force. No verdict of the registered grids changes.

**Limitations.**

- One share and one patience; a patience of 300 or 800 steps is not read.
- The settings in force were chosen on the earlier publication of the same stays; a rate that
  suits the new one better is not searched.
- The standard deviation of the chosen variant is read on the cells that chose it; the choice
  goes by the mean, not the spread, but the two are not independent.
- Ten seeds read a difference to about 0.016 at 50 stays and 0.013 at 200.
- The stays scored are a fifth of the tuning side the earlier selections and the stop's reading
  of 2026-10-03 were also scored on.

## 2026-10-04 — Colab G4: the recipe at 20, 50 and 200 stays and at 50 windows, and the seeds' spread under it

**Question.** Does the head solved first, or the stop counted in steps and divided by outcome,
replace an arm's setting at the curve's budgets, and how far does one seed move the area? The
design, the reading and the predictions are in the section declared on 2026-10-04 for the
recipe.

**Conditions.** Commit `4b07dc94`; campaigns `da27c29a…` (20 stays), `5098676c…` (50),
`e4ef8d16…` (200), `b9dc55a7…` (FD001, 50 windows), under the mixture of four
(`sha256:0d81c01e…`); one order each, four processes on one Colab G4, the longest order in
5,198 s. Scored on the fifth held out by seed 101, seeds 1 to 10: 800 stays, or 16 engines and
523 windows. Read with `data/report/l1/recipe-pairs.sh` (`scripts/campaign_pairs_report.py`);
CSV under `data/report/l1/recipe/`.

**Area under the ROC curve**, pooled over the ten seeds, and its standard deviation over them.
*Head* is the head solved first; *stop* the stop of 500 steps past the warm-up, divided by
outcome. The network from nothing runs at 0.003 at 200 stays and at the rate in force below.

| Candidate | 20 stays | 50 stays | 200 stays |
|---|---|---|---|
| network from nothing | 0.589 (0.038) | 0.625 (0.049) | 0.643 (0.047) |
| from nothing, head | 0.600 (0.057) | 0.630 (0.054) | 0.636 (0.060) |
| from nothing, stop | — | 0.622 (0.059) | 0.646 (0.063) |
| from nothing, head and stop | — | 0.629 (0.063) | 0.648 (0.070) |
| full fine-tuning | 0.573 (0.054) | 0.616 (0.032) | 0.689 (0.040) |
| fine-tuning, head | 0.550 (0.059) | 0.616 (0.047) | 0.672 (0.043) |
| fine-tuning, stop | — | 0.575 (0.035) | 0.663 (0.031) |
| fine-tuning, head and stop | — | 0.609 (0.050) | 0.653 (0.041) |
| probe in closed form (`frozen_ridge`) | 0.557 (0.054) | 0.603 (0.056) | 0.657 (0.033) |
| the same probe at initialisation | 0.541 (0.037) | 0.571 (0.045) | 0.606 (0.044) |

**Gain in area against the arm's setting**, paired over stays (95 % interval), and seed by seed
(mean ± standard error). A variant *replaces* where the interval lies above zero and the mean
exceeds twice its standard error.

| Budget | Variant | Paired gain | Seed by seed | Replaces |
|---|---|---|---|---|
| 20 | from nothing, head | +0.010 [−0.011; +0.032] | +0.010 ± 0.013 | no |
| 20 | fine-tuning, head | **−0.023 [−0.038; −0.008]** | −0.023 ± 0.016 | no, loses |
| 50 | from nothing, head / stop / both | +0.005 / −0.003 / +0.005, each interval across zero | ≤ 0.6 SE | no |
| 50 | fine-tuning, head | +0.000 [−0.016; +0.016] | +0.000 ± 0.015 | no |
| 50 | fine-tuning, stop | **−0.042 [−0.065; −0.019]** | −0.042 ± 0.014 | no, loses |
| 50 | fine-tuning, both | −0.007 [−0.029; +0.015] | −0.007 ± 0.015 | no |
| 200 | from nothing, head / stop / both | −0.006 / +0.004 / +0.005, each interval across zero | ≤ 0.4 SE | no |
| 200 | fine-tuning, head | **−0.017 [−0.033; −0.002]** | −0.017 ± 0.013 | no, loses |
| 200 | fine-tuning, stop | **−0.026 [−0.042; −0.009]** | −0.026 ± 0.012 | no, loses |
| 200 | fine-tuning, both | **−0.036 [−0.058; −0.014]** | −0.036 ± 0.013 | no, loses |

The probes, as measured: `frozen_ridge` against the network from nothing −0.033 [−0.059;
−0.007] at 20, −0.022 [−0.053; +0.010] at 50, +0.015 [−0.017; +0.045] at 200; against the same
probe at initialisation +0.016 [−0.005; +0.037], **+0.032 [+0.007; +0.057]** and **+0.051
[+0.016; +0.085]**.

**FD001 at 50 windows**, RMSE pooled over the ten seeds (standard deviation), and the reduction
against the arm's setting.

| Candidate | RMSE | Reduction | Seed by seed | Replaces |
|---|---|---|---|---|
| network from nothing | 16.2 (1.0) | | | |
| from nothing, head | 17.1 (1.5) | −0.91 [−1.39; −0.42] | −0.88 ± 0.47 | no, loses |
| from nothing, stop | 20.7 (6.3) | −4.50 [−6.71; −2.47] | −3.65 ± 1.94 | no, loses |
| from nothing, head and stop | 19.3 (2.7) | −3.10 [−4.12; −2.10] | −2.96 ± 0.82 | no, loses |
| full fine-tuning | 22.7 (1.6) | | | |
| fine-tuning, head | 21.9 (2.2) | **+0.84 [+0.21; +1.46]** | +0.89 ± 0.42 | **yes** |
| fine-tuning, stop | 22.9 (1.5) | −0.21 [−0.90; +0.44] | −0.21 ± 0.47 | no |
| fine-tuning, head and stop | 22.8 (3.1) | −0.03 [−1.09; +0.94] | +0.11 ± 1.07 | no |
| probe in closed form | 23.9 (4.6) | −7.74 against from nothing | | |
| the same probe at initialisation | 18.8 (1.9) | −2.65 against from nothing; +5.09 against the probe under the backbone | | |

**Conclusions.**

1. On the intensive-care task no variant replaces any arm's setting at any budget: the settings
   in force stand for the network from nothing and full fine-tuning at 20, 50 and 200 stays.
   Five variants of full fine-tuning lose by the interval; the stop costs it most (−0.042 at 50).
   On FD001 the head solved first replaces full fine-tuning's setting (−0.84 RMSE, 3.7 %), and
   every stop loses for the network from nothing.
2. Predictions: 1 holds at 50 (0.625) and misses by 0.002 at 200 (0.643); 2 holds on FD001 only;
   3 fails, the stop gains the network from nothing nothing at 200 (+0.004); 4 holds; 5 holds,
   the probe at initialisation at 0.571 and pretraining adding 0.032 to it at 50; 6 holds for the
   network from nothing (0.049) and fails for the probe, which spreads by 0.056, not 0.02–0.04.
3. By the reading declared beforehand the least gain is 0.055 at 20 stays, 0.060 at 50 and 0.050
   at 200 (the probe's spread at 20 and 50, the network's at 200), and the floor's fixed part
   0.035 (the network's 0.038 at 20, rounded down).
4. At the curve's first point the probe under the backbone does not beat the network from
   nothing at any budget, and beats the same probe at initialisation by 0.03 to 0.05 from 50
   stays up. On FD001 it is worse than the probe at initialisation by 5.1 RMSE: the mixture's
   states serve a linear reading of the remaining life worse than an untrained encoder's.
5. A head solved in closed form does not carry over to fine-tuning on the stays: the solved
   head is the probe's (0.603 at 50), and fine-tuning from it lands where fine-tuning from a
   drawn head does.

**Limitations.**

- The settings in force were chosen on the earlier publication of the same stays; no rate was
  searched again.
- The spread of the network from nothing is read on the cells that also chose its setting.
- One share and one patience of the stop; the probe phase is the closed form only.
- Ten seeds read a difference to about 0.015; at 20 stays the draw holds two or three deaths.

## 2026-10-04 — declared before the run: the first point of the curve over the scale of pretraining

**Question.** At the curve's first point, the mixture of four of about 10⁸ values, how far does
the probe solved in closed form stand from the network from nothing at 20, 50 and 200 stays and
at 50 windows of FD001, how far apart do two pretrainings of the mixture put it, and does the
larger shape, at either rate, move it? The claim, the points and the rules are in
`docs/preregistration.md`, "The scale of pretraining"; the recipe and thresholds were registered
on 2026-10-04.

**Design.** Three campaign files, each defined under every backbone of the point, scored on the
validation side over seeds 1 to 10, the same stays, engines and draws under every backbone.

| Campaign file | Budgets | Candidates |
|---|---|---|
| `campaigns/yardstick-low-physionet2012.toml` | 20, 50 stays | network from nothing; `frozen_ridge`; trained probe; full fine-tuning; `untrained_ridge` |
| `campaigns/yardstick-200-physionet2012.toml` | 200 stays | the same, the network from nothing at 0.003 |
| `campaigns/yardstick-50-fd001.toml` | 50 windows | the same at the turbofan's settings, full fine-tuning starting its head solved |

| Backbone | Weights | Shape |
|---|---|---|
| `backbone-mixed4-m` (seed 1) | `sha256:0d81c01e…` | 4.8M |
| `backbone-mixed4-m-seed2` | `sha256:94f00d72…` | 4.8M |
| `backbone-mixed4-512x8-m` (1e-3) | `sha256:d047a2d0…` | 25.3M |
| `backbone-mixed4-512x8-m-5e-4` | when accepted (`manual-handoff.md`, 2026-10-04) | 25.3M |

The network from nothing and `untrained_ridge` take each backbone's shape, so under the two
4.8M backbones they run twice; the second run checks that a cell repeats on the accelerator and
is not read otherwise.

**Reading, declared beforehand.** By `scripts/campaign_pairs_report.py`, pairs over the scored
units, pooled over the seeds.

- *Under each backbone*, at every budget: `frozen_ridge`, the trained probe and full fine-tuning
  against the network from nothing, read by the registered rules (the least gain 0.055, 0.060 and
  0.050 in area, 10 % in RMSE on FD001; the floor; Holm over the campaign's family); and
  `frozen_ridge` against `untrained_ridge`, pretraining's share over the probe's form.
- *The seeds' difference*, the bound a step of the curve must exceed: `frozen_ridge` under seed 2
  less under seed 1 at 50 stays, paired; its absolute mean is the difference the registered rule
  of the slope names. Reported at 20 and 200 stays and on FD001 beside it.
- *The shape*: `frozen_ridge` under the larger shape against today's at 50 stays, at the rate the
  pretraining note's rule keeps for the larger shape (`manual-handoff.md`, 2026-10-04: the half
  rate where its probe is higher by the paired interval, 1e-3 otherwise).
- The first point of each shape's curve is the one read here; the curve itself is read when the
  next point is.

**Predictions** — those registered for the stage, made specific to this point:

1. Under each 4.8M backbone the interval of `frozen_ridge` against the network from nothing holds
   zero at 50 and 200 stays. At 20 stays it lies below zero, as on the tuning side (−0.033),
   where the registered prediction has it hold zero; the registered one is the one judged.
2. `untrained_ridge` lies below the network from nothing at 50 stays by the interval, and below
   `frozen_ridge` by 0.02 to 0.05.
3. The two seeds' `frozen_ridge` differ by less than 0.01 at 50 stays.
4. The larger shape's `frozen_ridge` gains no more than 0.01 over today's at 50 stays, at either
   rate.
5. On FD001 `frozen_ridge` lies above the network from nothing in RMSE under every backbone, and
   above `untrained_ridge`, as on the tuning side (23.9 against 18.8).
6. Full fine-tuning gains over the network from nothing at 200 stays by the interval, 0.03 to
   0.05, as on the tuning side (+0.046), and is read as secondary.

**Limitations.**

- One point of the curve: no slope is read here.
- The validation side has been read before under the mixture of five and the stays alone at 50
  and 200 stays (grids of 2026-09-30); the backbones and publication are new, the stays are not.
- 20 stays hold two or three deaths in a draw; their cells are read descriptively.
- One pretraining run per backbone beyond the pair of seeds; the larger shape's seed spread is
  assumed to be today's shape's.
