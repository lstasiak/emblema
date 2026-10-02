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
