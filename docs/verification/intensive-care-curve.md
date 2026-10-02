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
