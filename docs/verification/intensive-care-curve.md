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
| Variants | each candidate at its selection's choice per budget, on the tuning side (`seed-spread`, selections of 2026-09-29) |
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
   backbone it clears no floor at any budget; under the stays alone it clears it at 50 and
   1,000, by 0.061 and 0.015, and not at every stay.
2. Under the mixed backbone no way of using it beats the network from nothing at any budget; the
   closed-form probe is worse at 1,000 and every stay. Under the stays alone the frozen probe
   gains 0.055, 0.047 and 0.031 in area at 200, 1,000 and every stay. The mixture, which learnt
   the stays about as well as eight epochs on them alone (`manual-handoff.md`, 2026-09-29), is
   the weaker backbone for this task. This comparison was named a diagnostic in the registration
   but given no prediction beforehand, so it is descriptive only.
3. Trees on per-channel features and MiniRocket are the best candidates from 200 stays up, 0.76
   at 200 and 0.86 and 0.85 at every stay, above every network. The patch model beats the
   pretrained arms at 50 and 200 under the mixed backbone.
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
