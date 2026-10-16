# The recipe of fine-tuning a pretrained backbone

Whether full fine-tuning reads the intensive-care stays better when the encoder's pretrained
weights learn at a share of the head's rate, with the head drawn or solved first, as Kumar et
al. (2022) prescribe. The head solved first alone replaced nothing (`intensive-care-curve.md`,
2026-10-05), and a smaller rate for the whole network moved fine-tuning by at most 0.012 at 200
stays; from 500 stays up fine-tuning reads the stays worse than the network from nothing
(`verdict-statistics.md`, 2026-10-13). The rules in force are in `docs/preregistration.md`,
the rows "The reader of the stage" and "The recipe"; this note changes no rule until a
replacement is read.

## 2026-10-16 — declared before the run: the encoder's share of the rate at 200 and 500 stays

**Question.** Does fine-tuning with the encoder's weights slowed to a tenth or a hundredth of
the schedule's rate, the head drawn or solved first, replace fine-tuning's setting at 200 stays,
where fine-tuning leads the network from nothing, and at 500, where it trails it? And does the
slowed fine-tuning at 500 still trail the network from nothing?

**The data.**

- *The knob.* `encoder_rate`, a knob of the training regime (ADR-0055): the encoder's weights
  learn at that share of the schedule's rate, the head and the rows grown for the task's new
  channels at the whole of it. Under a share of one the optimiser is handed what it always
  was; the loop under groups at the whole rate repeats the plain run bit for bit (test).
- *The campaigns.* `campaigns/recipe-encoder-rate-200-physionet2012.toml` and
  `-500-`: at each budget the network from nothing and fine-tuning at their settings in force
  (at 200 the rate 0.003 and the fixed epochs; at 500 the stop of ten epochs over a fifth
  held out, for both arms), and four variants of fine-tuning: the encoder at 0.1 and at 0.01,
  each with the head drawn and with the head solved first. Half the tuning side held out by
  seed 101, seeds 1 to 10: 60 cells a campaign.
- *The backbones.* The mixture of four at its first two pretraining seeds
  (`sha256:0d81c01e…`, `94f00d72…`), as the reader's networks ran: four campaigns, 240 cells,
  on Kaggle T4 ×2 over two sessions, about 400 s a cell at 200 stays and 110 to 230 s under
  the stop at 500 (`verdict-statistics.md`, 2026-10-13).
- *The trial.* Every plan of both files builds under the declared worker settings (the
  campaign-file test); the runtime under the knob is read on the test corpus (a slowed encoder
  moves less, the head still learns). The first cell of the first order is the trial on the
  platform, and its time is recorded in the measured section.
- *The reading.* `scripts/campaign_pairs_report.py`, paired over stays and seed by seed, under
  each pretraining: (R) each variant less fine-tuning's setting at its budget; (C) each variant
  less the network from nothing at its setting; (C0) fine-tuning's setting less the network
  from nothing, the pair already read in L-16, as the check. A variant *replaces* the setting
  where the interval of R lies above zero and the mean exceeds twice its standard error under
  both pretrainings, the rule the recipes were chosen by.

**Predictions.**

1. *At 200 stays no variant replaces fine-tuning's setting.* The gains of R lie within −0.02
   to +0.02: the rate variants of the whole network moved fine-tuning by at most 0.012 there,
   and the head solved first cost 0.017.
2. *At 500 stays the encoder at 0.1 with the head solved first replaces fine-tuning's setting*,
   by 0.01 to 0.03 under both pretrainings, and its pair against the network from nothing
   crosses zero, where fine-tuning's setting lies at −0.017 [−0.028; −0.006] and −0.011
   [−0.024; +0.001]. The features the pretraining made are kept where the encoder moves
   little while the head is already placed.
3. *The encoder at 0.01 reads below the encoder at 0.1* at both budgets, with the head drawn
   or solved: a hundredth of the rate adapts too little in the budget's steps.

**What follows.**

- If prediction 2 holds: fine-tuning's setting at 500 stays becomes the slowed variant by a
  row of the rule, the reading of the few-shot budgets (20, 50 and 100 stays) that the
  literature places the gain at runs under it, and the product's claim that a pretrained
  backbone is never a worse start than a draw is read there.
- If it fails: the encoder's rate is not the cause of the loss from 500 stays up, the setting
  stays, and the next step changes the pretext (a contrastive objective) or continues the
  pretraining on the task's own stays, not the recipe.
- A cell that refuses, or a campaign that does not run whole, is named and its backbone left
  out; a reading short of both pretrainings at 500 stops the replacement.

**Limitations.**

- One share over the whole encoder; a decay of the rate by layer is not read.
- The pairs are under two pretrainings of one recipe; the error between pretrainings s =
  0.0057 at 500 stays bounds what a replacement means for the recipe, not for this backbone.
- The budgets are the reader's; the few-shot budgets the literature places the gain at are
  read only after the recipe is settled.

## 2026-10-16 — Colab, one GPU: the encoder's share of the rate at 200 and 500 stays, measured

**Question.** The one declared above: does fine-tuning with the encoder slowed to a tenth or a
hundredth of the rate, the head drawn or solved first, replace fine-tuning's setting at 200 or
500 stays, and does it still trail the network from nothing at 500?

**Conditions.** Commit `be36f421`; four campaigns defined and ordered from it, run on Colab on
one G4 GPU as four concurrent processes under CUDA MPS (`data/report/l22/colab-cell.sh`). Wall
clock of an order of 60 cells: 1,963 and 2,288 s at 500 stays, 4,101 and 4,134 s at 200. A cell
took 17 to 52 s under the stop at 500 and 86 to 98 s at 200 while four processes shared the
GPU, 46 to 52 s once the 500 campaigns had ended. The first cell of the first order was the
trial. Accepted by `data/report/l22/accept.sh`; pairs by `data/report/l22/pairs.sh`
(`scripts/campaign_pairs_report.py`, 10,000 resamples); CSV under `data/report/l22/pairs/`.

**Area on the 2,000 stays held out**, mean over seeds 1 to 10 (the seeds' standard
deviation), under the first and the second pretraining.

| Budget | from nothing | fine-tuning's setting | encoder 0.1 | encoder 0.01 | 0.1, head solved | 0.01, head solved |
|---|---|---|---|---|---|---|
| 200 | 0.670, 0.660 (0.04, 0.03) | 0.682, 0.688 (0.02, 0.03) | 0.620, 0.673 (0.03) | 0.594, 0.680 (0.02, 0.03) | 0.632, 0.681 (0.04) | 0.623, 0.682 (0.04, 0.03) |
| 500 | 0.747, 0.745 (0.03) | 0.724, 0.733 (0.04, 0.02) | 0.663, 0.718 (0.04, 0.03) | 0.614, 0.626 (0.04, 0.08) | 0.685, 0.732 (0.02, 0.01) | 0.690, 0.716 (0.02) |

**The pairs**, paired over stays (95 % interval) and seed by seed (± standard error), under
the first and the second pretraining. A variant replaces where the interval lies above zero and
the mean exceeds twice its error under both.

| Kind | Budget | Variant | First pretraining | Second pretraining |
|---|---|---|---|---|
| R, less fine-tuning's setting | 200 | 0.1 | −0.062 [−0.079; −0.045] ± 0.009 | −0.014 [−0.028; −0.001] ± 0.012 |
| R | 200 | 0.01 | −0.088 [−0.111; −0.065] ± 0.008 | −0.007 [−0.025; +0.010] ± 0.013 |
| R | 200 | 0.1, solved | −0.050 [−0.069; −0.031] ± 0.009 | −0.007 [−0.020; +0.006] ± 0.013 |
| R | 200 | 0.01, solved | −0.059 [−0.082; −0.036] ± 0.010 | −0.005 [−0.020; +0.009] ± 0.012 |
| R | 500 | 0.1 | −0.061 [−0.076; −0.046] ± 0.019 | −0.015 [−0.026; −0.004] ± 0.011 |
| R | 500 | 0.01 | −0.110 [−0.132; −0.087] ± 0.021 | −0.107 [−0.129; −0.086] ± 0.023 |
| R | 500 | 0.1, solved | −0.038 [−0.056; −0.020] ± 0.012 | −0.001 [−0.015; +0.013] ± 0.006 |
| R | 500 | 0.01, solved | −0.033 [−0.054; −0.013] ± 0.010 | −0.018 [−0.034; −0.001] ± 0.008 |
| C, less the network from nothing | 200 | 0.1 / 0.01 | −0.050 / −0.076 | +0.014 / +0.021 |
| C | 200 | 0.1 / 0.01, solved | −0.038 / −0.047 | +0.021 [+0.004; +0.039] / +0.023 [+0.003; +0.042] |
| C | 500 | 0.1 / 0.01 | −0.084 / −0.133 | −0.027 / −0.119 |
| C | 500 | 0.1 / 0.01, solved | −0.061 / −0.056 | −0.013 [−0.030; +0.003] / −0.030 |
| C0, fine-tuning's setting less the network from nothing | 200 | | +0.012 [−0.001; +0.026] ± 0.012 | +0.028 [+0.013; +0.043] ± 0.010 |
| C0 | 500 | | −0.023 [−0.033; −0.013] ± 0.017 | −0.012 [−0.023; −0.000] ± 0.012 |

**The predictions.**

1. *At 200 stays no variant replaces fine-tuning's setting*: holds, every interval of R lies at
   or below zero. *The gains lie within ±0.02*: holds under the second pretraining (−0.005 to
   −0.014) and fails under the first (−0.050 to −0.088).
2. *At 500 stays the encoder at 0.1 with the head solved first replaces fine-tuning's setting by
   0.01 to 0.03*: fails. It trails by 0.038 [0.020; 0.056] under the first pretraining and
   ties under the second (−0.001 [−0.015; +0.013]). Its pair against the network from nothing
   crosses zero under the second pretraining only (−0.013 [−0.030; +0.003]); under the first it
   lies at −0.061.
3. *The encoder at 0.01 reads below the encoder at 0.1*: holds with the head drawn at 500, by
   0.05 and 0.09, and at 200 under the first pretraining; with the head solved the two shares
   lie within 0.02 of each other everywhere, and at 200 under the second pretraining all four
   variants lie within 0.01 of one another.

**Conclusions.**

- No variant replaces fine-tuning's setting at either budget under either pretraining; the
  setting in force stands, and the reading of the few-shot budgets under a slowed variant does
  not run.
- A slowed encoder keeps the pretraining's features, and keeps the pretrainings apart: under
  the first pretraining every variant lies below the same variant under the second, by 0.05
  to 0.09 in six of eight cells and by 0.01 and 0.03 for the encoder at 0.01 at 500, while fine-tuning at the whole rate under the two lies within 0.01 (0.682 and 0.688
  at 200; 0.724 and 0.733 at 500). Fine-tuning's gain at 200 and its loss from 500 up are
  therefore not made by the recipe moving the encoder too far; the whole rate moves it away
  from features that read the stays worse than the ones the labels build from nothing.
- The head solved first halves the loss of a slowed encoder (at 500, 0.1: −0.061 to −0.038
  under the first pretraining, −0.015 to −0.001 under the second) and holds the seeds' spread
  there at 0.01 to 0.02 against 0.03 to 0.08 under a drawn head: the linear-probe half of the recipe
  does what Kumar et al. describe, the slowed fine-tuning half adds nothing that full
  fine-tuning does not.
- The check C0 repeats the reader's reading of 2026-10-13 within 0.02 on another GPU and under
  four concurrent processes (+0.012 and +0.028 at 200 against +0.032 and +0.033; −0.023 and
  −0.012 at 500 against −0.017 and −0.011).
- What follows, as declared: the encoder's rate is not the cause of the loss from 500 stays
  up, and the next step changes what the encoder learns or the corpora it learns from, not the
  recipe that adapts it.

**Limitations.**

- The stop judges after the first epoch, so the head solved before any step is not a state it
  can restore; a solved variant can end below the closed-form probe it started from.
- Two pretrainings of one recipe; the gap between them under a slowed encoder (up to 0.09)
  is itself read on two points.
- The variants' cells at 200 stays ran at lower contention than the settings' cells; the
  time of a cell, not its reading, depends on that.
