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
