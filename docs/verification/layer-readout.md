# Reading the encoder at another layer

Whether what the pretext teaches lies in the encoder's states but below the last block, where no
head has read it, and how the backbones never read by the yardstick compare
([ADR-0053](../adr/0053-reading-the-encoders-states-at-another-layer.md)).

## 2026-10-08 — declared before the run: the probe at every layer, and two backbones never read

**Question.**

1. Does the probe solved in closed form read the intensive-care task better from a layer below
   the last block, from the mean of the blocks, or from the blocks side by side?
2. If it does, is that because of what pretraining put there, or because of the width a
   reading of every block gives?
3. How does the yardstick's probe read the two backbones never read by it: the mixture with the
   stays four times as often, and the small backbone over the stays alone?

**Design.** Every campaign runs only probes, on this machine's accelerator, ordered from the
commit that adds this section. Every campaign on the tuning side is scored on half of the tuning
stays held out by seed 101 (`one_in = 2`, about 2,000 stays, holding the fifth the recipe was
chosen on), over seeds 1 to 10. That is the design the replacement rule reads since
`verdict-statistics.md`, 2026-10-08. The validation side is read only by part (b).

| Part | Campaign file | Backbones | Budgets | Cells |
|---|---|---|---|---|
| (c) layers | `layers-50-physionet2012.toml` | `backbone-scale-1e9-m`, `backbone-scale-1e9-4ep-m` (4.8M) | 50 stays | 2 × 120 |
| (c) layers | `layers-512x8-50-physionet2012.toml` | `backbone-scale-1e9-512x8-m` (25.3M) | 50 stays | 140 |
| (c) layers | `layers-20-200-physionet2012.toml` | the three above | 20, 200 stays | 3 × 80 |
| (c) layers | `layers-50-fd001.toml`, `layers-512x8-50-fd001.toml` | the same, by shape | 50 windows | 2 × 120 + 140 |
| (b) yardstick | `yardstick-probes-physionet2012.toml` | `backbone-mixed4-stays-x4-m`, `backbone-stays-small-m` | 20, 50, 200 stays | 2 × 90 |
| (b) yardstick | `yardstick-probes-50-fd001.toml` | the same | 50 windows | 2 × 30 |
| the pretext rule's gap | `pretext-half-physionet2012.toml` | `backbone-mixed4-m` at seeds 1 and 2 | 20, 50, 200 stays | 2 × 90 |

The layer campaigns read, under each backbone:

- the closed-form probe at the last block (`frozen_ridge`);
- the same probe at every layer below it, from the tokens' embedding (`layer=0`) up;
- at the mean of the blocks (`layer=mean`) and at the blocks side by side (`layer=concat`);
- the same probe over the encoder at its initialisation, at its last block, its mean and its
  concatenation (`untrained_ridge`). These three are the width control.

Part (b) reads the yardstick's three probes on the validation side, on the same stays and draws
as the curve. The network from nothing is not run again: its cells under the mixture of four pair
with these.

About 1,420 cells. The cost is not measured at this size: from 10 s a cell at 800 stays in 4.8M,
about seven hours.

**Pairs.** `scripts/campaign_pairs_report.py`, 10,000 resamples, 95 %, with the mean over seeds
and its standard error. Within each layer campaign and backbone, every reading is paired against
the last block, and the controls against `untrained_ridge`. Each backbone's concatenation gain is
paired against its control's concatenation gain. In part (b), the probes under each backbone are
paired against the same probes under `backbone-mixed4-m` at seed 1 and against the network from
nothing. Under the mixture's two seeds, the probes are paired against each other.

**Reading, declared beforehand.**

- *Replacement of the last block.* `layer=mean` or `layer=concat` replaces the last block as the
  probe's reading when, at 50 stays, it meets the replacement rule against the last block under
  at least two of the three backbones. The rule requires a paired 95 % interval above zero and a
  mean over seeds above twice its standard error. The pretraining-seed condition does not apply:
  both sides read one set of weights. Where both readings replace, the larger mean gain over the
  three backbones is chosen. Per backbone, the rule replaces a reading that gains nothing in 0.02
  to 0.05 of readings, and replaces a true gain of 0.02 in about half and one of 0.03 in about 0.9
  (`verdict-statistics.md`, 2026-10-08). If the three backbones were independent, two of three
  would give under 0.01, about 0.5 and about 0.98. They share stays and draws, so the false rate
  lies somewhat above that.
- *What a replacement does.* It is registered as the probe's reading for the stage's later
  readings (the longer run of the larger shape, the mixture weights, the objective) by one row in
  `docs/preregistration.md` before any of them is read. Points already read are not read again.
- *Read descriptively, choosing nothing*: single layers, the controls, 20 and 200 stays, FD001,
  and part (b).
- *The pretext rule's third condition* is the absolute difference between the closed-form probe
  under the mixture's two pretraining seeds at 50 stays on the half, as measured here.

**Predictions.** Where no earlier interval bears on a prediction, it is marked as a judgement.

1. *Judgement.* Under at least two backbones, `layer=concat` lies above the last block at
   50 stays by 0.01 to 0.04. The rule replaces with `concat`.
2. *Judgement.* Under each 4.8M backbone at 50 stays, the best single layer is a block below the
   last (layers 1 to 5), not the embedding.
3. On FD001, the embedding (`layer=0`) reads the remaining life closer to the untrained encoder
   than the last block does. Under every backbone, the last block reads 3.9 to 5.9 RMSE worse
   than the untrained encoder (`docs/findings.md`), and layer zero holds no block trained on the
   pretext.
4. The width alone does not explain a gain. Under the untrained encoder, `concat` gains over its
   last block at 50 stays by less than the pretrained encoders' `concat` gains over theirs.
   Basis: on the fifth, pretraining added +0.032 ± 0.012 at 50 stays over the untrained probe
   (`pretext-variants.md`, 2026-10-08).
5. The mixture's two pretraining seeds differ at 50 stays on the half by between −0.03 and
   +0.02 for the closed-form probe. Basis: on the validation side, seed 2 less seed 1 is
   +0.004 ± 0.016; on the fifth, −0.008 ± 0.007.
6. *Judgement.* Under the mixture with the stays four times as often, the closed-form probe at
   50 stays on the validation side lies within ±0.03 of the probe under the mixture at seed 1.
   The small backbone's probe lies below it.

**Limitations.**

- One pretraining per backbone, so a gain of a reading is a gain under these weights.
- The concatenation hands the closed-form probe up to eight times more columns from 50 labels.
  The penalty is chosen by leave-one-out among the registered ones, and the control reads the
  same width.
- Half of the tuning side holds the fifth that the recipe and the pretext variants were read on.
