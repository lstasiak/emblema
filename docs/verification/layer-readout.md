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

## 2026-10-09 — M1 Pro, MPS: the probe at every layer, and two backbones never read

**Question.** The three declared above.

**Conditions.**

- Orders placed at `8e69f83d` and fulfilled one after another on this machine's accelerator (MPS,
  fp32), 2026-10-08 21:13 to 2026-10-09 11:49. The two last orders failed at once, because a
  draft of this section had left the tree uncommitted, and were fulfilled again at 12:06.
- 4.8M cells took about 28 s each on half of the tuning side and 25.3M cells about 74 s;
  1,420 cells in all.
- Pairs by `scripts/campaign_pairs_report.py` (10,000 resamples, 95 %); CSV under
  `data/report/l8/`.
- Gains are reductions of 1 − AUROC, positive favouring the candidate. Each cell gives the paired
  interval, then the mean over seeds ± its standard error.

**The replacement of the last block, at 50 stays on half of the tuning side.**

| Backbone | `layer=mean` | `layer=concat` |
|---|---|---|
| 10⁹ values, 4.8M | **+0.027** [+0.019; +0.035], +0.027 ± 0.009 | **+0.030** [+0.021; +0.039], +0.030 ± 0.009 |
| 10⁹ over four epochs, 4.8M | **+0.015** [+0.004; +0.026], +0.015 ± 0.007 | **+0.020** [+0.009; +0.030], +0.020 ± 0.007 |
| 10⁹ values, 25.3M | **+0.030** [+0.022; +0.038], +0.030 ± 0.010 | **+0.036** [+0.027; +0.044], +0.036 ± 0.011 |

Both readings meet the rule under all three backbones. The concatenation has the larger mean
gain (0.029 against 0.024), so **`layer=concat` replaces the last block** as the probe's reading.

**Every layer, at 50 stays**, against the last block:

| Layer | 10⁹, 4.8M | 10⁹ four epochs, 4.8M | 10⁹, 25.3M |
|---|---|---|---|
| 0 (embedding) | +0.025 [+0.005; +0.046] | −0.006 [−0.027; +0.016] | +0.018 [−0.000; +0.036] |
| 1 | +0.047 [+0.031; +0.062] | **+0.030** [+0.015; +0.045] | +0.051 [+0.037; +0.066] |
| 2 | **+0.049** [+0.033; +0.065] | +0.017 [+0.002; +0.031] | +0.042 [+0.030; +0.055] |
| 3 | +0.040 [+0.029; +0.052] | +0.011 [−0.003; +0.024] | **+0.053** [+0.040; +0.065] |
| 4 | +0.017 [+0.009; +0.025] | +0.007 [−0.005; +0.020] | +0.042 [+0.031; +0.052] |
| 5 | +0.005 [−0.001; +0.011] | +0.008 [−0.002; +0.019] | +0.022 [+0.016; +0.028] |
| 6 | | | +0.016 [+0.010; +0.021] |
| 7 | | | +0.004 [+0.003; +0.006] |

The best layer of each backbone is in bold.

**The width control, at 50 stays.**

| | 10⁹, 4.8M | 10⁹ four epochs, 4.8M | 10⁹, 25.3M |
|---|---|---|---|
| untrained encoder: `concat` less its last block | −0.002 ± 0.006 | (the same encoder) | +0.001 ± 0.003 |
| `concat` gain less the control's `concat` gain | +0.032 ± 0.014 | +0.022 ± 0.010 | +0.034 ± 0.010 |
| last block less the untrained encoder | −0.021 [−0.040; −0.002] | +0.008 [−0.015; +0.031] | −0.011 [−0.031; +0.009] |
| `concat` less the untrained encoder's `concat` | +0.011 [−0.011; +0.033], +0.011 ± 0.025 | +0.029 [+0.008; +0.050], +0.029 ± 0.020 | +0.023 [+0.001; +0.046], +0.023 ± 0.026 |
| best layer less the untrained encoder (chosen after the fact) | +0.028 ± 0.019 | +0.038 ± 0.018 | +0.041 ± 0.018 |

**20 and 200 stays**, `concat` against the last block (descriptive):

| Backbone | 20 stays | 200 stays |
|---|---|---|
| 10⁹, 4.8M | +0.019 [+0.013; +0.025] | +0.032 [+0.021; +0.043] |
| 10⁹ four epochs, 4.8M | +0.024 [+0.015; +0.033] | +0.014 [+0.002; +0.027] |
| 10⁹, 25.3M | +0.019 [+0.013; +0.026] | +0.011 [+0.004; +0.018] |

Every interval lies above zero, but at 20 stays no mean over seeds passes twice its standard
error (+0.019 ± 0.013, +0.024 ± 0.012, +0.019 ± 0.010); at 200 stays all three do. The mean of
the blocks gains less at every budget.

**The pretext rule's third condition.** On the half at 50 stays, the closed-form probe under the
mixture's pretraining seed 2 less seed 1 is −0.003 [−0.017; +0.011], −0.003 ± 0.012. The rule's
gap on the half is therefore 0.003. At 20 and 200 stays: −0.007 and +0.009. The trained probe
differs more: +0.012, +0.012 and **+0.029** [+0.008; +0.050] ± 0.010 at 20, 50 and 200 stays,
the last passing both conditions between two pretrainings of one recipe.

**The two backbones never read, on the validation side.** Each is compared with the same probe
under the mixture of four at seed 1.

| Probe | Stays | Stays four times as often | The stays alone, 77k |
|---|---|---|---|
| closed-form | 20 / 50 / 200 | −0.011 / −0.009 / −0.020 | −0.008 / **−0.031** / −0.009 |
| trained | 20 / 50 / 200 | −0.006 / −0.008 / −0.005 | −0.011 / +0.023 / **+0.057** |
| closed-form, against the network from nothing | 50 | −0.030 [−0.045; −0.015] | −0.052 [−0.068; −0.037] |

Bold marks a difference whose mean over seeds passes twice its standard error.

**FD001 at 50 windows** (descriptive), RMSE in cycles, lower is better: the layer campaigns on
the tuning half (40 engines), part (b) on the 21 validation engines.

| Reading | 10⁹, 4.8M | 10⁹ four epochs, 4.8M | 10⁹, 25.3M |
|---|---|---|---|
| untrained encoder, last block | 19.1 | 19.1 | 18.9 |
| last block | 23.5 | 23.7 | 24.0 |
| layer 0 (embedding) | 22.2 | 22.1 | **22.1** |
| best block | **19.9** (block 1) | **19.3** (block 1) | 22.8 (blocks 2 and 3) |
| `concat` | 22.7 | 21.8 | 23.7 |
| untrained encoder, `concat` | 19.1 | 19.1 | 18.7 |

On the validation engines, against the mixture of four at seed 1 (25.5 RMSE):

- the probe under the stays four times as often reads 23.6;
- the probe under the small backbone reads 18.5, level with its own untrained encoder (18.3;
  −0.2 [−0.7; +0.3]);
- the network from nothing reads 16.6.

**Conclusions.**

1. **The concatenation of the blocks replaces the last block** by the registered rule. It gains
   0.020 to 0.036 at 50 stays under every backbone. At 200 stays the gain passes both conditions
   too; at 20 stays its intervals lie above zero but the seeds do not settle it.
2. **The task lies early in the encoder.** Under every backbone the best single layer is block 1
   to 3. It gains 0.030 to 0.053 over the last block, more than the concatenation does, though
   the best of six or eight layers is chosen after the fact and so lies somewhat high. Above it
   the blocks read the task broadly less well, down to the last. This is consistent with a
   pretext that spends the top of the network on reconstruction.
3. **The gain is not the width's.** The untrained encoder gains nothing from reading every
   block (−0.002 and +0.001); the pretrained encoders gain 0.022 to 0.034 more from it than
   their controls do. What it recovers is mostly what the last block lost: read at its last
   block, each backbone lies from 0.021 below to 0.008 above the untrained encoder; read by the
   concatenation, 0.011 to 0.029 above it, with no mean over seeds passing twice its standard
   error. That pretraining helps this probe at 50 stays is still not shown by the rule.
4. **Longer training lifts the top blocks, so its gain depends on the reading.** Four epochs
   against two (paired, same half and seeds, descriptive): +0.028 [+0.008; +0.049] ± 0.014 at
   the last block, +0.018 [+0.000; +0.036] ± 0.017 under the concatenation. Blocks 4, 5 and the
   last read 0.02 to 0.03 better after the longer run, blocks 2 and 3 no better. The larger
   shape against 4.8M stays unsettled either way (+0.007 at the last block, +0.013 under the
   concatenation).
5. **On FD001 the same shape, weaker, and no layer helps.** Block 1 of the 4.8M backbones reads
   the remaining life within 0.2 to 0.8 RMSE of the untrained encoder, where the last block lies
   4.4 to 4.6 above it. The concatenation recovers less than half of that. In 25.3M no layer
   comes within 3 RMSE of the untrained encoder. No layer of any backbone beats it.

**Predictions.** All six held: `concat` gained 0.020 to 0.036 (1); the best 4.8M layers are 2
and 1 (2); on FD001 the embedding lies closer to the untrained encoder than the last block (3);
the controls gained nothing (4); the seeds' gap is −0.003 (5); the stays four times as often lie
0.009 below the mixture, and below it at every budget by less than the rule detects, while the
small backbone's closed-form probe lies below it (6).

**What follows.** The stage's later readings take `frozen_ridge@layer=concat` as the closed-form
probe, by a row in `docs/preregistration.md`; points already read are not read again. A single
early block reads better still, but the rule did not offer it, and choosing it per backbone
would be a selection of its own.

**Limitations.**

- One pretraining per backbone.
- The tuning half holds the fifth on which the recipe and the pretext variants were read.
- The trained probe was not read at other layers. Under the small backbone it gains at 200
  stays where the closed-form probe loses at 50, so the two heads read that backbone
  differently. That is not explored here.
- At 50 labels the concatenation hands the closed-form probe 1,536 or 4,096 columns, its
  penalty chosen by leave-one-out; the control reads the same width.

## 2026-10-09 — declared before the run: the pretraining seeds' difference under the blocks side by side

**Question.** The blocks side by side are now the closed-form probe's reading. How far apart does
that probe read the mixture of four's two pretraining seeds, on half of the tuning side and on the
validation side? Every rule that compares backbones by this probe takes that difference as a
threshold, so it is read before any of them.

**Design.** Only probes, on this machine's accelerator, ordered from a commit that holds this
section. Under `backbone-mixed4-m` at pretraining seeds 1 and 2, the closed-form probe over the
blocks side by side (`frozen_ridge@layer=concat`) and the same reading of the encoder at its
initialisation (`untrained_ridge@layer=concat`), at 20, 50 and 200 stays, seeds 1 to 10.

| Side | Campaign file | Cells |
|---|---|---|
| half of the tuning side (seed 101, `one_in = 2`) | `pretext-concat-half-physionet2012.toml` | 2 × 60 |
| validation | `yardstick-concat-physionet2012.toml` | 2 × 60 |

About 240 cells; from L-8's costs, about two hours.

**Pairs.** `scripts/campaign_pairs_report.py`, 10,000 resamples, 95 %, with the mean over seeds
and its standard error: seed 2 against seed 1 on each side at each budget; on each side, the
blocks side by side against the last block under seed 1 (the half's campaigns of 2026-10-08 and
the yardstick's campaigns score the same stays and draws).

**Reading, declared beforehand.**

- *The pretext rule's third condition* becomes the absolute difference between the two seeds by
  this probe at 50 stays on the half.
- *The difference between the pretraining seeds* that the slope and the length of training name
  becomes the absolute difference between the two seeds by this probe at 50 stays on the
  validation side (`docs/preregistration.md`, "The scale of pretraining").
- Everything else is descriptive.

**Predictions.**

1. On the half at 50 stays, seed 2 less seed 1 lies between −0.02 and +0.02. Basis: by the last
   block, −0.003 [−0.017; +0.011] (2026-10-09 above); a reading of other blocks may move it, so
   the range is rounded outwards.
2. On the validation side at 50 stays, it lies between −0.03 and +0.04. Basis: by the last block,
   +0.004 ± 0.016.
3. *Judgement.* Under seed 1 the blocks side by side lie above the last block at 50 stays on both
   sides. Under the three backbones of about 10⁹ values they gained 0.020 to 0.036 on the half.

**Limitations.**

- Two pretrainings give one difference. It is a point with a standard error of about 0.012 to
  0.016, used as a threshold because the rules are registered that way.

## 2026-10-09 — M1 Pro, MPS: the pretraining seeds' difference under the blocks side by side

**Question.** The one declared above.

**Conditions.**

- Orders placed at `36b83bf9` and fulfilled one after another on this machine's accelerator (MPS,
  fp32), 2026-10-09 18:27 to 21:23. Cells took about 28 s on the half and 53 s on the validation
  side; 240 cells in all.
- Pairs by `scripts/campaign_pairs_report.py` (10,000 resamples, 95 %); CSV under
  `data/report/l9/`. Gains as above: reductions of 1 − AUROC, the paired interval, then the mean
  over seeds ± its standard error.

**Seed 2 less seed 1, by the blocks side by side.**

| Side | 20 stays | 50 stays | 200 stays |
|---|---|---|---|
| half of the tuning side | −0.013 [−0.024; −0.003], −0.013 ± 0.011 | −0.005 [−0.016; +0.006], −0.005 ± 0.008 | +0.018 [+0.004; +0.033], +0.018 ± 0.008 |
| validation | −0.018 [−0.027; −0.010], −0.018 ± 0.008 | +0.003 [−0.003; +0.010], +0.003 ± 0.008 | +0.015 [+0.005; +0.026], +0.015 ± 0.010 |

The pretext rule's third condition is therefore **0.005**. The difference between the pretraining
seeds that the slope and the length of training take, read by this probe, is **0.003**.

**The blocks side by side against the last block, under seed 1.**

| Side | 20 stays | 50 stays | 200 stays |
|---|---|---|---|
| half of the tuning side | +0.011 [+0.005; +0.018] | +0.011 [+0.006; +0.017], +0.011 ± 0.006 | +0.007 [−0.002; +0.016] |
| validation | +0.019 [+0.014; +0.023] | +0.016 [+0.012; +0.020], +0.016 ± 0.003 | +0.008 [+0.003; +0.014] |

The untrained encoder's blocks side by side against its last block, on the validation side: −0.003,
−0.001 and −0.009 at 20, 50 and 200 stays.

**Against the untrained encoder's blocks side by side** (paired after the run, not declared;
descriptive):

| Side | Seed | 20 stays | 50 stays | 200 stays |
|---|---|---|---|---|
| half | 1 | +0.031 ± 0.025 | +0.020 [+0.003; +0.038], ± 0.017 | +0.019 ± 0.013 |
| half | 2 | +0.018 ± 0.021 | +0.015 [+0.001; +0.030], ± 0.018 | **+0.038** ± 0.013 |
| validation | 1 | **+0.032** ± 0.013 | **+0.026** [+0.015; +0.036], ± 0.012 | +0.030 ± 0.020 |
| validation | 2 | +0.013 ± 0.016 | **+0.029** [+0.021; +0.037], ± 0.008 | **+0.045** ± 0.012 |

Bold marks an interval above zero with a mean over seeds above twice its standard error.

**Conclusions.**

1. **The thresholds are small: 0.005 on the half, 0.003 on the validation side.** Both lie inside
   the predicted ranges.
2. **At 50 stays the two pretrainings agree more closely than at the other budgets.** At 20 and
   200 stays they differ by 0.013 to 0.018, with the sign turning between them, and in two of
   those four cells the difference passes both conditions. Net of the draws of labels and stays,
   the six differences put the pretraining's own spread of a difference between two backbones at
   about 0.007 to 0.010 (one standard deviation, from one pair of pretrainings). A threshold read
   at 50 stays alone is one draw of that spread, and here a small one.
3. **The blocks side by side read the mixture of four better than its last block**, by 0.011 on
   the half and 0.016 on the validation side at 50 stays: less than under the backbones of about
   10⁹ values (0.020 to 0.036), and less at 200 stays.
4. **Read this way, the mixture of four beats its untrained encoder at 50 stays on the validation
   side under both pretraining seeds** (+0.026 and +0.029, each passing both conditions and the
   seeds' difference). It was paired after the run and changes no verdict: points already read
   are not read again. On the half the same comparison does not pass the seed condition.
5. **Against the network from nothing** (yardstick cells on the same stays and draws, paired after
   the run): the blocks side by side lie −0.006 [−0.021; +0.009] and −0.003 [−0.015; +0.010]
   under the two seeds at 50 stays, where the last block lies −0.022 [−0.036; −0.007]; at 20
   stays they lie −0.040 and −0.058 below it.

**Predictions.** All three held: −0.005 on the half (1), +0.003 on the validation side (2), and
the blocks side by side above the last block on both sides (3).

**Limitations.**

- Two pretrainings give one difference per side and budget; the threshold at 50 stays is a
  point with a standard error of 0.008.
- The comparisons with the untrained encoder and the network from nothing were chosen after the
  run; they are leads for a declared reading, not results.

## 2026-10-10 — M1 Pro, MPS: the pretraining's spread from four seeds

**Question.** The error between backbones (`docs/preregistration.md`) needs s, the spread a
difference between two backbones takes from their pretraining alone, read from the mixture of
four's pretrainings at seeds 1 to 4 before the first reading under it.

**Conditions.** `backbone-mixed4-m-seed3` and `-seed4` pretrained on Kaggle (two T4s, one each,
ordered from `cbd986c8`) and accepted here; their campaigns under the same files as seeds 1 and 2
(`pretext-concat-half-physionet2012.toml`, `yardstick-concat-physionet2012.toml`), ordered from
`6c98ebb7` and fulfilled on this machine's accelerator (MPS, fp32), 2026-10-10 10:23 to 14:30.
Every pair of seeds at 20, 50 and 200 stays by `scripts/campaign_pairs_report.py`; s by the
registered formula (`data/report/l9/s.py`).

**The later seed less the earlier, by the blocks side by side** (mean over the draws of labels).

| Seeds | Half, 20 / 50 / 200 stays | Validation, 20 / 50 / 200 stays |
|---|---|---|
| 2 − 1 | −0.013 / −0.005 / +0.018 | −0.018 / +0.003 / +0.015 |
| 3 − 1 | −0.031 / −0.015 / +0.010 | −0.004 / −0.016 / −0.019 |
| 4 − 1 | −0.006 / −0.001 / +0.002 | −0.008 / +0.003 / +0.010 |
| 3 − 2 | −0.018 / −0.010 / −0.009 | +0.014 / −0.020 / −0.035 |
| 4 − 2 | +0.007 / +0.004 / −0.017 | +0.011 / −0.001 / −0.006 |
| 4 − 3 | +0.024 / +0.014 / −0.008 | −0.003 / +0.019 / +0.029 |

**s = 0.0060 on half the tuning side and 0.0104 on the validation side** (18 pairs and budgets
each). From seeds 1 and 2 alone the same formula gives 0.0076 and 0.0099; the values first
registered, 0.0072 and 0.0095, were computed from the rounded table and are corrected here.

**Conclusions.**

1. On the validation side a comparison of two backbones at 50 stays now needs a mean of about
   0.026 to 0.029 at the standard errors seen there (0.008 to 0.010); on the half, about 0.020.
2. Seed 3 reads the stays worse than the others at 50 stays on both sides (0.010 to 0.020), and
   seed 4 lies with seeds 1 and 2: one pretraining in four lands visibly apart, which is what the
   error between backbones is there to carry.
3. At 50 stays half the pairs differ by 0.005 or less and the rest by 0.010 to 0.020: a threshold
   read from one pair, as first registered, lands on either side by chance.

**Limitations.**

- Four pretrainings estimate s with a few degrees of freedom; the budgets are pooled as though the
  pretraining's effect at each were a fresh draw.

## 2026-10-13 — declared before the run: the stage's backbones by the reader at 500 stays, and the point of about 10⁹ values block by block

**Question.** Read at 500 stays and above, pretraining's share of the probe turns negative, and
the point of about 10⁹ values reads the stays worse than the mixture of four
([`verdict-statistics.md`](verdict-statistics.md), 2026-10-13). Where in the stack is that loss:
in every block, or in the blocks the pretext trains last? And what do the backbones read before
only at 20, 50 and 200 stays, or never, show under the reader of the stage? The reading is
descriptive: it changes no rule and no verdict.

**Design.** Only probes, on this machine's accelerator, ordered from a commit that holds this
section; half of the tuning side (seed 101, `one_in = 2`), 500 stays, seeds 1 to 10.

| Campaign file | Candidates | Under | Cells |
|---|---|---|---|
| `review-layers-500-physionet2012.toml` | `frozen_ridge` and `untrained_ridge` at layers 0 to 5 and the last | `backbone-scale-1e9-m`; `backbone-mixed4-m` at pretraining seeds 1 and 2 | 3 × 140 |
| `review-concat-500-physionet2012.toml` | `frozen_ridge@layer=concat`, `untrained_ridge@layer=concat` | the eighteen backbones below | 18 × 20 |

| Group | Backbones | Read before |
|---|---|---|
| the curve's points, 4.8M | `backbone-scale-1e8new-m`, `backbone-scale-3e8-m` | last block, 20 / 50 / 200 |
| length of training | `backbone-scale-1e9-4ep-m`; `backbone-scale-1e9-512x8-4ep-m` | last block and `concat`, 50 |
| the larger shape | `backbone-mixed4-512x8-m-5e-4`; `backbone-scale-1e9-512x8-m` at seeds 1 and 2 | last block, 20 / 50 / 200; at 10⁹ `concat`, 50 |
| corpora left out | `backbone-scale-1e9-without-{physionet2012,cmapss,physionet2019}-m`, `backbone-physionet2019-m`, `backbone-mixed4-without-physionet2012-m` | never by a probe |
| the stays' weight | `backbone-mixed4-stays-x4-m`, `backbone-stays-small-m` (81k) | validation side, 20 / 50 / 200 |
| pretext and objective | `backbone-mixed4-m-forecast`, `backbone-mixed4-m-harder`; `backbone-mixed4-latent-m` at seeds 1 and 2 | last block or `concat`, 50 |

780 cells. At 28 s a cell at 500 stays in 4.8M (2026-10-13), and about three times that in 25.3M, about
seven hours. The reader's campaigns under `backbone-mixed4-m` at seeds 1 to 4 and
`backbone-scale-1e9-m` (`reader-probes-physionet2012.toml`) score the same stays and draws, and
stand in for the mixture and the point of about 10⁹ values here.

The encoder at its initialisation depends on the backbone's shape, its vocabulary included:
under the mixture's four pretrainings its cells are identical, under `backbone-scale-1e9-m` they
are not (0.693 against 0.681 at 500 stays, 2026-10-13). So each backbone is read against the
initialisation of its own shape, and the backbones fall into seven shapes.

**Pairs.** `scripts/campaign_pairs_report.py`, 10,000 resamples, 95 %, with the mean over seeds
and its standard error.

| Kind | Pairs | Stands in for |
|---|---|---|
| L | each layer under a backbone less the same layer at its initialisation | where pretraining adds or takes away |
| L′ | each layer less the last, under each backbone and at initialisation | where the task sits |
| D | the point of about 10⁹ values less the mixture, layer by layer, as a difference of means over the mixture's two pretrainings | where more pretraining takes away |
| A | each backbone of the second campaign by `concat` less its own initialisation | its share of pretraining |
| B | each backbone by `concat` less the mixture's four pretrainings, as a difference of means | the stage's comparison |
| B″ | the named pairs within a group: four epochs less two; 25.3M less 4.8M at 10⁸ and at 10⁹; each corpus left out less `backbone-scale-1e9-m`; the stays alone less `backbone-scale-1e9-m` | comparisons read earlier at 50 stays and left unsettled |
| I | the initialisation's `concat` across the seven shapes | the spread of a draw of initial weights |

**Reading, declared beforehand.** Every pair is descriptive. Kinds B and B″ are set against the
stage's threshold, twice √(SE² + (s²/2)(1/n_c + 1/n_r)), s = 0.0057, n the pretrainings on each
side, and reported as *over* or *within* it; none is a verdict, and none enters
`docs/preregistration.md`. Kind A is reported beside B wherever the two sides differ in shape,
since there a difference in the initialisation enters B and not A. Identical shapes give
identical cells at initialisation; any difference there is a fault in the run, read before
anything else.

**Predictions.** Each from an interval read before, or marked a judgement.

1. *Kind L under the point of about 10⁹ values*: blocks 1 to 3 within ±0.02 of their
   initialisation, blocks 4, 5 and the last below it by 0.01 to 0.05. Basis: at 50 stays blocks 1
   to 3 lay 0.040 to 0.049 above the last block, which lay 0.021 [−0.040; −0.002] below its
   initialisation (2026-10-09); `concat` lost 0.007 to its initialisation at 500 stays
   (2026-10-13). A judgement for the split between blocks.
2. *Kind L under the mixture*: blocks 1 to 3 at or above their initialisation (0 to +0.03), the
   last block within −0.03 to +0.01. A judgement from `concat` at 500 stays, −0.000 and +0.017
   under seeds 1 and 2.
3. *Kind D*: the point of about 10⁹ values lies below the mixture at every layer, by most in the
   top blocks. Basis: −0.019 to −0.036 by `concat` (2026-10-13); the top blocks a judgement.
4. *The curve's points*, kind B: `backbone-scale-1e8new-m` and `backbone-scale-3e8-m` between
   the point of about 10⁹ values and the mixture, −0.04 to 0.00. A judgement: at 50 stays the
   points were not monotone (−0.021, +0.015, −0.021).
5. *Length of training*, kind B″: four epochs less two in 4.8M, 0.00 to +0.03 (by `concat` at 50
   stays +0.018 [+0.000; +0.036], 2026-10-09, from its lower edge); in 25.3M −0.03 to +0.01
   (−0.027 [−0.038; −0.016] against the first pretraining at 50 stays,
   [`intensive-care-curve.md`](intensive-care-curve.md), 2026-10-10; the second pretraining did
   not replicate the first's lead, 2026-10-11).
6. *The larger shape*, kind B″: 25.3M less 4.8M at 10⁹ within ±0.02, within the threshold; at 10⁸
   within ±0.02. Basis: +0.007 at the last block and +0.013 by `concat` at 50 stays (2026-10-09),
   and the 0.020 between the larger shape's two pretrainings (2026-10-11).
7. *The pretext and the objective*, kind B: each of the forecast tail, the harder masks and the
   objective of states below the mixture, by 0 to 0.04, none over the threshold for a gain. Basis:
   −0.017 [−0.038; +0.005] and −0.015 [−0.030; +0.000] at 50 stays on a fifth by the last block,
   and the objective of states −0.034 ± 0.014 by `concat`
   ([`pretext-variants.md`](pretext-variants.md), 2026-10-06 and 2026-10-11).
8. *Corpora left out* and *the stays' weight*: a judgement only, none read before at any budget
   by `concat` or none above 200 stays. The stays four times as often within ±0.02 of the mixture
   (−0.009 and −0.020 at 50 and 200 stays on the validation side, 2026-10-09).
9. *Kind I*: the seven initialisations spread by 0.005 to 0.015 in mean area. A judgement from
   one pair, 0.012.

**Limitations.**

- Block by block, two pretrainings of the mixture against one of the point of about 10⁹ values;
  the spread of pretraining at each layer is not measured, only by `concat`.
- Most groups hold one pretraining per backbone; kind B carries s from the mixture, which a
  larger shape exceeds (0.020 between its two pretrainings at 50 stays).
- 500 stays resolve 0.02 to 0.03 between backbones four times in five; a difference of 0.01
  stays unseen.
- The reading follows the stage's reader and nothing else; FD001 is not run.
