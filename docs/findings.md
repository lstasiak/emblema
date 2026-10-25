# Findings

What the project has measured, on one page. Every number is on the **validation side** and
preliminary: each task's frozen test side is opened once, at the end. Each comparison was judged
by a rule registered before its run ([preregistration](preregistration.md)); each row links the
note that holds its evidence ([`verification/`](verification/README.md)).

Last updated 2026-10-25.

## Status

- **The claim is not supported so far.** No way of using the pretrained encoder beats a
  well-tuned network trained from scratch beyond the instrument's resolution, on either task.
  On the small encoder over cells, fine-tuning gains +0.021 [+0.010; +0.031] at 200
  intensive-care stays over the same network at the rate in force, at the registered least
  gain; with each arm at the rate it keeps, the network from scratch gains +0.024 itself and the
  gain at 200 stays is −0.004 [−0.015; +0.007]; at 50 stays +0.014 [+0.006; +0.022] remains,
  below the least gain; at 20 nothing ([note](verification/cell-pretraining.md)).
- **Pretraining over cells leaves a representation above its initialisation, a first on the
  intensive-care stays.** A probe on the frozen encoder gains 0.060 to 0.091 over the same encoder untrained at
  500 stays under every recipe (0.757 against 0.677; the trees read 0.787), and its pretext
  reads unread stays above the best linear map. A forecast tail, the count term and added grids
  move nothing ([note](verification/cell-pretraining.md)).
- **Classical models lead at few labels.** Trees over per-channel summaries match or beat every
  network from 200 labels up on both tasks; on the intensive-care stays they lie 0.07 to 0.09 in
  area above the probe, and a logistic head over the same summaries about 0.07 above it, within
  0.006 of the trees at 200 stays (re-read on 2026-10-23 with its preparation corrected). The
  network trained from scratch on raw tokens trails the trees by 0.077 at 200 stays and, against
  set B, by 0.05 at every stay: more than any gain pretraining has shown. A small encoder from
  scratch over cells of channel and time bin closes most of that (below).
- **The summaries are in the token states; the probe's average loses them.** A linear map
  recovers a median 86 % of a channel's level from the mean of its own tokens, in every block and
  as before training, but 15 % from the mean over the stay (31 % before training). Pooling per
  channel lifts the probe by 0.04, still 0.03 below a head over the summaries, and
  pretraining has no share in that lift; no average over a stay's channels keeps it.
- **Below 500 labels the network's gap is representation; above it, the trees' table.** A
  network handed the trees' per-channel summaries reads above the trees at 20 and 50 stays
  (+0.023, +0.046; carried at 50) and leads the network on its tokens by 0.045 to 0.059 up to 200 stays. From 500 stays
  the network on its tokens matches it, and both stop 0.03 below the trees, as a linear head over
  the summaries does: the rest is what trees make of that table, not the network's training
  (a penalty and every drawn stay change nothing) ([note](verification/gap-to-the-trees.md)).
- **What transfers is the encoder's machinery, not its features for the task.** Fitting only the
  channel rows of a corpus the backbone never read, on its unlabelled windows, brings the pretext
  to within 0.006 of a backbone that read it, and carries part of a downstream task (+0.031 on
  sepsis at 4,000 patients).
- **Ruled out as levers:** more pretraining data, a five times larger model, continuing on the
  task's own domain, the adaptation recipe, and six variants of the pretext.
- **Reading the window as cells does not close the gap at few labels.** An encoder trained from
  scratch over cells of channel and time bin, carrying each bin's count, level, extremes and
  ends, reads within 0.01 of the same encoder on raw readings rerun beside it at 50 and 200
  stays (0.67 at 200 in that rerun, 0.65 in the earlier campaign), and 0.03 to 0.05 below a
  network handed the trees' table; the best of thirteen variants, six bins, gains 0.03 at 200
  stays and still trails that network; a learnt time encoding, several pooling
  queries and rows for the channels a stay never measured move nothing beyond the reader's
  resolution ([note](verification/encoder-candidates.md)).
- **A small encoder from scratch reaches the trees at 200 stays and leads them at 50.** Two
  blocks 64 wide over cells of six bins read 0.73 at 200 stays, level with the trees (+0.003
  [−0.011; +0.017]) and 0.027 above the network handed their table, and 0.66 at 50, 0.028 above
  the trees; the 4.8M encoder on the same cells reads 0.68 and 0.64. The gap at few labels was
  the encoder's size, not what it read; dropout, weight decay, a clip and a class weight add
  nothing ([note](verification/encoder-candidates.md)).
- **From 500 stays up the cells carry the gain and the trees keep 0.008.** On six-bin cells the
  4.8M encoder and the small one read alike (0.83 at every stay of the half), 0.022 above the
  network handed the trees' table and 0.008 below the trees, not distinguishable. The mean lies
  within the 0.015 the register asks of the half; the lower edge of that pair (−0.023) does not,
  so the set B rule would pass about one time in four at that standing
  ([note](verification/encoder-candidates.md)).
- **No input moves the small encoder at every stay.** Twelve and twenty-four bins, a learnt
  time encoding and four pooling queries read within 0.004 of six bins at every stay of the
  half, none carried; rows for the channels a stay never measured cost 0.009 [0.001; 0.018].
  Six bins under the registered recipe go to set B against the trees
  ([note](verification/encoder-candidates.md)).
- **On set B the small encoder from scratch reads 0.008 below the tuned trees and is not shown
  no worse by 0.015.** Learnt from every stay of set A, it reads 0.850 against the trees'
  0.858 at `learning_rate=0.1,rounds=300` (−0.008 [−0.017; +0.001], the lower edge 0.002 past
  the margin) and level with the trees at their default setting (+0.001 [−0.010; +0.011]); the
  same 0.008 as on the half. The registered rule closes this direction
  ([note](verification/encoder-candidates.md)).
- **Next:** data in the tasks' regime for both legs (raw asynchronous telemetry, open
  intensive-care sources), then anomaly tasks on telemetry without labels, read on the
  representation pretraining now gives. The clinical confirmation is declared as a curve
  against the network from scratch at its chosen rate.

## Setup

| | |
| --- | --- |
| Tasks | Remaining useful life on C-MAPSS FD001 (RMSE in cycles; 79 tuning and 21 validation engines); death in hospital on PhysioNet 2012 (AUROC over the first 48 h of a stay); sepsis on PhysioNet 2019 as a reader within one backbone |
| Pretraining corpora | Mixture of four: C-MAPSS, SKAB, ESA satellite telemetry, PhysioNet 2012 ([ADR-0029](adr/0029-pretraining-over-a-mixture-of-corpora.md)); up to about 10⁹ values over 26 publications for the scale curve; sampled like the intensive-care tasks: 4.5 % of the mixture's windows, 1 to 3 % of the scale curve's values ([note](verification/pretraining-corpus-sampling.md)) |
| Backbone | Set encoder over (channel, time, value) tokens, 4.8M parameters (25.3M for scale), pretrained by masked reconstruction ([ADR-0017](adr/0017-encoder-architecture.md)) |
| Arms | From scratch (control), frozen probe, probe solved in closed form, LoRA, full fine-tuning; trees, MiniRocket and a patch model as classical baselines |
| Statistics | Paired bootstrap over engines or stays, seeds paired; a backbone comparison counts past 2·√(SE² + (s²/2)(1/n₁ + 1/n₂)), s the spread between pretrainings ([ADR-0032](adr/0032-statistics-of-a-paired-comparison.md)) |
| Instrument | The closed-form probe over all blocks at 500 stays on half the tuning side, ten seeds; it resolves 0.02 to 0.03 in area, not 0.01 ([note](verification/verdict-statistics.md)) |

## Results

### Pretraining against training from scratch

FD001, one paired campaign, five seeds, the encoder pretrained on the four C-MAPSS subsets,
every arm tuned per budget by the same declared selection
([note](verification/label-efficiency-curve.md), [classical](verification/classical-baselines.md)).
RMSE, lower is better.

| Labelled windows | From scratch | Frozen probe | Full fine-tuning | Trees per channel | MiniRocket |
| --- | --- | --- | --- | --- | --- |
| 50 | **16.34** | 16.93 | 17.61 | 19.35 | 17.17 |
| 200 | 14.29 | 14.66 | 15.77 | **14.08** | 15.30 |
| 1,000 | 12.77 | 14.53 | 13.40 | **12.04** | 15.40 |
| all (2,568) | **11.47** | 14.18 | 12.71 | 11.76 | 15.42 |

Bold: the lowest mean. The registered endpoint, full fine-tuning at 200 windows, is 10.5 %
worse than from scratch (interval [−2.56, −0.54] RMSE).

PhysioNet 2012, half the tuning side, ten seeds ([note](verification/verdict-statistics.md);
1,000 and every stay of the classical rows and the diagnostic: [note](verification/gap-to-the-trees.md)).
AUROC, higher is better.

| Labelled stays | 200 | 500 | 1,000 | all (2,000) |
| --- | --- | --- | --- | --- |
| Network from scratch | 0.65 | 0.75 | 0.79 | 0.79 |
| Network from scratch, rerun beside the cells ([note](verification/encoder-candidates.md)) | 0.67 | | | |
| Network from scratch over cells of channel × time bin, 6 / 12 bins ([note](verification/encoder-candidates.md)) | 0.69 / 0.67 | | | |
| Small network from scratch (2 blocks, 64 wide) over 6-bin cells ([note](verification/encoder-candidates.md)) | 0.73 | 0.78 | | 0.83 |
| Network from scratch (4.8M) over 6-bin cells, 500 and every stay ([note](verification/encoder-candidates.md)) | | 0.77 | | 0.83 |
| Full fine-tuning, mixture of four | 0.69 | 0.74 | 0.76 | 0.79 |
| Probe, mixture of four | 0.65–0.67 | 0.69–0.71 | 0.72–0.74 | 0.74–0.76 |
| Probe, untrained encoder | 0.64 | 0.69 | 0.75 | 0.78 |
| Trees per channel | **0.74** | **0.79** | **0.82** | **0.84** |
| MiniRocket | 0.71 | 0.76 | 0.78 | 0.81 |
| Logistic head over the per-channel summaries | 0.73 | 0.77 | | 0.81 |
| Network over the per-channel summaries (diagnostic) | 0.71 | 0.75 | 0.79 | 0.81 |

Pretraining's share changes sign with the budget: the probe gains 0.02 to 0.04 over an untrained
encoder at 200 stays, nothing at 500 and loses 0.01 to 0.04 from 1,000 up. The classical rows
were read at 200 and 500 stays only ([note](verification/head-and-representation.md)). On the full grid
against set B, trees reach 0.76 at 200 stays and 0.86 at every stay, the network 0.68 and 0.81
([note](verification/intensive-care-curve.md)); the small encoder over six-bin cells, learnt from
every stay of set A, reads 0.85 on set B against the tuned trees' 0.86 and the default trees'
0.85 ([note](verification/encoder-candidates.md)). Under all four pretrainings, full fine-tuning
reads 0.58, 0.62 and 0.68 at 20, 50 and 200 stays, and the network from scratch 0.60, 0.62 and
0.66. The gain at 200 stays, +0.024 [+0.014; +0.033], is not replicated under the two
pretrainings never read there (+0.014 against 0.021), and neither 20 nor 50 stays passes
([note](verification/few-label-fine-tuning.md)).

The small encoder over six-bin cells, pretrained without labels on the intensive-care stays
(eight pretrainings), on the same half and seeds ([note](verification/cell-pretraining.md)):

| Labelled stays | 20 | 50 | 200 | 500 |
| --- | --- | --- | --- | --- |
| Full fine-tuning | 0.62 | 0.66 | 0.74 | |
| Network from scratch, rate in force / its chosen rate | 0.62 / 0.62 | 0.65 / 0.65 | 0.72 / 0.74 | |
| Probe, pretrained / untrained encoder | | | | 0.76 / 0.68 |
| Trees per channel | 0.62 | 0.63 | 0.74 | **0.79** |

### Is the pipeline sound?

| Check | Result | Source |
| --- | --- | --- |
| Generated data with known shared structure | the pretrained encoder's error 0.094 [0.089; 0.099] below a fresh one's, at a window spanning the process | [synthetic-transfer](verification/synthetic-transfer.md) |
| Published network (STraTS) on the same stays | 0.827; this network from scratch 0.818 once kept at its best epoch and shrunk to two blocks | [intensive-care-curve](verification/intensive-care-curve.md) |
| The network from scratch in two campaigns | differs by at most 0.005 | [verdict-statistics](verification/verdict-statistics.md) |
| Fine-tuning's gain on scored stays the backbone read, less on those it did not | −0.007 / −0.003 / −0.010 at 20 / 50 / 200 stays: no gain from having read them | [cell-pretraining](verification/cell-pretraining.md) |
| A pretext over cells of channel × time bin, answered without a model | on the stays a linear stack reads a hidden cell's mean at 0.12 and its presence at 0.59, from the other channels in the same bin; under a hidden tail the presence at 0.22; on the gridded corpora the presence is constant and the mean interpolated at 0.53 to 0.80 | [cell-pretext](verification/cell-pretext.md) |

### What changes the backbone

Each recipe read by the instrument at 500 stays against the mixture of four (four pretrainings).
Bold passes the threshold in force, s = max(0.014, the recipe's own): about 0.024 for one
pretraining against four, 0.021 for two.

| Recipe | Pretrainings | Gain in area | Source |
| --- | --- | --- | --- |
| Scale curve at about 10⁸ and 3·10⁸ values | 1 each | −0.018, −0.021 | [layer-readout](verification/layer-readout.md) |
| Scale curve at about 10⁹ values | 1 | 0.02 to 0.05 below every mixture pretraining | [verdict-statistics](verification/verdict-statistics.md) |
| 25.3M parameters at about 10⁹ values | 2 + 1 | −0.021 / +0.004; four epochs **−0.032** | [layer-readout](verification/layer-readout.md) |
| Intensive-care corpora alone | 4 | +0.015 [+0.005; +0.025], threshold 0.022 | [pretext-variants](verification/pretext-variants.md) |
| PhysioNet 2019 alone | 1 | +0.020 ± 0.005 | [layer-readout](verification/layer-readout.md) |
| The stays four times as often in the mixture | 1 | +0.012 ± 0.003 | [layer-readout](verification/layer-readout.md) |
| Masks over three quarters of a window | 1 | +0.008 ± 0.005 | [layer-readout](verification/layer-readout.md) |
| Masks over the forecast tail | 1 | −0.019 ± 0.005 | [layer-readout](verification/layer-readout.md) |
| Decoder of four blocks | 2 | **−0.037** | [pretext-variants](verification/pretext-variants.md) |
| Predicting a teacher's states instead of values | 2 | **−0.055** | [pretext-variants](verification/pretext-variants.md) |
| Contrast between a window's two halves | 2 | **−0.103** [−0.125; −0.081] | [pretext-variants](verification/pretext-variants.md) |
| Each channel's summaries of the hidden part beside reconstruction | 2 | not read: on the stays the term is learnt no better than a linear answer without a model (0.57 against 0.63 in R²) | [pretext-variants](verification/pretext-variants.md) |
| 10⁹ continued on the task's domain, rate 1e-4 / 1e-3 | 2 / 1 | **−0.025** / −0.002 | [domain-continuation](verification/domain-continuation.md) |

### How the backbone is used

| Change | Result | Source |
| --- | --- | --- |
| All blocks side by side instead of the last | +0.020 to +0.036 at 50 stays; registered as the reader | [layer-readout](verification/layer-readout.md) |
| Encoder slowed to 1/10 or 1/100 during fine-tuning (LP-FT) | loses up to 0.11; at the full rate within 0.01 of fine-tuning | [fine-tuning-recipe](verification/fine-tuning-recipe.md) |
| Mean per channel instead of over the stay, reduced to the probe's width | +0.037 / +0.038 at 200 / 500 stays; 0.699 / 0.735, 0.031 / 0.034 below the summaries' head, and the same at initialisation; a diagnostic, bound to the corpus's channels | [head-and-representation](verification/head-and-representation.md) |
| Mean over the channels a stay measured, free of their layout | −0.058 / −0.052 behind the per-channel mean at 200 / 500 stays, and no better than the mean over the stay; the same at initialisation | [head-and-representation](verification/head-and-representation.md) |
| Network kept at its best epoch on held-out labels | +0.027 at every stay, spread over seeds 0.018 → 0.006 | [intensive-care-curve](verification/intensive-care-curve.md) |

### Transfer to a corpus the backbone never read

| Reading | Result | Source |
| --- | --- | --- |
| Pretext, unseen channels left as drawn | loses to interpolation and cross-channel regression by 0.12 to 0.21 | [zero-shot-pretext](verification/zero-shot-pretext.md) |
| Pretext, only the channel rows fitted on unlabelled windows | within 0.006 of the backbone that read the corpus (2012, C-MAPSS); 0.02 behind on 2019 | [zero-shot-pretext](verification/zero-shot-pretext.md) |
| Pretext, rows copied from the same quantity in another corpus, no fit | within 0.01 to 0.02 of the fitted rows | [zero-shot-pretext](verification/zero-shot-pretext.md) |
| Sepsis task at 4,000 patients, rows fitted | +0.031 [+0.015; +0.047] over drawn rows; none at 500 stays on 2012 | [task-transfer](verification/task-transfer.md) |

### What the failures have in common

- **A better pretext loss is not a better backbone.** Thirteen passes over the mixture lower the
  pretext loss and lose 0.057 under fine-tuning to eight passes; continuing on the task's domain
  at the low rate lowers it and moves nothing the reader sees.
- **The loss sits in the top blocks.** Under the point of about 10⁹ values the early blocks read
  like an untrained encoder and the deficit grows block by block to the last.
- **The pretext removes what the task reads from the probe's average, not from the tokens.**
  The mean over a stay holds less of its levels, extremes and measurement counts than the same
  encoder at initialisation — half the level, from the first block — and the loss falls on the
  channels measured a few times a stay. Each channel's own tokens hold the level as before
  training; the probe reads values, and loses 0.06 to 0.13 when they are replaced
  ([note](verification/head-and-representation.md)).
- **Pretexts are solved by cues that are not the task.** Under masks over three quarters of a
  window the stays' single tokens are matched by interpolation alone; the contrast between a
  window's halves was solved by the stays' clock of measurements, which pairs the halves without
  error with no model; each channel's summaries of a stay's hidden part are read by a linear
  answer from the visible part, and a term asking for them learns nothing past it.
- **Pretraining reads little that is sampled like the intensive-care tasks.** Only the stays
  read their channels at their own times: a fifth to a third of a stay's channels at one instant,
  rates 14 to 24 times apart, 2 to 4 readings of a channel in a window. They are 4.5 % of the
  mixture's windows and 1 to 3 % of the values along the scale curve. The satellite telemetry,
  declared irregular, is read as a grid, and most of the rest are dense grids where a hidden value
  lies between readings of its own channel ([note](verification/pretraining-corpus-sampling.md)).

## Limitations

- **Validation only.** Each configuration was chosen on the same validation side it is reported
  on; the single test run removes that bias.
- **Resolution.** At 500 stays the instrument resolves 0.02 to 0.03 in area; published gains of
  self-supervision in the clinic at 10 % of labels or more are 0.004 to 0.013, below it.
- **Few independent units.** FD001 has 21 validation engines; its 95 % interval covers about
  92 % on known answers.
- **Two main tasks.** Transfer across corpora is read at the pretext and on sepsis within one
  pretraining; the intensive-care task learns from set A only, so its levels are not comparable
  with published benchmarks.
- **Controls' rates.** The network from scratch chose the lowest rate on its grid, as did
  fine-tuning; a lower one was not read. Between two runs it moved by 0.011 at 200 stays.
- **Accelerators.** Arms ran on Colab G4, A100 and Kaggle T4; between accelerators an arm drifts
  by up to 0.7 RMSE.

## Next

1. Pretraining data in the tasks' regime for both legs, then anomaly tasks on telemetry
   without labels and channels added without labels, read on the representation that
   pretraining over cells now gives; the clinical confirmation as a declared curve against the
   network from scratch at its chosen rate.
2. Read transfer to corpora the backbone never read on their tasks, across pretrainings.
3. Open the frozen test side once.
