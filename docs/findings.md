# Findings

What the project has measured, on one page. Every number is on the **validation side** and
preliminary: each task's frozen test side is opened once, at the end. Each comparison was judged
by a rule registered before its run ([preregistration](preregistration.md)); each row links the
note that holds its evidence ([`verification/`](verification/README.md)).

Last updated 2026-10-20.

## Status

- **The claim is not supported so far.** No way of using the pretrained encoder beats the same
  network trained from scratch beyond the instrument's resolution, on either task, with one
  exception: full fine-tuning at 200 intensive-care stays (+0.024 to +0.033 in area), below the
  registered least gain.
- **Classical models lead at few labels.** Trees over per-channel summaries match or beat every
  network from 200 labels up on both tasks; on the intensive-care stays MiniRocket does too.
- **What transfers is the encoder's machinery, not its features for the task.** Fitting only the
  channel rows of a corpus the backbone never read, on its unlabelled windows, brings the pretext
  to within 0.006 of a backbone that read it, and carries part of a downstream task (+0.031 on
  sepsis at 4,000 patients).
- **Ruled out as levers:** more pretraining data, a five times larger model, continuing on the
  task's own domain, the adaptation recipe, and five variants of the pretext.
- **Next:** measure what the task reads at few labels and whether the encoder's states carry it,
  before another objective is trained.

## Setup

| | |
| --- | --- |
| Tasks | Remaining useful life on C-MAPSS FD001 (RMSE in cycles; 79 tuning and 21 validation engines); death in hospital on PhysioNet 2012 (AUROC over the first 48 h of a stay); sepsis on PhysioNet 2019 as a reader within one backbone |
| Pretraining corpora | Mixture of four: C-MAPSS, SKAB, ESA satellite telemetry, PhysioNet 2012 ([ADR-0029](adr/0029-pretraining-over-a-mixture-of-corpora.md)); up to about 10⁹ values over 26 publications for the scale curve |
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

PhysioNet 2012, half the tuning side, ten seeds ([note](verification/verdict-statistics.md)).
AUROC, higher is better.

| Labelled stays | 200 | 500 | 1,000 | all (2,000) |
| --- | --- | --- | --- | --- |
| Network from scratch | 0.65 | 0.75 | 0.79 | 0.79 |
| Full fine-tuning, mixture of four | 0.69 | 0.74 | 0.76 | 0.79 |
| Probe, mixture of four | 0.65–0.67 | 0.69–0.71 | 0.72–0.74 | 0.74–0.76 |
| Probe, untrained encoder | 0.64 | 0.69 | 0.75 | 0.78 |

Pretraining's share changes sign with the budget: the probe gains 0.02 to 0.04 over an untrained
encoder at 200 stays, nothing at 500 and loses 0.01 to 0.04 from 1,000 up. On the full grid
against set B, trees reach 0.76 at 200 stays and 0.86 at every stay, the network 0.68 and 0.81
([note](verification/intensive-care-curve.md)).

### Is the pipeline sound?

| Check | Result | Source |
| --- | --- | --- |
| Generated data with known shared structure | the pretrained encoder's error 0.094 [0.089; 0.099] below a fresh one's, at a window spanning the process | [synthetic-transfer](verification/synthetic-transfer.md) |
| Published network (STraTS) on the same stays | 0.827; this network from scratch 0.818 once kept at its best epoch and shrunk to two blocks | [intensive-care-curve](verification/intensive-care-curve.md) |
| The network from scratch in two campaigns | differs by at most 0.005 | [verdict-statistics](verification/verdict-statistics.md) |

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
| 10⁹ continued on the task's domain, rate 1e-4 / 1e-3 | 2 / 1 | **−0.025** / −0.002 | [domain-continuation](verification/domain-continuation.md) |

### How the backbone is used

| Change | Result | Source |
| --- | --- | --- |
| All blocks side by side instead of the last | +0.020 to +0.036 at 50 stays; registered as the reader | [layer-readout](verification/layer-readout.md) |
| Encoder slowed to 1/10 or 1/100 during fine-tuning (LP-FT) | loses up to 0.11; at the full rate within 0.01 of fine-tuning | [fine-tuning-recipe](verification/fine-tuning-recipe.md) |
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
- **Pretexts are solved by cues that are not the task.** Under masks over three quarters of a
  window the stays' single tokens are matched by interpolation alone; the contrast between a
  window's halves was solved by the stays' clock of measurements, which pairs the halves without
  error with no model.

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
- **Accelerators.** Arms ran on Colab G4, A100 and Kaggle T4; between accelerators an arm drifts
  by up to 0.7 RMSE.

## Next

1. Measure what the task reads at few labels, where trees over per-channel summaries lead, and
   whether the mixture's states carry it, before another objective is trained.
2. Read transfer to corpora the backbone never read on their tasks, across pretrainings.
3. Open the frozen test side once.
