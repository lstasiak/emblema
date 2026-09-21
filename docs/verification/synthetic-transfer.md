# The transfer leg of the synthetic control

Purpose: read the two rules registered for the synthetic control (`docs/preregistration.md`, "The
synthetic control", and the sections of 2026-09-21 that state how the leg is run) on the corpora
built for it, so that a negative result on real data can be told from a broken pipeline. A pair
that shares latent structure must show an advantage of the pretrained arm over training from
scratch, above the practical floor with its whole interval above zero; a pair that shares none
must show equivalence, the whole interval within ±the floor.

Every number is validation, not test, and every run is at tier S on this machine.

## 2026-09-21 — Darwin arm64 (MacBook Pro M1 Pro, MPS, fp32): both pairs at the endpoint's budget

Code `0e9b0f5` for the runs, Python 3.14.7, torch 2.14.0. The pieces:

|  |  |
| --- | --- |
| Task | the exact reading of `s01` twelve time units past the window's end, on the second layout of each pair; four strata; error in the sensor's units (ADR-0033) |
| Backbones | `control-a-s` (`0e295940-…`, weights `sha256:f285de7f…`) and `null-a-s` (`061b9f6e-…`, weights `sha256:9a2e4814…`), each 24 epochs of batch 32 over its first layout, fp32 on MPS, 835 s; validation loss per hidden token 0.01371 and 0.32654, 0.014 and 0.334 of the trivial predictor's |
| Corpora | `control-b-wide` (`sha256:ddb58ea1…`) and `null-b-wide` (`sha256:23cbf2a8…`): the second layouts with 800 units and no other dial turned, published under the first layouts' vocabularies, windows of 32 at stride 12, half the units held out under seed 1 — 400 tuning units, 133 frozen, 266–267 validation units, 9,300 or so validation windows |
| The backbone over the second layout's channels | its channel table grown by rows for them, drawn at build time and trained under every mode (ADR-0033, 2026-09-21) |
| Runs | 200 labelled windows under seeds 1 to 5, the floor of 2,000 optimiser steps (154 epochs of batches of 16, 2,002 steps), the head's bias at the mean label, warm-up over the first tenth and a cosine decay to one per cent; peaks from the sweep below: from scratch 1e-3, frozen probe 1e-2, low-rank updates 3e-3 (rank 8, α = 16), full fine-tuning 1e-3 |
| Stored | `data/report/transfer/endpoint-<task>` (cells), `data/report/curve/endpoint-<task>` (readings); the sweeps under `data/report/transfer/sweep-<task>/<arm>-lr<peak>` |

### The sweep of the peaks, on each pair's own task

Three peaks per arm, seeds 1 to 3, the rule of 2026-09-20; mean validation RMSE over the three
seeds, the chosen cells in bold. Full fine-tuning was swept a decade above the turbofan grid after
a pilot on the narrow null corpus (`null-b`, 20 validation units) found the turbofan peak of 3e-5
a decade too small there: 0.508 against 0.185 from scratch.

| arm | peak | null pair | coupled pair |
| --- | --- | --- | --- |
| from scratch | 1e-3 / 3e-4 / 1e-4 | **0.194** / 0.195 / 0.202 | **0.368** / 0.405 / 0.423 |
| frozen probe | 1e-2 / 3e-3 / 1e-3 | **0.898** / 0.903 / 0.907 | **0.661** / 0.686 / 0.712 |
| low-rank updates | 3e-3 / 1e-3 / 3e-4 | **0.242** / 0.307 / 0.347 | **0.373** / 0.413 / 0.461 |
| full fine-tuning | 1e-3 / 3e-4 / 1e-4 | **0.232** / 0.319 / 0.376 | **0.369** / 0.395 / 0.426 |

Power, read off the chosen cells with the three seeds pooled per unit: the spread over the
validation units of the paired difference of per-unit RMSE between the control and full
fine-tuning is 0.033 on the null pair and 0.035 on the coupled pair, a half-width of 0.004 on
either against floors of 0.010 and 0.011; the 266 units are six times more than the floor asks.

### The endpoint, five seeds

Validation RMSE in the sensor's units per seed, and the paired reading against the control by
the registered rules (`scripts/label_curve_report.py`; interval over units, 2,000 resamples;
floor = max(3 % of the control's RMSE, its SD over the seeds)):

| pair | arm | seed 1 | 2 | 3 | 4 | 5 | mean | SD | trainable | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coupled | from_scratch | 0.376 | 0.366 | 0.352 | 0.375 | 0.356 | 0.365 | 0.011 | 1,788,481 | 48–52 |
| coupled | frozen_probe | 0.679 | 0.600 | 0.706 | 0.688 | 0.643 | 0.663 | 0.042 | 1,345 | 9–10 |
| coupled | lora | 0.387 | 0.381 | 0.370 | 0.481 | 0.414 | 0.407 | 0.045 | 99,649 | 51–54 |
| coupled | full_fine_tuning | 0.364 | 0.378 | 0.370 | 0.429 | 0.373 | 0.383 | 0.026 | 1,788,481 | 53–54 |
| null | from_scratch | 0.188 | 0.187 | 0.206 | 0.186 | 0.181 | 0.190 | 0.010 | 1,788,481 | 49–56 |
| null | frozen_probe | 0.934 | 0.837 | 0.922 | 0.859 | 0.731 | 0.857 | 0.081 | 1,345 | 9–10 |
| null | lora | 0.255 | 0.209 | 0.262 | 0.255 | 0.253 | 0.247 | 0.022 | 99,649 | 52–54 |
| null | full_fine_tuning | 0.209 | 0.211 | 0.245 | 0.235 | 0.244 | 0.229 | 0.018 | 1,788,481 | 53–54 |

The mean predictor scores 1.003 on the coupled pair's validation windows and 1.011 on the null
pair's. The trainable count of the probe and the low-rank arm includes the 1,088 weights of the
rows grown for the second layout's six channels, trained under every mode.

| pair | comparison against from scratch | control | candidate | reduction | 95 % interval | floor | rule | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| coupled | full fine-tuning | 0.365 | 0.384 | −0.019 | [−0.022, −0.015] | 0.011 | advantage above the floor, interval above zero | **fails**: worse, beyond the floor |
| coupled | low-rank updates | 0.365 | 0.408 | −0.043 | [−0.048, −0.039] | 0.011 | — | worse |
| coupled | frozen probe | 0.365 | 0.664 | −0.299 | [−0.310, −0.288] | 0.011 | — | worse |
| null | full fine-tuning | 0.190 | 0.229 | −0.040 | [−0.043, −0.036] | 0.0095 | equivalence, interval within ±floor | **fails**: worse by four floors |
| null | low-rank updates | 0.190 | 0.248 | −0.058 | [−0.061, −0.054] | 0.0095 | — | worse |
| null | frozen probe | 0.190 | 0.860 | −0.670 | [−0.689, −0.651] | 0.0095 | — | worse |

### What the leg says

- **The coupled pair fails its rule.** Full fine-tuning of the backbone pretrained on the first
  layout ends 0.019 above training from scratch at 200 labelled windows, with the whole interval
  below zero and beyond the floor; the low-rank arm is further behind. Under the registered
  reading the pipeline does not show transfer where transfer exists by construction, and until
  this is understood a result on real data cannot be told from a pipeline that would miss the
  effect if it were there.
- **The null pair fails its equivalence, in the same direction.** Every pretrained arm is worse
  than the control, full fine-tuning by four floors. A start from weights pretrained on
  unrelated structure is a worse start than a random one for this task at this budget, and 2,002
  steps at the largest peak swept do not close the gap.
- **The backbone did learn the shared structure, and it reaches the second layout's sensors.**
  The frozen probe stands at 0.66 on the coupled pair and at 0.86 on the null pair, twenty floors
  apart, over channels the backbone never saw and whose rows were drawn at build time: the states
  of a window computed by the backbone of the coupled pair carry the shared factors' phase and
  amplitude, and those of the null pair carry almost nothing beyond the mean (1.01). The failure
  is not that the encoder finds nothing; it is that what it finds does not beat a fresh encoder
  trained for 2,002 steps on 200 windows of a task a fresh encoder solves to 0.37, and that
  starting from it costs the fine-tuned arms something the fresh start does not pay.
- **The ordering holds at other budgets, under one seed.** Twelve cells of the coupled pair run
  before the grid was reduced to the endpoint (`data/report/transfer/grid-control-b-wide-forecast`,
  seed 1): at 50 labelled windows from scratch 0.473, full fine-tuning 0.536, low-rank 0.511; at
  1,000, 0.285, 0.284 and 0.269. Not a reading, one seed, but nothing in it points the other way.
- **Cost.** An arm that steps the encoder costs 48–56 s a run on MPS at this budget, the probe
  9–10 s; the two sweeps took 55 min, the forty endpoint runs 28 min; the S pretrainings 14 min
  for both, run together.

### What it does not say, and what comes next

The rules were written for a pipeline that either transfers or does not, and the leg answers
that it does not, on a task and a budget where a fresh encoder is strong. It does not say which
of three things is missing, and the next runs are the ones that separate them:

1. **The ceiling.** The two layouts with equal `trajectory_seed` watch the same factor
   trajectories (ADR-0018): a leak, not a control, and therefore the most transfer can give.
   If the fine-tuned arm does not beat the control even there, the fault is above the structure
   of the data — in the objective, the head, or the way a pretrained start is fine-tuned.
2. **The objective.** Masked reconstruction within a window teaches interpolation; the task asks
   for a reading twelve steps past the window. A probe that reaches 0.66 shows the states carry
   the factors; a fine-tuning that ends above the fresh start suggests the states are arranged
   for reconstruction in a way the forecasting head has to undo first.
3. **The task and budget.** Two hundred windows and 2,002 steps let a fresh encoder reach 0.37 on
   a target that is a smooth function of four factors; a control whose fresh arm is that strong
   leaves little for a pretrained start to add. A harder target — fewer windows, a longer horizon,
   more noise — is the registration's to change before it is run, not after.

Nothing on the turbofan task is read until the coupled pair passes: that is what the control is
for.
