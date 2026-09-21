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

## 2026-09-21 — the ceiling: the second layout over the first's own trajectories

Declared before it ran (`docs/preregistration.md`, 2026-09-21, "the ceiling of the synthetic
transfer"). The layout `control-b-shared` is `control-b` with the first layout's `trajectory_seed`
and its 160 units, so that unit *i* watches the factor trajectories unit *i* of `control-a` was
pretrained on, under the second layout's sensors, sampling, noise and private factors: a leak,
and so the most transfer between the two layouts can give. Published under the first layout's
vocabulary (manifest `sha256:d3a0317a…`), half the units held out under seed 1: 80 tuning units,
27 frozen, 53 validation units. The runs are the endpoint's, under the coupled pair's peaks and
the backbone `control-a-s`; code `3426e0c` with the layout and its task uncommitted at the time.

| arm | seed 1 | 2 | 3 | 4 | 5 | mean | SD | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| from_scratch | 0.384 | 0.368 | 0.358 | 0.383 | 0.344 | 0.368 | 0.017 | 47–59 |
| frozen_probe | 0.692 | 0.580 | 0.695 | 0.691 | 0.670 | 0.666 | 0.049 | 3–10 |
| lora | 0.359 | 0.389 | 0.368 | 0.440 | 0.360 | 0.383 | 0.034 | 50–67 |
| full_fine_tuning | 0.394 | 0.378 | 0.341 | 0.378 | 0.356 | 0.369 | 0.021 | 49–58 |

The mean predictor scores 0.982 on these windows; the floor is 0.017, the control's spread over
the seeds.

| comparison against from scratch | control | candidate | reduction | 95 % interval | reading |
| --- | --- | --- | --- | --- | --- |
| full fine-tuning | 0.368 | 0.370 | −0.002 | [−0.009, +0.005] | indistinguishable; the interval within the floor either way |
| low-rank updates | 0.368 | 0.384 | −0.016 | [−0.023, −0.010] | worse, at the floor |
| frozen probe | 0.368 | 0.667 | −0.299 | [−0.322, −0.279] | worse |

**What the ceiling says.** Even where the second layout's units are the first's trajectories
themselves, fine-tuning the pretrained backbone ends exactly where a fresh encoder ends, 0.369
against 0.368, and the frozen probe stands where it stood on the coupled pair, 0.67 against
0.66. The leak bought nothing. By the reading declared beforehand, the fault is above the data:
the coupled pair's failure is not a fact about its private factors, its sampling or the size of
its corpus, because removing the one thing that separates the two layouts' trajectories changes
no number. What the backbone learns under masked reconstruction of a 32-step window, and what
a forecasting head can take from it, is where the next question lies.

**What was checked before reading it.** The stored weights are loaded into the encoder
(`TrainedModel.build` loads the state dictionary and the restored encoder equals the stored one
weight for weight, by test); every cell of both pairs and of the ceiling ran with the intended
weights and manifest, 2,002 steps, batches of 16, the registered schedule and peaks, the low-rank
shape of rank 8 and α = 16, the run's seed equal to the draw's, as `runs.csv` records. As a
contrast, the frozen probe on the coupled pair's corpus under the *null* pair's backbone scores
0.80–0.89 over five seeds (`data/report/transfer/swap-check-control-b-wide-under-null-a`), against
0.66 under its own: the 0.66 comes from the weights pretrained on the shared structure, not from
the corpus or a mix-up of artefacts.

**What separates the remaining explanations**, each a registration before a run:

- *The window against the factors' periods.* A window spans 32 time units; the factors' periods
  run from 24 to 300. Masked reconstruction inside such a window is solved by local
  interpolation, which needs no model of the factors' dynamics, and a forecast twelve steps past
  the window needs exactly that model for the slow factors. A window several periods long
  (128 or 256) is the cheapest change: two publications, one S pretraining of minutes, twenty
  runs.
- *The regime of the task.* A fresh encoder reaches 0.37 in 2,002 steps at 200 windows and 0.285
  at 1,000 (one seed); the endpoint's budget may be one where the labels already teach what
  pretraining could have. Fifty windows, or a longer horizon, moves the comparison to where a
  representation should matter, at the cost of a floor read off a weaker control.
- *The pretext itself.* An objective that predicts forward in time rather than reconstructing
  in place would teach the dynamics the task needs; it is the largest change and the last to
  try, because the two above can be measured in an hour.

## 2026-09-21 — the window against the factors' periods: the coupled pair at a window of 128

Declared before it ran (`docs/preregistration.md`, 2026-09-21, "the window against the factors'
periods"). The coupled pair's corpora republished with windows of 128 time units at the same
stride of 12: `control-a` (manifest `sha256:eb997d64…`, validation 0.25, seed 1) and
`control-b-wide` under its vocabulary (`sha256:9fb6c129…`, half the units held out, seed 1; 266
validation units, 7,906 validation windows). `control-a-s` pretrained again on the new corpus
under the same experiment file (backbone `ace3fb98-…`, weights `sha256:488be6bd…`, 24 epochs of
batch 32 on MPS, 1,867 s; validation loss per hidden token 0.01722, 0.017 of the trivial
predictor's, against 0.01371 at the window of 32). The runs are the endpoint's under the peaks
registered for the pair at 32, carried over as declared; code `b463ed1`.

| arm | seed 1 | 2 | 3 | 4 | 5 | mean | SD | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| from_scratch | 0.466 | 0.416 | 0.335 | 0.376 | 0.341 | 0.387 | 0.055 | 97–104 |
| frozen_probe | 0.898 | 0.848 | 0.861 | 0.894 | 0.900 | 0.880 | 0.024 | 11–12 |
| lora | 0.290 | 0.294 | 0.292 | 0.325 | 0.276 | 0.296 | 0.018 | 117–129 |
| full_fine_tuning | 0.294 | 0.284 | 0.306 | 0.310 | 0.291 | 0.297 | 0.011 | 95–101 |

The mean predictor scores 1.002 on these windows; the floor is 0.055, the control's spread over
the seeds, which at this window is five times what it was at 32.

| comparison against from scratch | control | candidate | reduction | 95 % interval | floor | reading |
| --- | --- | --- | --- | --- | --- | --- |
| full fine-tuning | 0.387 | 0.297 | **+0.093** | [+0.088, +0.098] | 0.055 | **the rule's condition met**: above the floor, the whole interval above zero |
| low-rank updates | 0.387 | 0.296 | +0.094 | [+0.088, +0.099] | 0.055 | distinguishable, above the floor |
| frozen probe | 0.387 | 0.880 | −0.491 | [−0.508, −0.474] | 0.055 | worse |

**What the diagnostic says.** With windows that span the factors' periods, the pretrained
backbone transfers: full fine-tuning and the low-rank updates end a quarter below the fresh
encoder, on every seed, with the fresh encoder's own spread five times the pretrained arms'. The
window was the fault. A 32-step window under masked reconstruction taught interpolation; a
128-step one teaches something the forecasting head can use and a fresh encoder does not learn
from 200 windows in 2,002 steps. Two things in the same table keep the reading honest: the
control got worse and noisier at the longer window (0.387 ± 0.055 against 0.365 ± 0.011), so
part of the advantage is the fresh encoder's difficulty with four times the tokens at the same
budget of steps, which is a real property of the comparison and not an artefact; and the frozen
probe got worse (0.88 against 0.66), so the states the longer-window backbone produces are less
linearly readable while being a better start to fine-tune from.

**What was checked.** The cells ran under the backbone accepted for this corpus (`488be6bd…`),
the manifest at 128, the registered plan (2,002 steps, batches of 16, the schedule, the peaks
of the pair) and the run's seed equal to the draw's, as `runs.csv` records; the pretraining
converged to a loss of the same order as at 32.

**What follows, each a registration before a run.** By the reading declared beforehand, the
control's registration moves to the longer window before the pair is measured again under its
rule: the peaks swept at 128 on each pair's own task, the null pair republished and pretrained at
128 and measured for its equivalence, and the coupled pair read again with swept peaks. Only
then does the control pass "both ways", and only then is a result on real data read. The
turbofan task's own window, 50 cycles against lives of hundreds, is the same question asked of
the real data, to be stated in its registration before any grid.

