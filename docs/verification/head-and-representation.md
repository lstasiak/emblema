# Head, representation or task: where the network loses to the trees

Diagnostics on the validation side of the turbofan task, made before any network is tuned or
the label-efficiency curve is repeated. Each diagnosis is declared here, with its prediction and
the reading each outcome gets, before it runs.

## 2026-09-25 — declared before the run

**Question.** At 200 labels on the turbofan task the tuned trees per channel score 13.94, the
best pretrained arm 16.10, the network from scratch 18.86 and a patch model from scratch 18.40
([`classical-baselines.md`](classical-baselines.md),
[`label-efficiency-curve.md`](label-efficiency-curve.md); the arms are not paired with the
trees). The frozen probe (16.33) sits within a third of a cycle of full fine-tuning (16.68):
training the encoder adds nothing over its frozen states under a linear head. Three explanations,
not exclusive:

1. **The head.** Both networks average about a thousand token states over the whole window
   before a linear map. The mean loses where each channel stands as the window ends, which is
   what the trees read first.
2. **The representation.** The masked-reconstruction pretext did not put the state of
   degradation into the states, so no head could read it out.
3. **The task.** Remaining life at this window is a regression of the current readings, which
   the trees read directly and a network has to extract.

**Design.** Everything is read off frozen states, on the validation side, in minutes on the
host's accelerator; nothing is fine-tuned. Two scripts: `scripts/frozen_representations.py`
embeds and pools once, `scripts/head_and_representation_report.py` fits, pairs and renders in a
process without torch.

| | |
| --- | --- |
| Corpus, task | `cmapss` in force, manifest `sha256:d63f8e1b…`; FD001, 79 tuning / 21 validation engines, remaining life under a ceiling of 125 |
| Encoders | the backbone in force `sha256:6283c210…` (tier M, width 256, 4.78 M weights), frozen; an encoder of the same shape never trained, drawn under seed 1 |
| Labels | budgets 50 and 200, seeds 1–3, the draws a campaign makes (`LabelSample.drawn` over the tuning side); every probe of one cell learns from the same windows |
| Scored | every validation window; paired over the 21 engines |
| Inputs | the statistics per channel the trees read (126 channels × 10); their `last` columns alone; the states pooled: the mean over the window; the mean over the last 2, 10 and 20 % of it (1, 5 and 10 cycles); one mean per channel the tuning windows observe (21 × 256); the state of each channel's latest token (21 × 256) |
| Fitters | ridge with an intercept, columns scaled by their spread, penalty chosen by leave-one-out among the convolution baseline's ten (10⁻³ to 10³); boosted trees on the library's knobs at the depth the selection chose (3 at 200, 6 at 50), seed of the fit = seed of the draw, 4 threads |
| Statistics | repeats pooled per engine over the 3 seeds; percentile bootstrap over engines, 10,000 resamples, seed 1; no family correction, since these are diagnostics and not a verdict, and each prediction names its own contrast |
| Machine | MacBook Pro M1 Pro, 32 GB; MPS, fp32 for the states; CPU for the fits |

**Check before anything is read.** The trees on the statistics per channel are the trees of
campaign `ee69d456…` over the same rows and knobs: they must score 13.94 ± 0.71 at 200 and
18.25 ± 1.47 at 50 (mean ± SD over the three seeds). If they do not, the draws differ and none
of the rows below is read.

**Predictions**, at 200 labels; 50 is reported beside and read the same way.

- **P1, the head.** Under one ridge, the tail of 10 % reduces the error of the mean pooling by
  at least 5 % with the interval above zero, and each per-channel pooling (the mean per channel,
  the latest state per channel) by at least 10 %.
- **P2, the representation.** Trees on the latest state per channel land within 10 % of the
  trees on the statistics per channel; trees on the mean-pooled states land more than 10 % behind
  them. Under one fitter, the pretrained states beat the untrained encoder's by at least 10 % at
  the mean pooling.
- **P3, the task.** Ridge on the 21 last values lands within 10 % of the trees on the statistics
  per channel, that is at about 14 to 15.5; trees on the last values within 5 %.

**Reading, declared beforehand.**

- P1 holds, P3 holds and the per-channel clause of P2 holds: the head is the bottleneck. The
  arms and the patch model get a pooling knob that keeps the channel layout out of the head
  (the tail, learnt attention), confirmed by a fine-tuning campaign before the curve is
  repeated; the networks' hyperparameter search includes the pooling.
- The per-channel clause of P2 fails while the pretrained states do no better than the
  untrained ones: the representation is the problem. The pretext does not encode the state of
  degradation, and the objective is revisited ([ADR-0019](../adr/0019-self-supervised-objective.md),
  [ADR-0029](../adr/0029-pretraining-over-a-mixture-of-corpora.md)) before any head is tuned.
- P3 fails, the last values far behind the trees: the task needs the trend within the window
  and not only the level, and the poolings that keep the trend (per channel, attention) matter
  more than the tail.
- Mixed outcomes are read contrast by contrast. Whatever they show, the confirmation runs as a
  campaign through the harness, not through these scripts.

**Cost.** Two encoders over 3,186 windows on MPS; 18 probes × 2 budgets × 3 seeds = 108 fits of
seconds each; the comparisons in pure Python.

    uv run scripts/frozen_representations.py --out data/report/head-and-representation \
        --weights durable/sha256/6283c210… sha256:6283c210… \
        --manifest durable/sha256/d63f8e1b… sha256:d63f8e1b…
    uv run scripts/head_and_representation_report.py data/report/head-and-representation
