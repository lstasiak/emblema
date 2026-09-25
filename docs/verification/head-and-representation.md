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

## 2026-09-25 — M1 Pro, MPS, fp32: the head loses most of it; the task is not the last reading

**Question.** As declared above: with the backbone frozen, how much of the gap to the trees is the
pooling, how much the states, and how much the task's own shape? All numbers are **validation**,
tier S for the fits, the backbone at tier M.

**Conditions.** Commit `52db8a1` (the fix of a column constant up to rounding, made after the
first fit of the ridge stopped on it, changes no number of the trees and only which columns a
ridge reads). 2,568 tuning and 618 validation windows of 1,050 tokens; 21 channels read; both
encoders embedded in 43 s each on MPS; 108 fits of at most 2.7 s on the CPU. The ridge dropped
the statistics of the six sensors FD001 never moves, and reads 119 of the 1,260 hand columns.

**Check.** The trees on the hand statistics repeat campaign `ee69d456…` to the hundredth in
every seed (13.65 / 13.41 / 14.75 at 200; 18.25 / 19.71 / 16.78 at 50). The draws are the
campaign's, and every row below is read.

RMSE on the 21 validation engines, mean ± SD over 3 seeds. Inputs: `hand` = the trees' statistics
per channel; `last` = their last value per channel; `mean`, `tail_k` = the frozen states pooled
over the window or its last k %; `channel_mean`, `channel_last` = one pooled state per channel;
`fresh_*` = the same over an encoder never trained.

| probe | columns | 50 | 200 |
| --- | --- | --- | --- |
| hand/trees (reference) | 1260 | 18.25 ± 1.47 | 13.94 ± 0.71 |
| hand/ridge | 119 | 19.67 ± 0.93 | 16.47 ± 1.01 |
| last/ridge | 14 | 21.98 ± 0.67 | 21.56 ± 0.33 |
| last/trees | 126 | 24.64 ± 1.64 | 22.61 ± 0.88 |
| mean/ridge | 256 | 18.14 ± 0.86 | 16.57 ± 1.01 |
| mean/trees | 256 | 25.14 ± 2.54 | 20.64 ± 0.63 |
| tail_2/ridge | 256 | 18.64 ± 0.68 | 16.19 ± 0.37 |
| tail_10/ridge | 256 | 17.40 ± 0.16 | 15.27 ± 0.33 |
| tail_10/trees | 256 | 23.31 ± 1.94 | 17.98 ± 0.67 |
| **tail_20/ridge** | 256 | **17.06 ± 0.14** | **14.73 ± 0.54** |
| channel_mean/ridge | 5376 | 17.02 ± 1.06 | 14.76 ± 0.19 |
| channel_mean/trees | 5376 | 25.14 ± 2.32 | 17.83 ± 0.85 |
| channel_last/ridge | 5376 | 17.86 ± 0.20 | 15.97 ± 0.06 |
| channel_last/trees | 5376 | 22.61 ± 1.34 | 17.91 ± 0.49 |
| fresh_mean/ridge | 256 | 23.07 ± 1.00 | 17.89 ± 0.28 |
| fresh_mean/trees | 256 | 26.99 ± 1.46 | 23.38 ± 0.22 |
| fresh_channel_last/ridge | 5376 | 20.73 ± 0.13 | 18.44 ± 0.74 |
| fresh_channel_last/trees | 5376 | 23.75 ± 1.08 | 20.77 ± 0.20 |

The declared contrasts at 200, paired over engines with the three seeds pooled (10,000
resamples); reduction of the rival's error, positive when the candidate is better.

| prediction | candidate | rival | reduction | 95 % interval | p | declared | held |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1 head | tail_10/ridge | mean/ridge | +8.0 % | [+0.60, +2.09] | 0.0006 | ≥ 5 %, above zero | **yes** |
| P1 head | channel_mean/ridge | mean/ridge | +11.0 % | [+0.98, +2.66] | 0.0002 | ≥ 10 % | **yes** |
| P1 head | channel_last/ridge | mean/ridge | +3.8 % | [−0.69, +2.02] | 0.37 | ≥ 10 % | no |
| P2 representation | channel_last/trees | hand/trees | −28.4 % | [−5.65, −2.22] | 0.0004 | within 10 % | no |
| P2 representation | mean/trees | hand/trees | −48.0 % | [−8.93, −4.35] | 0.0002 | more than 10 % behind | yes |
| P2 representation | mean/trees | fresh_mean/trees | +11.7 % | [+0.97, +4.40] | 0.0024 | ≥ 10 % | **yes** |
| P3 task | last/ridge | hand/trees | −54.5 % | [−10.98, −4.07] | 0.0002 | within 10 % | **no** |
| P3 task | last/trees | hand/trees | −62.1 % | [−11.65, −5.61] | 0.0002 | within 5 % | **no** |

Beside the declared rows: the two best poolings against the trees at 200 are not distinguishable
from them (tail_20/ridge −5.6 %, [−2.02, +0.51], p 0.23; channel_mean/ridge −5.8 %,
[−2.10, +0.50], p 0.22), while the mean pooling is (−18.9 %, [−4.25, −1.02]). At 50 both beat the
trees' mean by 1.2 cycles, interval across zero. The pretrained states beat the untrained
encoder's in every pairing, under both fitters, at both budgets (+11.7 % to +26 %).

**Conclusions.**

1. **The head is the largest identified loss.** With the backbone frozen and a linear head,
   changing the pooling from the mean over the window to the mean over its last 20 % takes the
   error at 200 labels from 16.57 to 14.73, and the gap to the tuned trees from 18.9 % and
   distinguishable to 5.6 % and not. A per-channel mean does the same (14.76) at twenty times
   the width; the tail keeps the channel layout out of the head and matches it.
2. **The representation carries the task.** Read linearly per channel it lands within the
   trees' interval, and every pooling of the pretrained states beats the untrained encoder's.
   P2's per-channel clause failed under the trees and not under the ridge: boosted trees on
   200 rows of dense states are a poor reader of them (channel_mean: 17.83 under trees against
   14.76 under ridge), so that contrast measured the fitter as much as the states. The
   declared reading of a P2 failure — revisit the pretext — is not taken; the linear reading
   says the states are there.
3. **The task is not a regression of the last reading.** The last values alone lose half the
   trees' accuracy under either fitter (21.6 and 22.6 against 13.9). Remaining life at this
   window is read from how the readings move and spread within it, which is why the mean over a
   single cycle (`tail_2`, 16.19) is barely better than the mean over the whole window and
   the mean over ten cycles is the best: recent, and averaged over enough cycles to lose the
   sensor noise.
4. **What follows.** The arms and the patch model get a pooling knob that keeps the layout out
   of the head — the tail, and learnt attention, which can weight recency and noise on its own —
   confirmed by a fine-tuning campaign against the trees before the curve is repeated; the
   networks' hyperparameter search includes the pooling. A frozen backbone under a tail pooling
   and a ridge is itself a candidate worth a cell: at 50 labels it is the best probe here.
5. **Reproducibility.** The trees repeat the campaign bit for bit through this script's draws,
   so the draws, the features and the knobs are the campaign's own.

**Limitations.**

- Frozen states and linear or tree heads only; nothing here says what fine-tuning does under a
  different pooling. That is the confirmation campaign's question.
- The per-channel ridges chose the largest penalty of the grid (10³) in five of six fits at
  `channel_last`; a wider grid might read them a little better. The layout-free poolings sat in
  the middle of the grid.
- No family correction over the 46 comparisons, as declared; three seeds, tier S fits,
  validation only. The percentile interval over 21 engines runs short of its level
  ([`verdict-statistics.md`](verdict-statistics.md)).
- The window here is 50 cycles at one reading per cycle; the best share of the tail is a fact
  about this corpus and is a knob, not a constant.
