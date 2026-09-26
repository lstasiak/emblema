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

## 2026-09-26 — M1 Pro, MPS, fp32: the pooling under training, campaign `cb5ed115…`

**Question.** Does the pooling the frozen diagnostics singled out hold once the networks are
trained? The control arm, full fine-tuning and the frozen probe, each under the mean, under the
tail of 20 % and under a learnt attention, paired on the same 21 engines with the tuned trees.
Declared in `campaigns/pooling-fd001.toml` (commit `61ed17e`) before the run. All numbers are
**validation**, tier S.

**Conditions.** Code `61ed17e`; corpus, task, backbone and schedule as in the sections above
(30 epochs, at least 2,000 steps, batch 16, peak 1e-3, warm-up 0.1, cosine to 1 %); 200 labels,
seeds 1–3; the trees at the variant selection `3856e705…` chose. Network cells on the host's MPS
through an order run by `campaign_run` (27 cells, 22:44 to 04:15), the trees through an order of
the general pool on the host (seconds). Attention starts from a zero query, so a run under it
starts where a run under the mean starts.

RMSE on the 21 validation engines, mean ± SD over 3 seeds; the trees per channel 13.94 ± 0.71.

| pooling | from scratch | frozen probe | full fine-tuning |
| --- | --- | --- | --- |
| mean | 18.72 ± 0.28 | 19.75 ± 0.23 | 15.84 ± 0.40 |
| tail 20 % | **14.96 ± 0.49** | 16.83 ± 0.36 | 15.44 ± 0.73 |
| attention | 18.86 ± 0.68 | 19.29 ± 0.23 | 16.24 ± 0.30 |

Seconds per cell: 970–1,150 for the trained arms, 12 for the probe under the mean and the tail,
370 under attention (the encoder runs in the loop).

**The campaign's verdict**, by the registered rules: the endpoint is confirmed. Full fine-tuning
under the tail lowers the control's error by 17.4 % (18.72 → 15.45, interval [+2.07, +4.55],
floor 0.37); five of the eight secondary comparisons are distinguishable after Holm.

**Paired contrasts** beyond the design, the three seeds pooled, 10,000 resamples; reduction of
the rival's error, positive when the candidate is better. No family correction over these.

| candidate | rival | reduction | 95 % interval | p |
| --- | --- | --- | --- | --- |
| from scratch, tail | from scratch, mean | +20.1 % | [+2.14, +5.33] | 0.0002 |
| full fine-tuning, tail | full fine-tuning, mean | +2.5 % | [−0.18, +0.93] | 0.18 |
| frozen probe, tail | frozen probe, mean | +14.8 % | [+1.79, +4.02] | 0.0002 |
| from scratch, attention | from scratch, mean | −0.8 % | [−0.48, +0.23] | 0.40 |
| full fine-tuning, attention | full fine-tuning, mean | −2.5 % | [−0.75, −0.03] | 0.03 |
| frozen probe, attention | frozen probe, mean | +2.4 % | [+0.17, +0.77] | 0.0006 |
| **full fine-tuning, tail** | **from scratch, tail** | **−3.3 %** | **[−1.93, +0.79]** | **0.48** |
| full fine-tuning, mean | from scratch, tail | −5.9 % | [−2.37, +0.43] | 0.21 |
| from scratch, tail | trees per channel | −7.2 % | [−2.38, +0.51] | 0.18 |
| full fine-tuning, tail | trees per channel | −10.8 % | [−3.13, +0.04] | 0.06 |
| full fine-tuning, mean | trees per channel | −13.6 % | [−3.34, −0.32] | 0.02 |

**Reproducibility.** The control under the mean scores 18.72 against 18.86 in campaign
`ee69d456…` and 18.75 in `add35a93…`, within the drift of MPS between identical runs. The trees
repeat to the hundredth.

**Conclusions.**

1. **The tail holds under training, and most for the network trained from nothing.** From
   scratch under the tail lands at 14.96, 20 % below the same network under the mean, with
   every seed under the tail more than three cycles below every seed under the mean, and within
   the trees' interval (−7.2 %, [−2.38, +0.51]). The frozen diagnostics predicted the size
   of this move (16.57 → 14.73 under a ridge) by another method.
2. **Full fine-tuning gains little from the tail** (+2.5 %, interval across zero). A reading
   consistent with the numbers, not a measurement: an encoder trained on the task can route the
   end of the window through its own attention and partly repair the mean; a fresh encoder under
   2,000 steps cannot.
3. **Under the tail the advantage of pretraining at 200 labels is not distinguishable from
   zero here.** Full fine-tuning under the tail against from scratch under the tail: −3.3 %,
   [−1.93, +0.79]. The curve measured +12.3 % for this pair under the mean, at tier M with five
   seeds. This is the absence of evidence at three seeds and one budget, not evidence of absence:
   the interval spans ±9 % and the percentile interval over 21 engines runs short of its level.
   Part of the curve's margin was the mean pooling handicapping the control more than the
   pretrained arm. The repeat of the curve, every arm under the tail, five seeds, four budgets,
   registered before it runs, is where this is settled.
4. **Attention from a zero query does not help under this schedule**: within the noise for the
   control and the probe, 2.5 % worse for fine-tuning. The tail gives for free what the query
   would have to learn in 2,000 steps.
5. **The probe's head, trained by the schedule, is far from the ridge on the same states**: 19.75
   against 16.57 under the mean, 16.83 against 14.73 under the tail. A linear head under AdamW at
   the arms' rate does not reach the closed-form optimum in this budget, so the probe arm has
   understated what the representation carries in every campaign so far.

**Limitations.** Three seeds, one budget, tier S on MPS, validation only. Eleven contrasts beyond
the design without a family correction. The confirmed endpoint of this campaign compares the
tail-headed arm against the mean-headed control and says nothing about pretraining on its own;
conclusion 3 is the comparison that does, and it is underpowered by design.

## 2026-09-26 — declared before the run: the pilot selection of the trained arms' knobs

**Question.** Under the tail, the arm trained from nothing sits a cycle behind the trees and
full fine-tuning level with it, both under one peak rate, no weight decay and a share of the
window chosen off frozen states. How much of that is the knobs, and are the arms short of
steps? Asked on held-out tuning engines, never on the validation side, so that the repeat of the
curve starts from a settled default.

**Design.** Two selection campaigns, declared in `campaigns/selection-networks-fd001.toml` and
`campaigns/selection-networks-budget-fd001.toml` and registered in `docs/preregistration.md`
(2026-09-26). The first turns one knob at a time around the pooling campaign's setting for the
arm from nothing and for full fine-tuning: the tail's share (0.1, 0.2, 0.5), the peak rate (a
third, once, three times 1e-3) and a weight decay (0, 0.01); 12 variants, 200 labels, three
repeats of a division that holds 16 of the 79 tuning engines out, the one-standard-error rule with
the Nadeau–Bengio correction as for the baselines. The second runs the two arms and the probe
under the tail with the floor of steps doubled to 4,000, on the same repeats. Both through orders
run on a rented accelerator and accepted back.

**Predictions.** For the arm from nothing, a share above 0.2 and a rate below the peak are chosen;
for full fine-tuning the setting in force survives the one-standard-error rule. The doubled
budget lowers the arm from nothing by more than the practical floor and full fine-tuning by less.

**Reading.** Whatever is chosen is the default the repeated curve runs the arms under, and its
selection at four budgets searches around it. A budget effect above the floor for the arm from
nothing means the curve's compute budget is revisited before the repeat, as a configuration
change registered in its own right; below the floor the budget stands.
