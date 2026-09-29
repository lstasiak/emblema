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

## 2026-09-26 — Kaggle T4 and Colab L4, fp32: the pilot selection of the trained arms' knobs

**Question.** The one declared in the section above: which share of the tail, which rate and
whether a weight decay each trained arm wants at 200 labels, and whether the arms are short of
steps. Read on held-out tuning engines only; nothing here touches the validation side.

**Conditions.** Code `588e56f`; campaigns `a887a115…` (`campaigns/selection-networks-fd001.toml`,
36 cells) and `596c68bb…` (`campaigns/selection-networks-budget-fd001.toml`, 9 cells, declared and
run under `EMBLEMA_WORKER__SCHEDULE__MIN_STEPS=4000`), tier M. Every repeat holds 16 of the 79
tuning engines out (one in five, seeds 1–3); both campaigns divide them identically, checked cell
by cell. Orders run by `campaign_run`: 13 cells of the knobs on a Colab L4 (about 4.7 min per
trained cell), the rest and the whole budget campaign on a Kaggle T4 (about 12.5 min per trained
cell at the 2,000-step floor, 25 min at 4,000, 12 s for the probe), accepted back into the local
registry. RMSE on the held-out engines, mean ± SD over the three repeats; the reduction is the
paired bootstrap over the 48 (repeat, engine) pairs against the arm's setting the knobs were turned
around, 10,000 resamples, no family correction.

**Knobs, one at a time around the tail of 20 %, peak 1e-3, no weight decay.**

| knob | from scratch | reduction vs base | full fine-tuning | reduction vs base |
| --- | --- | --- | --- | --- |
| base (tail 0.2, rate 1e-3) | 15.89 ± 0.87 | — | 15.91 ± 0.63 | — |
| tail 0.1 | 16.90 ± 0.10 | −6.3 % [−1.65, −0.35] | 15.61 ± 0.87 | +1.9 % [−0.23, +0.87] |
| tail 0.5 | 15.07 ± 0.55 | +5.3 % [−0.11, +1.78] | 16.46 ± 0.60 | −3.4 % [−1.09, −0.01] |
| rate 3.3e-4 | 17.47 ± 0.73 | −9.8 % [−2.14, −0.96] | 15.68 ± 0.42 | +1.5 % [−0.47, +0.97] |
| rate 3e-3 | **14.64 ± 0.14** | **+8.0 % [+0.78, +1.71]** | 21.92 ± 1.22 | −37.8 % [−7.02, −4.98] |
| weight decay 0.01 | 15.87 ± 0.57 | 0.0 % [−0.28, +0.30] | 16.09 ± 0.44 | −1.1 % [−0.68, +0.34] |

**What the rule chose** (`campaign select`, one standard error with the Nadeau–Bengio correction,
ties broken towards the setting the knobs were turned around): for the arm from nothing
`from_scratch@learning_rate=0.003,pooling=tail,tail_share=0.2` — the only variant within one
standard error of the best, so no tie was broken; for full fine-tuning its base — the best variant
(tail 0.1) admits the base, the lower rate and the weight decay within its reach, and the base is
nearest. Reading the rule needed a fix: the campaign took the setting a selection turns around to
be the candidate under its bare name, which a selection declared around a variant does not hold;
it is now the one variant the others depart from by the fewest knobs in all, fixed before this
section was read. The choice is the same under either reading.

**Budget: the floor of steps doubled to 4,000, on the same repeats.**

| arm | 2,000 steps | 4,000 steps | reduction | 95 % interval |
| --- | --- | --- | --- | --- |
| from scratch | 15.89 ± 0.87 | 15.50 ± 0.93 | +2.4 % | [+0.04, +0.70] |
| full fine-tuning | 15.91 ± 0.63 | 15.91 ± 1.20 | 0.0 % | [−0.49, +0.48] |
| frozen probe | — | 16.05 ± 0.33 | — | — |

The practical floor at this error is 0.32; the doubled budget lowers the arm from nothing by
0.38, an interval reaching down to 0.04. The rate alone lowers it by 1.26 at the same 2,000
steps, and the arm at 3e-3 and 2,000 steps beats the arm at 1e-3 and 4,000 steps by 5.7 %
[+0.39, +1.36].

**The arms against each other on the tuning engines.** At the shared setting they are equal
(full fine-tuning against from scratch 0.0 %, [−0.85, +0.89]). At what the rule chose for each,
the arm from nothing beats full fine-tuning by 8.0 % [+0.47, +2.03], p = 0.003, and beats full
fine-tuning's best variant (tail 0.1) by 6.2 % [+0.05, +1.85].

**Against the predictions.** A longer tail for the arm from nothing: the direction holds (0.5
gains 5.3 %, interval touching zero) but the rule did not choose it, since the rate alone leaves
nothing else within reach. A lower rate for it: refuted — it wants three times the peak. Full
fine-tuning's setting surviving the rule: confirmed. The budget lowering the arm from nothing by
more than the floor: at the floor, not above it with any confidence; full fine-tuning by less:
confirmed, by nothing at all.

**Conclusions.**

- The arm trained from nothing was short of rate, not of steps: three times the peak buys 8 %,
  twice the steps 2.4 %. Its chosen rate is the largest the grid held, so the repeat's selection
  must reach further (1e-2).
- The two arms want different rates. At 3e-3 full fine-tuning loses 38 %: the pretrained weights
  are overwritten before the head has learnt. One peak for every arm, as the first curve ran,
  handicapped the control and not the pretrained arm; with the mean pooling that is the second
  handicap the curve's margin was read under.
- Once each arm runs at its own setting, the control is 6–8 % ahead of the pretrained arm at 200
  labels on the tuning engines, distinguishable at three repeats. The validation-side answer
  belongs to the repeat of the curve; what this pilot says is that a margin for pretraining at
  200 labels, if there is one, is not there to be found by tuning the pretrained arm's schedule.
- Weight decay of 0.01 changes nothing for either arm. The control is hurt by a shorter tail and
  helped by a longer one; full fine-tuning is indifferent to a shorter tail and hurt by a longer.
- The declared reading said a budget effect above the floor revisits the curve's compute budget.
  The effect sits at the floor, at twice the cost, and is a third of what the rate buys; the floor
  of 2,000 steps is kept for the repeat, and a rate chosen per budget is where the steps' worth
  is read next.

**Limitations.** Three repeats of 16 engines; the intervals above are as wide as the effects
they bracket for every knob but the rate. One budget, 200 labels; the repeat selects per budget.
The rate's grid ended where the choice landed. The probe under the tail was run at 4,000 steps
only, so it is not compared here.

## 2026-09-27 — declared before the run: the floor of steps at the rate the pilot chose

**Question.** The pilot doubled the floor of steps at the old rate and found the arm from
nothing gains 2.4 %, at the practical floor, while the rate alone gains 8 %. Whether the arm
still wants more steps once it runs at 3e-3, and whether full fine-tuning wants any at 1e-3, is
asked before the curve is repeated, since the floor every cell of the curve spends is settled
here.

**Design.** Two selection campaigns over the same three arms — the arm from nothing at the rate
the pilot chose (`from_scratch@learning_rate=0.003,pooling=tail,tail_share=0.2`), full
fine-tuning at its setting (`full_fine_tuning@pooling=tail,tail_share=0.2`) and the probe under
the tail as the family's member — at 200 labels, the tuning engines divided as the pilot
divided them (one in five held out, seeds 1–3), never on the validation side:
`campaigns/selection-networks-floor-fd001.toml` under the floor in force, 2,000 steps, and
`campaigns/selection-networks-floor-doubled-fd001.toml` under `EMBLEMA_WORKER__SCHEDULE__MIN_STEPS=4000`.
Both on one kind of accelerator, an NVIDIA A100, and read cell by cell on the same repeats by
`scripts/campaign_pairs_report.py`: the paired bootstrap over the 48 (repeat, engine) pairs,
10,000 resamples, the practical floor as the larger of 2 % of the control's error and its spread
over the repeats. The pilot's cells at 2,000 steps are not reused: they ran on a T4 and an L4,
and between accelerators an arm drifts by as much as the effect looked for.

**Predictions.** At 3e-3 the doubled floor lowers the arm from nothing by less than the practical
floor, since the rate bought what the steps were buying; full fine-tuning is lowered by less than
the floor, as it was in the pilot. The probe is unchanged.

**Reading.** A reduction above the practical floor for either trained arm, with an interval
above zero, makes 4,000 steps the floor of the repeated curve, registered as a configuration
change before any of its selections run; otherwise the floor of 2,000 stands, with this as its
evidence. Nothing here chooses a rate: the repeat's selections do that per budget.

## 2026-09-27 — Colab A100, fp32: the floor of steps at the rate the pilot chose

**Question.** The one declared above: at the rates the pilot chose, does either trained arm want
twice the floor of steps? Read on held-out tuning engines only.

**Conditions.** Code `d3c98928`; campaigns `fed1a76e…` (`campaigns/selection-networks-floor-fd001.toml`,
2,000 steps) and `1995ccb3…` (`campaigns/selection-networks-floor-doubled-fd001.toml`, declared
and run under `EMBLEMA_WORKER__SCHEDULE__MIN_STEPS=4000`), tier M, both on one NVIDIA A100
through orders run by `campaign_run` and accepted back (a trained cell 2 min at 2,000 steps,
4 min at 4,000; the probe 3–5 s). Every repeat holds 16 of the 79 tuning engines out (one in
five, seeds 1–3), both campaigns dividing them identically. Read by
`scripts/campaign_pairs_report.py`: RMSE on the held-out engines, mean ± SD over the three
repeats; the reduction is the paired bootstrap over the 48 (repeat, engine) pairs, 10,000
resamples; the practical floor is the larger of 2 % of the control's error and its spread. The
probe ran at the worker's one rate, 1e-3, not at its registered 3e-2: it is the family's member
here and settles nothing.

| arm | 2,000 steps | 4,000 steps | reduction | 95 % interval | p | floor |
| --- | --- | --- | --- | --- | --- | --- |
| from scratch at 3e-3 | 14.50 ± 0.04 | 14.14 ± 0.41 | +2.6 % (0.37) | [−0.04, +0.78] | 0.076 | 0.29 |
| full fine-tuning at 1e-3 | 15.92 ± 0.29 | 17.49 ± 3.35 | −10.7 % (−1.70) | [−2.81, −0.63] | 0.001 | 0.32 |
| frozen probe at 1e-3 | 17.23 ± 0.48 | 16.05 ± 0.33 | +6.9 % (1.20) | [+0.85, +1.56] | < 0.001 | 0.48 |

Full fine-tuning at 4,000 steps scored 15.75, 15.38 and 21.35 over the three repeats: one
repeat left the pretrained weights behind, as the arm did under three times the rate in the
pilot.

**The arms against each other on the tuning engines.** At 2,000 steps the arm from nothing beats
full fine-tuning by 9.7 % [+0.64, +2.16] and the probe by 19.0 % [+1.72, +3.73]; at 4,000 steps
it beats full fine-tuning by 24.7 %.

**Against the predictions.** The arm from nothing: the doubled floor lowers it by 0.37, above the
floor of 0.29 as a point but with an interval reaching zero, as the pilot found at 1e-3 (0.38,
[+0.04, +0.70]); the prediction of a gain below the floor held as a reading, not as a point.
Full fine-tuning: not lowered at all but raised, by more than the floor; the prediction held in
the direction that matters. The probe: refuted, it gains, but at a rate a thirtieth of its own.

**Conclusions.**

- The floor of 2,000 optimiser steps stands for the repeated curve: neither trained arm is
  lowered by the doubled floor with an interval above zero, and full fine-tuning is hurt by it.
  No configuration row is needed.
- Twice the steps costs the pretrained arm what three times the rate cost it in the pilot: more
  optimisation of the pretrained weights is what overwrites them. The curve's selection of a
  rate per budget is where this is read next, and where 1,000 labels and more, which lengthen
  every run past the floor, will show whether the arm survives them.
- The trained probe wants more steps at 1e-3, which is not its rate; the curve runs it at its
  registered peak, beside the probe solved in closed form (ADR-0044), which takes no step.

**Limitations.** Three repeats of 16 engines, one budget, one accelerator. The rate of each arm
is the pilot's choice at 200 labels; whether the floor binds at other budgets is not asked here,
and at 1,000 labels and above the schedule's epochs exceed it anyway.

## 2026-09-27 — declared before the run: the selection of each trained arm at every budget

**Question.** Which rate and which tail each trained arm runs at, at each of the four budgets of
the repeated curve, chosen on held-out tuning engines and never on the validation side.

**Design.** Three selection campaigns, one per arm so that one accelerator runs each whole:
`campaigns/selection-scratch-fd001.toml`, `campaigns/selection-fine-tuning-fd001.toml` and
`campaigns/selection-lora-fd001.toml`, each four variants — the setting in force, the rate a
third and three times it, and the other share of the tail — at 50, 200, 1,000 and every
labelled window, three repeats holding 16 of the 79 tuning engines out, under the floor of
2,000 steps the section above keeps. The arm from nothing turns around 3e-3 and the tail of
20 % (rates 1e-3 and 1e-2, tail 0.5); full fine-tuning around 1e-3 (3.3e-4 and 3e-3, tail 0.1);
the low-rank arm around its registered peak of 1e-4 under the tail (3.3e-5 and 3e-4, tail
0.5). The one-standard-error rule with the Nadeau–Bengio correction chooses per budget, ties
towards the setting in force. The arm from nothing on an A100, the other two on a Kaggle T4
each; the curve's own cells run on one kind of accelerator per budget, so a variant chosen on
one and run on another is a choice, not a comparison.

**Predictions.** For the arm from nothing the chosen rate does not fall with the budget: 3e-3
or 1e-2 at every budget, the tail of 0.5 chosen at 50 where the labels are fewest. For full
fine-tuning the smaller rate, 3.3e-4, is chosen at 1,000 and at all, where the run is longest
and the section above says more optimisation overwrites the pretrained weights; at 50 and 200
the setting in force survives. For the low-rank arm the setting in force survives at every
budget.

**Reading.** Whatever is chosen is what the curve runs each arm at, per budget, named in the
curve's campaign file by the selection that chose it. A rate chosen at the edge of its grid at
a budget is followed by one step beyond it at that budget before the curve runs, as the
registered edge rule says.

## 2026-09-27 — Colab A100 and G4, fp32: what each trained arm's selection chose per budget

**Question.** The one declared above: which rate and which tail each trained arm runs at, at
each budget of the repeated curve. Held-out tuning engines only.

**Conditions.** Code `a0a8970c`; campaigns `7a842dba…` (`campaigns/selection-scratch-fd001.toml`,
an A100, 2 min a cell at 2,000 steps and 4 min at every window), `199fb850…` and `83ecaced…`
(`campaigns/selection-fine-tuning-fd001.toml` and `campaigns/selection-lora-fd001.toml`, two
processes sharing one G4, 1:40–2 min a cell); 48 cells each, tier M, floor 2,000 steps, three
repeats holding 16 of 79 tuning engines out. RMSE on the held-out engines, mean ± SD over the
repeats; **bold** = chosen by the one-standard-error rule with the Nadeau–Bengio correction,
ties towards the setting in force.

**The arm from nothing**, around 3e-3 and the tail of 20 %.

| budget | 1e-3 | 3e-3 (in force) | 1e-2 | 3e-3, tail 0.5 |
| --- | --- | --- | --- | --- |
| 50 | 16.95 ± 0.10 | **16.07 ± 0.35** | 30.14 ± 3.21 | 21.46 ± 2.59 |
| 200 | 15.90 ± 0.73 | **14.69 ± 0.77** | 21.37 ± 1.77 | 14.28 ± 0.56 |
| 1,000 | 13.41 ± 1.09 | 13.66 ± 1.07 | 20.61 ± 1.99 | **12.94 ± 0.50** |
| all | 13.06 ± 0.44 | 13.25 ± 0.96 | 20.16 ± 0.84 | **12.52 ± 0.57** |

**Full fine-tuning**, around 1e-3 and the tail of 20 %.

| budget | 3.3e-4 | 1e-3 (in force) | 3e-3 | 1e-3, tail 0.1 |
| --- | --- | --- | --- | --- |
| 50 | **17.20 ± 0.62** | 18.24 ± 1.16 | 24.12 ± 0.35 | 17.70 ± 0.51 |
| 200 | 15.61 ± 0.17 | 16.08 ± 0.40 | 21.10 ± 2.98 | **15.54 ± 0.63** |
| 1,000 | 14.30 ± 1.05 | **13.79 ± 0.84** | 20.40 ± 1.56 | 13.82 ± 0.41 |
| all | 13.56 ± 0.45 | **13.32 ± 0.85** | 19.50 ± 1.32 | 16.73 ± 4.46 |

**The low-rank arm**, around 1e-4 and the tail of 20 %.

| budget | 3.3e-5 | 1e-4 (in force) | 3e-4 | 1e-4, tail 0.5 |
| --- | --- | --- | --- | --- |
| 50 | **18.35 ± 0.39** | 19.09 ± 0.82 | 19.74 ± 1.34 | 19.45 ± 1.34 |
| 200 | 16.60 ± 0.36 | **15.38 ± 0.67** | 16.82 ± 1.18 | 15.55 ± 0.55 |
| 1,000 | 16.62 ± 0.24 | **14.25 ± 0.50** | 13.85 ± 1.43 | 14.54 ± 0.72 |
| all | 14.64 ± 0.42 | **13.38 ± 1.25** | 13.29 ± 1.33 | 13.55 ± 1.31 |

**Against the predictions.** The arm from nothing keeps 3e-3 at every budget: held, and 1e-2
is no edge but a cliff, so no rate beyond it is asked for. Its longer tail at 50: refuted, the
tail of 0.5 costs a third of the error there and is chosen at 1,000 and at all instead, where
the labels reach further into the window. Full fine-tuning's smaller rate at 1,000 and at all:
refuted, the setting in force survives there; its setting surviving at 50 and 200: refuted the
other way, a third of the peak at 50 and the shorter tail at 200. The low-rank arm keeping its
setting: held at three budgets, refuted at 50, where a third of the peak is chosen.

**Conclusions.**

- Three times the peak, 3e-3, is what the arm from nothing wants at every budget, and a decade
  above it the arm does not learn. The tail's share is the knob that moves with the budget.
- The pretrained arms want less rate where labels are fewest: at 50 both choose the smallest
  rate their grids held, the edge of the grid, which the registered edge rule follows with one
  rate beyond it before the curve runs (declared below).
- At 1,000 and at all every arm keeps or nearly keeps its setting in force; the pretrained
  arms' selections are within one standard error of several variants there.

**Limitations.** Three repeats of 16 engines; every interval overlaps its neighbours at 1,000
and above. The arm from nothing ran on an A100, the other two on a G4: a selection compares
variants within one accelerator, and the curve's cells of a budget run on one kind.

## 2026-09-27 — declared before the run: the two rates beyond the edge, at 50 labels

**Question.** Full fine-tuning and the low-rank arm each chose the smallest rate of their grids
at 50 labels. Does a rate half a decade lower do better still, by the same rule?

**Design.** `campaigns/selection-fine-tuning-edge-fd001.toml` and
`campaigns/selection-lora-edge-fd001.toml`: at 50 labels, the same three repeats of the same
division, one knob at a time around the chosen setting — the rate a third of it (1e-4 and
1e-5) and the other share of the tail. Read by the same rule; the choice replaces the one above
at 50 labels and is what the curve runs the arm at there.

**Predictions.** Neither arm gains from the lower rate by more than one standard error, so the
chosen rates stand: 3.3e-4 for full fine-tuning, 3.3e-5 for the low-rank arm. If a lower rate is
chosen again, the rule allows one more step, and no more.

## 2026-09-27 — Colab G4, fp32: the rates beyond the edge, at 50 labels

**Conditions.** Code `98be0ab9`; campaigns `9be0af77…` and `f1b12cb1…`, nine cells each, two
processes on one G4, the same three repeats of the same division as above.

| arm | rate chosen above | a third of it | chosen, other tail |
| --- | --- | --- | --- |
| full fine-tuning | **3.3e-4: 17.33 ± 0.45** | 1e-4: 18.25 ± 1.71 | tail 0.1: 17.50 ± 0.16 |
| low-rank | **3.3e-5: 18.35 ± 0.39** | 1e-5: 23.37 ± 2.18 | tail 0.5: 19.09 ± 0.38 |

The prediction held: neither arm gains from the lower rate, the chosen rates stand and no
further step is asked for. The low-rank arm's cell at 3.3e-5 reproduced the selection above to
the second decimal on each repeat, as a run under one seed should.

## 2026-09-29 — declared before the run: the closed-form probe's penalty over outcomes on the intensive-care task

**Question.** Over outcomes the probe solved in closed form fits an L2-penalised logistic
regression on the frozen backbone's 256 pooled states, its penalty chosen by the log-loss of five
folds within the drawn labels among nine strengths from 0.001 to 100,000, a decade apart — the
list the regression task uses. MiniRocket's fit on the same task passed its list's strongest
step (`classical-baselines.md`, 2026-09-28). Does the probe's list reach far enough at every
budget, under both backbones of the task and both poolings a selection would choose between, and
does every fit converge?

**Conditions.** Commit of this section; MacBook Pro M1 Pro, the encoder on MPS in fp32, the head
in double precision on the host. Task `physionet2012-in-hospital-death` over manifest
`cef44de2…`; backbones `backbone-physionet2012-m-32` (weights `c5878c03…`) and
`backbone-mixed5-m` (`869ed545…`); the mean over the window and the tail of a fifth; budgets 50,
200, 1,000 and every stay of the tuning side, each under seeds 1, 2 and 3, drawn by the task's
own draw as a campaign cell draws them. The candidate is built and its stays embedded by the
campaign's code, sixteen at a time; the head is fitted once over a longer list — one decade below
the list in force and three above, 0.0001 to 100,000,000 — as the candidate fits it, and once per
strength alone on the same folds, so a strength that does not converge is named. Beside each draw
goes the folds' log-loss of answering with the prevalence. No window of the validation side is
read and nothing is scored. Before this was written the embedding was timed on one draw of seed
9, 200 stays; no head was fitted.

    M=cef44de241af45ebb9f99da55679445a72632ada9f8b982dc9651e8554e51a78
    P=c5878c0399789c43a2363170105c7b879a6beb639d67d933915d4da0a4262598
    X=869ed54532a47bca801de6f11049020be786e7550376f65120beee7a9c93ed55
    uv run --env-file .env.r2 scripts/probe_penalty_report.py \
        --manifest durable/sha256/$M sha256:$M \
        --backbone physionet-32 durable/sha256/$P sha256:$P \
        --backbone mixed5 durable/sha256/$X sha256:$X --out data/report/t42c/probe

**Prediction.**

1. At 200, 1,000 and every stay, every fit under both backbones and both poolings chooses a
   strength strictly inside the list in force, between 0.01 and 10,000.
2. At 50 stays, under each backbone and pooling, at least two seeds of three choose the strongest
   strength offered, where the folds' log-loss does not fall below the prevalence's.
3. Every fit converges at every strength of the longer list.
4. Embedding takes under 10 ms a stay and a fit over every stay under a minute; the whole run
   under half an hour.

**Reading, declared beforehand.**

- If any fit at 200, 1,000 or every stay chooses 100,000 or stronger, or 0.001 or weaker, the
  probe on outcomes gets a list of its own: the list in force extended by decades up to one
  beyond the most extreme choice, registered as configuration before any selection on the task
  runs. The regression task keeps its list. One list serves both poolings.
- A choice at 50 stays at either end of the longer list is read against the prevalence: where
  the folds' loss does not fall below it, no longer list would change the choice, and the list
  stands on that account; where it does, the choice is read by the first point.
- A fit that does not converge at a strength of the list in force is a defect of the candidate,
  fixed and tested before any selection runs, and this check is run again. A failure at a
  strength of the longer list alone is reported and changes nothing unless that strength joins
  the list.
- The probe's pooling on this task is not chosen here; that is a selection's to make.
