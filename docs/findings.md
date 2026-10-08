# Findings

What the project has measured so far, in one place. Every number is **preliminary and on the
validation side**: the frozen test side of every task is opened once, at the end. Rules for each
comparison were registered before the run they judge ([preregistration](preregistration.md)).
The evidence for each line is in [`docs/verification/`](verification/README.md).

Last updated 2026-10-05.

## The claim

A single variable-channel, permutation-invariant transformer, pretrained without labels on sensor
corpora that differ in channel count and sampling regime, lowers the number of labels a new task
needs. The claim is tested against the same architecture trained from scratch and against strong
classical baselines, with paired intervals over independent units.

## Setup

| | |
| --- | --- |
| Task | Remaining useful life on C-MAPSS FD001: 79 tuning engines, 21 validation engines, 100 frozen test engines; the second task, in-hospital death on PhysioNet 2012, is set out in section 6 |
| Corpus | The four C-MAPSS subsets, one channel per sensor and operating condition, scaled within the condition ([ADR-0034](adr/0034-the-turbofan-corpus-read-per-operating-condition.md)); window 50 cycles, stride 5 |
| Backbone | 4.75M-parameter encoder (256 wide, 6 blocks), masked reconstruction, 8 epochs |
| Arms | From scratch (control), frozen probe, frozen probe solved in closed form, low-rank updates (LoRA), full fine-tuning; one linear head over the tail of the window ([ADR-0030](adr/0030-transfer-modes.md), [ADR-0041](adr/0041-the-pooling-of-the-head-as-a-knob.md), [ADR-0044](adr/0044-the-probe-solved-in-closed-form.md)); rate and tail chosen per arm and budget by a declared selection |
| Statistics | Reduction in RMSE against the control, 95 % paired bootstrap over engines, Holm over the secondary cells (35 in the paired campaign), a practical floor ([ADR-0032](adr/0032-statistics-of-a-paired-comparison.md)) |
| Budgets | 50, 200 (endpoint), 1,000 and all 2,568 labelled windows; 5 seeds |

## Results

### 1. Repeated by the harness, the registered endpoint is not confirmed: the control leads

One paired campaign over ten candidates, four budgets and five seeds, every network pooling by
the tail of the window and every trained arm at the rate and tail its selection chose per budget
(tier M, Colab G4 and A100, fp32; [note](verification/label-efficiency-curve.md)). At 200
labelled windows full fine-tuning is **10.5 % worse** than training from scratch (14.29 → 15.77;
interval [−2.56, −0.54]; floor 0.50). No candidate beats the control at any budget under the
family's correction.

| Labelled windows | From scratch | Frozen probe | Probe, closed form | LoRA | Full fine-tuning |
| --- | --- | --- | --- | --- | --- |
| 50 | **16.34** | 16.93 (n.s.) | 17.41 (worse) | 20.19 (worse) | 17.61 (worse) |
| 200 | **14.29** | 14.66 (n.s.) | 15.12 (n.s.) | 14.96 (n.s.) | 15.77 (worse, endpoint) |
| 1,000 | **12.77** | 14.53 (worse) | 14.29 (worse) | 13.97 (n.s.) | 13.40 (n.s.) |
| 2,568 (all) | **11.47** | 14.18 (worse) | 13.61 (worse) | 12.76 (n.s.) | 12.71 (n.s.) |

RMSE in cycles, mean over five seeds; verdicts are the campaign's, paired over engines.

- The first curve (2026-09-22, every arm under the mean, one peak per arm) read +12.3 % for full
  fine-tuning at 200 labels. What that measured was the control's handicap: under the tail and
  at its own rate the control fell from 19.0 to 14.3 at 200 labels, full fine-tuning only from
  16.6 to 15.8.
- The pretrained arms are not under-tuned: each ran at what its own selection chose at that
  budget, and the selections found the pretrained weights hurt by more optimisation, not helped.
- Both probes place the pretrained states within about a cycle of the control at 200 labels and
  no closer. What the encoder learnt from the pretext is close to what a fresh encoder learns
  from 200 labelled windows of this task.

### 2. Classical baselines tie the control and beat every pretrained arm

In the same campaign, at the variants their selections chose per budget
([ADR-0038](adr/0038-the-tuned-baseline.md), [note](verification/classical-baselines.md)):

| Labelled windows | From scratch | Trees per channel | Trees across channels | MiniRocket | Patch model |
| --- | --- | --- | --- | --- | --- |
| 50 | **16.34** | 19.35 (worse) | 16.59 (n.s.) | 17.17 (n.s.) | 22.13 (worse) |
| 200 | 14.29 | **14.08** (n.s.) | 14.24 (n.s.) | 15.30 (n.s.) | 17.67 (worse) |
| 1,000 | 12.77 | **12.04** (n.s.) | 13.09 (n.s.) | 15.40 (worse) | 13.91 (n.s.) |
| 2,568 (all) | **11.47** | 11.76 (n.s.) | 13.32 (n.s.) | 15.42 (worse) | 12.54 (n.s.) |

The trees per channel and the control are indistinguishable from 200 labels up; the earlier
26 % margin of the trees over the network from scratch was the same handicap. Trees over the
spectrum trail everything (18.2 at 200). The patch model, run as published under the mean, is
24 % behind the control at 200 and within reach at 1,000 and above. **On FD001 a well-tuned
transformer trained from scratch is as good as a well-tuned classical method, and the pretrained
encoder adds nothing to either.**

### 3. The pipeline finds structure where it exists

A generated positive control has two sensor layouts over one latent process
([ADR-0018](adr/0018-synthetic-positive-control.md), [note](verification/synthetic-transfer.md)).

- With a window that spans the process's time scales (128 steps), the coupled pair transfers:
  0.094 below a fresh encoder, interval [+0.089, +0.099], floor 0.053.
- At a window of 32 everything fails, including the ceiling. The window was the fault, not the data.
- A backbone pretrained on noise is a worse start than none.
- The "null" pair shares the family of the signals, and that family is most of what transfers. It
  controls leakage, not structure.

### 4. Pretraining over a mixture of corpora learns every corpus

C-MAPSS, SKAB, SMD and ESA satellite telemetry under one chained vocabulary, 8 epochs on a Kaggle
T4 ([ADR-0029](adr/0029-pretraining-over-a-mixture-of-corpora.md),
[note](verification/manual-handoff.md)). Each held-out side is learnt against its channel-mean
predictor: C-MAPSS to 0.7 %, SKAB to 26 %, SMD to 37 %, ESA-AD to 28 %. SMD's held-out loss rises
while the mean falls, so a per-corpus weight is the next parameter.

Two findings shaped that run:

- A squared error let 0.13 % of satellite tokens carry 45 % of the gradient. A Huber loss and a
  split that puts excursions on both sides turned *not learnt* into *data-limited*
  ([ADR-0028](adr/0028-bounding-the-loss-of-an-excursion.md)).
- Scaling C-MAPSS sensors across all operating conditions made the pretext trivially solvable.
  The first grid failed its endpoint on that corpus
  ([ADR-0034](adr/0034-the-turbofan-corpus-read-per-operating-condition.md)).

### 5. The head cost most of the gap, and the pretraining margin at 200 labels shrinks with it

Every network above pooled a window's states by their mean before a linear head. Read on frozen
states ([note](verification/head-and-representation.md)), the mean is the largest identified loss
against the trees: a linear head over the last fifth of the window sits within the trees' interval
(14.73 against 13.94), the same head over the whole window 19 % behind. The last reading alone is
not the task (21.6): remaining life is read from the movement within the window. The pretrained
states beat an untrained encoder's everywhere.

Under training, in one paired campaign at 200 labels, tier S, three seeds:

| pooling | from scratch | frozen probe | full fine-tuning |
| --- | --- | --- | --- |
| mean | 18.72 | 19.75 | 15.84 |
| tail 20 % | **14.96** | 16.83 | 15.44 |
| attention | 18.86 | 19.29 | 16.24 |

- The network trained from nothing under the tail lands 7 % behind the trees, interval across
  zero; under the mean it was 26 % behind.
- **Under the tail, full fine-tuning against from scratch is −3.3 %, interval [−1.93, +0.79].**
  Part of the first curve's 12.3 % was the mean handicapping the control more than the
  pretrained arm. Three seeds and one budget could not tell a 5 % advantage from none; the repeat
  in section 1 settled it: all of it was.
- Attention pooling from a zero query does not help under this schedule; the frozen probe's head,
  trained by the arms' schedule, stays about three cycles above a ridge on the same states.
- A pilot selection on held-out tuning engines (three repeats of 16, never the validation side)
  finds the two arms want different rates: from nothing gains 8.0 % [+0.78, +1.71] at three times
  the peak, full fine-tuning loses 38 % at it and keeps its setting; doubling the steps buys the
  control 2.4 % [+0.04, +0.70]. At each arm's own setting the control leads full fine-tuning by
  8.0 % [+0.47, +2.03] on those engines. One peak for every arm was the curve's second handicap.

### 6. On in-hospital death the endpoint is not confirmed; the task's own backbone helps the probe

Death in hospital after an intensive-care stay (PhysioNet/CinC Challenge 2012), read over the
first 48 hours: learnt from set A's 3,997 stays, scored on set B's 3,994, AUROC with a paired
bootstrap over stays ([ADR-0046](adr/0046-comparing-candidates-by-auroc.md),
[note](verification/intensive-care-curve.md)). Two campaigns, one per backbone, every candidate at
its selection's choice per budget, five seeds; the registered least gain is 0.045 in area.

| Labelled stays | From scratch | Frozen probe | Full fine-tuning | Trees per channel | MiniRocket |
| --- | --- | --- | --- | --- | --- |
| 50 | 0.65 | 0.64 (0.67) | 0.67 (0.71) | 0.66 | 0.71 |
| 200 | 0.68 | 0.66 (0.75) | 0.70 (0.71) | 0.76 | **0.76** |
| 1,000 | 0.78 | 0.76 (0.82) | 0.77 (0.79) | **0.83** | 0.82 |
| 3,997 (all) | 0.81 | 0.79 (0.84) | 0.81 (0.80) | **0.86** | 0.85 |

AUROC, mean over five seeds, under the mixed backbone; in brackets under the backbone of the
stays alone.

- **The endpoint is not confirmed**: under the mixed backbone full fine-tuning gains 0.024 in area
  at 200 stays, interval [+0.012, +0.036], below a floor of 0.040 set by one seed's spread.
- Under the mixed backbone no way of using it beats the network trained from nothing at any
  budget. Under a backbone pretrained on the stays alone the frozen probe gains 0.055, 0.047 and
  0.031 at 200, 1,000 and every stay, above the floor. The two backbones side by side were not given a
  prediction beforehand, so this is a description, not a test.
- From 200 stays up a classical baseline has the highest mean area: MiniRocket at 200, the trees
  per channel at 1,000 and every stay; the stays-alone probe comes within 0.005 of the trees at
  1,000. At 50 stays full fine-tuning under the stays alone and MiniRocket are level.
- The methods that need a grid pay nothing measurable for it: the patch model leads the network
  on raw readings by 0.042 and 0.041 at 50 and 200 stays, MiniRocket by 0.04 to 0.08 at every
  budget. The patch model also differs in shape and dropout, so the grid alone is not isolated;
  the raw readings give the network no advantage at these budgets.
- Two possible handicaps of the networks are open: rates chosen at the edge of their grid, and a
  step floor confirmed on the turbofans only.

### 7. The network's deficit on the stays is in its training, not in its data

A diagnosis before transfer is measured, read on one fifth of set A's tuning stays (800 stays,
held out by one seed, the validation side unread), seeds 1 to 10, every step declared with a
prediction before it ran ([note](verification/intensive-care-curve.md), sections from
2026-10-01; [ADR-0047](adr/0047-a-training-regime-beside-the-schedule.md)). Levels here are on
that fifth and are not those of the table in section 6.

| Network from nothing, every stay | AUROC | Gain over the control |
| --- | --- | --- |
| control: 30 epochs of cosine decay, last weights | 0.783 | |
| kept at its best epoch on a fifth of its labels held out | 0.810 | +0.027 [+0.009, +0.044] |
| 64 wide, 2 blocks (77,000 parameters), under the published network's regime whole | 0.818 | +0.034 [+0.015, +0.054] |
| STraTS, the published network, on the same stays | 0.827 | |

- **The data are not the deficit.** The published network run on this project's tokens reaches
  0.827, on its own preparation 0.822; the tokens are the raw files.
- **Nor are the things tried first.** Static features set apart in the head, pooling under
  attention, dropout, readings on an hourly grid, a nonlinear value embedding and values
  bounded at five standard deviations each gain the network nothing measurable. The published
  network's own lead from its value embedding is a bound on the tails: a linear value with
  clipped readings recovers 0.034 of the 0.039 it loses.
- **The stop is the lever.** Kept at its best epoch on held-out labels the network gains 0.027
  from fewer labels (2,560 against 3,200) and its spread over seeds falls from 0.018 to 0.006;
  the network overfitted through the second half of its schedule and the protocol read the last
  weights. The stop is the registered recipe of every arm at the budget of every stay from
  2026-10-03. No registered verdict changes.
- **Two blocks instead of six close the rest.** Under the published regime whole the small shape
  lands 0.009 from the published network, an interval that includes it. It needs the regime's
  other parts (a class weight, withheld channels, dropout, a constant rate), which cost the
  large shape 0.011.
- **The stop does not scale down.** At 1,000 stays it gains 0.018 and 0.013 by the interval
  only; at 200 it gains nothing in the large shape and loses 0.038 in the small one, because a
  patience of ten epochs is 100 steps there, inside the warm-up. The fixed epochs stand below
  the budget of every stay.
- **The mixture.** Without SMD, 40 % of its steps, the mixture transfers as the mixture does, so
  that corpus teaches the task nothing; the same mixture at thirteen passes, with the better
  pretext loss, loses 0.057 under fine-tuning to its eight-pass twin. A backbone is chosen on
  the task, never by its pretext loss. The rate and the step floor of fine-tuning at 200 stays
  move it by at most 0.012.

### 8. At about 10⁸ values, pretraining does not carry a probe to the network from nothing

The first point of a curve of the gain over the data a backbone is pretrained on
([ADR-0049](adr/0049-the-scale-of-pretraining-as-a-stage.md),
[note](verification/intensive-care-curve.md), 2026-10-05): the mixture of four corpora, about 10⁸
values, read by the probe solved in closed form against the network from nothing, ten seeds, under
a recipe chosen on the tuning side beforehand
([ADR-0050](adr/0050-a-head-solved-first-a-patience-in-steps-and-a-probe-at-initialisation.md)).

| Gain of the probe, in area | 20 stays | 50 stays | 200 stays |
| --- | --- | --- | --- |
| against the network from nothing (two pretraining seeds) | −0.058, −0.071 | −0.022, −0.016 | −0.024, +0.013 |
| against the same probe over an untrained encoder | +0.011, −0.002 | +0.009, +0.013 | +0.012, +0.035 |

- **The probe lies below the network from nothing at 20 stays** under both pretrainings, past
  the spread a draw of labels makes; at 50 stays the gap (−0.022, −0.016) is within it. What
  pretraining adds over an untrained encoder's states at 20 and 50 stays cannot be told from zero
  ([note](verification/intensive-care-curve.md), 2026-10-08).
- **On FD001 the mixture's states are worse than an untrained encoder's** for a linear reading
  of the remaining life, by 3.9 to 5.9 RMSE at 50 windows.
- **Two pretrainings differ by 0.004 at 50 stays**, within a draw of labels (±0.016), so the
  gap between pretraining seeds is not measured. A shape five times larger gains nothing
  here, once its rate is halved.
- Full fine-tuning gains 0.030 and 0.036 over the network from nothing at 200 stays under
  today's shape, and loses 0.021 under the larger one at its rate.

This is the curve's left end; whether the gain grows with the data is read at the next points.
Two variants of the pretext, a forecast tail and masks over three quarters of a window, were
read by the same probe on a fifth of the tuning side ([note](verification/pretext-variants.md),
2026-10-06): neither beats the mixture's masks at 50 stays (−0.017 and −0.015 in area, intervals
crossing zero), so the curve continues under the masks it started with.

## Limitations

- **Validation only.** Every configuration choice (window, normalisation, corpus, backbone, peaks)
  was registered before its run, but each was read on the same 21 validation engines. A
  configuration kept because it did better there scores optimistically there. The single test-set
  run is what removes that bias.
- **21 engines** bound every interval on the turbofan task. On synthetic data with a known answer
  the percentile interval over 21 units covers about 92 % rather than 95 %, and its whole width
  lies above a true zero about 5 % of the time rather than 2.5 %
  (`docs/verification/verdict-statistics.md`). The confirmed endpoint sits far from that
  boundary; a result near it would need a bias-corrected interval, registered before the run.
- **Two supervised tasks so far.** Transfer across corpora (leave-one-corpus-out, zero-shot) is not
  yet measured. The intensive-care task learns from set A only, under half the stays the
  published benchmarks on this corpus learn from, so its levels are not comparable with theirs.
- **Two kinds of accelerator.** The repeated curve ran budgets 50 and 200 on an A100 and 1,000
  and all on a G4, one kind per budget as registered; the arm from nothing was selected on an
  A100 and the other arms on a G4. Between accelerators an arm drifts by up to 0.7 RMSE.
- **One pretraining corpus for the turbofan task.** Its backbone was pretrained on C-MAPSS itself;
  a fresh encoder has the same data to learn the task from. The mixture backbone is read on the
  intensive-care task only, where it is the weaker of the two.
- The patch model ran as published, under the mean, since no selection turned its pooling.
- The pretrained arms' spread over seeds is largest at 50 labels (LoRA 2.2) and at every window
  (full fine-tuning 1.65).

## Next

1. Measure transfer on the intensive-care task again under the stop at every stay, with
   backbones of eight passes over a mixture without SMD, in the control's shape and in two
   blocks on the stays alone; the stop for the pretrained arms is declared with a prediction
   before the first campaign.
2. Weigh the corpora that remain in the mixture, now that one of them is known to teach the
   task nothing.
3. Measure transfer across corpora and to unseen sensor layouts, where a fresh encoder has
   nothing of the target to learn from and the pretrained one has everything else.
4. Open the frozen test side once.
