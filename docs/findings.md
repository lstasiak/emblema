# Findings

What the project has measured so far, in one place. Every number is **preliminary and on the
validation side**: the frozen test side of every task is opened once, at the end. Rules for each
comparison were registered before the run they judge ([preregistration](preregistration.md)).
The evidence for each line is in [`docs/verification/`](verification/README.md).

Last updated 2026-09-27.

## The claim

A single variable-channel, permutation-invariant transformer, pretrained without labels on sensor
corpora that differ in channel count and sampling regime, lowers the number of labels a new task
needs. The claim is tested against the same architecture trained from scratch and against strong
classical baselines, with paired intervals over independent units.

## Setup

| | |
| --- | --- |
| Task | Remaining useful life on C-MAPSS FD001: 79 tuning engines, 21 validation engines, 100 frozen test engines |
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
- **One supervised task so far.** Transfer across corpora (leave-one-corpus-out, zero-shot) is not
  yet measured.
- **Two kinds of accelerator.** The repeated curve ran budgets 50 and 200 on an A100 and 1,000
  and all on a G4, one kind per budget as registered; the arm from nothing was selected on an
  A100 and the other arms on a G4. Between accelerators an arm drifts by up to 0.7 RMSE.
- **One pretraining corpus for this task.** The backbone was pretrained on C-MAPSS itself; a
  fresh encoder has the same data to learn the task from. The mixture backbone and transfer
  across corpora are not yet read on a task.
- The patch model ran as published, under the mean, since no selection turned its pooling.
- The pretrained arms' spread over seeds is largest at 50 labels (LoRA 2.2) and at every window
  (full fine-tuning 1.65).

## Next

1. Measure transfer across corpora and to unseen sensor layouts, where a fresh encoder has
   nothing of the target to learn from and the pretrained one has everything else.
2. Read the mixture backbone on a task, and a per-corpus weight in the mixture.
3. Open the frozen test side once.
