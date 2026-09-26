# Findings

What the project has measured so far, in one place. Every number is **preliminary and on the
validation side**: the frozen test side of every task is opened once, at the end. Rules for each
comparison were registered before the run they judge ([preregistration](preregistration.md)).
The evidence for each line is in [`docs/verification/`](verification/README.md).

Last updated 2026-09-26.

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
| Arms | From scratch (control), frozen probe, low-rank updates (LoRA), full fine-tuning; one linear head ([ADR-0030](adr/0030-transfer-modes.md)) |
| Statistics | Reduction in RMSE against the control, 95 % paired bootstrap over engines, Holm over 11 secondary cells, a practical floor ([ADR-0032](adr/0032-statistics-of-a-paired-comparison.md)) |
| Budgets | 50, 200 (endpoint), 1,000 and all 2,568 labelled windows; 5 seeds |

## Results

### 1. The registered endpoint is confirmed

At 200 labelled windows, full fine-tuning lowers validation RMSE by **12.3 %** against training
from scratch (19.02 → 16.68; interval [+1.49, +3.30]; tier M, Colab A100, fp32). Over five fresh
seeds no sweep had seen, the reduction is 12.8 % with almost the same interval
([note](verification/label-efficiency-curve.md)).

| Labelled windows | From scratch | Frozen probe | LoRA | Full fine-tuning |
| --- | --- | --- | --- | --- |
| 50 | 23.49 | 18.63 (+20.6 %) | 18.24 (+22.2 %) | 19.71 (+16.0 %) |
| 200 | 19.01 | 16.32 (+14.2 %) | 16.09 (+15.4 %) | 16.63 (+12.3 %, confirmed) |
| 1,000 | 15.20 | 16.08 (−5.4 %, n.s.) | 14.45 (+5.2 %, n.s.) | 13.87 (+9.0 %) |
| 2,568 (all) | 12.26 | 15.72 (−28.0 %, worse) | 12.80 (n.s.) | 12.88 (n.s.) |

RMSE in cycles, mean over five seeds; percentages are pooled paired reductions.

- The advantage is largest where labels are scarcest, and it disappears once every label is used.
- A frozen probe plateaus near 16: it places an engine's stage of life, not its last cycles.

### 2. Classical baselines beat every network on this task

Tuned by a declared selection on held-out tuning engines ([ADR-0038](adr/0038-the-tuned-baseline.md)),
tier S, M1 Pro ([note](verification/classical-baselines.md)):

| At 200 labelled windows | RMSE |
| --- | --- |
| Gradient-boosted trees, per-channel window statistics | **13.94** |
| Gradient-boosted trees, statistics aggregated across channels | 14.32 |
| MiniRocket + ridge | 15.24 |
| Best pretrained arm (LoRA, from the grid above) | 16.09 |
| Patch model from scratch, PatchTST-style (paired campaign with the trees) | 18.40 |
| Trees over Lomb–Scargle spectra | 18.47 |
| Network from scratch (same campaign as the trees) | 18.86 |

The per-channel trees beat the network from scratch by 26.0 % (interval [+2.86, +6.88],
confirmed). They are about 13 % below the best pretrained arm. That last comparison crosses two
campaigns on the same 21 engines and is **not paired**. At 50 labels the trees across channels
(16.59) and MiniRocket (16.81) also lead the best pretrained arm (18.24). An audit found no
leakage. A second architecture trained from nothing, a patch model in the style of PatchTST
([ADR-0039](adr/0039-the-patch-baseline.md)), ties the network from scratch at 200 labels
(18.40 against 18.75 in the same campaign; interval [−1.22, +1.82]) and trails the trees by the
same margin, so the gap is not specific to the set encoder. **On FD001 the pretrained encoder
has not yet beaten a well-tuned classical method.**

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
  Part of the 12.3 % in section 1 was the mean handicapping the control more than the pretrained
  arm. Three seeds and one budget cannot tell a 5 % advantage from none; the repeat of the curve
  under the tail is where this is settled.
- Attention pooling from a zero query does not help under this schedule; the frozen probe's head,
  trained by the arms' schedule, stays about three cycles above a ridge on the same states.

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
- **Unequal tuning.** Baselines were tuned at every budget; the network arms only at 200. This
  asymmetry can only make the network's claim harder.
- **Unequal tiers.** Baselines ran at tier S on an M1; the network grid at tier M on an A100. The
  network from scratch drifts by up to 1.1 RMSE per seed between identical MPS runs.
- Full fine-tuning's margin varies widely with the draw of labels (2.1–17.3 % per seed).
- **The head of section 1.** The curve was measured with every arm pooling by the mean, which
  section 5 shows handicapped the control most. The confirmed endpoint stands as registered; its
  reading as a pretraining advantage is provisional until the curve is repeated under the tail.

## Next

1. Repeat the curve as one paired campaign with the baselines, every arm pooling by the tail, the
   share and the rate chosen per arm and budget by the same selection protocol, five seeds, the
   configuration registered before the run.
2. Measure transfer across corpora and to unseen sensor layouts.
3. Open the frozen test side once.
