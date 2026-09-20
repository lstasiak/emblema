# ADR-0030: Transfer modes — four arms of one procedure, low-rank updates as a cap on degrees of freedom, and the encoder reached through a seam the process wires

- Status: accepted (first run on the registered backbone in the dated section below)
- Date: 2026-09-18

## Context

The label-efficiency curve compares the same architecture on the same task at the same budget of
labels, pretrained against trained from scratch. The task, its split and the draw of a budget
exist (ADR-0026); the pretrained backbone exists, registered with its weights (ADR-0024); what did
not exist was the procedure that turns a backbone into a candidate on the task and answers its
validation windows — the thing the curve's four methods differ in.

Three constraints shaped it. The preregistration names four methods and fixes them before any
number: from scratch, frozen backbone with a linear head, LoRA, full fine-tuning; the primary
endpoint compares the last against the first. The encoder is an adapter of the Pretraining
context, and a context may import another's published contracts and the shared packages only
(ADR-0005, ADR-0017); ADR-0017 left the mechanism by which Evaluation would reach it to the first
consumer, "with the wiring as the argument". And the model is small: 4.75 million parameters in
the published tier, adapted on at most 2,651 labelled windows.

## Decision

**The four methods are one closed vocabulary, the control arm on it.** `TransferMode` names what
the backbone's weights do while the task is learnt: `FROM_SCRATCH`, `FROZEN_PROBE`, `LORA`,
`FULL_FINE_TUNING`. Every member is the same architecture answering through the same head; they
differ in which weights train and where they start. The control is a member rather than a
separate axis because the curve has one axis of methods, the preregistration lists four, and a
cell of it is one comparison; an `AdaptationPlan` carries the mode together with the artifact the
weights start from, and refuses the two combinations that make no sense — the control arm naming
weights, a transfer mode naming none.

**Low-rank updates are an ablation of regularisation, not an economy.** On a model of this size
full fine-tuning costs minutes, so the argument that carries LoRA in the literature — memory and
time on billion-parameter models — does not apply and is not made. What the mode measures here is
what happens when a budget of fifty labels may spend fewer degrees of freedom: the pretrained
weights stay, a product of two thin matrices trains beside the attention's projections and the
feed-forward network's linears, and the head. The update starts at zero, so the pretrained
behaviour is the starting point rather than a perturbation of it. Rank, scale, dropout and which
layers are capped are a value object (`LoraSpec`) stated with the plan: updating the attention
alone and the feed-forward network too are two caps, and a reader must know which was measured.
A target names a run of consecutive segments of a layer's path — `attention.projection` and not
`projection`, because the time encoding has a projection of its own — and a target that reaches
no layer is refused rather than ignored, before any layer is wrapped, so a refusal leaves the
encoder as it was handed in. The layer is written here in forty lines; a library for it would add
a dependency to carry an argument this record declines to make.

**One head, linear, and fixed epochs.** Every mode answers through a linear head over the mean of
the observed token states, so the modes differ in the backbone alone and the probe asks what the
representation carries as it stands. A run trains for the epochs its schedule states and is
scored after the last: stopping on the validation error would spend the validation side on a
decision per run, and every number reported on it afterwards would be a little too good. Targets
are learnt in units of the task's label ceiling and put back into cycles before they leave. The
schedule states its weight decay rather than inheriting one, because a mode that caps degrees of
freedom is compared against modes that regularise otherwise, and what those did belongs on the
record. It is a schedule and not a budget: in this context a budget counts labels, and the
campaign will count compute under the same word, so the epochs, the batch, the rate and the decay
go under a name that collides with neither.

**The outcome is an answer per validation window, not a summary.** `AdaptationOutcome` carries the
plan, the task, the budget of labels and the seed they were drawn under, and one prediction per
window in block order. The error per unit — what a paired interval resamples — and the error over
all windows are arithmetic over those rows, and so are the metrics the preregistration reports
beside the endpoint. A run stored as its summary would have to be run again for the next
question.

**Two seeds, kept apart.** The seed of the plan fixes the head's weights, a fresh backbone's
weights, the low-rank updates and the order of windows; the seed of the draw fixes which labels.
Two runs over one sample differ in what they learnt and in nothing else. The seed sits on the plan
and not in its schedule, because another seed is a repeat of the same schedule; a row that carries
both names them `run_seed` and `sample_seed`. The head is drawn first under the seed, so every
mode starts it from the same weights — what the encoder draws afterwards is kept by the control
arm and overwritten by pretrained weights elsewhere, and neither can move the head.

**The encoder is reached through a seam the process wires.** Evaluation's torch adapter takes a
`BackboneFactory`: a structural protocol with `pretrained(weights)`, `fresh()` and `width`, its
modules typed as `nn.Module` taking the five tensors of a batch. Evaluation knows the shape of the
call and nothing of the class. The implementation lives with whoever knows both sides — the
process — as `RestoredBackbones` beside the command-line composition roots: it reads the model
Pretraining stored and hands out its encoder, or a fresh one of the same shape over the same
vocabulary. This is option (a) of ADR-0017. The wiring argument for it, at the moment of the first
consumer: the alternative that moves the encoder to the shared packages would take its
architecture value object into the shared kernel with it and touch every module and script that
builds an encoder, in the same weeks in which the pretraining mixture is being built on that very
code; the option that names the seam in Pretraining's published contracts cannot type it without
naming torch, which the contracts may not. The seam costs each process one small class.

**The pooling moves to the shared packages.** `MaskedMeanPooling` is now
`shared/adapters/tensors/masked_mean_pooling.py`. It carries no language of any context — one
state per window out of the states per token — and three things pool the same states the same
way: the export of a backbone, the head a task is answered with, the graph an inference server
runs. A probe that pooled differently from the graph it stands for would measure the difference.
The encoder stays where ADR-0017 put it.

**Reading windows by position is a method of the block.** `WindowBlock.at(positions)` selects the
windows a labelled task addresses, without copying a token; a side of a few thousand windows
materialised as Python objects would cost hundreds of megabytes. Fetching a block once into a
workspace and verifying it before it is mapped is `BlockWorkspace`, which verifies once per
workspace rather than per opening and now serves the Catalog's archive and Pretraining's reader
as well as Evaluation. Evaluation's two readers of blocks share one `PublishedCorpusBlocks`,
which decodes the manifest and opens the block in this context's words, so a run fetches and
verifies its block once rather than once per adapter — and the time a run reports starts only
once the block is at hand.

## Consequences

- Domain, in `evaluation/domain/transfer/`: `TransferMode`, `LoraSpec`, `AdaptationSchedule`,
  `AdaptationPlan`, `WindowPrediction`, `UnitError`, `AdaptationOutcome`. The task labels windows
  and refuses a sample drawn from another task. Port `AdaptationRuntime`, adapters
  `TorchAdaptationRuntime` and `InMemoryAdaptationRuntime` under one contract; use case
  `RunAdaptation` draws a budget, labels the validation side by the task's own scheme, adapts and
  scores, and never asks for the frozen test side.
- The tests the ticket asks for hold: under the frozen probe the backbone's weights keep every
  value bit for bit; under the low-rank mode the wrapped layers do too and the update alone
  receives a gradient; full fine-tuning moves the weights; the control arm never asks for the
  pretrained ones. A run repeats bit for bit on the host under one seed. Every mode trains on a
  published sample of the turbofan corpus through the real use cases and answers every validation
  window in cycles.
- The trainable counts over the published tier at rank 8 are the numbers the ablation turns on:
  the head alone is 257 weights, the low-rank updates some 200 thousand beside it, full
  fine-tuning 4.75 million. They are reported with every run.
- The first runs on the pretrained backbone are made with `scripts/transfer_modes_report.py`,
  which stores a row per run, per epoch and per validation window as CSV and renders the note's
  table from the files; the note is written where the runs happen and dated. The task's engines,
  ceiling, strata and frozen test source are stated in the script's `KnownTasks` until an
  Evaluation process owns them.
- The adaptation runs in single precision; the precision is not yet a parameter of the schedule.
  Encoder dropout during adaptation is whatever the factory built, which is zero for a restored
  model; the low-rank update has a dropout of its own.
- No import-linter contract changed. The seam is a protocol in Evaluation's torch adapters, its
  implementation in the entrypoints, and the shared packages gained a module with no context
  language in it.

### 2026-09-18 — the first run on the registered backbone

Measured on the M1 (MPS, fp32; `docs/verification/transfer-modes.md`): 200 labelled windows of
the turbofan task, one seed, 30 epochs, the report's default learning rate per mode, nothing
tuned. Validation RMSE over the 18 held-out engines (535 windows): from scratch 44.64, frozen
probe 38.92, low-rank 24.07, full fine-tuning 22.30; the mean predictor scores 41.11 on the same
windows. The trainable counts are the ones stated above (257 / 196,865 / 4,752,129). Every mode
trains and answers, which is what the ticket asked; the two arms that update the pretrained
encoder are far under the trivial predictor, the probe barely under it, and the control arm
above it — at this budget and schedule the fresh encoder learnt the mean, and two runs of it
alone at a tenth of the rate, or at three times the epochs, reach 34.9 (the note). The control
arm's schedule is therefore fixed on the validation side before the grid and declared as the
grid's, since the primary endpoint is measured against it.

## Alternatives considered

- **Three transfer modes and a separate axis for where the weights come from.** Would let a
  "frozen random features" arm be expressed. Rejected: the curve has one axis of methods and the
  preregistration lists four; a plan refuses the two combinations that make no sense instead.
- **A library implementation of low-rank updates.** Rejected: the argument the libraries exist for
  is not made here, the layer is forty lines, and the dependency would be larger than the code it
  replaces.
- **Early stopping on the validation error.** Rejected above; the number of epochs is a stated
  parameter of the schedule instead, and the curve reports it.
- **Closed-form least squares for the probe.** Cheaper and deterministic. Rejected for now: one
  loop for every mode keeps the modes comparable in what they were given, and the probe's cost is
  seconds either way; the states are encoded once and the head trained over them.
- **Moving the encoder to the shared packages** (option (c) of ADR-0017). Deferred with the wiring
  argument above; the threshold at which it flips is below.
- **An Open Host Service in Pretraining's contracts** (option (b)). Rejected: the seam cannot be
  typed without torch, which the contracts may not import.
- **A separate head per mode.** Rejected: the modes would then differ in two things.

## Revisit when

- A third consumer of the encoder arrives — Serving, exporting a trained backbone — and the seam
  is implemented a third time. Then option (c) of ADR-0017 is weighed again with three wirings on
  the table.
- The grid needs the adaptation in half precision to fit its budget of accelerator time. Then
  precision joins `AdaptationSchedule` the way it joined the pretraining configuration.
- A second kind of task arrives with a per-unit label. Then a second head, and the head's kind
  becomes a value the task states.
- The curve shows the probe far below fine-tuning at every budget. Then a two-stage arm — the
  probe first, the fine-tuning from it (Kumar et al., 2022) — is the next ablation, as a fifth
  member of the axis.

## Sources

- Hu, E. J. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR.
- Alain, G. and Bengio, Y. (2017). Understanding Intermediate Layers Using Linear Classifier
  Probes. ICLR workshop.
- Kornblith, S., Shlens, J. and Le, Q. V. (2019). Do Better ImageNet Models Transfer Better? CVPR.
- Kumar, A. et al. (2022). Fine-Tuning Can Distort Pretrained Features and Underperform
  Out-of-Distribution. ICLR.
- Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.

### 2026-09-19 — the rate has a shape, and every arm's peak is fixed before the grid

The constant rate of the first run is superseded. `AdaptationSchedule` states, beside the peak,
a warm-up as a share of the run's optimiser steps and the fraction of the peak the rate decays
to, and the runtime steps the rate under `LearningRateSchedule` — the same value object the
pretraining budget uses, moved to the shared kernel because it is a pure function of the step
with no context's language in it. A constant rate is the shape with no warm-up and a floor of
one, so the first run repeats under the new field (44.64 again, to the device's scatter). The
warm-up is a share rather than a count of epochs because an epoch is four steps at the smallest
budget and a hundred and sixty at the largest.

The shape and each arm's peak were fixed on the validation side at the endpoint's budget, one
seed, by a rule stated before the sweep — lowest validation RMSE among three peaks per arm, the
control's shape for every arm — and registered before the grid (`docs/preregistration.md`,
2026-09-19; the numbers in `docs/verification/label-efficiency-curve.md`). Under a warm-up over
a tenth of the run and a cosine decay to one per cent, the control arm goes from 44.64 to 22.38
at a peak of 3e-4, the low-rank arm to 20.70 at 3e-3, full fine-tuning to 21.10 at 3e-4, and
the probe stays at 38.73 at 1e-2. The three arms that step the encoder or an update beside it
therefore stand within two RMSE of one another at 200 labels under one seed, and the arm the
first run showed at the trivial predictor was the schedule's, not the data's, as the note
suspected. The outcome now also states how many labelled windows and how many units the
labels came from, so a budget is reported beside the engines behind it.

### 2026-09-20 — the grid's reading, and which conditions above it meets

The grid ran on two T4 accelerators in single precision, four modes over four budgets under
five seeds, and was read by the registered rules (`docs/verification/label-efficiency-curve.md`).
The endpoint is not confirmed: full fine-tuning at 200 labelled windows takes 21.5 % off the
control's error with its whole interval above zero, and the practical floor — the control's
spread over its seeds, 7.25 — swallows the 7.17. The control arm leaves the plateau of the mean
predictor under two seeds of five under the schedule fixed on one seed above; the pretrained
arm leaves it under four. Where both leave it the difference is six per cent. At a thousand
labels the three arms that step the encoder or an update beside it are indistinguishable, and at
every label the arm trained from scratch is better than both (14.4 against 18.1 and 21.1, the
intervals below zero). One repeat of the low-rank arm at the whole budget rose off its minimum
late in the decay, under a peak chosen over a run thirteen times shorter.

Of the conditions above, two are touched. The probe is far below fine-tuning at every budget
beyond the first (39.5 / 33.0 / 26.5 against 25.8 / 19.8 / 18.1), so the two-stage arm is the
next ablation of this axis when the axis is next extended. Half precision was not needed: the
longer shard took seven hours, above the six set beforehand as the point to reopen it, and
fitted one session; the
registration names single precision, and the decision stands. What the grid adds to the record
is a condition of its own: the schedule of an arm fixed on one seed at one budget does not fix
how often that arm leaves the plateau, the peaks chosen at 200 are a cost at 1,000 and above,
and a run measured in epochs gives the small budgets too few steps to converge — no cell of the
grid had, and the next run states its budget in optimiser steps.
The reading of this as a statement about the encoder waits for the synthetic control's transfer
leg, as the registration orders.
