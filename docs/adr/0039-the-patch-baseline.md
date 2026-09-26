# ADR-0039: The patch baseline — a network from nothing on the grid, behind a port of its own, on the arms' budget

- Status: accepted
- Date: 2026-09-25

## Context

The comparison owes a modern patch model trained from scratch (in the style of PatchTST; Nie et
al., 2023). The control arm measures what pretraining adds to one architecture; a patch model
measures what a different, widely used architecture does with the same labels.

A patch model reads a regular series. The grid it needs already exists for MiniRocket
(ADR-0038). What remained open was where this candidate lives among the existing runtimes, which
budget it is held to, and how PatchTST's defaults fit a remaining-life task.

## Decision

- **A port of its own, `PatchRuntime`.** It is neither of the existing two. `AdaptationRuntime`
  describes a run by a transfer mode over named weights, and a patch model has neither: a third
  mode would be a plan whose backbone and low-rank fields are ignored. `ClassicalRuntime` fits by
  a method's own procedure and shares no compute budget, while a patch model is a network trained
  by the same loop as the arms. The run is described by a `PatchPlan` (shape, schedule, seed) and
  answered with the plain `ScoredOutcome`, unchanged. A third kind of candidate fitting the
  shared outcome as it stands is the check that the outcome's shape was right.
- **The arms' compute budget and training loop.** The candidate is `NEURAL`, runs on the `ml`
  pool, and learns under the one `AdaptationSchedule` the arms learn under. `ComputeBudget.of(schedule)`
  derives the budget in one place, so two networks are held to equal budgets by construction. The
  loop (AdamW, the schedule's rate shape, the seeded window order, the refusal of a non-finite
  loss) moved into `ScheduledTraining`, used by both runtimes; the arms' results are bit-for-bit
  unchanged.
- **PatchTST's reading, with the grid's mask and without RevIN.** Each channel's row is padded at
  the end by one stride and cut into overlapping patches. A token is a patch of values beside the
  same patch of its mask, because the grid carries readings forward and the model must be able to
  tell a reading from a copy. One encoder is shared by all channels. Each channel's tokens are
  averaged, the channels are laid side by side, and a linear head reads the answer. PatchTST
  normalises every window by its own mean and spread (RevIN). Here the level of a reading is the
  state of degradation the task asks about, and the values are already on the corpus's scale, so
  there is no per-window normalisation.
- **Rows as the grid picks them.** The channels read are those the training windows observe,
  chosen by the rule MiniRocket uses (now `RegularGrid.rows_read`). A turbofan task holds 21 of
  the corpus's 126 channels, and a head as wide as the vocabulary would be mostly constant input.
- **Shape from the worker's settings.** `EMBLEMA_WORKER__PATCH__*` configures the shape, as the
  boosting and convolution knobs are configured, and the campaign records it in the candidate's
  method, so a worker set otherwise is refused the cell. The template's values are the small
  tier's encoder shape (width 192, 3 heads, 4 layers): 1.79 million weights over a turbofan
  window, which a test holds under 2 million.
- **The artifact is a torch document; no knob variants yet.** The kept model is its plan, its
  grid reading, its target scale and its state, read back with `weights_only`. Export to ONNX
  belongs with the export of every network artifact, not with this candidate. How the knobs of a
  network are tuned is decided once for every network, so a variant name is refused until then.

## Consequences

- A campaign can compare two architectures trained from nothing under one budget, paired on the
  same units, beside the pretrained arms and the classical baselines.
- On an irregular corpus the patch model and MiniRocket pay the cost of the grid, and the set
  encoder does not. The difference is a measurement the comparison can now make.
- The channels form an axis of the input, so a patch model does not transfer across layouts.
  That is expected of a baseline and the reason the transfer baseline exists.
- The per-channel mean discards where in the window a pattern sits. It was chosen over
  flattening all patches into the head, which would make the head grow with the window length.
  The pooling is a candidate knob once networks are tuned.

## Alternatives considered

- **A third transfer mode "from scratch, patch architecture".** Rejected: the plan would carry
  fields the mode must ignore, and the backbone factory would have to build a model that is not a
  backbone.
- **Shape from `compute_tiers.toml`.** Rejected: the profile describes the set encoder, and a
  catalogue does not know a campaign's tier.

## Amendments

- **2026-09-25 — the knobs the decision left open.** The shape's fields are turned by name like
  the schedule's knobs, and the pooling of each channel's patches is a knob shared with the arms
  ([ADR-0041](0041-the-pooling-of-the-head-as-a-knob.md)). Neither touches the budget. Status
  moves to accepted.
