# ADR-0044: The probe solved in closed form — a fifth arm whose head is a ridge over the pooled states, chosen leave-one-out, on the arms' budget

- Status: proposed
- Date: 2026-09-27

## Context

The frozen probe (ADR-0030) trains a linear head over the pooled states of the frozen encoder
under the arms' schedule. Read on the same states, that head sits several cycles behind a ridge
solved in closed form: 19.75 against 16.57 RMSE on the frozen backbone and 16.83 against 14.73
under the tail
([`head-and-representation.md`](../verification/head-and-representation.md)). The probe is the
arm that says what the representation carries as it stands, and a probe that understates it by
three cycles reports the optimiser, not the representation. The curve is about to be repeated
under the tail (ADR-0041), and a repeat that keeps only the trained probe would publish that
understatement as the representation's worth.

The convolution baseline already answers through a ridge whose penalty is chosen by
leave-one-out error among a declared grid (ADR-0038).

## Decision

- **A fifth `TransferMode`, `FROZEN_RIDGE`**: the pretrained weights held fixed, the linear head
  solved in closed form over the pooled states. It is a mode of the same axis, not a candidate of
  another kind: the same `AdaptedBackbone`, encoder, pooling and head, with the head's weights
  written by the solution rather than trained. What is kept, exported to the inference graph
  (ADR-0040) and served is therefore unchanged.
- **The penalty is chosen leave-one-out among a declared grid**, `RidgePenalties`, named in the
  plan exactly when the mode solves its head; the grid is the worker's setting
  (`EMBLEMA_WORKER__PROBE__RIDGE_PENALTIES`), its own rather than the convolution baseline's,
  since one is read over ten thousand features and the other over the encoder's width. The
  choice is the fit's, as it is for the baseline (ADR-0038): not a knob of the selection
  protocol.
- **Solved in the torch adapter, not through a library**: columns scaled by their spread, an
  unpenalised intercept by centring, one singular value decomposition, the leave-one-out
  residuals of a linear smoother read off the leverage for every penalty. Held to the reference
  regressor to working precision by a test. On the host in double precision, moved there before
  it is widened.
- **The arm carries the schedule and takes no step.** A campaign holds every network to one
  compute budget by construction, so the arm is declared under the schedule like every other and
  reports no training loss and zero optimiser steps. A learnt pooling is refused under it: a
  closed form has nothing to train a query with.
- **The trained probe stays.** The two probes are the same states under two heads, and the gap
  between them is a fact about the schedule that the curve reports.

## Consequences

- The repeated curve runs five arms; the old grid scripts run the four that learn under a
  schedule and never this one, and their stored columns are unchanged.
- A kept `frozen_ridge` candidate is a `torch-state` and an inference graph like any arm's; the
  penalties travel in the plan's parameters.
- A run costs one pass of the encoder over the sample and the validation windows: seconds, so
  the probe rides in every campaign at no cost worth naming.
- Fewer than two labelled windows leave nothing to leave out and are refused.

## Alternatives considered

- **A candidate of its own kind with its own runtime.** Would duplicate the embedding, the
  keeping, the export and the serving of a network for a head that differs by how its weights are
  found.
- **Reusing the convolution baseline's grid.** Binds at its upper edge over thousands of columns;
  the probe's columns are the encoder's width and want a grid of their own.
- **Dropping the trained probe.** Loses the reading of how much the schedule costs a linear head.
