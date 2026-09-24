# ADR-0026: A task inherits the corpus's split, names its frozen side rather than holding it, and takes labels through a port

- Status: accepted
- Date: 2026-09-16; amended 2026-09-20
- Full text before condensation: commit `5f14447`

## Context

The label-efficiency curve compares a pretrained encoder with the same architecture from scratch
at several label budgets on one task. Before any number is visible we must fix which units tune
and which measure, where labels come from, and what keeps the final test set out of every earlier
decision. The Catalog publishes unlabelled corpora split over units, with statistics from the
training side; the official turbofan test engines are in no corpus. Measured first: of FD001's 100
engines the published split holds out **18** (2,651 tuning / 535 held-out windows); **25.0 %** of
tuning windows sit at the label ceiling; **7** of 100 official test engines are shorter than a
window.

## Decision

- **The task's sides are the corpus's sides**, narrowed to the task's units: tuning = what the
  backbone trained on, validation = what the corpus held out. A fresh 80/20 split would take most
  validation engines from the side the backbone pretrained on and whose values normalised them.
- **The split is stored with the task, never recomputed.**
- **The frozen side is named, not materialised**: unit keys plus the source set name. No corpus
  holds them.
- **Opening the frozen side is a use case that publishes an event**; the aggregate refuses it to
  anything but the final run, so "the test set was read once" has a record behind it.
- **Labels come through an Evaluation port**; the turbofan adapter reads each record's length.
  The Catalog never carries answers.
- **Stratification is part of the task**, declared at definition with the label ceiling.
- **A budget is drawn by rank across strata**: each window ranked under the seed on its own key,
  strata cut by rank (the 25 % tie at the ceiling would pile into one stratum by value), drawn round
  robin. A smaller budget is a subset of a larger one at the same seed.

## Consequences

- Every interval on this task resamples 18 units — the binding constraint, stated before any run.
- A task pins one manifest; two publications are two tasks.
- Evaluation holds its own `UnitKey` and reads published names as text.
- A typo in frozen units surfaces only at the final run.

## Alternatives considered

- *A fresh 80/20 split*: transductive for most validation engines.
- *Three-way split inside the training side*: keeps the leak and wastes untouched units.
- *Publishing test engines now*: work the final run needs; must never enter pretraining.
- *A separate access-log port*: events already carry the fact.

## Revisit when

- A corpus publishes no split, or a held-out side too small to measure on.
- Campaigns bring a run identity → the opening event names the campaign.

## Amendments

- **2026-09-20** — a second task kind (forecasting on the synthetic control) turned
  `UnitLifetimes.failure_times(units)` into `GroundTruth.truths_of(windows)`: one number per window,
  read by the task's label scheme. The turbofan adapter answers every window with the failure
  moment; nothing above moves (ADR-0033).
