# ADR-0026: A task inherits the corpus's split, names its frozen side rather than holding it, and takes labels through a port

- Status: accepted
- Date: 2026-09-16

## Context

The label-efficiency curve compares a pretrained encoder against the same architecture trained
from scratch, at several budgets of labelled windows, on one supervised task. Before any of that
can be measured, three questions have to be answered in a way that cannot be revisited once the
numbers are visible: which units tune the task and which it is measured on, where the labels come
from, and what keeps the final test set out of every decision taken before the end.

The Catalog publishes corpora without labels, divided by its own split over units, and fits the
statistics that normalise every token on the training side of that split alone. The turbofan
corpus it publishes is read from the run-to-failure half of the source only: the official test
engines, whose ground truth is published beside them, are in no corpus this system holds.

Three facts were measured before deciding, on the corpus itself:

- of the 100 engines of the subset the task uses, the published split holds out **18**; the
  tuning side therefore holds 2,651 windows and the held-out side 535, at the window the corpus
  was published under;
- **25.0 %** of the tuning windows carry the label ceiling (24.1 % over the whole subset), because
  early life is not distinguishable and the convention holds it flat;
- **7** of the 100 official test engines are shorter than one window.

## Decision

**The task's sides are the corpus's sides, narrowed to the units the task covers.** The tuning
side is what the corpus trained on, the validation side is what the corpus held out. A task that
drew its own 80/20 split would take most of its validation engines from the side the backbone was
pretrained on — the model would have read those windows without their labels, and their values
would have contributed to the statistics normalising them at evaluation. Inheriting the corpus's
own held-out units costs two engines of an already small side and removes both objections.

**The split is stored with the task and never recomputed.** Deriving it on demand would move it
whenever the corpus was published again, under another seed, or with another unit — and every
number measured before the move would silently stop being comparable with the ones after it.

**The frozen side is named, not materialised.** It carries its unit keys and the name of the set
they come from, and no corpus in the system holds them. Publishing the official test engines into
the corpus the backbone pretrains on would hand the model the data it is finally judged on; giving
them a corpus of their own is work the final run needs and nothing before it does.

**The way to the frozen side is a use case that publishes an event.** The aggregate refuses to
hand it to anything but the final run, and the use case records every opening. The claim the
project makes is that the test set was read once, and a claim of that shape is worth exactly as
much as the record behind it.

**Labels enter through a port of this context.** The turbofan ground truth is the length of each
run-to-failure record, read by an adapter of Evaluation, never by the Catalog: a corpus that
carried its answers would put them in front of the pretraining, which is the one place they must
not be.

**The stratification is part of the task, like the label ceiling.** How many groups of the target
a budget spreads over changes which windows are drawn, so a count chosen at the moment of asking
is a knob that can be turned once the errors are visible. It is declared when the task is defined
and stored with it.

**A budget is drawn by rank, across strata of the target.** Each window is ranked under the seed
on its own key, strata are cut by rank rather than by value, and the draw goes round the strata
one at a time. Cutting by value would pile the quarter of the windows tied at the ceiling into one
stratum and leave others empty; ranking each window on its own rather than shuffling the pool
keeps the draw independent of the order the windows arrived in, so it repeats elsewhere. A
consequence worth naming: a smaller budget is a subset of a larger one at the same seed, so two
points of the curve differ by their budget and not also by their sample.

## Consequences

- Every interval this task reports is a resampling over 18 units. That is the binding constraint
  of the design, and it is written down before the first run rather than discovered in the width
  of an interval afterwards.
- A task is defined against one manifest and pins it. Two publications of the same data are two
  tasks, which is the same rule corpus versions already follow.
- Evaluation holds its own `UnitKey` and reads published unit names as text. The duplication is
  the price of the contexts staying separable; the alternative is reaching past the published
  language for a convenience.
- The frozen units cannot be checked against anything until they are published, so a typo in them
  surfaces at the final run. The source descriptor is what makes that publication a defined piece
  of work rather than a guess.
- Nothing here persists to a database yet: the repository is a port with an in-memory adapter, as
  the context has no tables until its campaigns need them.

## Alternatives considered

**A fresh 80/20 split owned by the task.** Keeps the split a property of the task rather than of a
publication, and holds the round number the design was first written with. Rejected: it makes the
comparison transductive for most of the validation engines, and the first question a reader asks
would be about the design rather than the result.

**Splitting three ways inside the corpus's training side.** Keeps the official test set for later
and leaves the corpus's held-out units unused. Rejected: it keeps the leak whole and wastes the
only units nothing has touched.

**Publishing the official test engines as a corpus now.** Would let the frozen side be checked and
windowed like any other. Deferred rather than rejected: it is work the final run needs, it must
never enter the pretraining mixture, and doing it now would put units in the system that nothing
is allowed to read for months.

**A separate access log port for openings of the frozen side.** Rejected: events already carry
facts out of this context, and a second mechanism beside them would record the same thing in
another place.

**Hiding the frozen units behind an accessor on the value object.** Rejected: the split checks
them against its other sides and a repository stores them, so the field exists either way; two
ways to the same units would leave the weaker one in place.

## Revisit when

- A task appears whose corpus publishes no split of its own, or whose held-out side is too small
  to measure on — then the task draws its own split from the corpus's held-out side, and this
  record is amended rather than the other way round.
- A second kind of task arrives (a classification with per-unit labels): the lifetimes port is
  shaped for a failure time, and a second shape is the moment to ask whether one port with a
  richer result is better than two.
- Campaigns arrive and bring a run identity: the event would then name the run that opened the
  frozen side, not only the task.

### 2026-09-20 — a second kind of task arrived, and the port became one with a richer result

The transfer leg of the synthetic control poses a forecasting task whose truth varies along a
unit. `UnitLifetimes.failure_times(units)` is now `GroundTruth.truths_of(windows)`: one number
per window, read by the task's label scheme, which the aggregate holds as a closed set of two.
The turbofan adapter answers every window of an engine with the moment it failed, so nothing
this record decided about the remaining-life task moves. ADR-0033 records the choice.
