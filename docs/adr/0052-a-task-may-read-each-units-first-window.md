# ADR-0052: A task may read each unit's first window: the corpus is published for pretraining, the moment of prediction is the task's

- Status: proposed
- Date: 2026-10-06

## Context

The sepsis task over PhysioNet 2019 asks, at the end of a stay's first day, whether sepsis
follows later in the stay. A stay runs from 8 to 336 hours, so one stay holds many windows of
a day, and every window after the first reads hours the question is asked before.

Until now the windows of a task were the windows of its publication. The intensive-care task of
2012 read one window per stay because its stays last 48 hours by construction, and its corpus
was published with one 48-hour window. Publishing 2019 the same way would cut the unlabelled
corpus to the shape of one downstream question: 5.1M of its 10.5M recorded values instead of
8.4M under windows of a day every twelve hours, and a pretraining distribution equal to the
task's input.

## Decision

- **A task states which of a unit's windows it reads**: every window, as before and by default,
  or the first window of each unit. The choice is a closed vocabulary on the task, fixed when the
  task is defined and stored with it, since a choice made per run is a knob that can be turned
  once numbers are visible.
- **Drawing a budget and scoring a run read the same windows**: the task narrows the windows of
  its units before any label is read.
- **The corpus stays independent of the task**: PhysioNet 2019 is published with windows of 24
  hours every 12, and the turbofans keep pretraining on whole trajectories while their task reads
  windows by the position of the label.
- **The first window is checked against the stay, not trusted**: the sepsis ground truth answers
  only a window that ends one window after the stay's first recorded hour, and refuses a window
  in which the label has already turned. A stay whose first day holds no measurement has no
  first window to read, so its earliest published window is a later one; such stays, and every
  other stay the task does not read, are named as ineligible beside the frozen side, by the same
  rule, so that the task's sides hold only the stays it reads.
- **A document of a task carries the choice only when it is not the default**, so every task and
  order written before keeps its bytes.

## Consequences

- One more column on the task and one migration; a task reads one window per unit on a corpus
  published for many.
- The moment of prediction is the end of the first recorded day, not the admission plus a day:
  the reader starts a stay at its first row.
- A second question over the same corpus at another moment needs a corpus published with that
  window length, not a second publication of the data.

## Alternatives considered

- *One window per stay, as 2012*: no code, but a corpus cut to one task, read by every backbone.
- *Two publications of the same stays*, one for pretraining and one for the task: no change to
  the task, but its safety rests on both publications drawing the same sides; a seed or an
  exclusion list that differs would put validation stays into pretraining without a sound.
