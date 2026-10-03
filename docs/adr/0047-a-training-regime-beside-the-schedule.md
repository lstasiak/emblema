# ADR-0047: A training regime beside the schedule — a stop on held-out labels, a class weight and withheld channels, as declared knobs

- Status: accepted (2026-10-03; the stop alone gained the network from nothing 0.027 in area at every stay, the other knobs cost 0.011 beside it: `docs/verification/intensive-care-curve.md`, 2026-10-03)
- Date: 2026-10-03

## Context

Every network of a campaign learns a task under one schedule: a fixed number of epochs on a
floor of steps, the last weights kept (ADR-0030, ADR-0035). The rule has two reasons. The arms
spend one compute budget, so a comparison between them measures their weights and not where
each happened to stop. And the validation side is never read while a run trains, so what is
reported on it is not selected on it.

Published networks on the same task train otherwise: they hold out part of their labelled data,
stop when it stops improving, weight the rarer outcome and withhold whole variables while they
learn. Turned towards this project's regime one part at a time, such a network lost at most 0.011
to any one part; turned towards it in all of them, 0.043 (`docs/verification/intensive-care-curve.md`,
2026-10-02). This project's network tried the parts it had one at a time and gained nothing. The
parts together have never been read in it, and nothing else measured there separates the two
networks any more.

## Decision

- **A `TrainingRegime` sits beside the `AdaptationSchedule` on a plan and an arm.** The schedule
  says how long a run may take and at what rate; the regime says what else is done inside that
  budget. Its knobs are `stop_share`, `patience`, `class_weight` and `channel_dropout`; the
  standard regime is what every campaign ran under before, and a description names only knobs
  turned away from it, so stored descriptions stay valid.
- **The stop reads labels the run was given, never the validation side.** Whole units are held
  out of the labelled sample, ranked by a digest of the run's seed and the unit, and the run
  learns from the rest. After each epoch the held-out units are scored, by the sum of the areas
  under the ROC and precision-recall curves for an outcome and the negative squared error for a
  quantity; the weights of the best epoch are kept and the run gives up after `patience` epochs
  without a better one.
- **The schedule's epochs become a cap.** A stopped run is held to the whole sample's count of
  epochs and spends at most the budget it was declared under; the steps it took are read off
  its outcome, as the floor's already are.
- **The class weight is the ratio of negatives to positives among the labels learnt from**, and
  only an outcome has one. **Withholding channels** marks a window's channel as padding for one
  step, static features excepted, and never empties a window; it is refused where the encoder
  states every window once, as the dropout is.
- **The standard regime stays the default where the stop was not read.** At the budget of every
  stay the stop (a fifth held out, patience 10) is the registered recipe of every arm of the
  backbone from 2026-10-03 on (`docs/preregistration.md`, that date); at smaller budgets, where
  a held-out fifth is a handful of units, the standard regime stands until a stop is read there.
  The other three knobs stay knobs a selection may turn.

## Consequences

- A cell's budget is no longer exactly what it spends: under a stop it spends less, and the
  learning side is a fifth smaller. Both are stated in the run's record, and the comparison
  against the standard regime is a comparison of two recipes on one cap, which is what the
  question asks.
- At small label budgets a held-out fifth is a handful of units and the stop reads noise; the
  regime is read at every stay, and nothing here says it transfers to 50 or 200 stays.
- The stop's score for an outcome is the published network's, not this project's measure; the
  campaign still reads the validation side by the registered measure.
- The optimiser steps of a stopped run are counted off the epochs it ran over the learning
  side's batches, so a run's cost in steps is read, not planned.

## Alternatives considered

- **A stop on the validation side.** Rejected: it selects on what is reported.
- **Choosing the epochs by a nested selection instead of a stop.** Keeps the budget exact, but a
  selection reads one setting for every seed where a stop reads each run; it is the fallback if
  the stop gains and the budget's exactness is wanted back.
- **Transplanting the regime whole as one knob.** Simpler to declare, but it hides which part
  carries a gain; four knobs let the stop be read alone first.
