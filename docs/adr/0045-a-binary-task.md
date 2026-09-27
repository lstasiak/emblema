# ADR-0045: A binary task — the kind of the target decides the head's link, the draw and what a cell keeps

- Status: proposed (accepted once the first campaign over outcomes has run)
- Date: 2026-09-27

## Context

Every task so far read a quantity: remaining life, or a sensor's reading ahead of the window.
The next one reads an outcome: whether an intensive-care stay ended in death in hospital
(PhysioNet/CinC 2012, about one stay in seven). A quantity and an outcome differ in how a network
should learn them, in how a budget of labels should be drawn and in what a comparison must keep
to read the result. The harness had none of these as a parameter: squared error, a head scaled by
the label ceiling and strata cut by rank of the target were built in.

## Decision

- **`TargetKind` on the label scheme** (`CONTINUOUS`, `BINARY`). A new scheme,
  `OutcomeScheme(outcome)`, reads one named column of a unit's record as zero or one; every
  window of a unit carries its unit's outcome. The kind is a property of the scheme, so every
  adapter learns it from the task it already receives and no port changes.
- **The head's meaning is a link function, as in a generalised linear model.** One linear head
  for every network; `TargetLink` maps the kind to the loss (squared error, or binary
  cross-entropy on the head's output taken as a logit), the bias it starts from (the mean
  label in the scale, or the log-odds of the prevalence) and the answer (multiplied back into the
  task's unit, or the sigmoid). The sigmoid is inside the exported inference graph, so a served
  candidate answers a probability without Serving knowing the task.
- **No class weighting.** The area under the ROC curve does not depend on prevalence, and
  weighting the loss would move the probabilities off the outcomes that the Brier score reads.
- **Budgets over outcomes are drawn in proportion.** `ClassStrata` takes one window at a time
  from whichever outcome lies furthest below its share of the pool, in integers, members ranked
  by the seed. Every prefix holds each outcome within one window of its share, so budgets stay
  nested; a draw that holds one outcome only is refused. Equal shares per outcome were rejected:
  a candidate taught at even odds answers the probabilities of a population that does not exist.
- **Heads solved in closed form are calibrated.** The ridge probe and MiniRocket keep their
  ridge on zero-one targets (its leave-one-out error is the leave-one-out Brier score) and add a
  logistic calibration (Platt) fitted on the leave-one-out answers of the chosen penalty. The
  calibration is monotone, so the ranking is the ridge's. Platt's smoothed targets keep the fit
  finite when the scores separate the outcomes. Gradient-boosted trees grow under
  `binary:logistic`; the stored trees answer the probability themselves.
- **A cell keeps its answers.** Beside the squared error per unit, every new cell keeps each
  window's target and answer (`campaign_window_prediction`, handoff documents alike), because a
  ranking measure does not split into sums per unit. Cells recorded earlier keep none and are
  read by squared error as before.
- **A candidate fitted over several tasks refuses a source of another kind**: a probability and
  a quantity share no scale.

## Consequences

- The first binary task is `physionet2012-in-hospital-death`: stays of sets A and B, whose
  outcomes are read from the challenge's `Outcomes-a.txt` and `Outcomes-b.txt`; set C is the
  frozen side, named from the published listing.
- Every kept form records its link or calibration; forms kept before this read as a quantity in
  its scale.
- Squared error per unit remains the stored error for every task; for an outcome it is the
  Brier score of the unit.

## Alternatives considered

- **A classification head class beside the regression head.** The layer is identical; the loss
  and the transform of the output would still have to be switched in the loop, the closed-form
  probe, the patch model and the export.
- **Refusing the closed-form heads on outcomes.** Would drop the probe that reads the
  representation without an optimiser, and MiniRocket, whose reference form is a ridge
  classifier.
