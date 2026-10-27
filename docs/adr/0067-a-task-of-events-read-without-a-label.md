# ADR-0067: A task of events, fitted without a label and read by the events it sets apart

- Status: proposed
- Date: 2026-10-27

## Context

The protocol for anomaly detection was declared before the first run: a detector fitted on
normal data and scored on the rest, with no budget of labels. Until now the Evaluation context
could store such a task and run nothing over it: every campaign spreads candidates over budgets
of labels, every runtime takes a sample of labels, and the measures read a quantity or an
outcome.

Three corpora carry annotated events: the raw ESA satellite telemetry (a second reader reads
each mission's whole span, every reading at its own instant), the SKAB testbed and the OPS-SAT
fragments. Their events run from a point to forty days, and the benchmarks score them with
threshold-dependent, event-wise metrics.

## Decision

- **A third kind of target.** A detection task reads each window by the event it overlaps:
  zero for none, the event's number otherwise, and a mark for a window the task does not score
  (a gap in the downlink). The number is the target, so every window of one event shares it and
  the stored answers say which event a score was given for. The task carries no stratification
  and draws no budget.
- **Normal data is a side of the split, not a filter by label.** The detector fits on every
  window of the task's tuning side; no label reaches the fit. Which units form that side is the
  publication's: the first half of each mission's months, the authors' training fragments, the
  anomaly-free recording. The benchmarks fit the same way, with the contamination they publish.
- **One cell per detector.** A campaign over such a task runs at the one budget of every
  window; a grid of several budgets is refused, since the axis does not exist. A selection run
  is refused too: a detector's settings are declared with the campaign, since choosing them on
  labels would spend what the protocol says it has not got.
- **The measure is the event-weighted area under the precision-recall curve.** Windows are
  swept from the highest score down; every event weighs one, spread evenly over its windows,
  and every normal window weighs one. A month-long event and an hour-long one count the same,
  a false alarm costs the same at any hour, and nothing is maximised over an event's windows,
  so a long event gains nothing from the number of draws it offers. The comparison is paired
  by unit and resampled by unit, as every other measure is.
- **The detector is the same under every arm.** A window is read as the pooled states of a
  frozen encoder, pretrained or drawn at its initialisation, or as the summaries of its
  channels; the score is the mean distance to the nearest fitted windows, every feature
  standardised over the fitting side; the alarm is the quantile of the fitting windows' own
  scores that flags a declared share of them. A run answers the distance above the alarm, so
  the alarm is set without a label and the ranking is unchanged by it.
- **Two readings stand beside the area**, both with the alarm's semantics: the share of events
  found with at most one per cent of the normal windows flagged, and the benchmark's corrected
  event-wise F0.5 at each detector's alarm, a run of flagged windows touching no event being
  one false alarm and the precision scaled by the share of normal windows left quiet.

## Consequences

- A fragment shorter than a window yields one window of the nominal length (a policy of the
  window specification, stored in the manifest), so the fragments' durations stay readable.
- The benchmark's timing and range metrics are not computed: at a window's resolution they
  would measure the window, not the detector. Published figures are compared with, never
  claimed as reproduced: the channels, the resolution and the level of detection differ.
- A detector is a candidate of its own kind: it shares no compute budget, needs the stack that
  runs an encoder, and nothing serves it yet. The register restates the protocol in these terms.

## Alternatives considered

- *PR-AUC with every window weighing one*: three events of the satellite's test half span
  1,650 of about 1,750 positive windows, so the measure would read those three alone.
- *Recall over events by the maximum score of their windows*: the benchmark's semantics, but a
  long event's maximum over hundreds of draws beats a point event's single draw under any
  detector, chance included.
- *Fitting on windows the labels mark as normal*: spends labels at the fit and is not what the
  benchmarks do.
- *An adaptation plan as the detector's description*: carries a schedule and penalties the
  detector never uses.
