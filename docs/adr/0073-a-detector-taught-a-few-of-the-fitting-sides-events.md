# ADR-0073: A detector taught a few of the fitting side's events, a knob of the candidate

- Status: accepted
- Date: 2026-11-01

## Context

The detection tasks (ADR-0067) fit every detector on the fitting side without a label. Read so,
no distance over the pretrained encoder's states beats the same distance over its initialisation.
The field's lever where a zero-label score stays weak is a handful of labelled anomalies: with
about one in a hundred training anomalies labelled, most semi-supervised detectors beat the best
unsupervised one, and tree ensembles with the same labels match them. An operator of a mission
has that handful. No study of time series lays the grid out as {pretrained, initialisation} ×
{none, a few labels} under one head, so whether pretraining's advantage grows with labels is open.

The campaign's grid spreads candidates over budgets of labelled windows, an axis a detection
task has not got. The first mission's fitting half holds 105 scored events, 41 of them a single
window, and about a tenth of its windows lie in events nobody has marked.

## Decision

- **The count of events a detector is taught is a knob of the candidate**, `events=k`, never a
  budget: the grid stays candidates by seeds, the protocol keeps its one budget, and three counts
  are three candidates. The task still guards the labels: it accepts taught windows from its
  tuning side alone, and a task that spends a budget teaches none. The events are drawn by the
  seeded rank of each under the run's seed, so the draws nest: the events taught at three are
  among those taught at five. Every event is taught whole, anomalies and rare nominal events
  alike, as the task scores them. The scored months follow the fitting months, so the split by
  time and by event holds by construction; the rest of the fitting side keeps whatever events it
  holds unmarked, as the field's semi-supervised setting does.
- **Three heads replace the distance**, chosen by `head=`, each weighing every taught event one
  over its windows, as the register's measure weighs it. `direction`, the default: Fisher's
  direction under the shrunk covariance of the fitting rows outside the taught windows, the
  whitened coordinate along the line from their mean to the taught events' mean, signed. The
  fitting rows spread by about one along it; the estimate carries noise of about the square
  root of features over taught windows, small for the states and large for the cells.
  `combined`: the mean of that coordinate and the squared whitened distance, each over its
  fitting median and spread, so an event unlike the taught ones still raises the distance; a
  product candidate, read descriptively. `trees`: scikit-learn's gradient-boosted trees over the
  same standardised features, the taught rows against the rest and weighing as much, grown by
  one rule stated in the code under the cell's seed and kept as the arrays of their nodes, so
  the document outlives the library. XGBoost would crash beside torch in the one process that
  fits detectors on the host, and the histogram trees have no document form.
- **The same alarm, reference and refusals.** The alarm is set over every fitting row's own
  score; a detector taught events names no distance and no neighbours, a head is named after
  the events, and a detector scored by an error learns from none.

## Consequences

- The fitting side is label-free by default, not by construction: a taught detector says so in
  its name and its record.
- Teaching replaces the scoring rule rather than adding to it: the squared whitened distance is
  the log-density under one Gaussian, the direction the log-likelihood ratio of two, so at three
  events a heterogeneous mission may score below none; that is the reading, not a defect.
- One seed draws both the events and the untrained encoder's initialisation, so the paired
  difference between encoders mixes the two draws; comparisons across counts pair by seed.
- A screen that reads taught detectors declares: the interaction between encoders and counts
  paired by seed; the trees with the same labels as the product bar at each count; the scored
  events split by whether their class was taught; the recall at one in a hundred normal windows
  flagged; the trees under the fitting-side reference, since their log-odds saturate within a
  month; the cells as descriptive; and its own threshold and power, since the registered least
  gain was set for zero-label detectors.
- Considered and not taken: a budget axis in events and a task per count, which multiply what
  one grid reads; a logistic head, which adds a penalty to choose where the shrunk covariance
  already is the ridge; a sum of the zero-label and taught scores, which has no weight to stand
  on and is read as the combination instead.
