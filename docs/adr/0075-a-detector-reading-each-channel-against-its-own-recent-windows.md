# ADR-0075: A detector reading each channel against its own recent windows, from the windows it is handed

- Status: accepted
- Date: 2026-11-06

## Context

On both satellite missions, a score with no model and no label read the events about as well as
the best bar of the field, or better. Each channel of an hour is compared with the same channel
over the previous five hours, and the hour is scored by its largest departure: 0.179 against the
cells' farthest feature at 0.160 on the first mission, and 0.885 against 0.627 on the second.
At one normal window in a hundred flagged, it finds as many events as Fisher's direction taught
the events (`telemetry-channel-local.md`). A backbone cannot be held to a host report's
reading; a campaign must be able to fit the detector.

A campaign cell hands a detection runtime two flat lists of windows: the fitting side and the
scored side. Each window carries its unit, its position in the block and when it ends. The
detectors over features score each window's row alone (`FeatureDetector.scores`). This detector
needs the hours before each window as well.

## Decision

- **A new runtime of the existing port.** `ChannelLocalDetectionRuntime` implements
  `DetectionRuntime`. It is routed by the features `channel_departures`, which name a
  `ChannelHistory`: the span in window lengths, whether the window's summary or each of its six
  spans departs, and whether the window takes its largest channel or the mean of its channels.
  It passes the same contract test as the other runtimes.
- **The history comes from the windows handed in, by when they end.** Every window of either
  side is laid on six spans per channel. For each window, the history is the windows that end
  nearest to one, two, and up to `span - 1` window lengths earlier, each within half a length.
  No grid is assumed, so a month published 16 seconds off the hour still finds its place. The
  runtime reads nothing it was not handed, so a history never reaches a frozen month. Units are
  one history when they follow one another on one time axis, as the months of a mission do.
  The runtime refuses units whose windows interleave in time, because their ends then lie on
  separate axes.
- **The arithmetic is the report's.** A channel's summary is the mean of its spans' means, the
  least minimum, the largest maximum, the last reading and the log of the count. The departure
  is the distance from the history's median, divided by 1.4826 times its median absolute
  deviation, and the largest over the five values. The spread is floored at a tenth of the
  channel's spread over the fitting side. A channel with readings in fewer than half of its
  earlier windows departs by nothing. The alarm and the per-unit reference are every detector's
  (`ScoreAlarm`).
- **The fitting side carries no label.** The report dropped the fitting windows that overlap a
  downlink gap, which it learned from the ground truth; the runtime fits on the windows the
  campaign hands it, as every detector does, which moves the floor slightly.
- **Acceptance before any reading relies on it.** On each mission, one campaign cell is checked
  against the report's answers for the arm the report read there (`hour-6h` on the first
  mission, `bins-6h` on the second). The ranks within every month must agree with Spearman's
  coefficient of at least 0.9999, and every window in the report's top 1 % must stay in the
  cell's. The values need not match bit for bit, since the floor differs.

## Alternatives

- **Republish windows that carry their history.** A block of six-hour windows every hour would
  be six times the size of the one-hour block (10.5 GB on the first mission). It would also
  change the unit a task scores. The report was built over one-hour windows for the same
  reasons.
- **Pass when each window ends to `FeatureDetector.scores`.** This would put time into every
  detector over features, only for one of them.
- **Let the runtime read neighbouring windows from the block.** This is a hidden input. A
  window at the edge of a validation month could reach into a frozen one.

## Consequences

- The detector is a bar a backbone must beat on the telemetry tasks.
- Tasks whose units each keep their own time axis, such as the testbed's experiments, are
  refused, not silently mixed.
- The departures are rows of channels per window. A detector taught events could read them
  later without changing this runtime.
- A window's history is limited to the windows the campaign hands in. On the scored side, those
  are only the windows that are scored, so an hour after a downlink gap reads less history than
  it would from the whole block. The report had the same limit.
