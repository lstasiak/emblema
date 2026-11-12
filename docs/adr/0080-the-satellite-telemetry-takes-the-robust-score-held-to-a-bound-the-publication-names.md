# ADR-0080: The satellite telemetry takes the robust score held to a bound of 100, and a publication names the bound of its robust score

- Status: accepted
- Date: 2026-11-12
- Extends ADR-0079; replaces its consequence that the telemetry keeps the standard score, and its fixed bound of 10

## Context

ADR-0079 kept the standard score for the satellite telemetry on the ground that its detectors read
deviations in those units, and bounded every robust score at 10 robust deviations. An audit of the raw
ESA telemetry's fitting side ([`telemetry-input.md`](../verification/telemetry-input.md),
2026-11-12) found otherwise.

The fitting side's standard deviation is set by the benchmark's own labelled events in the
channels that matter: on Mission1 the events hold 44 to 94 % of the variance of ten of the 48
channels while holding 0.3 to 1.1 % of their readings, and on Mission2 three channels have a
standard deviation 534, 22 and 9 times their robust deviation, nearly all of it from the labelled
events, almost all of them rare events. Under the standard score the ordinary range of those
channels spans 0.02 to 0.7 of a deviation, so a network with linear embeddings reads them as
nearly constant. The telemetry is about half of the windows the base recipe (the stays of
PhysioNet 2012 and 2019, the eICU demo and the raw ESA telemetry) is pretrained on, and the
backbones of the clinical confirmation are the ones the project keeps, so their input is chosen
before they are pretrained.

The product's detector reads each channel's departure from its own recent windows over a robust
spread floored at a share of the fitting side's. Its value summaries are invariant to any affine
map of a channel's values, which is what two normalisations differ by without a bound (to 10⁻¹⁰
absolute on the recorded run). Only a bound, and the floor of its readings-count summary
(Consequences), reach it; the 10 of ADR-0079 would flatten a quarter of the event readings of
Mission1's channels 41 to 46.

## Decision

- **A publication names the bound of its robust score** (`--robust-bound`, in robust
  deviations), kept with the channel's statistics in the manifest. A manifest that names none is
  held to 10, so every robust manifest published before keeps its bytes; one held to another
  bound is written at a third version of the manifest format, which an older reader refuses
  rather than clipping it wrongly.
- **The raw ESA telemetry is published under the robust score held to 100.** On the fitting side
  six of 50 million normal readings lie beyond 50 robust deviations: one on Mission1's channel 74
  (at 208) and, on Mission2, one on channel 18 (51), three on channel 19 (up to 124) and one on
  channel 20 (520); every other normal reading lies within 21. A bound of 100 flattens 0 to 7 % of
  Mission1's event readings and 6 to 31 % of Mission2's three heavy-tailed channels', all far
  beyond any alarm; the encoder has been pretrained on standard scores reaching 226, so inputs
  up to 100 ask nothing new of it.
- The intensive-care corpora keep the bound of 10 of ADR-0079; nothing of their publications
  changes.

## Consequences

- The chain of publications the base recipe is pretrained on holds with the same channel names
  (PhysioNet 2012 → PhysioNet 2019 → eICU demo → ESA), so no earlier publication is republished
  for it; the telemetry tasks and the bars they are read against are defined again over the new
  publication.
- The channel-local detector's ranking changes with the publication only through the floor of
  its readings-count summary, which borrows the typical spread of the value summaries in the
  publication's units: under the exact map between the two scores the two arms the verdict reads
  keep 98 and 99.5 % of their top-1 % windows. That floor is a flaw of the instrument, queued as
  its own change.
- The screen of two pretrainings on the new clinical input also checks the new telemetry input's
  mechanics: the loss falls, no value is non-finite, Mission2's channels 18 to 20 reconstruct no
  worse than before.

## Alternatives considered

- *The standard score, as ADR-0079 had it*: changes no verdict so far, but bakes three unreadable
  channels into every backbone handed on.
- *The robust score without a bound*: rare events of Mission2's channel 18 reach 9,800 robust
  deviations; a bound of 100 keeps every normal reading but five and the order of the readings
  up to it.

## Revisit when

- The screen of two pretrainings shows the telemetry input harming the pretext's mechanics →
  the bound is lowered, or the telemetry returns to the standard score.
- The readings-count floor is fixed at its source → the bars are chosen again on the new
  publication under the fixed detector.
