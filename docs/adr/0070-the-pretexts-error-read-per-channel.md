# ADR-0070: The pretext's error read per channel, over each channel's typical miss

- Status: accepted
- Date: 2026-11-01

## Context

A detector scored by the pretext's error (ADR-0068) reads a window as one mean of the backbone's
loss over its hidden cells, in the corpus's normalised units. On the satellite telemetry it did
not pass a ridge over the same cells, and on the first mission's subtle events, 87 % of them
without a reading beyond ten deviations, it read near chance like every other detector
([composition-screen](../verification/composition-screen.md)).

The pretext does not miss every channel alike. A channel with heavy tails is missed widely in
every window, normal or not; a nearly constant one is answered to a hair. One mean over the cells
follows the channels missed widely, and a quiet channel's departure, however many deviations of
its own, barely moves it. Detectors built for spacecraft telemetry read each channel's error
against that channel's own history for this reason, and the one published pretrained forecaster
run on this telemetry scales each channel by its own training statistics and warns that a
channel whose variation is physically negligible otherwise inflates the error.

## Decision

- **How the errors are read is a knob of every detector scored by an error**, `error`, with two
  values: `raw`, one mean over the window's scored cells as before, and `channel`, the mean over
  the window's scored channels of each channel's mean error over that channel's scale. The
  default stays `raw`, so every campaign declared so far reads as it was stored, and a stored
  campaign names the reading only where it is per channel.
- **A channel's scale is the median of its per-window error over the fitting windows that scored
  it.** The scales are floored at a tenth of their own median: below that the division amplifies
  rounding rather than departures. Where that median is itself nought, as a linear answer's can
  be on channels held constant, the floor is the smallest scale above nought. A channel no
  fitting window scored is unknown rather than quiet: it takes the scales' median, so a window
  that holds it is read neither up nor down for it.
- **The same cells, draws, alarm and reference.** The errors are summed per channel as they are
  scored, so the reading costs no further pass through the backbone; the alarm is set over the
  fitting windows' errors read the same way, and a window without a scored cell scores the
  fitting side's median as before. The ridge and the interpolation read their errors the same
  way, so the bars move with the detector.
- **What a fit keeps** gains the channels' scales.

## Consequences

- The screen of the reformulated telemetry leg reads `pretext_error@error=channel` beside the
  raw reading, with `linear_answer_error` and `interpolation_error` under the same knob. The
  choice was made on the two backbones the screen reads, so whatever it chooses is confirmed on
  the fresh pretrainings of the clinical confirmation, under a rule written before they run.
- The floor is a parameter of the detector, set before any reading and named here; a different
  floor is another decision.
- A window's score under `channel` no longer tells how large its misses were in normalised
  units, only how unusual they were for their channels; the two readings are not on one scale.
- A reading by rank within each channel's error distribution would stand up to heavier tails than
  the median does; it waits until the median has been read.
