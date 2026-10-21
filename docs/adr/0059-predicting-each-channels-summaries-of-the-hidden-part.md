# ADR-0059: Predicting each channel's summaries of the hidden part — a term beside masked reconstruction, one unit per channel and window

- Status: proposed
- Date: 2026-10-21

## Context

On the intensive-care stays a logistic head over each channel's summaries reads 0.05 to 0.06 in
area above the closed-form probe, trees 0.07 to 0.09
([`head-and-representation.md`](../verification/head-and-representation.md)). The summaries are
in the token states — the mean of a channel's tokens holds a median 86 % of its level — but as
much at initialisation, and a probe over those states reads the same under the mixture as under
an untrained encoder. A pretext earns its place only by putting into the token states what
initialisation does not.

Masked reconstruction scores each hidden token alone, so a channel weighs by how often it is
measured. An audit with no model
([`pretext-variants.md`](../verification/pretext-variants.md), 2026-10-21) set each channel's
summaries of a window's hidden part against answers fitted without the scored windows: the
visible part, a line through it, a regression on the other channels, the schedule alone, and a
ridge stack of all of these. On the stays, for a channel that keeps visible tokens:

- the stack reads a median R² of 0.67 for the hidden mean and 0.60 to 0.63 for the extremes;
- the schedule alone reads at most 0.08, so the target is no clock;
- nothing reads the hidden part's slope (at most 0.06), the mixture's reconstruction included;
- the mixture's reconstruction, summarised over the same tokens, reads the mean 0.03 and the
  extremes 0.16 to 0.22 below the stack: it smooths the hidden part.

## Decision

- **The target is the hidden part's level and extremes per channel**: for every window and
  channel with an observed hidden token, the mean, minimum and maximum of its hidden normalised
  values. The slope is left out, since nothing reads it; so is the schedule, since on the stays
  the probe reads values and a target the schedule answers is a shortcut.
- **The head reads the decoder.** The decoder's last normalised states at the channel's hidden
  tokens are averaged and mapped linearly to the three summaries. The decoder already places each
  hidden token by channel and time, so a channel hidden whole is answered from the others, and
  the gradient reaches the encoder as the reconstruction's does. It is thrown away with the
  decoder.
- **One unit per channel and window.** A unit costs the mean of its three summaries' losses under
  the run's reading of a miss, however many tokens it hid. The term weighs against one hidden
  token's loss as the contrast did — the units' mean loss times the weight times the hidden
  tokens scored — so a batch divided by its tokens costs the mean reconstruction plus the weight
  times the mean summary loss. Each summary's trivial predictor is zero, the channel's mean.
- **One knob of the experiment file**, `[objective] summary_weight`. Zero renders nothing, so
  every signature, document and artefact written before is unchanged. The term is stated beside
  the value target only and not with the contrast. The artefact carries the weight.
- **The masks stay the experiment file's.** The audit leaves 0.07 to 0.12 more beyond the stack
  under the forecast tail than under the mixture's masks, but under reconstruction alone the tail
  read below the mixture's masks
  ([`pretext-variants.md`](../verification/pretext-variants.md), 2026-10-05), so the pilot reads
  the term under both, by a rule written before it runs.
- **A subclass of masked reconstruction**, the head built after the encoder and the decoder, so a
  run under the term draws the mixture's encoder and decoder from its seed and differs in the
  term alone.

## Consequences

- On SKAB and the satellite telemetry the stack reads these summaries at 0.90 to 0.96, so the term
  teaches mostly from the stays and the turbofans.
- The sparse laboratory values are hidden whole, and nothing reads them from the rest of the
  window (median R² about 0); their units train towards the channel's mean.
- The objective is read by the stage's rules: a tier-S pilot of the weight and of both mask sets
  read as a dose, two pretrainings on the mixture of four, the stage's reader at 500 stays as a
  screen against harm, and the probe over the states pooled per channel against initialisation.

## Alternatives considered

- **Summaries of the visible part**: initialisation already holds them in the token states.
- **The encoder's states of the channel's visible tokens as the head's input**: a channel hidden
  whole has none, and the visible part's summary is the shortcut the audit names.
- **The summaries instead of the reconstruction**: two changes at once.
- **A weight per channel equalising the channels' total cost**: the channels that would gain
  weight are those nothing answers.
