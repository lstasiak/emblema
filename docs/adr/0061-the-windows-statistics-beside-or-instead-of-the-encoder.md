# ADR-0061: The window's statistics per channel beside or instead of the encoder — a diagnostic of the network trained from nothing

- Status: accepted (2026-10-22)
- Date: 2026-10-22

## Context

On the intensive-care stays the network trained from nothing lies 0.04 to 0.08 in area below
boosted trees over the window's statistics per channel (count, mean, deviation, extremes, first,
last, slope, step, gap). A logistic head over the same statistics lies within 0.006 of the
trees at 200 stays and 0.018 below them at 500
(`docs/verification/head-and-representation.md`, 2026-10-23). The gap therefore has at least two
parts. One is what a network makes of the statistics with few labels against a tree. The other is
what the encoder fails to build from the readings that the statistics already hold, which a new
encoder is meant to close. Reading them apart needs a network that is handed the statistics, run
under the same regime, stop, draws and registry as the network it is compared with.

## Decision

- **`summaries`, a knob of the encoder setting**: `none` by default, `beside` or `only`. Under
  `beside` the head reads the pooled states and a state made from the statistics, side by side.
  Under `only` it reads the statistics' state alone; the encoder is built but never run, and none
  of its weights train.
- **The statistics are the trees' own table.** The runtime reads them from the raw readings with
  the reader the trees use. A small branch keeps the columns the drawn windows define with at
  least two distinct values, bounds each to the range the drawn windows span, standardises it by
  the mean and spread of the values they define, puts a missing statistic at that mean, and
  passes them through one hidden layer as wide as the encoder. Moments of the values defined, not
  of the filled column, and the bound keep a scored window holding a rarely measured channel from
  landing thousands of deviations out; beyond the drawn range a tree is constant too. A linear map
  would only repeat the logistic head already measured. The columns, bounds and moments are
  buffers, kept with the weights.
- **Only the network trained from nothing takes it**, with a head it trains, over raw readings
  and every channel. A backbone, a head solved in closed form, a grid and withheld channels are
  refused, by the plan and by the catalogue for the same reason. Under `only`, every knob that
  only shapes the encoder (dropout, own build, clip, layer, pooling, encoder rate) is refused,
  since it would change nothing.
- **A diagnostic, never a candidate.** The statistics hand the head what an encoder is meant to
  learn, so a result under this knob says how far an encoder falls short. It does not say how
  good a network is, and no declaration may carry a claim on it.
- **Kept in its measured form alone.** The inference graph takes raw tokens; a candidate reading
  the statistics would answer them as something other than what was scored.
- A plan that reads no statistics draws its head and encoder exactly as before. A candidate's
  description names the knob only where it is turned, so stored descriptions stay what they
  were; the flattened plan a kept candidate records carries it always, as every knob.

## Consequences

- Three differences can be read on the same stays and seeds. The trees against `only` give the
  cost of a network over the trees' table. `only` against the plain network gives what the
  encoder fails to build of it. `beside` against `only` gives what the encoder adds to it. The
  first is not a bound on every architecture: an encoder may read more than ten statistics hold.
- The branch is as wide as the corpus has channels times ten, so it is bound to one corpus's
  layout. That is acceptable for a diagnostic and is the reason it can never become a candidate:
  a backbone must not depend on the count of channels.

## Alternatives

- **The statistics as extra tokens.** They would enter the mean over the window, which loses
  such summaries (`head-and-representation.md`, 2026-10-21), and they would change the encoder's
  input. Which tokens the encoder reads is the question this diagnostic is meant to inform.
- **The statistics computed in torch from the tokens.** That would be a second implementation of
  the trees' table, and the two could drift apart.
- **A linear branch.** It repeats the logistic head over the statistics.
- **A knob of the head's pooling.** The pooling is shared with the patch model, and `only` reads
  no pooled state at all.
