# ADR-0034: The turbofan corpus read per operating condition — a channel per sensor and condition, scaled within it, and the default reading left as it was

- Status: accepted
- Date: 2026-09-22
- Full text before condensation: commit `5f14447`

## Context

The C-MAPSS reader reads 21 sensors as channels, one scale per channel over the training units.
FD002 and FD004 fly six flight conditions, and the condition explains a median 100 % (never less
than 88 %) of a sensor's variance. Under one scale per sensor over all four subsets, FD001 spans
0.5 % of the scale for the median sensor; masked reconstruction is solved by reading the condition
off other channels (loss 0.7 % of the trivial predictor's), and pretrained arms lost the turbofan
task to training from scratch (`docs/verification/label-efficiency-curve.md`). FD001 and FD003
alone (one condition each) restored both — full fine-tuning 10.5 % below from scratch — but drop
509 of 709 engines (71 % of rows). The settings of every row lie within 0.009 kft and 0.002 Mach of
one of the six conditions.

## Decision

- **Read per operating condition, a sensor is a channel per condition**, named
  `<sensor>@<altitude>kft-M<Mach>-TRA<throttle>`. A cycle's 21 readings land on the channels of its
  condition. The per-channel scale is then a scale within the condition; tokeniser, statistics and
  manifest are unchanged.
- **The six conditions are reader constants** (Saxena et al., 2008). A row matches within 0.5 kft,
  0.01 Mach and 0.5° throttle — at least 5× the released scatter and 8× under the closest two
  conditions' distance; a row matching none is malformed.
- **Single-condition subsets fly sea level**, so FD001's engines keep their readings, scaled over
  more engines at the same condition.
- **The default reading is unchanged.** `--per-operating-condition` on the publishing CLI (refused
  for other corpora). Same checksum, different schema, so the registry keeps the readings apart.

## Consequences

- A window still holds 50 × 21 tokens. The vocabulary is 126 channels over four subsets. In FD002/4
  windows a channel is sampled only at its condition's cycles — irregular, which the encoder takes.
- 28 channels are constant within their condition (scale one; the identity carries the condition).
- Measured over the 98 varying channels: the most compressed subset's units move over 0.56 of a
  channel's scale (median), against 0.004 by sensor and 0.53 for FD001+FD003 alone; lag-1
  correlation within a unit 0.63, against 0.02 and 0.75.
- The default reading's description and all 3,367,539 observations hash identically before and after.
- Test engines will be read in the mode of the corpus the backbone was pretrained on.
- **Measured**: on the same 21 validation engines the four-subset backbone beats FD001+FD003 by
  3.2 %, a narrow margin, and the endpoint is confirmed on it, so this reading is the configuration.

## Alternatives considered

- *Statistics per condition in the tokenisation scheme*: a Catalog concept for one corpus.
- *Settings as channels*: the pretext stays solvable from them.
- *FD001 and FD003 alone*: gives up 71 % of the rows.
- *A scale per unit*: removes an engine's own level, part of what tells wear apart.

## Revisit when

- Another corpus flies discrete regimes that set its levels cycle by cycle.
- The test-engine reader is written.

## Sources

Saxena, Goebel, Simon and Eklund (2008), "Damage Propagation Modeling for Aircraft Engine
Run-to-Failure Simulation", PHM.
