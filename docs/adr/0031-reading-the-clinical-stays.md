# ADR-0031: Reading the clinical stays — a stay is the unit on the protocol's 48-hour axis, descriptors are timeless tokens, and the test set stays out

- Status: accepted
- Date: 2026-09-18; amended 2026-09-19 (set B held out as a part)
- Full text before condensation: commit `5f14447`

## Context

PhysioNet/CinC Challenge 2012: 12,000 ICU stays in sets A, B and C of 4,000, each a file of
`Time,Parameter,Value` rows — six descriptors at `00:00`, then up to 37 clinical variables over 48
hours. It is the first corpus with static descriptors on real units, the first with units that
may hold no measurement (3 in set A), and the sparsest (median 422 values per stay over 37
channels). Measured before writing the reader: descriptors not always in the first six rows;
height unrecorded (`-1`) in 47 % of stays; rows stamped `48:00`; `Weight` both a descriptor and a
series; duplicate (time, variable) pairs; a few impossible values (negative temperatures, pH in
the hundreds).

## Decision

- **A stay is the unit** (`set-a/132539`). **The extent is the protocol's**, `[0, 48 h + 1 min)`
  for every stay; a stay that ended early has empty later hours. **Time is hours since admission**
  at minute resolution.
- **Descriptors are timeless tokens.** Age, height, gender as numbers; ICU type as a presence token
  of one of four ward channels (no order to invent). A descriptor recorded as `-1` or absent is no
  feature.
- **Admission weight is the first measurement of the `Weight` series** at hour zero, not a second
  channel for one quantity (3,674 observations more than the spike counted).
- **Only sets A and B are read.** Set C is the classification task's frozen side (ADR-0026).
- **Values are read raw**: a cleaning rule would be a second corpus under the same checksum.
- **Rows are trusted in file order**; a row going back in time is refused.

## Consequences

- Set A: 4,000 units, 44 channels (37 timed, 7 timeless), 1,737,654 observations; describing takes
  10 s, the sanity pass 58 s (`docs/verification/window-sanity.md`).
- Rows stamped `48:00` are read and checksummed but fall in no whole-stay window.
- Five channels never vary (four wards, `MechVent`); scale falls back to one, the value is zero, the
  identity carries the information.
- 23 channels have values beyond eight deviations (pH 88σ); one more argument for statistics per
  corpus.
- The four-stay sample in `tests/data/physionet2012/` is byte-exact and covers the format's edge
  cases.

## Alternatives considered

- *The stay's own extent*: whole-stay windows would differ per unit.
- *Minutes as the axis*: other corpora price windows in hours.
- *ICU type as codes 1–4*: invents an order.
- *One-hot gender*: says it twice.
- *Admission weight as its own channel*, or *dropped*: one quantity learnt twice, or lost.
- *Cleaning in the reader*, *set C as a subset*: rejected above.

## Revisit when

- The classification task arrives → a per-unit label port; set C named as frozen.
- A cleaned reading is wanted → a second known corpus.
- Normalisation across corpora is proposed → these tails argue against it.

## Amendments

- **2026-09-19 — set B is the held-out side.** A third split policy, `PartSplit`, holds out every
  unit of a named part (`--hold-out-subset set-b`), so numbers sit beside the challenge's own
  division. Unit keys must name their part; SMD machines are now keyed within their group
  (`1/machine-1-1`), so a seeded republication of SMD draws a different fifth — published versions
  are untouched. Every one of the 44 channels is measured in both sets (rarest: `Cholesterol`, 315 in
  A), asserted by a test where the raw files exist.

## Sources

Silva et al. (2012), Computing in Cardiology 39; Goldberger et al. (2000), Circulation 101(23).
