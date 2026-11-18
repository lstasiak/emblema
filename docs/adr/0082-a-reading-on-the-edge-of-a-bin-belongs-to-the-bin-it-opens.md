# ADR-0082: A reading on the edge of a bin belongs to the bin it opens, by one rule for the grids and the cells

- Status: accepted
- Date: 2026-11-16

## Context

A published window stores a reading's time as a fraction of the window in single precision.
A corpus sampled on a regular clock therefore stores times that sit exactly on the edges of any
division of the window its clock divides: the 2019 intensive-care stays are 24 hours of hourly
rows, so every time is k / 24, and some of those land a hair below k / 24.

The grids of the Evaluation assign such a reading to the step it opens (`steps_of`, a
tolerance of a thousandth of a step). The cells of channel
and time bin (ADR-0062), written later, took a plain floor. On the 2019 stays that floor put
a reading one bin early wherever its time was stored below the edge
(`docs/verification/cell-pretext.md`, 2026-11-16):

| bins | readings on an edge | readings a bin holds, against an even share |
| --- | --- | --- |
| 6 | 0.24 | 0.72 to 1.24 |
| 12 | 0.49 | 0.47 to 1.51 |
| 24 | 1.00 | 7 bins of 24 empty, their neighbours doubled |

The 2012 stays, the eICU demo and the satellite telemetry, timed to the minute or the second,
have at most 0.022 of their readings on an edge and no empty bin at any count up to 48.

## Decision

- **One rule, in the shared adapters.** `steps_of` moves to
  `shared/adapters/arrays/steps.py`; the grids and `BinnedTokens` both call it. A reading
  within a thousandth of a bin below an edge falls in the bin that edge opens; the window's very
  end stays in the last bin.
- **The offset of a cell's last reading is held to `[0, 1]`**, so a reading stored a hair before
  the edge of the bin it opens stands at that bin's start.
- **No knob.** The floor was an accident of rounding, not a layout anyone chose; a setting that
  restored it would keep a defect selectable.

## Consequences

- Every candidate over cells reads a corpus sampled on its bins' edges differently from before:
  on the 2019 stays up to a quarter of the readings at six bins move one bin later, so each
  bin holds four hours of rows rather than three to five. On the other corpora the change
  can touch about 2 % of readings at 48 bins and under 1 % at six.
- Backbones over cells pretrained before this record laid the 2019 stays out by their rounding,
  in pretraining and in every evaluation alike, so their readings stand as measured; they are
  not continued or compared cell for cell with backbones pretrained after it. A comparison
  across the change is rerun whole under the rule.
- A division finer than a corpus's own clock still leaves bins that can never hold a reading:
  32 bins over the 2019 stays' 24 hourly rows leave 8 empty by construction. No rule of
  assignment removes that; the count of bins is chosen with the corpora's clocks in view.

## Alternatives considered

- **Stamping an hourly row at the middle of its hour, in the reader.** It would cure the 2019
  stays by a new publication and leave the layout unstable for the next corpus on a clock.
- **A tolerance in the cells alone.** Two statements of one rule, free to drift apart.
- **Keeping the floor for stored backbones behind a flag.** The readings they gave stand in the
  notes at their commits; nothing reruns them cell for cell.
