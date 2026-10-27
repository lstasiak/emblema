# ADR-0066: Reading the eICU demo as 2012 frames a stay — 48 hours from admission to the unit, the charted variables in the 2012 names

- Status: accepted
- Date: 2026-10-27

## Context

The eICU Collaborative Research Database demo (version 2.0.1, ODbL), 2,520 unit stays of 1,841
patients in 186 hospitals, joins the pretraining of the 2012 and 2019 stays unlabelled. Its tables
keep whole stays and measure the vital signs twice: a monitor every five minutes (`vitalPeriodic`,
`vitalAperiodic`) and a nurse's charting (`nurseCharting`). Measured before the reader was
written, on the first 48 hours of 100 stays drawn by seed 1 (medians of the coverage report's
statistics):

| Read as | Co-observed | Gap variation | Rate spread | Readings / stay |
|---|---|---|---|---|
| the 2012 stays | 0.22 | 0.46 | 1.38 | 417 |
| monitor and laboratory | 0.14 | 0.21 | 2.45 | 1,609 |
| charting and laboratory, every label | 0.13 | 0.42 | 1.42 | 343 |
| charting, laboratory and urine, in the 2012 variables | 0.24 | 0.43 | 1.07 | 134 |

Half the charted non-invasive pressures are the monitor's reading at that minute. In the first
48 hours: 63 % of stays leave the unit before their end; 286 laboratory results repeat at one
minute, 128 with another value; urine volumes sit under 27 labels of catheters and voids; 6 % of
oxygen fractions are written as fractions rather than percent; 98 temperatures above 45 °C sit in
the Celsius cell; ages above 89 are written `> 89`.

## Decision

- **A unit stay is the unit**, keyed `<hospital>/<stay>`, over 48 hours from admission to the
  unit as the 2012 challenge frames a stay. A reading before admission, after the patient left the
  unit or after hour 48 is not the stay's; a shorter stay keeps the extent and leaves its tail
  empty. Every stay of the patient table is a unit, one without a reading too.
- **The variables are the 2012 ones the demo charts**, 36 of 37 (the ventilation flag is not
  read): vital signs and coma score from the nurse's charting, laboratory results by name (serum
  bicarbonate, arterial saturation), urine output from the output cells holding a volume (named
  for urine, a void, a catheter or a nephrostomy; counts and mixtures with stool excluded).
- **Units are the demo's**: the laboratory table gives one per variable, and the schema states
  it. Values are read as published, except the oxygen fraction: a value of 1 or less is a
  fraction, since the percent is at least 21, and is read as a percent.
- **One reading per variable and minute**: a laboratory result's latest revision, one never
  revised last, otherwise the first row of its table.
- **Descriptors are timeless tokens** as the 2012 stays carry them: age (`> 89` read as 90),
  gender, height, and the unit type as a presence token of one of the database's eight types. The
  weight at admission is the first reading of the `Weight` series, at minute zero.
- **The checksum covers the four tables read**, in a fixed order.

## Consequences

- The demo's stays sit in the 2012 regime: read through the reader, 0.21, 0.42 and 0.90 against
  0.22, 0.46 and 1.38 (`docs/verification/task-regime-sources.md`).
- A stay holds fewer readings than a 2012 one (median 100 tokens a window against 417), because
  most stays leave the unit before hour 48; a stay of 48 hours charts its vital signs about as
  densely as a 2012 one (heart rate 51 readings against 56).
- The empty tail of a short stay, which the 2012 stays lack, reads as channels not measured.
- A split draws stays, not patients; the demo feeds only a pretraining, whose held-out side gives
  a loss, not a reading.
- A key names the hospital, so hospitals can be held out as parts.
- The registry records the ODbL with the corpus; a block published where others can fetch it
  carries the attribution and the same licence beside it.

## Alternatives

- **The monitor's vital signs.** Rejected: a five-minute grid is another regime (gap variation
  0.21), four times the readings of a 2012 stay, and a schedule a pretext could read as a clock.
- **Every charted and laboratory label.** Rejected: further from the 2012 regime.
- **The extent ending when the stay leaves the unit.** Rejected: a window is cut only where it
  fits, so two thirds of the stays would yield no 48-hour window.
- **The oxygen fraction as published.** Rejected: fractions would sit a hundred times too low.
- **One stay per patient, or stays of 48 hours only.** Rejected: a quarter or two thirds of the
  stays lost for a held-out loss.
