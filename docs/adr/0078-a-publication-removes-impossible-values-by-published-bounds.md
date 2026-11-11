# ADR-0078: A publication removes impossible values by bounds published with the literature, and the manifest states the bounds and what they removed

- Status: accepted
- Date: 2026-11-11
- Supersedes: the point "values are read raw" of ADR-0031

## Context

The clinical corpora hold values no patient can have. PhysioNet 2012 set A has eleven pH readings
of 94 to 735 and of 1 and 3, and 129 temperatures from −18 to 3.6 °C. The eICU demo has FiO2
readings of 3,000 to 10,000 % and 74 Fahrenheit temperatures in its Celsius field. Each of these
inflates its channel's standard deviation: the clinical range of pH in 2012 spans 0.05 of a
deviation once standardised, so a network reads the channel as nearly constant. Trees split on
order and ignore the scale, but a window's summary they read, such as its mean, carries the error. ADR-0031 read values raw and said a cleaned reading would be a
second corpus with its own reader.

The raw files and their checksum do not change when an impossible value is dropped. What changes
is what a publication makes of them, as the split of units already is.

## Decision

- **A publication may name a cleaning policy**: per channel, in the corpus's own unit, an outer
  pair of bounds outside which a value is removed and an inner pair to which a value between the
  two is clipped. A removed value becomes an unmeasured one.
- **The bounds are taken from MIMIC-Extract's variable ranges** (Wang et al., CHIL 2020,
  `resources/variable_ranges.csv`, SHA-256 `f4bf5853…`): outer bounds are its outlier bounds,
  inner bounds its valid bounds. Each corpus has a file of bounds under `corpora/cleaning/`,
  mapping its channel names to that table's rows and converting units where the corpus records
  another (FiO2 as a percent in the eICU demo). A channel the table does not name is left as it is.
- **The policy applies before fitting and before tokenising**, to timed observations and
  timeless descriptors alike, so the statistics and the tokens see the same values. The bounds
  are fixed in advance and fitted on nothing, so no held-out unit informs them. A corpus that
  adopts another's statistics (ADR-0077) adopts the bounds they were fitted under.
- **The manifest states the bounds and, per channel, how many values were removed and clipped**,
  only where a policy was named, so every manifest published without one keeps its bytes.

## Consequences

- A cleaned publication of a corpus has its own checksum; tasks and campaigns defined on the raw
  publication are not reused across the two.
- The counts in the manifest make the cleaning auditable without the raw files.
- Fahrenheit values in a Celsius field are removed rather than converted: converting would guess
  the unit from the value, and 74 of 13,619 temperatures are lost.
- The satellite telemetry is not cleaned: its extreme readings are the events the detectors look
  for.

## Alternatives considered

- *A second known corpus with its own reader*, as ADR-0031 foresaw: three readers that differ from
  the raw ones by one filter, and the rule kept in code rather than beside the data it changed.
- *Clipping every value to its valid bounds*: a pH of 735 is a pH of 7.35 written without its
  point; clipped to 8.4 it becomes a severe alkalosis.
- *Bounds of our own*: no record of where they came from; the literature's table is cited and
  checksummed.
- *Robust scaling alone* (ADR-0079): bounds a tail's effect on the scale, but leaves the wrong
  values in place as extreme ones.

## Revisit when

- A corpus records a quantity in a unit the bounds' table does not → the file converts it.
- A value inside the outer bounds is shown to be a unit error → a narrower bound, with its source.

## Sources

Wang S. et al. (2020). MIMIC-Extract: a data extraction, preprocessing, and representation
pipeline for MIMIC-III. CHIL 2020. Audit: `data/report/l48/audit/` (local).
