# ADR-0084: Reading a wind farm's turbine tables — a turbine's month is the unit, a farm's turbines share sixty signals as channels, and the rows of one instant are one reading

- Status: accepted
- Date: 2026-11-20

## Context

Two wind farms are a fleet of systems of one design: six turbines at Kelmarsh and fourteen at
Penmanshiel, exported by their operator from one system, Greenbyte, as ten-minute tables of some
three hundred columns. What the archives hold is measured in
[`wind-fleet-sources.md`](../verification/wind-fleet-sources.md). Three of its findings force a
decision:

- Sixty of the columns are the averages of the signals the manufacturer's mapping names for
  both farms; the rest are statistics of the same ten minutes and numbers the export derives.
- The tables of 2023 and 2024 are cumulative: a table restarts its period every few hundred
  rows and repeats every instant before, with values in the newest rows alone. An instant
  stands on up to 92 rows; no two rows of one ever disagree.
- A signal's column is not one name: a unit changed spelling in 2023 (`mm/ss` to `mm/s2`), and
  one title is cased otherwise in the tables than in the mapping.

Unpacked, the tables are ten times their archives of 11 GB.

## Decision

- **One reader for the export, a corpus per farm** (`kelmarsh`, `penmanshiel`): two records
  under two attributions, and a reading that leaves a farm out composes corpora.
- **A turbine is a subset and a turbine's calendar month is the unit**, keyed
  `<turbine>/<YYYY-MM>`; only months holding a reading are units. The turbine stays the
  independent unit, as a mission is for the satellites.
- **A channel is a signal, shared by every turbine of the farm.** The turbines of a farm are
  one machine with one set of sensors, so a new turbine brings no new channel. The mapping's
  sixty signals are the channels of both farms, named by their titles.
- **Only averages are read**, a signal matched by its title, whatever its case, and its unit;
  a title followed by a comma is a statistic, and one over another unit is refused.
- **The rows of one instant are one reading.** A signal's value at an instant is the one value
  its rows hold, across the repeats of a table and the tables of a turbine; the data is refused
  where two of them differ, and so is a table holding one signal in two columns.
- **An average lies at its row's stamp**, in hours since 2016-01-01 00:00 UTC on one axis for
  both farms. A turbine's first month opens at the whole hour of its first reading and its last
  closes at the whole hour after its last, so windows of whole hours lie on clock hours.
- **The archives are read compressed**, each table as a stream, and the fetch script leaves
  them packed. The checksum covers a statement of the farm, turbines and signals, the months
  excluded, and every archive in name order.
- **Months are excluded by name**, as the satellites' are; a name matching no month is refused.
- The reader declares a regular regime, as measured.

## Consequences

- A reader keeping the first or the last row of an instant would lose nearly every value of
  the cumulative years without an error; a test reads a whole turbine on the real files.
- Values are read raw, the sentinels of a failed sensor included (a transformer at −273.2 °C);
  what a publication removes and how it scores a channel is the publication's.
- Angles are read as published and wrap at 360°. The status logs are not read.
- A farm read whole is held in memory, about a quarter of a gigabyte a turbine.

## Alternatives considered

- *A channel per turbine and signal*, as the satellites' channels are named by mission:
  rejected, since 6 to 14 rows of identity would stand for one sensor of one machine type.
- *One corpus of both farms with shared channels*: rejected for now; the farms are two turbine
  models, and whether their signals are one channel is a question for a reading.
- *Every column as a channel*: rejected; five times the tokens for summaries of the same ten
  minutes and computed numbers.

## Revisit when

- A reading needs the turbines' stops → a ground truth over the status logs.
- The multi-system corpus reads both farms as one design → channels shared across farms, or
  counterparts.
