# ADR-0065: Reading the Water Quality Portal — a location is the unit over fixed years, a reading has an instant of its own, and only the widely measured characteristics are channels

- Status: accepted
- Date: 2026-10-26

## Context

The US Water Quality Portal joins the intensive-care stays as a corpus of another domain sampled
asynchronously (`docs/verification/task-regime-sources.md`). Its narrow result profile is one row
per result: organisation, location, activity, medium, date, time and zone, depth, characteristic,
fraction, value, unit code, statistic, database. Measured on the fetched batches before writing
the reader:

- results of sediment, habitat, air and tissue beside water, and quality-control samples;
- a fifth to a quarter of the results dated without a time of day;
- statistics of continuous records (minimum, mean, maximum): 13 % of the readings of four batches;
- logger deployments whose thousands of readings all carry the activity's start, plus depth
  profiles and repeats: half the readings shared location, characteristic and instant with another;
- one characteristic in several unit codes, some differing only in case; fractions labelled
  inconsistently (pH as `Total` or blank);
- 749 characteristic names over 100 locations, the most widely measured 64 holding 84 % of the
  readings; 1,079 of 114,340 location identifiers holding a slash.

## Decision

- **A location is the unit**, keyed `<database>/<location identifier>`; the first slash is the
  part separator, so an identifier may hold its own.
- **The extent is the fetched years** for every location, 2010-01-01 to 2024-01-01 UTC, in hours
  since the epoch, so one window length cuts every location alike.
- **A reading** is a result taken in water, not a quality-control sample, measured rather than
  computed (statistical base blank, `MPN` or `Instantaneous`; a mean, maximum, ratio or recovery
  is not), with a finite number, reported by the organisation with most of the location's rows.
  Unknown zone codes are read as UTC.
- **A reading has an instant of its own.** Results dated alone are dropped, and so are the
  readings of an activity reporting one characteristic, fraction, unit and depth more than once.
  Either would give a pretext a clock to read.
- **One reading per characteristic, unit and instant**: the water as taken before a dissolved
  fraction, then the shallowest, then the first in the file. A depth profile says nothing about
  when a channel is measured.
- **A characteristic is read in its most frequent unit**, codes compared regardless of case;
  results in other units are dropped. Names are kept as published; fractions are not told apart.
- **The 64 characteristics measured at the most locations are the channels**, counted by
  locations, so one dense location does not choose the vocabulary. A location enters with at
  least 50 readings on them.
- **The corpus is a prefix of the audit's frame** in its seeded order, fetched a hundred locations
  a request into `batches/batch-<n>.zip`; the checksum covers the batches in name order. Values
  are read raw.

## Consequences

- A channel's values are in one unit, so per-channel normalisation is meaningful.
- Every channel is measured at many locations, so a seeded split cannot leave one without
  training statistics, and the per-corpus shortcut audit, whose design grows with the channel
  count, stays affordable.
- About half the readings passing the first filters are dropped, so the corpus needs about twice
  the locations; the depth structure of lakes is lost.
- A characteristic under two names stays two channels; reading through one organisation removes
  most such pairs. Total and dissolved fractions at different instants share a channel.
- Units come batch by batch and the last batch is kept, so a pass over the units parses each batch
  once.

## Alternatives

- **One channel per unit code, or per fraction.** Rejected: both double channels for one quantity
  and read them as never measured together.
- **Converting units.** Rejected: a conversion table per characteristic risks more than it recovers.
- **Every characteristic as a channel.** Rejected: hundreds seen at a handful of locations, some
  only on the held-out side, which stops a publication.
- **Keeping repeats at one instant**, as the 2012 stays keep rare duplicate pairs (ADR-0031).
  Rejected: here they are half the readings, in groups of hundreds.
- **Results dated alone at midnight**, as the audit's declared frame read them. Rejected: dropping
  them moves the regime measures by at most 0.03 and removes the clock.
