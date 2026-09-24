# ADR-0025: Reading the satellite telemetry — the mission is the unit of independence, the month the unit of splitting

- Status: accepted
- Date: 2026-09-16
- Full text before condensation: commit `5f14447`

## Context

ESA-AD entered the mix as an ingredient (ADR-0008) with a subsampling strategy fixed before a
reader existed. A mission is a directory of one zip per channel, each a pickled pandas data frame
(15.4M rows per Mission1 channel), millisecond instants, `float32`, no missing values. Channels of
a mission share instants; the two missions share nothing. The split holds out whole units and fits
statistics on training units: two missions at fraction 0.2 hold out nothing, and at 0.5 the
held-out mission's channels have no statistics at all.

## Decision

- **Units are calendar months of a mission's training half** (`ESA-Mission1/2003-05`). The mission
  stays the independent unit (eligibility count = 2); the month is the unit of splitting, so a
  seeded split holds out months of both missions and every channel keeps training months. A held-out
  month adjoins training months at two windows in ~700 — the price of a split on two independent
  units.
- **The subsampling is constants in the reader**: channels 41–46 (Mission1) and 18–28 (Mission2),
  the cut at the bin boundary at or before the half of the span, the earliest observation per
  30-second bin. A bytes checksum cannot tell policies apart, so another policy is another corpus.
  The reader reproduces the spike's 64,328,525 observations exactly.
- **Channels are named by mission** (`ESA-Mission1/channel_41`): numbers mean different quantities
  on different satellites.
- **Time is hours since the mission's start**, double precision, millisecond resolution.
- **pandas is in its own extra** (`corpora`), imported inside the reader, forbidden in the core. No
  upper bound: 2.3.3 and 3.0.5 both read the pickles. Unpickling runs the stream; the files are
  trusted as far as the fetch checksums vouch.
- **The checksum covers the seventeen channel archives** in mission and channel order. The last
  mission read is cached, so memory is bounded by the largest mission.
- **Mission3 stays out.**

## Consequences

- 105 units, 64.3M observations; describing takes seconds, streaming under a minute on the Mac
  (`docs/verification/window-sanity.md`).
- The test sample in `tests/data/esa_ad/` is thinned (one row in two days) and re-pickled; its
  README states the derivation.
- A one-hour window stores instants at 0.2 ms in the block's fp32; windows over ~3 hours need the
  float64 block (ADR-0014).

## Alternatives considered

- *Missions as units at 0.5*: the held-out mission has no statistics.
- *Statistics on every unit*: nothing held out may reach the statistics.
- *Days or weeks as units*: more held-out/training boundaries.
- *A contiguous split*: needs a second split policy in the domain.
- *Policy as parameters*: two readings of one file set would share a version.
- *Reading pickles without pandas*: reimplements pandas internals.
- *Converting pickles once*: the checksum would cover our conversion.

## Revisit when

- Mission3 is asked for → a channel table entry.
- Windows over ~3 hours → float64 block.
- A second subsampling policy → a second known corpus.
