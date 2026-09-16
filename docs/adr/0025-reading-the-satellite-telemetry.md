# ADR-0025: Reading the satellite telemetry — the mission is the unit of independence, the month the unit of splitting

- Status: accepted
- Date: 2026-09-16

## Context

The ESA Anomaly Dataset entered the pretraining mix as an ingredient (ADR-0008), with a subsampling
strategy fixed before any reader existed: the benchmark's lightweight channels, the first half of
each mission by time, at most one observation per channel per thirty seconds with native instants.
The mix is the next thing to build, so the reader had to be written now, and the strategy had to
become code that reproduces the counts the decision was taken on.

The files are unlike the three corpora read so far. A mission is a directory of one zip per
channel, each holding a pickled pandas data frame — 15.4 million rows for a Mission1 channel, 7.4
million for a Mission2 one, instants at millisecond resolution, a `float32` column, no missing
value in any of the seventeen channels read. The channels of a mission share their instants; the
two missions share nothing, not even channel numbers that mean the same thing. Mission3 is
published too, and the benchmark leaves it out.

Two units are the problem. The reader port says a unit is what a corpus is split on
(ADR-0009), the split holds out whole units and rounds the held-out count down, and the
tokenisation scheme fits its statistics on the training units alone. Two missions at a validation
fraction of 0.2 hold out nothing; at 0.5 they hold out one mission, whose channels then have no
statistics at all, because no training unit ever observed them. A corpus whose units have disjoint
channels cannot be split on those units.

## Decision

**Units are the calendar months of a mission's training half**, keyed `ESA-Mission1/2003-05`. The
mission remains the independent unit — the count that decides eligibility in ADR-0008 stays at two
— and the month is the unit of splitting: a seeded split holds out months of both missions, every
channel keeps training months to fit its statistics on, and no window straddles two months. A month
in which no telemetry reached the ground is a unit that yields nothing, like a hospital stay with
descriptors only. Months adjoin in time, so a held-out month's first and last windows sit next to
training windows; that is two windows in some seven hundred, and the price of a split on a corpus
that has two independent units.

**The subsampling is code, and it is constants.** The lightweight channels (41–46 of Mission1,
18–28 of Mission2), the cut at the first bin boundary at or before the exact half of the span the
selected channels cover, and the earliest observation of each thirty-second bin are fixed in the
reader, not taken as parameters. A version's checksum covers bytes (ADR-0009) and cannot tell two
policies apart, so a policy that could vary would let two different corpora share one identity; a
different policy is a different corpus, registered under another name. The rule is the one the
data spike counted with, and the reader reproduces its 64,328,525 observations exactly.

**Channels are named by their mission**: `ESA-Mission1/channel_41`. Channel 41 names one quantity
on one satellite and another on the next; the schema requires unique names, and a collision has to
be impossible rather than merely absent from the two subsets chosen today.

**Time is hours since the mission's start**, the whole second of its first kept observation, on
one axis shared by the mission's months, at the source's millisecond resolution in double
precision. The budget file prices this corpus in hours, so a window of one hour is `--window 1`.

**pandas is a dependency of the package, in an extra of its own** (`corpora`), imported inside
the reader so that publishing any other corpus needs nothing of it, and forbidden in the core by
the import linter like every other framework. There is no upper bound: the pickles were written by
a 2.x pandas and both 2.3.3 and 3.0.5 read them. Unpickling runs the stream it reads; the files
are trusted as far as the fetch script's checksums vouch for them.

**The checksum covers the seventeen channel archives** in mission order and channel-number order,
hashed in one-megabyte pieces. **The mission last read is kept in memory** — a month is a slice of
its mission's arrays, and unpickling a gigabyte for each of a mission's eighty-four months would
cost minutes per pass; the cache holds one mission at a time, so memory is bounded by the largest.

**Mission3 stays out.** Its channel set is a decision of its own, and ADR-0008 names it as the
lever to pull if the mix does not saturate.

## Consequences

- The corpus reads as 105 units of 64.3 million observations; describing it takes seconds and
  streaming it under a minute on the development Mac (`docs/verification/window-sanity.md`).
- The description's unit count (105) and the independence count (2) are different facts; the
  full-corpus test checks both against the spike's measurements.
- The sample in `tests/data/esa_ad/` is thinned rather than copied — one row in two days, 365 rows
  per channel, re-pickled with the locked pandas — so that a training half spans a year of months;
  its README states the derivation.
- A one-hour window stores instants at a resolution of 0.2 ms in the block's single precision,
  above the source's millisecond; a window of six hours would put instants a millisecond apart on
  one stored position and the block writer would refuse it. Longer windows for this corpus need
  the double-precision block ADR-0014 anticipates.
- The Python 3.12 leg that runs without extras skips the tests that write pickles, as it skips
  those that need torch.

## Alternatives considered

- **Missions as units at a validation fraction of one half.** Rejected: the held-out mission's
  channels have no statistics.
- **Fitting statistics on every unit for this corpus.** Rejected: nothing held out may reach the
  statistics, and one corpus does not get its own rule.
- **Days or weeks as units.** Rejected: three thousand day-units multiply the boundaries at which
  held-out and training windows adjoin; a month keeps that to two windows in seven hundred.
- **Fixed thirty-day segments from the mission's start.** Rejected for calendar months, which a
  reader can find in the benchmark's own label files.
- **A contiguous split, the last fifth of each mission held out.** Rejected: it needs a second
  split policy in the domain, while a seeded rank of months gives a blocked split with what exists.
- **The policy as constructor parameters.** Rejected: bytes-only checksums cannot tell policies
  apart, and two readings of one file set must not be one version.
- **Reading the pickles without pandas**, with an unpickler that stubs pandas' classes. Rejected:
  it reimplements pandas' own internals and breaks with them.
- **Converting the pickles once to a plain format.** Rejected: the checksum would cover our
  conversion rather than the publisher's bytes, and the road from fetch to publish would gain a
  step for one corpus.
- **Unqualified channel names.** Rejected: collision-free today by the accident of which subsets
  were chosen.
- **`pandas<3`**, as the fetch spike anticipated. Rejected once tested.

## Revisit when

- The saturation measurement asks for Mission3: an entry in the channel table and a decision on
  which of its channels to read.
- A window of more than about three hours is wanted for this corpus: the block in double precision.
- A second subsampling policy is wanted: a second known corpus, not a parameter.
