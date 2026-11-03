# ADR-0072: The bars of the field over the cells themselves, three distances from the fitting side

- Status: accepted
- Date: 2026-11-01

## Context

The detection tasks (ADR-0067) hold a representation to the summaries per channel, their counts
of readings, and, for the pretext's error, a ridge and an interpolation over the same cells
(ADR-0068). The benchmarks the field reads hold detectors to other bars: a nearest-neighbour
distance over the raw subsequence, the norm of the input itself, the classical reading of
principal components with Hotelling's statistic and the squared residual, and a threshold on how
many deviations a reading lies from its channel's training statistics. The strongest of them,
on sound metrics, keep pace with pretrained models; a reading that does not name them beside its
own leaves a reader to wonder.

Every one of them reads the input, not a representation. For a window of irregular readings the
input the encoder reads is its cells: every channel of the corpus over every span of the window,
with what the span held. A detector over those cells, held to the same distances as the states,
is the bar of the input itself, and the states earn their place only above it.

## Decision

- **The cells themselves are features a detector reads**, `raw_cells`: every window laid out
  over the corpus's channels on the backbone's spans, as the encoder would read it, flattened,
  and no encoder between. The runtime that fits detectors over the states lays them out, since
  the spans are the backbone's; a backbone over one token per reading has no cells to read and
  refuses. The register names the detector `cells_knn`, and its variants turn the same knobs as
  the detectors over states.
- **Three more distances** join the nearest neighbours and the whitened distance, for every
  detector over features: `centre`, the squared Euclidean distance from the fitting mean over
  standardised features, the norm of the input read from the fitting side; `farthest`, the
  largest standardised departure of any one feature, which flags a window for one reading far
  out as a threshold on a channel's deviations would; and `subspace`, Hotelling's statistic
  within the leading components that carry nine tenths of the fitting variance plus the squared
  residual outside them, each over its fitting median so neither swamps the other, the median
  floored at a tenth of what the fitted covariance would give it, since fewer fitting rows than
  features leave a residual of nought; no more components are kept than the rows span.
- **The same alarm, reference and refusals.** Each is set over the fitting rows' own scores at
  the declared contamination, read against a unit where asked, and names no neighbours; a name
  that turns both is refused whichever comes first.
- **No bar is read as a detector of its own kind.** The random score is a number, the share of
  windows in events, not a candidate.

## Consequences

- The screen of the reformulated telemetry leg reads `cells_knn` under the five distances beside
  the summaries and the counts, and the detectors over states under the same distances, so the
  gap between a representation and its input is read under one distance at a time.
- A window lays out every channel the corpus names, both missions' 59 together, over 6 spans
  with 9 features each, 3,186 features: on the first mission the nearest-neighbour distance holds
  the fitting side's 61,000 rows in memory, about 1.6 GB in double precision, and the other
  distances decompose one 3,186 × 3,186 covariance.
- `cells_knn` is a bar, declared beside a representation and never confirmed as a claim of its
  own.
- A reading of the components truncated at another share, or a residual weighed by its own
  components, would be another decision.
