# ADR-0079: How a publication normalises a channel is a property of the publication: the standard score, a robust score, or a normal score of its quantile

- Status: accepted (2026-11-12): the intensive-care corpora take the robust score ([`clinical-input.md`](../verification/clinical-input.md))
- Date: 2026-11-11

## Context

Every publication normalises a channel by the standard score fitted on its training side
(ADR-0011). On skewed laboratory values this compresses the bulk of the clinical range: in
PhysioNet 2012 set A the standard deviation of ALT is 28 times its robust deviation (1.4826 × the
median absolute deviation), of AST 24, of troponin T 11; in the eICU demo troponin I 99. A network
with linear embeddings reads such a channel near zero except for its tail; trees are invariant to
any monotone transform of a channel. A benchmark of deep tabular models transforms each feature
to the normal score of its quantile by default, calls preprocessing "vital for DL models", and
reports one dataset where it harmed them (Gorishniy et al., NeurIPS 2021); richer encodings of
numbers let simple networks "compete with GBDT on some traditionally GBDT-friendly benchmarks"
(Gorishniy et al., NeurIPS 2022). Whether it helps here is measured, not assumed.

## Decision

- **A publication names one normalisation for the channels it fits**; the standard score stays the
  default and every manifest that names none keeps its bytes.
  - *standard*: (x − mean) / sd, as today.
  - *robust*: (x − median) / (1.4826 · MAD), bounded to ±10.
  - *quantile*: Φ⁻¹ of the value's rank, interpolated between 1,001 quantiles of the training side
    (ties take the midpoint of their rank), the rank bounded to [10⁻⁷, 1 − 10⁻⁷], so values lie
    within about ±5.2.
- **The statistics are facts kept beside the normalisation they imply**: count, mean and standard
  deviation always; median and MAD, or the quantiles, where the normalisation reads them. Fitting
  reads the training side once and holds each channel's values only where a median or a quantile
  is needed.
- **A corpus that adopts another's statistics (ADR-0077) adopts its normalisation and the
  bounds its values were held to (ADR-0078)**; naming either beside them is refused.
- **The intensive-care corpora take the robust score**, after cleaning by ADR-0078. A declared
  reading of the network from scratch and the trees on four preparations of PhysioNet 2012 kept
  it: +0.028 at 20 stays and +0.012 at 50 over the raw values, harm at no budget; the quantile
  score moved the network at no budget (`clinical-input.md`, 2026-11-12).

## Consequences

- The satellite telemetry keeps the standard score: its detectors read deviations from a
  channel's own history in those units.
- Normalising back is exact for the standard and robust scores within the bound, and approximate
  for the quantile score, to the interpolation between quantiles.
- Values beyond the training side's range map to the bound under the quantile score: the encoder
  sees "beyond the highest seen", not how far beyond.

## Alternatives considered

- *A logarithm of skewed positive channels*: the choice of channels and offset is ours to make
  per channel; the quantile score needs neither.
- *Clipping the standard score* (measured on the intensive-care task before cells, no gain): a
  bound does not undo what the tail did to the scale.
- *Normalising inside the encoder*: every reader of the block would have to repeat it; the
  publication is where a value is made what a model sees.

## Revisit when

- A reading of PhysioNet 2019 or the eICU demo shows the robust score harming the network from
  scratch → that corpus returns to the cleaned standard score.
- The satellite telemetry's own reading of its input names another normalisation → a new
  record for it.
- Which part of the robust score, its scale or its bound, gives the gain matters to a task →
  read the two apart.

## Sources

Gorishniy Y. et al. (2021). Revisiting Deep Learning Models for Tabular Data. NeurIPS.
Gorishniy Y. et al. (2022). On Embeddings for Numerical Features in Tabular Deep Learning. NeurIPS.
