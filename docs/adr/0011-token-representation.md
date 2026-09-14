# ADR-0011: Token representation — a window is a set of channel–time–value tokens, normalised per channel, timed relative to the window, with the gap to the previous token of its channel

- Status: accepted
- Date: 2026-09-11

## Context

The encoder reads a window of sensor data as a set of tokens. ADR-0007 fixed the five tensors the
exported graph declares — `features [batch, tokens, 2]`, `channel_ids`, `timestamps`, `timeless`,
`padding_mask` — and measured that a window must carry at least one observed token. What a token
*is* was still open: how a raw value becomes a feature, what the timestamp is relative to, what
the second feature carries, how static features of a window enter, how channels of different
corpora share one identifier space, and what happens with a channel the model has never seen.

Constraints the representation must satisfy: no assumption of a regular time step or of a fixed
number of observations per window; every scale parameter in configuration;
normalisation statistics from the training side of a unit-level split only; static features as
timeless tokens without an invented timestamp; a mixed corpus of four regular and two
irregular sources from the first pretraining run (ADR-0008); and a stated answer to the unknown
channel, because the transfer matrix rests on it.

## Decision

### The token

A token is `(channel_id, value, time, gap, timeless)`.

- **`value`** is the observed value after the z-score of its channel: `(x − mean) / std` with the
  population standard deviation of the training data. A channel that never varied in training
  (six of the 21 C-MAPSS sensors are constant in FD001: T2, P2, epr, farB, Nf_dmd, PCNfR_dmd)
  divides by one instead, so a value that deviates later stays finite and in raw units; the test
  for zero spread is exact, because a channel with any spread at all is scaled up correctly.
- **`time`** is the position within the window, `(t − window start) / window length`, in `[0, 1)`.
  Never the absolute instant: an absolute timestamp reveals which part of a data set a window
  comes from and does not transfer between corpora collected in different years.
- **`gap`** is the time since the previous token of the same channel *in the window*, as a fraction
  of the window length; for the first token of a channel it is the time since the window start.
  It never exceeds `time`. For the first token the gap is a left-censored lower bound of the true
  gap — the predecessor is outside the window, so it lies at least this far back — which is
  honest and computable from the window alone. Two alternatives were rejected: a zero gap for the
  first token (the GRU-D convention) asserts a gap that is false and is indistinguishable from a
  genuine duplicate at the same instant; the true predecessor from before the window is known to
  the tokeniser of a corpus but not to the API that tokenises a single request, so it would build
  a train/serve mismatch into the representation.
- **`timeless`** marks a static feature of the unit (a patient's age, a device rating) carried into
  every window of that unit. Such a token has `time = gap = 0` and the encoder skips time encoding
  for it: substituting a timestamp would make the model learn to ignore a vector it cannot
  predict.
- A **missing value is no token**. A set carries absence naturally; readers emit finite values only,
  and the masked-reconstruction objective masks tokens of its own.

Tokens of a window are kept in one canonical order — timeless tokens first, by channel and value,
then timed tokens by time, channel and value — so two windows built from the same tokens in any
order compare equal; the constructor rejects any other order and a factory establishes it. A
channel observed twice at the same instant is allowed (the second occurrence has gap zero);
whether to deduplicate is the reader's decision.

**A window carries at least one timed token.** This is stricter than ADR-0007's "observed": a
window of static features alone holds no measurement, so it is not a window. The tokeniser never
produces one, and a unit whose data is static features only yields nothing — three PhysioNet stays
are exactly that.

### Windows

`WindowSpec(length, stride)` is expressed in the unit's own time unit — cycles for C-MAPSS, hours for
PhysioNet and ESA — and never in samples. Windows are laid from the start of the unit's time
extent every `stride`; only a window that fits inside the extent counts; a window into which no
observation falls is not produced (ADR-0008: an empty window costs no compute and teaches
nothing). The extent is a fact the reader states — an engine of `L` cycles spans `[1, L + 1)`, a
hospital stay its protocol's 48 hours — not the span of the observations. Because the rule is the
same for a regular and an irregular corpus and `SamplingRegime` is never consulted, independence
from the sampling step holds by construction rather than by discipline. The counts reproduce the data spike's: 34 windows and
35 700 tokens for the two sample engines, 25 395 windows and 26 664 750 tokens for the four C-MAPSS
subsets at the default `w50s5`.

Two things this leaves for later, deliberately. Start-aligned windows drop the tail of a unit
that no full window covers — for run-to-failure data that is the failure end, harmless for
self-supervised pretraining but decisive for the remaining-useful-life task, whose windows the
Evaluation context will anchor at the end (an `anchor` parameter when that task is defined).
And there is no time-unit label on a spec or an extent: a mismatch is caught today by the
consistency test against the measured window counts; the second corpus decides whether a label is
worth carrying.

### Normalisation statistics and the split

`ChannelStatistics(count, mean, std)` per channel are kept as facts about the training data rather
than folded into an affine transform: they are the provenance of a model, the numbers a model card
reports, and the only description a channel without metadata offers. They are fitted on the
training side of a `UnitSplit` — units, never windows, ordered by a seeded digest of each key
before any window is cut, so that overlapping windows of one unit can never straddle the split —
and over *all* observations of those units, whether or not a window later covers them,
because the statistics describe the data and not the windowing. A channel the training units never
show has no statistics, and tokenising it fails rather than passing a raw value through.

**Zero-shot protocol.** For a corpus the backbone has not seen, the statistics come from the same
fit on the unlabelled training portion of the target corpus, and its channels get a new block of
the vocabulary. This is what "zero-shot" means in the transfer matrix: zero *labels* in
adaptation, not zero target data, and it must be stated as such there.

### The channel vocabulary and the unknown channel

The vocabulary is one registry over every registered corpus: `(corpus, channel) → id`, identifiers
`1..n` in registration order, `0` reserved for padding so a padding position can never be read
as a channel even where a mask is mishandled. Spaces are disjoint per corpus — `T2` of one corpus
and `T2` of another never share an identifier — and the registry is append-only: extending it
with a corpus adds the channels it has not seen and changes nothing it has, so identifiers stay
valid for every artifact that carries them. Static channels are declared in the corpus's
`ChannelSchema` (`Channel.timeless`) so that the vocabulary knows them, and the tokeniser enforces
the declaration per token: an observation on a timeless channel, or a static feature on a timed
one, is an error. Entries keep the channel's unit.

**Variant A is the mechanism: a new channel gets a new identifier and an embedding learned from
data.** What transfers to an unseen sensor set is the encoder body, the shared value projection and
the time encoding — everything but one vector per channel. The precedent is direct: Artetxe, Ruder
and Yogatama (2020) freeze a transformer body and learn new input embeddings for an unseen
language from unlabelled text, then transfer the task head; STraTS (Tipirneni and Reddy, 2022)
uses the same `(time, variable, value)` triplet with learned variable embeddings; Raindrop (Zhang
et al., 2022) learns sensor embeddings and tolerates missing sensors. The terminology follows:
*variable-channel* and *permutation-invariant*, not *channel-agnostic*, because the model does not
understand the semantics of a channel it has not seen.

Variant B — an embedding derived from a channel's description — is the more ambitious hypothesis,
not the more professional default, and it cannot be the default for this corpus mix: SMD's 38
metrics and ESA-AD's channels are anonymised, without unit or type, and ESA-AD supplies two thirds
of the unique values that legitimise the reference model (ADR-0008); the statistical descriptors
that remain are exactly what per-channel normalisation removes. B is therefore a second,
measured arm on the sub-mix with descriptors (C-MAPSS, SKAB, PhysioNet), swapped in at one module
boundary of the encoder and reported beside A. Nothing here forecloses it: entries
carry the unit, statistics carry mean and spread, and a `kind` field on `Channel` is a non-breaking
addition. Variant C — channels identified by position in a sorted order — is rejected: it breaks
the invariance the representation exists for. A shared *unknown* embedding, as the true
zero-adaptation lower bound of the transfer matrix, costs one appended identifier when the matrix
wants it.

### Layout in arrays

`TokenBatch` lays windows out as the five arrays of ADR-0007, padded to the longest window:
`features` carries value and gap, `padding_mask` is `True` where the position is padding (the
PyTorch convention, so the mask reaches attention unchanged), padding positions carry channel
identifier `0` and zeros, tokens keep their canonical order, and the floating type is a parameter
because the encoder's input precision is a property of the model artifact. The layout decodes back
to the windows it was built from, exactly in double precision and to `1e-7` in single.

## Consequences

- The tokeniser is held to the definition, not to itself: a brute-force reconstruction of every
  window from the observations inside it, on hypothesis-generated units with random instants,
  random channel subsets per instant and a random window, must equal the streaming adapter's
  output. The mandatory tests of the ticket — permutation invariance, synthetic irregularity, no
  leakage from validation units (with the positive control that changing training units *does*
  change the statistics), masking — are in `tests/catalog/ports`, `tests/shared/kernel` and
  `tests/shared/adapters/arrays`.
- **Throughput, measured, not assumed.** Tokenising the four C-MAPSS subsets — reading 160 359
  rows, fitting and producing 26 664 750 tokens in pure Python — took 49 s on the x86 development
  laptop; FD001 alone takes 6 s and is the test that runs where the raw data is present. ESA-AD at
  ~150M tokens per pass is of the order of five minutes. This is a one-off cost of the
  preprocessing step; the dataset layer reads the stored artifact and never
  re-tokenises per epoch. Where the seam for a faster implementation lies is ADR-0012's subject.
- What this leaves to the code that comes after. The manifest persists the vocabulary, the statistics and the *membership* of the
  unit split — the seed reproduces it (each unit is placed by a digest of its own key, so the
  split survives a Python upgrade and, but for the units at the cut, a corpus that has grown), and
  the membership records what was actually used — and counts the units that yielded no window. A
  served request must carry the window's origin and length explicitly, or every first token gets
  `time = 0` and the distribution shifts. Categorical static features (PhysioNet's gender, ICU
  type) pass through a z-score that means nothing; the PhysioNet reader decides their encoding.
  The channel embedding stays one module so variant B is a swap.
- The `w50s5` consistency test asserts a regular grid for one corpus; it is a fact about C-MAPSS,
  not an invariant of the representation.

## Alternatives considered

- **Position index or resampling onto a grid.** Rejected: both assume a regular step and
  erase the irregularity the second axis of the thesis needs.
- **Time differences as an attention bias.** Rejected: quadratic
  in the window and coupled to the attention decision; the gap feature carries the same
  information per token.
- **Two windowing modes, samples for regular and hours for irregular corpora**, as the data spike's
  scripts do. Rejected: the samples mode is a grid inside the tokeniser.
- **An affine `(offset, scale)` per channel** as the stored normalisation. Estimator-agnostic and
  smaller, but it discards the count and the meaning of the numbers; the statistics are kept and
  the transform is derived.
- **Robust scaling (median, interquartile range).** Not rejected — a different estimator producing
  a different statistics type, added when an ablation wants it; the representation does not change.
- **A token for a missing value, with a mask.** Rejected: absence is already a property of a set,
  and the objective's mask is a different mask.
- **A `kind` enum on `Channel` instead of a flag.** Deferred: two kinds exist, the flag names the
  property that matters (whether the token is timed), and the enum is a non-breaking addition when
  a third kind or variant B needs it.

## Sources

- Artetxe, M., Ruder, S. and Yogatama, D. (2020). On the Cross-lingual Transferability of
  Monolingual Representations. ACL.
- Tipirneni, S. and Reddy, C. K. (2022). Self-Supervised Transformer for Sparse and Irregularly
  Sampled Multivariate Clinical Time-Series. ACM TKDD.
- Zhang, X., Zeman, M., Tsiligkaridis, T. and Zitnik, M. (2022). Graph-Guided Network for
  Irregularly Sampled Multivariate Time Series (Raindrop). ICLR.
- Che, Z. et al. (2018). Recurrent Neural Networks for Multivariate Time Series with Missing Values
  (GRU-D). Scientific Reports.
- Horn, M. et al. (2020). Set Functions for Time Series (SeFT). ICML.
