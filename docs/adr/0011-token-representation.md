# ADR-0011: Token representation — a window is a set of channel–time–value tokens, normalised per channel, timed relative to the window, with the gap to the previous token of its channel

- Status: accepted
- Date: 2026-09-11
- Full text before condensation: commit `5f14447`

## Context

ADR-0007 fixed the five tensors of the exported graph. What a token *is* was open: how a value
becomes a feature, what time is relative to, how static features enter, how channels of different
corpora share identifiers, and what happens with an unseen channel. Constraints: no regular time
step or fixed observation count; scale parameters in configuration; normalisation from the
training side of a unit-level split only; a mix of regular and irregular sources (ADR-0008).

## Decision

- **A token is `(channel_id, value, time, gap, timeless)`.**
  - `value`: z-score of its channel with training mean and population deviation. A channel
    constant in training divides by one, so a later deviation stays finite.
  - `time`: position in the window, `(t − start) / length`, in `[0, 1)`. Never the absolute
    instant, which leaks position in the data set and does not transfer.
  - `gap`: time since the channel's previous token *in the window*, as a fraction of the length;
    for a channel's first token, time since the window start — a left-censored lower bound,
    computable from the window alone (a serving request has no earlier context).
  - `timeless`: a static feature of the unit, `time = gap = 0`; the encoder skips time encoding.
  - A missing value is no token.
- **Canonical order**: timeless tokens first, then timed tokens by time, channel, value; equal sets
  compare equal. A window carries at least one timed token.
- **Windows.** `WindowSpec(length, stride)` in the unit's own time unit (cycles, hours), never
  samples. Laid from the start of the unit's extent (a fact the reader states); only windows that
  fit count; empty windows are not produced. `SamplingRegime` is never consulted, so independence
  from the step holds by construction. The RUL task anchors its windows at the unit's end.
- **Statistics** `ChannelStatistics(count, mean, std)` are stored as facts, fitted on the training
  side of a `UnitSplit` — units ordered by a seeded digest of each key, before windowing — over all
  their observations. A channel without statistics fails to tokenise.
- **Zero-shot** means zero *labels*: statistics come from the unlabelled training portion of the
  target corpus, and its channels get a new block of the vocabulary.
- **Vocabulary**: one append-only registry `(corpus, channel) → id`, ids `1..n`, `0` for padding.
  Spaces are disjoint per corpus. Timeless channels are declared in `ChannelSchema` and enforced
  per token.
- **An unseen channel gets a new id and an embedding learned from data** (variant A). The encoder
  body, value projection and time encoding transfer; one vector per channel does not (Artetxe et
  al., 2020; STraTS, Tipirneni and Reddy, 2022; Raindrop, Zhang et al., 2022). Terminology:
  *variable-channel*, *permutation-invariant*, not *channel-agnostic*.
  - Variant B, an embedding from the channel's description, is a measured arm on the sub-mix with
    descriptors (C-MAPSS, SKAB, PhysioNet): SMD and ESA-AD channels are anonymised.
  - Variant C, channels by sorted position, breaks the invariance and is rejected.
  - A shared *unknown* embedding, the zero-adaptation lower bound, costs one appended id.
- **Array layout** `TokenBatch`: padded to the longest window; `features` = value and gap;
  `padding_mask` true on padding (PyTorch convention); padding carries id 0; float type is a
  parameter. It decodes back to its windows exactly in fp64, to 1e-7 in fp32.

## Consequences

- The tokeniser is tested against a brute-force reconstruction on hypothesis-generated irregular
  units, plus permutation invariance, no leakage from validation units (with a positive control)
  and masking.
- Tokenising the four C-MAPSS subsets (26.7M tokens, pure Python) took 49 s on the x86 laptop; a
  one-off preprocessing cost — the dataset reads the stored artifact.
- The manifest persists vocabulary, statistics and split membership. A served request must carry
  the window's origin and length, or every first token gets `time = 0`.

## Alternatives considered

- *Position index or resampling to a grid*: assume a regular step.
- *Time differences as an attention bias*: quadratic, coupled to attention; the gap carries it.
- *Windowing in samples for regular corpora*: a grid inside the tokeniser.
- *Stored affine transform*: discards count and meaning; it is derived instead.
- *Robust scaling*: a different statistics type, addable for an ablation.
- *A missing-value token*: absence is already a property of a set.
- *Zero gap for the first token* (GRU-D): false, and indistinguishable from a duplicate.

## Sources

Artetxe, Ruder and Yogatama (2020), ACL. Tipirneni and Reddy (2022), ACM TKDD. Zhang et al.
(2022), ICLR (Raindrop). Che et al. (2018), Scientific Reports (GRU-D). Horn et al. (2020), ICML
(SeFT).
