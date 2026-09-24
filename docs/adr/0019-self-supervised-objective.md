# ADR-0019: The self-supervised objective — masked reconstruction with the hidden tokens removed from the encoder, a mixture of channel, block and token masks, and a trivial baseline per kind

- Status: accepted
- Date: 2026-09-13
- Full text before condensation: commit `5f14447`

## Context

Masked reconstruction goes quietly wrong in two ways: a loss counted on never-observed positions
looks low without predicting anything, and a single hidden value in a dense channel is recovered by
interpolating neighbours, which teaches nothing that transfers. The answer is a mixture of masks —
blocks of a channel's time, whole channels, single tokens as the minority — with a trivial baseline
per kind, so a kind the model does not beat is known to teach nothing. On the positive control
(ADR-0018) the answers are known in advance: a whole channel is a linear function of the others
(ridge is near optimal), a block needs the other channels, a single dense token is interpolated.

## Decision

- **A hidden token leaves the encoder's input.** It is marked as padding for the encoder; a
  separate decoder puts a learned placeholder at each hidden position, adds the encoder's channel
  and time encodings, reads the visible states through one block and predicts the value with a
  linear head (asymmetric masked autoencoder, He et al., 2022). The backbone never meets a `[MASK]`
  token it will not meet at inference; attention pays only for visible tokens; a whole hidden
  channel is a window of fewer channels. The decoder is discarded after pretraining.
- **The decoder shares the encoder's channel embedding and time encoding**, so a channel hidden
  whole still trains its own embedding.
- **Three independent draws, rates as primitives**: whole channel (`channel_rate`), one block of
  `block_span` of the window's length per channel (`block_rate`), single token (`token_rate`).
  Share hidden = `1 − (1 − c)(1 − b·s)(1 − t)`, stated and measured. Blocks are spans of *time*, not
  runs of tokens. Timeless tokens are never in a block. Every window keeps a visible token.
- **`MaskingStrategy` is a domain value**: four numbers with invariants and the derived share.
  Decoder depth is an argument of the torch objective.
- **Loss = mean squared error over hidden tokens that were observed**, averaged over the batch's
  scored tokens. Padding is excluded by the loss whatever the masks claim. The target is the
  normalised value; the gap is not reconstructed (the mask hides values, not sampling pattern).
- **Kind is read off the outcome**: *channel* = no visible token of the channel remains; *block* =
  hidden by the block draw with the channel visible elsewhere; *token* = the rest.
- **One trivial baseline per kind, scored by the same loss over the same tokens**: linear
  interpolation for blocks and tokens; for whole channels a ridge regression on the other channels'
  nearest values, fitted on unmasked training windows. Constant and timeless channels are tallied
  apart.
- **Spectral diagnostic**: least-squares fit on whole cycles per window plus constant and trend,
  on the tokens' own instants, for truth and residual. Whole cycles, because half-cycles against a
  constant gave a condition number near 27,000.
- **The evidence comes from a report script** (`scripts/masked_reconstruction_report.py`), a plain
  Adam loop superseded by the training runtime (ADR-0021).

## Consequences

- `tests/ml`: one fixed-mask batch of both layouts; 150 Adam steps take the loss from ~1 to < 0.01.
- Tests hold every rule above, e.g. scrambling hidden values leaves predictions unchanged.
- **Measured on the control** (`docs/verification/masked-reconstruction.md`). Dense layout:
  validation 0.37 → 0.032 in twelve epochs, against 0.47 for interpolation and 0.24 for ridge;
  blocks learnt >10× over interpolation, whole channels 5× over ridge; single tokens trivial (0.0167
  vs 0.0135). Sparse layout: 0.21 against baselines above 0.7, every kind learnt. The model gives
  back 94 % (dense) and 61 % (sparse) of the energy at one cycle per window.
- The single-token rate is the ablation's first dial, not a default to defend.
- Code in `pretraining/adapters/objective/` and `pretraining/adapters/diagnostics/`.

## Alternatives considered

- *`[MASK]` in the encoder input* (BERT, TST): changes the signature and shows the backbone a token
  absent at inference.
- *Decoder with its own tables*: a whole hidden channel would not train its embedding.
- *Blocks as runs of tokens*: assumes a meaningful step.
- *Huber loss*: different units from the least-squares baselines (revisited in ADR-0028).

## Revisit when

- The mixture ablation runs → single-token rate first.
- A second objective (contrastive) → the objective becomes a port.

## Sources

He et al. (2022), CVPR; Devlin et al. (2019), NAACL; Clark et al. (2020), ICLR; Zerveas et al.
(2021), KDD; Li et al. (2023), Ti-MAE; Dong et al. (2023), SimMTM, NeurIPS; Hoerl and Kennard
(1970); VanderPlas (2018), ApJS.
