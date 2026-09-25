# ADR-0017: The encoder — full self-attention over a set of tokens, time at fixed frequencies, one learned vector per channel

- Status: accepted
- Date: 2026-09-13; amended 2026-09-18, 2026-09-24
- Full text before condensation: commit `5f14447`

## Context

A window is a set of tokens (ADR-0011) as five tensors with one dynamic axis (ADR-0007). The export
spike fixed four construction constraints: `scaled_dot_product_attention`, one dynamic axis,
branch-free timeless handling, non-empty windows. Open: the attention mechanism, and how a token's
position in its window is encoded. Target budget 5–30M parameters.

## Decision

- **Full, exact self-attention; window length bounds its cost.** An approximation needs a
  justification exact attention does not — a measured window that does not fit. Patching along
  time and channel × time factorisation are rejected: both assume a regular grid. A latent
  bottleneck (Perceiver) is deferred as the entry point for another modality.
- **Time as sines and cosines at fixed frequencies** `π · 2^k`, `k = 0 … F−1`, projected to the
  width. Fixed is parameter-free, deterministic and cannot overfit; a learned variant must earn its
  parameters. `F` is an architecture field bounded by the data's finest spacing (Tancik et al.,
  2020): twelve for a 48-hour stay at minute resolution. Angles are formed in fp32; the module
  stores octaves (small integers), because top-frequency angles overflow fp16.
- **A timeless token gets a zero time encoding via `where`**, not an `if`.
- **One learned vector per channel** (`LearnedChannelEmbedding`), vocabulary + 1 rows, row 0 the
  padding row with no gradient. Everything else transfers to an unseen sensor set.
- **The encoder returns a state per token**; `MaskedMeanPooling` averages observed states and
  yields zeros for an empty window. The export composes the two.
- **Pre-normalisation** blocks with a final norm (Xiong et al., 2020), GELU, dropout gated on
  training mode.
- **`EncoderArchitecture`** (width, heads, layers, feed-forward width, time frequencies) is a
  Pretraining domain value with an exact `parameter_count(vocabulary)` that needs no torch; a test
  holds the built model to it. Vocabulary size and dropout are arguments, not fields.
- **The torch modules are Pretraining adapters** in `pretraining/adapters/encoder/`: the encoder is
  this context's "model". It implements no port.
- **Input modules are injected** (`SetEncoder(architecture, channel_embedding=…, time_encoding=…)`)
  so a variant replaces one module.
- **The export suite exports the real encoder** from here on; the stand-in is gone.

## Consequences

- Tests: permutation invariance, padding and hidden-value invariance, finite non-zero gradients
  (none on the padding row), timeless ignores its timestamp bit for bit, all-padding gives finite
  states, parameter count exact, seeded builds identical, dropout only in training; the same
  invariants through ONNX Runtime.
- **Reference shape** (256 wide, 4 heads, 6 blocks, feed-forward 1024, 12 frequencies) holds
  **4.78M parameters** over 121 channels — below the target; the saturation measurement decides the
  tier (ADR-0008).
- `docs/verification/encoder.md` (`scripts/encoder_budget_report.py`): at batch 8 in fp16, the
  longest default window's attention buffers are ~10 GiB materialised, inside a 16 GiB device.
  Exact attention fits; approximation has no case yet.
- `N_FEATURES` moved to `shared/kernel/tokens.py` so the domain can count parameters.

## Alternatives considered

- *Inducing points* (Set Transformer): an approximation for windows that do not fit.
- *PatchTST* and *channel × time attention*: assume a grid.
- *`nn.TransformerEncoderLayer`*: uses `nn.MultiheadAttention`, NaN on an empty window.
- *Learned time encoding* (Time2Vec, mTAN): the ablation arm behind the same call.
- *Post-normalisation*: needs warm-up; this project trains many small runs.
- *Architecture only as adapter configuration*: parameter count would need torch.

## Revisit when

- A corpus's window does not fit the device even with fused attention → inducing points or latents,
  compared against exact attention on a shorter window.
- The learned time encoding wins its ablation.
- The description-derived channel embedding is measured (ADR-0011, variant B).

## Amendments

- **2026-09-18 — the first consumer decided.** Evaluation fine-tunes the encoder through a
  structural `BackboneFactory` (`pretrained(weights)`, `fresh()`, `width`) implemented by the
  process (`RestoredBackbones`); the encoder stays in Pretraining (ADR-0030).
  `MaskedMeanPooling` moved to `shared/adapters/tensors/`: export, task heads and inference pool the
  same way.
- **2026-09-24 — the third consumer did not need the module.** Serving runs the inference graph a
  fitted candidate is exported to (ADR-0040) and never holds the encoder as a torch module; the
  export is made in Evaluation, which already holds it through `BackboneFactory`, so that seam
  stays the only one.

## Sources

Vaswani et al. (2017); Lee et al. (2019), Set Transformer; Jaegle et al. (2021), Perceiver; Nie et
al. (2023), PatchTST; Xiong et al. (2020); Tancik et al. (2020); Kazemi et al. (2019), Time2Vec;
Shukla and Marlin (2021), mTAN; Horn et al. (2020), SeFT.
