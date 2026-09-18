# ADR-0017: The encoder — full self-attention over a set of tokens, time at fixed frequencies, one learned vector per channel

- Status: accepted
- Date: 2026-09-13

## Context

The representation is settled: a window is a set of tokens, each a channel, a normalised value,
the gap to the previous token of its channel, and a position in the window (ADR-0011); a batch of
windows is five tensors with one dynamic axis, the token count (ADR-0007); a loader puts those
tensors in front of a model (ADR-0013). The model that reads them did not exist. What it has to be
is constrained from several sides, and each constraint was measured or decided before this ticket
rather than assumed here.

The export spike fixed four things about the encoder's construction (ADR-0007): attention through
`scaled_dot_product_attention`, the only one of three implementations that returns numbers rather
than NaN on a window with no observed token; one dynamic axis beyond the batch, so that a
channel-by-time grid cannot leak back into the graph; a timeless token handled without a Python
branch, because a branch does not survive tracing; and a window carrying at least one observed
token, which the tokeniser upholds. The token representation fixed the meaning of time (relative
to the window, never absolute) and moved the gap into a feature of the token rather than into the
attention (ADR-0011), and chose that a channel the model has not seen gets a new identifier and a
vector learned from data — variant A — with the description-derived vector as a second, measured
arm swapped in at one module boundary of the encoder.

Two decisions were left open by the design and are settled here. The attention mechanism: full
self-attention against approximations of it. And the encoding of a token's position in its window:
fixed Fourier features against a learned encoding. The parameter budget aimed at for the published
tier is 5–30 million. The Pretraining bounded context had no domain yet; it begins here.

The repository's constraints: torch may be imported by adapters and entrypoints only; a context
imports from another only its `contracts` or `shared/`; every scale parameter lives in
configuration, never in code; no assumption of a regular time step anywhere in the model.

## Decision

**Full self-attention, exact, and the window length is the knob that bounds its cost.** Every token
attends to every observed token of its window; the cost grows with the square of the token count.
This is the exact mechanism, not the expensive option. Inducing points and a latent bottleneck are
approximations of it, and an approximation needs a justification where the exact solution does not:
the question a reviewer asks is why an approximation was used while the windows fit in memory. It
enters only when a measurement shows that the window a corpus needs does not fit — and then the
record compares it against exact attention on a shorter window. Two mechanisms that are tempting
for their efficiency are rejected outright: patching along time assumes a regular step inside a
patch, and attention factorised into channel × time assumes a grid; both assume the regular step
the representation refuses. A latent
bottleneck — cross-attention from a fixed set of latents — is deferred, not rejected: it is the
entry point through which a further modality would join, and it is described as an extension.

**The attention is `scaled_dot_product_attention` and nothing else.** ADR-0007 measured the choice;
this ticket implements it and removes the two losing variants from the test suite, where they had
stayed so that this decision had something to pick from.

**Time enters as sines and cosines at fixed frequencies.** A token's position in its window, in
`[0, 1]`, is multiplied by angular frequencies `π · 2^k` for `k = 0 … F − 1` — half a cycle per
window, doubling — and the sines and cosines are projected to the encoder's width. The frequencies
are not learned: a fixed encoding is parameter-free, deterministic, extrapolates to positions
training never showed and has nothing to overfit, so the burden of proof lies on a learned variant
to earn its parameters against it. `F` is a field of the architecture, so a corpus whose default
window holds thousands of tokens can ask for the resolution it needs, and the data bounds it from
above rather than taste: `2^F` at or past the instants a window holds gives neighbouring instants
distinct encodings, while a frequency whose period is shorter than the data's own spacing makes
neighbouring instants unrelated rather than nearby. That is aliasing, not margin, and it is why the
mapping is matched to the bandwidth of the signal (Tancik et al., 2020). The finest spacing on this
project's list is a 48-hour PhysioNet stay at minute resolution, so the tiers ask for twelve
frequencies. The angles are formed in single precision whatever precision the module is held in, and
what the module stores is the octave of each frequency rather than the frequency itself: a wide
encoding reaches some 10^5 radians per window at its top frequency, which half precision cannot
hold, while the octaves are small integers every floating type holds exactly. The learned variant
is a different module behind the same call — the ablation swaps it in without the rest of the encoder
noticing — and it is written when that ablation is scheduled, not now. Rejected, as in ADR-0011:
a positional index, which assumes a grid; and the time difference between pairs of
tokens as an attention bias, which is quadratic in the window and couples the encoding of time to
the choice of attention, while the gap feature already carries the information per token.

**A timeless token gets a zero time encoding, chosen by `where`.** Substituting a timestamp would
make the encoder spend capacity learning to ignore a vector it cannot predict; a Python `if` would
not survive the exporter. The test that a timeless token ignores its timestamp holds bit for bit.

**One learned vector per channel, and the padding row stays at zero.** `LearnedChannelEmbedding`
wraps an embedding table of `vocabulary size + 1` rows with row zero reserved for padding, which
carries no gradient, so a padding position contributes nothing even where a mask is mishandled.
This is the one module that knows a channel; everything else — the blocks, the projection of a
token's value and gap, the encoding of its time — transfers to an unseen sensor set untouched.
Deriving the vector from a description of the channel instead is a module behind the same call.

**The encoder returns a state per token; pooling is a separate module.** The consumers disagree
about what to do with the states: the self-supervised objective scores the tokens it masked, a
probe and the inference graph want one embedding per window. `SetEncoder` therefore returns
`[batch, tokens, width]`, padding positions included, and `MaskedMeanPooling` averages the observed
states into `[batch, width]` — weighting the padding out rather than slicing it, so the token count
stays the one dynamic axis, and yielding zeros for a window with no observed token rather than
dividing by zero. The export composes the two.

**Blocks normalise before each sub-layer and the stream ends in a normalisation.** Pre-normalisation
keeps the residual stream unnormalised and makes depth cost nothing in stability, without the
warm-up schedule that post-normalisation needs (Xiong et al., 2020). GELU in the feed-forward
network, dropout on the residual branches and inside the attention, both gated on training mode.

**The architecture is a value object of the Pretraining domain.** `EncoderArchitecture` — width,
heads, layers, feed-forward width, time frequencies — carries the invariants (positive counts,
width a multiple of heads) and an exact `parameter_count` for a given
vocabulary size, arithmetic that needs no torch: the channel table with its padding row, the two
input projections, each block's two normalisations, attention and feed-forward network, the final
normalisation. A test holds the built model to that number exactly, so the budget of a tier can be
checked from configuration alone and the formula is proven rather than trusted. The vocabulary
size is an argument, not a field: the same architecture trained over two vocabularies is the same
architecture with two embedding tables, which is what "one backbone for any number of channels"
means. Dropout is not a field either: it is regularisation of a run, changes the shape of no
weight, and a checkpoint trained with it loads into a model without it — so the torch module takes
it as an argument and the run configuration owns it. A backbone is this shape plus the weights
learned into it, so the shape is part of what a trained model is identified and reproduced by; the
aggregate that carries it arrives with the registration of a trained backbone.

**The torch module is an adapter of Pretraining**, in `pretraining/adapters/encoder/`, one class per
module — `LearnedChannelEmbedding`, `FourierTimeEncoding`, `SelfAttention`, `EncoderBlock`,
`SetEncoder`, `MaskedMeanPooling` — named for their subject, not the library. It implements no
port: nothing in the application layer calls a model, the training runtime and the exporter will,
and both are adapters. It does not live in `shared/`: the encoder is the core of this context's
language — "model" means "backbone with weights" here and nowhere else — and moving it out would
hollow the context the same way moving the tokeniser out would have hollowed the Catalog
(ADR-0012). That the loading layer went to `shared/` (ADR-0013) is not a precedent against this:
the loader is how a batch reaches *any* model, the encoder is *the* model. Two further contexts will
need the module — Evaluation fine-tunes it, Serving runs it in PyTorch — and a context may not
import another's adapters. The mechanism is decided when the first of them arrives, not now; the
options are listed under *Revisit when*.

**The input modules are injected.** `SetEncoder(architecture, channel_embedding=…,
time_encoding=…)` takes the two modules that a variant would replace; `SetEncoder.for_vocabulary`
builds the standard pair, and `parameter_count` describes that pair. A variant passes its own
module and counts its own parameters.

**The export harness exports the encoder itself from this ticket on.** The stand-in of ADR-0007 was
a set of constructions that had to pass through the exporter while the architecture could still
change cheaply. The architecture has now changed for the last time cheaply, so the four constraints
are checked on the thing that inherits them, on every commit, and the test-only copy of an
architecture is gone. `PooledSetEncoder`, a few lines of test apparatus composing the encoder and
the pooling, stands where the inference adapter will.

## Consequences

- The tests the design calls for — permutation invariance, a mask that removes padding's influence,
  gradient flow — are in `tests/pretraining/adapters/encoder`, beside the ones the design adds: one
  batch holds windows of three and thirty channels; permuting the tokens permutes their states and
  leaves the pooled embedding unchanged; padding appended to a window, or other values hidden under
  the padding, leave the observed states unchanged; every parameter receives a finite, non-zero
  gradient and the padding row receives none; a timeless token ignores its timestamp exactly; a
  window of nothing but padding yields finite states; the built model holds exactly the parameters
  the architecture counts; the same seed builds the same encoder; dropout acts in training mode
  only; a variant input module takes the standard one's place. The export suite runs the same
  invariants through ONNX Runtime on the real encoder.
- **The reference shape of the published tier — 256 wide, 4 heads, 6 blocks, feed-forward 1024,
  12 frequencies — holds 4.78M parameters over the 121 channels of the measured corpora, below the
  5–30 million aimed at.** The coarse 12 · d² · L estimate (4.72M) already was. This record does not
  settle the tier's shape: the corpus-saturation measurement that precedes the first full
  pretraining does, and it now has an exact count to work with. Seven blocks or a width of 288 cross
  the line; whether they should is a question for that measurement.
- Measured cost, in `docs/verification/encoder.md`, reproducible with
  `scripts/encoder_budget_report.py`: the exact parameter count per tier, what a window's attention
  weighs with the scores materialised, and a training step per window at the default window of
  every measured corpus — from PhysioNet's ~430 tokens to SMD's ~3800. The x86 leg is recorded
  there; the M1 leg is added when the branch is verified on it. The arithmetic states the verdict
  for the published tier: at a batch of eight in half precision, the attention buffers of the
  longest default window come to about 10 GiB when materialised, inside a 16 GiB device with 4 GiB
  of headroom, and shrink to the linear term wherever a fused kernel runs. The exact mechanism fits
  the windows the corpora produce; the approximation has no case yet.
- `N_FEATURES`, the number of continuous features of a token, moves from the array codec to
  `shared/kernel/tokens.py`: it is a fact of the representation, and the domain of Pretraining needs
  it to count parameters while it may not import an adapter. Recorded as a kernel extension.
- The budget file's tiers now state a whole shape (`heads`, `feedforward_width`,
  `time_frequencies`), and `scripts/budget_file.py` turns a tier into an `EncoderArchitecture` in
  one place for every report script, over the one vocabulary the measured corpora add up to — so a
  report cannot label a model with a tier's name and build it at some other size.
- No import-linter contract changed: the wildcards over `*.domain` and `*.adapters` cover the new
  packages, and the entry that keeps a domain from its own contracts waits for the first contract
  of Pretraining, because a listed module must exist.
- The throughput note of ADR-0013 timed a step of the stand-in; the script now times the encoder,
  and the note gets its number when the branch runs on the M1.

## Alternatives considered

- **Inducing points** (Set Transformer, Lee et al., 2019): attention through `m` learned inducing
  points, linear in the window. An approximation whose case is a window that does not fit; the
  threshold is measured, not assumed, and is below.
- **A latent bottleneck** (Perceiver, Jaegle et al., 2021): cross-attention from a fixed array of
  latents. Deferred as the extension through which another modality would enter; not the default
  for a single modality whose windows fit.
- **Patching along time** (PatchTST, Nie et al., 2023): tokens are patches of consecutive samples.
  Rejected: a patch assumes a regular step inside it, the assumption the project refuses to make.
- **Attention factorised into channel and time**: rejected for assuming a channel-by-time grid.
- **`nn.TransformerEncoderLayer`**: rejected because it attends through `nn.MultiheadAttention`,
  which returns NaN on a fully masked window in PyTorch and in the exported graph alike
  (ADR-0007).
- **A learned time encoding** (Time2Vec, Kazemi et al., 2019; the learned embedding of mTAN, Shukla
  and Marlin, 2021): the ablation arm, behind the same call, written when the ablation is run.
- **Time differences as an attention bias**: rejected in ADR-0011 and again here — quadratic and
  coupled to the attention decision.
- **Post-normalisation**: rejected on Xiong et al. (2020) — it needs a learning-rate warm-up that
  pre-normalisation does not, and this project trains many small runs rather than one large one.
- **The encoder returning the pooled embedding only**: rejected — the objective scores tokens.
- **The architecture as a pydantic configuration in the adapters only**: rejected — the parameter
  count would need torch, the domain of Pretraining would still be empty, and the aggregate that
  registers a trained backbone needs the shape as a value.
- **The module in `shared/adapters`**, where the loader went: rejected above; the threshold at which
  it flips is below.
- **Keeping the stand-in until the exporter adapter exists**: rejected — a construct that the
  exporter cannot trace would have surfaced four stages later, and two architectures with one
  signature would have lived in the repository meanwhile.

## Revisit when

- A corpus needs a window that does not fit the published device even with a fused attention
  kernel. Then inducing points or a latent bottleneck enter, compared in a record against exact
  attention on a shorter window of the same corpus.
- Evaluation needs to fine-tune the encoder, or Serving to run it in PyTorch. Then one of: the
  composition root injects a factory of the module into the consumer's adapter, typed structurally
  as a torch module taking the five tensors; an Open Host Service in `pretraining/contracts` names
  the seam without naming torch; or the module moves to `shared/adapters` and this record is
  superseded on that point. The first consumer decides, with the wiring as the argument.
- The learned time encoding wins its ablation. Then the default flips at the module boundary.
- The description-derived channel embedding is measured on the sub-mix that carries descriptors.
  Then the record of variant B says what transfers through the input layer.

## Sources

- Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
- Lee, J. et al. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant
  Neural Networks. ICML.
- Jaegle, A. et al. (2021). Perceiver: General Perception with Iterative Attention. ICML.
- Nie, Y. et al. (2023). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers.
  ICLR.
- Xiong, R. et al. (2020). On Layer Normalization in the Transformer Architecture. ICML.
- Tancik, M. et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low
  Dimensional Domains. NeurIPS.
- Kazemi, S. M. et al. (2019). Time2Vec: Learning a Vector Representation of Time. arXiv:1907.05321.
- Shukla, S. N. and Marlin, B. M. (2021). Multi-Time Attention Networks for Irregularly Sampled
  Time Series. ICLR.
- Horn, M. et al. (2020). Set Functions for Time Series. ICML.

## 2026-09-18 — the first consumer decided: a factory the process wires

Evaluation fine-tunes the encoder from this date, and the mechanism is the first of the three
listed under *Revisit when*: the consumer's torch adapter takes a structural `BackboneFactory` —
`pretrained(weights)`, `fresh()`, `width`, modules typed as `nn.Module` over the five tensors of a
batch — and the process implements it beside its composition roots (`RestoredBackbones`), reading
the model Pretraining stored. The encoder stays in `pretraining/adapters/encoder/`. The wiring
argument and the threshold at which the choice flips are in ADR-0030.

One module named in this record moved: `MaskedMeanPooling` is now
`shared/adapters/tensors/masked_mean_pooling.py`. It carries no language of this context and
three things pool the same states the same way — the export, a task's head, an inference graph —
so it is shared the way the loading layer is (ADR-0013). Nothing else about the encoder changed.
