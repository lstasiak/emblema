# ADR-0019: The self-supervised objective — masked reconstruction with the hidden tokens removed from the encoder, a mixture of channel, block and token masks, and a trivial baseline per kind

- Status: accepted
- Date: 2026-09-13

## Context

The encoder exists (ADR-0017) and the road from a corpus to a batch in front of it is open
(ADR-0018): both control layouts publish through the real use cases and meet in one batch. What
did not exist was a reason for the encoder to learn anything. The plan asks for masked
reconstruction — hide part of a window, predict what was hidden — and states the two ways it goes
quietly wrong. The loss counted on positions that were never observed, padding or values that
were never there, looks low without the model having predicted anything. And a single hidden
value in a densely sampled channel is recovered by interpolating its neighbours, so a model trained
to fill single gaps learns to interpolate and nothing that transfers. The answer taken here to the
second is a mixture of masks — blocks of a channel's time, whole channels, and single tokens as the
minority ingredient — with a high overall share, and a mandatory check: a trivial baseline matched
to each kind of mask, so that a kind the model does not beat is known not to teach anything. A
spectral view of what the model gives back completes it: a model that recovers only the slow
components has learnt smoothness.

Three things about the representation shape the design. A window is a set of tokens with no axis
for channels (ADR-0011), so "remove a token" is a mask, not a slice. The encoder returns a state
per token, padding positions included, so a consumer decides which states count (ADR-0017). And
the batch carries no positions for values that were never observed: an unobserved instant is not a
token, so the only positions that are not tokens are padding.

The positive control is the first data the objective meets (ADR-0018). Its channels are linear
projections of a handful of shared latent factors with a little noise, sampled on a grid in one
layout and irregularly in the other. That gives the diagnostics a known answer before any real
corpus: a channel hidden whole is a linear function of the others up to noise, so the ridge
baseline is near what any method can do there and the model is expected to match it rather than
beat it by much; a block of a channel's time is where the other channels carry what interpolation
cannot see, so the model is expected to beat interpolation clearly; and a single token between
two dense neighbours is recovered by interpolation to within the noise, so the model is expected
not to beat it. The last is the warning above, made measurable.

## Decision

**A hidden token leaves the encoder's input; it does not become a placeholder.** The objective
marks hidden tokens as padding for the encoder's call, which for a set encoder is the same as their
never having been observed, and a separate decoder puts a learned placeholder at every hidden
position, adds the channel and time encodings the encoder itself uses, reads the visible states
through one block of the encoder's kind, and predicts the value with a linear head. This is the
asymmetric arrangement of the masked autoencoder (He et al., 2022), and the set representation
makes it cost nothing: no change to the encoder module, its signature or its export graph. Three
things follow. The backbone never meets a placeholder it will not meet at inference, the mismatch
that a `[MASK]` token in the input creates (Devlin et al., 2019; Clark et al., 2020 built a whole
objective around avoiding it). The encoder's attention pays only for the tokens it can see, a
fourfold saving at half the tokens hidden. And a channel hidden whole is, for the encoder, a window
of fewer channels — the robustness to a missing sensor that is wanted later comes with the
objective. The decoder is discarded after pretraining; the backbone is the encoder, as
ADR-0017 defines it.

**The decoder borrows the encoder's channel embedding and time encoding.** The question the
decoder asks — what does channel `c` show at time `t` — is spelled in the encoder's own vocabulary
of channels and positions. Sharing the two input modules costs no parameters and has one
consequence that matters: a channel hidden whole still trains its own embedding, through the
question the decoder asks about it, in a window where the encoder never saw it. A decoder with its
own tables would leave that channel's embedding untouched by the very windows that test it.

**Three draws, with their rates as the primitives.** Over every window, independently: each
channel present is hidden whole with probability `channel_rate`; each channel loses one block of
its time — a span of `block_span` of the window's length, placed uniformly inside it — with
probability `block_rate`; each token is hidden on its own with probability `token_rate`. A token is
hidden if any draw reaches it. The rates are what an ablation turns and what a vectorised draw
needs: a channel-level draw is one uniform number per window and channel, looked up per token
through its identifier, so a batch of any width is masked in a handful of tensor operations and a
channel is hidden whole or not at all. The share of a window they hide together is derived —
`1 − (1 − c)(1 − b·s)(1 − t)`, exact for channels of equal density — stated by the strategy and
measured on data rather than promised. Blocks are spans of time, not runs of tokens: how many
tokens a block swallows follows from how densely the channel is sampled there, which is the fact
the model is not allowed to assume. A timeless token has no time and is never inside a
block; it is hidden whole or singly like any other. Every window keeps at least one visible token,
because a window with nothing visible asks the model to predict from nothing.

**`MaskingStrategy` is a value object of the domain.** Four numbers with invariants — rates in the
unit interval, a span inside the window, something hidden, something left visible — and the
derived share. It is what a run is configured and reproduced by, alongside the architecture, and
it needs no torch. The decoder's depth is an argument of the torch objective, like dropout: it
changes nothing about the backbone and the run configuration owns it.

**The loss is the mean squared error over the hidden tokens that were observed, and nothing
else.** The rule is applied on the prediction, by the loss, whatever the masks claim: padding is
excluded there even when a hand-built mask marks it, so no producer of masks can lower the loss.
The mean runs over the scored tokens of the whole batch, not per window, so a window that hid
three tokens does not weigh as much as one that hid three hundred. The target is the normalised
value; the gap is not reconstructed, because the positions of hidden tokens are given to the
decoder — the mask hides values, not the sampling pattern, exactly as the interpolation baseline
knows where the hidden tokens are. Squared error rather than a robust or a distributional loss,
because the baselines it is compared with are least-squares estimators: the comparison is between
two answers to one question in one unit.

**The kind of a hidden token is read off the outcome, not the draw.** A hidden token whose
channel keeps no visible token in the window cannot be interpolated, whatever hid it — a channel
of three tokens emptied by the single-token draw is, for every purpose that matters, hidden
whole. So the channel kind means "no visible token of this channel remains", the block kind means
"hidden by the block draw, with the channel still visible somewhere", and the token kind is the
rest; each token has one kind.

**One trivial baseline per kind, scored by the objective's own loss over the same tokens.** For a
block or a single token: linear interpolation between the channel's visible neighbours, the end
value carried past the last one. For a channel hidden whole: one ridge regression per channel on
the values the other channels show at their instants nearest to the token, fitted on the training
windows with nothing hidden, the token's own column zero so a channel never explains itself, one
untuned penalty on everything but the bias. Both produce a prediction over the same batch under
the same masks as the model, and the same loss scores all three, so the comparison cannot drift.
Channels the verdict would flatter are tallied apart: a constant channel is recovered by every
method, and a timeless channel has no neighbours in time. The verdict is per kind: a kind whose
baseline does as well as the model taught nothing the baseline did not know.

**The spectrum is a least-squares fit on whole cycles per window, with a trend.** Every channel
hidden whole is fitted on sines and cosines at one to `K` cycles per window, plus a constant and a
linear trend, once for the truth and once for the residual; the share of the truth's energy the
residual no longer holds, per frequency, is what the model gives back. Whole cycles, because a
half-wave over the window is nearly a constant and a basis holding both trades them off in the
thousands — measured at a condition number near 27 000 before the change. The trend absorbs what is
slower than a cycle, which would otherwise leak into every frequency. The fit is on the tokens'
own instants, so it serves the irregular layout as it serves the regular one, at the price that
the frequencies are not orthogonal there and the shares are approximate. How many cycles a corpus
can be fitted up to follows from how many tokens a channel holds in a window.

**The training loop that answers the ticket is a report script, not the training runtime.** The
loss has to fall and the diagnostics have to be run on a trained model before the objective is
believed, and the runtime that trains a backbone for real — checkpoints, resumption, a tracker,
a precision policy — is the next ticket's. A plain loop of Adam over the control corpus, in
`scripts/masked_reconstruction_report.py`, is the smallest thing that produces the evidence, and
it is superseded rather than extended. The report publishes the corpus through the same use cases
the command line runs, so a failure anywhere on the road is the report's failure too.

## Consequences

- The single test that would catch a pipeline learning nothing is in `tests/ml`: one batch of control
  windows of both layouts, one fixed draw of masks, a small encoder and 150 steps of Adam drive the
  loss from about one to below 0.01. The masks are held fixed on purpose — that a fixed set of
  hidden values can be recovered from the visible ones is the claim; a target that moves every step
  is a different, harder one.
- The objective's tests hold the rules stated above: what the prediction says at visible and
  padding positions does not count, padding claimed hidden is still not scored, nothing hidden
  scores zero and backpropagates, the mean runs over tokens and not windows; the encoder never
  sees the value of a hidden token (scrambling the hidden values leaves every prediction unchanged
  to reduction precision); the prediction for a hidden token depends on which channel and when;
  every parameter of encoder and decoder receives a gradient and the padding row none; the
  backbone is the encoder alone.
- The masking is held to its arithmetic on a regular grid: each draw alone hides at its own rate,
  the three together hide the derived share within two points, a channel is hidden whole or kept,
  a block is one span of one channel's time of the declared length, a timeless token is never in
  a block, padding is never hidden, every window keeps a visible token, the kinds partition the
  hidden tokens, and the same seed draws the same masks.
- The baselines and the diagnostics have known answers of their own, checked without training: a
  linear channel is interpolated exactly and a curved one is not; a channel that is a linear
  combination of the others is recovered by the ridge to four decimals and the coefficients read
  off; the features are the nearest visible instants; an oracle is trivial nowhere and a model
  that is its baseline is trivial everywhere; each kind is scored against its own baseline;
  channels apart get rows of their own; a perfect prediction recovers every frequency, a smooth
  one the slow component only, and a trend is not a frequency.
- Measured on the control in `docs/verification/masked-reconstruction.md`, reproducible with
  `scripts/masked_reconstruction_report.py`. The Windows leg trains an encoder cut down from
  tier S, because the tier's shape costs seconds a step on that CPU; the M1 leg trains the tier.
  The known answers came out as written above. On the dense layout the validation loss fell from
  0.37 to 0.032 in twelve epochs, against 0.47 for interpolation and 0.24 for the ridge over the
  same hidden tokens; blocks were learnt by more than ten times over interpolation and whole
  channels by five times over the ridge, and single tokens were trivial — 0.0167 against
  interpolation's 0.0135. On the sparse, irregular layout the loss fell threefold to 0.21 against
  baselines above 0.7, every kind was learnt, single tokens narrowly. The truth's energy sits at
  one cycle per window on the control by construction, and the model gives back 94 % of it on the
  dense layout and 61 % on the sparse; the rest is at the noise floor, which no method recovers.
- The single-token draw is therefore the ablation's first question, not a default to defend: on a
  densely sampled corpus it teaches nothing interpolation does not know, and it stays in the
  mixture as the minority ingredient only until the ablation of the mixture says otherwise. The strategy the first full pretraining uses is a decision of its configuration.
- The spectral diagnostic is uninformative on the control beyond the first cycle, because the
  control's factors are slower than its window by design. It is in place for the corpora whose
  windows hold faster structure, and the report says at which frequencies the truth holds energy
  before it says what was recovered there.
- The diagnostics are tests as well as figures: the report's verdicts are code, exercised
  from both sides in `tests/scripts`, so a regression fails a build rather than a reading.
- `pretraining/adapters` gains two packages: `objective/` (the masks, the sampler, the decoder,
  the objective, the loss) and `diagnostics/` (the two baselines, the triviality tally, the
  spectrum, and the host-side arrays they work from). No import-linter contract changed: the
  wildcards over `*.domain` and `*.adapters` cover them, and Pretraining still has no contracts of
  its own.

## Alternatives considered

- **A `[MASK]` placeholder in the encoder's input** (BERT-style; TST, Zerveas et al., 2021, feeds
  zeros at hidden positions): rejected — it changes the encoder's signature or the token
  representation, shows the backbone a token it never sees at inference, and spends the encoder's
  attention on tokens it cannot see.
- **A decoder with its own channel and time modules**, as the masked autoencoder gives its decoder
  its own positional embedding: rejected — a channel hidden whole would then leave its own
  embedding untouched by the windows that test it.
- **Recomputing the gap feature over the visible tokens**: rejected — the gap is a fact of the
  data the tokeniser states (ADR-0011), the mask hides values and not the sampling pattern, and
  recomputing it here would move part of the tokeniser into the objective (ADR-0012).
- **The overall share and the kinds' proportions as the primitives**, with a budgeted draw per
  window: rejected — the draw becomes sequential and the proportions approximate anyway once a
  channel holds more tokens than its budget. The rates are exact; the share is derived and
  measured.
- **Blocks as runs of tokens**, as in TST's geometric spans: rejected — a run of tokens assumes the
  step between them means something, which is the assumption the project refuses.
- **A robust or distributional loss** (Huber; a Gaussian head): the second is the calibration
  ticket's business and the first would put the model and its baselines in different units.
  Both are one line to swap later.
- **Attributing kinds by the draw**: rejected — a channel emptied by another draw is not
  interpolable, and the baseline would be wrong for it.
- **Half-cycles in the spectral basis**: rejected on measurement — nearly collinear with the
  constant.
- **A discrete Fourier transform for the spectrum**: rejected — it needs a grid, and half the
  point of the diagnostic is to run on the irregular layout.
- **Deferring the trained-model evidence to the training runtime**: rejected — the ticket's
  definition of done is that the loss falls and the diagnostic is negative, and a plain loop in a
  report answers it without pre-empting the runtime's design.

## Revisit when

- The ablation of the mixture runs. Then the single-token rate is the first dial, and the
  proportions the first full pretraining used are compared against the strategy that wins.
- A corpus with slowly sampled channels arrives, where a block of half the window swallows one
  token. Then `block_span` is a per-corpus setting rather than a constant of the report.
- The decoder's depth or width turns out to matter. Then it is a shape of its own, a value object
  beside the architecture, rather than an argument.
- A second objective enters, such as the contrastive variant. Then the objective becomes a port
  the runtime is given, and this record's module is one adapter of it.
- The diagnostics are wanted on a real corpus with constant channels. Then the channels apart come
  from the manifest's statistics, as the report already does, and the tally says so in its rows.

## Sources

- He, K. et al. (2022). Masked Autoencoders Are Scalable Vision Learners. CVPR.
- Devlin, J. et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language
  Understanding. NAACL.
- Clark, K. et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than
  Generators. ICLR.
- Zerveas, G. et al. (2021). A Transformer-based Framework for Multivariate Time Series
  Representation Learning. KDD.
- Li, Z. et al. (2023). Ti-MAE: Self-Supervised Masked Time Series Autoencoders. arXiv:2301.08871.
- Dong, J. et al. (2023). SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling.
  NeurIPS.
- Hoerl, A. E. and Kennard, R. W. (1970). Ridge Regression: Biased Estimation for Nonorthogonal
  Problems. Technometrics 12(1), 55–67.
- VanderPlas, J. T. (2018). Understanding the Lomb–Scargle Periodogram. ApJS 236(1), 16.
