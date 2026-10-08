# ADR-0053: Reading the encoder's states at another layer — a knob of the frozen modes, every layer through the final normalisation

- Status: proposed
- Date: 2026-10-08

## Context

Every head so far read the encoder's last block. Under the mixture's backbones, the closed-form
probe gains little or nothing over the same probe on an untrained encoder at 20 and 50 stays,
and on FD001 it reads worse than the untrained encoder
([`intensive-care-curve.md`](../verification/intensive-care-curve.md)). One explanation is that
the pretext, reconstructing hidden readings, spends its last blocks on the reconstruction and
leaves what a task wants in the middle of the network. Whether that holds is a question about
the states the head reads, not about the backbone, and it can be answered on the backbones that
already exist.

The encoder (ADR-0017) is a pre-norm stack: each block adds to a residual stream, and one
`LayerNorm` normalises the stream after the last block. A state read below the top has never
been normalised.

## Decision

- **A knob of the variant grammar, `layer`**, held by `EncoderSetting`:
  - `last`: the default, what every head read before;
  - a layer's number: `0` for the tokens' embedding, `1` to the number of blocks;
  - `mean`: the mean of the blocks' states;
  - `concat`: the blocks' states side by side.

  The combinations leave the embedding out. The value is `LayerReading` in the domain.
  Campaigns name `frozen_ridge@layer=3` or `frozen_ridge@layer=concat,pooling=tail`.
- **Only where the encoder does not train.** The plan and the catalogue refuse a layer other than
  the last under the network from nothing, full fine-tuning and LoRA. Under those modes, reading
  below the top would change which blocks train, which is a different method. The closed-form
  probe, the trained probe and the probe on an untrained encoder take it.
- **Every layer passes through the encoder's final normalisation.** The last layer read by number
  is then exactly the state every head read before. The closed-form probe scales and centres
  each column, so for it this normalisation is equivalent to one without weights.
- **The encoder answers with all its layers by name.** `SetEncoder.layer_states` returns
  `[blocks + 1, batch, tokens, width]`, and its `forward` is unchanged. Evaluation may not import
  the encoder's class, so it reaches the method through a `Protocol`, `LayeredEncoder`. This is a
  named seam of the adapter, like `BackboneFactory`, not a port. A module that does not answer
  is refused. The wrapper `LayerReadout` holds no weights and sits where the encoder sat, inside
  the value clip.
- **The factory states its number of blocks** (`BackboneFactory.layers`), so the head is sized
  for a concatenation before the encoder is drawn. The head is still drawn first, as before.
- **The reading travels with the candidate**: as `encoder_layer` in the plan's parameters, in the
  campaign's description of the method only when turned, and in the inference graph. The graph
  is exported and checked against eager answers like any other.

## Consequences

- Descriptions stored before this knob are unchanged, so every campaign recorded before reads as
  it ran.
- A reading below the top computes every layer and keeps one or all. It costs the whole encoder
  as before, plus a normalisation per layer.
- Whether a multi-layer reading replaces the last layer is decided by a rule registered before
  the reading, on the tuning side, not by this record.

## Alternatives considered

- **Forward hooks on the blocks.** These need no change to the encoder, but tie Evaluation to the
  names of another context's modules and survive the export less predictably than a method.
- **A learnt weighting of the layers** (a scalar mix). It adds weights the closed-form probe
  cannot solve, and it is a second question on top of whether the layers differ at all.
- **The raw residual stream below the top.** Its scale grows with depth, so a trained probe would
  meet each layer at another scale. The closed-form probe would only lose the per-token
  normalisation every head has had.
