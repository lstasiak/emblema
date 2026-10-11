# ADR-0054: Predicting the teacher's states at the hidden tokens — a second objective beside masked reconstruction, chosen by the experiment file

- Status: proposed (accepted or rejected by the reading registered for it, `docs/preregistration.md`, "The pretext")
- Date: 2026-10-11

## Context

Every backbone so far was pretrained by masked reconstruction (ADR-0019), which named the failure
it guards against — a hidden value in a dense channel is answered by a line through its
neighbours, which teaches interpolation — and deferred a second objective. Three readings since
point at that failure rather than at the amount of data:

- On FD001 the closed-form probe reads the remaining life worse from every pretrained backbone than
  from an untrained encoder, by 3.9 to 5.9 RMSE in today's shape, at every layer
  ([`layer-readout.md`](../verification/layer-readout.md)).
- On 21 to 22 of the 25 UTSD datasets a single hidden token is answered better by a line through
  its neighbours than by the model ([`intensive-care-curve.md`](../verification/intensive-care-curve.md)).
- Two variants of the masks replaced nothing; the forecast tail unlearnt the stays
  ([`pretext-variants.md`](../verification/pretext-variants.md)).

A different target, not a different mask, is the lever left untried.

## Decision

- **The target is a state, not a value.** A teacher — a copy of the encoder that sees the whole
  window — gives every token a state; the predictor answers it at the hidden tokens. A line
  through the neighbours' values does not answer it (Baevski et al. 2022, data2vec; Assran et
  al. 2023, I-JEPA).
- **The teacher trails the encoder.** After every optimiser step its weights move toward the
  encoder's by an exponential moving average, the momentum rising linearly from 0.996 to one over
  the run (Grill et al. 2020). It takes no gradient, trains no dropout and sees no mask; it moves
  on skipped steps too, as the schedule counts them, so a resumed run takes the same path.
- **The target is the mean of the teacher's top four of six blocks, each normalised without
  parameters.** The encoder's own normalisation is a moving average of the student's, so a scale
  drifting to zero would turn every target into one constant; a normalisation without parameters
  cannot be pulled there. The embedding layer is never a target: it is the value projected.
- **The encoder meets the hidden tokens as before**, as padding. The predictor is the decoder's
  shape with a head as wide as the state. The loss is the run's reading per coordinate, averaged
  over the width and summed over the hidden observed tokens, so a token weighs what it weighed.
- **The trivial predictor is the mean state of the window's visible tokens**, as the channel mean
  is for a value. The kept epoch is still the lowest loss over trivial. A trivial loss of zero —
  targets collapsed onto one point — stops the run with its cause.
- **The objective is a knob of the experiment file**: `[objective] target = "latent"` with
  `target_layers`, `momentum` and `final_momentum`; `value` is the default and renders nothing
  new, so every signature, document and registry row written before is unchanged. The artefact
  carries the target; bytes written before it existed read as the value objective.
- **Not a port**, as ADR-0019 expected. An objective speaks in tensors, which no interface of the
  application could, and the run is already behind `TrainingRuntime`. Both objectives subclass
  `PretrainingObjective`, an abstract module in the adapter; `ObjectiveShape` knows both.
- **A diagnostic of its own.** `scripts/latent_prediction_report.py` reads the predictor against
  the line between the neighbours' states and the nearest neighbour's, per kind of mask with
  intervals over units, and reports the targets' variance and effective rank, where a collapse
  shows.

## Consequences

- The teacher's forward pass adds about thirty per cent to a step and doubles the artefact; the
  checkpoint carries the teacher inside the model's state, so a resume is exact on the host.
- Evaluation is untouched: the backbone is the student encoder, read as any other.
- The objective is read against the mixture of four by the registered rule, pretrained twice and
  compared as a difference of means over pretrainings. This record decides how the second
  objective is built, not whether it replaces the first.

## Alternatives considered

- **A contrastive objective** (Yue et al. 2022, TS2Vec): its hierarchy of poolings along time
  assumes a grid, and it needs augmentations and negatives the set encoder has no natural form for.
- **A block in time across every channel**: measured already as the forecast tail.
- **Reconstruction from the other channels alone**: whole channels are learnt today and the probe
  gains nothing from them.
- **The teacher as the backbone** (Caron et al. 2021): deferred, so the comparison changes one
  thing.
- **The encoder's own detached states as targets**: nothing then holds the states apart; the
  moving average does.
