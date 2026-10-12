# ADR-0054: Predicting the teacher's states at the hidden tokens — a second objective beside masked reconstruction, chosen by the experiment file

- Status: accepted as the design of the second objective; by the registered reading it does not replace masked reconstruction
- Date: 2026-10-11
- 2026-10-11: read ([`pretext-variants.md`](../verification/pretext-variants.md)). The objective reads the stays worse than the mixture's masks by 0.03 at 50 stays, past the spread of pretraining; the mixture's masks stay the curve's pretext.
- 2026-10-12: read again (same note). The states concentrate to about half the mixture's own effective rank, not a collapse, and lose their channel and time identity; the pretext is barely learnt on the stays. Weaknesses for a successor: the target carries the embedding, so at initialisation it is mostly the channel the predictor is handed; the kept epoch is chosen over a baseline that moves with the teacher; the collapse check compared with epoch 1, after the targets had narrowed.

## Context

Every backbone so far was pretrained by masked reconstruction (ADR-0019), which deferred a second
objective. Three readings since point at the objective rather than at the amount of data: on FD001 the closed-form probe reads the remaining life worse from
every pretrained backbone than from an untrained encoder
([`layer-readout.md`](../verification/layer-readout.md)); on 21 to 22 of the 25 UTSD datasets a
single hidden token is answered better by a line through its neighbours than by the model
([`intensive-care-curve.md`](../verification/intensive-care-curve.md)); two variants of the masks
replaced nothing ([`pretext-variants.md`](../verification/pretext-variants.md)). A different
target, not a different mask, is the lever left untried.

## Decision

- **The target is a state, not a value.** A teacher — a copy of the encoder that sees the whole
  window — gives every token a state; the predictor answers it at the hidden tokens (Baevski et
  al. 2022, data2vec; Assran et al. 2023, I-JEPA).
- **The teacher trails the encoder.** After every optimiser step its weights move toward the
  encoder's by an exponential moving average, the momentum rising linearly from 0.996 to one over
  the run (Grill et al. 2020). It takes no gradient, trains no dropout, sees no mask, and moves on
  skipped steps too, so a resume takes the same path.
- **The target is the mean of the teacher's top four of six blocks, each normalised without
  parameters.** The encoder's own normalisation is a moving average of the student's, so a scale
  drifting to zero would turn every target into one constant. The embedding layer is never a
  target on its own: it is the value projected.
- **The encoder meets the hidden tokens as before**, as padding. The predictor is the decoder's
  shape with a head as wide as the state. The loss is the run's reading per coordinate, averaged
  over the width and summed over the hidden observed tokens.
- **The trivial predictor is the mean state of the window's visible tokens.** The kept epoch is
  still the lowest loss over trivial. A trivial loss of zero stops the run with its cause.
- **The objective is a knob of the experiment file**: `[objective] target = "latent"` with
  `target_layers`, `momentum` and `final_momentum`; `value` is the default and renders nothing
  new, so everything written before is unchanged. The artefact
  carries the target; bytes written before it existed read as the value objective.
- **Not a port**, as ADR-0019 expected: an objective speaks in tensors, and the run is already
  behind `TrainingRuntime`. Both objectives subclass `PretrainingObjective`, an abstract module in
  the adapter; `ObjectiveShape` knows both.
- **Diagnostics of its own.** `scripts/latent_prediction_report.py` reads the predictor against
  the neighbours' states per kind of mask, and the targets' variance and effective rank;
  `scripts/state_spread_report.py` reads that spread, with its channel and time share, off any
  backbone, so a concentration is held against the other objective's.

## Consequences

- The teacher's forward pass adds about a quarter to a step and doubles the artefact; the
  checkpoint carries the teacher, so a resume is exact.
- Evaluation is untouched: the backbone is the student encoder, read as any other.
- The objective is read against the mixture of four by the registered rule, pretrained twice and
  compared as a difference of means over pretrainings. This record decides how the second
  objective is built, not whether it replaces the first.

## Alternatives considered

- **A contrastive objective** (Yue et al. 2022, TS2Vec): its poolings along time assume a grid,
  and it needs augmentations and negatives the set encoder has no natural form for.
- **A block in time across every channel**: measured as the forecast tail.
- **Reconstruction from the other channels alone**: whole channels are learnt today without gain
  for the probe.
- **The teacher as the backbone** (Caron et al. 2021): deferred, so one thing changes.
- **The encoder's own detached states as targets**: nothing then holds the states apart.
