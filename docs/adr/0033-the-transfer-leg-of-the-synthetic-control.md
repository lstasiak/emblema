# ADR-0033: The transfer leg of the synthetic control — a forecasting task whose truth is generated, one ground-truth port for every task, and the generator's specification shared by both sides

- Status: accepted
- Date: 2026-09-20; amended 2026-09-21 (grown channel rows)
- Full text before condensation: commit `5f14447`

## Context

The synthetic control makes a negative result readable (ADR-0018): a coupled pair must transfer,
an uncoupled pair must not. The first label-efficiency grid was negative on the endpoint
(ADR-0030), so the control's transfer leg now decides what that says about the encoder. Evaluation
knew one kind of task (remaining life from a failure time); a generated corpus has no failures;
and the generator lived in the Catalog's adapters, which Evaluation may not import.

## Decision

- **The task forecasts one sensor's exact reading a fixed time past the window.**
  `ForecastScheme(channel, horizon)`: the target is the noiseless value of the named sensor at
  `ends_at + horizon`, in its own units. Both tasks name the first sensor of each pair's second
  layout, twelve steps ahead (half the shortest factor period). A forecast needs the frequencies
  behind the signal — what the coupled pair shares — and is not the reconstruction pretext. The same
  task on the null pair is as learnable from labels; only pretraining is uninformative there.
- **One ground-truth port for every task, keyed by window**: `GroundTruth.truths_of(windows)`
  replaces `UnitLifetimes.failure_times(units)`. The aggregate holds a closed set,
  `RemainingLifeScheme | ForecastScheme`; each scheme turns truths into targets and states the unit
  targets are learnt in. The revisit ADR-0026 named.
- **The generator's specification moves to `shared/adapters/synthetic`**; the reader stays in the
  Catalog. `SensorSignal` is the noiseless model: the reader adds noise, gaps and rounding; the
  ground truth reads the same signal at any instant. Corpus bytes unchanged (pinned checksums).
- **A generated corpus's frozen side is a third of its held-out units**, ranked under a seed at task
  definition.

## Consequences

- Reports take a task by name; shards of two tasks are refused as one curve. Remaining-life
  readings are blank for a forecast; the endpoint is RMSE under either scheme.
- The forecasting task travels the whole road on miniature control corpora in a test.
- A classification task enters through the same port with a third scheme.
- Budgets, units, seeds and the equivalence floor are registered in `docs/preregistration.md`.

## Alternatives considered

- *The hidden factor as the label*: unlearnable on the null pair, so equivalence would hold
  trivially.
- *The reading at the window's end*: too close to the pretext.
- *A second port per scheme*: two ports for one question.
- *A function handed over by the composition root*: keeps the letter of the boundary, breaks its
  sense.
- *Duplicating the signal model*: the corpus and its truth could quietly disagree.

## Revisit when

- A truth is not one number per window → the port returns a value object.
- Campaigns own the task registry → the frozen share becomes a task field.

## Amendments

- **2026-09-21 — a backbone grows rows for the channels a task's corpus adds.** The first pilot cell
  failed with an index past the channel table: the second corpus of a pair continues the first's
  vocabulary, and a backbone pretrained on the first has no rows for it (the end-to-end test's
  stand-in had been sized to the larger vocabulary). `SetEncoder.grown_to(vocabulary_size)` keeps
  learnt rows and draws new ones (`GrownChannelEmbedding`); `pretrained(...)` and `fresh(...)` take
  the task's vocabulary size, so the control's parameter count equals full fine-tuning's. **Grown
  rows train under every mode**, like the head — a frozen probe over random identities for the
  task's own sensors would measure nothing. They are marked by `GrownParameters`, a Protocol in
  `shared/adapters/tensors`. Publishing the pairs in reverse order was rejected: it hides a general
  need behind an ordering. No number had been measured when this changed.
