# ADR-0024: A run is ordered here, made anywhere, and accepted back against the order

- Status: accepted
- Date: 2026-09-16
- Full text before condensation: commit `5f14447`

## Context

Published runs happen on rented GPUs, in notebook sessions that may drop and reach nothing of this
system but the bucket. The system must still say what was trained, on which data, by which code,
and where the weights are. Missing: a backbone registry, a way to read a published corpus inside
Pretraining without the Catalog, and a way to carry a run across machines that keeps the registry
honest.

## Decision

- **Three steps**: `order` registers a `Backbone` as ordered and places an order in the artifact
  store; `run` reads the order on any machine and trains through the same use case as a local run,
  writing a result to the store; `accept` holds the result to the order, replays it through the same
  use case with a runtime that reads results, and delivers the weights. The local machine is a
  platform like any other, so the path is tested once.
- **`Backbone` is ordered before it is ready.** It carries configuration, `PretrainingInput`
  (corpus version, manifest, block checksum, vocabulary), run name, code revision, run signature,
  and after delivery the artifact. Delivery refuses a second one.
- **Orders and results are durable documents** behind `HandoffExchange` (JSON in the store;
  in memory). A result names its order by a content-addressed reference.
- **Acceptance names what differs**: another configuration (naming the parameter), other data (both
  checksums), other code (both commits), another checkpoint — each checked before the run
  signature. The signature digests the corpus's *shape* too, so a machine that never held the
  windows is held to the same digest.
- **The training machine refuses before training**: other code or another corpus stop the run
  before the first epoch; acceptance checks again.
- **The replay is the third `TrainingRuntime` adapter** and passes the port's contract through a
  simulated platform.
- **`TrainingCorpusReader`**: `describe` reads the manifest via the Catalog's published codec;
  `read` fetches the block once into the workspace under its digest, hashing a found block before
  mapping it.
- **Persistence like the Catalog's** (ADR-0016): `pretraining` schema, configuration as JSONB with
  seed, parameter count and commit as columns, `status` held to the artifact columns by a check.
- **The code revision is read, not typed**: from distribution metadata for a `git+…@<sha>` install;
  from `HEAD` for an editable install, marked `-dirty` with uncommitted changes. Orders are placed
  only from a committed tree. `--commit` overrides where it cannot be read.

## Consequences

- A registry row states experiment, run, corpus, code and weights, wherever it was trained.
- `accept` replays the run into MLflow, so a curve from a platform without a tracker lands with the
  others; the tracking URI is required there.
- The ready row names its result, from which the order, checkpoint and epochs are reachable.
- An open order is a row without an artifact; nothing deletes it.
- The pretraining CLI needs the `ml` and `tracking` extras.
- Two racing acceptances are not guarded; the process is manual.
- A resumed run's result holds the epochs after its checkpoint only; the curve is not stitched.

## Alternatives considered

- *One adapter that writes a specification, waits and accepts*: `train()` cannot write before and
  read after; registering weights is a use case.
- *Registering only finished backbones*: acceptance would compare against what the operator typed
  again, not what was ordered.
- *Documents as files carried by hand*: one more thing to get wrong.
- *Pinning the corpus in the experiment file*: the file says what; the order says on what.
- *SQLModel or `MappedAsDataclass`*: too little repetition to justify it.

## Revisit when

- A backbone reads several manifests → `PretrainingInput` becomes a tuple (done in ADR-0029).
- Partial results epoch by epoch are wanted from a platform without a tracker.
- Another context needs a backbone by identity → `BackboneId` into `pretraining/contracts`.
