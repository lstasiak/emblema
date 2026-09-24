# ADR-0023: Runs are tracked through a port with an MLflow adapter, logged epoch by epoch while they run

- Status: accepted
- Date: 2026-09-15
- Full text before condensation: commit `5f14447`

## Context

A curve is worth more while the run is alive: platform sessions drop without warning, and a process
that writes its numbers on exit writes nothing when killed. The report stored finished runs as CSV,
once, at the end. The local stack already served MLflow against the shared Postgres and bucket, and
nothing logged to it.

## Decision

- **`ExperimentTracker` is a Pretraining port with three calls in order**: configuration once, each
  epoch as it finishes, the outcome at the end. One tracker records one run.
- **A run begun and never ended is an interrupted run**, not an error; nothing tidies it up.
- **MLflow is the adapter, through its client**, not the module-level API with its global active
  run. Experiment = configuration name; parameters = `parameters()` (ADR-0022); losses, share
  hidden and seconds are metrics stepped by epoch; corpus, tier and latest checkpoint are tags.
- **An in-memory adapter is production code** for smoke runs, laptop reports and ML tests.
- **The dependency is its own extra**: `mlflow-skinny` under `tracking`.
- **The CSV store stays** for what the assessment and figures read (per-unit tallies, spectra,
  windows). The tracker follows a run; the store holds a result. They are joined by the tracked name
  and the weights checksum, kept as a tag.

## Consequences

- A dropped session leaves its curve up to the last finished epoch.
- The report logs to memory by default and to a server with `--track`.
- Tests use a SQLite file with the same client and store, one database per session (MLflow's
  migration is expensive).
- MLflow 3's file backend (`mlruns`) is in maintenance mode, so "no server" means a database file.

## Alternatives considered

- *Weights & Biases*: hosted by default; the project runs on a laptop and in a container without an
  account.
- *CSV written per epoch*: a platform's filesystem disappears with the session, and comparison
  tooling would have to be written.
- *TensorBoard*: scalars without the record of configuration.

## Revisit when

- A long-lived process tracks many runs → a tracker per task, explicit in the composition root.
- Artifacts beyond references need logging → keep weights in the artifact store; the reference is
  the identity (ADR-0006).
