# ADR-0023: Runs are tracked through a port with an MLflow adapter, logged epoch by epoch while they run

- Status: accepted
- Date: 2026-09-15

## Context

A run of the pretraining objective produces a curve, and the curve is worth more while the run is
alive than after it: the sessions the published results are trained on drop without warning, and a
process that only writes its numbers on the way out writes nothing at all when it is killed. The
report already stores a finished run as CSV and assesses it against an earlier one, but that store
is written once, at the end, by the machinery that draws the figures.

The local stack has served an MLflow server since the environment ticket, against the same Postgres
and the same bucket the rest of the system uses. Nothing was logging to it.

## Decision

**`ExperimentTracker` is a port of Pretraining with three calls in order**: the configuration once,
each epoch as it finishes, the outcome at the end. One tracker records one run, so nothing has to
carry a run identifier through calls that already know which run they belong to, and a run that has
ended cannot quietly gain another epoch.

**A run that was begun and never ended is an interrupted run, not an error.** Nothing is written
after the fact to tidy it up: the tracked run holds the epochs it managed, which is the truth about
it.

**MLflow is the adapter, used through its client rather than its module-level API.** A global
active run is process state, and a process that trains two things would have to remember which one
it is logging to. The experiment is the configuration's name, so runs of one configuration line up
and runs of different ones do not; the configuration's `parameters()` are the run's parameters
(ADR-0022); the losses, the share hidden and the seconds are metrics stepped by epoch; the corpus
and the tier are tags; and the latest checkpoint is a tag too, because it changes as the run goes
on and is what a dropped session is resumed from.

**In memory is the other adapter**, and it is production code, not a test double: a smoke run, a
report on a laptop and the ML tests all want the epochs as they arrive without a server being up.

**The dependency is its own extra.** `mlflow-skinny` under `tracking`, because only a process that
trains logs runs, and the API and the inference worker have no use for its stack.

**The CSV store stays where it is.** It holds what the assessment reads and what the figures are
drawn from — per-unit tallies, the spectrum, the drawn windows — which is the report's shape and
not a tracker's. The two are separate on purpose: the tracker follows a run, the store holds a
result.

## Consequences

- A session that drops leaves its curve behind, up to the last epoch it finished.
- The report logs to memory by default and to a server when told (`--track`), so nothing needs
  Docker to run and everything can be compared in one place when it is up.
- Tests run against a SQLite file: the same client and the same store the served instance uses,
  without the network. MLflow migrates a fresh database before its first write, which costs more
  than the tests it serves, so one database is shared across the session.
- MLflow's file backend — the `mlruns` directory — is in maintenance mode as of version 3 and
  raises unless explicitly enabled, so "no server" means a database file, not a directory.

## Alternatives considered

**Weights & Biases.** Better live views, and the hosted tier is free for this scale. Rejected:
it is a hosted service by default, the runs of this project include the data they were trained on
by reference, and the project's stated environment is one that runs on a laptop and in a container
without an account.

**CSV only, with the store extended to be written per epoch.** No dependency and no server. It is
what the tracker's in-memory adapter plus the existing store nearly is, and it was rejected as the
only answer because a run on a GPU platform writes to a filesystem that disappears with the
session, and because comparing runs then means writing the comparison tooling MLflow already is.

**TensorBoard.** Ubiquitous and cheap to write. Rejected: it is a view of scalars, not a record of
what a run was configured to do, and the parameters are the half of this that matters — the curve
is already in the CSV store.

## Revisit when

- A second process starts tracking runs: the tracker is process-scoped by design and a worker will
  want one per task, which the composition root has to make explicit.
- Artifacts other than references need logging: today the weights live in the artifact store and
  the tracker records the reference. Logging the artifact to MLflow as well would put the same
  bytes in two places under two lifecycles, and the rule that a reference is the artifact's
  identity (ADR-0006) is what keeps that from happening.
