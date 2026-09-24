# ADR-0022: An experiment is a file — every parameter of scale written down, the shape from its tier, the value objects deciding what is legal

- Status: accepted
- Date: 2026-09-15; amended 2026-09-16
- Full text before condensation: commit `5f14447`

## Context

Scale is a parameter of this project and method is not: a smaller model or fewer seeds is a
legitimate result if declared and reported with the number. Scale had lived in a report script's
flags and constants, so reproducing a run meant remembering a command line. The training runtime
(ADR-0021) needs the same description to run an experiment and to decide whether a checkpoint
belongs to it.

## Decision

- **One file per experiment in `experiments/`**, stating everything: corpus, tier, precision,
  dropout, decoder depth, masking rates, budget (epochs, batch, accumulation, peak rate, warmup,
  floor, seed), checkpoint frequency. A run is reproduced by running the file.
- **The shape comes from the tier** (`compute_tiers.toml`); the file may override it field by field,
  so a cut-down leg is as reproducible as the full one.
- **An adapter reads the file; domain values decide what is legal.** `ExperimentFile` (pydantic, in
  `pretraining/adapters/experiments/`) refuses unknown keys and builds `ExperimentConfiguration`,
  `TrainingBudget`, `MaskingStrategy` and `CheckpointPolicy`, where ranges are enforced.
- **One flattening**: `parameters()` in the domain is what the tracker logs and what the run
  signature digests, so any difference changes the signature.
- **The device is not in the file.** Precision belongs to the method; the device to the machine,
  and the stored run records it.

## Consequences

- The report's flags shrink to machine and report concerns (device, tracking URI, output, smoke
  cut, recorded epochs override).
- An impossible experiment fails when the file is read, before anything is published.
- `corpus_budget.toml` keeps estimates; experiment files keep run configuration.

## Alternatives considered

- *Flags only*: nobody can repeat the run six months later.
- *pydantic-settings and the environment*: an experiment is a committed, cited artifact, not an
  environment property.
- *Hydra*: takes over the entry point and working directory; a loop over files suffices today.
- *One file for all experiments*: an experiment is cited by path.

## Revisit when

- A hundred experiments → composition (base plus differences), and Hydra again.

## Amendments

- **2026-09-16** — the corpus is pinned by the order, not the file (ADR-0024): the file names the
  corpus; the order carries the manifest's reference, checksum and the run signature over the
  windows read.
