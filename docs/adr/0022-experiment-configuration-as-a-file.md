# ADR-0022: An experiment is a file — every parameter of scale written down, the shape from its tier, the value objects deciding what is legal

- Status: accepted
- Date: 2026-09-15

## Context

Scale is a parameter of this project and method is not: a smaller model, a tenth of the corpus or
three seeds instead of five is a legitimate result as long as it is declared and reported with the
number. Until now the scale of a run lived in the flags of a report script and in constants beside
its code — the masking rates, the decoder depth, the warmup, the peak rate — with the epochs in a
TOML file of their own because a measurement had to survive a change to the code. Reproducing a
run therefore meant remembering a command line, and two runs of "the same" configuration differed
wherever a default had moved.

The training runtime (ADR-0021) needs the same description from two directions: to run the
experiment, and to decide whether a checkpoint belongs to it.

## Decision

**One file per experiment, in `experiments/`, and it states everything.** The corpus it reads, the
tier it declares, the precision, the dropout, the decoder depth, the masking rates, the budget
(epochs, batch, accumulation, peak rate, warmup, floor, seed) and how often to checkpoint. A run
is reproduced by running the file, not by remembering flags.

**The shape comes from the tier, and the file may override it field by field.** `compute_tiers.toml`
already states what each tier's model is; an experiment names a tier and inherits it. A machine
slower than the tier assumes runs a smaller encoder by writing the difference into its own file —
which keeps the cut-down leg of a measurement as reproducible as the full one, instead of living
in a command line that no note records.

**The file is read by an adapter, and the value objects decide what is legal.** `ExperimentFile`
is a pydantic model in `pretraining/adapters/experiments/`: it reads TOML, refuses keys nobody
reads, and hands the numbers to `ExperimentConfiguration`, `TrainingBudget`, `MaskingStrategy` and
`CheckpointPolicy`. Ranges are stated once, in the domain, where they are also enforced for a
configuration built in a test.

**The configuration flattens to scalars, once, in the domain.** `parameters()` is the one
rendering: the tracker logs it as the run's parameters and the run signature digests it. Two
configurations that differ anywhere differ in that mapping, which is what makes the signature
honest.

**The device is not in the file.** Precision is a property of the method and belongs to the
experiment; the device is a property of the machine, and the same tier is MPS on one laptop and the
CPU on another. The runtime takes it from the machine or from its caller, and the stored run
records which one it was.

## Consequences

- `scripts/masked_reconstruction_epochs.toml` is gone: the epochs a corpus was measured for are in
  its experiment file, beside everything else that decides the run.
- The report's flags shrink to what belongs to the machine and to the report: the device, the
  tracking URI, where to write, a corpus cut short for a smoke run, and an epochs override that is
  recorded with the run.
- An experiment that asks for something impossible — a warmup with no epoch left to decay over, a
  strategy that hides nothing — fails when the file is read, before anything is published.
- `corpus_budget.toml` continues to hold estimates of a budget rather than the configuration of a
  run; keeping the two in step is a task for the ticket that revises the tier shapes.

## Alternatives considered

**Flags only, as before.** Nothing new to learn and nothing new to keep in step. Rejected: a run
described by a command line is a run nobody can repeat six months later, and the note that reports
it has to restate every default to be readable.

**pydantic-settings and the environment.** The configuration machinery already exists and is used
for the process's settings. Rejected because an experiment is not a property of an environment: two
experiments run in one process, and the file is an artifact to be committed, diffed and cited,
which environment variables are not.

**Hydra.** Composition, sweeps and overrides out of the box, which the ablation tickets would use.
Rejected for now: it takes over the entry point and the working directory, and this project's
entry points are ordinary classes with one use case behind them. The sweep it would buy is a loop
over files today; if the matrix grows to where that stops being true, this is the record to revisit.

**One file with every experiment in it.** Fewer files, and a diff shows the difference between two
runs directly. Rejected: an experiment is cited by name in a note and in a model card, and a file
per experiment is what makes that citation a path.

## Revisit when

- The transfer matrix turns a dozen experiments into a hundred: composition (a base file and the
  differences) starts paying for itself, and Hydra deserves a second look.
- A run's corpus has to be pinned as tightly as its configuration: the file names a corpus, while
  the published artifact it should name is a reference and a checksum. That belongs to the ticket
  where Pretraining reads a published corpus.
