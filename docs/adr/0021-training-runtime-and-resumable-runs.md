# ADR-0021: The training loop is an adapter behind a port that reports epochs, and a run is resumable mid-epoch from state written on a step boundary

- Status: accepted
- Date: 2026-09-15
- Full text before condensation: commit `5f14447`

## Context

The training loop lived in a report script. Three things were missing: a run must survive the
session it starts in (free GPU sessions drop without warning, and an epoch there takes hours); a
run must be watchable while it runs; and where training happens must not be part of how it is
asked for — a later runtime emits a job and waits for its result from another platform.

## Decision

- **The loop is an adapter behind `TrainingRuntime`**:
  `train(configuration, corpus, resume_from) -> Iterator[EpochOutcome]`. `PretrainBackbone`
  consumes the iterator and logs each epoch to `ExperimentTracker` (ADR-0023). Nothing trains until
  the first outcome is asked for.
- **Weights never cross the port**: checkpoints and the trained model are artifact references
  (ADR-0006). Two runs ending in the same weights end in the same reference.
- **A checkpoint holds the whole state**, written only after an optimiser step: weights, optimiser
  state, position, the mask generator, the host and accelerator generators. The learning rate is
  not stored — the schedule is a pure function of the step.
- **Resume inside the epoch**: the position carries the epoch and micro-batches consumed; the loader
  is set to that epoch's order and skips them. On the host the resumed run's weights are bit for bit
  those of the uninterrupted run (`tests/ml/test_resume_matches_uninterrupted.py`).
- **A checkpoint belongs to one run**: `RunSignature` digests the configuration and the corpus
  (artifact checksum and shape). Resuming into another configuration, other data or a finished run
  is refused.
- **Everything is stored on the CPU**, so a checkpoint loads anywhere and exports (ADR-0007).
- **Precision is declared** (`TorchPrecision`): fp32 everywhere, fp16 on CUDA and MPS, bf16 on CUDA
  and host; an impossible pair is refused. Gradient scaling only where it exists (CUDA).
- **A run that stops being finite stops** (`DivergedRunError`), checked on the loss and on the
  gradients before each step (except under a scaler, which skips overflowing steps itself).
- **Adam with weight decay stated as zero**, not inherited from a default.
- **Accumulated micro-batches are summed over tokens** and divided once by the tokens the group
  hid, so batches are weighted as the loss weights tokens.

## Consequences

- A dropped session costs the steps since the last checkpoint.
- An interrupted run leaves a tracked run with its epochs and no ending.
- The report no longer trains; it reads the kept model from the store and diagnoses it.
- Two runtimes share a contract test: torch and in-memory.
- A step the scaler skips still advances the schedule (the count is of steps attempted).
- Summed accumulation scales the loss with hidden tokens; on MPS without a scaler, a divergence now
  stops the run rather than writing non-finite weights.
- On MPS runs do not repeat bit for bit; the resume there is held to a tolerance between repeat
  spread and a broken resume (`docs/verification/training-loop.md`).
- **Known defect**: a resumed run's training loss for the re-entered epoch covers only the batches
  after the checkpoint. Weights, validation and best-epoch choice are unaffected.

## Alternatives considered

- *PyTorch Lightning*: owns the program's shape, puts training policy into the model, resume is
  epoch-granular by default, and it would load wherever the model loads.
- *Hugging Face Accelerate*: assumes a loader and script it drives; its value is distributed
  training, which this project does not do.
- *Loop in the use case*: a port per tensor operation, or torch in the application.
- *Returning a final result*: nothing to watch while a session runs.
- *Epoch-granular resume*: an epoch on the platform is hours.

## Revisit when

- Multi-device training → position, generators and averaging per rank.
- fp16 on MPS overflows under summed accumulation → divide per micro-batch instead.
- An ablation varies the objective, encoder or optimiser → a builder passed to the runtime.
- A schedule with state beyond the step (plateau) → its state goes into the checkpoint.
- Checkpoint size dominates a session → store less for runs that only need finishing.
