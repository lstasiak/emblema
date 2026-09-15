# ADR-0021: The training loop is an adapter behind a port that reports epochs, and a run is resumable mid-epoch from state written on a step boundary

- Status: accepted
- Date: 2026-09-15

## Context

The objective trains (ADR-0019) and what a run measured is judged by rules of its own (ADR-0020),
but until now the loop that connected them lived in a report script: it built the model, stepped
the optimiser, scored the validation windows and kept everything in memory. Three things that the
project needs were therefore missing.

A run has to survive the session it starts in. The published results are trained on a free GPU
platform whose sessions drop without warning, and an epoch at that scale is measured in hours. A
run that cannot be picked up is a run that has to be repeated.

A run has to be watchable while it runs, not only readable after it. A curve that appears when the
process exits is no use to a session that never exits.

And where the training happens must not be part of how it is asked for. The same run is meant to
be started on a laptop for a smoke test and on rented hardware for a result, and a later ticket
adds a runtime that does not train at all — it emits a fully specified job, waits for the result to
come back from a platform outside the system, and registers it.

The immediate consumer is the masked-reconstruction report, which trains three corpora, diagnoses
what was learnt, and draws figures.

## Decision

**The loop is an adapter behind `TrainingRuntime`, and the port reports one outcome per epoch.**
`train(configuration, corpus, resume_from) -> Iterator[EpochOutcome]` — the iterator is the seam.
The application (`PretrainBackbone`) consumes it, logs each epoch to `ExperimentTracker` as it
arrives (ADR-0023), and assembles the outcome; the adapter keeps everything that touches torch.
Nothing is trained until the first outcome is asked for, and a caller that stops consuming stops
the run.

**Weights never cross the port.** A checkpoint and the trained model are references into the
artifact store, so a runtime that trains elsewhere reports what a local one reports. The store is
content-addressed (ADR-0006), which gives a property worth having: two runs that end in the same
weights end in the same reference.

**A checkpoint carries the whole state of the run, and is written on a step boundary.** Weights,
optimiser state, the position in the run, the generator the masks are drawn from, the host's
global generator and the accelerator's own. The learning rate is not stored: the schedule is a
pure function of the step, so restoring the step restores the rate. Checkpoints are written only
after an optimiser step, when no gradient is pending, so a resumed run never double-counts a
partial group of micro-batches.

**A resumed run picks up inside the epoch it stopped in.** The position carries the epoch and the
micro-batches consumed in it; the resumed run sets the loader to that epoch's order and skips
exactly those micro-batches. That, with the restored generators, makes the resumed run the run it
resumed: on the host the weights come out bit for bit identical, which is what
`tests/ml/test_resume_matches_uninterrupted.py` asserts, and equality of the content-addressed
reference is the same claim stated once more.

**A checkpoint belongs to one run and says so.** `RunSignature` digests everything the
configuration states and what the corpus is: the checksum of the artifact its windows were read
out of, and the shape beside it, because a corpus read in part is not the corpus its checksum
names. Resuming into another configuration is refused rather than silently producing a model that
reports parameters it did not train under; so is resuming on other data of the same shape, which
the counts alone would have let through; and so is resuming a run that has finished.

**Everything is stored on the CPU.** A tensor saved from an accelerator loads only where that
accelerator exists, and a model kept on MPS cannot be exported at all (ADR-0007). The device a run
happened on stays out of what the run leaves behind.

**Precision is declared, and an unsupported pair is refused.** `TorchPrecision` writes out which
precisions each backend runs — single everywhere, half on CUDA and MPS, bfloat16 on CUDA and the
host — and raises where a pair is impossible. Gradient scaling follows the same rule: it is CUDA's
answer to half precision's underflow, so a run that cannot scale says so by not scaling rather
than by pretending to. A run never reports a precision it did not use.

**The optimiser is Adam at the configured rate and nothing implicit.** Weight decay is stated as
zero rather than inherited from a default, because a regularisation this project never chose
should not be one a reader has to look up; the day it varies it becomes a field of the experiment,
where every other parameter of scale already is (ADR-0022).

**Accumulated micro-batches are summed over tokens, not averaged per batch.** The objective's
error is a mean over hidden tokens; a group of micro-batches therefore accumulates the summed
error and divides the gradients once, by the tokens the group hid. Averaging the per-batch means
would weigh a batch that hid three tokens like one that hid three hundred — the very rule the loss
applies within a batch, broken across them.

## Consequences

- A dropped session costs the steps since the last checkpoint, and nothing else.
- An interrupted run leaves a tracked run with the epochs it managed and no ending, which is what
  an interrupted run is.
- The report no longer trains: it assembles the use case, reads the model back out of the store,
  and diagnoses that. What is diagnosed is therefore what was kept.
- Two runtimes exist against the port — the torch one and an in-memory one that reports the shape
  of a run without the arithmetic — so the contract is a shared test rather than a description.
- A step the gradient scaler skips, because half precision overflowed, still advances the learning
  rate schedule: the schedule is a function of the step count, and the count is of steps attempted.
  At the rate a scaler settles to, that is a handful of steps in a run.
- The summed accumulation makes the loss magnitude that reaches `backward` proportional to the
  tokens a batch hid. Under CUDA half precision the scaler absorbs it by construction; on MPS,
  where no scaler exists, a large batch in half precision is the case to watch.

## Alternatives considered

**PyTorch Lightning.** It provides the loop, checkpointing, resume, mixed precision and
accumulation, and would have been less code. Rejected for three reasons. It owns the shape of the
program: a `LightningModule` is where the model, the optimiser and the step live together, which
puts training policy back inside the model and makes the objective's own tests harder to write
against plain modules. Its resume is epoch-granular by default and its mid-epoch support needs the
loader to cooperate, so the property this ticket exists for would still be ours to verify. And it
is a dependency that must load wherever the model loads, including the inference container.

**Hugging Face Accelerate.** Lighter than Lightning and it solves precision and devices well. It
assumes a `DataLoader` it wraps and a training script it drives, and what we need wrapped is a port
whose other adapter does not train at all. Its value is concentrated in distributed training, which
this project does not do at any tier.

**A loop in the use case, with torch behind narrow ports.** Keeps the application in charge of the
epochs, at the cost of a port for every tensor operation the loop performs — or of torch in the
application layer, which the architecture rules forbid and for good reason: the domain and the
application are where the method lives, and the method is not framework-shaped.

**Returning a final result instead of an iterator.** Simpler port, and the tracker would move into
the adapter. Rejected: the application would then learn what happened only when everything had
happened, and the curve that is watched while a session runs is the one that matters.

**Epoch-granular resume.** Simpler state, and enough for a laptop. Rejected because it is not
enough for the machine the published results are trained on: an epoch there is hours, and losing
one is losing the session the checkpoint was for.

## Revisit when

- A runtime trains on more than one device at a time: the position, the generators and the
  gradient-averaging rule are written for one process and would need the same care per rank.
- Half precision on MPS is measured on a real batch and the loss magnitude under summed
  accumulation turns out to overflow: the fallback is to divide each micro-batch by the group's
  size, at the cost of the weighting this record argues for.
- A scheduler arrives whose state is not a function of the step — a plateau schedule reading the
  validation loss, say. The rate is restored here by rebuilding the schedule at the step the run
  reached, which is exact for a schedule that is a pure function of it (measured: the rebuilt
  scheduler matches the stepped one on `last_epoch`, `base_lrs`, `_last_lr` and the rate itself,
  differing only in the counter that drives a torch warning). A stateful schedule would need its
  own state in the checkpoint, and its metric history with it.
- A checkpoint's size becomes the cost that dominates a session: the optimiser's state is two
  thirds of it, and a run that only needs to be finished, not compared, could store less.
