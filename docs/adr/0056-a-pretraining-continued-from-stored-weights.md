# ADR-0056: A pretraining continued from stored weights — the start and the trained parameters as knobs of the experiment file

- Status: accepted (2026-10-17)
- Date: 2026-10-17

## Context

Every pretraining so far drew its weights from its seed. Two readings of the stage now need
a run that starts elsewhere. The diagnostic of `docs/verification/zero-shot-pretext.md`
(2026-10-17) fitted the channel rows of a corpus a backbone never read, every other weight
frozen, on that corpus's unlabelled windows, and found the pretext restored to within 0.006
of a backbone that had read the corpus. That fit lived in a script and left no backbone: the
evaluation could not read what the rows carry to a task. The next question, whether a broad
backbone continued on the task's own corpora reads the task better, needs the same start with
everything free.

A checkpoint resumes the same run; it cannot start another. The evaluation's modes train a
backbone's weights on labels under the task's head; a pretext on unlabelled windows is the
pretraining context's job.

## Decision

- **Two knobs of the experiment file**, both fields of the configuration and so of its
  parameters, its signature, the registry's document and the order handed to the machine that
  trains: `[start]`, the stored model the run continues, named by its key and checksum as the
  evaluation names a backbone; and `trained`, which parameters move — `everything`, the
  default, or `channel_rows`, the channel table alone. A file that states neither is the run
  it always was, and renders as it did before the knobs existed.
- **The run draws its objective from the seed and then carries the stored weights into it.**
  Every tensor is copied whole but the channel table, which is copied row by row as far as
  both tables reach. The run's table is its own vocabulary's, the mixture's: a row the stored
  model never had keeps what the seed drew, and a stored row past the run's vocabulary is left
  behind. So a run over a corpus a backbone never read gets drawn rows for that corpus's
  channels, and one over a corpus from the middle of the backbone's chain reads under the
  shorter table, as every backbone reads under its mixture's.
- **Under `channel_rows` the optimiser is handed the channel table and nothing else.** A row
  of a channel the run's windows never carry receives no gradient and, under Adam without
  weight decay, stays exactly where it started, so over a corpus the backbone never read this
  fits that corpus's rows alone — the fit of the diagnostic, which masked the other rows'
  gradients, by another route. One step of each over the same windows and masks agrees to
  rounding (the test in `tests/scripts/test_pretext_triviality_report.py`); later steps draw
  masks by each one's own protocol.
- **Ordering traces and refuses.** The order finds the ready backbone holding the stored
  weights and registers it as the run's parent, a column of the registry; it refuses weights no
  ready backbone holds, a backbone of another shape, decoder or objective, and corpora whose
  vocabulary does not lie on the parent's chain of publications, since a row is carried by its
  identifier alone. The machine that trains refuses the shape mismatch from the stored model
  itself, so a platform without a registry stops before it trains.
- **A control without a start.** `trained = "channel_rows"` with no `[start]` fits the rows
  on an encoder the seed drew: the control that tells a backbone's machinery from its rows.

## Consequences

- A backbone adapted without labels to a corpus it never read is a registered backbone with
  provenance, and the evaluation reads it as it reads any other. Domain-adaptive continuation
  of the broad backbone is the same knob with everything free.
- The parent's initialisation is not a start the file can name; a continued backbone refuses
  the diagnostic's reading at initialisation and points at its parent.
- The diagnostic's fit and the registered run differ in how masks are drawn after the first
  step and in which windows they read, not in the arithmetic of a step. The diagnostic keeps
  its protocol, so its readings before this record reproduce.
- A stored model that predicts states is continued with its teacher copied; nothing here
  reads that case.

## Alternatives considered

- Naming the parent by its registry identifier in the file: a configuration that only a
  registry can resolve, and a signature over an identifier rather than over the data.
- A table of the larger of the two vocabularies: a backbone whose stored table disagrees with
  the vocabulary and parameter count the registry holds for it.
- Making the diagnostic run the pretraining: the pretraining keeps the best epoch by the
  validation it would then be scored on, and the diagnostic's readings would no longer
  reproduce.
