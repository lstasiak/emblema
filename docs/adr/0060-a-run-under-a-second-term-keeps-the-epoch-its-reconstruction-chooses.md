# ADR-0060: A run under a second term keeps the epoch its reconstruction chooses

- Status: accepted
- Date: 2026-10-22

## Context

A run keeps the weights of its best epoch, read as the mean over corpora of each corpus's
validation loss over the trivial predictor's (ADR-0029). Under a term beside masked
reconstruction — the contrast of ADR-0058, the summaries of ADR-0059 — that loss was the sum of
both terms, and so was its trivial cost: ADR-0058 kept "the lowest combined loss over trivial".

A term run is read against a run without the term, trained on the same mixture, seed and
schedule. With the combined loss the two runs keep their epochs by different rules, so a
difference between them may be the term or the epoch it chose. At one epoch the choice is empty;
a run of eight epochs on a GPU picks among eight. A term whose weight moves its share of the
combined loss also moves which epoch is kept, so a sweep over the weight confounds the two.

## Decision

**Validation reads the hidden tokens' own term alone.** Every objective of several terms names
that term `reconstruction`; the runtime scores each corpus's held-out side on it and on its
trivial cost, and the other terms are logged beside it, per corpus, as before. An objective of
one term is scored on its whole loss, which is the same number. The training loss, the gradient
and the term's weight are unchanged.

## Consequences

- A run with a term and a run without it keep their epochs by the same rule, and their
  validation, logged and tracked as `relative_validation`, compares directly.
- A run under a single term — every run before the contrastive term, and every value or teacher
  run — scores the same numbers as before; a test holds the term runs' validation to their
  logged reconstruction.
- Under a term, the training loss is still the loss stepped on, both terms, while validation is
  the reconstruction alone: the gap between them is no generalisation gap, and the log names each
  term beside both. A term's held-out cost is in the log, not among the tracked metrics.
- An objective of several terms that leaves `reconstruction` unnamed is refused at its first
  batch, so the rule cannot lapse silently.
- A contrastive run of several epochs, retrained, may keep another epoch than the one recorded.
  Its recorded `relative_validation` is the combined loss, not comparable with a run's from
  here on. The contrastive term is rejected and no reading depends on either.
- A term that harms reconstruction while it helps the representation still competes with it
  for the kept epoch; the representation is read downstream, not here.

## Alternatives considered

- **Keep the combined loss.** Rejected: the selection rule then differs between the runs a
  reading compares.
- **Keep the last epoch.** Gives the same rule to both runs but drops the protection against a
  late epoch that overfits, which every other run keeps.
