# ADR-0027: Measuring corpus saturation — one budget of steps over nested shares of whole units, the share a parameter of the run, and a verdict read by rules fixed beforehand

- Status: accepted
- Date: 2026-09-16; amended 2026-09-17 (a fourth verdict)
- Full text before condensation: commit `5f14447`

## Context

ADR-0008 admitted the reference encoder on a ratio fitted to text trained for one epoch, on the
condition that a measurement corrects it: does more of a corpus still lower the loss, or has the
model learnt what the corpus holds? To be honest the measurement needs a definition of "a share of
the corpus" that does not leak, a fair basis of comparison between shares, and the share recorded
with the run.

## Decision

- **A share is a share of training units, whole, ranked by the run's seed** (`CorpusShare`), using
  the Catalog's digest framed apart from the split. Shares of one seed are nested. The validation
  side is never shared.
- **The share is a parameter of the experiment** (`corpus_fraction`), inherited from the tier and
  overridable; it enters parameters, signature, order and result. The corpus reader takes it
  (`read(manifest, share)`).
- **Every share spends the same optimiser steps**: 20 epochs over a tenth against 2 over the whole;
  warmup keeps its share of the run. Points more than 15 % apart in steps are refused.
- **One curve per corpus, at the small tier**; the mix would entangle saturation with transfer.
- **The verdict is code with thresholds fixed beforehand** (`judge`), in the order of what
  invalidates what: validation / training > 1.5 at the whole corpus → *overfitting*; last doubling
  lowers validation by < 5 % → *plateau*; otherwise → *data-limited*. The same thresholds as
  ADR-0020.
- **Each side is read against its own channel-mean predictor**: relative losses, so the satellite
  corpus's harder held-out months do not read as overfitting.
- **Runs are stored as they go and resumed** through ADR-0021's mid-epoch resume; a directory with
  an outcome is never trained again; figures come from files alone.

## Consequences

- Every configuration's signature changed (share and warmup rendered); order and result documents
  moved to version 2.
- A smoke run at the small tier reads a tenth of a real corpus by stating nothing.
- Shares repeat windows different numbers of times; the verdict holds at this budget, and a full
  run's own curve checks it.

## Alternatives considered

- *A share of windows*: overlapping windows would leak inside the training side.
- *Equal epochs per share*: measures compute, not data.
- *The fraction as a report flag*: the run's provenance would lie.
- *Independent draws per share*: adds subset variation to size.
- *A curve over the mix*: answers a transfer question.

## Revisit when

- The size axis disagrees with the small tier's curve.
- Units differ in length by an order of magnitude → rank by observations.
- The mixed run's curve is wanted → a tuple of shares.

## Amendments

- **2026-09-17 — a fourth verdict, added after the first curves.** On ESA-AD no share beat the
  channel mean on the held-out months (1.00 at best, 1.29 at the whole) while every share learnt its
  training months; the rules said *overfitting*, 3.77×, and prescribed the wrong remedy. One
  held-out month held 88 % of the side's squared magnitude. `judge` now reads **not learnt** first,
  when relative validation at the largest share is ≥ 1 — the trivial predictor's own error, not a
  tuned threshold. Declared as a post-hoc addition here and in `docs/verification/corpus-saturation.md`,
  beside what the original rules read. Tables now carry each run's best epoch beside its last.
