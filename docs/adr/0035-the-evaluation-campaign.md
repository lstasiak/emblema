# ADR-0035: The evaluation campaign — a grid of candidates declared before it runs, a verdict only once it is whole, and one narrow message out of it

- Status: accepted
- Date: 2026-09-22
- Full text before condensation: commit `5f14447`

## Context

The first label-efficiency curve came from a report script: CSV files, a shard per seed, a reader
that rebuilt the comparison. Three things could not be promised: an interrupted run left cells to
spot by hand; a curve could be read with cells missing — and missing cells correlate with what they
found (the diverging arm is the one that crashes); and budgets, seeds and rules were flags, choosable
after the numbers were visible. Serving must also learn, through an event and not a foreign key,
which artifacts a finished comparison measured.

## Decision

- **A campaign is a design plus the record of running it.** `CampaignDesign` states competitors,
  budgets, seeds, the control, the one endpoint comparison and the verdict rules — written once,
  before anything runs. The aggregate afterwards only gains cells.
- **A candidate carries what it was set to, as named strings the campaign does not interpret**
  (rate, decay, warm-up, LoRA targets, tree depth), beside the compute budget all neural arms share.
- **The grid's axis is the candidate**, not the transfer mode, so a classical method enters on the
  same terms. Candidates arrive through `CandidateProvider`: describe a candidate, answer a cell
  with an error per unit.
- **No finish while a cell is pending; no verdict before the finish** — aggregate invariants.
  Recording a cell twice, or after closing, is refused.
- **Resuming is asking what is pending.** State is database rows, not queue messages; an already
  recorded cell is answered from storage, so delivery twice is safe.
- **The artifact is the fitted candidate, kept at one cell** named by the design before anything
  ran (the endpoint's budget under the first seed), so keeping one is not a selection on results.
- **What leaves is narrow**: `CampaignCompleted` carries the campaign, task, a sentence, and per
  candidate its name, kind, kept artifact, one figure per budget and a three-valued standing
  against the control. The internal verdict taxonomy stays inside.

## Consequences

- A partial grid produces no number — the intended cost.
- A permanently failing cell means the campaign never finishes: fix the cell or design without it.
- The design is a document; results are a long table of per-unit errors.
- Concurrent consumers need an optimistic lock (added in ADR-0036).
- A provider serving other weights than the candidate records refuses the cell.
- Method parameters are held in name order and compared as text.
- A final campaign opens the frozen side once per cell; the promise is per campaign, and the record
  must be read that way.

## Revisit when

- An unsupervised detection campaign (no budget axis) → a different shape.
- A candidate fitted once and scored on several tasks.
- More than one process consumes cells at a time (done in ADR-0036).
