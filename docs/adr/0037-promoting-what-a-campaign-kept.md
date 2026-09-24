# ADR-0037: Promoting what a campaign kept — a projection fed by the campaign's announcement, a served model per period of service, and a manual relay for a delivery that failed

- Status: accepted
- Date: 2026-09-24
- Full text before condensation: commit `5f14447`

## Context

Serving keeps one claim: nothing is served that a finished campaign did not measure — a fitted
tree model as much as an adapted backbone. The facts live in Evaluation, and the schema-per-context
rule forbids a foreign key from `serving` to `evaluation`. The contract is one event,
`CampaignCompleted`, published in process by whichever process closes the campaign.
`CompleteCampaign` stores first and publishes second; if a subscriber fails in between, the
campaign is closed, the message lost, and closing cannot be repeated.

## Decision

- **A projection fed by the event is the only source of what may be promoted.**
  `PromotableArtifact` records origin (campaign, task, competitor), kind, artifact reference and
  checksum, standing, scores per operating point and finish time. `serving/adapters/acl/evaluation.py`
  is the only module reading Evaluation's messages; it drops competitors with no kept artifact.
  `ServedModel.promoted(projection, …)` is the only constructor, so the invariant holds by
  construction. Scores become Serving's own `CampaignScore`.
- **Promotion names an artifact by checksum** and trusts nothing else in the request. When the same
  bytes were kept more than once, the request must name the origin. The store is asked whether the
  bytes exist; it verifies content on every read.
- **Standing and scores are recorded, never enforced.** Ranking is the verdict's job.
- **A served model is one period of service**; withdrawal is final; promoting again creates a new
  model. At most one active model per artifact — a partial unique index in the database, the same
  rule in memory.
- **A failed delivery is repaired by announcing again.** `AnnounceCampaign` rebuilds the event from
  the stored closed campaign with the closing assembler, dated at closing;
  `campaign announce --campaign <id>` runs it and prints every kept checksum. The projection is
  keyed by origin, so repetition is safe. The handler is registered in `CampaignProcess`.

## Consequences

- Either kind of kept artifact is promotable on the same terms; anything outside a finished
  campaign is refused with a named reason. Which runtime loads an artifact is decided with the
  prediction endpoint; the kind is carried for it.
- Every process that can close a campaign writes to `serving`; a failure there fails the task
  after closing, repaired by one idempotent invocation.
- The projection copies what it needs; a served model copies its origin rather than joining.
- Serving publishes no contracts yet.

## Alternatives considered

- *A cross-schema foreign key*: couples contexts below their contract.
- *Asking Evaluation at promotion time*: promotion would depend on another context being reachable.
- *An outbox now*: closes the gap automatically, but the only subscriber runs in the publishing
  process and the repair is one idempotent command; deferred until the threshold of ADR-0005.
- *Gating on standing*: re-decides the campaign's question and would block a classical winner.
- *A per-task aggregate of the serving model*: nothing needs "one model per task" yet.
- *A state column beside `withdrawn_at`*: two fields that must agree.

## Revisit when

- A subscriber must survive the publishing transaction or run elsewhere → the outbox.
- The prediction endpoint needs "the model serving task X" → a per-task aggregate.
- An efficiency variant is served in place of its source → the served model names the variant.
