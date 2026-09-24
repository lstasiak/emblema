# ADR-0037: Promoting what a campaign kept — a projection fed by the campaign's announcement, a served model per period of service, and a manual relay for a delivery that failed

- Status: accepted
- Date: 2026-09-24

## Context

The claim the Serving context exists to keep is short: nothing is served that a finished campaign
did not measure. It applies to every candidate a campaign can keep, a fitted gradient-boosting
model as much as an adapted backbone. If the verdict says the trees won, the platform has to be
able to serve the trees rather than the loser.

The facts that claim depends on live in another context. Evaluation knows which campaigns
finished, what each one kept and how each competitor did. The schema-per-context rule forbids the
obvious enforcement, a foreign key from `serving` to `evaluation`. Such a key would be a channel
between the two contexts outside their published contract, and the moment it existed the
contexts could no longer be put in separate databases. The contract between them is one event,
`CampaignCompleted`, published in process by whichever process closes the campaign. That is a
worker of either pool, or the command line that advances a grid whose last cell has already run.

Publishing in process has a gap that this is the first consumer to expose. `CompleteCampaign`
stores the closed campaign first and publishes second, so nothing is announced that was not
recorded. If a subscriber fails after the store and before the delivery, the campaign is closed
and the message is lost. Closing cannot be repeated: a retried cell finds the campaign closed and
is refused.

## Decision

**A projection, fed by the event, is the only source of what may be promoted.**
`PromotableArtifact` is Serving's own copy of one kept artifact. It records the origin (campaign,
task, competitor), the kind, the artifact reference with its checksum, the standing against the
control, the scores per operating point, and the moment the campaign finished.
`serving/adapters/acl/evaluation.py` is the only module in the context that reads Evaluation's
messages. It translates each announcement into a command and drops every competitor the campaign
kept no artifact of. A `ServedModel` is created only by `ServedModel.promoted(projection, …)`, so
the invariant holds because of how the model is constructed, not because some service checks it.
Serving uses the identities and the two closed vocabularies Evaluation publishes (`CandidateKind`,
`CandidateStanding`). The scores are translated into a value object of its own
(`CampaignScore`), not kept as the other context's message type.

**Promotion names an artifact by its checksum and trusts nothing else in the request.** The
reference that gets served is the one found in the projection. When the same bytes were kept
more than once — by several campaigns, or by one campaign as two competitors — the promotion is
refused until the request names the origin, the campaign and where needed the competitor,
because a served model must state exactly one provenance. The store is asked whether the bytes
are present. It is not asked
to hash them, because it is content-addressed and verifies every read: a model whose artifact is
present will load the bytes that were measured, and one whose artifact is missing would never load.

**The standing and the scores are recorded, never enforced.** A candidate the campaign found
worse than its control can still be promoted. Ranking candidates is the verdict's job, and a gate
here would be a second verdict under rules nobody registered.

**A served model is one period of service.** Its one transition, withdrawal, is final. Promoting
the same artifact again creates a new model, so each record keeps its own start and end as the
provenance of every answer it gave. At most one model not yet withdrawn may serve a given
artifact. This rule spans many stored models, so the repository holds it: a partial unique index
over the models not yet withdrawn, whose violation the database adapter translates into the
domain's error. The in-memory adapter enforces the same rule on save.

**A delivery that failed is repaired by announcing the campaign again.** `AnnounceCampaign`
rebuilds `CampaignCompleted` from the stored closed campaign using the same assembler as closing,
dated at the moment the campaign closed. `campaign announce --campaign <id>` runs it and prints
the checksum of every artifact the campaign kept, which is what a promotion names. The projection
is keyed by origin, so a second delivery replaces the first rather than adding to it, and
repeating an announcement is safe. The handler is registered in `CampaignProcess`, the part
shared by every process that can close a campaign.

## Consequences

- A kept artifact from a campaign of either kind of candidate can be promoted on the same terms;
  one outside any finished campaign is refused with a domain error that names the reason. What can
  *run* a promoted artifact is not decided here: the kind is carried so that the inference
  runtime can choose a loader, and that runtime arrives with the prediction endpoint.
- Every process that can close a campaign now writes to the `serving` schema through the engine
  it already holds. A worker that fails there fails its task after the campaign has been closed.
  Nothing corrects that on its own; the repair is one invocation, and it is idempotent.
- The projection copies what it needs and never reads Evaluation's tables. A served model copies
  its origin rather than joining it to the projection, so it can go on saying where its artifact
  came from whatever happens to the projection later.
- Serving has no published contracts of its own: nothing outside it refers to a served model. Its
  domain therefore needs no entry in the linter's rule on sharing only identity with its own
  contracts. The entry comes with the first contract module.

## Alternatives considered

- **A foreign key from `serving.served_model` to Evaluation's tables.** The strongest guarantee
  a database can give, and the one the schema-per-context rule exists to forbid: it would couple
  the contexts below their contract and prevent separating the databases. Rejected.
- **Asking Evaluation at promotion time**, through a port Serving defines and an adapter over
  Evaluation's repository. This avoids the projection, but it makes a promotion depend on another
  context being reachable, and Serving would need a view of Evaluation's campaign that its
  published language does not give. Rejected; the projection is the pattern the event
  relationship implies.
- **An outbox** in the `evaluation` schema, written in the transaction that closes the campaign
  and relayed afterwards. It closes the gap without anyone running anything, and the port shape
  was chosen so that it can be added without changing a use case. Rejected for now: the only
  subscriber is in the same process as the publisher, and the gap is repaired by one idempotent
  invocation. A relay process and its failure modes would cost more than the problem it solves
  today.
- **Gating promotion on the standing** (refusing `WORSE`, or anything not `ESTABLISHED`). It
  guards against a mistake, but it re-decides a question the campaign has already answered, and
  it would stop the platform serving a classical winner of a campaign whose neural endpoint lost.
  Rejected.
- **A per-task aggregate holding the one model that serves the task.** This would put a "one
  model per task" invariant in the domain, but nothing needs that rule yet. The prediction
  endpoint addresses a model by identity, and designing the aggregate now would be guessing at
  that endpoint's shape. Rejected until the endpoint needs a default model.
- **A state column beside the withdrawal timestamp.** It spells out the state, but it is a second
  field that must agree with the first. Rejected: the state is derived from `withdrawn_at`.

## Revisit when

- A subscriber must survive the transaction that publishes, or runs in another process: that is
  the outbox threshold of ADR-0005, and it fires here first.
- The prediction endpoint needs "the model serving task X": introduce the per-task aggregate, and
  move the one-per-artifact rule into it if it fits there.
- An efficiency variant (quantised, distilled) has to be served in place of its source: the served
  model gains the variant it serves, and the provenance extends from the campaign to the source
  artifact.
