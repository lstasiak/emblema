# ADR-0005: Context integration — published contracts, shared vocabulary and in-process events

- Status: accepted
- Date: 2026-09-08

## Context

Four bounded contexts share one repository and one process. They must talk to each other
(Catalog → Pretraining and Evaluation by published language, Pretraining → Evaluation through an
anti-corruption layer, Evaluation → Serving by an event) without one context importing another's
interior. Import-linter has enforced from the first commit that a context may import only another context's
`contracts` package and that `contracts` depends on `emblema.shared` alone. The first real contract,
the Catalog's reference to a frozen corpus version, exposed three questions the rules did not
answer: where an identifier that both the domain and the published language use lives, what an
event is and how it travels, and where a consumer translates a foreign contract into its own model.

## Decision

**`contracts` holds two kinds of things.** *Vocabulary* is what the domain model and the
published language must agree on: identifiers of the entities other contexts reference
(`catalog/contracts/identifiers.py`, `CorpusVersionId`), and later closed enumerations whose
semantics are identical on both sides. The domain imports vocabulary from its own `contracts`.
*Messages* are references and events (`CorpusVersionRef`, `CorpusVersionFrozen`); the application
layer builds them from the domain and the domain never sees them. The builder is a named
assembler, one per published message family (`CorpusVersionRefAssembler` in
`catalog/application`): the upstream side of a published-language relationship, translating the
context's own model outwards. It is not an anti-corruption layer, which is the downstream side's
defence. Event metadata (`event_id`, `occurred_at`) is stamped by the use case, which owns the
`Clock` and `IdGenerator`, not by the assembler. Import-linter enforces the split per context
(`domain-shares-only-identity-with-contracts`): a domain may import from its own `contracts`
nothing but `identifiers`. Identifiers that are not published (`CorpusId`) stay in the domain.

**A message that travels through the artifact store carries its wire form with it.** The
description of a published corpus (`PublishedCorpusManifest`) is read by the Pretraining
and Evaluation contexts from the store, not from a call, so the JSON it is stored as is part of
the contract: `catalog/contracts/published_corpus_manifest_json.py` encodes and decodes it with the standard
library and nothing else, and a consumer needs no module outside `contracts` to read a manifest.
Its assembler (`PublishedCorpusManifestAssembler`) translates both ways, because the Catalog also
reads its own manifests back; the reverse direction is still the provider's translation of its own
message, not an anti-corruption layer.

**Messages validate their own shape.** A reference rejects an empty or unsorted channel list and
duplicate names; a channel rejects a blank name. The producer's mapper always satisfies these
rules; the checks protect consumers that build references in their own tests and guarantee that
two references to the same data compare equal.

**Events.** `emblema.shared.events.DomainEvent` is a frozen, keyword-only dataclass carrying
`event_id: EventId` and `occurred_at: UtcDateTime`, stamped by the publishing use case from the
injected `IdGenerator` and `Clock`. Concrete events subclass it inside the publishing context's
`contracts`. Two ports in `emblema.shared.ports` split the mechanism by who depends on it:
`EventPublisher.publish(event)` is what a use case receives; `EventSubscriber.subscribe(type,
handler)` is what the composition root and handler-registering modules receive. A handler
subscribed to a type receives every event of that type or a subtype, in subscription order.
Adapters map one-to-one onto ports: `InMemoryEventSubscriber` is the registry that decides which
handlers an event goes to; `InMemoryEventPublisher` takes that registry and invokes the handlers
synchronously, in process, letting a failing handler propagate to the publisher and stop the
dispatch. They share the registry object because in-process delivery has no broker to hold it.

**Anti-corruption layer placement.** An ACL implements a port of the consuming context or handles
a foreign event, so it is an adapter of the consumer: `<context>/adapters/acl/<upstream>.py`
(`serving/adapters/acl/evaluation.py` for the `CampaignCompleted` handler feeding
`PromotableArtifact`). One named module per upstream context, never mapping functions scattered
across adapters. Published-language relationships (Catalog → Pretraining) need no ACL: the
consumer uses `CorpusVersionRef` as is. No ACL exists yet; this record fixes its place so the
first one does not have to.

## Consequences

- Importing `emblema.catalog.contracts.*` loads no module of `catalog.domain`, `.application` or
  `.adapters`; `tests/architecture/test_context_isolation.py` proves it in a fresh interpreter.
- Consumers import contract modules, never the bare `contracts` package: the independence
  exemption covers submodules only, and `__init__` re-exports nothing.
- The per-context linter entry must be extended when Pretraining, Evaluation and Serving get a
  domain; a wildcard cannot tell a context's own `contracts` from another's. The meta-test on
  violations keeps the list honest.
- A use case publishes after the repository saved the new aggregate state; the immutable
  aggregates (ADR-0003) do not collect events. Forgetting to publish is a visible omission in the
  use case and a failing application test, not a silently lost event.
- **Threshold for a second adapter.** In-process delivery is adequate while every subscriber's
  work may fail the publishing use case and finish inside it. The moment a subscriber must outlive the
  publishing transaction, run in another process (Celery worker) or survive a crash between save
  and publish, the second adapter is an outbox: events stored with the aggregate in the same
  transaction and relayed afterwards. The port shape does not change; `event_id` exists so that
  at-least-once delivery is safe for subscribers.

## Alternatives considered

- **Two identifier classes** (domain `CorpusVersionId` and a contract-side copy, mapped by the
  application). Orthodox for distributed systems where the contract is a schema and each side
  wraps the identifier. Rejected: in one process the duplicate buys no isolation, two classes of
  one name in one context confuse readers, and per-class equality of `EntityId` makes a domain
  identifier unequal to the contract identifier of the same version, a trap in tests.
- **Raw `UUID` in the reference.** Rejected: it breaks the kernel's promise that identifiers of
  different kinds never mix exactly at the boundary where mixing is most likely.
- **A context-level identifiers module outside the layers** (`catalog/identifiers.py`, imported
  by both domain and contracts). Rejected: consumers need the type for annotations, so `contracts`
  would have to re-export it, which is the chosen design with one more hop.
- **A single `EventBus` port.** Rejected: a use case would receive `subscribe`, which it must
  never call; the two ports name who depends on what.
- **One `InMemoryEventBus` adapter implementing both ports.** Rejected after review: it breaks
  the `<Technology><Port>` naming rule (there is no `EventBus` port), and future adapters diverge
  anyway: an outbox publisher writes rows and has no subscription side, the relay that reads them
  is another object. Two adapters sharing a registry cost one constructor argument.
- **Module-level mapping functions instead of an assembler class.** Rejected after review: a
  stateless module named after a category (`published_language`) read as a grab-bag, and the
  singular/plural pair of functions differed by one letter. A class names the responsibility and
  is injected like any other collaborator.
- **Bus interface inside `shared/events`.** Rejected for consistency: every port lives in
  `shared/ports`; `events` holds types only and sits between `ports` and `kernel` in the layer
  contract.
- **Events collected on aggregates and published by the repository.** Rejected: aggregates are
  immutable values (ADR-0003) and would have to return `(state, events)` pairs, and the events
  that cross contexts are built from the published language anyway, which only the application
  layer may do.
- **ACL inside `contracts`.** Rejected: an ACL produces the consumer's domain objects, so it must
  import the consumer's domain, which `contracts` may not.
- **The manifest codec inside `catalog/adapters`, the manifest a domain value.** The first shape
  of the published corpus. Rejected after review: the manifest exists so that the handoff adapter of the
  Pretraining context can check an artifact without fetching it, and that adapter may import
  neither `catalog/domain` nor `catalog/adapters`. A codec there would have left every consumer
  to duplicate it or break the independence contract.

## 2026-09-22 — the Evaluation side of the backbone relationship is not an ACL

This record first named `evaluation/adapters/acl/pretraining.py` as the place where Evaluation
would defend itself against Pretraining. What was built there defends against nothing: it
implements `CandidateProvider` over four ways of using one backbone, and it imports not a line of
the upstream context, because the weights reach it as an artifact reference and a checksum —
published language, used as it stands. The translation that does happen, from a stored trained
model to an encoder, lives in the process that knows both sides (`entrypoints/restored_backbones.py`,
ADR-0017), and a process is not a context.

So the provider is an ordinary adapter, `evaluation/adapters/candidates/`, and the rule above is
unchanged: an ACL is named after the upstream context it defends against, and the first one will
be the `CampaignCompleted` handler in Serving. A directory named after an upstream context that
imports nothing of it promises a seam that is not there.
