# ADR-0005: Context integration — published contracts, shared vocabulary and in-process events

- Status: accepted
- Date: 2026-09-08; amended 2026-09-22 and 2026-09-24 (where the anti-corruption layer is)
- Full text before condensation: commit `5f14447`

## Context

Four bounded contexts share one repository and one process. They talk to each other (Catalog →
Pretraining and Evaluation by published language, Evaluation → Serving by an event) without
importing each other's interior. Import-linter enforces that a context imports only another
context's `contracts`, and that `contracts` depends on `emblema.shared` alone. The first contract
left three questions open: where an identifier shared by domain and published language lives,
what an event is and how it travels, and where a consumer translates a foreign contract.

## Decision

- **`contracts` holds vocabulary and messages.**
  - *Vocabulary*: identifiers of entities other contexts reference (`CorpusVersionId` in
    `catalog/contracts/identifiers.py`) and closed enumerations with identical meaning on both
    sides. The domain may import from its own `contracts` nothing but `identifiers` (linter
    contract `domain-shares-only-identity-with-contracts`). Unpublished identifiers (`CorpusId`)
    stay in the domain.
  - *Messages*: references and events (`CorpusVersionRef`, `CorpusVersionFrozen`). The application
    layer builds them through a named assembler (`CorpusVersionRefAssembler`) — the upstream side
    of published language, not an anti-corruption layer. The use case stamps event metadata.
- **A message that travels through the artifact store carries its codec.** A published corpus
  manifest is read from the store, so its JSON is part of the contract:
  `published_corpus_manifest_json.py` in `catalog/contracts`, standard library only. Its assembler
  translates both ways because the Catalog reads its own manifests back.
- **Messages validate their shape** (no empty or unsorted channel list, no duplicates), so two
  references to the same data compare equal.
- **Events.** `DomainEvent` is a frozen keyword-only dataclass with `event_id` and `occurred_at`,
  stamped from injected `IdGenerator` and `Clock`. Concrete events live in the publisher's
  `contracts`. Two ports: `EventPublisher.publish(event)` for use cases,
  `EventSubscriber.subscribe(type, handler)` for composition roots. `InMemoryEventSubscriber` is
  the registry; `InMemoryEventPublisher` takes it and calls handlers synchronously, in
  subscription order; a failing handler propagates.
- **An anti-corruption layer is an adapter of the consumer**: `<context>/adapters/acl/<upstream>.py`,
  one named module per upstream context. Published language needs no ACL.
  - 2026-09-22: Evaluation's backbone provider imports nothing of Pretraining (weights arrive as an
    artifact reference and checksum), so it is an ordinary adapter in
    `evaluation/adapters/candidates/`, not an ACL. A directory named after an upstream context that
    imports nothing of it promises a seam that is not there.
  - 2026-09-24: the first ACL is `serving/adapters/acl/evaluation.py`, handling `CampaignCompleted`
    (ADR-0037). It is the first subscriber that can fail after the publisher committed; a failed
    delivery is repaired by `AnnounceCampaign`, which republishes a closed campaign's message. The
    consumer is idempotent.

## Consequences

- Importing `emblema.catalog.contracts.*` loads no module of that context's domain, application or
  adapters; `tests/architecture/test_context_isolation.py` proves it in a fresh interpreter.
- Consumers import contract modules, never the bare `contracts` package; `__init__` re-exports
  nothing.
- The per-context linter entry grows with each context's first domain module.
- A use case publishes after the repository saved; immutable aggregates (ADR-0003) collect no
  events. A missing publish is a visible omission and a failing application test.
- **Threshold for a second publisher**: when a subscriber must outlive the publishing transaction,
  run in another process, or survive a crash between save and publish, add an outbox (events
  stored with the aggregate, relayed afterwards). The ports do not change; `event_id` makes
  at-least-once delivery safe.

## Alternatives considered

- **Two identifier classes** (domain and contract copy): no isolation in one process, and equality
  per class makes the two unequal — a trap in tests.
- **Raw `UUID` in references**: identifiers of different kinds would mix at the boundary.
- **A single `EventBus` port / one in-memory bus adapter**: a use case would receive `subscribe`;
  an outbox publisher has no subscription side anyway.
- **Module-level mapping functions**: read as a grab-bag; a class names the responsibility.
- **Events collected on aggregates**: aggregates are immutable values, and cross-context events are
  built from published language, which only the application layer may do.
- **ACL inside `contracts`**: an ACL imports the consumer's domain, which `contracts` may not.
- **Manifest codec in `catalog/adapters`**: consumers may not import it, so each would duplicate it.
