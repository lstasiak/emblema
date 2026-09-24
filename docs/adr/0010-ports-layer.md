# ADR-0010: Ports are a layer of the bounded context — Protocols only, between domain and application

- Status: accepted
- Date: 2026-09-11
- Full text before condensation: commit `5f14447`

## Context

The initial rules placed a context's ports inside `domain/`, following the DDD view that a
repository interface belongs to the model. The first two ports (`CorpusReader`,
`CorpusRepository`) showed that nothing in the domain calls a port — the model is immutable
(ADR-0003), and the use case stores what an aggregate returns — and that a `domain/` holding
aggregates, value objects and interfaces read as a bag.

## Decision

- **A `ports/` layer per context, between `domain` and `application`.** The dependency rule is
  `domain ← ports ← application ← adapters ← entrypoints`, enforced by the import-linter contract
  `hexagonal-layers`. A context reads as the hexagon: `contracts` is what it publishes, `ports` what
  it requires, `adapters` what supplies it. The layout mirrors `shared/{kernel,ports,adapters}`.
- **Only the inside owns ports, and only `application` uses them.** Adapters implement ports and
  never define them. If a domain service ever needs a port, this record is revisited.
- **Driving ports are the use cases.** A use case is a class with dependencies in the constructor
  and one `__call__(command)`, so it satisfies `Callable[[Command], Result]`. Commands are frozen,
  keyword-only `<UseCase>Command` dataclasses in the use case's module.
- **`ports/` holds Protocols and nothing else**, one per module. The types a port speaks live in
  `domain/`; its failures are domain exceptions in `domain/exceptions.py`, so one hierarchy serves
  one API error handler. `tests/architecture/test_ports_are_protocols.py` fails on any module that
  defines anything but exactly one `Protocol`.
- **The `shared/` asymmetry stays** (ADR-0006): `shared/` has no domain, so its port exceptions and
  `Retention` live in `shared/ports`.
- **Naming**: a verb for the method, a noun of the context for the result, never confusable —
  `CorpusReader.describe() -> CorpusDescription`, not `read() -> CorpusReading`.

## Consequences

- Port contract tests live in `tests/<context>/ports/`.
- Four linter contracts include `emblema.*.ports`, each with a deliberate-violation test.
- A new context adds `ports/` with its first port; the Protocol-only test picks it up.
- mypy strict applies to `ports` as to `domain` and `application`.

## Alternatives considered

- *Ports inside `domain/`*: claims a dependency that does not exist here.
- *`application/ports/`* (Clean Architecture gateways): hides the boundary a level down, and
  nothing would stop a port importing a use case.
- *A `ports/` package in every layer*: two of three would be empty.
- *`infrastructure/` instead of `adapters/`*: "adapters" pairs with "ports"; sub-packages per
  technology say what is inside.
- *Exceptions and enums in `ports/`*: a context has a domain to hold them.
