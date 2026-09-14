# ADR-0010: Ports are a layer of the bounded context — Protocols only, between domain and application

- Status: accepted
- Date: 2026-09-11

## Context

The project rules written at the start placed a context's ports (the `Protocol` interfaces its use
cases depend on) inside `domain/`, following the DDD stance that a repository interface is part of
the domain model. The first two context ports were `CorpusReader` and
`CorpusRepository` — and showed two things. Nothing in the domain calls a port: the model is
immutable (ADR-0003), an aggregate returns a new instance and the use case stores it, so every port
is used by `application` alone. And `domain/` with eleven modules of three different kinds
(aggregate and entity, value objects, interfaces) read as a bag. The review asked where ports
belong, whether each layer owns some, and how to keep the answer from becoming a dumping ground.

## Decision

**A `ports/` layer per context, between `domain` and `application`.** The dependency rule becomes
`domain ← ports ← application ← adapters ← entrypoints`, enforced by the `hexagonal-layers`
contract of import-linter. Ports import the domain and `shared/`, nothing above them; the domain
cannot import ports. Alongside `contracts/` the top level of a context now reads as the hexagon:
`contracts` is what the context publishes, `ports` is what it requires, `adapters` is what supplies
it. The layout mirrors `shared/{kernel,ports,adapters}`.

**Only the inside owns ports, and only `application` uses them.** Adapters implement ports and
never define them, so `adapters/ports/` is an empty category. The domain defines none by design;
should a domain service ever need one, the decision is revisited here rather than by adding a
package. Per-layer `ports/` sub-packages were rejected for that reason: two of the three would be
empty, which the project forbids.

**Driving ports are the use cases themselves.** A use case is a class with its dependencies in the
constructor and `__call__(command)` as its single operation, so it satisfies
`Callable[[Command], Result]` and can be wrapped (logging, a transaction) without a named
interface. Commands are frozen, keyword-only dataclasses named `<UseCase>Command`, kept in the use
case's module: they name the intent, they serialise to a worker, and they mirror events.

**`ports/` holds Protocols and nothing else.** One `Protocol` per module. The types a port speaks —
`CorpusDescription`, `ChannelSchema`, `SamplingRegime` — are the domain's and live in `domain/`.
The failures a port reports — `CorpusReadError`, `CorpusNotFoundError`, `CorpusNameTakenError` —
are domain exceptions in `domain/exceptions.py`, under the context's `CatalogError`, so one error
hierarchy serves one handler at the API. The rule is: an exception lives in the lowest layer whose
language it speaks. An architecture test (`tests/architecture/test_ports_are_protocols.py`) walks
`emblema.<context>.ports` and fails on any module that defines other than exactly one `Protocol`,
or any function; a bag is impossible rather than discouraged.

**The `shared/` asymmetry is kept.** `shared/ports/exceptions.py` and the `Retention` enum beside
`ArtifactStore` stay where ADR-0006 put them: `shared/` has no domain and its kernel holds value
objects only, so the ports layer is the lowest one that speaks the port's language. The
architecture test covers bounded contexts only; `shared/ports` is finite and governed by ADR-0006.

**Naming.** A port's method is a verb for what it does and its result a noun of the context's
language, chosen so the two cannot be confused: `CorpusReader.describe() -> CorpusDescription`,
not `read() -> CorpusReading`.

## Consequences

- `catalog/domain` is back to the model: aggregate, entity, five value objects, identifiers,
  exceptions. `catalog/ports` has two modules. Contract tests of ports live in
  `tests/catalog/ports/`, mirroring `tests/shared/ports/`.
- Four import-linter contracts gained `emblema.*.ports` (`hexagonal-layers`, `pure-core`,
  `config-only-at-the-edges`, `contracts-depend-only-on-shared`), and
  `domain-shares-only-identity-with-contracts` lists `emblema.catalog.ports` beside the domain.
  Each has a deliberate-violation test.
- A new context adds `ports/` with its first port and joins the identity contract with its first
  domain and ports modules; the Protocol-only test picks the package up automatically.
- mypy strict applies to `ports` as to `domain` and `application`.

## Alternatives considered

- **Ports inside `domain/`** (as before, or as `domain/ports/`). Evans's placement of repository
  interfaces; honest for a domain that calls its own services. Rejected: here the domain never
  calls a port, so the placement claimed a dependency that does not exist, and the domain package
  mixed three kinds of thing.
- **Ports inside `application/`** (`application/ports/`, the Clean Architecture gateway ring).
  Semantically exact — the application owns its boundary — and free of linter changes. Rejected
  because it hides the boundary one level down and cannot be policed: nothing would stop
  `application/ports/x.py` from importing a use case.
- **A `ports/` sub-package in every layer.** Rejected: adapters never own ports and the domain owns
  none here, leaving two packages built ahead of a tenant.
- **Renaming `adapters/` to `infrastructure/`.** Onion and Clean vocabulary; "adapters" is the
  hexagonal one and pairs with "ports". The question "what is in there" is answered by the
  sub-package per technology (`persistence/`, `readers/`, `in_memory/`, `acl/`). Rejected.
- **Exceptions and vocabulary enums in `ports/`**, as `shared/ports` does. Rejected for contexts:
  they are of a different nature from an interface, and the context has a domain to hold them.
