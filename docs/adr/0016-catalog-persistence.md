# ADR-0016: The Catalog persists its aggregate through record classes, migrated by one Alembic tree

- Status: accepted
- Date: 2026-09-12

## Context

Publishing a corpus registers it, freezes a version of its data and tokenises that version into an
artifact. Until now nothing of the registration outlived the process: the repository was a
dictionary, so a second publication of the same data minted a second version identifier and a
second manifest for one block. The manifest is the reference a training run pins and the thing the
handoff adapter checks an artifact against by version, so the first consumer that needs a
registration to survive a process is the next ticket, not a later stage.

The plan settles the shape of persistence but assigns it to no ticket: one PostgreSQL database,
one schema per bounded context, no foreign key across schemas, configurations as JSONB, checksums
as first-class columns, one Alembic tree with the schema stated on every table. The aggregate it
has to store is an immutable value (ADR-0003): a `Corpus` with a tuple of `CorpusVersion`s, every
operation returning a new instance, the repository storing whole states.

## Decision

**A persistence model beside the domain model.** `CorpusRecord` and `CorpusVersionRecord` in
`catalog/adapters/persistence/` are SQLAlchemy declarative classes over one `Base`, whose metadata
fixes the `catalog` schema and one naming convention for constraints. The aggregate stays a value;
a record is how it is laid in rows and rebuilt from them, and `from_corpus` / `to_corpus` on the
record classes are the whole mapping. Neither model knows the other's shape beyond that.

**One read, one write, both whole.** `get` and `find_by_name` load a record with its versions
(eagerly, in number order) and rebuild the aggregate. `save` merges the record of the state the
caller holds inside one transaction: the row is inserted or updated by identity, each version
likewise, and a version the aggregate no longer holds is an orphan the relationship deletes. What
is stored afterwards is exactly the state the caller held, which is the contract the in-memory
adapter already kept. The uniqueness of names is a constraint of the schema; the adapter turns a
violation of that constraint, and only that one, into the domain's `CorpusNameTakenError`.

**Columns follow the data model.** The channel schema is JSONB, a small set of records whose shape
changes more often than it is queried. The checksum is two columns, algorithm and digest, because
other contexts pin a version by it. `frozen_at` is `timestamptz`; the database answers in the
session's time zone and the record normalises to UTC when it rebuilds the value object, keeping
the offset-zero rule at the boundary that produced the value. A check constraint states what the
aggregate already enforces: a frozen version has content.

**One migration tree, at the repository root.** `alembic.ini` and `migrations/` hold every
context's migrations; `env.py` connects the way the application does, from `Settings`, and lists
every context's metadata so that a comparison covers the whole database. The first migration
creates the Catalog schema if it is missing, so it stands without the local stack's init script.
An integration test compares the migrated database with the model and expects no difference; CI
migrates before it tests.

**Nothing about the connection has a default.** The application reads host, port, database name,
role and password from `EMBLEMA_DATABASE__*`; compose creates the database and its role from the
same variables and hands them to MLflow's backend store, as it already maps the artifact key onto
Garage. A default host would make a process reach a database nobody named.

**The process registers only what is missing.** The repository gained `find_by_name`, the
aggregate `frozen_version_describing`, and `PublishCorpus` uses both: a corpus registered under the
name is reused, a frozen version over exactly the data the reader sees is reused, and only what is
absent is registered. Publishing the same data under the same name and configuration twice yields
the same manifest. `RegisterCorpusVersion` keeps refusing data already frozen: a caller who asks
for a new version and gets an old one has been misled, so the reuse is the process's decision.

## Consequences

- The publishing process connects to the database by default; the command line fails on startup
  without database settings. Tests override the repository with the in-memory adapter and never
  open a connection.
- The repository contract runs against both adapters; the database leg, the migration comparison
  and the two-process publication test are marked `integration` and need the local stack, which
  the CI job starts and migrates. The Windows development machine cannot run them.
- Publishing twice gives one block and one manifest, the promise the DoD's identical checksum
  makes for the artifact a run pins.
- Three dependencies join the core set: SQLAlchemy, Alembic, psycopg. The import-linter contract
  that keeps them out of the domain, ports, application and contracts was already in place.

## Alternatives

**Core tables declared at module level, Core statements in the repository.** The first shape.
Rejected after review: two `Table(...)` calls and a metadata object read like a migration script,
and the mapping between rows and value objects had nowhere to live but the repository, which grew
to carry it. Record classes name the persistence model and own the mapping.

**Mapping the domain dataclasses themselves.** Rejected: the identity map and change tracking
exist for mutable objects that live across a unit of work; a frozen dataclass replaced wholesale
would fight the mapper's assumptions or need mutable mirrors anyway. The mirrors are the records,
declared openly.

**SQLModel for the record classes.** One class as table model and pydantic schema, with less
column boilerplate. Not adopted: its gain is a table model that doubles as an API schema, which is
the coupling the read-model rule keeps out; the columns that matter here (JSONB, `timestamptz`, a
check constraint, a naming convention) fall back to SQLAlchemy `Column` objects anyway; and it
tracks SQLAlchemy 2.0's typing at a distance, on a pre-1.0 API. Kept as an idea to weigh again when
a second context brings tables — against SQLAlchemy's own `MappedAsDataclass` and `Annotated`
column aliases, which remove the same boilerplate without a new dependency.

**SQLite for development and tests.** Rejected: no schemas, no JSONB, and a database in tests that
is not the database in deployment, the reasoning that kept testcontainers out in favour of the
compose stack.

**A file-backed repository (JSON).** Rejected: a third adapter with its own consistency problems,
that this one would replace anyway.

**Migrations inside the package.** Rejected: the package is the application, migrations are its
deployment; one tree at the root is what `alembic` finds unaided.

**Idempotency inside `RegisterCorpusVersion`.** One line, and a caller registering a new version
over unchanged data would silently get the old one. Rejected: the error is deliberate, and the
process is the place that knows it is republishing.

## Revisit when

- A second context gets tables. The engine is then shared between contexts and the question of a
  unit of work spanning a use case arises; today one `save` is one transaction and no use case
  writes to two aggregates.
- A use case must publish an event and persist state atomically: the outbox threshold of
  ADR-0005, and the first table in the tree not owned by a context's aggregate.
- The API needs queries the repository does not have, such as listing or paging. Those are read
  models over the same tables, not methods on the repository.

## 2026-09-16 — the second context with tables

Pretraining persists its backbone (ADR-0024) in the same shape: record classes over a declarative
base for the `pretraining` schema, one more migration in the tree, the configuration as JSONB with
the columns that are queried beside it. The two contexts share one engine from the settings and
still write one aggregate per transaction, so no unit of work spans a use case yet. SQLModel and
`MappedAsDataclass` were weighed again and not adopted — the reasoning is in ADR-0024 — and the
one thing the two persistence models repeated, the naming convention, moved to
`shared/adapters/persistence/naming.py`.
