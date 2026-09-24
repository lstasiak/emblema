# ADR-0016: The Catalog persists its aggregate through record classes, migrated by one Alembic tree

- Status: accepted
- Date: 2026-09-12; amended 2026-09-16, 2026-09-18, 2026-09-24
- Full text before condensation: commit `5f14447`

## Context

Until now a registration did not outlive the process: the repository was a dictionary, so
publishing the same data twice minted a second version and a second manifest for one block. The
manifest is what a training run pins. Persistence shape: one PostgreSQL database, a schema per
context, no cross-schema foreign keys, configurations as JSONB, checksums as columns, one Alembic
tree. The aggregate is an immutable value (ADR-0003).

## Decision

- **A persistence model beside the domain model.** `CorpusRecord` and `CorpusVersionRecord` in
  `catalog/adapters/persistence/` are SQLAlchemy declarative classes over one `Base` (schema
  `catalog`, one naming convention). `from_corpus` / `to_corpus` are the whole mapping.
- **One read, one write, both whole.** `get` / `find_by_name` load a record with its versions
  eagerly and rebuild the aggregate. `save` merges the whole state in one transaction; orphaned
  versions are deleted by the relationship. A violation of the name-uniqueness constraint, and only
  that one, becomes `CorpusNameTakenError`.
- **Columns**: channel schema as JSONB; checksum as algorithm and digest; `frozen_at` as
  `timestamptz`, normalised to UTC on rebuild; a check constraint that a frozen version has content.
- **One migration tree at the root** (`alembic.ini`, `migrations/`). `env.py` connects from
  `Settings` and lists every context's metadata. Each context's first migration creates its schema.
  An integration test expects no difference between the migrated database and the model.
- **No connection default**: `EMBLEMA_DATABASE__{HOST,PORT,NAME,USER,PASSWORD}`; compose creates the
  database and role from the same variables.
- **The process registers only what is missing.** `PublishCorpus` reuses a corpus by name and a
  frozen version over exactly the same data, so publishing twice gives the same manifest.
  `RegisterCorpusVersion` still refuses data already frozen; reuse is the process's decision.

## Consequences

- The publishing process needs database settings; tests override the repository in memory.
- The repository contract runs against both adapters; database legs are `integration` tests on the
  compose stack.
- SQLAlchemy, Alembic and psycopg join the dependencies; the linter keeps them out of the core.

## Alternatives considered

- *Core tables at module level*: read like a migration script; the mapping had no home.
- *Mapping the frozen domain dataclasses*: fights the identity map and change tracking.
- *SQLModel*: couples table models to API schemas; JSONB and constraints fall back to SQLAlchemy.
- *SQLite for tests*: no schemas, no JSONB, not the deployed database.
- *Idempotency inside `RegisterCorpusVersion`*: a caller asking for a new version would silently
  get an old one.

## Revisit when

- A use case writes two aggregates → a unit of work.
- A use case must publish and persist atomically → the outbox (ADR-0005).
- The API needs listing or paging → read models over the tables, not repository methods.

## Amendments

- **2026-09-16** — Pretraining persists its backbone the same way (ADR-0024). The shared naming
  convention moved to `shared/adapters/persistence/naming.py`.
- **2026-09-18** — integration tests use their own database, the configured name plus `_test`.
  A test run had truncated the development registry between a handoff order and its result.
  `tests/support/database.py` sets the name before `Settings()` is built, creates the database if
  missing, and refuses to truncate one without the suffix.
- **2026-09-24** — Serving's tables (ADR-0037, migration 0006): every context now has tables. The
  first cross-row rule is a partial unique index (at most one active served model per artifact);
  Alembic does not compare index predicates, so the adapter's contract test holds it. UTC
  normalisation moved to `shared/adapters/persistence/datetimes.py`, which refuses a naive value.
