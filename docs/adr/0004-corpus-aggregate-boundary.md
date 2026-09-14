# ADR-0004: `Corpus` is the aggregate root; `CorpusVersion` is an entity inside it

- Status: accepted
- Date: 2026-09-08

## Context

The Data Catalog owns corpora and their versions. A version is a snapshot of the data described by
its checksum, channel schema and sampling regime; once frozen it is immutable and other contexts
(Pretraining, Evaluation) reference it by identifier and checksum. The invariants that involve more
than one version are: version numbers form the sequence 1..n; version identifiers are unique
within a corpus; no two frozen versions describe the same data (same checksum, schema and regime).

## Decision

`Corpus` is the aggregate root and the only consistency boundary. `CorpusVersion` is an entity
held in the root's `versions` tuple; it enforces its own lifecycle (draft → content recorded →
frozen) and the root enforces everything that spans versions (numbering, lookup, duplicate
detection at freeze time). All changes go through the root:

```
Corpus.add_version(...)     -> new draft numbered n+1
Corpus.record_content(...)  -> content on a draft; CorpusVersionFrozenError otherwise
Corpus.freeze_version(...)  -> frozen; CorpusVersionNotValidatedError / SameDataAlreadyFrozenError
```

`CorpusVersion` keeps its own identifier (`CorpusVersionId`) because it is what other contexts
reference; the identifier is stable even though the version is not a root.

## Consequences

- One repository (`CorpusRepository`) and one transaction per corpus
  change. Numbering and duplicate detection need no cross-aggregate coordination or database
  uniqueness rules.
- The published language of the Catalog is built from `Corpus.frozen_versions`; drafts
  never leave the context.
- Loading a corpus loads all its versions. Corpora have a handful of versions each, so this is
  not a cost worth designing around; the persistence schema still gets a `corpus_version` table
  with a foreign key to `corpus` inside the `catalog` schema.
- Freezing a version is a decision about the corpus (does this data already exist as a version?),
  which is why the duplicate rule lives in the root and not in the version.

## Alternatives considered

- **Two aggregates** (`Corpus` and `CorpusVersion`, versions referencing `corpus_id`): smaller
  units, separate repositories. Rejected: the multi-version invariants (numbering, uniqueness of
  data) would move to the repository or the database, i.e. out of the domain, and freezing would
  need a check across aggregates. The database schema nests versions under the corpus for the
  same reason.
- **Versions as value objects without identity**: rejected because other contexts must reference
  a version by a stable identifier, and a version's state changes (draft → frozen) over time.
