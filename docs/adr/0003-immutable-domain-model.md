# ADR-0003: Immutable domain model — aggregates and value objects as frozen dataclasses

- Status: accepted
- Date: 2026-09-08

## Context

The domain layer is plain Python with no framework (import-linter contract `pure-core`). Its
central invariants are about things that must not change once fixed: a frozen corpus version, a
backbone after creation, a test split after freezing, a promoted artifact identified by checksum.
The first aggregate, `Corpus` with its `CorpusVersion` entities, had to settle a style that every
later context (Pretraining, Evaluation, Serving) will copy: mutable aggregate roots whose methods
change state in place, or immutable aggregates whose methods return a new instance.

## Decision

Every domain object — value object, entity and aggregate root — is a `dataclass(frozen=True)`.
An operation on an aggregate is a method that validates the transition, builds the new state with
`dataclasses.replace` and returns it. The receiver is never modified:

```python
corpus = corpus.add_version(version_id, schema, regime, licence)
corpus = corpus.record_content(version_id, content)
corpus = corpus.freeze_version(version_id, clock.now())
repository.save(corpus)
```

Invariants live in `__post_init__` and in the transition methods. Because `replace` re-runs
`__post_init__`, an aggregate cannot be reconstituted (from persistence or a test) in a state its
constructor rejects.

Collections inside aggregates are tuples. Lookup is linear; the collections in question (versions
of a corpus, runs of a campaign) are small by construction.

## Consequences

- **No hidden state.** Two references to a corpus never disagree; a use case sees exactly the
  value it computed. The only way to change a frozen version is a method that refuses to.
- **Repository semantics are explicit.** A use case saves the value returned by the last
  operation. Forgetting to save is visible in the code, not a silently unpersisted mutation.
- **Equality is structural.** Two snapshots of the same aggregate at different lifecycle stages
  compare unequal; identity comparison uses `.id`. Tests assert on values, not on object identity.
- **Hashable by default.** Domain objects can be dictionary keys and set members without extra
  code; the stateful property test of `Corpus` relies on this to detect changes to frozen versions.
- **Boilerplate.** Each transition is `replace(self, ...)`; a root that changes one element of a
  collection rebuilds the tuple (`Corpus._replace_version`). The cost is a few lines per aggregate.
- **Threshold at which the choice flips.** An aggregate whose operations append to or update a
  collection of thousands of elements per transaction (e.g. per-token state) would pay O(n) per
  operation. No planned aggregate has that shape; if one appears, that aggregate may be mutable
  with a documented reason, while value objects stay frozen.

## Alternatives considered

- **Mutable roots, frozen value objects** (the pattern of *Architecture Patterns with Python*):
  `corpus.freeze_version(id, at)` mutates in place; entities are `dataclass(eq=False)` with
  identity equality. Less boilerplate and closer to common Python DDD examples. Rejected: any code
  path holding a reference can mutate state around the invariants, `replace`-style validation on
  reconstitution is lost, and the immutability the domain talks about (frozen versions, immutable
  backbones) would be a convention rather than a property of the objects.
- **Type-state classes** (`DraftCorpusVersion` / `FrozenCorpusVersion`): illegal transitions
  become type errors. Rejected for now: doubles the number of classes per lifecycle, and the
  aggregate root must still dispatch at runtime on a union; the constructor invariant
  "frozen implies content" gives most of the safety at a fraction of the surface.
