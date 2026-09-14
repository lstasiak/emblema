# ADR-0012: Where tokenisation lives — the window in the shared kernel, the array codec beside it, the tokeniser a port of the Catalog

- Status: accepted (the Serving section is a direction, not yet an implementation)
- Date: 2026-09-11

## Context

Three bounded contexts touch the token representation of ADR-0011. The Catalog produces it —
"token schema" is in the Catalog's language and `Tokeniser` is one of its ports. Pretraining
trains on it: the dataset layer needs windows before the artifact pipeline exists. Serving infers on it: a request of raw observations must become the five
tensors exactly as training did, or the model is served a different distribution than it learned.

The dependency rules make the placement a real decision. A context imports from another only its
`contracts` or `shared/`; `shared/` imports no context; `contracts` depends on `shared/` only; and
the framework-free core — every `domain`, `ports`, `application`, `contracts`, and `shared/kernel`,
`shared/ports`, `shared/events` — imports no numpy or torch (`pure-core`). So a type shared by three
contexts can live in a `contracts` package or in the shared kernel, but an array codec over it can
live only where numpy is allowed and where all three can reach it.

## Decision

**`TokenWindow` is a value object of the shared kernel** (`shared/kernel/tokens.py`), with the
row view `Token`. It carries the invariants of ADR-0011 and nothing else, like `Checksum` or
`ArtifactRef`. This widens the kernel beyond what it held — identifiers, checksums, artifact
references, time — to *the model's input representation*, the largest growth of the kernel in the
project, recorded in `DEVIATIONS.md` with the rewritten rule: the kernel holds value objects and
closed vocabularies of identical semantics in several contexts, validated but without domain
behaviour. The defence is Evans's: a shared kernel is a subset of the model that the owning
teams agree to share and change together, code included; three contexts read this type with one
meaning, and the meaning of a window's identifiers and values is supplied by the manifest the
Catalog publishes beside it, the way a `CorpusVersionRef` gives meaning to a checksum.

**`TokenBatch` is a codec in `shared/adapters/arrays`**, the five arrays of ADR-0007 with padding
and mask, reversible to windows. It implements no port — it is the wire format of the
representation, as `shared/adapters/storage/layout.py` is the key format of the artifact store —
and it can live nowhere else: not in any `contracts` (numpy), not in the Catalog's adapters (the
other contexts could not import it), not once per consumer (two copies of the code that decides
what the model sees). The package is named for its subject, `arrays`, not for the library, because
a package called `numpy` beside `import numpy` in the same tree misleads a reader and some tools.

**`Tokeniser` is a port of the Catalog** (`catalog/ports/tokeniser.py`, ADR-0010: a Protocol, its
types in `catalog/domain`, its failures in `domain/exceptions.py`), with `fit` over a flat stream
of training values and `tokenise` over one unit. The rules — how windows are laid
(`WindowSpec.windows_over`), how a value is normalised (`ChannelStatistics.normalise`), what a
window is (`TokenWindow`) — are domain value objects; the adapter is bookkeeping. The reason it is
a port and not a domain service is specific to this repository: `pure-core` forbids numpy in the
domain, so a domain service would fix tokenisation at pure-Python speed for good, while a port is
the seam through which a vectorised implementation can arrive — the port is the boundary to the
array library, in Cockburn's sense of a boundary to a technology. The two usual reasons for a
second adapter, a deterministic fake and an ablation axis for the estimator, do not hold up: a
fake of a deterministic computation passes no content test and buys the publishing use case
nothing that the real tokeniser on two sample engines does not, and the estimator is a Strategy
inside an adapter.
Hence **one adapter**, `SlidingWindowTokeniser`, and a contract test parametrised over a list of
one, the precedent being `CorpusRepository` — recorded as a deviation from the project rule that
every port has a second, in-memory adapter.

**The reader port streams units and observations separately.** `read_units()` yields `CorpusUnit`
— a value object of key, time extent and static features — and `read_observations(key)` streams a
unit's observations in time order. The first design had the observations as a lazy field of the
unit; it was a cursor dressed as a value: equality by identity, consumed once, and unable to check
that its observations lie inside its extent. With two calls the use case iterates units once and
observations twice (fitting, then tokenising), the tokeniser checks every observation against the
unit's extent, and a unit of any size still streams. The C-MAPSS reader answers
`read_observations` by scanning the subset's file to the engine's block and stopping after it,
which it can do because the rows of an engine are contiguous — a property the reader now
enforces.

**Serving reuses the Catalog's tokeniser through an Open Host Service** — the direction, to be
implemented and measured when Serving is built. The Catalog publishes, in its `contracts`, the frozen scheme
(vocabulary and statistics) and a Protocol for tokenising one window of raw observations; the
Catalog's adapter implements it; the composition root injects it into Serving. Serving depends on
the contract only, the import-linter is satisfied, and one implementation serves training and
inference. This section stays a direction because the project's rule is that a decision is
defended by an implementation and a measurement, and neither exists yet.

## Consequences

- Import-linter: every contract kept without a new exception. The kernel's `tokens.py` imports
  `kernel/exceptions.py` only; `shared/adapters/arrays` imports the kernel and numpy; the Catalog's
  ports import the domain and the kernel; `pure-core` catches any indirect path from a core module
  to the codec.
- `tests/architecture/test_ports_are_protocols.py` covers `catalog/ports/tokeniser.py` with no
  change; the contract test of the reader gained the streaming methods and runs against both
  adapters; the contract test of the tokeniser runs against one.
- The stand-in encoder's test harness draws channel identifiers from 1, because 0 is padding.
- numpy stays in the `ml` extra; the codec's tests skip where it is absent (the local 3.12 leg).

## Alternatives considered

- **`TokenWindow` in `catalog/contracts/tokens.py`**, imported by the domain like `identifiers`.
  Names the owner faithfully. Rejected: the array codec would then have to live in the Catalog's
  adapters, unreachable from Pretraining and Serving, or be duplicated in both — a train/serve
  skew waiting to happen — and the linter contract that keeps a domain from its own messages would
  need a second exception.
- **The token schema only in `catalog/domain`, crossing contexts as the published artefact.** No rule
  changes at all. Rejected: the dataset layer would have no input until the artifact
  format exists, or Pretraining would write a decoder for a format not yet defined.
- **Tokenisation as a shared-kernel service**, so that Serving imports the code directly.
  Evans permits behaviour in a shared kernel. Rejected: the *representation* has identical
  semantics in three contexts; the *procedure* that produces it is the Catalog's, with the
  Catalog's vocabulary and split, and moving it would hollow the context that is defined by it.
- **Serving re-implements the per-token arithmetic from the published scheme.** Fifty lines, and a
  cross-context test could pin the two implementations to each other. Rejected: two encoders of
  the model's input are the classic source of serving skew, and the test would import both
  contexts' internals.
- **Normalisation and gaps computed inside the ONNX graph**, so that Serving sends raw values.
  Per-channel affine as a gather is easy; the gap needs a per-channel sort in the graph and the
  name-to-identifier lookup stays outside anyway. Kept as an option for the export adapter, not a
  solution.
- **A domain service instead of a port**, with the port introduced when a use case needs a fake.
  DDD-canonical for pure computation. Rejected for the vectorisation seam above; the threshold
  at which it flips is below.

## Revisit when

- Serving ends up *not* consuming `TokenWindow` — the Open Host Service returns arrays, or
  tokenisation moves into the graph. Then the type has one producer and one consumer and belongs
  in `catalog/contracts/tokens.py`.
- A second real corpus arrives and there is still one adapter of `Tokeniser` and no measured need
  for a faster one.
  Then the port collapses into a domain service and this record is superseded on that point.
- A corpus arrives whose unit does not fit the streaming budget of one window's observations, or
  the pure-Python envelope of ADR-0011 (about 1.8 µs per token) becomes the bottleneck of the
  preprocessing step. Then the second adapter vectorises the inner loop; if the object stream of
  the reader port itself is the limit, the port grows a columnar block type in the kernel.
