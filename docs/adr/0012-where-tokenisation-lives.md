# ADR-0012: Where tokenisation lives — the window in the shared kernel, the array codec beside it, the tokeniser a port of the Catalog

- Status: accepted (the Serving section is a direction, not yet an implementation)
- Date: 2026-09-11
- Full text before condensation: commit `5f14447`

## Context

Three contexts touch the token representation (ADR-0011). The Catalog produces it, Pretraining
trains on it, Serving infers on it — and a served request must become the five tensors exactly as
in training. A context imports only another's `contracts` or `shared/`, and the framework-free core
(`pure-core`) forbids numpy. So a type shared by three contexts can live in `contracts` or the
kernel, but an array codec over it only where numpy is allowed and all three can reach it.

## Decision

- **`TokenWindow` is a value object of the shared kernel** (`shared/kernel/tokens.py`), with the
  row view `Token`. It carries ADR-0011's invariants and nothing else. This widens the kernel to
  *the model's input representation*; the rule becomes: value objects and closed vocabularies of
  identical meaning in several contexts, validated, without domain behaviour. Evans's shared kernel
  is a subset the owning teams change together.
- **`TokenBatch` is a codec in `shared/adapters/arrays`**: the five arrays of ADR-0007, reversible
  to windows. It implements no port — it is the wire format of the representation — and there is
  one copy, because two copies of the code deciding what the model sees is serving skew.
- **`Tokeniser` is a Catalog port** with `fit` over training values and `tokenise` over one unit.
  The rules (`WindowSpec.windows_over`, `ChannelStatistics.normalise`, `TokenWindow`) are domain
  value objects; the adapter is bookkeeping. It is a port, not a domain service, because the core
  forbids numpy and the port is the seam for a vectorised implementation. It has **one adapter**,
  `SlidingWindowTokeniser` (a fake of a deterministic computation passes no content test), with a
  contract test over a list of one.
- **The reader streams units and observations separately**: `read_units()` yields `CorpusUnit`
  (key, extent, static features), `read_observations(key)` streams one unit in time order. The
  tokeniser checks each observation against its unit's extent.
- **Serving will reuse the Catalog's tokeniser through an Open Host Service**: the Catalog
  publishes the frozen scheme and a Protocol for tokenising one window; the composition root
  injects the adapter. A direction until implemented and measured.

## Consequences

- Every import-linter contract holds without a new exception.
- The reader contract test covers both adapters; the tokeniser contract runs against one.
- Channel id 0 is padding, so test harnesses draw ids from 1.
- numpy stays in the `ml` extra; codec tests skip where it is absent.

## Alternatives considered

- *`TokenWindow` in `catalog/contracts`*: the codec would then be unreachable from Pretraining and
  Serving, or duplicated.
- *Only the published artifact crosses contexts*: the dataset layer would wait for the format.
- *Tokenisation as a shared-kernel service*: the representation is shared; the procedure, with the
  Catalog's vocabulary and split, is the Catalog's.
- *Serving re-implements the arithmetic*: two encoders of the input are the classic source of skew.
- *Normalisation and gaps inside the ONNX graph*: the gap needs a per-channel sort in the graph;
  kept as an export option.
- *A domain service*: fixes tokenisation at pure-Python speed.

## Revisit when

- Serving does not consume `TokenWindow` → move it to `catalog/contracts`.
- A second real corpus arrives with still one tokeniser adapter and no need for speed → the port
  collapses into a domain service.
- A unit exceeds the streaming budget, or ~1.8 µs per token becomes the preprocessing bottleneck →
  a vectorised adapter, possibly a columnar block type in the kernel.
