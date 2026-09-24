# ADR-0013: Feeding the model — a sequence of windows, an order from a digest, a loader the contexts share

- Status: accepted
- Date: 2026-09-12
- Full text before condensation: commit `5f14447`

## Context

Between the tokeniser (ADR-0012) and a training loop four things were missing: something that
holds windows, an order to visit them in, tensors for the model, and replay from a seed. The
Pretraining domain and the memory-mapped published corpus did not exist yet. A run is prepared on
one machine and executed on another (a notebook platform with its own Python and torch), so any
order that depends on a library's random generator replays differently.

## Decision

- **A dataset is a `Sequence` of windows.** `WindowDataset` satisfies torch's `Dataset`; the
  memory-mapped store implements the same protocol later. No port: `Sequence` already is one.
- **An epoch's order is a ranking by digest, defined once in the kernel.**
  `seeded_rank(seed, *parts)` (`shared/kernel/ordering.py`) ranks an item by a digest of what
  identifies it. `SeededShuffleSampler` ranks positions under seed and epoch; `UnitSplit.by_seed`
  ranks units under the seed. The order depends only on seed and item — identical under any torch,
  numpy or Python. The epoch is sampler state, so a resumed run continues the sequence.
- **Each key part is framed by its length**, not joined by a separator, so `"3:17"` and `(3, 17)`
  never collide.
- **`seeded_rank` is behaviour in the kernel** — a narrow exception: no domain vocabulary, and the
  only alternative is a copy per context that must not drift.
- **The loading layer is shared** (`shared/adapters/loaders`: `WindowDataset`,
  `SeededShuffleSampler`, `WindowLoader`). Evaluation and batch inference feed the same encoder, and
  a batch built differently for a probe than for its run measures the difference, not the model.
- **`TokenTensors`** (`shared/adapters/tensors`) turns a `TokenBatch` into five tensors sharing
  memory and moves them to a device and precision. `args` spells the order out: it is the model's
  calling convention.
- **Asking for batches names the epoch and draws the first batch** (`batches_of(n)`), replacing
  torch's forgettable `set_epoch`.
- **Workers persist across epochs** where workers are used; a pool restarted each epoch measured
  two orders of magnitude worse.
- Packages are named for their subject (`tensors`, `loaders`), not their library.

## Consequences

- `docs/verification/loader-throughput.md`: on the accelerator a training step costs tens of times
  what collating and transferring a batch does; the MPS leg is a test
  (`tests/ml/test_loader_keeps_up_on_mps.py`).
- **A 1,050-token window costs ~93 KiB as Python objects**: full C-MAPSS would take ~2.3 GB. This
  made the memory-mapped published corpus a requirement.
- Ordering an epoch by digest costs tens of milliseconds on C-MAPSS — noise against a minute-long
  epoch.
- `shared/` now holds *epoch*, *shuffle* and *batch*; if no second caller appears, the layer moves
  to Pretraining.

## Alternatives considered

- *Loader in `pretraining/adapters`*: evaluation would need a cross-context import or a copy.
- *`torch.Generator` + `randperm`* or *numpy `Generator`*: order tied to a library's stream, which
  neither guarantees across versions.
- *A `WindowStore` port*: its types would need a domain invented for the occasion.
- *Tokenising per epoch*: ~49 s of pure Python per pass on C-MAPSS.

## Revisit when

- Rebuilding window objects from the block costs a measurable part of a step → batched reads.
- Evaluation or Serving do not load windows this way → the layer moves into Pretraining.
- A corpus passes ~1M windows → compute the ranking once into an array.
