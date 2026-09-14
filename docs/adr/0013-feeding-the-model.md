# ADR-0013: Feeding the model — a sequence of windows, an order from a digest, a loader the contexts share

- Status: accepted
- Date: 2026-09-12

## Context

The tokeniser produces token windows (ADR-0012) and the array codec lays a list of them into the
five arrays the encoder declares (ADR-0007). Between those two and a training loop four things are
missing: something that holds windows, an order to visit them in, tensors to call the model with,
and the guarantee that all of it replays from a seed.

Two things that will exist later do not exist yet, and the design must not assume either. The
Pretraining domain arrives with the encoder; until then that context has no `domain` and therefore
no place for a port, because a port's types live in the domain it serves (ADR-0010). The published,
tokenised corpus arrives with its memory-mapped artefact; until then windows are Python objects in
memory.

The constraints are the repository's usual ones. Torch may be imported by adapters and entrypoints
only (`pure-core`); `shared/` may not import a bounded context; a context imports from another only
its `contracts` or `shared/`. One more constraint is specific to this project: a run is *prepared*
on one machine and *executed* on another — the accelerator is a free notebook platform with its own
Python and its own torch — so anything whose output depends on a library's random number generator
replays differently than it was recorded.

## Decision

**A dataset is a `Sequence` of windows.** `WindowDataset` holds one and answers by position; it
exists to satisfy torch's nominal `Dataset` and to name the seam, not to transform anything. The
memory-mapped store of the published corpus implements the same protocol, so it takes this place
without the loader changing. No port is introduced: `Sequence` already is one, the technology seam
here is torch and it is named by the adapter package, and a port of Pretraining today would mean
inventing that context's domain four tickets early.

**The order of an epoch is a ranking by digest, and the rule lives in the shared kernel.**
`seeded_rank(seed, *parts)` (`shared/kernel/ordering.py`) ranks an item by a digest of what
identifies it; `SeededShuffleSampler` ranks a position under the seed and the epoch, and
`UnitSplit.by_seed` ranks a unit under the seed. Both had written the same rule separately, and a
split fitted under one and an epoch replayed by the other come from the same recorded seed: a
change to either has to be a change to both, which only one definition can guarantee. The order
depends on nothing but the seed and the item, so it is identical on the machine that recorded a run
and on the machine that replays it, under any version of torch, numpy or Python. The epoch is state
of the sampler, because a run resumed from a checkpoint must continue the sequence rather than
restart it.

Each part of the key is **framed by its length** rather than joined by a separator. Joining made
`seeded_rank(seed, "3:17")` and `seeded_rank(seed, 3, 17)` the same item, so a unit whose name
happened to spell an epoch and a position would have shared a rank with it. No corpus in the
project names units that way, but this is the one function the reproducibility claim rests on and
its key should not depend on what a future reader calls its units. The digests it produces changed
with the framing; nothing had been recorded under the old ones, and the two pinned tests were
re-pinned.

**`seeded_rank` is behaviour in the shared kernel**, which the kernel had not held before — its
rule is value objects and closed vocabularies of identical meaning in several contexts, validated
but without behaviour (`DEVIATIONS.md`). The exception is narrow and the alternative is worse: the
function has no domain vocabulary of its own, it takes integers and strings and returns bytes, and
the only other home reachable from both the Catalog's domain and a shared adapter is a duplicate in
each. Evans's shared kernel is a subset of the model that the owning teams change together, code
included; this is a rule they must change together or not at all.

**The loading layer is a shared adapter, not one context's.** `WindowDataset`,
`SeededShuffleSampler` and `WindowLoader` live in `shared/adapters/loaders`, beside the tensor
codec they produce. They were first written under `pretraining/adapters` on the argument that
epochs and shuffling are training vocabulary. That argument does not survive the second consumer:
the evaluation harness feeds the same encoder the same way, and so does batch inference, and a
batch built differently for a probe than for the run it probes measures the difference between the
two rather than the model. The same reasoning put the array codec in `shared/` in ADR-0012. What
Pretraining will own is the training loop, not the machinery that puts a batch in front of it.

**`TokenTensors`** (`shared/adapters/tensors`) turns a `TokenBatch` into the five tensors, sharing
their memory, and moves them to a device and a precision without touching identifiers or masks. Its
`args` spells the order out rather than reading it off the fields, because that order is the model's
calling convention: the export names its graph inputs alongside it, and reordering the fields must
be a visible change rather than a silent one.

**Asking for batches names the epoch, and draws the first one.** `WindowLoader.batches_of(n)` sets
the order, pulls the first batch and returns it ahead of the rest. Torch's own convention is a
separate `set_epoch` call, and forgetting it is a silent failure: every epoch then visits the
windows in the same order, and the symptom is a training curve nobody can explain. Making the epoch
an argument of iteration removes the call that can be forgotten. Drawing the first batch is what
makes it true: torch's batch sampler is a generator that does not read the sampler until a batch is
asked for, so an epoch set after the iterator was made would have reordered one already being read.

**Workers outlive the epoch.** `persistent_workers` is on wherever workers are asked for. The whole
cost of a worker is the handover it pays at startup, and a pool restarted each epoch measured two
orders of magnitude worse than one that persists. Torch rejects the flag without workers to keep
alive, so it is conditional rather than constant.

**Packages are named for their subject, not their library** — `tensors` beside `arrays`, `loaders`
beside `readers` and `tokenisation`. A directory named `torch` next to `import torch` in one tree
misleads a reader and some tools; the same argument kept the numpy codec out of a package called
`numpy` out of a package called `numpy` when the codec was written.

## Consequences

- Measured on both development machines and recorded in
  `docs/verification/loader-throughput.md`. On the accelerator a training step costs tens of
  times what putting a batch in front of it does, which is the question answered; on the
  x86 machine, with no accelerator, it is three orders of magnitude. The comparison counts
  collating **and** the transfer to the device, because a batch built in host memory has to cross
  before the step starts. The accelerator leg is a test rather than a number in a note, so it keeps
  holding: `tests/ml/test_loader_keeps_up_on_mps.py`, at the margin the report prints.
- **A window of 1050 tokens costs about 93 KiB held as Python objects**, so the 25 395 windows of
  full C-MAPSS would cost roughly 2.3 GB and a corpus ten times larger will not fit at all. This is
  the measurement that turns the memory-mapped published artefact from a preference into a
  requirement, and it is why the dataset takes a `Sequence` rather than a list.
- Ordering an epoch costs tens of milliseconds at the scale of a full C-MAPSS and under a second at
  ten times that — the price of ranking by digest rather than permuting with a generator. Against
  an epoch measured in minutes this is noise; against a corpus of millions of windows it would not
  be.
- `shared/` now holds the words *epoch*, *shuffle* and *batch*. That is the cost of the decision
  above and the thing to watch: if a second caller never appears, the layer belongs to the one
  context that uses it.
- All nine import-linter contracts hold with no new exception.

## Alternatives considered

- **The loading layer in `pretraining/adapters/loaders`**, with Serving and Evaluation reaching for
  it through their own adapters later. Names an owner honestly and keeps training vocabulary out of
  `shared/`. Rejected after it was built that way: the second consumer is the evaluation harness,
  which must feed a candidate exactly as the run it compares it to was fed, and the only ways to
  get there are a cross-context import or a second implementation.
- **`torch.Generator().manual_seed(seed)` with `randperm`.** One line, linear rather than
  linearithmic, and what every PyTorch example does. Rejected: it ties the order of a recorded run
  to torch's generator, and this project's runs cross machines and torch versions by design. The
  cost of the alternative is measured above and is small at the scales the project reaches.
- **`numpy.random.default_rng(seed).permutation`.** Independent of torch and fast. Rejected for the
  same reason in a different library: numpy guarantees stream compatibility for the legacy
  `RandomState`, not for `Generator`.
- **`seeded_rank` duplicated in the Catalog's domain and the shared adapter**, leaving the kernel
  free of behaviour. Rejected: two copies of a rule that must not drift is exactly the failure the
  kernel exists to prevent, and the duplicate would be discovered only when a replayed run came out
  different.
- **A `WindowStore` port of Pretraining with an in-memory and a memory-mapped adapter.** Follows the
  letter of the rule that every port has at least two adapters. Rejected: the port would need types,
  and its types would have to live in a domain invented for the occasion; `Sequence` carries the
  same substitution with no new vocabulary.
- **A loader that also accepts a ready-made `WindowDataset`.** Written and removed: its only caller
  was the test written for it, and the `isinstance` branch behind it would still have rejected any
  dataset that was not a `WindowDataset`. A store that reads windows from a memory map is a
  `Sequence` and needs nothing added.
- **The item of a dataset is an index, and collation reads columns from the store.** Avoids
  rebuilding window objects from a memory-mapped file. Rejected as premature — there is no such
  store yet — but it is a non-breaking addition (`__getitems__`) and the memory figure above says
  it is the likely shape once the corpus is published as a block.
- **The dataset tokenises a unit per epoch**, holding no windows at all. Rejected on the envelope of
  ADR-0011: 27 million tokens of C-MAPSS take about 49 s of pure Python, which an epoch would pay
  again for nothing.

## Revisit when

- The corpus is published as a block. If reconstructing window objects from it costs a measurable part of
  a step, the dataset grows a batched read and the collation moves to the store; the seam is already
  where that change lands. Worker processes become cheap in the same move, since the handover is
  then a memory map rather than a corpus of objects.
- The evaluation harness or Serving arrive and turn out **not** to load windows this
  way. Then `shared/adapters/loaders` has one consumer, and the layer moves into Pretraining with
  only the tensor codec left shared.
- A corpus passes roughly a million windows, at which point ordering an epoch by digest costs
  seconds. Then the ranking is computed once into an array and indexed per epoch, or the order comes
  from a counter-based generator.
- A third caller of `seeded_rank` needs an ordering that is *not* replayable across corpora — then
  the shared rule has become a constraint rather than a guarantee, and it splits again with each
  copy named for what it orders.
