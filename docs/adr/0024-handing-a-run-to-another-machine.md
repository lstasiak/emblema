# ADR-0024: A run is ordered here, made anywhere, and accepted back against the order

- Status: accepted
- Date: 2026-09-16

## Context

The runs whose results are published happen on rented GPUs, in notebook sessions that may drop and
that reach nothing of this system but the bucket. The system still has to say what was trained,
on which data, by which code, and where the weights are: a backbone whose provenance lives in a
notebook's output cell is not one the evaluation can pick up or the serving can promote.

What existed before this record: a training runtime behind a port that reports epochs as they
finish (ADR-0021), an experiment file that states every parameter of a run (ADR-0022), a tracker
that records the run while it runs (ADR-0023), and a published corpus that any machine with the
bucket's credentials can mount from its manifest (ADR-0014). What did not exist: a registry of
backbones, a way to read a published corpus from inside Pretraining without touching the Catalog,
and a way to carry a run across machines that keeps the registry honest.

## Decision

**Three steps, two machines, one backbone.** `order` registers a `Backbone` as ordered and places
an order in the artifact store; `run` reads the order on whichever machine trains, trains through
the same use case a local run goes through, and reports a result into the store; `accept` reads
the result back here, holds it to the order, replays the run through the same use case again with
a runtime that reads results rather than training, and delivers the weights to the backbone. The
machine that trains may be this one: the local runtime is a platform like any other, so the path
is one and is tested once.

**The `Backbone` aggregate is ordered before it is ready.** It carries the configuration, the
published corpus it reads (`PretrainingInput`: which version of which corpus, the manifest, the
block's checksum, the vocabulary), the run's name, the revision of the code, the signature of the
run, and — once delivered — the artifact and the time of delivery. Delivery is a method that
returns a new backbone and refuses a second delivery; `ready` is the presence of the artifact.
The registry knows an order is open, and a result is compared with what the registry says was
ordered, not with arguments typed a second time.

**Orders and results are documents in the artifact store**, behind a `HandoffExchange` port with
two adapters (JSON in the store; in memory). The store is the one channel the machine that trains
already has credentials for; content addressing means a result names the order it fulfilled by a
reference nothing can change under it; and both documents are durable because they are the
provenance of a backbone, not an intermediate of the run.

**A result states facts, and acceptance names what differs.** The result carries the configuration
the run trained under, the shape of the corpus it read — name, checksum of the block, windows on
each side, vocabulary — the revision of the code, the checkpoint it was picked up from and the
outcome. Acceptance refuses a result of another configuration naming the parameter, a result over
other data naming both checksums, a result made with other code naming both commits, and a result
picked up from another checkpoint; the refusal for the configuration and the refusal for the data
are different sentences because they are fixed in different places, and each is checked before
the run's signature, which by then can differ only in how many windows each side held. The
signature of a run (ADR-0021) is now computed from the corpus's *shape* as well as from the
corpus, so a machine that never held the windows can be held to the same digest; the parameters
it digests render every rate as a float, so a configuration built with an integer zero signs as
the one built with a float zero. What the replaying runtime yields is held to the result accepted
as well, because the runtime is built on a result of its own and only that comparison says the
two are one document.

**The machine that trains stops before it trains.** An order names the revision of the code and
the signature of the run; the machine fulfilling it refuses other code and another corpus before
the first epoch, because a result the acceptance would refuse is a session of an accelerator
wasted. The acceptance checks both again, as the second line.

**The handoff runtime is the third adapter of `TrainingRuntime`.** Reading a result and reporting
its epochs is a way of fulfilling the port: the use case that records runs does not know whether
the epochs it logs were computed a moment ago or on another continent, and the tracker holds the
curve of a run made elsewhere as it holds any other. The port's contract runs against it through a
simulated platform — the in-memory runtime training on the far side of an exchange — so the replay
is held to everything a runtime is held to.

**Pretraining reads a published corpus through a port of its own.** `TrainingCorpusReader` has
`describe`, which reads the manifest through the Catalog's published codec and costs kilobytes,
and `read`, which fetches the block once into the workspace and maps it. The block is kept under
its digest, the layout the Catalog's archive keeps its own blocks in, so the machine that published
a corpus orders and accepts without fetching what it wrote; a block found there is hashed before it
is mapped, because the workspace is written by another process and outlives every one of them.
The Catalog's `CorpusArchive` stays the Catalog's.

**Pretraining persists its aggregate the way the Catalog does** (ADR-0016): record classes over a
declarative base whose metadata fixes the `pretraining` schema, one migration in the shared tree,
the configuration as JSONB with the seed, the parameter count and the commit as columns beside it
because they are asked about, and a `status` column held true to the artifact columns by a check
constraint. The naming convention the two contexts share moved to `shared/adapters/persistence`.

**The revision of the code is read, not typed.** A package installed from a repository at a
revision records it in its distribution metadata, which is what a notebook that installed
`git+…@<sha>` runs; a package installed editable records the tree instead, and that tree's `HEAD`
is the answer, marked `-dirty` when the tree has uncommitted changes, because a run made on such a
tree is not a run of that commit. An order is placed only from a committed tree: it names what
another machine installs, and a marked revision is one nobody can install. A run on a dirty tree
carries the mark and is refused against an order from the committed one. `--commit` states the
revision where it cannot be read, or where the operator decides the tree is what the run is of.

**The second composition root.** The pretraining command line is the second process assembled by
hand (ADR-0015). What the two roots repeat is the store and the engine built from settings, which
became two functions in `entrypoints/cli/configured.py`; the rest of each root names adapters the
other has no use for.

## Consequences

- A backbone in the registry says which experiment, which run, which corpus, which code and which
  weights, whether it was trained here or on a rented GPU; the evaluation harness can read that
  without asking a notebook.
- `accept` replays the run into MLflow, so the curve of a run made on a platform with no tracking
  server lands where every other run's curve is. The tracking URI is required there: the backbone
  refuses a second delivery, so a replay into a tracker that dies with the process would be the
  one chance to record the run spent.
- The ready row names the result it was delivered with, beside the weights, so the order it
  fulfilled, the checkpoint it was picked up from and every epoch are reachable from the registry.
  The order's own reference is not stored: the exchange is content-addressed and the order is
  recomputed from the row.
- Accepting reads the block, from the workspace where the machine that ordered already holds it,
  or once from the store on a machine that does not — not to check the result, which is held to
  the signature the registry keeps, but because the replay goes through the training runtime's
  port, whose input is a corpus. A fresh read catches only a reader that changed.
- An open order is a row with no artifact. Nothing deletes it: a run that never came back is a
  fact about the experiment, and a second order of the same run is a second backbone.
- The pretraining command line needs the `ml` and `tracking` extras, because one process may
  train and the same process may record to MLflow; the publishing command line needs neither.
- A single-corpus backbone. The mixture of corpora that later experiments train on will read more
  than one manifest, and `PretrainingInput` becomes a tuple then; the table already keys the input
  by the backbone alone and gains a position column with that change.
- Two acceptances of different results for one backbone racing each other are not guarded: the
  open check and the save are two statements, and the last save wins. The process is manual and
  one operator's; the guard belongs with the first process that accepts concurrently.
- A resumed run's result holds the epochs after the checkpoint, and so does the curve the replay
  records: the epochs before it are the interrupted run's, in its tracker if it had one. The
  registry leads to the checkpoint through the result; the curve is not stitched.

## Alternatives considered

**One adapter that generates the specification, waits, and accepts.** The shape the design was
first described in. Rejected: an adapter behind `train()` cannot both write an order before a run
and read a result after it, and registering the weights is a use case over the registry, not an
adapter's side effect. The port still gets its third adapter — the replay — and the writing and
the reading became use cases.

**Registering only the finished backbone.** No open orders, the signature recomputed at acceptance
from an experiment file and a manifest given again. Rejected: the registry would not know that a
run is outstanding, and acceptance would compare the result with whatever the operator typed the
second time rather than with what was ordered.

**Exchanging documents as files.** No credentials needed to read an order; rejected because the
result has to come back from a machine that has the bucket and nothing else, and a file carried by
hand is one more thing to get wrong than a reference to content that cannot change.

**Pinning the published corpus in the experiment file.** The revisit threshold of ADR-0022. Not
adopted: the file says what to do and the order says on what; a corpus published under several
windows would need one file per publication, and the control experiments publish their corpus on
the fly with no reference to pin.

**SQLModel or `MappedAsDataclass` for the second context's records** (the revisit threshold of
ADR-0016). Not adopted: the columns that carry meaning here are JSONB, `timestamptz` and three
check constraints, which SQLModel hands back to SQLAlchemy anyway; two contexts with two record
classes each repeat a UUID key and a checksum pair, which is too little to justify a dataclass
mapping or a set of column aliases. The naming convention was the one thing both repeated, and it
moved.

## Revisit when

- A backbone reads more than one manifest: `PretrainingInput` becomes a tuple and the table gains
  a position.
- A result is wanted before the run ends, epoch by epoch, so that a session that drops leaves its
  curve behind on a platform with no tracking server: the exchange would then carry partial
  results, or the tracker would have a store-backed adapter.
- Another context needs a backbone by identity rather than by an opaque reference and a checksum:
  `BackboneId` moves to `pretraining/contracts`.
- The evaluation worker becomes the third process: what the three roots repeat decides ADR-0015.
