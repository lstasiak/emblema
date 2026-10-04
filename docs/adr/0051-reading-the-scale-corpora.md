# ADR-0051: Reading the scale corpora: one corpus per UTSD dataset, sepsis stays cut by name

- Status: proposed
- Date: 2026-10-04

## Context

Pretraining at ten times the data needs corpora the Catalog has not read: UTSD, a collection of
datasets published as 80 Arrow shards of univariate records named `<dataset>_<a>_<b>`; the
PhysioNet 2019 sepsis challenge, 40,336 hourly stays with the task's label in every row and no
labelled test set; and the Tennessee Eastman simulations, two R data frames of 10,500 runs.

The collection's two numbers mean different things per dataset: in a multivariate dataset `a`
numbers a series and `b` its variate; in a collection of separate series the publisher numbered
the series in `a`, in `b`, or in `b` within blocks of a thousand numbered by `a`. Every dataset
sits in the same shards, so a block of the whole would be 35 GB. The challenge's test side has to
be cut from the training sets rather than left out as a directory, as set C of 2012 was.

## Decision

- **A dataset of the collection is a corpus of its own**, named `utsd/<dataset>`, published
  separately and chained through the vocabulary. The datasets measure unrelated things; a corpus
  per dataset keeps their channels apart and cuts the publication where the data is cut, so each
  block has its own checksum and manifest.
- **The layout of each dataset is declared** in a registry the reader carries and **checked
  against the records**: a multivariate dataset is a unit per series and a channel per variate,
  and is refused as malformed when its series do not all carry the same variates; a collection is
  a unit per record on one channel, named by both numbers. Values sit at their index, the regime
  is regular, and a value stored as not-a-number is no observation.
- **The checksum of a dataset covers its records alone**, names and values in canonical order:
  two datasets of one shard describe differently, and a dataset the same wherever the shards cut.
- **The collection is read from a local snapshot** pinned to a repository revision and checked by
  the digests the repository stores, not streamed: the reader is bound to a directory like every
  other, and a description is a function of bytes, not of a connection. Where the repository's
  digests contradict its sizes, as for the smaller volume, they are taken from downloaded files.
- **The sepsis stays are the challenge's two sets less the stays excluded by name**: out of the
  units, the counts and the checksum. A name that matches no stay is refused, since an exclusion
  that silently missed would let a test stay be pretrained on. Which stays are excluded is the
  task's decision, recorded with its protocol.
- **A label column is never a channel**: the sepsis label is read only to be checked, the fault
  number only to name the run.
- **A simulation run is a unit**, keyed by its fault and number within it; the two training files
  are the corpus's parts and the testing files stay out.
- **R data files are read with `rdata`** (pure Python, MIT) rather than the C-backed reader under
  the AGPL: the repository is Apache-2.0, and a one-off parse at publication buys no copyleft.

## Consequences

- Eighteen more names on the publishing command line, generated from the registry; a run names
  the datasets it mixes and weighs them with `passes` and `fraction`.
- A declaration the data contradicts is found at the first description on the machine that holds
  the shards, where the registry is completed; a dataset the shards hold and the registry does
  not is found by a test that lists them there.
- The block writer keeps a few machine words per window rather than Python objects, so a dataset
  of a hundred million values cut at a short stride publishes within memory.
- The exclusion list is an input of the publication, read from a file; only the corpus whose
  publisher drew no test set accepts it.

## Alternatives considered

- *One corpus `utsd` with the datasets as subsets*: one block of the whole, and one vocabulary
  across unrelated datasets.
- *Every record a unit on one channel*: loses the variates of a multivariate dataset.
- *Inferring the layout from the records*: a refused declaration is louder than a guess.
- *Streaming the shards*: every pass of a publication is a download, or a snapshot in a hidden
  cache.
- *The sepsis label as a channel*: leaks the task into the pretext.

## Revisit when

- A dataset of the collection holds one series only: it cannot be split on units and is read as a
  collection or left out.
- The sepsis task names its frozen stays: the publication reads the list from the task's protocol.
