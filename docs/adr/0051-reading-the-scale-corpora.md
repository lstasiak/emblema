# ADR-0051: Reading the scale corpora: one corpus per UTSD dataset read on each series' scale, sepsis stays cut by name

- Status: proposed
- Date: 2026-10-04

## Context

Pretraining at ten times the data needs corpora the Catalog has not read: UTSD, a collection of
datasets published as 80 Arrow shards of univariate records named `<dataset>_<a>_<b>`; the
PhysioNet 2019 sepsis challenge, 40,336 hourly stays with the task's label in every row and no
labelled test set; and the Tennessee Eastman simulations, two R data frames of 10,500 runs.

The collection's two numbers mean different things per dataset, and every dataset sits in the
same shards, so a block of the whole would be 35 GB. The challenge's test side has to be cut from
the training sets, not left out as a directory as set C of 2012 was.

## Decision

- **A dataset of the collection is a corpus of its own**, `utsd/<dataset>`, chained through the
  vocabulary: the datasets measure unrelated things, and each block has its own manifest.
- **The layout of each dataset is declared** in a registry the reader carries and **checked
  against the records**, for all 29 datasets of the volume: series of variates (`a` the series),
  variates of series (`a` the variate, as ERA5's grid), or a collection. Four datasets of one
  series are excluded with that reason, since no split divides one unit.
- **How a dataset is read is a publication option, apart from its layout**: its variates as one
  unit's channels or every record a unit of its own (`--channels-independent`), and each record
  on its own scale, the default, or as stored (`--scale-as-published`). Measured over the volume,
  a series spans a median of 0.006 of its channel's scale in web traffic, 0.002 in TDBrain and
  0.019 in IEEEPPG when every series shares one scale per channel, the compression that made the
  turbofan pretext trivial (0.004, ADR-0034). A reading that leaves one unit is refused.
- **The checksum covers the reading and the records**, names and stored values in canonical
  order: two readings of a dataset publish as two versions, and a dataset describes the same
  wherever the shards cut it.
- **The collection is read from a local snapshot** pinned to a revision and checked by digest,
  not streamed: a description is a function of bytes, not of a connection.
- **The sepsis stays are the challenge's two sets less the stays excluded by name**, out of the
  units, the counts and the checksum; a name matching no stay is refused, since a silent miss
  would pretrain on a test stay. Which stays is the task's decision.
- **A label column is never a channel**: the sepsis label is read only to be checked, the fault
  number only to name the run.
- **A simulation run is a unit**, keyed by its fault and number within it; the two training files
  are the corpus's parts and the testing files stay out.
- **R data files are read with `rdata`** (pure Python, MIT) rather than the C-backed reader under
  the AGPL: the repository is Apache-2.0, and a one-off parse at publication buys no copyleft.

## Consequences

- Twenty-five more names on the publishing command line, generated from the registry; a run
  names the datasets it mixes and weighs them with `passes` and `fraction`.
- Which reading and which window a mixture takes is a publication's choice: either reading of a
  dataset can be published later beside the other, for an ablation, without code.
- A record is a slice of its mapped batch, read in microseconds rather than milliseconds.
- The block writer keeps a few machine words per window, not Python objects, so a dataset of a
  hundred million values publishes within memory.

## Alternatives considered

- *One corpus `utsd` with the datasets as subsets*: one block of the whole, and one vocabulary
  across unrelated datasets.
- *Every record a unit on one channel*: loses the variates of a multivariate dataset.
- *Inferring the layout from the records*: a refused declaration is louder than a guess.
- *A layout chosen for the experiment*: it would record a modelling choice as a fact of the data.
- *One scale per channel, as for the task corpora*: the measurement above; the level of an
  unrelated series carries nothing a task could use, while a vital sign's does.
- *Streaming the shards*: every pass of a publication is a download.
- *The sepsis label as a channel*: leaks the task into the pretext.

## Revisit when

- A long single series is worth cutting into segments as units.
- The sepsis task names its frozen stays: the publication reads the list from the task's protocol.
