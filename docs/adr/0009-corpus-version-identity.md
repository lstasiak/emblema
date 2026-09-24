# ADR-0009: Corpus version identity — the checksum covers the bytes of the file set the reader declares

- Status: accepted
- Date: 2026-09-11
- Full text before condensation: commit `5f14447`

## Context

A frozen `CorpusVersion` is what other contexts pin, by `(version_id, checksum)` (ADR-0005). The
checksum is the cross-context identity of the data, and the rule that no two frozen versions
describe the same data (ADR-0004) compares it. The first reader (NASA C-MAPSS: four subsets of
run-to-failure trajectories as text files, plus test trajectories and RUL targets) had to decide
what the checksum covers, which part of a data set a version covers, and what the counts mean.

## Decision

- **Port** `CorpusReader.describe() -> CorpusDescription` in `catalog/ports`. An adapter is bound
  to its location at construction (`CmapssCorpusReader(root, subsets)`); the port carries no path.
  One call validates, checksums and counts, in one pass. Failures are domain errors
  (`CorpusDataNotFoundError`, `MalformedCorpusDataError`, both `CorpusReadError`).
- **The checksum is SHA-256 over the raw bytes of the corpus's files in a canonical order the
  adapter fixes**, streamed through `Checksum.of_chunks`. For C-MAPSS: `train_FD00x.txt` of the
  selected subsets, always in FD001–FD004 order. One changed byte is a new version — bytes are
  what a publisher ships and a mirror can be checked against.
- **A version covers the pretraining side.** The C-MAPSS reader registers training trajectories
  only; test trajectories and targets are labelled evaluation data. A checksum over training files
  alone proves by construction that no test engine took part in pretraining.
- **Counts** follow ADR-0008: a unit is what a split happens on, an observation is one channel at
  one instant; overlapping windows multiply neither. A unit may hold no observation.
- **Channels are the 21 sensors** named after Saxena et al. (2008) in file column order. The three
  operational settings are not channels.
- **Registration is one transaction.** `RegisterCorpusVersion` describes, opens, records, freezes
  and saves once, then publishes `CorpusVersionFrozen`. A failure leaves the corpus as it was.
  Creating a corpus is a separate use case (`RegisterCorpus`), so a misspelt name fails instead of
  creating a second corpus. Names are unique.

## Consequences

- Each reader states its canonical file set and order in its class docstring; anyone can recompute
  a version's checksum from the published files.
- Registering the same files twice fails (`SameDataAlreadyFrozenError`); modified files yield a new
  version. Both are contract tests of the port.
- The two-engine FD001 sample in `tests/data/cmapss/` is byte-exact (rows end with two spaces); a
  trimming editor changes the checksum the tests pin.
- Streaming values to a tokeniser is a second method on the port; `describe()` keeps its
  single-pass contract.

## Alternatives considered

- *Checksum over parsed values*: invariant to reformatting, but a parser bug changes the identity
  of unchanged data, and it cannot be checked without our code.
- *Checksum of the downloaded archive*: covers test files and documentation too, so it cannot say
  what was pretrained on.
- *One checksum per file*: a version needs one identity; per-file digests can be a diagnostic.
- *A version covering the whole data set*: the proof that pretraining excluded the test set would
  move from the checksum to a manifest.
- *`describe(location)`*: puts a file path into a port of the hexagon.
- *One find-or-create use case*: a typo becomes a new corpus.
