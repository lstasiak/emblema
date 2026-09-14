# ADR-0009: Corpus version identity — the checksum covers the bytes of the file set the reader declares

- Status: accepted
- Date: 2026-09-11

## Context

A frozen `CorpusVersion` is what other contexts pin: Pretraining records which versions a backbone
was trained on, Evaluation which version a task draws from, both by `(version_id, checksum)`
(ADR-0005). The checksum is therefore the cross-context identity of the data, and the rule that no
two frozen versions describe the same data (ADR-0004) compares it. Until this ticket nothing
computed it: the domain took a `CorpusContent` from a reader that did not exist.

Three questions had to be answered by the first reader, and their answers bind every reader after
it: what the checksum is computed over, which part of a published data set a version covers, and
what the content counts mean. The first corpus is NASA's C-MAPSS, four subsets of run-to-failure
engine trajectories shipped as space-separated text files, with separate test trajectories and
remaining-useful-life targets for each subset.

## Decision

**The port.** `CorpusReader.describe() -> CorpusDescription` lives in `catalog/ports` (ADR-0010).
An adapter is bound to the location and format of its corpus at construction
(`CmapssCorpusReader(root, subsets)`), the way an artifact store is bound to its bucket; the port
carries no location, so the hexagon has no notion of a file. One call validates the data,
checksums it and counts it, so the data is scanned once however large it is. The description
carries the channel schema, the sampling regime and the content, which is everything the
aggregate needs to open, fill and freeze a version; it is a domain value object. Failures are the
domain's: `CorpusDataNotFoundError` when the source holds nothing where the adapter expects it,
`MalformedCorpusDataError` when the data violates its own format; both extend `CorpusReadError`.

**What the checksum covers.** SHA-256 over the raw bytes of the files that make up the corpus, in
a canonical order the adapter fixes, streamed through `Checksum.of_chunks` so a corpus of several
gigabytes never sits in memory. For C-MAPSS the files are `train_FD00x.txt` for the selected
subsets, always in FD001–FD004 order whatever order the caller named them in, so two readers over
the same files agree and the checksum is the provenance of exactly the bytes that were read. A
change of a single byte — a value, a separator, a line ending — is a different version. That is
accepted: bytes are what a publisher ships and what a mirror can be checked against; equivalence
under reformatting is not a property the Catalog promises.

**A version covers the pretraining side of a data set.** The C-MAPSS reader registers the training
trajectories only. The official test trajectories and their remaining-useful-life targets are
labelled evaluation data: labels are the language of Evaluation, not of the Catalog, and a version
whose checksum covers only the training files is, by construction, the proof that no test engine
took part in pretraining. How the test portion is catalogued is decided when the first downstream
task is designed: as a second corpus of the same source with its own checksum, or through
an adapter of the Evaluation context. Not here, and not by widening this version.

**What the counts mean.** `CorpusContent` carries `unit_count` and `observation_count` in the
vocabulary of ADR-0008: a unit is what a split happens on (an engine, a machine, a stay), an
observation is one channel at one instant, and overlapping windows multiply neither. A unit may
hold no observation (three PhysioNet stays carry descriptors only), so the two counts are
independent. The reader's counts over the full corpus are checked against the facts the data spike
measured from the same files, where the raw data is present.

**Channels are the 21 sensors, named after the source.** The reader names channels and units after
the table in Saxena, Goebel, Simon and Eklund (2008) in file column order (`T2` in °R, `Nf` in
rpm, ratios without unit). The three operational settings are inputs that vary per cycle in FD002
and FD004 and are not channels. A mislabelled sensor would be corrected by a new version, which is
the mechanism the aggregate already has for a change of schema.

**Registration is one transaction.** `RegisterCorpusVersion` describes the data, opens a version,
records the content, freezes it and saves the corpus once; the event `CorpusVersionFrozen` is
published after the save. A failed reading or data already frozen under another version leaves the corpus as it
was, and no subscriber learns of a version that was not stored. Creating a corpus is a separate
use case (`RegisterCorpus`), so a misspelt name fails as an unknown corpus instead of creating a
second one. Corpus names are unique across the repository, enforced by `CorpusRepository.save`.

## Consequences

- Every reader defines its canonical file set and order in one place and documents it in its class
  docstring; the checksum of a version can be recomputed from the published files by anyone.
- Registering the same files twice fails with `SameDataAlreadyFrozenError`; registering modified
  files yields a new version with a different checksum and the same schema and regime. Both are
  contract tests of the port, run against every adapter.
- The sample of two FD001 engines in `tests/data/cmapss/` is kept byte-exact (rows end with two
  spaces); a whitespace-trimming editor would change the checksum the tests pin.
- A reader that also has to stream values into a tokeniser grows a second method on the
  same port; `describe()` keeps its single-pass, keep-nothing contract.
- `CorpusRepository` has one adapter until the first persistence ticket adds the SQLAlchemy one;
  the contract test is parametrised by a list of adapters so that the second joins with one entry.

## Alternatives considered

- **Checksum over parsed values in a canonical serialisation.** Invariant under reformatting, but
  it needs a serialisation per corpus and a parser bug changes the identity of unchanged data.
  Rejected: bytes are checkable against the publisher without our code.
- **Checksum of the downloaded archive.** Cheapest, and the fetch script already records it. But
  it covers documentation, test files and targets alike, so it cannot say what was pretrained on,
  and it breaks as soon as the data is registered from unpacked files. Rejected.
- **One checksum per file, stored as a list.** More diagnostic on a mismatch, but a version needs
  one identity to compare and to publish; the ordered concatenation carries the same information
  for that purpose. Rejected for now; per-file digests can be a reader-side diagnostic later.
- **A version covering the whole data set, test trajectories and targets included.** One
  provenance for both consuming contexts, with the official split persisted by Evaluation. Rejected
  because the Catalog would hold bytes it does not model, and the proof that pretraining excluded
  the test set would move from the checksum to a tokenisation manifest.
- **`describe(location)` on the port.** Separates the format (chosen by composition) from the
  location (data of the request). Rejected because it puts a file path into a port of the hexagon
  and gives the in-memory adapter an artificial key; binding the adapter to its source matches
  the other adapters in the repository.
- **One find-or-create use case for corpus and version.** One command for a future CLI, but upsert
  semantics turn a typo into a new corpus and widen the repository port. Rejected.
