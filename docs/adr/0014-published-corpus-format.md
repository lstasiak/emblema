# ADR-0014: A published corpus is a block of windows beside a manifest, in a format we own

- Status: accepted
- Date: 2026-09-12
- Full text before condensation: commit `5f14447`

## Context

Preprocessing runs once, on a machine that holds the raw data; every training run then mounts the
result. The tokenised corpus is an artifact that must say what it holds, whether a run wants it,
and whether producing it again gives the same bytes. Measured on full C-MAPSS at the default
window: 25,395 windows, 26.7M tokens; **3.00 GB** as Python objects, **454 MB** as flat columns;
the overlap multiplies the data by **7.9**. A token has no meaning outside its window (time and the
first gap are relative to it).

## Decision

- **Two artifacts: a block and a manifest**, both content-addressed. The manifest (a few KiB) names
  the block by checksum, so a run can check suitability before fetching the corpus.
- **The manifest is the Catalog's published language**: `PublishedCorpusManifest` (corpus version
  and checksum, block reference, window, vocabulary with statistics, unit keys in block order,
  split sides), key-ordered JSON by a standard-library codec in `catalog/contracts` (ADR-0005).
  The domain's `TokenisationManifest` is translated by an assembler.
- **The block holds materialised windows**, not units plus a window index. The latter is ~8×
  smaller but puts half the tokeniser on the reading side — train/serve skew, refused in ADR-0012
  and ADR-0013.
- **Our own format**: magic number, length-prefixed JSON header, columns aligned to 64 bytes. The
  header gives each column's name, type with explicit byte order, length and offset, so the file
  can be mapped from the header alone. Per window: unit index, span of the unit's time axis, offset
  of the first token. Per token: channel id, value, time, gap, timeless flag.
- **Units are indices in the block and names in the manifest**, so the block codec lives in
  `shared/adapters/windows` and imports no context.
- **Measurements at the encoder's width** (float32 by default, recorded in the header). The writer
  casts and rebuilds every window and refuses one that stops being valid (two tokens collapsing to
  one float), once at write time instead of in a run.
- **The writer streams**: token columns go to scratch files beside the block (never the system tmp,
  which may be memory in a container), copied in chunks at the end. Memory is flat; transient disk
  is twice the artifact.

## Consequences

- Full C-MAPSS: block 454 MB, manifest 19.7 KiB; publishing 39.9 s on the M1 (5.0 s of it the
  write-time check). Reading a window from the map costs 0.27 ms; a batch of eight costs 2.2 ms
  against a 221 ms model step (`docs/verification/published-corpus.md`).
- The same data and configuration give the same block. The manifest gets a new version id per
  registration until registrations persist.
- The format is a liability we accept: a reader outside this repository needs the description.

## Alternatives considered

- **Arrow IPC**: zero-copy and Parquet-ready, but per-window random access would go through
  pyarrow's buffers — the read path we tune would be someone else's code.
- **safetensors**: a standard for weights; its numpy API reads tensors whole.
- **A directory of `.npy` files**: several keys whose combined checksum is defined outside the
  store.
- **float64 measurements**: ~47 % more disk on every corpus to guard a case the corpora in scope do
  not have.
- **Manifest codec in `catalog/adapters`**: the Pretraining adapter that reads it may not import it.

## Revisit when

- An artifact exceeds the free bucket (10 GB) or overlap exceeds ten → store units with a window
  index, splitting tokenisation into a window-independent part and a shared window-applying part.
- Per-read window validation shows in a training step → a batch read from columns to tensors.
- The precision check refuses a window of a corpus in scope → float64 for that block.
- Public distribution → a Parquet export beside the block.
