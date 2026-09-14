# ADR-0014: A published corpus is a block of windows beside a manifest, in a format we own

- Status: accepted
- Date: 2026-09-12

## Context

Preprocessing is done once, on a machine that holds the raw data and has no session to run out of;
every training run afterwards mounts the result and starts training without parsing a source file.
That makes the tokenised corpus an artifact, and an artifact has to answer three questions: what
it holds, whether it is the one a run wants, and whether producing it again produces the same
thing.

Three measurements decide the shape of the answer. The full C-MAPSS corpus under the default
window is 25,395 windows of 26,664,750 tokens; held as Python objects those windows cost **3.00 GB**
and as flat columns **454 MB**. The overlap of the default window multiplies the data by **7.9**:
there are 3.4 million observations behind those 26.7 million tokens. And a window's tokens are not
independent of the window — `time` is a fraction of the window's own length and the first token of
a channel measures its gap from the window's start — so a token has no meaning outside the window
it was cut for.

The constraints are the repository's usual ones. The application layer may not import numpy
(`pure-core`), so serialisation cannot happen there. `shared/` may not import a bounded context, so
nothing in `shared/adapters` may know what a Catalog manifest is. A context may not import another
context's domain or adapters, so anything the Pretraining and Evaluation contexts both read has to
live in `shared/` or in the Catalog's `contracts` — the argument that produced ADR-0012 and
ADR-0013, now for a third codec of the same representation and for the first message that travels
through the artifact store.

## Decision

**A published corpus is two artifacts: a block and a manifest.** The block holds the windows; the
manifest describes them and names the block by checksum. Both are content-addressed in the artifact
store, so a manifest can be fetched and read — a few kilobytes — to decide whether a corpus suits a
run before the corpus is fetched at all. That is what the compatibility check of the manual handoff
adapter needs, and it is also what makes the separation more than tidiness.

**The manifest's stored form is the Catalog's published language.** The handoff adapter that reads
it belongs to the Pretraining context, which may import nothing of the Catalog but `contracts`. So
the manifest travels as `PublishedCorpusManifest` — corpus version and checksum, block reference,
window, the vocabulary with each channel's statistics beside it, unit keys as text in the order the
block indexes them, the sides of the split sorted — encoded as key-ordered JSON by a codec in
`catalog/contracts` that needs nothing but the standard library. The Catalog's own value,
`TokenisationManifest`, stays in its domain; an assembler in the application layer translates it
outwards and, because the Catalog reads its own manifests back, inwards again (ADR-0005).

**The block holds materialised windows, not units with a window index.** Storing tokenised units
and cutting windows at read would be about eight times smaller, but a stored token would then be a
*pre-token*: the window-relative time and the first gap would have to be computed on the reading
side, putting half the tokeniser in a second place. That is the train/serve skew ADR-0012 and
ADR-0013 already refused twice. The corpus pays 454 MB instead of 57 MB for keeping one definition
of a token.

**The format is ours: one file, a magic number, a length-prefixed JSON header, then columns each
aligned to 64 bytes.** The header names every column, its element type with an explicit byte order,
its length and its offset, so a reader that has the format description and the header can map the
file without any code from this repository. Columns addressed by window are the unit index, the
span of that unit's time axis, and the offset of the window's first token; columns addressed by
token are the channel identifier, value, time, gap and the timeless flag.

Units are **indices** in the block and **names** in the manifest. The block therefore holds numbers
and nothing a corpus would call a word, which is what lets its codec live in `shared/adapters/windows`
beside the numpy and torch codecs of the same representation, importing no context.

**Measurements are stored at the width the encoder reads** — float32 by default, recorded in the
header. Storing them wider would keep precision that `TokenTensors` discards on its way to the
model. The price is that a block is a round trip *up to that width*, and the writer pays it at
write time: every window is cast to the stored width and rebuilt before it is buffered, and one
that is no longer a valid window — rounding is monotone but not one-to-one, so two tokens that
differ in the last places of a double can collapse into one float and stand in the wrong order —
is refused there, once, instead of failing the run that reads it. A block written at float64 has
no such case and reads back exactly; the header says which.

**The writer streams.** Each token column goes to a scratch file beside the block — never in the
system's temporary directory, which inside a container may be memory — as it fills, and the
columns are copied into the block a chunk at a time at the end, when the header can state where
each begins. Memory does not grow with the corpus; transient disk is twice the artifact.

## Consequences

A block of the full C-MAPSS corpus is 454 MB and its manifest 19.7 KiB; publishing takes 39.9 s
on the M1 that publishes it, once, of which the write-time check is 5.0 s — the writer pays per
window what every reader pays per window (0.20 ms) plus the cast of five columns. Reading a window
back out of the map costs 0.27 ms, so a batch of eight costs 2.2 ms against a model step of
221 ms on the same machine, a margin of about 100 (`docs/verification/published-corpus.md`;
the Windows sections there were taken on a slower, then a loaded, machine and say so).

Because the artifact store addresses by content, the block of a corpus republished from the same
data and configuration is the same artifact. Its manifest is not, today: each run registers the
corpus afresh and mints a new version identifier, because no registration outlives the process.
That is a gap in persistence rather than in the format, and it has to close before the handoff
adapter pins a manifest by version rather than by block checksum.

The block just published stays in the archive's workspace under its digest, where a fetch would
have put it, so a run on the publishing machine reads it from disk.

The format is a liability we accept: it is ours to keep working, and a reader outside this
repository needs the description above. The alternative was to inherit someone else's reader.

## Alternatives

**Arrow IPC.** Genuinely zero-copy over a memory map, ragged data through offsets, and Parquet for
free if the corpus is ever distributed. Rejected on ownership of the read path, not on checksums:
random access to one window out of a mapped file, at the granularity of a token, would go through
pyarrow's buffers and record batches, and the reader this project measures and tunes per window
would be someone else's code. The checksum argument sometimes made against Arrow is weaker than it
sounds — Arrow IPC is a versioned specification and the bytes are deterministic for fixed writer
options — and is not what decided this.

**safetensors.** One file, a JSON header, mmap, and the standard of the ecosystem — but the
standard for *weights*, not for datasets, and its numpy API reads a tensor whole rather than
mapping it, so random access per window would stop being zero-copy. If the corpus were ever
published, a dataset repository would want Parquet, which safetensors does not give either.

**A directory of `.npy` files.** `np.load(mmap_mode="r")` works out of the box and needs no format
of ours. Rejected because the artifact would become several content-addressed keys with a checksum
of the whole defined outside the store, which leaks the format into the store's key space and makes
"the same configuration yields the same checksum" a statement about a convention rather than about
a file.

**Storing units with a window index** — see above; rejected on the train/serve skew, not on size.

**Measurements at float64.** Removes the write-time check and the precision caveat for 8 more bytes
per token, about 47 % more disk on every corpus. Rejected: the case the check guards against is a
sub-float32 separation of two channels' observations, which the corpora in scope do not have, and
the cost would be paid by every corpus for it.

**The manifest as a domain value with a codec in `catalog/adapters`.** The first shape. Rejected
after review: the one adapter that has to read the manifest lives in another context and may not
import it.

## Revisit when

- An artifact exceeds the storage tier (10 GB on the free bucket) or the window overlap exceeds
  ten. At 17 bytes per token and ×7.9 overlap, ESA-ADB subsampled to the 50–100 million
  *observations* the corpus budget allows would be a block of 6.8–13.6 GB and cross the tier; subsampled to that
  many *tokens after windowing* it would be 0.85–1.7 GB and not. Which reading the budget meant is
  settled when the ESA subsampling strategy is, before the first paid run. Then storing tokenised units with a
  window index becomes worth its cost — and the way to do it without a second tokeniser is to cut
  the tokenisation in two, a part independent of the window and a part that applies the window,
  with the second shared by the writer and the reader.
- Checking a window's invariants on every read starts to show in a training step. The block would
  then grow a batch read that goes from columns to tensors without building a window object, which
  is the extension ADR-0013 already anticipated for the dataset.
- The write-time precision check refuses a window of a corpus in scope. Then the stored width for
  that corpus becomes float64, per block, through the header the format already has.
- The corpus is distributed publicly, or a reader outside this repository has to open it. Then a
  Parquet export is written *beside* the block, not in place of it: the internal format is chosen
  for the loader, a distribution format for its audience.
