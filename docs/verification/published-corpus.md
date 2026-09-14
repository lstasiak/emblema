# Published corpus

What a tokenised corpus costs once it is an artifact, and what publishing it twice gives. The
tables below are produced by `scripts/published_corpus_report.py`, which publishes the corpus into
a local directory store so that the numbers are about the format and the machine rather than about
a network. The remote half — publishing into the bucket and mounting it from another workspace —
is a test rather than a report: `tests/entrypoints/cli/test_publishing_to_the_bucket.py`, marked
`integration`.

Run it where the raw corpus is:

```
uv sync --all-extras
uv run scripts/published_corpus_report.py --corpus cmapss
```

## 2026-09-12 — Windows AMD64

|  |  |
| --- | --- |
| Machine | Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Corpus | cmapss, window length 50 stride 5 |
| Units | 709 indexed, 0 yielding no window |
| Split | 568 training, 141 validation, seed 1 |
| Windows | 25,395 |
| Tokens | 26,664,750 |
| Block | 454.0 MB (17.5 KiB per window) |
| Manifest | 19.5 KiB |
| Publishing | 61.5 s |
| Data spike | 709 units, 3,367,539 observations |

Publishing the same corpus twice gives one block and **two manifests**.
The blocks agree, so the data is one artifact. The manifests differ because each run registers the
corpus afresh and mints a new version identifier: no registration outlives the process yet. Once
one does, the second run finds the version already frozen and describes it rather than minting
another.

| Reading back | over 512 windows |
| --- | --- |
| One window from the map | 0.42 ms |
| A batch of eight | 3.4 ms |
| Held as objects | 115.2 KiB per window, 3.00 GB in all |
| On disk | 17.5 KiB per window, 0.45 GB in all |

### What the numbers say

**The memory map is a requirement, not a preference.** The same windows cost 3.00 GB as Python
objects and 454 MB as a block — six and a half times less, and the difference is the reason a
corpus this size can be opened at all inside a session with a few gigabytes of RAM. The measurement
agrees with the one taken in `loader-throughput.md` on a smaller sample (93.4 KiB per window there, 115.2 KiB here
on the full corpus, where windows are the same shape but the sample of them is larger).

**Reading a window back costs 0.42 ms**, so a batch of eight costs 3.4 ms against a model step of
221 ms measured on MPS in `loader-throughput.md` — a margin of about 65. The loader stays off the critical path,
but the margin is no longer the 100 measured over windows already in memory, and the cost is
almost all Python: converting a column to a tuple takes 0.11 ms and checking the window's
invariants takes the remaining 0.29 ms. A first version of the reader converted element by element
and cost 1.8 ms per window; `tolist` on each column took that to 0.42. What is left is the case
for a batch read that skips the per-window object altogether, which is written down as a revision
threshold in the ADR.

**The counts agree with the data spike** — 709 units and 3,367,539 observations become 25,395
windows of 26,664,750 tokens under the default window, exactly the figures the tokeniser measured. The
overlap of the default window multiplies the data by 7.9, which is what the block pays for on
disk and what the ADR names as the trigger for storing units with a window index instead.

**Publishing is 61.5 s** for the full corpus on this machine, of which tokenisation is about 49 s
when it was written. That is once, locally, and never again inside a session on rented hardware.

### Open

- ~~Publishing twice gives one block and two manifests, because nothing yet stores the
  registration of a corpus.~~ Closed the same day: the process now registers only what is missing
  and the registry is the metadata database (ADR-0016). See the last section below.
- ~~The remote leg has not run on this machine: it needs the object store from the settings.~~
  Run on the Mac (2026-09-12) against Garage locally and R2 remotely; see the last section.
- ~~The database leg has not run on this machine either.~~ Run on the Mac (2026-09-12): the
  repository contract against PostgreSQL, the comparison of the migrations with the model, the
  two-process publication and the bucket tests passed on the local stack, after
  `uv run alembic upgrade head`; `alembic check` reports nothing to migrate.
- ~~The remote bucket (R2) run is the one leg still outstanding.~~ Run on the Mac (2026-09-12):
  the store contract and the bucket test passed against R2 by configuration alone, and the full
  corpus was then published there. See the last section.

## 2026-09-12, after review — Windows AMD64

The same corpus, the same machine, after review changed the writer (every window is
cast to the stored width and rebuilt before it is buffered; scratch files beside the block; the
columns copied a chunk at a time) and moved the manifest's stored form into the Catalog's
published language. The block is unchanged: 454.0 MB, one artifact across both publications. The
manifest grew by 0.2 KiB with each channel's statistics beside it.

|  |  |
| --- | --- |
| Machine | Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Corpus | cmapss, window length 50 stride 5 |
| Units | 709 indexed, 0 yielding no window |
| Split | 568 training, 141 validation, seed 1 |
| Windows | 25,395 |
| Tokens | 26,664,750 |
| Block | 454.0 MB (17.5 KiB per window) |
| Manifest | 19.7 KiB |
| Publishing | 196.4 s |
| Data spike | 709 units, 3,367,539 observations |

Publishing the same corpus twice gives one block and **two manifests**, as in the morning.

| Reading back | over 512 windows |
| --- | --- |
| One window from the map | 1.02 ms |
| A batch of eight | 8.2 ms |
| Held as objects | 115.2 KiB per window, 3.00 GB in all |
| On disk | 17.5 KiB per window, 0.45 GB in all |

| Writing | over 512 windows |
| --- | --- |
| One window into the block | 1.87 ms |
| Of which checking it reads back as a window | 0.74 ms |
| Checking alone, over the whole corpus | 18.8 s |

### What the numbers say

**The machine, not the code, is what changed on the read side.** The reader was not touched by
the review, yet a window reads in 1.02 ms against 0.42 ms in the morning. The yardstick is the
last row of the writing table: checking a window's invariants is the same Python loop as before
and took 0.29 ms in the morning, 0.74 ms now — a factor of 2.5 on everything, from an IDE indexing
the fresh 454 MB files in the background. Every number in this section is to be read against that
factor; at the morning's speed the read is the same 0.4 ms and publishing is about 77 s.

**The write-time check costs what one read costs, once.** A window goes into the block in
1.87 ms, of which 0.74 ms is the rebuild that guarantees it reads back as a window; the rest is
casting five columns to the stored width and buffering them. Over the whole corpus the check is
18.8 s at this speed, about 7 s at the morning's — the writer now pays once what every reader
pays per window. Without the check a window in a block could fail its own invariants on read,
which the review reproduced with two channels observed closer together than a float32 tells
apart (`tests/shared/adapters/windows/test_window_block.py`).

**Publishing went from 61.5 s to about 77 s at like-for-like speed**, of which the check is 7 s
and the cast and buffering of already-rounded values most of the rest. That is once, locally,
and still a quarter of what a session on rented hardware would spend parsing the source files.

## 2026-09-12, publishing twice into one registry — Windows AMD64

The FD001 subset, published twice by two separately assembled processes sharing one registry.
The report keeps the registry in memory so that the numbers are about the format and the
machine; the database plays the same part for two real processes, in the integration test
`tests/entrypoints/cli/test_publishing_with_the_registry.py`.

|  |  |
| --- | --- |
| Corpus | cmapss FD001, window length 50 stride 5 |
| Units | 100 indexed, 0 yielding no window |
| Split | 80 training, 20 validation, seed 1 |
| Windows | 3,186 |
| Tokens | 3,345,300 |
| Block | 57.0 MB (17.5 KiB per window) |
| Manifest | 5.9 KiB |
| Publishing | 25.1 s |

Publishing the same corpus twice gives one block and **one manifest**: the second process found
the corpus under its name and the frozen version over exactly the data its reader saw, and
tokenised that version again into the same block, described by the same manifest under the same
reference. The DoD's identical checksum now holds for the artifact a run pins, not only for the
block behind it.

## 2026-09-12 — Darwin arm64

The reference machine, after the review and the persistence work, on the branch as pushed
(`771bf27`). The integration tests against the local stack and `alembic upgrade head` /
`alembic check` passed before this run.

|  |  |
| --- | --- |
| Machine | macOS-26.6.2-arm64-arm-64bit-Mach-O, arm |
| Corpus | cmapss, window length 50 stride 5 |
| Units | 709 indexed, 0 yielding no window |
| Split | 568 training, 141 validation, seed 1 |
| Windows | 25,395 |
| Tokens | 26,664,750 |
| Block | 454.0 MB (17.5 KiB per window) |
| Manifest | 19.7 KiB |
| Publishing | 39.9 s |
| Data spike | 709 units, 3,367,539 observations |

Publishing the same corpus twice gives one block and **one manifest**.

| Reading back | over 512 windows |
| --- | --- |
| One window from the map | 0.27 ms |
| A batch of eight | 2.2 ms |
| Held as objects | 115.2 KiB per window, 3.00 GB in all |
| On disk | 17.5 KiB per window, 0.45 GB in all |

| Writing | over 512 windows |
| --- | --- |
| One window into the block | 0.58 ms |
| Of which checking it reads back as a window | 0.20 ms |
| Checking alone, over the whole corpus | 5.0 s |

### What the numbers say

**These are the numbers the ADR quotes.** The M1 is where the corpus is published and where the
loader was measured in `loader-throughput.md`, so its figures are the like-for-like ones: publishing the full
corpus takes 39.9 s including the write-time check, which costs 5.0 s of it; a window reads back
in 0.27 ms, a batch of eight in 2.2 ms against a model step of 221 ms — a margin of about 100,
back where `loader-throughput.md` measured it over windows already in memory.

**The block is the same file on both machines**: 454.0 MB, and the same bytes for the same data
and configuration, as the identical-checksum tests assert. With the registry in the database, the
second publication finds the first one's version and the manifest is the same reference as well.

## 2026-09-12 — published to the remote bucket

The full C-MAPSS corpus, published from the Mac into the Cloudflare R2 bucket with the registry
in the local PostgreSQL, after the branch was merged (`1b3a6b6`, PR #12):

```
uv run --env-file .env.r2 python -m emblema.entrypoints.cli.publish_corpus \
  --corpus cmapss \
  --root "data/raw/cmapss/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData" \
  --window 50 --stride 5
durable/sha256/a00c3865aba466147cc2fb731cc1903afcb91bdc94c2027379fab5232ca789c1
sha256:a00c3865aba466147cc2fb731cc1903afcb91bdc94c2027379fab5232ca789c1
```

The two lines are the reference a training run pins: the manifest's key and checksum. The key
carries no environment prefix; the store adapter puts the object under the prefix its settings
name. The registry was emptied first (`TRUNCATE catalog.corpus_version, catalog.corpus`) so that
the registrations left by the integration tests could not answer for the corpus. The `.env.r2`
used for publishing carries the real database variables of the local stack; the `unused` values
suggested for it in `env.example` fit the store tests alone.
