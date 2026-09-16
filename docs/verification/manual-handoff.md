# The manual handoff: a run ordered here, made elsewhere, accepted back

Purpose: show that a backbone trained on a machine this system only hands work to ends up in the
registry with its provenance, and record what each step of the handoff costs. The decisions are
in ADR-0024; this note records what was observed, where, and with which versions.

The properties that must hold everywhere are tests, not numbers here: that a result of another
configuration, over other data, picked up from another checkpoint, made with other code, for
another backbone, or delivered a second time is refused, each with its own sentence
(`tests/pretraining/domain/handoff`, `tests/pretraining/adapters/handoff`,
`tests/pretraining/application/use_cases`); that the replaying runtime keeps every promise the
port makes (`tests/pretraining/ports/test_training_runtime_contract.py`, driven through a
simulated platform); that the three steps assembled by the composition root make one backbone
over the adapters they share (`tests/entrypoints/cli/pretrain`); that the registry's tables are
the ones the model describes (`tests/test_migrations.py`, `integration`). What this note adds is
the run on real infrastructure: the remote bucket, the local metadata database and the local
MLflow server, with this machine's accelerator standing in for the platform.

Method on any machine with the local stack up and the remote bucket configured:

```sh
uv sync --all-extras
docker compose up -d --wait && uv run alembic upgrade head
uv run --env-file .env.r2 python -m emblema.entrypoints.cli.publish_corpus \
    --corpus control-a --window 32 --stride 12 --validation-fraction 0.25
uv run --env-file .env.r2 python -m emblema.entrypoints.cli.pretrain order \
    --experiment experiments/control-a-s.toml --corpus <manifest key> <checksum> --run m1-leg
uv run --env-file .env.r2 python -m emblema.entrypoints.cli.pretrain run \
    --order <order key> <checksum> --device mps
uv run --env-file .env.r2 python -m emblema.entrypoints.cli.pretrain accept \
    --result <result key> <checksum> --track http://127.0.0.1:5000
```

The publishing step registers the corpus in the local Catalog and puts its block and manifest in
the remote bucket; the three pretraining steps then touch the bucket, the local registry and the
local MLflow server exactly as a run split between this machine and a notebook would, except that
the `run` step happens here too.

## Runs

### 2026-09-16 — Darwin arm64, M1 Pro, MPS, fp32, against Cloudflare R2

|  |  |
| --- | --- |
| Machine | macOS 26.6.2, Apple M1 Pro, 32 GB; Python 3.14.7; torch 2.14.0; mlflow 3.16.0; sqlalchemy 2.0.52; alembic 1.20.0 |
| Code | `1e61ecdc73f73fe3089d8d2a506dd40d5610a91e-dirty` — the working tree of this ticket before its commits, which is what the order and the result both carry |
| Experiment | `control-a-s`: tier S, 24 epochs, batch 32, seed 1, checkpoint every 200 steps |
| Corpus | `control-a`, window 32 / stride 12, validation 0.25, seed 1; manifest `durable/sha256/fbab5af1…`, block `39c03b8c…`; 4,732 training and 1,534 validation windows over 9 channels; version `1b10a81c-4d15-446c-ab69-512c40967367` in the local Catalog |
| Backbone | `ab200781-e406-4591-84f5-fd58ab2b2fb2`, `control-a-s/m1-leg`, 1,787,136 parameters |
| Order | `durable/sha256/fafd6df4…`, 917 bytes |
| Result | `durable/sha256/bca11f65…`, 9,239 bytes: 24 epochs, 17 checkpoints (all transient), weights `durable/sha256/88bce440…` (9.0 MB) |

| Step | Wall clock | What it did |
| --- | --- | --- |
| `publish_corpus` | 20.1 s | generated the corpus, registered it, wrote the block and the manifest to R2 |
| `order` | 3.6 s | described and read the corpus from R2, computed the signature, saved the ordered backbone in Postgres, placed the order in R2 |
| `run` (MPS) | 12 min 23 s | read the order and the corpus from R2, trained 24 epochs, uploaded 17 checkpoints and the weights to R2, reported the result to R2 |
| `accept` | 6.9 s | read the result from R2 and the backbone from the registry, held the one to the other, replayed 24 epochs into MLflow, delivered the weights, saved the ready backbone |
| `accept` again | refused | `BackboneAlreadyDeliveredError: backbone control-a-s/m1-leg already holds durable/sha256/88bce440…` |

Epochs took 19.1 s at the least, 33.2 s at the median and 41.1 s at the most, 733 s in all; the
same experiment took 19–23 s an epoch when its checkpoints went to memory
(`masked-reconstruction.md`, 2026-09-16), so the difference is the upload of a 27 MB checkpoint
every 200 steps to a bucket on another continent. The validation loss fell from 0.3949 to 0.0138
under a hidden share of 0.460, which is the curve the experiment produces (`training-loop.md`).

What the registry holds afterwards (`pretraining.backbone` and `pretraining.pretraining_input`,
read back through SQL):

```
status = ready            seed = 1        parameter_count = 1787136       tier = S
git_commit = 1e61ecdc73f73fe3089d8d2a506dd40d5610a91e-dirty
signature = 21c07d44e1c35c5cce640083b1b95e7e157dbd96ce4113e7e6eb2062fb602c67
artifact_key = durable/sha256/88bce4402412397f5474443187b1bb15e5e3435fc16a5929e8a034ff32e80adb
ordered_at = 2026-09-16 10:57:07 UTC      delivered_at = 2026-09-16 11:10:07 UTC
input: corpus = control-a, corpus_version = 1b10a81c-…, manifest = durable/sha256/fbab5af1…,
       block = 39c03b8c…, vocabulary_size = 9
```

What MLflow holds: one run named `m1-leg` under the experiment `control-a-s`, status `FINISHED`,
24 points on `validation_loss` from 0.3949 to 0.0138, and the tag `backbone_checksum` equal to the
checksum of the artifact the registry names — the run was recorded by the replay, thirteen minutes
after it happened on the accelerator, and reads as though it had been recorded live.

### What reading a published corpus costs

The C-MAPSS corpus published in the previous ticket (`published-corpus.md`, manifest
`durable/sha256/a00c3865…`, block 454 MB) read through the new reader, from R2, on the machine that
published it and therefore holds the block under its digest in `data/artifacts/`:

| Call | Wall clock | What it fetched |
| --- | --- | --- |
| `describe` | 0.41 s | the manifest, 19.5 KiB: corpus `cmapss`, version `e143f4b4-…`, 21 channels |
| `read` | 0.23 s | nothing: the block was mapped from the workspace; 20,237 training and 5,158 validation windows |
| first window | 1.6 ms | 1,050 tokens out of the map |

A machine without the block fetches it once, 454 MB from the bucket, and maps it from then on;
that cost is the bucket's transfer rate and was not measured here. These numbers predate the
reader hashing a block it finds in the workspace before mapping it; `read` now pays one pass over
the file, to be measured when the leg is repeated.

### What the numbers say

- A backbone trained by a process that never touched the registry ends up in it with everything
  the acceptance checked: the configuration, the corpus as published, the code's revision, the
  signature of the run and the weights. The `-dirty` suffix on the commit is right: the tree was
  not at its commit, and a result from a clean checkout at the same commit would have been
  refused. The run predates the rule that an order is placed only from a committed tree.
- The handoff itself costs seconds; the run costs what the run costs, plus the checkpoints'
  journey to the bucket. On a platform with the bucket in the same region the second term shrinks.
- Ordering reads the corpus's windows, not only its manifest, because the signature covers how
  many windows each side of the split holds and the manifest states only their total. Accepting
  reads them too, but for another reason: the check against the order needs no windows, since the
  signature is in the registry; the replay goes through the training runtime's port, whose input
  is a corpus. On the machine that published the corpus that is a map of a file already there; on
  another machine it is one download.

### Open

- The checkpoint reference a dropped session should be resumed from is known to MLflow when the
  run is tracked and to nobody when it is not: the notebook has to print it. Which of the two
  remedies ADR-0024 names is taken is decided with the first run on a platform.
- Every run of this note was made on a dirty tree and cannot be reproduced from a commit alone;
  the commit the note names is the base. An order is now refused from such a tree, so the leg is
  to be repeated from the commit that carries these changes, which will also remeasure `read`
  and fill the result columns the registry gained since.
