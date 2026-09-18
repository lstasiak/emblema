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

### 2026-09-18 — Kaggle, Tesla T4, CUDA 12.8, fp32 and fp16, against Cloudflare R2

The first runs on the platform the handoff was designed for: three orders placed here from a
committed tree, fulfilled in one Kaggle notebook that installed the package at that commit, and
accepted back into the local registry. The corpus is C-MAPSS as published to the bucket
(`published-corpus.md`; manifest `durable/sha256/a00c3865…`, block `bbee7fcd…`, 454 MB, 21 channels, 20,237
training and 5,158 validation windows of 1,050 tokens); the experiments are the three files of
`experiments/backbone-cmapss-m*.toml`, which differ in the shape and the precision alone and spend
the budget the saturation measurement spent over this corpus, 2,532 steps of batch 32.

|  |  |
| --- | --- |
| Platform | Kaggle notebook, accelerator "GPU T4 x2", one device used; Python 3.12; torch 2.10.0+cu128 as preinstalled, CUDA 12.8, `Tesla T4`, capability (7, 5); numpy 2.5.3, boto3 1.43.36, mlflow-skinny 3.16.1, pydantic 2.12.3 after `pip install "emblema[ml,tracking] @ git+https://github.com/lstasiak/emblema@061fd05…"` |
| This machine | macOS 26.6.2, Apple M1 Pro; Python 3.14.7; torch 2.14.0 — orders and acceptances only |
| Code | `061fd0543bfdab7049ace288701d7b100e91b0ee`, read by the notebook from the installed package's `direct_url.json` and by this machine from the tree; every order and every result carries it |
| Bucket | the remote bucket, six `EMBLEMA_ARTIFACT_STORE__*` variables from the notebook's secrets |

Before any order, the notebook confirmed what the wheels note had only read off a repository:
the platform's Python is 3.12, the package installs from a commit without a token, the commit is
read back from the installation, the platform's own CUDA torch satisfies `torch>=2.9` and is
kept, and the manifest and the block are readable from the notebook with the bucket's
credentials. pip reported version conflicts against packages preinstalled on the platform
(`numba`, `ydata-profiling`, `google-colab`, `moviepy`) over the numpy it raised to 2.5.3; none
of them is on this package's path.

| Experiment | Shape | Precision | Backbone | Result | Weights | Steps | Wall clock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `backbone-cmapss-m-small-fp32` | 192×4, 1,789,440 | fp32 | `b6b04ba2-…` | `durable/sha256/88ba80d1…` | `durable/sha256/8c79085b…` | 2,532 | 22 min 59 s |
| `backbone-cmapss-m-small` | 192×4, 1,789,440 | fp16 | `5da9b8ce-…` | `durable/sha256/6b80fa2e…` | `durable/sha256/3d208e70…` | 2,532 | not recorded; 7 min 26 s of epochs |
| `backbone-cmapss-m` | 256×6, 4,751,872 | fp16 | `de3815c9-…` | `durable/sha256/dd966da6…` | `durable/sha256/6830e117…` | 2,532 | 13 min 12 s |

The parameter counts are the encoder's, as the registry records them; each run's objective adds
one decoder block. Each run wrote five checkpoints to the bucket's transient prefix, one every
500 steps, and the notebook printed nothing between fetching the block and printing the result's
reference — a run in this process has no log, and its progress is read off the platform's GPU
gauge, which stood at 90–100 % throughout: the accelerator, not the loader in the main process,
is what a step waits for.

Loss per hidden token at the end of each epoch, validation over the whole held-out side (2,511,314
hidden tokens, the same for every run; the channel mean scores 1.0208 there):

| Epoch | fp32 192×4, training / validation | fp16 192×4, training / validation | fp16 256×6, training / validation | Seconds, fp32 / fp16 small / fp16 256×6 |
| --- | --- | --- | --- | --- |
| 1 | 0.3314 / 0.01237 | 0.3436 / 0.01199 | 0.3520 / 0.01100 | 329 / 104 / 193 |
| 2 | 0.00777 / 0.00630 | 0.00779 / 0.00636 | 0.00732 / 0.00617 | 336 / 110 / 190 |
| 3 | 0.00572 / 0.00528 | 0.00590 / 0.00551 | 0.00558 / 0.00507 | 338 / 112 / 190 |
| 4 | 0.00491 / 0.00498 | 0.00515 / 0.00524 | 0.00475 / 0.00469 | 343 / 120 / 202 |

Per optimiser step, validation passes and checkpoint uploads included: 0.53 s in fp32 and 0.18 s
in fp16 at the small shape, 0.31 s in fp16 at the reference shape. The saturation measurement's
run of the same experiment on this machine (MPS, fp32, validation on every fourth window) ended
at 0.0049 training and 0.0047 validation.

What the registry holds afterwards: three rows in `pretraining.backbone` in state `ready`, tier
`M`, seed 1, commit `061fd05…`, each with its signature, its artifact key and its input row
naming the corpus version `e143f4b4-…`, the manifest and the block; the acceptances replayed the
epochs into the local MLflow server under the three experiment names, run `kaggle-t4` each.

#### What the numbers say

- **The path works end to end on the platform it was made for**, from a committed tree: the
  notebook read the order and the corpus from the bucket, trained, uploaded its checkpoints and
  its weights, reported, and the registry accepted each result against the order it had placed.
  The one `-dirty` run of the first leg is superseded.
- **The CUDA path reproduces the development machine's curve.** In fp32 the training loss of the
  last epoch agrees with the MPS run to three significant figures (0.00491 against 0.0049); the
  validation figure differs by the windows it is scored on, all of them here against a quarter
  there.
- **Half precision costs 5 % of the loss at this budget and buys a threefold speed-up.** The
  small shape ends at 0.00524 in fp16 against 0.00498 in fp32 on the same windows, and at
  0.00515 against 0.00491 in training. Two things separate the runs: what fp16 does to the
  arithmetic, and the steps the gradient scaler skipped. The loss is summed over a batch's hidden
  tokens rather than averaged, so the scaler's starting scale of 65,536 overflows fp16 by
  construction; emulated on the host before the run (CPU autocast to float16, the loop's own
  order of operations), the first steps were skipped with the scale halving each time, and a
  backward pass fitted at a scale of 2⁴ and overflowed at 2⁶ for either shape on a batch of 8 —
  so on a batch of 32 the scaler skipped a dozen or so of the 2,532 steps, all within the
  warm-up, and settled near 2³. The platform confirmed the mechanism without counting it: torch
  warned, once, that the scheduler had stepped before the optimiser, which is what a skipped
  step looks like from the scheduler's side. Which of the two accounts for the 5 % is
  not separated here; a second run of either configuration would put a device's own scatter
  beside it.
- **The reference shape lowers the loss at equal steps, a little.** 0.00469 against 0.00524
  under the same precision, 11 % lower on the held-out side, for 1.7× the time per step; the
  saturation measurement's reading that C-MAPSS is learnt to the objective's floor by the small
  shape stands, with the floor a little lower for the larger one.
- **The platform's cost per step is what prices the mixed run.** At the reference shape, in half
  precision, a step over 32 windows of 1,050 tokens costs 0.31 s on a T4 with everything
  included; the mixed run's windows are longer and its attention is quadratic in them, so this
  number is the floor of that estimate, not the estimate.

### Open

- The checkpoint reference a dropped session should be resumed from is known to MLflow when the
  run is tracked and to nobody when it is not: the notebook has to print it. The three platform
  runs were short enough not to need it, and the result document carries the last checkpoint of
  each epoch once the run has ended; a run of hours on the platform still needs one of the two
  remedies ADR-0024 names, decided before the mixed run.
- Resuming from a remote checkpoint has been tested through the port and on this machine's
  accelerator, not yet on the platform: the platform runs of 2026-09-18 left five checkpoints
  each in the bucket, and a second run of one of their orders from one of them is the test that
  remains, read against the scatter of two uninterrupted runs of the same order.
- The cost of `read` on a machine that fetches the block from the bucket was not measured on the
  platform: the run's wall clock includes it and the run itself does not time it.
