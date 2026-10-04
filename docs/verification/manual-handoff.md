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

### 2026-09-19 — Kaggle, Tesla T4, fp16, against Cloudflare R2: the backbone over the mixture

The first full pretraining: one order over the four corpora chained through one vocabulary
(ADR-0029), placed here from `main` and fulfilled in one Kaggle session with "Save & Run All",
accepted back the next morning. Everything the run is made of:

|  |  |
| --- | --- |
| Experiment | `experiments/backbone-mixed-m.toml`: tier M (256 wide, 6 blocks, 4,751,872 encoder parameters over 84 channels), fp16, Huber δ = 1, dropout 0, 8 epochs, micro-batch 16 with 2 accumulated, checkpoint every 2,000 steps, seed 1 |
| Corpora | C-MAPSS 50 / 5 (`a00c3865…`, channels 1–21) → SKAB 100 / 10 s (`166ca516…`, 22–29) → SMD 50 / 10 min (`c8a2c9ae…`, 30–67) → ESA-AD 1 / 1 h (`f0842859…`, 68–84); 186.6M tokens an epoch after windowing, of which ESA-AD 51.3M — the "50–100M tokens" of the programme are counted after windowing |
| Code | `265fc911958f6833d951dcc8a96909101c5fabf7`, installed in the notebook from the commit |
| Backbone | `058188f2-cba7-46fc-91dd-6a4663090146`, order `durable/sha256/138ecdda…`, result `durable/sha256/64bc7796…`, weights `durable/sha256/d52dfef2…` (the seventh epoch's) |
| Platform | Kaggle, GPU T4 x2 with one device used, Python 3.12, the platform's CUDA torch; the run's log on standard error, read live in the version's log and kept with it |
| Cost | 0.39–0.42 s an optimiser step (4,493 steps an epoch), 2,055–2,071 s an epoch with its four validation passes, 4.6 h of epochs in one session; no out-of-memory at 16 windows of 1,900 tokens in half precision, the memory itself not read off the platform |

Validation loss per hidden token relative to the channel-mean predictor's over the same
tokens, each corpus on its own held-out side, under the run's bounded reading; the mean over the
corpora is what keeps the epoch (`scripts/pretraining_curve_report.py`, curve stored under
`data/report/pretraining/<backbone>`; figure `figures/pretraining-curve-backbone-mixed-m.png`):

| epoch | training | cmapss | skab | smd | esa_ad | mean relative | seconds |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.09439 | 0.070 | 0.331 | 0.348 | 0.399 | 0.2870 | 2071 | kept |
| 2 | 0.05242 | 0.045 | 0.308 | 0.383 | 0.335 | 0.2679 | 2066 | kept |
| 3 | 0.04409 | 0.030 | 0.281 | 0.354 | 0.325 | 0.2477 | 2060 | kept |
| 4 | 0.03924 | 0.015 | 0.270 | 0.379 | 0.309 | 0.2433 | 2059 | kept |
| 5 | 0.03428 | 0.010 | 0.268 | 0.397 | 0.290 | 0.2411 | 2063 | kept |
| 6 | 0.03119 | 0.009 | 0.265 | 0.375 | 0.294 | 0.2356 | 2055 | kept |
| 7 | 0.02911 | 0.007 | 0.263 | 0.368 | 0.280 | 0.2295 | 2060 | backbone |
| 8 | 0.02835 | 0.007 | 0.263 | 0.373 | 0.281 | 0.2308 | 2058 |  |

Every number is validation, not test. What the run says:

- **Every corpus is learnt under the mixture**, none sits at the trivial predictor: C-MAPSS to
  0.7 % of it, the three others to 26–37 %. The satellite corpus, which its own curve under the
  bounded reading took four epochs to bring to 0.27, stands at 0.28 here after seven, so the
  mixture costs it nothing.
- **SMD rises from its first epoch** (0.348 → 0.397 at the fifth, 0.373 at the last) while the
  mean over the corpora still falls: the first condition ADR-0029 names for revisiting the
  weights of the mix. Its best epoch is the first, which the rule that keeps one epoch for the
  whole mixture cannot honour; the mixed backbone carries SMD at 0.368 rather than 0.348.
- **The mean flattens at the end**: 0.2295 at the seventh epoch, 0.2308 at the eighth, so the
  budget of eight epochs is not what limits this run and ADR-0008's second condition does not
  fire.
- **The first order of this run failed after one epoch** (`36416db3…`, commit `b9f3161`, 36
  minutes of the accelerator): the validation pass scored the loss in the prediction's half
  precision outside autocast, and the sum over one batch of a corpus with excursions of hundreds
  of deviations passed 65,504 and came back infinite. Training was never affected — autocast
  computes the loss in single precision — and the three C-MAPSS runs above never summed past a
  few tens. Fixed on `main` before the second order: the losses are read in single precision at
  least, whatever the prediction came in.
- **PyTorch warned once** that the scheduler stepped before the optimiser: the gradient scaler
  skips the first steps whose gradients overflow while it finds its scale, and the schedule
  counts them. A handful of steps out of 35,944, and the same on the fp16 C-MAPSS runs.
- **The log is what the dropped-session remedy came to** (ADR-0029): a progress line every 100
  steps with the pace and the time left, a line per checkpoint with its reference, a line per
  epoch with every corpus's loss. This run fitted one session, so the resume was tested on its
  own, below.
- **Resumed on the platform from a checkpoint inside an epoch.** A second session ran the same
  order from the checkpoint written at step 26,000, inside the sixth epoch (`transient/sha256/d67c6cde…`,
  the last checkpoint that epoch wrote), and finished the sixth, seventh and eighth epochs in
  618 + 2,018 + 2,022 s; result `durable/sha256/15baac32…`, which names the checkpoint it
  resumed from and signs the same run. Read against the uninterrupted run, relative validation
  per corpus:

  | epoch | cmapss | skab | smd | esa_ad | mean relative | uninterrupted mean |
  | --- | --- | --- | --- | --- | --- | --- |
  | 6 | 0.008 | 0.264 | 0.379 | 0.290 | 0.2354 | 0.2356 |
  | 7 | 0.007 | 0.263 | 0.365 | 0.282 | 0.2294 | 0.2295 |
  | 8 | 0.007 | 0.263 | 0.368 | 0.282 | 0.2299 | 0.2308 |

  The largest difference in any corpus's number is 0.005 (SMD, eighth epoch), the mean agrees to
  0.001 or better, and the resumed run keeps the same epoch, the seventh, as its backbone — an
  epoch's worth of the accelerator's own half-precision scatter, of the size the resume test of
  `training-loop.md` calibrated on this machine's accelerator. The resumed run inherited the best
  epoch the checkpoint carried (the fifth's 0.2411), improved on it at the sixth and seventh,
  and reports the sixth epoch's training loss over the batches after the checkpoint alone, as
  `training-loop.md` says it does. Its result is not accepted: the backbone was delivered by the
  first run and refuses a second delivery, which is the registry doing its job.

### 2026-09-21 — Kaggle, Tesla T4, fp16, against Cloudflare R2: the C-MAPSS backbone to its plateau

Four orders of the C-MAPSS backbone's experiment with the budget doubled each time — 8, 16, 32
and 64 epochs, files `experiments/backbone-cmapss-m-{8,16,32,64}.toml`, otherwise the file of the
backbone of 2026-09-18 — placed here from `96da833` (the first three) and `a726b7b` (the fourth),
fulfilled in two Kaggle sessions on "GPU T4 x2" (the runs of 32 on one device and of 8 then 16 on
the other in the first session, the run of 64 alone in the second) and accepted here. Every run:
batch 32, one micro-batch, peak 1e-3 with a quarter-epoch warm-up and a cosine decay to one per
cent spanning the run, checkpoints every 1,000 (8 epochs), 2,000 (16 and 32) or 4,000 steps (64).

| Experiment | Backbone | Result | Weights | Steps | Best epoch | Validation loss | Training loss | Minutes of epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `backbone-cmapss-m-8` | `86dd091c-…` | `durable/sha256/0c3b0cb8…` | `durable/sha256/c68cd866…` | 5,064 | 7 | 0.00291 | 0.00276 | 25 |
| `backbone-cmapss-m-16` | `0dade051-…` | `durable/sha256/37767e7d…` | `durable/sha256/d1692f10…` | 10,128 | 14 | 0.00208 | 0.00212 | 50 |
| `backbone-cmapss-m-32` | `9a1007fe-…` | `durable/sha256/b57ebb74…` | `durable/sha256/9759bb67…` | 20,256 | 30 | 0.00172 | 0.00166 | 96 (5,910 s wall) |
| `backbone-cmapss-m-64` | `ccd28046-…` | `durable/sha256/3f60cfa6…` | `durable/sha256/78b3c201…` | 40,512 | 64 | 0.00176 | 0.00175 | 189 (11,451 s wall) |

Validation loss per hidden token over the whole held-out side, as on 2026-09-18, where the run of
four epochs ended at 0.00469; the curves are stored under `data/report/pretraining/<backbone>`
by `scripts/pretraining_curve_report.py`, which reads the results of these runs and not the
older document of the four-epoch run.

What the runs say:

- **The doublings fall by 38, 29 and 17 per cent, then by nothing.** Under the rule the
  registration fixed for this (`docs/preregistration.md`, 2026-09-20 and 2026-09-21), the run of
  64 epochs is the backbone: the fourth doubling ended 2.5 per cent above the third, which is the
  size of the last epochs' own movement in either run.
- **Within a run the last third is flat and the schedule is why.** In the run of 32 the epochs
  from 27 on stand at 0.0017; in the run of 64 the epochs from 50 on stand at 0.0017–0.0018. A
  cosine decay to one per cent of the peak leaves the last epochs little to move by, so a run's
  own tail does not say whether a longer run would go lower; the doubling does.
- **Cost per step fell with the session**: 0.31 s on 2026-09-18, 0.29 s in the run of 32 and
  0.27–0.28 s in the run of 64, the platform's variation rather than anything in the code.
- **The weights of every new best epoch are written durably**, about 22 MB each: the run of 32
  left some twenty-five such objects in the bucket and the run of 64 some forty. The registry
  names one of them; the others are reachable by the result document alone.

The two sessions cost about 6.3 hours of the platform's GPU quota in all.

### 2026-09-21 — Colab, NVIDIA L4, fp16, against Cloudflare R2: the backbone over FD001 and FD003 to its plateau

The same ladder over `cmapss` published again with the subsets FD001 and FD003 alone, one
operating condition each (`durable/sha256/a9c73709…`; why, in `docs/preregistration.md`,
2026-09-21): the experiment files of 4, 8, 16, 32 and 64 epochs, placed here from `c1c8a9c` as
run `colab-fd13`, fulfilled on one L4 of a paid notebook with all five runs sharing the device,
and accepted here. Every run as before: batch 32, peak 1e-3 with a quarter-epoch warm-up and a
cosine decay to one per cent spanning the run, fp16, seed 1.

| Experiment | Backbone | Result | Weights | Best epoch | Validation loss | Share of the trivial predictor's | Training loss | Minutes of epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `backbone-cmapss-m` | `c2a16506-…` | `durable/sha256/a146b33e…` | `durable/sha256/16d9d2e7…` | 4 | 0.13830 | 0.212 | 0.1253 | 3 |
| `backbone-cmapss-m-8` | `101274ae-…` | `durable/sha256/8faf5a71…` | `durable/sha256/259fdc70…` | 8 | 0.13427 | 0.206 | 0.1230 | 16 |
| `backbone-cmapss-m-16` | `e0d2f3ab-…` | `durable/sha256/459fbe28…` | `durable/sha256/1999ecf7…` | 15 | 0.13207 | 0.203 | 0.1189 | 32 |
| `backbone-cmapss-m-32` | `7359e47a-…` | `durable/sha256/3971157d…` | `durable/sha256/696d75c2…` | 32 | 0.13079 | 0.201 | 0.1173 | 22 |

Validation loss per hidden token over the whole held-out side; the order of 64 epochs
(`7e010b17-…`) was stopped after about thirteen epochs, once the rule had named its backbone, and stays
ordered and unfulfilled.

What the runs say:

- **The doublings fall by 2.9, 1.6 and 1.0 per cent.** Under the rule of 2026-09-20 the first
  doubling already gains less than five per cent, so the run of 8 epochs is the backbone
  (`259fdc70…`); the run of 32 was let finish as the curve's own evidence, and 8 to 32 epochs
  together take 2.6 per cent off.
- **The pretext is a task again.** Over the four subsets the backbone's loss fell to 0.7 per cent
  of the trivial predictor's, because the operating condition, read off the other channels, gave
  every masked value away; normalised within one condition the loss stays at a fifth of the
  trivial predictor's, and what is left to predict is the engine's own state.
- **Cost**: one epoch takes about 30 s on the L4 alone and 120 s with four runs sharing it; the
  minutes above are the runs' own epochs, however many ran beside them.

### 2026-09-22 — Colab, NVIDIA A100, fp16, against Cloudflare R2: the backbone over the four subsets read per operating condition to its plateau

The same ladder over `cmapss` published again with all four subsets read per operating condition
(ADR-0034; manifest `durable/sha256/d63f8e1b…`, 126 channels, 20,160 training and 5,235 held-out
windows; why, in `docs/preregistration.md`, 2026-09-22). The experiment files of 4, 8, 16, 32 and
64 epochs were placed here from `fdf8053` as run `colab-cond`, fulfilled one after another on one
A100 of a paid notebook, and accepted here. Every run as before: batch 32, peak 1e-3 with a
quarter-epoch warm-up and a cosine decay to one per cent spanning the run, fp16, seed 1.

| Experiment | Backbone | Result | Weights | Best epoch | Validation loss | Share of the trivial predictor's | Training loss | Minutes of epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `backbone-cmapss-m` | `72917a2a-…` | `durable/sha256/27885a9f…` | `durable/sha256/96d9ec20…` | 4 | 0.22541 | 0.325 | 0.1641 | 8 |
| `backbone-cmapss-m-8` | `ed3b1fa4-…` | `durable/sha256/1a728b19…` | `durable/sha256/6283c210…` | 8 | 0.22326 | 0.322 | 0.1637 | 19 |
| `backbone-cmapss-m-16` | `d2913c98-…` | `durable/sha256/0759b266…` | `durable/sha256/8cd60452…` | 16 | 0.22136 | 0.319 | 0.1624 | 14 |
| `backbone-cmapss-m-32` | `eecf8ec4-…` | `durable/sha256/d6ac405f…` | `durable/sha256/28350a30…` | 31 | 0.21579 | 0.311 | 0.1551 | 55 |

Validation loss per hidden token over the whole held-out side. The order of 64 epochs
(`durable/sha256/ea532e01…`) never started: the stream that ran the rungs one after another was
stopped once the rule had named its backbone, while the run of 32 was already under way, and the
order stays placed and unfulfilled.

What the runs say:

- **The doublings fall by 0.95, 0.85 and 2.5 per cent.** Under the rule of 2026-09-20 the first
  doubling already gains less than five per cent, so the run of 8 epochs is the backbone
  (`6283c210…`); the runs of 16 and 32 finished as the curve's own evidence, and 8 to 32 epochs
  together take 3.3 per cent off.
- **The loss is not comparable in level with the ladder over FD001 and FD003**: the corpus, its
  channels and its held-out side differ. Within each ladder the doublings gain little.
- **Cost**: the device was shared with the sweeps of the task for most of the ladder, so an
  epoch took from 51 to 181 s; the minutes above are the runs' own epochs, however many ran
  beside them.

### 2026-09-29 — Kaggle, Tesla T4 x2, fp16, against Cloudflare R2: the intensive-care backbones

Two backbones for the intensive-care task: one over the stays of set A alone, named by the
doubling rule, and the mixed backbone of 2026-09-19 with those stays added as a fifth corpus.
Five orders placed here from `8b1b4349` as run `kaggle-t42c`, fulfilled in one "Save & Run All"
session with one device per stream, and accepted here the next morning.

|  |  |
| --- | --- |
| Experiments | `experiments/backbone-physionet2012-m-{8,16,32,64}.toml` and `experiments/backbone-mixed5-m.toml`: tier M, fp16, Huber δ = 1, dropout 0, micro-batch 16 with 2 accumulated, peak 1e-3 with a quarter-epoch warm-up and a cosine decay to one per cent spanning the run, seed 1. The mixture's file is the four-corpus file with `physionet2012` added and nothing else changed; its configuration in the registry differs from `058188f2`'s in the name and the corpora only |
| Corpora | PhysioNet 2012 set A (`cef44de2…`, channels 85–128 of the chained vocabulary); the mixture adds it after the four manifests of 2026-09-19 |
| Code | `8b1b4349641c28b601f076bd7eafa082bd659579`; between `265fc911` and this commit the training path changed only by modules moved |
| Parameters | 4,779,264 in every run, the registry's count: 11,264 more than `058188f2`, which is 44 channel rows of width 256 |
| Platform | Kaggle, GPU T4 x2: the mixture on one device, the four rungs one after another on the other |
| Cost | the rungs 25 s an epoch of 125 steps, 3 to 27 minutes a run; the mixture 0.43 s a step, 4,618 steps and 2,127–2,153 s an epoch, 4.8 h of epochs. `accept` took 5–9 s a run |

The ladder over the stays alone. Validation loss per hidden token over the whole held-out side;
the best epoch is the last in every rung:

| Experiment | Backbone | Result | Weights | Best epoch | Validation loss | Share of the trivial predictor's | Training loss | Minutes of epochs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `backbone-physionet2012-m-8` | `3c1a33cb-…` | `durable/sha256/b0a644ad…` | `durable/sha256/a64b19e3…` | 8 | 0.16837 | 0.478 | 0.1676 | 3 |
| `backbone-physionet2012-m-16` | `4743a67a-…` | `durable/sha256/2f9aac34…` | `durable/sha256/f55e5aba…` | 16 | 0.15161 | 0.430 | 0.1517 | 7 |
| `backbone-physionet2012-m-32` | `187235a1-…` | `durable/sha256/1cef575e…` | `durable/sha256/c5878c03…` | 32 | 0.14618 | 0.415 | 0.1442 | 14 |
| `backbone-physionet2012-m-64` | `9bcec41c-…` | `durable/sha256/8738ecf9…` | `durable/sha256/32311e43…` | 64 | 0.14280 | 0.405 | 0.1374 | 27 |

The mixture of five, `backbone-mixed5-m` (`2cfacb10-…`, result `durable/sha256/574e6566…`,
weights `durable/sha256/869ed545…`, the eighth epoch's), against the four-corpus backbone
epoch by epoch. Share of the trivial predictor's loss per corpus, four corpora then five:

| epoch | cmapss | skab | smd | esa_ad | mean of the four | physionet2012 | mean of the five |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.070 / 0.065 | 0.331 / 0.366 | 0.348 / 0.360 | 0.399 / 0.385 | 0.2870 / 0.2940 | 0.590 | 0.3533 |
| 2 | 0.045 / 0.048 | 0.308 / 0.291 | 0.383 / 0.393 | 0.335 / 0.333 | 0.2679 / 0.2662 | 0.552 | 0.3233 |
| 3 | 0.030 / 0.046 | 0.281 / 0.279 | 0.354 / 0.378 | 0.325 / 0.322 | 0.2477 / 0.2562 | 0.533 | 0.3116 |
| 4 | 0.015 / 0.035 | 0.270 / 0.265 | 0.379 / 0.364 | 0.309 / 0.333 | 0.2433 / 0.2493 | 0.519 | 0.3033 |
| 5 | 0.010 / 0.031 | 0.268 / 0.263 | 0.397 / 0.379 | 0.290 / 0.292 | 0.2411 / 0.2412 | 0.506 | 0.2942 |
| 6 | 0.009 / 0.030 | 0.265 / 0.263 | 0.375 / 0.361 | 0.294 / 0.291 | 0.2356 / 0.2362 | 0.494 | 0.2878 |
| 7 | 0.007 / 0.028 | 0.263 / 0.262 | 0.368 / 0.365 | 0.280 / 0.279 | 0.2295 / 0.2335 | 0.489 | 0.2846 |
| 8 | 0.007 / 0.026 | 0.263 / 0.260 | 0.373 / 0.364 | 0.281 / 0.277 | 0.2308 / 0.2318 | 0.487 | 0.2828 |

Every number is validation, not test. What the runs say:

- **The doublings fall by 9.96, 3.58 and 2.31 per cent** of the best epoch's loss. The first
  doubling to gain less than five per cent is 16 to 32 epochs, so by the rule stated in the
  experiment files before the ladder ran the backbone over the stays alone is the run of 32
  epochs (`187235a1-…`, weights `c5878c03…`). The run of 64 finished as the curve's own
  evidence.
- **The mixture teaches the stays about as much as the rung of equal exposure.** The stays take
  125 of 4,618 steps an epoch, 1,000 steps in all, as many as the rung of 8 epochs; the mixture
  ends at 0.487 of the trivial predictor's loss on them, the rung at 0.478.
- **Adding the stays leaves SKAB, SMD and ESA-AD where they were** or slightly lower at the last
  epoch; the mean over the four shared corpora ends at 0.2318 against 0.2308.
- **C-MAPSS ends about four times higher: 0.026 of the trivial predictor's loss against
  0.007.** The two runs part at the third epoch and the gap does not close. The manifest is the
  same (the same held-out tokens and the same trivial loss), the configurations differ in the
  corpora only, and the training code did not change. The cause is not known: one run of each
  mixture cannot separate the added corpus from the run's own spread. The intensive-care task
  does not read C-MAPSS, so its campaigns are not affected.

### 2026-10-02 — Kaggle, Tesla T4, fp16, against Cloudflare R2: the mixture without SMD

Two backbones over the mixture of five less SMD, for the question whether SMD teaches the
intensive-care task anything (`intensive-care-curve.md`, 2026-10-02): A at the mixture's eight
passes and B at thirteen, which spends about the mixture's number of steps on the four corpora
that remain. Ordered here from `b6569a82`, fulfilled on Kaggle, one device each, and accepted
here the same day.

|  |  |
| --- | --- |
| Experiments | `experiments/backbone-mixed5-nosmd-m.toml` and `experiments/backbone-mixed5-nosmd-m-13.toml`: the mixture's file without `smd`, everything else the same; B states 13 epochs and keeps the mixture's share of the run warming up |
| Code | `b6569a82`; the training path unchanged since the mixture's run |
| Parameters | 4,779,264 in both, the mixture's count: SMD's channel rows stay in the table, untrained |
| Platform | Kaggle, GPU T4, one device a run |
| Cost | 0.33 s a step: 896–920 s an epoch of about 2,790 steps, 2.0 h for A and 3.3 h for B; `accept` took 4–5 s a run |

The kept epoch is the best by the mean over the four corpora of the validation loss's share of
the trivial predictor's. Every number is validation, not test.

| Experiment | Backbone | Result | Weights | Epochs | Kept | cmapss | skab | esa_ad | physionet2012 | Mean of the four | Training loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `backbone-mixed5-nosmd-m` (A) | `1c1c57c6-…` | `durable/sha256/a78bfd6d…` | `durable/sha256/994d1642…` | 8 | 8 | 0.006 | 0.266 | 0.294 | 0.489 | 0.2639 | 0.03550 |
| `backbone-mixed5-nosmd-m-13` (B) | `2523c734-…` | `durable/sha256/21a5422b…` | `durable/sha256/b4d3ad42…` | 13 | 12 | 0.006 | 0.267 | 0.286 | 0.480 | 0.2597 | 0.03372 |
| `backbone-mixed5-m`, for comparison | `2cfacb10-…` | `durable/sha256/574e6566…` | `durable/sha256/869ed545…` | 8 | 8 | 0.026 | 0.260 | 0.277 | 0.487 | 0.2625 (and 0.364 on SMD) | — |

Epoch by epoch, as `scripts/pretraining_curve_report.py` writes them
(`data/report/pretraining/{1c1c57c6…,2523c734…}/epochs.csv`), A then B:

| epoch | cmapss | skab | esa_ad | physionet2012 | mean of the four |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.079 / 0.071 | 0.314 / 0.404 | 0.416 / 0.448 | 0.597 / 0.629 | 0.3514 / 0.3881 |
| 2 | 0.047 / 0.020 | 0.305 / 0.315 | 0.351 / 0.366 | 0.573 / 0.568 | 0.3188 / 0.3172 |
| 3 | 0.038 / 0.020 | 0.299 / 0.320 | 0.335 / 0.358 | 0.557 / 0.566 | 0.3071 / 0.3161 |
| 4 | 0.010 / 0.013 | 0.279 / 0.283 | 0.331 / 0.345 | 0.527 / 0.555 | 0.2869 / 0.2990 |
| 5 | 0.009 / 0.010 | 0.275 / 0.291 | 0.301 / 0.331 | 0.514 / 0.523 | 0.2746 / 0.2888 |
| 6 | 0.007 / 0.008 | 0.269 / 0.280 | 0.304 / 0.337 | 0.499 / 0.515 | 0.2699 / 0.2848 |
| 7 | 0.007 / 0.008 | 0.267 / 0.276 | 0.291 / 0.309 | 0.491 / 0.501 | 0.2640 / 0.2733 |
| 8 | 0.006 / 0.006 | 0.266 / 0.271 | 0.294 / 0.288 | 0.489 / 0.492 | 0.2639 / 0.2644 |
| 9 | — / 0.009 | — / 0.272 | — / 0.294 | — / 0.493 | — / 0.2669 |
| 10 | — / 0.006 | — / 0.268 | — / 0.284 | — / 0.486 | — / 0.2609 |
| 11 | — / 0.006 | — / 0.268 | — / 0.284 | — / 0.482 | — / 0.2600 |
| 12 | — / 0.006 | — / 0.267 | — / 0.286 | — / 0.480 | — / 0.2597 |
| 13 | — / 0.006 | — / 0.267 | — / 0.286 | — / 0.480 | — / 0.2597 |

What the runs say:

- **Without SMD, C-MAPSS ends at 0.006, where the mixture of four ended (0.007).** The mixture
  of five's 0.026 came with SMD's share of the run, not with the stays: both runs without SMD
  hold the stays and lose SMD, and both return C-MAPSS to its level.
- **The stays and the other corpora end where the mixture of five left them**, within 0.02 of
  the trivial predictor's share; A's eighth epoch matches the mixture's on the stays (0.489
  against 0.487).
- **Five more passes buy B 0.004 on the mean and 0.009 on the stays**, and its best epoch is
  the twelfth, not the last. What the two backbones do on the intensive-care task is in
  `intensive-care-curve.md` (2026-10-02): B's fine-tuning loses 0.057 to A's.

### 2026-10-04 — Kaggle, Tesla T4 x2, fp16, against Cloudflare R2: the mixture of four and its leave-one-corpus-out variants

Seven backbones for the transfer matrix, on a vocabulary chained anew from C-MAPSS read per
operating condition (`durable/sha256/d63f8e1b…`): SKAB (`3e9c8450…`, channels 127–134), the
satellite telemetry (`77350303…`, 135–151) and the intensive-care stays (`717bc832…`, 152–195,
one window of 48 hours and a minute per stay). SMD is left out, since it taught the task nothing
(2026-10-02). Ordered here from `a3633a54`, fulfilled on Kaggle in one session of two T4s, and
accepted here the next morning.

|  |  |
| --- | --- |
| Experiments | `experiments/backbone-mixed4-m.toml` (the four corpora), `backbone-mixed4-without-{cmapss,skab,esa_ad,physionet2012}-m.toml` (the same publications with one left out), `backbone-mixed4-stays-x4-m.toml` (the stays read four times an epoch, `[passes]`), `backbone-stays-small-m.toml` (64 wide, 16 heads, 2 blocks, feed-forward 128, the stays alone) |
| Code | `a3633a54`; new since the mixture's run: a corpus read more than once an epoch, each pass in its own order |
| Parameters | 4,796,416 over 195 channels; 4,785,152 without the stays (151 channels); 81,408 for the small shape |
| Budget | 8 epochs, micro-batch 16, two to a step, peak 1e-3, a quarter of an epoch of warm-up, cosine to 1 %, Huber at one deviation, seed 1; the kept epoch is the best by the mean relative validation |
| Platform | Kaggle, two T4s, three or four runs in sequence on each |
| Cost | 0.33 s a step throughout: the mixture 881 s an epoch (2.0 h), the stays read four times 962 s (2.1 h), without C-MAPSS 726 s (1.6 h), without SKAB 861 s (1.9 h), without the satellite corpus 222 s (0.5 h), without the stays 899 s (2.0 h), the small shape 71 s (10 min); `accept` 5–8 s a run |

Validation loss as a share of the trivial predictor's, at the kept epoch (the eighth in every run).
Every number is validation, not test.

| Experiment | Backbone | Weights | cmapss | skab | esa_ad | physionet2012 | Mean | Training loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `backbone-mixed4-m` | `2aaca0ba-…` | `durable/sha256/0d81c01e…` | 0.311 | 0.273 | 0.272 | 0.490 | 0.3366 | 0.05521 |
| `backbone-mixed4-stays-x4-m` | `78c27106-…` | `durable/sha256/8b202cf2…` | 0.311 | 0.276 | 0.255 | **0.455** | 0.3241 | 0.06056 |
| `backbone-mixed4-without-cmapss-m` | `6f2aaf27-…` | `durable/sha256/274c719f…` | — | 0.277 | 0.270 | 0.505 | 0.3508 | 0.04881 |
| `backbone-mixed4-without-skab-m` | `066ed3d8-…` | `durable/sha256/bf13ecd5…` | 0.310 | — | 0.279 | 0.498 | 0.3620 | 0.05380 |
| `backbone-mixed4-without-esa_ad-m` | `6fb484fd-…` | `durable/sha256/12691b5c…` | 0.302 | 0.265 | — | 0.489 | 0.3521 | 0.08588 |
| `backbone-mixed4-without-physionet2012-m` | `bda16abf-…` | `durable/sha256/f7e1e088…` | 0.311 | 0.274 | 0.277 | — | 0.2874 | 0.05432 |
| `backbone-stays-small-m` | `46a20b5a-…` | `durable/sha256/221c5206…` | — | — | — | 0.638 | 0.6377 | 0.22397 |
| `backbone-mixed5-nosmd-m` (2026-10-02, the earlier chain), for comparison | `1c1c57c6-…` | `durable/sha256/994d1642…` | 0.006 | 0.266 | 0.294 | 0.489 | 0.2639 | 0.03550 |

Epoch by epoch: `scripts/pretraining_curve_report.py --report-only data/report/pretraining/<backbone>`.

What the runs say:

- **Read per operating condition, C-MAPSS is no longer a solved pretext.** 0.311 of the trivial
  predictor's loss against 0.006 under the global reading, and 0.322 for the backbone over
  C-MAPSS alone read per condition (2026-09-22). The mean over the corpora is therefore not
  comparable with the earlier mixtures'; the other three corpora end where they ended.
- **Four passes over the stays buy them 0.035**: 0.455 against 0.490, between eight passes alone
  (0.489) and the thirty-two the stays-alone backbone of the task had (0.415). The satellite
  corpus also ends lower (0.255 against 0.272), under 12 % more steps and a longer decay. What
  this does on the task is for the matrix.
- **Leaving one corpus out moves the others by at most 0.01**: C-MAPSS 0.302–0.311, SKAB
  0.265–0.277, the satellite corpus 0.270–0.279, the stays 0.489–0.505 across the five mixtures.
  On the pretext the corpora learn beside one another rather than from one another; whether a
  backbone that never saw a corpus still helps its task is the matrix's question, not this one.
- **The small shape learns a third less of the pretext**: 0.638 on the stays against 0.490 for the
  large shape in the mixture; on the task the same shape from nothing was the better network
  (2026-10-03), so the pretext loss again says nothing about the task until it is read there.
- One run per backbone; the spread of a pretraining run is still unmeasured.

### 2026-10-04 — Kaggle, Tesla T4 x2, fp16, against Cloudflare R2: the mixture of four at a second seed and in the shape of 512 by 8

The first point of the curve over the scale of pretraining (`docs/preregistration.md`, "The
scale of pretraining") needs the mixture of four pretrained at a second seed, whose difference a
step of the curve must exceed, and in the larger shape, so that shape has a slope too. Ordered
here from `f00c82cf`, fulfilled on Kaggle in one session of two T4s, and accepted here.

|  |  |
| --- | --- |
| Experiments | `experiments/backbone-mixed4-m-seed2.toml` (seed 2, nothing else changed), `experiments/backbone-mixed4-512x8-m.toml` (width 512, 8 heads, 8 blocks, feed-forward 2,048; micro-batch 8, four to a step, the same 32 windows a step) |
| Parameters | 4,796,416 and 25,334,784 over 195 channels |
| Budget | as the mixture of four: 8 epochs, peak 1e-3, a quarter of an epoch of warm-up, cosine to 1 %, Huber at one deviation; the kept epoch is the best by the mean relative validation |
| Cost | 945 s an epoch (2.1 h) and 2,360 s an epoch (5.2 h), on one T4 each; the larger shape took 2.5 times today's, under the 4.7 its arithmetic per token gives |

Validation loss as a share of the trivial predictor's, at the kept epoch. Every number is
validation, not test.

| Experiment | Backbone | Weights | Epoch | cmapss | skab | esa_ad | physionet2012 | Mean | Training loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `backbone-mixed4-m` (seed 1, above) | `2aaca0ba-…` | `sha256:0d81c01e…` | 8 | 0.311 | 0.273 | 0.272 | 0.490 | 0.3366 | 0.05521 |
| `backbone-mixed4-m-seed2` | `3d0c4468-…` | `sha256:94f00d72…` | 8 | 0.313 | 0.269 | 0.271 | 0.503 | 0.3392 | 0.05576 |
| `backbone-mixed4-512x8-m` | `1def13c0-…` | `sha256:d047a2d0…` | 7 | 0.322 | 0.284 | 0.303 | 0.557 | 0.3664 | 0.06518 |

Epoch by epoch: `scripts/pretraining_curve_report.py --report-only data/report/pretraining/l1-seed2`
and `…/l1-wide`.

What the runs say:

- **A second seed moves the pretext by 0.003 on the mean**, 0.013 on the stays and at most
  0.004 elsewhere. The spread a step of the curve has to exceed is read on the task, not here.
- **The larger shape learns the pretext worse than today's**: 0.3664 against 0.3366, every corpus
  higher, and a higher training loss as well (0.063 against 0.055 at the eighth epoch), with
  the validation rising between epochs 2 and 3 and again at 8. More parameters over the same
  data should fit the training side at least as well; that this one does not points at its
  optimisation, the rate first, rather than at its capacity.
- A correction to the section above: leaving a corpus out moved the stays by up to 0.015 (0.490
  to 0.505), not 0.01; the reading does not change.

### 2026-10-04 — declared before the run: the larger shape at half the rate

**Question.** Does the larger shape's poorer pretext come from a rate of 1e-3 being too high for
its width? `experiments/backbone-mixed4-512x8-m-5e-4.toml` is the same run at 5e-4, the rate
that halving with each doubling of width gives, and nothing else changed.

**Predictions.**

1. The training loss at the eighth epoch lies below 1e-3's 0.063, and at or below today's
   shape's 0.055.
2. The mean relative validation lies below 0.3664, and within 0.01 of today's shape's 0.3366 or
   below it.
3. The validation loss falls at every epoch after the second.

**Reading, declared beforehand.** The pretext decides nothing about which rate the larger shape
keeps: that is read on the task, by the closed-form probe at 50 stays under both runs, paired
over the same stays and seeds, the run whose probe is higher by the paired interval kept for
the curve, and 1e-3 kept where neither is. If the half rate wins on the task, the larger shape
at about 10⁹ values runs at it.

### Open

- ~~The checkpoint reference a dropped session should be resumed from is known to nobody when
  the run is not tracked.~~ Closed on 2026-09-19: the runtime logs every checkpoint it writes,
  and the platform keeps the log (ADR-0029).
- ~~Resuming from a remote checkpoint has been tested through the port and on this machine's
  accelerator, not yet on the platform.~~ Closed on 2026-09-19: the mixed run's order was resumed
  on the platform from a checkpoint inside its sixth epoch (section above).
- The cost of `read` on a machine that fetches the block from the bucket was not measured on the
  platform: the run's wall clock includes it and the run itself does not time it.
