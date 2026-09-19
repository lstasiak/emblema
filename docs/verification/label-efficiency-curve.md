# Label-efficiency curve: the first reading, preliminary and on the validation side

Purpose: draw the curve the programme is about — four transfer modes over four budgets of
labels under five seeds, on the turbofan task, from the registered backbone — and read it by
the rules registered before any of its numbers existed (`docs/preregistration.md`). Two things
happen here in order: the schedule of every arm is fixed at one budget on the validation side,
recorded as an amendment to the preregistration before the grid; then the grid runs on the GPU
platform and is read through the Evaluation context's statistics (ADR-0032).

**Every number here is validation, not test, and the result is preliminary.** The frozen test
side of the task is never opened. The evaluation harness repeats this comparison once it
exists; the reading it makes is the one published, and the difference from this one is recorded
when it is known.

Method on a machine with the remote bucket's credentials and the downloaded corpus under
`data/raw` (the assertions the curve stands on are tests: `tests/evaluation`,
`tests/scripts/test_transfer_modes_report.py`, `tests/scripts/test_label_curve_report.py`):

```sh
uv sync --all-extras
uv run pytest tests/evaluation tests/scripts
# one cell, or a sweep of them, on this machine — the budget the machine's tier serves
uv run --env-file .env.r2 scripts/transfer_modes_report.py \
    --weights <key> <checksum> --manifest <key> <checksum> \
    --budget 200 --seed 1 --mode from_scratch --device mps --out data/report/transfer/<name>
# the grid, fetched back from the bucket, read and drawn
uv run --env-file .env.r2 scripts/transfer_modes_report.py --fetch <key> <checksum> --out <shard>
uv run scripts/label_curve_report.py <shard> [<shard> ...] --out data/report/curve/<name>
uv run scripts/label_curve_figures.py data/report/curve/<name> \
    --figure docs/verification/figures/label-efficiency-curve.png
```

The transfer report stores a grid cell by cell — a row per run, per epoch and per validation
window — skips the cells a directory already holds, and publishes a directory to the bucket as
one archive; the curve report reads the shards as one curve and writes the three files a note
and a figure are rendered from. Nothing here is computed from what is still in memory.

On the platform, the grid is two processes in one session, one per accelerator, sharded by
seed. The cell runs in one notebook cell that waits for both, and both log to the notebook's
own output, because a session run non-interactively ends when its last cell returns and its
working directory does not outlive it:

```sh
git clone https://github.com/lstasiak/emblema && cd emblema && git checkout <commit>
pip install -e ".[ml]"                       # the scripts are not in the wheel
python scripts/fetch_corpora.py cmapss       # the ground truth is read from the raw files
# EMBLEMA_ARTIFACT_STORE__* from the notebook's secrets, as for the pretraining runs
python scripts/transfer_modes_report.py --weights ... --manifest ... \
    --seed 1 --seed 3 --seed 5 --device cuda:0 --out shard-0 --publish 2>&1 \
    | sed 's/^/[shard-0] /' &
python scripts/transfer_modes_report.py --weights ... --manifest ... \
    --seed 2 --seed 4 --device cuda:1 --out shard-1 --publish 2>&1 \
    | sed 's/^/[shard-1] /' &
wait
```

Every cell, once stored, is published as one archive and its reference printed on the cell's
own line, so a session that drops has lost at most the cell it was in; a cell that raises
stops its shard after publishing what it holds. A fresh session fetches each shard's last
reference into the same directory name and runs the same command: the cells it holds are
skipped, and a directory reopened under another plan, backbone or commit is refused rather
than resumed. The first cells of a shard are the cheapest budget, so a session reports its cost
per cell early.

```sh
python scripts/transfer_modes_report.py --fetch <key> <checksum> --out shard-0
```

## 2026-09-19 — Darwin arm64 (MacBook Pro M1 Pro, MPS, fp32): the schedule of every arm

Tree of the ticket branch on top of `6f5e9c3` (the code of the sweep is what the branch's
commits carry), Python 3.14.7, torch 2.14.0. Backbone `backbone-cmapss-m` (`de3815c9-…`,
weights `sha256:6830e117…`), corpus manifest `sha256:a00c3865…`. Task `turbofan-fd001`,
ceiling 125, 4 strata; 200 labelled windows drawn under seed 1, which reach 73 of the 82 tuning
engines; validation over the 18 held-out engines, 535 windows. Every arm: 30 epochs, batches of
16, no weight decay, run seed 1; the low-rank arm at rank 8, α = 16, beside `qkv`,
`attention.projection` and `feedforward`. Two shapes of the rate: constant, and a linear
warm-up over the first tenth of the run's steps followed by a cosine decay to one per cent of
the peak. The rule, fixed before the sweep: each arm takes the peak with the lowest validation
RMSE of three, and the shape chosen for the control arm applies to every arm. The sweep first
covered the control and the probe; when the control came out level with full fine-tuning, the
two remaining arms were swept over three peaks the same way (`docs/preregistration.md`,
2026-09-19). Stored under `data/report/transfer/sweep-20260919/<arm>-lr<peak>-<shape>`.

| arm | peak | shape | trainable | RMSE | seconds |
| --- | --- | --- | --- | --- | --- |
| from_scratch | 1e-3 | constant | 4,752,129 | 44.64 | 197 |
| from_scratch | 3e-4 | constant | 4,752,129 | 39.14 | 211 |
| from_scratch | 1e-4 | constant | 4,752,129 | 34.85 | 194 |
| from_scratch | 1e-3 | warm-up + cosine | 4,752,129 | 25.75 | 244 |
| **from_scratch** | **3e-4** | **warm-up + cosine** | 4,752,129 | **22.38** | 193 |
| from_scratch | 1e-4 | warm-up + cosine | 4,752,129 | 29.69 | 195 |
| frozen_probe | 1e-2 | constant | 257 | 38.92 | 10 |
| frozen_probe | 3e-3 | constant | 257 | 39.31 | 10 |
| frozen_probe | 1e-3 | constant | 257 | 40.08 | 10 |
| **frozen_probe** | **1e-2** | **warm-up + cosine** | 257 | **38.73** | 10 |
| frozen_probe | 3e-3 | warm-up + cosine | 257 | 40.14 | 10 |
| frozen_probe | 1e-3 | warm-up + cosine | 257 | 40.81 | 10 |
| **lora** | **3e-3** | **warm-up + cosine** | 196,865 | **20.70** | 237 |
| lora | 1e-3 | warm-up + cosine | 196,865 | 21.84 | 229 |
| lora | 3e-4 | warm-up + cosine | 196,865 | 22.90 | 231 |
| **full_fine_tuning** | **3e-4** | **warm-up + cosine** | 4,752,129 | **21.10** | 199 |
| full_fine_tuning | 1e-4 | warm-up + cosine | 4,752,129 | 22.08 | 321 |
| full_fine_tuning | 3e-5 | warm-up + cosine | 4,752,129 | 22.16 | 195 |

What the sweep says:

- **The constant rate reproduces the first run**: 44.64 at 1e-3, to the sixth decimal the
  number of `transfer-modes.md` (44.637796 there, 44.637797 here), so the schedule added to the
  loop is the identity where it says it is. The constant rate at 1e-4 repeats the diagnostic of
  that note as well (34.85).
- **The shape moves the control arm more than any peak does.** Under a constant rate the fresh
  encoder does best at the smallest peak and is still falling after thirty epochs; under the
  warm-up and the decay it does best at 3e-4 and its training loss settles (0.148 → 0.033 in
  units of the ceiling squared). The first run's control at the trivial predictor was the
  schedule's result.
- **The three arms that step the encoder, or an update beside it, end within two RMSE of one
  another** at this budget and seed: 22.38 from scratch, 21.10 with full fine-tuning, 20.70
  with the low-rank updates. The probe stays near the mean predictor (41.11 on these windows)
  whatever its rate, and the decay changes it by a fifth of a point.
- **Cost on MPS, fp32**: 190–245 s for an arm that steps the encoder at this budget, 10 s for
  the probe; the 321 s of one run is the machine doing something else at the time.

What it does not say: anything about the endpoint. One seed at one budget is a reading of the
schedule, not of the claim; the grid measures the claim over five seeds with its interval.

The path the grid takes was run once end to end on this machine the same evening, at the
smallest budget under one seed (`data/report/transfer/smoke-grid`): the four cells cost 54 / 7 /
64 / 53 s on MPS at 50 labelled windows from 33 engines (from scratch 41.11 — the mean
predictor's number —, probe 40.82, low-rank 40.78, full fine-tuning 34.81), the directory
published to the bucket as one archive (`durable/sha256/b7f62891…`), a second invocation
skipped every cell as already stored, the archive fetched back byte for byte, and the curve
report and the figure rendered from it, with the grid reported as incomplete and the endpoint
as not measured. Those numbers are a check of the path, not a reading of the curve; the one
secondary cell they reject, the probe at fifty, is rejected among the three cells that ran and
not among the registered eleven (`docs/preregistration.md`, the second section of 2026-09-19).
