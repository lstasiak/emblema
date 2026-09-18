# Transfer modes: the pretrained backbone meets the turbofan task

Purpose: show that every transfer mode of ADR-0030 trains on the registered backbone and answers
the task's validation engines, and record what the first run of each cost and scored. The
properties that hold everywhere are tests, not numbers here: under the frozen probe and the
low-rank mode the backbone keeps every weight bit for bit, full fine-tuning moves them, the
control arm never asks for the pretrained ones, and a run repeats bit for bit on the host under
one seed (`tests/evaluation/adapters/torch`, `tests/evaluation/test_transfer_modes_over_a_published_corpus.py`).
What this note adds is a run on the real backbone and the real task, on an accelerator.

**Every number here is validation, not test.** The frozen test side of the task is never opened
by these runs. The numbers are one run under one seed at one budget: a first reading, not the
curve, which is drawn from the grid the preregistration states.

Method on a machine with the remote bucket's credentials and the downloaded corpus under
`data/raw`:

```sh
uv sync --all-extras
uv run pytest tests/evaluation                                              # the assertions
uv run --env-file .env.r2 scripts/transfer_modes_report.py \
    --weights <key> <checksum> --manifest <key> <checksum> --budget 200 --device mps
uv run scripts/transfer_modes_report.py --report-only data/report/transfer/<run>   # rerender
```

The report stores a row per run, per epoch and per validation window as CSV and renders its
table from the files.

## 2026-09-18 — Darwin arm64 (MacBook Pro M1 Pro, MPS, fp32)

Commit `4d01929`, Python 3.14.7, torch 2.14.0. Backbone `backbone-cmapss-m`
(`de3815c9-…`, 256 wide, 6 blocks, 4,751,872 encoder parameters, pretrained in half precision on
a T4; weights `sha256:6830e117…`), corpus manifest `sha256:a00c3865…` (window 50, stride 5).
Task `turbofan-fd001`, ceiling 125, 4 strata; 200 labelled windows drawn under sample seed 1
from the 82 tuning engines; validation over the 18 engines the corpus holds out, 535 windows.
Every mode ran 30 epochs at batch 16, weight decay 0, under run seed 1, with the learning rate
the report defaults to for it; low-rank updates at rank 8, α = 16, beside `qkv`,
`attention.projection` and `feedforward` of every block. Nothing was tuned.

| mode | budget | run seed | epochs | lr | trainable | RMSE | seconds | device |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| from_scratch | 200 | 1 | 30 | 0.001 | 4752129 | 44.64 | 196 | mps |
| frozen_probe | 200 | 1 | 30 | 0.01 | 257 | 38.92 | 9 | mps |
| lora | 200 | 1 | 30 | 0.001 | 196865 | 24.07 | 232 | mps |
| full_fine_tuning | 200 | 1 | 30 | 0.0001 | 4752129 | 22.30 | 225 | mps |

Stored under `data/report/transfer/20260918-224536`.

What the files say beyond the table:

- **The training loss falls in every mode.** In units of the ceiling squared, over the 200
  windows: from scratch 4.40 → 0.13 (lowest 0.11 at epoch 13), the probe 1.03 → 0.18 (lowest
  0.11 at epoch 26, oscillating under its learning rate of 0.01), low-rank 0.12 → 0.027, full
  fine-tuning 0.12 → 0.027 (lowest 0.024 at epoch 29). The two pretrained-and-updated arms
  start where the probe ends, because the restored encoder already places the windows.
- **Baselines read off the same predictions.** Predicting the validation mean (72.3 cycles,
  SD 41.1) scores an RMSE of 41.11; predicting the ceiling, 66.85. The control arm trained from
  scratch, at 44.64, is **worse than the mean predictor**, and its training loss ended at about
  the mean predictor's level as well (0.13 against 0.108 for the mean over a side of this
  spread): at 200 windows, 30 epochs and a learning rate of 0.001 without warm-up, the fresh
  encoder learnt the mean and little else. The probe, at 38.92, is barely under it. The two
  arms that update the pretrained encoder are far under both.
- **Secondary readings the preregistration names**, from the same 535 predictions: RMSE over the
  430 windows whose target lies below the ceiling — 35.70 / 38.21 / 19.77 / 19.31; RMSE on the
  last window of each engine — 53.00 / 71.34 / 8.71 / 5.59, in the table's order. The last-window
  numbers are the ones comparable with published results on this task, over 18 engines only.
- **Cost.** The probe is seconds; the three arms that step the encoder or an update beside it
  cost 196–232 s each on MPS at this budget, the low-rank arm no cheaper than full fine-tuning,
  as ADR-0030 says it would not be.

Whether the control arm is a schedule or a defect was checked the same evening with two more
runs of it alone, same budget, seed and labels (`data/report/transfer/diag-scratch-*`): at a
learning rate of 0.0001 for 30 epochs the training loss falls from 0.38 to 0.093 and is still
falling, validation RMSE 34.85; at 0.001 for 90 epochs it sits on the plateau of the mean
through the thirtieth epoch and leaves it around the sixtieth, ending at 0.046, validation RMSE
34.87. The fresh encoder learns past the mean once it is given a rate it can start at or the
epochs to escape the plateau, so the number in the table is the default schedule's, not the
code's; and both variants stay far above the arms that update the pretrained encoder.

What this run does not settle: the schedule of the control arm. The primary endpoint of the
preregistration compares full fine-tuning with training from scratch, so a control arm at the
trivial predictor would make that comparison pass for the wrong reason. The control arm's
learning rate, warm-up and epochs are to be fixed before the grid, on this validation side, and
stated as the grid's — the preregistration allows one set of hyperparameters per mode declared
beforehand, not a set per cell.
