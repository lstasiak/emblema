# Label-efficiency curve on the turbofan task, preliminary and on the validation side

Purpose: draw the curve the programme is about — four transfer modes over four budgets of
labels under five seeds, on the turbofan task — and read it by the rules registered before any
of its numbers existed (`docs/preregistration.md`), through the Evaluation context's statistics
(ADR-0032). The sections are dated and appended. The first grid (2026-09-20) was not confirmed;
the configuration then changed, each change registered before its run, until the last section:
on the four C-MAPSS subsets read per operating condition (ADR-0034) the endpoint is confirmed and
the curve is drawn.

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
working directory does not outlive it. The prefix is added unbuffered (`sed -u`): through a pipe
`sed` otherwise holds its output until a block fills, and the log falls silent for many cells
while the bucket already holds them:

```sh
git clone https://github.com/lstasiak/emblema && cd emblema && git checkout <commit>
pip install -e ".[ml]"                       # the scripts are not in the wheel
python scripts/fetch_corpora.py cmapss       # the ground truth is read from the raw files
# EMBLEMA_ARTIFACT_STORE__* from the notebook's secrets, as for the pretraining runs
python scripts/transfer_modes_report.py --weights ... --manifest ... \
    --seed 1 --seed 3 --seed 5 --device cuda:0 --out shard-0 --publish 2>&1 \
    | sed -u 's/^/[shard-0] /' &
python scripts/transfer_modes_report.py --weights ... --manifest ... \
    --seed 2 --seed 4 --device cuda:1 --out shard-1 --publish 2>&1 \
    | sed -u 's/^/[shard-1] /' &
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

## 2026-09-20 — Linux x86_64 (Kaggle, two Tesla T4, fp32): the grid

Commit `0277f98` (the ticket branch rebased onto `86b0521`), Python 3.12, torch 2.10.0+cu128 as
preinstalled, package installed editable from the clone; the corpus fetched from its source of
record for the ground truth; the backbone and the manifest of the sweep. One session, two
processes sharded by seed, each cell published as it landed (80 archives, the last one of a
shard holding the whole shard): `cuda:0` seeds 1, 3, 5 — 48 cells, 7.53 h inside the cells,
finished 02:52 UTC; `cuda:1` seeds 2, 4 — 32 cells, 4.71 h, finished 00:02 UTC. Fetched here as
`data/report/transfer/grid-20260920/shard-{0,1}` (final references
`durable/sha256/e69ebef2…` and `durable/sha256/ad92964c…`), read into
`data/report/curve/grid-20260920` and drawn from there.

**Cost on a T4, fp32**: 41–45 / 4 / 42 / 41 s for the four arms at 50 labelled windows,
~150 s at 200, 740–770 s at 1,000 and 1,950–2,030 s at 2,651 for every arm that steps the
encoder or an update beside it — 0.37 to 0.41 s per optimiser step, the same whichever arm —
and 4 to 27 s for the probe. The first cells put the longer shard at seven hours, above the six
per accelerator at which the decision on half precision was to be reopened; the grid was left to
run because it fitted one session and the registration names single precision.

**The platform reproduces this machine.** Under seed 1 the four arms at 50 windows score
41.11 / 40.82 / 40.78 / 34.81, the smoke run's numbers to the second decimal, and at 200 they
score 22.38 / 38.73 / 20.69 / 21.02 against the sweep's 22.38 / 38.73 / 20.70 / 21.10 on MPS.

**Endpoint RMSE per cell, mean ± SD over seeds** (the engines column is how many engines the
budget's labels came from, over the seeds):

| budget | engines | from_scratch | frozen_probe | lora | full_fine_tuning |
| --- | --- | --- | --- | --- | --- |
| 50 | 33-38 | 40.97 ± 0.30 | 41.08 ± 0.25 | 39.58 ± 1.27 | 39.61 ± 2.91 |
| 200 | 70-78 | 32.78 ± 7.25 | 39.47 ± 0.47 | 22.26 ± 1.37 | 25.76 ± 5.56 |
| 1000 | 82 | 18.52 ± 0.96 | 33.03 ± 0.56 | 18.19 ± 1.67 | 19.75 ± 1.10 |
| all | 82 | 14.43 ± 0.48 | 26.52 ± 0.30 | 21.05 ± 11.54 | 18.14 ± 1.45 |

Trivial predictors over the same windows: mean predictor 41.11, ceiling predictor 66.85.

**Against the control arm**, pooled over the seeds, the interval a bootstrap over the 18
engines (10,000 resamples); the endpoint stands alone, the other eleven are the registered
family under the Holm correction at 5 %:

| mode | budget | control | candidate | reduction | relative | 95 % interval | p | floor | rejected | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_probe | 50 | 40.97 | 41.08 | -0.11 | -0.3% | [-0.32, +0.08] | 0.2600 | 1.23 | no | indistinguishable |
| lora | 50 | 40.97 | 39.60 | +1.37 | +3.3% | [+0.86, +1.84] | 0.0002 | 1.23 | yes | distinguishable |
| full_fine_tuning | 50 | 40.97 | 39.69 | +1.28 | +3.1% | [+0.98, +1.56] | 0.0002 | 1.23 | yes | distinguishable |
| frozen_probe | 200 | 33.41 | 39.47 | -6.06 | -18.1% | [-7.34, -4.77] | 0.0002 | 7.25 | yes | worse |
| lora | 200 | 33.41 | 22.29 | +11.12 | +33.3% | [+9.07, +13.50] | 0.0002 | 7.25 | yes | distinguishable |
| full_fine_tuning | 200 | 33.41 | 26.24 | +7.17 | +21.5% | [+5.92, +8.57] | 0.0002 | 7.25 | primary | distinguishable, practically nil |
| frozen_probe | 1000 | 18.54 | 33.03 | -14.49 | -78.2% | [-17.56, -11.48] | 0.0002 | 0.96 | yes | worse |
| lora | 1000 | 18.54 | 18.25 | +0.29 | +1.6% | [-1.71, +2.28] | 0.7947 | 0.96 | no | indistinguishable |
| full_fine_tuning | 1000 | 18.54 | 19.77 | -1.23 | -6.6% | [-2.93, +0.55] | 0.1622 | 0.96 | no | indistinguishable |
| frozen_probe | all | 14.44 | 26.53 | -12.09 | -83.7% | [-13.97, -10.14] | 0.0002 | 0.48 | yes | worse |
| lora | all | 14.44 | 23.44 | -9.00 | -62.4% | [-10.48, -7.57] | 0.0002 | 0.48 | yes | worse |
| full_fine_tuning | all | 14.44 | 18.18 | -3.75 | -25.9% | [-5.57, -1.98] | 0.0002 | 0.48 | yes | worse |

**Verdict on the registered endpoint: not confirmed — distinguishable, practically nil.** Full
fine-tuning at 200 labelled windows takes 21.5 % off the control's error (7.17 RMSE, interval
[5.92, 8.57], five seeds pooled per engine), which clears the registered share and keeps the
whole interval above zero; the practical floor at that budget is 7.25, and 7.17 does not clear
it. The floor is the control's own spread over its five seeds, and that spread is the finding:

- **The control arm at 200 is bimodal.** Under seeds 1 and 5 it leaves the plateau of the mean
  predictor (22.4 and 28.1; training loss 0.033 and 0.046 in units of the ceiling squared) and
  under seeds 2, 3 and 4 it stays on it (36.2 to 39.0; loss 0.083 to 0.101, the variance of the
  target being 0.108). The sweep chose its peak under seed 1, one of the two seeds that escape.
- **The pretrained arm escapes more often, not further.** Four of its five repeats leave the
  plateau (21.0 to 25.8) and one stays (seed 2, 35.2, loss 0.087). Under seed 1, where both
  arms escaped, the difference is 22.38 against 21.02 — six per cent, under the registered ten.
  The endpoint's reduction is therefore mostly a difference in how often thirty epochs get an
  arm off the plateau, which is what the measured part of the floor exists to catch.
- Under the rules as they stood before the second amendment of 2026-09-19 the verdict would have
  read *confirmed*; the amendment binding the endpoint to the floor was registered before any
  cell ran, and this cell is the case it names.

**The budget of optimisation, a limit of this design.** Every arm trains for thirty epochs, so
the number of optimiser steps a cell gets grows with its budget of labels, and no cell of the
grid had converged when it was scored (the training losses per epoch are in each shard's
`epochs.csv`, in units of the ceiling squared; the variance of the target is 0.108):

| budget | optimiser steps | the training loss at the thirtieth epoch |
| --- | --- | --- |
| 50 | 120 | every arm on the plateau of the mean predictor: a loss of 0.09–0.11 and answers spread under 2.5 cycles in 17 of the 20 runs |
| 200 | 390 | the repeats that leave the plateau are still falling steeply (the control under seed 1: 0.072 → 0.051 → 0.033 over the last ten epochs); the others sit at 0.08–0.10 |
| 1,000 | 1,890 | every arm still falling (0.014–0.021) |
| all | 4,980 | every arm still falling (the control 0.011 → 0.009 over the last five epochs) |

The registration read the cell at fifty as a budget of optimisation before the grid ran; the
grid says the same of the cell at 200, where the decay to one per cent of the peak closes on
the repeats that start to leave the plateau late (the control under seed 4 drifts off 0.10 only
from the eighteenth epoch, with the rate already at half its peak and at a tenth by the
twenty-fifth, and ends at 0.083). What the low
budgets measure is therefore how fast an arm leaves the plateau under a fixed and small number
of steps, not how many labels the task needs; a budget stated in optimiser steps rather than
epochs, fixed before the next run, separates the two.

**The shape of the curve.**

- At 50 windows — 120 optimiser steps — every arm sits within 1.5 RMSE of the mean predictor;
  the two cells the family rejects there clear a floor of 1.23 by 0.05 and 0.14 RMSE. The
  amendment recorded this cell as a budget of optimisation before the grid ran.
- **The one secondary cell that clears its floor by a margin is the low-rank arm at 200**: all
  five of its repeats leave the plateau (20.7 to 23.9, 22.26 ± 1.37 over seeds) where the
  control's leave under two, and its reduction of 11.12 RMSE, interval [9.07, 13.50], clears
  the floor of 7.25 under the Holm correction. This is the ablation the transfer modes were
  framed as (ADR-0030): fewer degrees of freedom beside a pretrained encoder where labels are
  few. Read with the limit above: under seed 1, where the control leaves the plateau too, the
  two arms end at 22.38 and 20.70, so the cell's margin is mostly how reliably thirty epochs
  get an arm moving, and a longer budget of steps may close it.
- At 1,000 the three arms that step the encoder or an update beside it are indistinguishable
  (18.2 to 19.8).
- At every label the arm trained from scratch (14.4) is better than full fine-tuning (18.1) and
  than the low-rank updates (21.1), the whole interval of each below zero: the pretrained
  weights cost error where labels are plentiful, under the peaks chosen at 200. The two arms
  end at nearly the same training loss (0.010–0.015 against 0.008–0.012), and the pretrained
  one starts lower (0.047 against 0.112 after the first epoch), so the pretrained weights speed
  up the first epochs and then settle in a region that generalises worse to the held-out
  engines, in all five repeats.
- The probe is far below every other arm at every budget beyond the first (39.5 / 33.0 / 26.5
  against 25.8 / 19.8 / 18.1 for full fine-tuning): the mean of the encoder's states is not a
  linear summary of remaining life. This is the condition ADR-0030 named for a two-stage arm.
- **One repeat of the low-rank arm at the whole budget diverged late**: seed 1 scores 41.47
  where the other four score 13.4 to 18.1; its training loss fell to 0.033 by the tenth epoch
  and rose back to 0.110 by the last, under a peak of 3e-3 that was chosen over 390 steps and
  here ran 4,980. It is one repeat, reported and not dropped; the SD of 11.5 is this run.

**Readings beside the endpoint**, mean ± SD over seeds — read, not thresholded:

RMSE below the ceiling:

| budget | engines | from_scratch | frozen_probe | lora | full_fine_tuning |
| --- | --- | --- | --- | --- | --- |
| 50 | 33-38 | 38.88 ± 0.96 | 39.43 ± 1.28 | 37.79 ± 1.34 | 37.97 ± 3.07 |
| 200 | 70-78 | 30.96 ± 6.86 | 37.48 ± 0.43 | 21.09 ± 1.10 | 24.51 ± 5.30 |
| 1000 | 82 | 18.35 ± 1.08 | 30.59 ± 1.22 | 17.46 ± 1.55 | 18.80 ± 1.13 |
| all | 82 | 14.74 ± 0.50 | 24.28 ± 0.69 | 20.55 ± 10.93 | 17.75 ± 1.39 |

RMSE on the last window of each engine (on the validation side an engine's last window ends within four cycles of its failure, so this is the error at the end of life; the benchmark's protocol, and the only reading comparable with published numbers, is the same over the test side's trajectories cut short of failure):

| budget | engines | from_scratch | frozen_probe | lora | full_fine_tuning |
| --- | --- | --- | --- | --- | --- |
| 50 | 33-38 | 73.62 ± 2.24 | 74.72 ± 2.95 | 71.01 ± 2.87 | 70.83 ± 8.11 |
| 200 | 70-78 | 50.61 ± 21.94 | 70.23 ± 1.15 | 10.49 ± 3.25 | 22.60 ± 22.21 |
| 1000 | 82 | 4.12 ± 0.75 | 51.82 ± 3.18 | 6.23 ± 1.72 | 6.62 ± 1.27 |
| all | 82 | 2.52 ± 0.47 | 32.34 ± 1.60 | 18.64 ± 32.24 | 4.93 ± 1.46 |

Alpha-lambda accuracy (share of answers within 20 % of the label):

| budget | engines | from_scratch | frozen_probe | lora | full_fine_tuning |
| --- | --- | --- | --- | --- | --- |
| 50 | 33-38 | 0.20 ± 0.00 | 0.21 ± 0.01 | 0.22 ± 0.01 | 0.22 ± 0.02 |
| 200 | 70-78 | 0.30 ± 0.10 | 0.21 ± 0.00 | 0.45 ± 0.02 | 0.39 ± 0.07 |
| 1000 | 82 | 0.58 ± 0.03 | 0.25 ± 0.01 | 0.55 ± 0.05 | 0.50 ± 0.02 |
| all | 82 | 0.67 ± 0.03 | 0.37 ± 0.01 | 0.54 ± 0.18 | 0.56 ± 0.04 |

Asymmetric score (mean per window, lower is better; its exponential tail is carried by a few engines):

| budget | engines | from_scratch | frozen_probe | lora | full_fine_tuning |
| --- | --- | --- | --- | --- | --- |
| 50 | 33-38 | 152.24 ± 29.72 | 170.96 ± 42.57 | 143.47 ± 34.16 | 144.76 ± 81.00 |
| 200 | 70-78 | 78.27 ± 60.02 | 119.76 ± 10.99 | 14.55 ± 4.50 | 39.54 ± 50.85 |
| 1000 | 82 | 8.67 ± 1.41 | 52.32 ± 14.66 | 8.41 ± 2.01 | 10.48 ± 1.56 |
| all | 82 | 4.15 ± 0.42 | 18.67 ± 2.24 | 41.77 ± 81.01 | 8.21 ± 1.78 |

The last-window reading is the error at the end of life on this side and is not comparable with
published test-side numbers (`docs/preregistration.md`, 2026-09-19). Two of them are worth a
sentence: at the whole budget the arm trained from scratch answers within 2.5 cycles at the end
of an engine's life and within a fifth of the label on two thirds of the windows, and at 200 the
low-rank arm carries the lowest asymmetric score of the grid (14.6 against 78.3 for the control).

**What the registered rules say to read next.** A negative result is read in a fixed order
before it is a statement about pretraining: the synthetic control (the transfer leg, not yet
run), whether the pretraining run was shown to have stopped learning (it was not: the backbone
spent 2,532 steps of batch 32 — four epochs, thirteen minutes on a T4 — and its validation loss
was still falling by seven per cent an epoch at the last one, `manual-handoff.md`; doubling the
budget of the same objective on the control corpora still cut the loss by a quarter or more,
`masked-reconstruction.md`; and the fine-tuning at the whole budget, 4,980 steps, spends more
compute than the pretraining did), and whether the control at this budget is already at the
floor of what the data support (it is not: two of its seeds reach 22 to 28 and at 1,000 it
reaches 18.5). The pretraining corpus is the tuning side of the task — the same 82 engines,
unlabelled — so at the whole budget the pretrained arm holds no data the control does not, and
the claim lives in the small budgets alone. What this grid establishes on its own is
that at 200 labelled windows the comparison is decided by which repeats leave the plateau under
thirty epochs, and that beyond 1,000 the pretrained weights are a cost. Whether the encoder
carries anything the task can use is what the synthetic control answers; the decision on the
next stage waits for it.

## 2026-09-20 — Linux x86_64 (Kaggle, two Tesla T4, fp32): the sweep under the floor

Code `605a5e0`, installed in the notebook from the commit: the floor of optimiser steps and the
head's start registered the same day (`docs/preregistration.md`, 2026-09-20), the platform's
torch 2.10.0+cu128. The same backbone, corpus, task and validation windows as the sweep of
2026-09-19: `backbone-cmapss-m` (`de3815c9-…`, weights `sha256:6830e117…`), manifest
`sha256:a00c3865…`, `turbofan-fd001`, the 18 held-out engines and their 535 windows. What
differs: 200 labelled windows drawn under three seeds, 1, 2 and 3 (73, 70 and 76 engines), and
every run lifted from thirty epochs to the floor of 2,000 steps — 154 epochs, 2,002 steps in
batches of 16, the warm-up over the first tenth and the cosine decay to one per cent spanning
the lengthened run, the head's bias starting at the mean label of the draw. The three peaks per
arm are the ones swept on 2026-09-19. The rule, fixed before the sweep: an arm takes the peak
with the lowest mean validation RMSE over the three seeds. The twelve configurations were dealt
by hand over the two accelerators, each published to the bucket as its own archive and fetched
back here into `data/report/transfer/sweep-20260920-kaggle/<arm>-lr<peak>` (archives
`durable/sha256/6354b176…`, `c02da1e2…`, `0b5d1461…` for the control at 1e-3, 3e-4, 1e-4;
`cedf1500…`, `20ae11f6…`, `93fa1aa1…` for the probe at 1e-2, 3e-3, 1e-3; `c0d5b199…`,
`3b2d8652…`, `be14e4b6…` for the low-rank arm at 3e-3, 1e-3, 3e-4; `0d81ba53…`, `95c90ea6…`,
`1c5360c5…` for full fine-tuning at 3e-4, 1e-4, 3e-5). Validation RMSE in cycles, the chosen
peaks in bold:

| arm | peak | trainable | seed 1 | seed 2 | seed 3 | mean | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **from_scratch** | **1e-3** | 4,752,129 | 20.70 | 20.92 | 20.97 | **20.86** | 755–762 |
| from_scratch | 3e-4 | 4,752,129 | 23.39 | 21.06 | 21.28 | 21.91 | 760–763 |
| from_scratch | 1e-4 | 4,752,129 | 22.20 | 21.37 | 21.69 | 21.75 | 760–763 |
| **frozen_probe** | **1e-2** | 257 | 32.70 | 32.86 | 31.70 | **32.42** | 6–12 |
| frozen_probe | 3e-3 | 257 | 37.15 | 37.02 | 36.54 | 36.90 | 6–8 |
| frozen_probe | 1e-3 | 257 | 39.55 | 39.45 | 39.35 | 39.45 | 6–8 |
| lora | 3e-3 | 196,865 | 26.35 | 23.49 | 25.50 | 25.11 | 785–798 |
| lora | 1e-3 | 196,865 | 25.87 | 26.04 | 25.27 | 25.73 | 801–804 |
| **lora** | **3e-4** | 196,865 | 21.21 | 23.56 | 22.77 | **22.51** | 751–756 |
| full_fine_tuning | 3e-4 | 4,752,129 | 23.11 | 25.71 | 24.10 | 24.31 | 718–720 |
| full_fine_tuning | 1e-4 | 4,752,129 | 22.01 | 26.10 | 25.81 | 24.64 | 719–721 |
| **full_fine_tuning** | **3e-5** | 4,752,129 | 21.69 | 23.73 | 24.00 | **23.14** | 719–721 |

What the sweep says:

- **The control's peak moves up, to 1e-3, and its seeds agree.** Under 390 steps the same peak
  scored 25.75 and 3e-4 won; under 2,002 the order reverses, and the three seeds at 1e-3 lie
  within 0.27 of one another (SD 0.14). Every seed of every control peak leaves the plateau of
  the mean predictor (41.11 on these windows): the bimodality the grid was read under was a
  fact about 390 steps.
- **The peaks of the two arms that update the pretrained encoder move down a notch**: the
  low-rank updates from 3e-3 to 3e-4, full fine-tuning from 3e-4 to 3e-5, by 2.6 and 1.2 RMSE
  over the peak chosen at 390 steps. A run five times longer wants a smaller peak.
- **On this backbone, under the floor, the control is the best arm at 200**: 20.86 against
  22.51 with the low-rank updates, 23.14 with full fine-tuning and 32.42 for the probe, and
  the two pretrained arms spread nine times wider over the seeds (SD 1.2 and 1.3 against 0.14).
  This is a reading at one budget on the validation side, on a backbone that trained for
  2,532 steps with its validation loss still falling and is being trained again; the peaks of
  the three arms that start from it are swept again under the retrained one before any cell
  of the next grid. What it already says is that the first grid's reading at 200, a pretrained
  arm ahead of a control on the plateau, does not survive giving the control the steps to
  leave it.
- **The probe gains six points from the steps and is still limited by its rate**: 32.42 at
  1e-2 against 38.73 under thirty epochs, and monotone in the peak over the three swept, as
  it was on 2026-09-19.
- **Cost on a T4, fp32**: 0.36–0.40 s per optimiser step, 718–804 s per run for an arm that
  steps the encoder or an update beside it, 6–12 s for the probe; the twelve configurations
  took 3.3 h over the two accelerators.

What it does not say: anything about the endpoint, which the grid measures over five seeds
with its interval, or anything about the retrained backbone.

The same script ran the same twelve configurations on this machine as the device check (M1 Pro,
MPS, fp32, `data/report/transfer/sweep-20260920/<arm>-lr<peak>`, 20–21 September, the later
directories under revisions up to `a726b7b`, none of which touches the adaptation). Validation
RMSE per seed on MPS, and its difference from the same run on the T4:

| arm | peak | MPS seed 1 / 2 / 3 | MPS mean | T4 mean | MPS − T4 per seed | seconds |
| --- | --- | --- | --- | --- | --- | --- |
| from_scratch | 1e-3 | 20.76 / 22.35 / 21.53 | 21.55 | 20.86 | +0.06 / +1.44 / +0.56 | 1,067–1,195 |
| from_scratch | 3e-4 | 21.94 / 22.82 / 21.58 | 22.12 | 21.91 | −1.45 / +1.76 / +0.30 | 976–1,117 |
| from_scratch | 1e-4 | 22.20 / 21.37 / 21.69 | 21.75 | 21.75 | 0 / 0 / 0 | 973–1,008 |
| frozen_probe | 1e-2, 3e-3, 1e-3 | as on the T4 | 32.42, 36.90, 39.45 | the same | 0 | 10–12 |
| lora | 3e-3 | 26.12 / 24.23 / 26.18 | 25.51 | 25.11 | −0.23 / +0.74 / +0.68 | 1,148–1,206 |
| lora | 1e-3 | 24.96 / 26.04 / 25.27 | 25.42 | 25.73 | −0.91 / 0 / 0 | 1,143–1,211 |
| lora | 3e-4 | 21.21 / 23.56 / 22.77 | 22.51 | 22.51 | 0 / 0 / 0 | 1,148–1,150 |
| full_fine_tuning | 3e-4 | 24.02 / 25.61 / 23.83 | 24.48 | 24.31 | +0.90 / −0.10 / −0.27 | 973–1,003 |
| full_fine_tuning | 1e-4 | 22.02 / 26.37 / 25.76 | 24.72 | 24.64 | 0 / +0.28 / −0.05 | 1,014–1,136 |
| full_fine_tuning | 3e-5 | 21.69 / 23.73 / 24.00 | 23.14 | 23.14 | 0 / 0 / 0 | 1,046–1,062 |

The rule chooses the same four peaks on either accelerator. What the differences say: at the
smallest peak of every arm, and for the probe at every peak, the two accelerators agree to the
hundredth in every seed, and the disagreement grows with the peak — up to 1.8 RMSE in one seed
at 3e-4 from scratch. A run that steps further amplifies the last bits of the arithmetic the two
devices differ in; a run that steps less stays on one trajectory. The chosen peaks of the three
arms that step the encoder or an update beside it are the smallest swept for two of them, so the
grid's numbers under them should travel between accelerators better than the first grid's did.
An arm at this budget costs 1.3–1.6× on MPS what it costs on the T4.

## 2026-09-21 — Linux x86_64 (Kaggle, two Tesla T4, fp32): the sweep of the pretrained arms under the retrained backbone

The same sweep as on 2026-09-20 for the three arms that start from the backbone — three peaks
per arm, 200 labelled windows under seeds 1, 2 and 3, 2,002 optimiser steps, the head's start —
under `backbone-cmapss-m-64` (`ccd28046-…`, weights `sha256:78b3c201…`, 40,512 steps; the four
doublings that chose it are in `manual-handoff.md`, 2026-09-21), code `5658c88`. The control
was not run again: it never sees the backbone, and its cells of 2026-09-20 stand. Nine
configurations dealt over the two accelerators, fetched back into
`data/report/transfer/resweep-20260921-kaggle/<arm>-lr<peak>` (archives `9fe3a98e…`,
`e90aeb5f…`, `1c5f9ca9…` for the probe at 1e-2, 3e-3, 1e-3; `3276792a…`, `dceae235…`, `f402449d…`
for the low-rank arm at 3e-3, 1e-3, 3e-4; `048ba765…`, `17279f94…`, `749902ec…` for full
fine-tuning at 3e-4, 1e-4, 3e-5). Validation RMSE in cycles, the chosen peaks in bold, and the
mean of the same cell under the first backbone (2026-09-20) beside it:

| arm | peak | seed 1 | seed 2 | seed 3 | mean | under the first backbone | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **frozen_probe** | **1e-2** | 23.57 | 22.73 | 22.69 | **23.00** | 32.42 | 7 |
| frozen_probe | 3e-3 | 24.43 | 23.46 | 23.42 | 23.77 | 36.90 | 7 |
| frozen_probe | 1e-3 | 25.05 | 24.33 | 24.16 | 24.51 | 39.45 | 7 |
| lora | 3e-3 | 21.40 | 22.89 | 24.09 | 22.79 | 25.11 | 790–792 |
| lora | 1e-3 | 22.59 | 21.58 | 22.81 | 22.33 | 25.73 | 790 |
| **lora** | **3e-4** | 20.67 | 23.24 | 21.91 | **21.94** | 22.51 | 790 |
| **full_fine_tuning** | **3e-4** | 23.28 | 22.34 | 23.24 | **22.95** | 24.31 | 767 |
| full_fine_tuning | 1e-4 | 24.57 | 22.70 | 23.71 | 23.66 | 24.64 | 767 |
| full_fine_tuning | 3e-5 | 24.53 | 22.32 | 23.20 | 23.35 | 23.14 | 769 |

What the sweep says:

- **The retrained backbone is a far better fixed representation.** The probe falls from 32.4 to
  23.0 at every peak alike, nine points, and now stands within a point of the two arms that
  update the encoder; on the first backbone it stood ten points above them. Sixteen times the
  pretraining steps bought the frozen features most of what fine-tuning had been adding.
- **The arms that update the encoder gain half a point to a point**, the low-rank updates from
  22.51 to 21.94 and full fine-tuning from 23.14 to 22.95 at their best peaks, and full
  fine-tuning's peak moves up a decade, to 3e-4: a converged backbone tolerates a larger step.
- **The control still stands below all three at this budget**: 20.86 under three seeds, against
  21.94, 22.95 and 23.00. On the validation side at 200 labelled windows, under the floor and
  the retrained backbone, pretraining has not yet shown an advantage over training from scratch;
  the grid measures it over five seeds and four budgets.
- **Cost on a T4, fp32**: 0.38–0.40 s a step, as before; the nine configurations took 2.0 h
  over the two accelerators.


## 2026-09-21 — Linux x86_64 (Colab, NVIDIA L4, fp32): the sweep on the corpus normalised within one operating condition

The corpus the backbones had been pretrained on joined the four C-MAPSS subsets under one
normalisation per channel, and in FD002 and FD004 the operating condition swamps every sensor, so
the task's subset occupied a sliver of the scale (`docs/preregistration.md`, 2026-09-21). The
corpus published again with FD001 and FD003 alone (`durable/sha256/a9c73709…`), its backbone
chosen by the doublings of `manual-handoff.md` (2026-09-21, FD001 and FD003: 8 epochs, weights
`sha256:259fdc70…`), and all four arms swept on it by the rule of 2026-09-20: 200 labelled windows
under seeds 1, 2 and 3, 2,002 optimiser steps, the head's start, grids centred where the edge
rule had pointed on the old corpus. The task keeps its window, labels, strata and test engines and
takes its tuning and validation engines from this version's division — 79 and 21, against 82 and
18 before — so its numbers compare with the sections above in kind, not to the hundredth. Code
`c1c8a9c`, five processes sharing the device; each configuration published as it finished and
fetched back into `data/report/transfer/fd13-sweep/<arm>-lr<peak>` (archives, in the table's
order: `fd46eb61…`, `2ec07503…`, `c06abeb5…`, `02a6a293…`; `a6e55796…`, `66c7bea3…`, `43ec6e50…`;
`b2c49d62…`, `6911f941…`, `4442bb04…`; `8aad577b…`, `eae69043…`, `f3c9aa97…`). Validation RMSE in
cycles, the lowest mean of each arm in bold:

| arm | peak | seed 1 | seed 2 | seed 3 | mean | SD | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| from_scratch | 1e-2 | 22.77 | 20.65 | 22.38 | 21.93 | 1.13 | 1,530–1,537 |
| **from_scratch** | **3e-3** | 17.45 | 19.38 | 19.32 | **18.72** | 1.10 | 1,180–1,189 |
| from_scratch | 1e-3 | 17.87 | 20.99 | 20.56 | 19.81 | 1.69 | 1,179–1,188 |
| from_scratch | 3e-4 | 19.96 | 22.08 | 21.62 | 21.22 | 1.11 | 1,180–1,189 |
| **frozen_probe** | **3e-2** | 19.73 | 19.88 | 20.16 | **19.93** | 0.22 | 52–53 |
| frozen_probe | 1e-2 | 21.47 | 21.59 | 21.65 | 21.57 | 0.09 | 52–55 |
| frozen_probe | 3e-3 | 23.23 | 23.27 | 23.43 | 23.31 | 0.11 | 50–52 |
| lora | 1e-3 | 16.69 | 18.88 | 18.52 | 18.03 | 1.17 | 1,487–1,837 |
| lora | 3e-4 | 16.53 | 20.94 | 19.57 | 19.01 | 2.26 | 730–935 |
| **lora** | **1e-4** | 15.95 | 18.44 | 17.39 | **17.26** | 1.25 | 730–932 |
| **full_fine_tuning** | **1e-3** | 16.44 | 17.00 | 17.30 | **16.91** | 0.44 | 1,533–1,538 |
| full_fine_tuning | 3e-4 | 17.27 | 20.08 | 18.59 | 18.65 | 1.40 | 1,533–1,537 |
| full_fine_tuning | 1e-4 | 18.80 | 22.23 | 19.61 | 20.21 | 1.79 | 1,533–1,538 |

What the sweep says:

- **The pretrained arms stand below the control for the first time.** Full fine-tuning 16.91 and
  the low-rank updates 17.26 against 18.72; on the four subsets under the backbone of 64 epochs
  they stood above it, 22.95 and 21.94 against 20.86. The probe falls from 23.00 to 19.93:
  the frozen features now carry the engine's state, where before they carried its operating
  condition.
- **The control gains too**, from 20.86 to 18.72, because the task's windows are normalised by
  the same corpus's statistics, so its input spans the scale as well.
- **Three of the four peaks sit at an edge** — the probe and full fine-tuning at the top, the
  low-rank updates at the bottom — and go on to the edge rule; the control's edge at 1e-2 came
  out worse, which leaves it inside its grid at 3e-3.
- **Every arm that updates the encoder fits its 200 windows exactly**: the training loss ends
  at 0.0000 over the 2,002 steps, without weight decay; the low-rank updates at 1e-4 end at
  0.006–0.008 and the probe cannot fit them at all. The schedule is the same for every arm.
- **The runs are stamped `c1c8a9c…-dirty`** because the run directory lay untracked inside the
  checkout, which the revision reads as a change; the code is that commit's.
- **Cost**: 1,180–1,540 s a run with five sharing the device, about 2.5 hours for the thirteen
  configurations.

## 2026-09-22 — Darwin arm64 (MacBook Pro M1 Pro, MPS, fp32): the edges and the endpoint on FD001 and FD003

The rule the endpoint is read by settled before either ran (`docs/preregistration.md`,
2026-09-22): confirmed when the relative reduction is at least 10 per cent, the whole 95 per cent
interval lies above zero and the practical floor does not swallow it. Code `d5a181e`, one run at
a time, overnight; each edge run only when the edge rule asked for it, then five seeds of the
four arms under the peaks it named, one directory per arm and seed
(`data/report/transfer/fd13-edges/`, `data/report/transfer/fd13-endpoint/`), read together by
`scripts/label_curve_report.py` (`data/report/curve/fd13-endpoint/`). The edges, three seeds
each:

| arm | peak | seed 1 | seed 2 | seed 3 | mean | SD | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_probe | 1e-1 | 18.91 | 19.06 | 19.19 | 19.05 | 0.14 | 12–14 |
| **frozen_probe** | **3e-1** | 18.30 | 18.53 | 19.15 | **18.66** | 0.44 | 12–13 |
| lora | 3e-5 | 20.45 | 21.49 | 22.26 | 21.40 | 0.91 | 1,143–1,268 |
| full_fine_tuning | 3e-3 | 26.64 | 21.10 | 20.40 | 22.71 | 3.42 | 970–971 |

The probe's peak moved twice and stops at 3e-1, still at the top edge, where the rule allows no
third step; the low-rank updates stay at 1e-4 and full fine-tuning at 1e-3, both now inside their
grids. The endpoint, validation RMSE per seed:

| arm | peak | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| from_scratch | 3e-3 | 18.75 | 19.22 | 19.24 | 18.75 | 20.55 | 19.30 | 0.74 |
| frozen_probe | 3e-1 | 18.30 | 18.53 | 19.15 | 19.06 | 19.29 | 18.86 | 0.43 |
| lora | 1e-4 | 15.95 | 18.44 | 17.39 | 17.32 | 17.58 | 17.34 | 0.89 |
| full_fine_tuning | 1e-3 | 17.12 | 17.19 | 17.72 | 17.82 | 16.56 | 17.28 | 0.51 |

Against the control, pooled over the five seeds, the interval a bootstrap over the 21 validation
engines (10,000 resamples); trivial predictors over the same 618 windows: the mean 40.84, the
ceiling 67.07.

| mode | control | candidate | reduction | relative | 95 % interval | p | floor | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_probe | 19.31 | 18.87 | +0.44 | +2.3 % | [−1.08, +1.93] | 0.5625 | 0.74 | indistinguishable |
| lora | 19.31 | 17.35 | +1.96 | +10.1 % | [+0.58, +3.44] | 0.0034 | 0.74 | distinguishable |
| **full_fine_tuning** | 19.31 | 17.29 | **+2.02** | **+10.5 %** | **[+0.98, +3.13]** | 0.0002 | 0.74 | **confirmed** |

Beside it, mean over the seeds: RMSE below the ceiling 18.87, 17.56, 17.20 and 16.90 in the
table's order of arms (control first); on each engine's last window 4.89, 9.10, 6.69 and 4.27;
the share within 20 per cent of the label 0.53, 0.48, 0.54 and 0.56; the asymmetric score 10.55,
6.45, 6.49 and 6.30.

How much the verdict rests on, read the same way over fewer seeds:

| seeds read | control | full fine-tuning | relative | 95 % interval | verdict |
| --- | --- | --- | --- | --- | --- |
| all five | 19.31 | 17.29 | +10.5 % | [+0.98, +3.13] | confirmed |
| without seed 1 | 19.45 | 17.33 | +10.9 % | [+0.98, +3.31] | confirmed |
| without seed 2 | 19.33 | 17.31 | +10.5 % | [+0.89, +3.21] | confirmed |
| without seed 3 | 19.33 | 17.18 | +11.1 % | [+1.25, +3.14] | confirmed |
| without seed 4 | 19.45 | 17.15 | +11.8 % | [+1.13, +3.53] | confirmed |
| without seed 5 | 18.99 | 17.47 | +8.0 % | [+0.35, +2.74] | below the registered reduction |
| seeds 1–3 | 19.07 | 17.35 | +9.0 % | [+0.31, +3.15] | below the registered reduction |

What the endpoint says:

- **Confirmed by the registered rule, on the validation side**: at 200 labelled windows full
  fine-tuning from the backbone takes 10.5 per cent off training from scratch, its whole interval
  above zero and more than twice the floor. The low-rank updates come within a tenth of a point of
  it and are distinguishable in the family. Preliminary: the grid over four budgets and the single
  test run are still to come.
- **The margin over 10 per cent is thin.** The interval says the reduction is real; whether it
  reaches a tenth rests on the control's fifth seed, its worst at 20.55: without that seed the
  reduction is 8.0 per cent, and over seeds 1–3 it is 9.0 here and 9.7 on the L4. The
  reduction lies near the threshold, and one more run of five seeds could fall on either side.
- **The two accelerators agree where a run steps little.** The low-rank updates at 1e-4
  reproduce the L4's three seeds to the hundredth; the arms at larger peaks drift by up to 0.7
  (full fine-tuning) and 1.3 (the control's first seed), as on 2026-09-20. The edges read on
  this machine against the sweep on the L4 are decided by far larger differences — 21.40 and
  22.71 against 17.26 and 16.91 — so the drift did not choose a peak.
- **The probe is weakest at the end of life**, 9.10 on each engine's last window against 4.27 to
  6.69 for the others, while on the windows below the ceiling it stands ahead of the control: a
  fixed linear read-out of the features places an engine's stage well and its last cycles poorly.
- **Cost**: 970 s a run of full fine-tuning or from scratch, 1,144 s of the low-rank updates,
  13 s of the probe; the edges and the endpoint, 32 runs, 6.2 hours.

## 2026-09-22 — Darwin arm64 (MacBook Pro M1 Pro, MPS, fp32): whether a longer backbone helps the task, exploratory

The doublings over FD001 and FD003 (`manual-handoff.md`, 2026-09-21) named the backbone of 8
epochs by the pretext's loss, which fell a further 2.6 per cent from 8 to 32 epochs. Whether the
task gains what the pretext no longer shows was registered as a check with nothing to be chosen
by it (`docs/preregistration.md`, 2026-09-22). It ran at `69aaa71` and `f0161e7`, whose code is
the endpoint's `d5a181e`, at 200 labelled windows under seeds 1 to 3 with the endpoint's
schedule, one run at a time (`data/report/transfer/fd13-longer/`). The cells under the backbone
of 8 epochs are the endpoint's and its edges'. Validation RMSE in cycles, mean over the three
seeds:

| backbone | weights | probe 1e-1 | probe 3e-1 | probe 1 | full fine-tuning 1e-3 |
| --- | --- | --- | --- | --- | --- |
| 4 epochs | `16d9d2e7…` | 19.49 | 19.21 | 19.37 | — |
| 8 epochs | `259fdc70…` | 19.05 | 18.66 | 18.77 | 17.34 |
| 16 epochs | `1999ecf7…` | 16.93 | **16.82** | 17.47 | **16.65** |
| 32 epochs | `696d75c2…` | 19.86 | 19.71 | 21.43 | 17.49 |

Full fine-tuning per seed: 17.12, 17.19 and 17.72 under 8 epochs; 16.13, 17.19 and 16.63 under
16; 18.48, 16.37 and 17.60 under 32.

What the check says:

- **The registered prediction failed.** Under 32 epochs full fine-tuning is worse than under 8 on
  the first seed and better on the second. The pretext's plateau is therefore read as the
  task's, and what a longer backbone over these two subsets lacks is data rather than steps.
- **The task's best backbone is not the pretext's.** Both arms score lowest under 16 epochs: the
  probe 1.8 and full fine-tuning 0.7 below the backbone of 8. Under 32 epochs both are worse than
  under 8. The rule names a backbone by the pretext's loss alone, and the task does not follow
  that loss monotonically. Three seeds at one budget are too few to choose by, and nothing was
  chosen.
- **The probe peaks at 3e-1 under every backbone**, so the ranking of the backbones does not
  depend on one learning rate.
- **Cost**: about 1,000 s a run of full fine-tuning and 13 s a run of the probe; about 1.9 hours in
  all.

## 2026-09-22 — Linux x86_64 (Colab, NVIDIA A100, fp32): the sweep, the endpoint and the replacement over the four subsets read per operating condition

**The configuration.**

- *The corpus.* The four C-MAPSS subsets are read with a channel per sensor and operating
  condition, each channel scaled within its condition (ADR-0034). They are published at window 50
  and stride 5 (`durable/sha256/d63f8e1b…`): 126 channels and 25,395 windows, against 7,192 over
  FD001 and FD003 alone.
- *The split.* FD001's and FD003's units are held out exactly as the version of those two holds
  them out. The task therefore keeps its 79 tuning engines, its 21 validation engines and its 618
  validation windows.
- *The backbone.* The run of 8 epochs chosen by the doublings (`manual-handoff.md`, 2026-09-22;
  weights `sha256:6283c210…`).
- *How it ran.* Everything below was registered before it ran (`docs/preregistration.md`,
  2026-09-22). It ran at `fdf8053`, whose code differs from the endpoint's `d5a181e` only in the
  reader and the command that publishes a corpus. The device was one A100 of a paid notebook,
  shared by five processes, and each configuration was published as it finished.

**The sweep.** 200 labelled windows under seeds 1, 2 and 3, 2,002 optimiser steps and the head's
start. The grids are centred on the peaks of FD001 and FD003 and extended by the edge rule. Each
configuration is fetched into `data/report/transfer/cond-sweep/<arm>-lr<peak>`; the archives, in
the table's order, are `0979baeb…`, `1c535374…`, `0dce0cc5…`, `1a9b5a3e…`; `1f784539…`,
`ea6c18b8…`, `7f6bec0b…`, `779b6420…`, `39339414…`; `b02f66fd…`, `cb70480f…`, `78150c05…`;
`885dc9eb…`, `7ef10679…`, `43837b78…`. Validation RMSE in cycles, with the lowest mean of each arm
in bold:

| arm | peak | seed 1 | seed 2 | seed 3 | mean | SD | seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| from_scratch | 1e-2 | 23.41 | 23.95 | 21.26 | 22.87 | 1.42 | 477–482 |
| from_scratch | 3e-3 | 19.19 | 19.75 | 20.32 | 19.75 | 0.57 | 477–481 |
| **from_scratch** | **1e-3** | 19.07 | 18.32 | 18.92 | **18.77** | 0.40 | 477–482 |
| from_scratch | 3e-4 | 21.01 | 24.28 | 21.13 | 22.14 | 1.85 | 624–749 |
| frozen_probe | 1 | 17.20 | 17.32 | 16.55 | 17.02 | 0.41 | 39–45 |
| frozen_probe | 3e-1 | 16.27 | 16.54 | 16.35 | 16.38 | 0.14 | 40–45 |
| frozen_probe | 1e-1 | 16.21 | 16.35 | 16.17 | 16.24 | 0.10 | 40–45 |
| **frozen_probe** | **3e-2** | 16.23 | 16.23 | 16.00 | **16.15** | 0.13 | 3–5 |
| frozen_probe | 1e-2 | 16.24 | 16.23 | 16.05 | 16.17 | 0.10 | 3–5 |
| lora | 3e-4 | 16.49 | 17.37 | 17.29 | 17.05 | 0.49 | 693–731 |
| **lora** | **1e-4** | 15.74 | 15.60 | 16.85 | **16.07** | 0.69 | 693–731 |
| lora | 3e-5 | 18.25 | 19.63 | 18.39 | 18.76 | 0.76 | 604–728 |
| full_fine_tuning | 3e-3 | 22.59 | 21.43 | 24.26 | 22.76 | 1.42 | 249–369 |
| **full_fine_tuning** | **1e-3** | 15.59 | 15.99 | 15.92 | **15.83** | 0.21 | 258–405 |
| full_fine_tuning | 3e-4 | 15.71 | 16.19 | 16.15 | 16.02 | 0.26 | 685–751 |

- **Where the peaks landed.** The control's best grid point sat at the lower edge, and the edge
  rule's extra step put its peak at 1e-3. The probe's peak moved down twice and stopped at 3e-2.
  Both now lie inside their grids. The low-rank updates stay at 1e-4 and full fine-tuning at
  1e-3.
- **The control is where it was**: 18.77 here against 18.72 over FD001 and FD003, because the
  task's own channels keep their scale under either corpus. Every gain of the pretrained arms
  came with the backbone.
- **The probe changed most.** A frozen encoder with a linear head scores 16.15 against the
  control's 18.77. Over FD001 and FD003 it scored 19.93, worse than the control.
- **Cost**: the seconds differ with how many processes shared the device at the time. The sweep
  took about an hour.

**The endpoint.** Five seeds of the four arms at 200 labelled windows under those peaks. Beside
them, full fine-tuning under the backbone of FD001 and FD003 ran again on the same device, with
its weights (`sha256:259fdc70…`), its corpus and its peak of 1e-3, so that both sides of the
replacement rule come from one accelerator. The runs are fetched into
`data/report/transfer/cond-endpoint-colab/<arm>-s<seed>` and
`data/report/transfer/cond-endpoint-colab-old/`. Validation RMSE per seed:

| arm | peak | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | mean | SD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| from_scratch | 1e-3 | 19.18 | 18.37 | 19.28 | 18.52 | 19.71 | 19.01 | 0.56 |
| frozen_probe | 3e-2 | 16.23 | 16.23 | 16.00 | 17.19 | 15.95 | 16.32 | 0.50 |
| lora | 1e-4 | 15.74 | 15.60 | 16.85 | 16.57 | 15.68 | 16.09 | 0.58 |
| full_fine_tuning | 1e-3 | 15.13 | 15.63 | 16.08 | 18.46 | 17.85 | 16.63 | 1.45 |
| full_fine_tuning under the backbone of FD001 and FD003 | 1e-3 | 16.55 | 17.42 | 17.76 | 17.67 | 16.72 | 17.22 | 0.56 |

Against the control, pooled over the five seeds, with the interval from a bootstrap over the 21
validation engines (10,000 resamples):

| mode | control | candidate | reduction | relative | 95 % interval | p | floor | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_probe | 19.02 | 16.33 | +2.69 | +14.2 % | [+1.44, +4.04] | 0.0002 | 0.57 | distinguishable |
| lora | 19.02 | 16.10 | +2.92 | +15.4 % | [+1.46, +4.46] | 0.0002 | 0.57 | distinguishable |
| **full_fine_tuning** | 19.02 | 16.68 | **+2.34** | **+12.3 %** | **[+1.49, +3.30]** | 0.0002 | 0.57 | **confirmed** |

The replacement rule compares full fine-tuning under the new backbone with the same arm under the
old, paired on the same engines and windows, by the Evaluation context's paired bootstrap over
the engines (10,000 resamples; `scripts/backbone_comparison_report.py`). It gives +0.55 (+3.2 per cent), interval
[+0.05, +1.03], p 0.036. The interval lies above zero and the endpoint is confirmed, so the four
subsets read per operating condition and this backbone became the configuration.

How much each verdict rests on, read over four seeds of the five:

| seeds read | endpoint | endpoint's verdict | replacement | replacement's interval |
| --- | --- | --- | --- | --- |
| all five | +12.3 % | confirmed | +3.2 % | [+0.05, +1.03] |
| without seed 1 | +10.2 % | confirmed | +2.0 % | [−0.16, +0.85] |
| without seed 2 | +11.7 % | confirmed | +1.5 % | [−0.39, +0.81] |
| without seed 3 | +11.2 % | confirmed | +1.6 % | [−0.33, +0.83] |
| without seed 4 | +15.3 % | confirmed | +5.3 % | [+0.32, +1.53] |
| without seed 5 | +13.1 % | confirmed | +5.7 % | [+0.51, +1.47] |

What it says:

- **The endpoint is confirmed with a wider margin than over FD001 and FD003.** Here the
  reduction stays above the tenth in every reading over four seeds; there, one such reading fell
  to 8.0.
- **The replacement is narrow.** The new backbone beats the old one by 3 per cent, and in three
  of the readings over four seeds the interval includes zero. Under seeds 1 to 3 full fine-tuning
  is better under the new backbone; under seeds 4 and 5 it is worse (18.46 and 17.85 against
  17.67 and 16.72). The rule was read as registered, and the configuration moved on it.
- **The arms that step the whole encoder drift between identical runs on this device.** The probe
  and the low-rank updates repeat the sweep's seeds 1 to 3 to the hundredth. The control and full
  fine-tuning move by up to 0.46, for example full fine-tuning under seed 1: 15.59 in the sweep
  and 15.13 here.
- **Cost**: about 620 s a run of the arms that step the encoder, with five processes sharing the
  device, and 22–58 s a run of the probe. The 25 runs took under an hour.

## 2026-09-22 — Linux x86_64 (Kaggle, two Tesla T4, fp32): the grid on FD001 and FD003, the record of that configuration

This grid was registered before it ran (`docs/preregistration.md`, 2026-09-22). It finished
after the configuration had moved to the four subsets (section above), so it is reported as the
record of that configuration, not as the curve.

- *Code and inputs.* Code `d5a181e`, the commit of the endpoint measured on this machine; torch
  2.10.0+cu128 as preinstalled; the backbone of 8 epochs over FD001 and FD003 (weights
  `sha256:259fdc70…`) and the corpus `a9c73709…`.
- *The peaks the edges named*: from scratch 3e-3, probe 3e-1, low-rank updates 1e-4, full
  fine-tuning 1e-3.
- *How it ran.* One session with a process per device, and each seed's four arms on one device.
  `cuda:0` ran seeds 1 to 3 at 50 and 1,000 and seeds 1 and 2 at all, 6.8 hours inside the cells;
  `cuda:1` ran the rest in 6.9 hours.
- *Where it is.* Fetched as `data/report/transfer/b5-kaggle/{d0,d1}-{small,all}` (final
  references `b936a5d9…`, `3d1e53cf…`, `a3d23dca…` and `5add2613…`) and read together with the
  endpoint (`data/report/curve/fd13-grid`).

Relative reduction of the validation RMSE against the control, read by the registered family, in
bold where it is distinguishable or confirmed; the control scored 22.70, 19.31, 16.24 and 15.74
at the four budgets (`data/report/curve/fd13-grid/comparisons.csv` holds the intervals):

| budget | frozen_probe | lora | full_fine_tuning |
| --- | --- | --- | --- |
| 50 | +1.9 % | **+12.0 %** | **+12.5 %** |
| 200 (the endpoint on this machine) | +2.3 % | **+10.1 %** | **+10.5 %, confirmed** |
| 1000 | −12.6 % | +2.7 % | **+12.5 %** |
| all | −5.2 % | **+15.2 %** | −3.6 % |

What it says:

- **Up to 1,000 labelled windows full fine-tuning leads the control by 10.5 to 12.5 per cent**
  and is distinguishable at every budget. The low-rank updates lead at 50 and 200 and not at
  1,000; the probe leads nowhere.
- **The column at all labelled windows does not compare settled arms.**
  - The control at 3e-3 ended its 4,830 steps at a training loss of 0.0065–0.0135, where at
    1,000 it had reached 0.0007–0.0020.
  - Full fine-tuning's loss jumped by 1.6 to 6.8 times within the run under every seed.
  - The low-rank updates' +15.2 per cent is therefore a lead over a control that did not fit its
    windows. Over the four subsets, where the control's peak is 1e-3, the control reaches 12.28
    at this budget against 15.74 here, and the low-rank updates 12.81 against 13.34.
- **Cost on a T4**: 0.29–0.32 s a step at 50 and 0.37–0.41 s at 1,000 and at all, for every arm
  that steps the encoder or an update beside it; 7–27 s a run of the probe.

## 2026-09-22 — Linux x86_64 (Colab, NVIDIA A100, fp32): the grid over the four subsets read per operating condition

The grid under the floor on the configuration the replacement rule chose, registered before it
ran (`docs/preregistration.md`, 2026-09-22).

- *What ran.* Budgets of 50, 1,000 and all 2,568 labelled windows; the four arms; seeds 1 to 5;
  the peaks of the sweep above; 2,000 optimiser steps at least and the head's start. The cell at
  200 is the endpoint above and did not run again.
- *Where it ran.* At `fdf8053`, the endpoint's commit, on the A100, five processes at once. Each
  directory holds one budget and seed with its four arms and was published as it landed.
- *Where it is.* Fetched as `data/report/transfer/cond-grid/b<budget>-s<seed>` (fifteen archives,
  one per budget and seed) and read with the endpoint (`data/report/curve/cond-grid`). Read here
  again from the fetched archives, the comparisons match the notebook's digit for digit.

**Endpoint RMSE per cell, mean ± SD over seeds** (the engines column is how many engines the
budget's labels came from, over the seeds):

| budget | engines | from_scratch | frozen_probe | lora | full_fine_tuning |
| --- | --- | --- | --- | --- | --- |
| 50 | 33-39 | 23.49 ± 1.19 | 18.63 ± 1.37 | 18.24 ± 1.59 | 19.71 ± 1.35 |
| 200 | 68-75 | 19.01 ± 0.56 | 16.32 ± 0.50 | 16.09 ± 0.58 | 16.63 ± 1.45 |
| 1000 | 79 | 15.20 ± 1.36 | 16.08 ± 0.23 | 14.45 ± 0.39 | 13.87 ± 0.30 |
| all | 79 | 12.26 ± 0.65 | 15.72 ± 0.33 | 12.80 ± 0.37 | 12.88 ± 0.80 |

Trivial predictors over the same windows: the mean predictor 40.84, the ceiling predictor 67.07.

**Against the control arm**, pooled over the seeds both arms hold, with the interval from a
bootstrap over the 21 engines (10,000 resamples). The endpoint stands alone; the other eleven
cells are the registered family under the Holm correction at 5 %:

| mode | budget | control | candidate | reduction | relative | 95 % interval | p | floor | rejected | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen_probe | 50 | 23.51 | 18.67 | +4.84 | +20.6% | [+2.64, +7.17] | 0.0002 | 1.19 | yes | distinguishable |
| lora | 50 | 23.51 | 18.30 | +5.22 | +22.2% | [+3.58, +6.91] | 0.0002 | 1.19 | yes | distinguishable |
| full_fine_tuning | 50 | 23.51 | 19.74 | +3.77 | +16.0% | [+2.55, +5.03] | 0.0002 | 1.19 | yes | distinguishable |
| frozen_probe | 200 | 19.02 | 16.33 | +2.69 | +14.2% | [+1.44, +4.04] | 0.0002 | 0.57 | yes | distinguishable |
| lora | 200 | 19.02 | 16.10 | +2.92 | +15.4% | [+1.46, +4.46] | 0.0002 | 0.57 | yes | distinguishable |
| full_fine_tuning | 200 | 19.02 | 16.68 | +2.34 | +12.3% | [+1.49, +3.30] | 0.0002 | 0.57 | primary | confirmed |
| frozen_probe | 1000 | 15.25 | 16.08 | -0.83 | -5.4% | [-1.90, +0.31] | 0.1558 | 1.36 | no | indistinguishable |
| lora | 1000 | 15.25 | 14.45 | +0.80 | +5.2% | [-0.26, +1.78] | 0.1302 | 1.36 | no | indistinguishable |
| full_fine_tuning | 1000 | 15.25 | 13.87 | +1.38 | +9.0% | [+0.42, +2.32] | 0.0038 | 1.36 | yes | distinguishable |
| frozen_probe | all | 12.28 | 15.72 | -3.44 | -28.0% | [-4.94, -1.90] | 0.0002 | 0.65 | yes | worse |
| lora | all | 12.28 | 12.81 | -0.53 | -4.3% | [-1.89, +0.82] | 0.4310 | 0.65 | no | indistinguishable |
| full_fine_tuning | all | 12.28 | 12.90 | -0.62 | -5.1% | [-1.55, +0.26] | 0.1668 | 0.65 | no | indistinguishable |

![Label-efficiency curve](figures/label-efficiency-curve.png)

**Conclusion.** Confirmed on the registered endpoint: at 200 labelled windows, full fine-tuning
lowers the validation RMSE by +12% (+2.34, 95 % interval [+1.49, +3.30] over 5 seeds); among the
secondary budgets the advantage of full fine-tuning holds under the Holm correction and above the
floor at 50 and 1000, not at the full label set. Preliminary; validation, not test.

How much the curve rests on, full fine-tuning read the same way over fewer seeds:

| seeds read | 50 | 200 (endpoint) | 1000 | all |
| --- | --- | --- | --- | --- |
| all five | +16.0 %, distinguishable | +12.3 %, confirmed | +9.0 %, distinguishable | −5.1 %, indistinguishable |
| without seed 1 | +14.4 %, distinguishable | +10.2 %, confirmed | +9.4 %, distinguishable, practically nil | −8.7 %, indistinguishable |
| without seed 2 | +17.7 %, distinguishable | +11.7 %, confirmed | +11.9 %, distinguishable | −2.5 %, indistinguishable |
| without seed 3 | +13.9 %, distinguishable | +11.2 %, confirmed | +6.5 %, indistinguishable | −4.0 %, indistinguishable |
| without seed 4 | +17.6 %, distinguishable | +15.3 %, confirmed | +9.9 %, distinguishable | −3.9 %, indistinguishable |
| without seed 5 | +16.4 %, distinguishable | +13.1 %, confirmed | +7.2 %, indistinguishable | −6.4 %, indistinguishable |

In every one of these readings the probe and the low-rank updates stay distinguishable at 50 and
at 200, and the probe stays worse at all labelled windows.

**Seed by seed**, at 50 and at 200 labelled windows every pretrained arm scores below the control
under every one of the five seeds (from +6.0 to +31.1 per cent at 50, from +0.3 to +21.1 at 200).
The sweep chose the peaks at 200 under seeds 1 to 3, so at that budget only seeds 4 and 5 are
fresh. Over them the probe and the low-rank updates keep what they show over seeds 1 to 3
(+13.3 and +15.6 per cent against +14.7 and +15.2), but full fine-tuning falls from +17.6 to +5.0
(+0.3 under seed 4, +9.4 under seed 5). Its sweep was flat near the peak (15.83 at 1e-3, 16.02 at
3e-4), so the choice of the peak explains little of the gap. Full fine-tuning, which fits its 200
windows exactly with 4.8 million weights, varies most with the draw of the labels; its margin over
the tenth rests on the seeds the peak was chosen on, where the low-rank updates' does not.

What the curve says:

- **The endpoint holds, and the advantage is largest where labels are scarcest.** At 50 labelled
  windows, from 33 to 39 engines, every pretrained arm leads the control by 16 to 22 per cent,
  distinguishable in every reading over four seeds. The two arms that change the fewest weights
  lead most there: the low-rank updates by 22.2 per cent and the probe by 20.6.
- **The low-rank updates are the steadiest arm**: distinguishable at 50 and 200 in every reading,
  and as far ahead of the control on the fresh seeds as on the others (above).
- **Every configuration choice was read on the same 21 validation engines**: the window, the
  normalisation, the corpus, the backbone and the peaks. Each was registered before its run, but
  a configuration kept because it did better on these engines scores optimistically on them; the
  single run on the frozen test engines is what measures it without that bias.
- **At 1,000 the lead of full fine-tuning is fragile.** It is 9.0 per cent overall, and over four
  seeds it clears its floor in two readings of five.
- **Once every label is used, the pretrained arms that update the encoder only match the
  control.** They are indistinguishable from it (−5.1 and −4.3 per cent), and the probe is worse
  (−28.0). With 2,568 windows from 79 engines the control reaches 12.28, and the pretrained
  weights do not improve on it. Two things the column does not settle:
  - Full fine-tuning at 1e-3 does not settle under two seeds of five at this budget: its training
    loss ends at 0.007–0.008, one run after a jump of 3.2 times, while the control ends at
    0.0002–0.0004 under every seed. A peak chosen over 2,002 steps may not suit 4,830.
  - At this budget the control sees every tuning engine labelled, and the backbone saw those
    same engines unlabelled, so what the backbone adds here is only the other subsets.
- **The probe's error stops falling near 16** from 1,000 labelled windows on (16.08, then
  15.72), while the arms that update the encoder go on to 12.8–12.9. From 200 windows on, its
  error on each engine's last window stays at 8.4–11.2, against 2.4–4.8 for the control at the
  same budgets. A linear head over frozen features places an engine's stage of life and not its
  last cycles.
- **Up to 1,000 windows the pretrained arms err late less often.** The asymmetric score, which
  punishes a late answer exponentially, is 8.4–11.7 for them against 21.2 for the control at 50
  windows, and 4.7–6.7 against 10.0 at 200.
- **Against the record of FD001 and FD003** (section above), this configuration leads by more at
  50 and 200. Its control is stronger at all labelled windows, where that grid's column compared
  arms that did not settle.

**Two readings beside the endpoint**, mean ± SD over seeds — read, not thresholded:

RMSE on the last window of each engine (the error at the end of life on this side, not
comparable with published test-side numbers):

| budget | engines | from_scratch | frozen_probe | lora | full_fine_tuning |
| --- | --- | --- | --- | --- | --- |
| 50 | 33-39 | 11.84 ± 5.95 | 17.46 ± 3.84 | 7.29 ± 2.49 | 7.20 ± 2.65 |
| 200 | 68-75 | 4.80 ± 1.34 | 11.15 ± 1.50 | 5.06 ± 0.41 | 4.31 ± 1.08 |
| 1000 | 79 | 2.81 ± 0.27 | 9.55 ± 0.45 | 4.45 ± 0.80 | 3.12 ± 0.45 |
| all | 79 | 2.36 ± 0.33 | 8.36 ± 0.63 | 3.70 ± 0.82 | 3.37 ± 0.81 |

Asymmetric score (mean per window, lower is better; its exponential tail is carried by a few
engines):

| budget | engines | from_scratch | frozen_probe | lora | full_fine_tuning |
| --- | --- | --- | --- | --- | --- |
| 50 | 33-39 | 21.24 ± 2.57 | 8.58 ± 3.35 | 8.40 ± 3.12 | 11.74 ± 3.37 |
| 200 | 68-75 | 10.00 ± 2.05 | 4.68 ± 0.50 | 5.64 ± 0.78 | 6.68 ± 2.44 |
| 1000 | 79 | 4.75 ± 1.44 | 4.53 ± 0.48 | 3.78 ± 0.19 | 3.78 ± 0.23 |
| all | 79 | 2.71 ± 0.41 | 4.25 ± 0.55 | 2.89 ± 0.28 | 3.01 ± 0.51 |

**Cost** on the A100 shared by five processes: a run of an arm that steps the encoder or an update
beside it took 507–536 s at 50 labelled windows, 646–675 s at 1,000 and 1,567–1,622 s at all; a
run of the probe 17–34 s. The 60 runs took about 2.3 hours, as declared.
