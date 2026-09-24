# Classical baselines: the spectrum, the grid, MiniRocket, and the selection of their knobs

What is measured here: that the MiniRocket implemented in this project computes what the
reference implementation computes; that the corpus a comparison runs over is the one in force,
reproduced from the raw files; which setting of each classical baseline a declared selection
chose, budget by budget; and where those baselines stand on the turbofan task beside a network
trained from scratch, run on this machine through the order that stands in for a queue.

Every number of a comparison below is **validation**: it is read on the validation side of the
task or, for the selection, on units held out of the tuning side, and none of it is the result the
project publishes. Tier S, one repeat of the procedure.

## 2026-09-24 — Darwin arm64 (M1 Pro, 32 GB): the baselines of the turbofan task

**Where it ran.** The classical cells ran in the `worker-general` image built at `eb62475`
(Linux arm64 under Docker Desktop, four threads per fit, CPU); the cells of the network trained
from scratch ran on the host's MPS in fp32, through an order placed at `4058a62`
(`durable/sha256/5082ac7c…`), run by `emblema.entrypoints.cli.campaign_run` and accepted back.
MiniRocket's comparison with aeon ran natively under `uv run --with aeon` (aeon 1.6.0), by
`scripts/classical_baselines_report.py`.

### MiniRocket against aeon

Where no draw differs — one channel and a transform fitted to one series, or 42 channels fitted
under aeon's own random draws replayed through this project's fit — the features agree cell by
cell (`agreement.csv`):

| case | steps | features | largest difference | cells apart |
| --- | --- | --- | --- | --- |
| one channel, one series | 64 | 840 | 2.6e-08 | 0 |
| one channel, one series | 128 | 9996 | 7.8e-03 | 5.0e-06 |
| one channel, one series | 40 | 84 | 2.4e-08 | 0 |
| 42 channels, aeon's draws, seed 1 | 64 | 9996 | 2.6e-08 | 0 |
| 42 channels, aeon's draws, seed 2 | 64 | 9996 | 1.6e-02 | 5.0e-06 |
| 42 channels, aeon's draws, seed 3 | 64 | 9996 | 2.6e-08 | 0 |

A feature is a share of positions, so a cell apart is one position counted on the other side of
its bias: a convolution landing exactly on a bias in aeon's float32 and just past it in float64.
Five cells in a million.

Fitted on a published corpus — `cmapss` over all four subsets at window 50, 21 channels, on a grid
of 64 steps — and scored on 2,000 windows of a held-out 30 per cent of the engines, the target the
share of its engine's run a window ends at, twenty seeds, one ridge over the same scaled features
(`multivariate.csv`):

| budget | this project | aeon | paired gap | this project lower in | under aeon's draws |
| --- | --- | --- | --- | --- | --- |
| 200 | 0.2189 ± 0.0035 | 0.2175 ± 0.0038 | +0.0014 ± 0.0053 | 8 of 20 | at most 1.8e-03 from aeon |
| 1000 | 0.2006 ± 0.0041 | 0.1970 ± 0.0034 | +0.0035 ± 0.0050 | 6 of 20 | at most 7.5e-04 from aeon |

Under its own draws this project's fit is 0.6 and 1.8 per cent above aeon's; under aeon's draws
it lands on aeon's error to the third decimal. What differs is the random stream — NumPy's
`Generator` here, the legacy `RandomState` there, the same distribution — and not the arithmetic.
A fit costs 3.4 s at 200 windows and 4.7 s at 1,000, against aeon's JIT-compiled 1.2 and 1.8 s.
Both implementations read the same grid, whose readings the rounding error below never touched at
64 steps over 50 and whose 21 channels all hold readings, so this comparison stands as run.

### The corpus in force, published again

The corpus the configuration names (`cmapss`, every sensor a channel per operating condition,
all four subsets, window 50 at stride 5) was published again on this machine from the raw files
into the local store, with the 141 held-out units of the manifest in force named one by one:

    uv run python -m emblema.entrypoints.cli.publish_corpus --corpus cmapss \
      --root "data/raw/cmapss/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData" \
      --window 50 --stride 5 --per-operating-condition --hold-out <the 141 units>

The manifest came out `sha256:d63f8e1bb4b8b659e9482141fc3204a2f8f39c98857c984af0f49f219e3d3477`,
the manifest the configuration names, byte for byte. A first attempt with the held-out units
drawn by the default split gave another manifest (`560fc361…`): the same corpus version, source
checksum, units, windows and tokens, a different split — the one in force names its units rather
than drawing them. The task over it holds 79 tuning, 21 validation and 100 frozen engines.

### The selection of the knobs

Declared in `campaigns/selection-classical-fd001.toml`, committed before it ran (`305c2e9`), and
read by the rule the preregistration states (2026-09-24, *how a classical baseline is tuned*):
ten repeats, each holding out 16 of the 79 tuning engines and scoring the variants on them; the
rule of one standard error, with the variance scaled by 1/J + n_test/n_train for the overlap of
the repeats; the variant closest to the published setting among those within reach. The
defaults are the library's — XGBoost's 100 rounds at a rate of 0.3, depth 6, every row and column
— and the paper's for MiniRocket: 9,996 features, the ridge penalty chosen among ten from 10⁻³ to
10³ by leave-one-out error, and the series read on its own cadence, one step per cycle.

Selection `3856e705…`. RMSE on the held-out tuning engines, mean ± SD over the ten repeats;
**bold** is the variant the rule chose.

| candidate | variant | 50 | 200 |
| --- | --- | --- | --- |
| trees per channel | default (depth 6) | **20.83 ± 2.69** | 15.64 ± 0.88 |
| | depth 3 | 19.95 ± 3.19 | **14.48 ± 0.67** |
| | depth 9 | 20.81 ± 2.71 | 16.02 ± 1.24 |
| | rate 0.1, 300 rounds | 21.32 ± 3.64 | 15.16 ± 0.92 |
| trees over the spectrum | default | 27.08 ± 2.24 | 20.15 ± 1.17 |
| | depth 3 | **24.95 ± 1.98** | **18.99 ± 1.54** |
| | depth 9 | 27.08 ± 2.25 | 20.70 ± 1.30 |
| | rate 0.1, 300 rounds | 27.45 ± 2.49 | 19.68 ± 1.09 |
| trees across channels | default | 17.66 ± 1.45 | 14.79 ± 0.79 |
| | depth 3 | **16.90 ± 1.21** | **14.09 ± 0.90** |
| | depth 9 | 17.69 ± 1.50 | 15.08 ± 0.91 |
| | rate 0.1, 300 rounds | 17.83 ± 1.33 | 14.30 ± 0.77 |
| MiniRocket | grid × 0.5 | 19.03 ± 0.93 | 18.42 ± 0.92 |
| | default (grid × 1) | **17.40 ± 0.78** | **16.24 ± 1.26** |
| | grid × 2 | 17.93 ± 1.44 | 16.86 ± 0.88 |
| | grid × 4 | 18.08 ± 1.32 | 17.16 ± 0.95 |

- **The trees choose to be shallow.** Five of the six choices fall on a depth of 3; the sixth,
  trees per channel at 50 labels, keeps the default because the spread between repeats there
  (3.2) puts the default within reach of depth 3.
- **MiniRocket reads the series as recorded.** One step per cycle is best at both budgets, and
  every other resolution is worse.
- **More, smaller steps of boosting never pass the rule.**

### The comparison

Declared in `campaigns/baselines-fd001.toml` and `campaigns/classical-only-fd001.toml` with the
choices above named per budget and their selection (`4058a62`); declaring them read the selection
again and found the same choices, and each variant described exactly as the selection ran it. The
network trained from scratch is the control, under the configuration in force (30 epochs on a
floor of 2,000 optimiser steps, batches of 16, peak 10⁻³). Campaign `ee69d456…`; RMSE on the 21
validation engines, mean ± SD over three seeds.

| candidate | 50 | 200 |
| --- | --- | --- |
| trees per channel | 18.25 ± 1.47 | **13.94 ± 0.71** |
| trees across channels (the one that can cross a layout) | **16.59 ± 1.36** | 14.32 ± 0.92 |
| MiniRocket | 16.81 ± 0.79 | 15.24 ± 0.15 |
| trees over the spectrum | 21.61 ± 1.21 | 18.47 ± 0.25 |
| network from scratch (control) | 22.51 ± 1.61 | 18.86 ± 0.23 |

Against the control, pooled over the seeds, with the interval from a bootstrap over the 21
engines (10,000 resamples):

| candidate | budget | reduction | relative | 95 % interval | floor | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| **trees per channel** | **200** | **+4.91** | **+26.0 %** | **[+2.86, +6.88]** | 0.38 | **confirmed** |
| trees per channel | 50 | +4.26 | +18.9 % | [+0.82, +7.46] | 1.61 | indistinguishable |
| trees across channels | 200 | +4.53 | +24.0 % | [+3.02, +6.13] | 0.38 | distinguishable |
| trees across channels | 50 | +5.92 | +26.3 % | [+3.40, +8.25] | 1.61 | distinguishable |
| MiniRocket | 200 | +3.62 | +19.2 % | [+1.83, +5.40] | 0.38 | distinguishable |
| MiniRocket | 50 | +5.73 | +25.4 % | [+2.91, +8.35] | 1.61 | distinguishable |
| trees over the spectrum | 200 | +0.40 | +2.1 % | [−1.97, +2.72] | 0.38 | indistinguishable |
| trees over the spectrum | 50 | +0.91 | +4.1 % | [−2.99, +4.68] | 1.61 | indistinguishable |

What it says:

- **The same cells, run again, are the same numbers.** The two comparisons share the classical
  cells of seeds 1 and 2, and all sixteen agree to the last bit; the cells of the trees agree to the
  last bit with the same cells run before the corrections below, which touched no tree. The
  network from scratch moves by up to 1.1 in a seed between two identical runs on this device —
  the drift of an arm that steps the whole encoder, described in `label-efficiency-curve.md` —
  and scores 18.86 here where that note's grid scored 18.77 over the same seeds on an A100.
- **Every baseline but the spectrum is ahead of the network trained from scratch**, the best by a
  quarter of its error. The spectrum is no help on this task, whose target is a trend and not an
  oscillation.
- **Beside the pretrained arms.** On this corpus and task, under the same configuration,
  `label-efficiency-curve.md` reads at 200 labels: low-rank updates 16.10, the frozen probe
  16.33, full fine-tuning 16.68, all over five seeds. The trees per channel, at 13.94, are about
  13 per cent below the best of them, and MiniRocket, at 15.24, about 5 per cent. This is a reading
  across two campaigns, not a paired comparison, and the baselines were tuned at every budget
  while the arms were tuned at 200 alone; the comparison that pairs them on the same engines is
  still to be run.

### What a cell costs

Read off the cells as the campaigns recorded them: fitting and answering, the corpus already at
hand.

| candidate | 50 | 200 |
| --- | --- | --- |
| trees per channel | 0.31 s | 0.43 s |
| trees over the spectrum | 0.71 s | 0.98 s |
| trees across channels | 0.87 s | 1.14 s |
| MiniRocket (grid × 0.5 to × 4) | 2.7 s | 3.3 s |
| network from scratch (MPS, fp32) | 812 s | 979 s |

The selection's 320 cells took 14 minutes on one worker. The network's six cells took 90
minutes: the floor of 2,000 optimiser steps is what a cell of it costs at these budgets, not the
epochs.

### What the runs found, and what they superseded

The numbers above are the second run of the selection and the comparisons. The first
(selection `77e4f5f8…`, comparisons `62fb1f09…` and `b1e8712f…`, at `851efa9`) is superseded for
every MiniRocket cell, and stands for every other:

- **A reading at the start of a step was laid one step early.** A token's time is stored in
  single precision, so k / 50 is a hair below k / 50, and a floor put it in the step before: on
  the grid of one step per cycle, 16 of every channel's 50 steps were empty and their neighbours
  doubled. Corrected in `afe5199`; the grid of the turbofan windows is now observed at every step.
- **The grid was laid over the whole vocabulary.** The corpus in force holds 126 channels, a
  sensor per operating condition, and a window of FD001 — one condition — holds 21 of them, so
  most of what MiniRocket convolved was channels nothing ever reported. The grid is now laid over
  the channels the fitted windows hold, with a mask only where they miss a step (`2032804`).
- Under both, MiniRocket scored 17.08 and 18.29 in the first comparison against 15.24 and 16.81
  now, and its first selection chose a grid of two steps per cycle, a choice made by the first
  error rather than by the method.
- **Closing a selection raised.** A selection's repeats score different units, so no verdict pairs
  them, and the close was stored before the verdict was read, so the campaign was closed with its
  message lost. A campaign now reads its verdict before its close is stored, and a selection
  closes announcing nothing (`9168856`).
