# The pretext over cells

What a pretext that hides cells of channel × time bin asks, and whether it can be answered
without a model, corpus by corpus; whether it trains at the tier a laptop runs; and how far the
small encoder it will be read on moves between seeds at few labels. The objective is
[ADR-0064](../adr/0064-a-pretext-over-cells-masked-before-the-layout.md); the layout
[ADR-0062](../adr/0062-tokens-as-cells-of-channel-and-time-bin.md).

## 2026-10-24 — declared before the run: the hidden cells answered without a model, the pilot, and the small shape's spread

**Questions.** (1) On each corpus the next pretraining may read — the intensive-care stays of
2012 and 2019, and the turbofans, the pumps and the satellite telemetry of the mixture of four —
how much of a hidden cell's presence, count and summaries does the strongest linear answer
without a model read, and how much is left for a model? Under the pilot's masks and under the
tail. (2) Does the objective train end to end on the stays at tier S — the loss falls, nothing
diverges, each term has a scale — and what does a step cost against one over raw readings?
(3) How far does the small encoder over six-bin cells move between seeds at 20, 50 and 200
stays on the fixed fifth the registered least gains were read on, and how far the closed-form
probe over the same encoder at its initialisation, the other half of the registered rule?

**Conditions.**

- *Commit*: the one that adds this section. *Machine*: M1 Pro; the audit and the campaign's
  trees on the CPU, the pilot and the campaign's networks on MPS, fp32, tier S for the pilot,
  tier M's recipe for the campaign.
- *Audit* (`scripts/cell_pretext_audit.py`): the corpora as two registered backbones read them —
  `backbone-clinical-m` (`1fa0ef3a-…`: 2012 at `717bc832…`, 2019 at `bf7de2fa…`) and
  `backbone-mixed4-m` (`2aaca0ba-…`: C-MAPSS, SKAB, ESA-AD and 2012) — 2,000 training and
  2,000 validation windows each, spread evenly; the masks of
  `experiments/backbone-cells-physionet2012-s.toml` (six bins; channel 0.15, block 0.6 over half
  the window, cell 0.1) and, on the stays, of `experiments/backbone-cells-tail-physionet2012-s.toml`
  (the same plus a tail drawn for half the windows, a quarter to a half of the window). Answers:
  `carry`, `interpolation`, `cross`, `schedule`, `stacked`, as the script's docstring states; the
  ridge penalty chosen among 1, 10, 100 and 1,000 on the half the stack is then fitted on; a
  map's answer for the presence clipped to [0.01, 0.99]; a value target read as R² against
  the channel's mean, the presence as the log loss removed against the training windows' share
  of hidden cells that held a reading, the share and the carried share both smoothed by one
  imagined cell of either kind.

      uv run --env-file .env.r2 scripts/cell_pretext_audit.py --backbone 1fa0ef3a-ba5a-44c5-877c-d7f3ddf756ee \
          --masks experiments/backbone-cells-physionet2012-s.toml
      uv run --env-file .env.r2 scripts/cell_pretext_audit.py --backbone 1fa0ef3a-ba5a-44c5-877c-d7f3ddf756ee \
          --masks experiments/backbone-cells-tail-physionet2012-s.toml --corpus physionet2012
      uv run --env-file .env.r2 scripts/cell_pretext_audit.py --backbone 2aaca0ba-ec68-4385-8531-ffed9aa29894 \
          --masks experiments/backbone-cells-physionet2012-s.toml

- *Pilot*: `experiments/backbone-cells-physionet2012-s.toml` and its tail variant over every stay
  of set A (`717bc832…`), four epochs; `experiments/backbone-cells-clinical-s.toml` over a tenth
  of 2012 and 2019, two epochs, for the cost alone; ordered, run on MPS and accepted by the
  `pretrain` command line; read in the run's log and MLflow, not downstream.
- *Spread*: `campaigns/seed-spread-small-physionet2012.toml` — the small shape on six bins from
  nothing at its setting in force, and the closed-form probe over the same shape and cells at
  its initialisation, at 20, 50 and 200 stays, ten seeds, scored on one tuning stay in five
  fixed by seed 101; the validation side is not read; defined under the registered worker
  settings with the mixture's backbone as the shape the arms borrow; the 4.8M network's spread
  on this fifth is the registered one (`verdict-statistics.md`, 2026-10-04) and is not read
  again; `scripts/seed_spread_report.py` over the accepted cells.

**Trial.** The audit command ran on 200 training and 200 validation windows of the 2012 stays
before this section was written, twice: the first run showed the carried presence read as a raw
share losing to the constant (−1.09) and the cross-channel ridge at a penalty of one over about
400 columns reading the values below zero (−0.42), so the share was smoothed and the penalty
chosen on the stacking half; the second run printed the tables (stacked: presence 0.45, mean
0.34; cross: presence 0.38; schedule: presence 0.15, values ≤ 0.13; carry: presence 0.09, mean
0.18; log count 0.95 under every map). The predictions below are drawn from the earlier audit
of the summaries over raw tokens and from the mechanism; the trial's numbers are named here so
that nothing reads as foreseen that was seen.

**Predictions** (intervals from the earlier readings, the lower edge where a gain is claimed).

| Reading | Prediction | Source |
| --- | --- | --- |
| Stays, every hidden cell, `stacked` on the mean | R² in [0.25, 0.50] | the stack over raw tokens read the hidden mean at 0.67 with the extremes at 0.60–0.63 (`pretext-variants.md`, 2026-10-21); a cell hidden whole leaves less to read than tokens hidden among visible ones |
| Stays, `schedule` on the values | within ±0.10 of zero | the schedule alone read the summaries at ≤ 0.08 there |
| Stays, presence | `cross` above `carry`; `stacked` in [0.30, 0.60] | which channels are measured together is the clinic's protocol, and the stays' schedule alone carried the task at 0.67 / 0.70 in area (`head-and-representation.md`, 2026-10-21) |
| 2019 stays | presence read higher than on 2012 by every map, log count ≥ 0.95 | the 2019 stream is hourly, so a channel's count is nearly constant and its presence a pattern |
| C-MAPSS, SKAB, ESA-AD | every cell holds a reading (held share ≥ 0.99); presence read at the edge by every map; `interpolation` on the mean ≥ 0.80 on SKAB and ESA-AD | the regime audit (`pretraining-corpus-sampling.md`): 50 to 1,024 readings of a channel in a window; the stack over tokens read their summaries at 0.90–0.96 |
| Pilot | the reconstruction's relative validation below 1 after the first epoch, the presence term's trivial cost above the model's by the second; no non-finite loss; an epoch over set A's 3,997 stays in under 10 min on MPS, the second layout under a third of a step | the small shape over cells trains a cell in 9–15 s on a G4 and 2012 alone over raw tokens took 31 min an epoch on MPS in the 4.8M shape (`task-transfer.md`, 2026-10-17) |
| Spread, small shape | one seed's SD in [0.02, 0.06] at 20, [0.02, 0.05] at 50, [0.01, 0.03] at 200 | the candidates' spread at 50 on the half 0.03–0.08, the small shape's at every stay 0.007–0.012 (`encoder-candidates.md`); one seed moved the 4.8M network by 0.041 at 200 on the fixed fifth (`verdict-statistics.md`, 2026-09-29) |
| Spread, probe at initialisation over the small shape's cells | one seed's SD at or below the small shape's at every budget, and at or below the registered least gains 0.055 / 0.060 / 0.050 | a head solved in closed form moves with the drawn labels alone, the encoder's draw adding little; the registered least gains were the larger of the network's and the probe's SD on this fifth (`docs/preregistration.md`, 2026-10-04) |

**What follows.** The audit is a diagnostic and decides the masks of the next pretraining, not
a verdict: where the stack reads the stays' presence at 0.90 or above the presence term is a
shortcut and is left out of the next experiment file; where it reads the stays' mean at 0.90 or
above the cells are interpolated and the block span grows before anything pretrains; otherwise
the pilot's masks stand. On the grids the pretext is expected to be interpolation, which is why
the mixture of four is the control arm of the next pretraining and not its body. The pilot is
read for mechanics and cost alone, as a dose; no reading of the representation is drawn from it.
The spreads are what the least gains of the pretraining's reading at these budgets are derived
from, by a register row written before that reading, with the rule as registered: the larger of
the network's and the probe's spread at each budget.

**Limitations.** The audit's answers are linear maps per channel on a few thousand cells; a
map that reads less than a model could is a floor on the margin, not a measurement of it. The
penalty is chosen on the half the stack is then fitted on, so the stack's weights lean slightly
towards that half; the scores do not, being read on the validation windows. The trial printed
the tables on a tenth of the windows the audit reads. The spread is read on a
fifth of the tuning side, as the registered least gains were, so it is not the spread of the
half the pretraining's reading will be scored on.

## 2026-10-24 — M1 Pro, CPU and MPS, fp32: the hidden cells answered without a model, the pilots, and the small shape's spread

**Question.** As declared above: how much of a hidden cell the strongest linear answer without
a model reads, corpus by corpus; whether the objective trains at tier S and what a step costs;
and the small shape's spread over seeds on the fixed fifth beside the probe's at initialisation.

**Conditions.** Declaration committed as `7d517a89`; the audit, the trees and the exports on the
M1 Pro's CPU, the pilots and the campaign's networks on MPS, fp32. Audit as declared, 7 to 130 s
a corpus (`data/report/l32b3a/audit-*/`). Spread: campaign `a2aa2234-…`, 60 cells, about 9.5 min
a budget; `campaign_pairs_report.py` then `seed_spread_report.py`. Pilots by the `pretrain`
command line (`data/report/l32b3a/pilot.sh`; its parsing of the order's and the run's output
failed twice and was fixed, each time resuming from the reference already placed, so nothing
was ordered or run twice). The host layout timed apart over twenty batches of 32 stays.

**The windows as cells.**

| corpus | windows | rows / window | cells / window | hidden cells | held |
| --- | --- | --- | --- | --- | --- |
| physionet2012 | 2,000 | 26.8 | 160.7 | 149,443 | 0.62 |
| physionet2019 | 2,000 | 21.3 | 127.9 | 119,849 | 0.52 |
| cmapss | 2,000 | 98.2 | 589.0 | 547,082 | 0.78 |
| skab | 768 | 8.0 | 48.0 | 16,920 | 0.99 |
| esa_ad | 2,000 | 7.0 | 41.7 | 38,629 | 1.00 |

**The stays of 2012, every hidden cell**, R² against the channel's mean (values) and log loss
removed against the constant share (presence); `cells` is how many the target is defined on.

| target | cells | carry | interpolation | cross | schedule | stacked |
| --- | --- | --- | --- | --- | --- | --- |
| presence | 149,443 | 0.082 | 0.082 | 0.537 | 0.238 | **0.593** |
| log_count | 93,257 | 0.881 | 0.881 | 0.955 | 0.956 | **0.959** |
| mean | 93,257 | 0.026 | 0.064 | 0.053 | 0.005 | **0.119** |
| minimum | 93,257 | 0.046 | 0.129 | 0.239 | 0.130 | **0.405** |
| maximum | 93,257 | 0.003 | 0.006 | 0.012 | 0.007 | 0.017 |

**The stays of 2019, every hidden cell**, `stacked`: presence 0.653 (`cross` 0.623, `schedule`
0.351, `carry` 0.125), log count 0.976, mean 0.376 (`interpolation` 0.250), minimum 0.385,
maximum 0.406.

**The grids**: the presence is a near certainty (held 0.78 to 1.00), so the constant share
already sits at the clip's edge and every other answer loses to it (SKAB −2.5, ESA-AD −10⁴ and
worse); the values are read as follows.

| corpus | target | carry | interpolation | cross | stacked |
| --- | --- | --- | --- | --- | --- |
| cmapss | mean | 0.005 | −0.001 | 0.348 | **0.552** |
| skab | mean | 0.367 | **0.530** | 0.189 | 0.224 |
| esa_ad | mean | 0.618 | **0.804** | 0.047 | 0.442 |
| esa_ad | minimum / maximum | 0.552 / 0.571 | 0.723 / 0.711 | 0.110 / 0.093 | 0.532 / −1.836 |

**The stays of 2012 under the tail's masks, by kind of mask** (`horizon` = the window's last
quarter to half, every channel at once).

| kind | cells | presence: carry | cross | schedule | stacked | mean: carry | interpolation | stacked |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| channel | 57,306 | −0.020 | 0.435 | 0.194 | 0.460 | 0.000 | 0.000 | 0.139 |
| block | 64,633 | 0.100 | 0.495 | 0.249 | 0.558 | 0.019 | 0.037 | 0.074 |
| token | 14,322 | 0.069 | 0.532 | 0.251 | 0.597 | 0.190 | 0.384 | 0.501 |
| horizon | 49,834 | 0.087 | 0.188 | 0.247 | **0.224** | 0.131 | 0.131 | **0.387** |
| all | 186,095 | 0.057 | 0.395 | 0.232 | 0.440 | 0.031 | 0.050 | 0.118 |

**The spread over seeds on the fixed fifth** (ten seeds, each drawing its labels; SD over the
ten areas, and the standard error of one area over the fifth's 800 stays).

| candidate | stays | mean area | SD over seeds | SE over stays |
| --- | --- | --- | --- | --- |
| small shape on six-bin cells, from nothing | 20 | 0.602 | **0.035** | 0.027 |
| probe at initialisation, same shape and cells | 20 | 0.549 | 0.032 | 0.028 |
| small shape, from nothing | 50 | 0.629 | **0.054** | 0.026 |
| probe at initialisation | 50 | 0.578 | 0.053 | 0.027 |
| small shape, from nothing | 200 | 0.706 | 0.019 | 0.024 |
| probe at initialisation | 200 | 0.617 | **0.023** | 0.027 |

By the registered reading (the larger of the two, rounded up to 0.005) the least gains over this
encoder would be **0.035 / 0.055 / 0.025** at 20 / 50 / 200 stays, against 0.055 / 0.060 / 0.050
registered for the 4.8M encoder; a register row before the pretraining's reading states which
rule applies.

**The pilots** (tier S, small shape, six bins, Huber 1.0, presence 1.0, count 0.5; validation on
set A's held-out stays, each term as a share of its trivial predictor's cost).

| pilot | epochs | s / epoch | s / step | reconstruction | presence | count | diverged |
| --- | --- | --- | --- | --- | --- | --- | --- |
| stays, the mixture's masks | 4 | 20–21 | 0.11–0.12 | 0.93 → 0.88 | 0.87 → 0.60 | 0.44 → 0.42 | no |
| stays, with the tail | 4 | 20–21 | 0.11 | 0.93 → 0.88 | 0.85 → 0.74 | 0.39 → 0.36 | no |
| 2012 + 2019, a tenth each | 2 | 40–41 | 0.10–0.11 | 2012: 1.04 → 1.00; 2019: 0.97 → 0.96 | 0.95 → 0.90; 0.73 → 0.71 | 0.87 → 0.65; 0.40 → 0.39 | no |

The two host layouts of a batch of 32 stays cost 5.2 ms (`collate`) + 5.8 ms (`scored`) + 0.2 ms
(masks) = 11 ms against a step of 110–120 ms on MPS, a tenth; the whole pilot of four epochs
over 3,997 stays ran in 93 s. At a tenth of each clinical corpus the 2012 stays have 25 steps an
epoch and reach their trivial cost only at the second epoch, the 2019 stays at the first.

**Conclusions.**

1. **On the stays the pretext leaves most of the values and about two fifths of the presence to
   a model.** The stack reads the hidden mean at 0.12 (2012) and 0.38 (2019), below the 0.25–0.50
   predicted for 2012; the presence at 0.59 and 0.65, inside the predicted [0.30, 0.60] on 2012;
   the count at 0.96–0.98 from the schedule alone. By the declared rule the pilot's masks and the
   presence term stand, and the count term is close to a shortcut.
2. **What reads the presence is the other channels in the same bin** (`cross` 0.54 against the
   schedule's 0.24 and the carry's 0.08): the clinic's protocol of co-measurement, not the
   channel's own rhythm. Under the tail the maps read the presence at 0.22 and the mean at 0.39:
   forecasting which channels are measured next is where a model has the most room.
3. **On the grids the pretext is interpolation**, as predicted: the presence is constant and the
   mean is carried or interpolated at 0.53 (SKAB) to 0.80 (ESA-AD); C-MAPSS per condition reads
   its cells across channels (0.55), an artefact of one sensor standing as six channels. The
   mixture of four is the control arm of the next pretraining, not its body.
4. **The objective trains at tier S**: every term falls under its trivial cost from the first
   epoch, no loss diverges, and the layout on the host is a tenth of an MPS step; on a GPU, where
   the small shape's step is several times shorter, the layout will be a larger share and is to be
   measured there before the next order.
5. **The small shape's spread over seeds is the labels' draw at 50 stays** (0.054, as the 4.8M
   network's), smaller at 20 (0.035) and 200 (0.019); the probe at initialisation moves as much.
   Two predictions missed by a little: 0.054 against [0.02, 0.05] at 50, and the probe above the
   network at 200 (0.023 against 0.019).

**Limitations.** R² under the squared error is governed by the heaviest values: on 2012 the
sum of squares of the hidden maxima is thirty times the minima's (2.8·10⁶ against 9.4·10⁴),
so the maximum's 0.017 says that a few excursions dominate it, not that it is unread; the
training loss bounds them (Huber), the audit does not. A stack fitted on another half can read
below its parts where the cells are few or the target degenerate (ESA-AD's maximum, SKAB's
channel kind). The pilots read mechanics alone. The spread is on a fifth of the tuning side, as
the registered least gains were.

## 2026-11-15 — declared before the run: finer cells and an interval hidden in every channel, answered without a model and piloted

**Questions.** (1) On each corpus of the composition the stage pretrains over — the stays of
2012 and 2019, the eICU demo and the raw satellite telemetry, as published under the robust
score — how much of a hidden cell does the strongest linear answer without a model read at 6,
12, 24, 32 and 48 bins under the masks in force, and at 6, 32 and 48 bins where the block of
each channel gives way to one interval hidden in every channel at once
([ADR-0081](../adr/0081-an-interval-of-the-window-hidden-in-every-channel.md))? (2) Does the
objective train end to end on the stays at each of those eight settings at tier S — the loss
falls, nothing diverges, each term has a scale against its trivial cost, the hidden cells of
each kind are answered — and what does an epoch cost? Nothing about the representation is read.

**Conditions.**

- *Commit*: the one that adds this section; the interval's draw and the eight experiment files
  are in the commits before it. *Machine*: M1 Pro; the audit on the CPU, the pilots on MPS, fp32, tier S.
- *Audit* (`scripts/cell_pretext_audit.py --manifest`): the publications the last screen read
  (`composition-screen.md`, 2026-11-12) — 2012 `d2d86d29…`, 2019 `79364c26…`, the eICU demo
  `5f24d626…`, the raw telemetry `490c1547…` — 2,000 training and 2,000 validation windows
  each, spread evenly, the demo all of its windows; the bins and masks of
  `experiments/backbone-cells-clinical-bins{6,12,24,32,48}-s.toml` (channel 0.15, block 0.6 over
  half the window, cell 0.1) and of `…-bins{6,32,48}-interval-s.toml` (the block's rate and span
  given to the interval); the answers, penalties and scores as on 2026-10-24.

      uv run --env-file .env.r2 scripts/cell_pretext_audit.py --manifest <key> <checksum> \
          --corpus <corpus> --masks experiments/<file>.toml --out data/report/l52/audit/<corpus>-<file>

- *Pilots*: the eight files over every 2012 stay of set A and a tenth of the 2019 stays
  (3,997 and 4,658 training windows), four epochs, batches of eight summed four to a step;
  ordered, run on MPS and accepted by the `pretrain` command line, one after another; read in
  the run's log and MLflow, not downstream.

**Trial, and what was seen before this section.** The cost of a step was timed on MPS over the
2012 stays and the telemetry: a stay is 164 / 325 / 647 / 862 / 1,323 cells at 6 / 12 / 24 / 32 /
48 bins, a telemetry window 249 to 1,904; a batch of 32 stays at 48 bins does not fit the
machine's memory, a batch of eight takes 0.59 s a step. The audit's command ran on 200 training
and 200 validation windows, a tenth of what it reads: the 2012 stays at 6 and 48 bins under both
sets of masks, the other three corpora at 48 bins under the masks in force. It printed, for the
stack: 2012 at 6 bins presence 0.49 and mean 0.46, at 48 bins 0.61 and 0.41, the mean, minimum
and maximum within 0.01 of one another there; 2019 at 48 bins presence 0.73, mean 0.27, the
count read exactly by every answer; the demo 0.68 and 0.39; the telemetry 0.71 and 0.62 with
0.89 of its cells holding a reading. Under the interval the stack read the 2012 presence of the
interval's cells at 0.23 (6 bins) and 0.38 (48 bins), where `cross` read 0.16 and 0.31. Eight
steps of the 48-bin interval arm ran on MPS without error. The predictions below name what the
trial showed, so that nothing reads as foreseen that was seen; the pilots were not run.

**Predictions.**

| Reading | Prediction | Source |
| --- | --- | --- |
| 2012 at 6 bins, masks in force, stack | presence in [0.45, 0.62], mean in [0.30, 0.55] | presence 0.59 on the publication before the robust score (2026-10-24), 0.49 in the trial; the mean 0.12 then, 0.46 in the trial, the robust score having bounded the excursions the squared error was read by |
| Stays, 12 to 32 bins, stack | presence and mean between the readings at 6 and 48 bins, within 0.05 | nothing but the bin changes; the trial's two ends |
| Stays at 48 bins, stack | presence in [0.55, 0.78], below 0.90 on both corpora; mean in [0.20, 0.50]; mean, minimum and maximum within 0.03 of one another | the trial; a cell of one hour holds one reading |
| 2019, count at 24 bins and finer | read at 0.99 or above by `carry` alone | the 2019 stream is hourly; the trial at 48 bins |
| Telemetry at every count | held share above 0.80; mean read by `interpolation` at 0.45 or above | the trial at 48 bins: held 0.89, 0.53 |
| Interval's cells, stays, presence | `cross` at least 0.15 below its reading on the block's cells of the same bins; the stack below 0.45 | the trial: 0.16 against 0.44 at 6 bins, 0.31 against 0.58 at 48; the tail's cells read 0.22 on 2026-10-24 |
| Interval's cells, stays, mean | `interpolation` within 0.15 of its reading on the block's cells | either leaves the channel's own readings on both sides of a span of half the window |
| Pilots, every arm | no non-finite loss; on 2012 the reconstruction's share of its trivial cost below 1 by the fourth epoch, the presence's below 0.90 | six bins on 2012: 0.93 → 0.88 and 0.87 → 0.60 over four epochs of 125 steps (2026-10-24); here about 270 steps an epoch over both corpora |
| Pilots, finer cells | the presence's share at the fourth epoch no higher at 48 bins than at 6 | more of the presence is answerable without a model at 48 bins (0.61 against 0.49) |
| Pilots, the interval | the presence's share on the interval's cells above the block arm's on the block's cells, and below 1 | the answers without a model read half as much there |
| Pilots, count on 2019 at 48 bins | its trivial cost below a hundredth of its cost at 6 bins | one reading a cell, so the count is a constant |
| Cost | an epoch with its validation within a factor of two of 2 / 3 / 6 / 8 / 15 min at 6 / 12 / 24 / 32 / 48 bins | 8 to 74 ms a 2012 stay on MPS in batches of eight; validation scores 15,232 windows an epoch |

**What follows.** The audit and the pilots decide the design of the screen that follows, by a
declaration of its own; they close nothing. Where the stack reads the stays' presence at 0.90 or
above at a count, the presence is a shortcut there and that count is not screened under the
masks in force; where it reads the mean at 0.90 or above, the cells are interpolated and the
count is not screened. A count whose pilot diverges, or whose reconstruction stays at or above
its trivial cost on both corpora after four epochs, is not screened until that is explained. A
term whose trivial cost is near zero is judged by its own loss beside the reconstruction's, not
by its share; where the count is a constant on a corpus its weight is a question for the
screen's files. No count is dropped for its cost. The interval enters the screen as a family of
its own if its pilots train. The pilots are read for mechanics and cost, as a dose; no reading
of the representation is drawn from them.

**Limitations.** The audit's answers are linear maps per channel on a few thousand cells: a
floor on what a model has left, not a measurement of it. The trial printed the tables on a
tenth of the windows, so the predictions for those cells are informed. R² under the squared
error follows the heaviest values. The pilots read a tenth of the 2019 stays at tier S, on
MPS; the cost on the accelerator the screen will run on is measured there before its order.

## 2026-11-16 — M1 Pro, CPU and MPS, fp32: finer cells leave a model as much as six bins do, the interval takes the same-bin answer away, and the 2019 stays were laid out by their rounding

**Question.** As declared on 2026-11-15: what the answers without a model read of a hidden cell
at each count of bins and under the interval, and whether the eight pilots train. Beside,
found while reading them: why the 2019 stays read apart from the other corpora.

**Conditions.** Declaration `41e7c5d5`, the code of every run. The audit on the CPU, 32 runs of
14 to 828 s (`data/report/l52/audit/`); the pilots on MPS, each ordered, run and accepted
(`data/report/l52/pilot/`); each pilot's decoder scored on the audit's hidden cells by
`scripts/cell_pretext_audit.py --backbone <id> --model trained`, at 48 bins on the CPU, where
its batch of 32 fits (`data/report/l52/kinds/`). Validation side, tier S.

**The stack on every hidden cell, masks in force** (presence: log loss removed; mean: R²).

| bins | 2012 presence | 2012 mean | 2019 presence | 2019 mean | demo presence | demo mean | telemetry mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 | 0.595 | 0.482 | 0.652 | 0.367 | 0.628 | 0.445 | 0.508 |
| 12 | 0.646 | 0.480 | 0.682 | 0.384 | 0.663 | 0.470 | 0.517 |
| 24 | 0.647 | 0.455 | 0.738 | 0.395 | 0.671 | 0.452 | 0.603 |
| 32 | 0.647 | 0.446 | 0.739 | 0.395 | 0.685 | 0.443 | 0.634 |
| 48 | 0.656 | 0.436 | 0.769 | 0.391 | 0.697 | 0.430 | 0.633 |

The cells holding a reading fall from 0.62 to 0.28 of the 2012 stays' cells and from 0.52 to
0.16 of the 2019 stays'; the telemetry's stay at 0.89 or above, its presence a constant at 6
and 12 bins. At 48 bins the stack reads the 2012 mean, minimum and maximum at 0.436, 0.428 and
0.429.

**Presence on the cells of the block and of the interval**, the interval in the block's place.

| corpus | bins | block: `cross` | block: stack | interval: `cross` | interval: `schedule` | interval: stack |
| --- | --- | --- | --- | --- | --- | --- |
| 2012 | 6 | 0.532 | 0.593 | 0.190 | 0.258 | 0.258 |
| 2012 | 32 | 0.598 | 0.651 | 0.312 | 0.381 | 0.385 |
| 2012 | 48 | 0.607 | 0.660 | 0.344 | 0.415 | 0.417 |
| 2019 | 6 | 0.607 | 0.644 | 0.338 | 0.357 | 0.351 |
| 2019 | 32 | 0.717 | 0.738 | 0.545 | 0.561 | 0.566 |
| 2019 | 48 | 0.754 | 0.766 | 0.610 | 0.617 | 0.622 |

The mean of an interval's cells is interpolated as a block's is (2012: 0.47 against 0.47 at 6
bins, 0.33 against 0.34 at 48). On the telemetry the mean of either is interpolated at 0.75 to
0.96.

**The pilots**, four epochs, each term on the validation side as a share of its trivial cost
after the first and the fourth epoch; an epoch with its validation.

| arm | s / epoch | 2012 reconstruction | 2012 presence | 2019 reconstruction | 2019 presence |
| --- | --- | --- | --- | --- | --- |
| 6 bins | 70 | 0.94 → 0.84 | 0.82 → 0.36 | 0.97 → 0.85 | 0.72 → 0.32 |
| 6, interval | 70 | 0.94 → 0.80 | 0.85 → 0.68 | 0.97 → 0.81 | 0.73 → 0.66 |
| 12 | 118 | 0.94 → 0.80 | 0.86 → 0.55 | 0.97 → 0.78 | 0.73 → 0.49 |
| 24 | 309 | 0.97 → 0.82 | 0.88 → 0.75 | 0.98 → 0.78 | 0.60 → 0.50 |
| 32 | 502 | 0.98 → 0.77 | 0.86 → 0.78 | 0.98 → 0.76 | 0.57 → 0.52 |
| 32, interval | 503 | 0.97 → 0.78 | 0.88 → 0.80 | 0.98 → 0.75 | 0.56 → 0.53 |
| 48 | 1,138 | 0.99 → 0.83 | 0.88 → 0.77 | 1.00 → 0.82 | 0.58 → 0.43 |
| 48, interval | 1,118 | 0.99 → 0.85 | 0.89 → 0.79 | 1.00 → 0.82 | 0.56 → 0.43 |

No loss was non-finite. The count's trivial cost on the 2019 stays is 0.016 at 6 bins and
below 2·10⁻⁶ at 32 and 48, the model's 1.2·10⁻⁴ to 1.6·10⁻⁴ there.

**The pilots' decoders against the stack, 2012, presence** on the cells of the arm's own span.

| arm | cells of | stack | model |
| --- | --- | --- | --- |
| 6 bins | block | 0.593 | 0.666 |
| 32 | block | 0.651 | 0.419 |
| 48 | block | 0.660 | 0.449 |
| 6, interval | interval | 0.258 | 0.304 |
| 32, interval | interval | 0.385 | 0.394 |
| 48, interval | interval | 0.417 | 0.421 |

The mean of those cells is read below the stack by every pilot: 0.17 to 0.29 against 0.47 to
0.60.

**The 2019 stays on the bins' edges** (`data/report/l52/bin_aliasing.py`: every 400th training
window of each publication, the layout's own assignment of a reading to its bin). The 2019
windows are 24 hours of hourly rows, 24 distinct times; the layout took a plain floor of a time
stored in single precision.

| corpus | distinct times | readings within 0.001 bin of an edge, at 6 / 12 / 24 / 32 / 48 bins | bins holding under a tenth of an even share |
| --- | --- | --- | --- |
| 2019 stays | 24 | 0.24 / 0.49 / 1.00 / 0.33 / 1.00 | 0 / 0 / 7 / 8 / 24 |
| 2012 stays | 2,881 | 0.004 at every count | none |
| eICU demo | 2,880 | 0.005 to 0.022 | none |
| telemetry | 31,509 | 0.000 to 0.003 | none |

A bin of the 2019 stays holds 0.72 to 1.24 of an even share at 6 bins and 0.47 to 1.51 at 12.

**Predictions.** Four missed: the 2019 count read by `carry` at 24 bins, 0.951 against 0.99 or
above; the stack on the interval's cells of the 2019 stays, 0.566 and 0.622 against below 0.45;
`cross` on those cells at 48 bins, 0.144 below the block's against 0.15; the presence's share
at 48 bins against 6 on 2012, 0.77 against 0.36 where no higher was predicted. The others held,
the epochs within a factor of two of their predicted cost.

**Conclusions.**

1. **No count of bins makes the pretext a shortcut on the stays.** The stack reads the presence
   at 0.60 to 0.77 and the mean at 0.37 to 0.48, far from 0.90 at every count; a finer cell
   holds one reading, so its five values are one target.
2. **The interval removes the answer from the same bin.** On the 2012 stays `cross` falls from
   0.53–0.61 to 0.19–0.34 and the stack to 0.26–0.42; what is left is the channel's own rhythm.
3. **The pilots train, and finer cells learn the presence more slowly in four epochs**: the
   2012 presence stays at 0.75 to 0.80 of its trivial cost from 24 bins up, against 0.36 at 6,
   and the decoder reads it below the stack there. Under the interval the decoder reaches the
   stack on the interval's cells at every count. A budget of epochs set at 6 bins does not
   carry over.
4. **The 2019 stays were laid out by their rounding.** Every reading sits on an hourly edge, and
   the floor put those stored below it one bin early: uneven bins at 6 and 12, 7 empty bins at
   24. Above 24 bins the cells are finer than the corpus's clock, so 8 of 32 and 24 of 48 can
   hold nothing: the 2019 columns of this section at 24, 32 and 48 bins read that comb, which
   is where three of the four missed predictions lie. The layout takes the grids' tolerant rule
   ([ADR-0082](../adr/0082-a-reading-on-the-edge-of-a-bin-belongs-to-the-bin-it-opens.md)).
5. **The screen that follows reads 24 bins, not 32 or 48**: the finest count at which no corpus
   of the composition is cut below its clock, an hour on the 2019 stays and two on the others.

**Limitations.** Tier S, one seed, four epochs: mechanics and cost alone, and at finer cells the
decoder is below the answers without a model, so nothing about the representation could be
read here had it been asked. The 2019 readings at 24 bins and finer are of the layout before
ADR-0082 and are not repeated under it in this section; the 2012, demo and telemetry readings
move by at most 2 % of readings. Backbones over cells pretrained before the rule read the
2019 stays one bin early at the edges, alike in pretraining and evaluation.

## 2026-11-16 — declared before the run: the audit and the pilots at 6, 12 and 24 bins under the rule for the bins' edges

**Question.** Under [ADR-0082](../adr/0082-a-reading-on-the-edge-of-a-bin-belongs-to-the-bin-it-opens.md),
what do the readings of 2026-11-15's declaration give at the counts the screen can use — 6, 12
and 24 bins under the masks in force, 6 and 24 under the interval — and does the arm not yet
run, 24 bins under the interval, train?

**Conditions.** The commit that adds this section and
`experiments/backbone-cells-clinical-bins24-interval-s.toml`; otherwise as declared on
2026-11-15: the four publications, 2,000 and 2,000 windows, the same answers; the pilots at 6
and 24 bins, each under both sets of masks, four epochs on MPS; each pilot's decoder scored on
the audit's cells. The edges are counted again by `data/report/l52/bin_aliasing.py` with the
layout's rule.

**What was seen before this section.** Everything in the section above. Under the rule, the
unit tests of the layout; the audit's command ran on 200 and 200 windows of the 2012 stays under
the new file, to files left unopened.

**Predictions.**

| Reading | Prediction | Source |
| --- | --- | --- |
| 2019 stays, bins against an even share | no bin under a tenth at 6, 12 or 24; 0.9 to 1.1 at 6 and 12 unless the hours are unevenly charted | every reading sits on an hourly edge and now falls in the hour it opens |
| 2019 at 24 bins | cells holding a reading in [0.28, 0.40]; count read at 0.99 or above by `carry`; presence: stack in [0.60, 0.74], `schedule` below 0.50 | held 0.25 with 7 of 24 bins empty; the stack 0.65 at 6 bins and 0.74 over the comb, `schedule` 0.55 over it |
| 2019 at 6 and 12 bins, stack | within 0.03 of 0.652 / 0.682 (presence) and 0.367 / 0.384 (mean) | the same rows, evenly binned |
| 2012, demo, telemetry, every count | within 0.01 of the section above | under 1 % of their readings sit on an edge |
| Interval's cells, 2012 at 24 bins | `cross` at least 0.15 below the block's; the stack below 0.45 | 0.31 against 0.60 and 0.385 at 32 bins |
| Pilots at 6 bins | every share within 0.05 of the section above | the 2012 stays unchanged, the 2019 stays evenly binned |
| Pilots at 24 bins | no non-finite loss; 2012 shares within 0.05 of 0.82 and 0.75; the 2019 presence's share above 0.50; an epoch within a factor of 1.5 of 309 s | the pilot over the comb read 0.50 on 2019, part of it the comb's pattern |
| 24 bins, interval | trains; the decoder within 0.05 of the stack on the interval's cells of 2012 | 0.39 against 0.385 at 32 bins |

**What follows.** As declared on 2026-11-15, read at 24 bins: a stack at 0.90 or above on the
stays' presence or mean, a divergence, or a reconstruction at or above its trivial cost on both
corpora keeps 24 bins out of the screen until explained.

**Limitations.** As on 2026-11-15. The predictions for the 2019 stays under the rule rest on
readings of the layout before it.

## 2026-11-16 — M1 Pro, CPU and MPS, fp32: under the rule for the edges the 2019 stays fill every bin, and 24 bins with the interval trains

**Question.** As declared above: the audit and the pilots at the counts the screen can use,
under [ADR-0082](../adr/0082-a-reading-on-the-edge-of-a-bin-belongs-to-the-bin-it-opens.md).

**Conditions.** Declaration `25bc09b8`, the code of every run; 20 audits, four pilots ordered as
run `pilot-s-edge`, their decoders scored on the audit's cells
(`data/report/l52/{audit,pilot,kinds}-edge/`). Validation side, tier S.

**The 2019 stays' bins against an even share**: 0.97 to 1.02 at 6 bins, 0.96 to 1.08 at 12,
0.78 to 1.17 at 24, none empty (before: 0.72 to 1.24, 0.47 to 1.51, 7 of 24 empty).

**The stack on every hidden cell.**

| corpus | bins | held | presence | `schedule` on presence | mean | count by `carry` |
| --- | --- | --- | --- | --- | --- | --- |
| 2012 | 6 / 12 / 24 | 0.62 / 0.49 / 0.36 | 0.591 / 0.646 / 0.647 | 0.238 / 0.307 / 0.342 | 0.482 / 0.480 / 0.454 | 0.881 / 0.903 / 0.923 |
| 2019 | 6 / 12 / 24 | 0.52 / 0.40 / 0.32 | 0.653 / 0.686 / 0.710 | 0.351 / 0.434 / 0.490 | 0.372 / 0.384 / 0.401 | 0.927 / 0.968 / 1.000 |
| demo | 6 / 12 / 24 | 0.51 / 0.37 / 0.26 | 0.628 / 0.663 / 0.671 | 0.258 / 0.327 / 0.363 | 0.445 / 0.470 / 0.452 | 0.870 / 0.897 / 0.918 |

The telemetry reads as before the rule at every count.

**Presence on the cells of the block and of the interval.**

| corpus | bins | block: `cross` | block: stack | interval: `cross` | interval: stack |
| --- | --- | --- | --- | --- | --- |
| 2012 | 6 | 0.522 | 0.589 | 0.190 | 0.258 |
| 2012 | 24 | 0.596 | 0.652 | 0.297 | 0.373 |
| 2019 | 6 | 0.607 | 0.645 | 0.339 | 0.351 |
| 2019 | 24 | 0.674 | 0.712 | 0.476 | 0.510 |

**The pilots**, each term as a share of its trivial cost after the first and fourth epoch, and
the decoder against the stack on the cells of the arm's own span (presence).

| arm | s / epoch | 2012 reconstruction | 2012 presence | 2019 reconstruction | 2019 presence | 2012 stack / decoder | 2019 stack / decoder |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 6 bins | 71 | 0.94 → 0.84 | 0.83 → 0.36 | 0.97 → 0.85 | 0.72 → 0.32 | 0.589 / 0.671 | 0.645 / 0.720 |
| 6, interval | 70 | 0.94 → 0.80 | 0.85 → 0.68 | 0.97 → 0.81 | 0.73 → 0.66 | 0.258 / 0.306 | 0.351 / 0.359 |
| 24 | 317 | 0.96 → 0.77 | 0.86 → 0.75 | 0.97 → 0.73 | 0.73 → 0.68 | 0.652 / 0.431 | 0.712 / 0.524 |
| 24, interval | 311 | 0.96 → 0.78 | 0.88 → 0.79 | 0.97 → 0.74 | 0.74 → 0.70 | 0.373 / 0.392 | 0.510 / 0.518 |

No loss was non-finite. The count's trivial cost on the 2019 stays at 24 bins is below 2·10⁻⁶.

**Predictions.** One missed: the 2012 reconstruction's share at 24 bins, 0.77 against within
0.05 of 0.82. The others held: no empty bin; at 24 bins on the 2019 stays 0.32 of cells held,
the count read exactly, the stack at 0.710 and `schedule` at 0.490; 6 and 12 bins and the
other corpora within their bounds; the interval's cells on 2012 at 24 bins with `cross` 0.30
below the block's and the stack at 0.373; the pilots at 6 bins within 0.05; the interval's
decoder at 24 bins 0.02 above the stack.

**Conclusions.**

1. **The rule removes the comb.** The 2019 stays fill every bin at 6, 12 and 24, and at 24 bins
   each cell that holds anything holds one row, so the count is a constant there.
2. **24 bins pass the audit**: the stack reads the stays' presence at 0.65 to 0.71 and the mean
   at 0.40 to 0.45, and what the channel's own rhythm reads of the 2019 presence falls from
   0.55 over the comb to 0.49.
3. **All four arms train**, the 2019 presence at 24 bins now learnt from 0.73 to 0.68 of its
   trivial cost where the comb had given 0.60 to 0.50 by its pattern. As before, at 24 bins the
   decoder is below the stack on the block's cells and at it on the interval's.
4. **The screen reads 6 and 24 bins under the block and under the interval**, by a declaration
   of its own, with the epochs of each new recipe read off a run of its own.

**Limitations.** As on 2026-11-15: mechanics and cost at tier S, one seed, four epochs.
