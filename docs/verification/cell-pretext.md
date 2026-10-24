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
