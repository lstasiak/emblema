# Candidates for an encoder trained from nothing on the intensive-care stays

## 2026-10-23 — declared before the run: cells, a learnt time and several queries at 50 and 200 stays

**Question.** Below 500 stays the network trained from nothing loses 0.045 to 0.059 in area to
the same network handed the trees' table (D2, `gap-to-the-trees.md`, 2026-10-23): with few labels
the encoder fails to build from single readings what ten statistics per channel hold. Does an
encoder that reads the window as cells of channel and time bin, with what each bin held and
whether anything was measured, close that gap at 50 and 200 stays? Do a time encoding that learns
its frequencies and a pooling under several learnt queries add to it? And is the absence of a
whole channel, which the trees see as a count of zero, part of the gap?

**Why these candidates.** The token states carry a channel's summaries and the mean over the
window loses them (`head-and-representation.md`, 2026-10-21); networks published on irregular
clinical series read cells of one variable over one span (DuETT, t-PatchGNN); a patch model from
nothing on a grid led this network by 0.043 [0.018; 0.069] at 200 stays (`intensive-care-curve.md`,
2026-10-01); a head over one state per channel reads 0.037 above the probe's mean and is bound to
the channel layout, which these queries are not (ADR-0063); a learnt time encoding is the ablation
ADR-0017 left open (mTAN: +0.013). ADR-0062 holds the design.

**Conditions.**

| | |
|---|---|
| Code | the commit of this declaration; campaigns `campaigns/candidates-{50,200}-physionet2012.toml` |
| Task | in-hospital death, `a6a9c653…`; half of the tuning side by seed 101 (1,997 stays learnt from, 2,000 scored); seeds 1 to 10; tier M |
| Worker | `data/report/t42c/worker-env.sh` under the mixture of four at its first seed (`0d81c01e…`), whose 4.8M shape the network from nothing borrows |
| Recipe | registered per budget: `from_scratch` at 50, `@learning_rate=0.003` at 200; no stop at either |
| Candidates | the control on its tokens; cells over the window's channels at 6, 12 and 24 bins; the same over every channel the corpus names (`cells=corpus`); 12 bins with the learnt time; the learnt time on raw tokens; 12 bins under four and eight queries; four queries on raw tokens; 12 bins with the outcomes weighed alike (`class_weight=ratio`) — thirteen per budget |
| Machines | network cells on one platform GPU (Colab G4 under CUDA MPS, or Kaggle T4 ×2 with one order per card), fp32; the trial on the M1 Pro (MPS) |
| Reused cells | the trees, MiniRocket and the network over the trees' table at 50 and 200 from the gap's campaigns (`f7611c9f…`, `9458a4d9…`, `6a388747…`; M1 Pro CPU and Colab G4), on the same stays and seeds |

**Readings** (each the paired difference in area, second over first, over the same stays and
seeds):

| Name | Pair | Budgets |
|---|---|---|
| C | control on its tokens → cells at 12 bins over the window | 50, 200 |
| C′ | control → the best candidate of the campaign | 50, 200 |
| A | cells at 12 bins over the window → the same over the corpus's channels | 50, 200 |
| T | cells at 12 bins → the same with the learnt time; control → the learnt time on raw tokens | 50, 200 |
| Q | cells at 12 bins → the same under four queries; four → eight; control → four queries on raw tokens | 50, 200 |
| W | cells at 12 bins → the same with the outcomes weighed alike | 50, 200 |
| Bins | cells at 6 → 12 → 24 bins, over the window and over the corpus | 50, 200 |
| Bar | the best candidate → the trees, → the network over the table, → MiniRocket | 50, 200 |

**Rule.** Paired bootstrap over stays in the two outcomes, 10,000 resamples, 95 %; a difference is
carried when its interval excludes zero and its mean exceeds twice its standard error over seeds.
No term for the spread between pretrainings: nothing here is pretrained. C at 50 and 200 is the
one family under Holm (two): the cells at 12 bins are the candidate named before the run, not the
best of thirteen. Every other reading is descriptive, C′ among them, since the best of thirteen
chosen on the stays it is scored on reads high; C′ is the estimate of size for the next step, not a
test. The candidate the campaign's selection chooses at 200 stays, by its rule of one standard
error, goes on to 500 and every stay of the half and then to set B under the registered parity
(`d18618a1`); the choice at 50 stays is read beside it. The detectable difference T on the half,
from the intervals of the gap's pairs at these budgets, is about 0.021 at 50 and 0.018 at 200.

**Predictions** (the edge of an earlier interval named as its source):

1. C is carried at both budgets and lies at no less than +0.018 at 200 and +0.027 at 50, the
   lower edges of the patch model's lead on the raw network at those budgets
   (`intensive-care-curve.md`, 2026-10-01 and 2026-09-30).
2. The cells at 12 bins close D2 at 200 stays: C lies at no less than +0.041, the lower edge of
   D2 there, and the best candidate's interval against the network over the table (0.706) holds
   zero. At 50 stays C lies at no less than +0.027, the lower edge of D2, and the best candidate
   reads within T of the table's 0.674.
3. A lies at or above zero at both budgets and under +0.021, the lead of a head over the
   schedule of measurements on the raw network at 200 (`head-and-representation.md`, 2026-10-21):
   the absence of a channel is part of the gap, and a small part.
4. T lies within ±T at both budgets, on cells and on raw tokens (mTAN's +0.013 is below the
   reader).
5. Q, four queries over the mean on cells, lies at no less than +0.010 at 200, the lower edge
   of the per-channel probe's lift over the probe (`head-and-representation.md`, 2026-10-21);
   eight against four lies within ±T.
6. W lies within ±T: the parts of the published regime beside the stop, the class weight among
   them, moved the control's shape by 0.011 at every stay (`intensive-care-curve.md`,
   2026-10-02). Twelve bins read within T of six and of twenty-four.

**Levels of the outcome.** *Closes D2*: the best candidate's interval against the network over
the table holds zero, or lies above it, at a budget. *Real, small*: C carried below the edge of D2.
*Noise*: C not carried. *Harm*: C carried below zero.

**What follows from each outcome.**

- *Closes D2 at 200 or at 50*: the chosen candidate goes to 500 and every stay of the half under
  the registered recipes, then to set B at every stay; the declaration of that step names it.
- *Real, small*: the chosen candidate goes on the same way, and the next family of candidates
  (robust normalisation, the recipe) is talked through before set B.
- *Noise or harm*: cells are not the answer at few labels; the reading is talked through before
  any further candidate is built.
- *Checks before reading*: the control rerun on the G4 reads within 0.02 in its mean of the
  gap's cells at 200 (`9458a4d9…`) and of the T4 cells at 50 (`reader-low`, `d4e39bae…`), and
  every cell's stored description names the knobs it was run under; either failing stops the
  reading until it is explained.

**Limitations.** The budgets are the two where D2 is largest; what cells do from 500 stays up,
where the encoder already rebuilds the table, is the next step's question. The recipe per budget
was chosen for the network on raw tokens; a network over cells may want another rate, and the
selection turns none here. The queries and the learnt time are read on cells at 12 bins alone,
the candidate named before the run. A trial ran the exact path on one point per family at seed
1 on the M1 Pro (MPS), printing counts of cells only: six cells at 50 stays (the control, cells
over the corpus at 12 bins, cells at 24 bins, the learnt time, four queries, the outcomes weighed
alike) in 270 to 571 s each, and three at 200 (the control, cells over the corpus, eight queries)
in 332 to 569 s; without a stop every cell takes its 2,000 steps, and a cell over twelve bins
costs less than one over the raw tokens. The stored descriptions name each knob only where it is
turned.
