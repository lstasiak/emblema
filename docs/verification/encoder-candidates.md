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

## 2026-10-23 — Colab G4, fp32: cells, a learnt time and several queries do not close the gap at 50 and 200 stays

**Question.** The one declared above: whether an encoder reading cells of channel and time bin,
with a learnt time encoding and several queries in its pooling, closes the gap to the trees'
table at 50 and 200 stays, and whether the absence of a whole channel is part of it.

**Conditions.** Campaigns `04df2a13…` (50) and `a23ee174…` (200) defined and ordered at
`e0c98e06`; 260 network cells on one Colab G4 under CUDA MPS, two lanes, 79 and 83 minutes
(about one minute a cell; the card read 95 % busy). Accepted by `data/report/l32b2/accept.sh`;
pairs by `scripts/campaign_pairs_report.py` (`data/report/l32b2/pairs.sh`), CSV under
`data/report/l32b2/pairs/`. The reused cells as declared.

**Checks before reading.** Both held. The control rerun reads 0.666 at 200 stays against 0.647
in the gap's campaign on the same kind of device (−0.018 [−0.029; −0.007] paired, within the
declared 0.02) and 0.624 at 50 against 0.625 on the T4 (+0.001). Every stored description names
the knobs it was run under and nothing else. Two runs of one control on one kind of device
differ by 0.018 in their means over ten seeds: the run-to-run noise of a mean is of the size of
the effects read here.

**Area under the ROC curve**, mean over seeds 1 to 10 (the spread over seeds 0.03 to 0.04 at
200 stays and 0.03 to 0.08 at 50):

| Candidate | 50 | 200 |
|---|---|---|
| Control on its tokens | 0.624 | 0.666 |
| Cells, 6 bins | 0.624 | **0.694** |
| Cells, 12 bins (named before the run) | 0.620 | 0.673 |
| Cells, 24 bins | 0.634 | 0.668 |
| Cells over the corpus's channels, 6 / 12 / 24 bins | 0.629 / 0.623 / 0.615 | 0.674 / 0.672 / 0.666 |
| Learnt time: on 12 bins / on raw tokens | 0.608 / 0.633 | 0.681 / 0.675 |
| Four queries: on 12 bins / on raw tokens | **0.647** / 0.641 | 0.671 / 0.665 |
| Eight queries on 12 bins | 0.632 | 0.668 |
| Outcomes weighed alike, 12 bins | 0.613 | 0.688 |
| Network over the trees' table (reused) | 0.674 | 0.706 |
| Trees per channel (reused) | 0.628 | 0.736 |
| MiniRocket (reused) | 0.655 | 0.713 |

**The pairs** (gain of the second over the first in area, 95 % interval over stays, seed by
seed mean ± SE; **bold** = carried):

| Pair | 50 | 200 |
|---|---|---|
| C: control → 12 bins (family, Holm over two) | −0.004 [−0.021; +0.013] ± 0.016 | +0.007 [−0.006; +0.021] ± 0.010 |
| A: 12 bins → over the corpus | +0.003 [−0.010; +0.016] ± 0.024 | −0.000 [−0.014; +0.013] ± 0.010 |
| T: 12 bins → learnt time | −0.012 [−0.019; −0.004] ± 0.013 | +0.008 [−0.002; +0.018] ± 0.013 |
| T: control → learnt time on raw tokens | +0.009 [+0.000; +0.017] ± 0.007 | +0.009 [−0.003; +0.021] ± 0.009 |
| Q: 12 bins → four queries | +0.027 [+0.015; +0.038] ± 0.020 | −0.002 [−0.013; +0.010] ± 0.006 |
| Q: four → eight queries | −0.015 [−0.024; −0.006] ± 0.014 | −0.003 [−0.015; +0.009] ± 0.009 |
| Q: control → four queries on raw tokens | +0.017 [+0.008; +0.027] ± 0.009 | −0.000 [−0.012; +0.012] ± 0.010 |
| W: 12 bins → outcomes weighed alike | −0.007 [−0.016; +0.003] ± 0.011 | **+0.015** [+0.003; +0.026] ± 0.007 |
| Bins: 6 → 12 | −0.004 [−0.013; +0.004] ± 0.014 | **−0.021** [−0.034; −0.009] ± 0.008 |
| Bins: 12 → 24 | **+0.014** [+0.006; +0.023] ± 0.006 | −0.005 [−0.016; +0.006] ± 0.012 |
| Bins over the corpus: 6 → 12, 12 → 24 | −0.006 ± 0.008, −0.008 ± 0.007 | −0.002 ± 0.006, −0.007 ± 0.015 |
| Bar: 12 bins → trees | +0.008 [−0.007; +0.023] ± 0.018 | **+0.063** [+0.050; +0.076] ± 0.009 |
| Bar: 12 bins → network over the table | **+0.054** [+0.036; +0.071] ± 0.018 | **+0.033** [+0.019; +0.048] ± 0.010 |
| Bar: 12 bins → MiniRocket | +0.035 [+0.013; +0.057] ± 0.023 | **+0.040** [+0.022; +0.058] ± 0.010 |

The best candidate of each budget against the control, from the means and not paired: four
queries on 12 bins +0.023 at 50, six bins +0.028 at 200; against the network over the table
−0.027 and −0.012.

**Predictions.** 1 failed: C is −0.004 and +0.007, neither carried, below the lower edges of
the patch model's lead. 2 failed: no candidate reaches the table's network at either budget;
the best at 200 lies 0.012 below it, within T, and the one named before the run 0.033 below,
carried. 3 held in the letter and says nothing: A is +0.003 and 0.000, inside ±T. 4 held: the
learnt time lies within ±T everywhere (+0.009 on raw tokens at both budgets, −0.012 on cells at
50). 5 failed: four queries on cells gain nothing at 200; at 50 they gain +0.027 by the stays'
interval but not by the seeds (SE 0.020). 6 failed in part: the class weight is carried at 200
(+0.015, within T); six bins beat twelve at 200 by 0.021, more than T, and twenty-four beat
twelve at 50 by 0.014.

**Conclusions.**

1. **By the declared rule the outcome is *noise*.** Cells at 12 bins, named before the run,
   change nothing against the network on raw tokens at 50 or 200 stays, and every variant of
   the cells lies 0.012 to 0.054 below the network over the trees' table and 0.04 to 0.06 below
   the trees at 200. The representation gap D2 at few labels is not closed by reading the
   window as cells with their aggregates and presence.
2. Which channels get a row, the learnt time and the queries move nothing beyond T; the
   outcomes weighed alike and fewer bins move the 200-stay reading by 0.015 to 0.02, and in
   the direction of less input (6 bins over 12), which points at how much a 4.8M encoder can
   learn from 200 stays rather than at what it is handed.
3. The two bars the candidates had to reach stand: the trees and the table's network lead
   every candidate at 200, and the table's network leads every candidate at 50.
4. As declared for this outcome, no further candidate is built before the reading is talked
   through; the step at 500 and every stay, and the confirmation on set B, do not follow from
   this campaign.

**Limitations.** The recipe per budget was chosen for the network on raw tokens and no rate was
turned for cells. At 50 stays it takes 2,000 steps of 16 windows, 640 passes over the drawn
stays, with no stop, no dropout and no decay, for an encoder of 4.8M parameters; the network
over the table under the same recipe is a far smaller function class, and the patch model whose
lead the predictions were drawn from was narrower, shallower and dropped a fifth of its
activations. The seconds per cell scale with the tokens each layout makes (14 s at 6 bins, 25 s
at 12, 54 s at 24, 45 s on raw tokens), so each candidate ran as the network it names. At 50 stays the spread over seeds (0.03 to 0.08) is larger than any effect
read; the budget tells candidates apart only by the stays' interval. The run-to-run noise of a
control's mean over ten seeds on one device, 0.018, bounds what a single campaign can resolve at
200 stays. The best of thirteen is a descriptive number.

## 2026-10-23 — declared before the run: the recipe and the capacity of the encoder from nothing at 50 and 200 stays

**Question.** The candidates above handed the encoder what the trees' table holds and it did not
close the gap; what moved the reading at 200 stays was less input (six bins over twelve) and a
balanced loss. At 50 stays the registered recipe takes 2,000 steps of 16 windows over 50 stays,
640 passes, with no stop, no dropout and no decay, for 4.8M parameters; the table's network,
one hidden layer over 410 features, beats it by 0.05 under the same steps. Is the gap at few
labels the cost of learning a large encoder from few stays, which a smaller shape and
regularisation close, rather than of what the encoder reads? Does a clip on the cells' values,
which inherit the readings' tails where the table's branch bounds them, add anything?

**Why this reading.** The patch model's lead over the raw network at 200 stays, +0.043
[+0.018; +0.069] (`intensive-care-curve.md`, 2026-10-01), came with a narrower, shallower
network under a dropout of a fifth; the reading above isolated the input and found about zero,
so the lead was capacity and regularisation. The small shape of the published regime gained
+0.019 at every stay (`intensive-care-curve.md`, 2026-10-03). On tabular-like data at hundreds
of examples, deep networks from nothing lose to trees for want of inductive bias and for excess
capacity (Grinsztajn et al., 2022); the candidates here take the capacity down.

**Conditions.** As the selection above, at the commit of this declaration; campaigns
`campaigns/capacity-{50,200}-physionet2012.toml`, 13 candidates at 50 stays and 15 at 200, each on
cells of six bins unless named otherwise: the control on raw tokens; six bins alone; the small
shape (two blocks, 64 wide, feed-forward 128, 16 heads); dropout 0.2; weight decay 0.1; a clip
at three deviations on the cells' values; the outcomes weighed alike; the small shape with
dropout (the candidate named before the run); the small shape with dropout, the class weight
and the decay; dropout with the clip; the small shape on raw tokens, with and without dropout;
the published regime whole without its stop (channel dropout, class weight, dropout, small
shape, a constant rate of 0.0005); at 200 stays also six bins under a stop with a fifth of the stays held out and a patience of
500 steps past the warm-up (a patience in epochs ends inside the warm-up at this budget), alone
and with the small shape and dropout. Network cells on
one Colab G4 under CUDA MPS in four lanes per order (`campaign_run --lane`), the first
measurement of the lanes' throughput against two; the trial on the M1 Pro (MPS).

**Readings.** R: six bins → the small shape with dropout on six bins, at 50 and 200, the one
family under Holm (two). Descriptively: each knob against six bins (shape, dropout, decay,
clip, class weight, stop); the combinations against the small shape with dropout; the small
shape on raw tokens against the control; the best candidate against the control, the network
over the table (0.674 / 0.706), the trees (0.628 / 0.736) and MiniRocket (0.655 / 0.713) of the
gap's campaigns, on the same stays and seeds; the control against the selection's control
(0.624 / 0.666) and the gap's (0.625 / 0.647). Rule as above: paired bootstrap over stays, a
pair carried when its interval excludes zero and its mean exceeds twice its standard error over
seeds; T about 0.021 at 50 and 0.018 at 200.

**Predictions.**

1. R is carried at 200 and lies at no less than +0.018, the lower edge of the patch model's
   lead, now read as the lead of a smaller, regularised network; at 50 it lies at or above
   zero, since the seeds' spread there (0.03 to 0.08) hides anything smaller than 0.04.
2. The best candidate reaches the network over the table within T at 200 stays (at or above
   0.688); at 50 it lies within T of 0.674 or above it.
3. The small shape alone gains less than the small shape with dropout at both budgets; the
   small shape on raw tokens gains over the control by no less than +0.019, the lead of that
   shape at every stay, within T.
4. The clip lies within ±T at both budgets (a clip at five deviations on raw tokens moved
   nothing, `intensive-care-curve.md`, 2026-10-01); the class weight reads +0.015 within T, as
   above; at 200 the stop on six bins lies at or above zero (at 500 stays a stop was worth 0.014
   over every drawn stay, `gap-to-the-trees.md`, 2026-10-23).
5. The published regime whole without its stop reads below the small shape with dropout at
   both budgets: its constant rate and withheld channels were tuned with the stop.

**Levels of the outcome.** *Reaches the table*: the best candidate's interval against the
network over the table holds zero or lies above it at a budget. *Real, small*: R carried below
that. *Noise*: R not carried at 200. *Harm*: R carried below zero.

**What follows.** *Reaches the table* at 200 or at 50: the recipe goes to 500 and every stay of
the half under the registered stop, then to set B at every stay under the registered parity;
its declaration names it. *Real, small* or *noise*: the encoder from nothing at few labels is
held back by sample complexity, not by its input, and no architecture from nothing closes the
gap; the ticket closes with that mechanism and the stage's next step is talked through.
*Checks before reading*: the control rerun within 0.02 of both earlier controls at each budget;
every stored description names its knobs; four lanes' cells per minute against two lanes'
(about one a minute) decide the lanes of later orders. Either check failing stops the reading.

**Limitations.** The rate stays the raw network's at each budget; the small shape may want
another. No stop at 50 stays: a fifth of 50 stays holds one or two deaths. The best of fifteen
is a descriptive number; the family is the one pair named before the run. A trial ran the exact
path on one point per family at seed 1 on the M1 Pro (MPS), printing counts of cells only: at
50 stays the control, the small shape with dropout, dropout with the clip and the published
regime (427, 65, 150 and 62 s); at 200 the control, the clip and the two stop candidates (564,
162, 62 and 26 s). A first trial with a patience of ten epochs at 200 stays ended its cell in
7 s, inside the warm-up, which is why the stop is stated in steps past it.

## 2026-10-24 — Colab G4, fp32: the small shape on six bins reaches the trees at 200 stays and leads them at 50

**Question.** The one declared above: whether a smaller, regularised encoder from nothing closes
the gap to the trees' table at 50 and 200 stays, and whether a clip on the cells' values adds
anything.

**Conditions.** Campaigns `0b337917…` (50) and `91170939…` (200) defined and ordered at
`0e98ca3a`; 280 network cells on one Colab G4 under CUDA MPS in four lanes per order
(`campaign_run --lane i/4`), 27 to 30 minutes wall-clock for all eight lanes (the card 93 %
busy); the eight results accepted lane by lane (`data/report/l32b2/accept-capacity.sh`). Pairs
by `scripts/campaign_pairs_report.py` (`pairs-capacity.sh`), CSV under
`data/report/l32b2/capacity-pairs/`. The reused cells as declared.

**Checks before reading.** Both held. The control rerun reads 0.615 at 50 and 0.655 at 200,
within 0.010 of the selection's controls (0.624, 0.666) and of the earlier campaigns' (0.625,
0.647), by the paired intervals. Every stored description names its knobs. Throughput: four
lanes per order on the same card answered 280 cells in 30 minutes against 260 in 83 minutes
under two lanes; the cells of the small shape are four times cheaper (12 against 45 s), so the
lanes gave about one and a half to two times the throughput, not four.

**Area under the ROC curve**, mean over seeds 1 to 10 (every candidate on six bins unless
named; "small" = two blocks, 64 wide, feed-forward 128, 16 heads):

| Candidate | 50 | 200 |
|---|---|---|
| Control on raw tokens | 0.615 | 0.655 |
| Six bins | 0.637 | 0.679 |
| **Small shape** | **0.656** | **0.733** |
| Small shape with dropout 0.2 (named before the run) | 0.655 | 0.715 |
| Small shape, dropout, class weight, decay 0.1 | 0.637 | 0.713 |
| Dropout 0.2 / decay 0.1 / clip at 3 / class weight | 0.623 / 0.629 / 0.621 / 0.618 | 0.685 / 0.678 / 0.683 / 0.687 |
| Dropout with the clip | 0.625 | 0.669 |
| Small shape on raw tokens, without / with dropout | 0.636 / 0.638 | 0.682 / 0.691 |
| The published regime whole, no stop | 0.645 | **0.735** |
| Stop (500 steps past the warm-up): six bins / small with dropout | | 0.689 / 0.684 |
| Network over the trees' table (reused) | 0.674 | 0.706 |
| Trees per channel (reused) | 0.628 | 0.736 |
| MiniRocket (reused) | 0.655 | 0.713 |

**The pairs** (gain of the second over the first in area, 95 % interval over stays, seed by
seed mean ± SE; **bold** = carried):

| Pair | 50 | 200 |
|---|---|---|
| R: six bins → small with dropout (family, Holm over two) | +0.018 [+0.009; +0.027] ± 0.011 | **+0.036** [+0.024; +0.049] ± 0.007 |
| Six bins → small shape | +0.019 [+0.010; +0.028] ± 0.012 | **+0.054** [+0.042; +0.066] ± 0.009 |
| Six bins → dropout / decay / clip / class weight | −0.014 / −0.008 / −0.017 / −0.019, none carried | +0.006 / −0.001 / +0.003 / +0.007, none carried |
| Small with dropout → small alone | +0.001 [−0.001; +0.003] ± 0.003 | **+0.018** [+0.012; +0.024] ± 0.006 |
| Small with dropout → with class weight and decay too | **−0.018** [−0.023; −0.014] ± 0.005 | −0.002 [−0.009; +0.005] ± 0.008 |
| Dropout → dropout with the clip | +0.002 [−0.005; +0.009] ± 0.010 | −0.017 [−0.027; −0.006] ± 0.011 |
| Small with dropout → the published regime whole | −0.011 [−0.019; −0.003] ± 0.008 | +0.019 [+0.010; +0.028] ± 0.010 |
| Control → small shape on raw tokens | +0.021 [+0.011; +0.031] ± 0.013 | +0.027 [+0.015; +0.038] ± 0.015 |
| Control → six bins | +0.022 [+0.006; +0.039] ± 0.018 | +0.024 [+0.010; +0.038] ± 0.014 |
| Control → small shape on six bins | **+0.041** [+0.025; +0.058] ± 0.014 | **+0.078** [+0.064; +0.092] ± 0.014 |
| Six bins → with the stop; small with dropout → with the stop | | +0.010 [−0.002; +0.023] ± 0.017; −0.032 [−0.040; −0.023] ± 0.025 |
| Bar: small shape → trees | **−0.028** [−0.043; −0.013] ± 0.011 | +0.003 [−0.011; +0.017] ± 0.006 |
| Bar: small shape → network over the table | +0.018 [+0.001; +0.035] ± 0.010 | **−0.027** [−0.043; −0.011] ± 0.004 |
| Bar: small shape → MiniRocket | −0.001 [−0.021; +0.019] ± 0.018 | **−0.020** [−0.040; −0.001] ± 0.005 |
| Bar: the published regime → trees, → the table | | +0.001 [−0.012; +0.015] ± 0.008; **−0.029** [−0.044; −0.014] ± 0.008 |

**Predictions.** 1 held: R is carried at 200 (+0.036, above +0.018) and at 50 lies above zero
(+0.018, not carried by the seeds). 2 held: at 200 the small shape reads 0.027 above the
network over the table, carried, and within 0.003 of the trees, not distinguishable; at 50 it
reads 0.018 below the table, within T, and 0.028 above the trees, carried. 3 failed: the small
shape alone reads above the small shape with dropout at 200 (+0.018, carried) and level with it
at 50; the small shape on raw tokens gains +0.021 and +0.027 over the control, within T of the
+0.019 predicted. 4 held in the letter: the clip, the decay and the class weight lie within ±T
at both budgets, all at or below zero at 50; the stop on six bins lies above zero at 200. 5
failed at 200: the published regime whole reads 0.019 above the small shape with dropout and
level with the small shape alone (0.735 against 0.733); it held at 50.

**Conclusions.**

1. **The gap at few labels was the encoder's size, not its input.** Shrinking the encoder from
   4.8M to two blocks 64 wide, on cells of six bins, lifts the network from nothing by 0.041 at
   50 stays and 0.078 at 200, both carried, where every change of what the encoder reads had
   moved it by 0.01 or less. By the declared rule the outcome is *reaches the table* at 200:
   the small shape reads 0.027 above the network handed the trees' table and level with the
   trees (+0.003 [−0.011; +0.017]). At 50 it reads within T of the table's network and 0.028
   above the trees, carried.
2. **Regularisation does not help a network this small at these budgets.** Dropout costs 0.018
   at 200 and nothing at 50; the decay, the clip on the cells' values and the class weight move
   nothing beyond T, and at 50 all four read below six bins alone. The published regime whole
   matches the small shape at 200 through its constant rate and withheld channels, and loses at
   50.
3. The two contributions are of one size at 200: six bins alone +0.024 over the control, the
   small shape on raw tokens +0.027, and together +0.078, more than their sum. At 50 the small
   shape on raw tokens is worth as much as on six bins (+0.021 against +0.019 on top of bins).
4. The stop in steps past the warm-up adds nothing at 200 on six bins and costs 0.032 with the
   small shape and dropout, where it stops a network still improving; the stop stays a knob for
   500 stays and up, where it was read.
5. As declared for *reaches the table*: the small shape on six bins goes to 500 and every stay
   of the half under the registered stop, then to set B at every stay under the registered
   parity, with its own declaration.

**Limitations.** The small shape without dropout was the best of fifteen at 200 and of thirteen
at 50, chosen on the stays it is scored on; the candidate named before the run, with dropout,
reads 0.018 below it at 200 and is the conservative estimate (+0.036 over six bins, 0.009 above
the table's network). The rate was the raw network's (0.003 at 200, 0.001 at 50); a small
network may want another, which the next step turns on the half. Ten seeds at 50 stays leave a
standard error of 0.01 to 0.02 on a mean; the differences among the regularisers there are not
read. Two of the three bars at 200 are carried in the candidate's favour (above the table's
network and MiniRocket); parity with the trees is a non-finding at this budget, not a
demonstrated equality, and is read on set B by the registered rule.

## 2026-10-24 — declared before the run: the small shape on six bins at 500 and every stay of the half

**Question.** At 200 stays the small encoder on six bins stands level with the trees and above
the network handed their table; at 50 it leads the trees. From 500 stays up the 4.8M encoder
already rebuilds the table, and what remains to the trees there is what they make of the table
that a smooth model does not (D1, +0.030 at every stay, `gap-to-the-trees.md`, 2026-10-23).
Does the small shape keep its standing at 500 and every stay of the half, where the registered
stop holds a fifth of the stays out, and which rate does it want there? The reading names the
variant that goes to set B.

**Conditions.** As above, at the commit of this declaration; campaigns
`campaigns/capacity-{500,all}-physionet2012.toml`, seven candidates each: the 4.8M encoder on raw
tokens (the control) and on six bins, the small shape on six bins under the recipe registered at
the budget (a stop with a fifth held out and a patience of ten epochs; at every stay a rate of
0.000333), the same at two other rates (0.003 and 0.0003 at 500; 0.001 and 0.003 at every stay),
the same with a dropout of a fifth, and the same learning from every drawn stay with no stop.
The trees, the network over the table and MiniRocket at 500 and every stay are read from the
gap's campaigns (`d1741c48…`, `105c9081…`, `6a388747…`) on the same stays and seeds; at every
stay the trees are the variant chosen there (`learning_rate=0.1,rounds=300`). Network cells on
one Colab G4 under CUDA MPS, four lanes per order; the trial on the M1 Pro (MPS).

**Readings.** S: six bins → the small shape under the registered recipe, at 500 and every stay,
the one family under Holm (two). Descriptively: the control → the small shape; each rate, the
dropout and the run without a stop against the registered recipe; the small shape, at its
registered recipe and at the best of its variants, against the trees, the table's network and
MiniRocket. Rule as above. **The variant for set B** is the small shape's variant with the
highest mean at every stay of the half, the registered recipe on a tie within 0.005; it is named
in the register before the set B campaign is defined, with its own declaration. The best of five
chosen on the stays it is scored on reads high; set B is where it is tested.

**Predictions.**

1. S is carried at both budgets and lies at no less than +0.024, the lower edge of the small
   shape's gain over six bins at 200 under dropout (the conservative estimate above).
2. At every stay the small shape under the registered recipe reads between the table's
   network (0.810) and the trees (0.839): the pair against the trees lies within [−0.030; +0.003],
   D1's distance below and the 200-stay reading above, and the pair against the table's network
   at or above zero. The registered parity on set B needs the lower edge against the trees above
   −0.015; on the half, the same bound is the level that makes the step to set B worth its cost.
3. At 500 the small shape reads within T (0.018) of the trees (0.787) and above the table's
   network (0.754) by no less than +0.011, the lower edge of its lead at 200.
4. A higher rate than the registered one gains at every stay (the registered 0.000333 was chosen
   for the 4.8M encoder; the small shape read best at 0.003 at 200), by no more than T; dropout
   lies within ±T; learning from every drawn stay with no stop lies at or below the stop, as it
   did for the network over the table (−0.014 at 500, `gap-to-the-trees.md`, 2026-10-23).

**Levels of the outcome.** *Within the margin*: the small shape's pair against the trees at
every stay has its lower edge above −0.015. *Below the margin*: the pair's lower edge below
−0.015 while S is carried. *Lost*: S not carried at every stay.

**What follows.** *Within the margin*: the named variant goes to set B at every stay, learning
from every stay of set A, against the trees at their chosen setting, ten seeds, under the
registered rule (non-inferiority at δ = 0.015, then superiority), with a register row naming it
before the campaign is defined. *Below the margin*: the small shape has closed what the 4.8M
encoder could not at few labels and not what the trees make of the table at every stay; the
step to set B waits for a reading above the table (the next question of the stage), and the
ticket closes with that mechanism. *Lost*: the small shape's standing at 200 does not carry to
larger budgets; talked through. *Checks before reading*: the controls within 0.02 of the gap's
cells at 500 and every stay (0.749, 0.796 on the G4); every stored description names its
knobs.

**Limitations.** Set B is read once; the variant is chosen on the half, so its reading there is
an estimate, not a test. The rate grid is three points. The stop's patience in epochs is sound
from 500 stays up (at 500, ten epochs are 250 steps against a warm-up of 200), though a small
network under it may stop early: in the trial on the M1 Pro (MPS), one point per family at
seed 1 printing counts only, the small shape under the stop took 10 s at 500 and 34 s at every
stay against 54 and 97 s without it; the 4.8M encoder on six bins took 28 and 95 s.

## 2026-10-24 — Colab G4, fp32: at 500 and every stay the cells carry the gain, the small shape adds nothing, and the trees keep 0.008

**Question.** The one declared above: whether the small shape on six bins keeps its standing at
500 and every stay of the half, which rate it wants there, and which variant goes to set B.

**Conditions.** Campaigns `c995ae4d…` (500) and `892f50a0…` (every stay) defined and ordered at
`49452626`; 140 network cells on one Colab G4 under CUDA MPS in four lanes per order, 7.5 and
11 minutes wall-clock per order (the card 90 % busy); the eight results accepted lane by lane
(`data/report/l32b2/accept-large.sh`). Pairs by `scripts/campaign_pairs_report.py`
(`pairs-capacity-large.sh`), CSV under `data/report/l32b2/capacity-large-pairs/`. The reused cells
as declared.

**Checks before reading.** Both held: the control reads 0.743 at 500 and 0.799 at every stay,
+0.006 and −0.003 from the gap's cells (0.749, 0.796) by the paired intervals; every stored
description names its knobs.

**Area under the ROC curve**, mean over seeds 1 to 10 (the stop with a fifth held out and a
patience of ten epochs unless named; at every stay the registered rate 0.000333):

| Candidate | 500 | every stay |
|---|---|---|
| Control, 4.8M on raw tokens | 0.743 | 0.799 |
| 4.8M on six bins | 0.766 | 0.829 |
| **Small shape on six bins, registered recipe** | **0.775** | **0.831** |
| Small shape, rate 0.003 / 0.0003 (500); 0.001 / 0.003 (every stay) | 0.775 / 0.731 | 0.823 / 0.823 |
| Small shape with dropout 0.2 | 0.772 | 0.829 |
| Small shape, every drawn stay, no stop | 0.767 | 0.828 |
| Network over the trees' table (reused) | 0.754 | 0.810 |
| Trees per channel (reused; at every stay `learning_rate=0.1,rounds=300`) | 0.787 | 0.839 |
| MiniRocket (reused) | 0.756 | 0.809 |

**The pairs** (gain of the second over the first in area, 95 % interval over stays, seed by
seed mean ± SE; **bold** = carried):

| Pair | 500 | every stay |
|---|---|---|
| S: six bins → small shape (family, Holm over two) | +0.009 [+0.002; +0.017] ± 0.007 | +0.002 [−0.003; +0.007] ± 0.004 |
| Control → small shape | **+0.032** [+0.017; +0.047] ± 0.009 | **+0.032** [+0.019; +0.046] ± 0.004 |
| Small shape → other rates | −0.000 ± 0.004 (0.003); −0.044 ± 0.031 (0.0003) | −0.008 ± 0.004 (0.001); **−0.008** [−0.014; −0.001] ± 0.003 (0.003) |
| Small shape → with dropout | −0.004 [−0.007; −0.000] ± 0.002 | −0.002 [−0.003; −0.001] ± 0.001 |
| Small shape → no stop | −0.008 [−0.017; −0.000] ± 0.005 | −0.003 [−0.009; +0.002] ± 0.002 |
| Bar: small shape → trees | +0.012 [−0.002; +0.025] ± 0.007 | +0.008 [−0.007; +0.023] ± 0.002 |
| Bar: small shape → network over the table | **−0.022** [−0.040; −0.004] ± 0.006 | **−0.022** [−0.039; −0.005] ± 0.002 |
| Bar: small shape → MiniRocket | **−0.019** [−0.036; −0.003] ± 0.008 | **−0.022** [−0.040; −0.005] ± 0.003 |
| Bar: small shape at 0.001 / 0.003 → trees | | **+0.016** [+0.001; +0.030]; **+0.016** [+0.000; +0.031] |

**Predictions.** 1 failed: S is +0.009 at 500 and +0.002 at every stay, neither carried; the
4.8M encoder on six bins reaches the small shape from 500 stays up. 2 held: the small shape
reads 0.008 below the trees at every stay, inside [−0.030; +0.003], and 0.022 above the table's
network, carried. 3 held: at 500 it reads 0.012 below the trees, within T, and 0.022 above the
table's network. 4 failed in its first part: both higher rates lose 0.008 at every stay, the
registered rate is the best of three; dropout lies within ±T and learning from every drawn stay
reads at or below the stop, as predicted.

**Conclusions.**

1. **From 500 stays up the cells carry the gain and the shape carries nothing.** Six bins lift
   the 4.8M encoder by 0.023 and 0.030 (0.743 → 0.766, 0.799 → 0.829), and the small shape adds
   0.009 and 0.002 on top, neither carried. At few labels the shape was the lever and the cells
   were not; at many labels it is the other way round. The small shape on six bins is the one
   candidate that holds at every budget: +0.041 / +0.078 / +0.032 / +0.032 over the control at
   50 / 200 / 500 / every stay, all carried.
2. **The cells close most of what the trees made of the table.** At every stay the table's
   network read 0.030 below the trees (D1, `gap-to-the-trees.md`); the network on six bins reads
   0.022 above the table's network, carried, and 0.008 below the trees, not distinguishable.
   An encoder that reads the window as cells does hold more than ten statistics per channel,
   which the declared outcome of L-32b1 asked for.
3. **By the declared levels the step to set B does not follow from this reading; by the
   register's gate it may.** The pair against the trees at every stay is +0.008 in the trees'
   favour with an interval of [−0.023; +0.007] for the network, so its lower edge lies below
   −0.015, and S is not carried, which the declaration names *lost*; the word is wrong for a
   candidate that reads 0.008 below the trees, and the level that fits is *below the margin*:
   the mechanism is not the shape but the 0.008 the trees keep at every stay. The declaration's
   bound on the half (the lower edge above −0.015) was its own and stricter than the register's:
   the registered rule (2026-10-22) lets one configuration go on to set B when its mean at every
   stay of the half lies within 0.015 of the trees', and 0.008 is within it. On set B (3,994
   stays) the interval narrows to about ±0.011; at a true difference of −0.008 the registered
   rule passes about one time in four, at −0.004 one in two, at zero four in five, before any
   gain from learning on twice the stays. The declaration's *below the margin* branch closes the
   ticket with the mechanism; whether to spend set B at these odds is decided in the next
   section.
4. The registered rate (0.000333) is the best of three at every stay for the small shape as it
   was for the 4.8M encoder; dropout and a run without the stop cost 0.002 to 0.008.

**Limitations.** The small shape was chosen on the half at 50 and 200 and read here at 500 and
every stay without a choice, so this reading is a fresh one for it; the bar against the trees
is read at the trees' setting chosen on these same stays. Two of the three rates at every stay
are a two-point grid around the registered one; a rate between 0.000333 and 0.001 was not run.
The candidates at every stay differ by 0.002 to 0.008, within the run-to-run noise of a mean
over ten seeds (0.01 to 0.02 at 200; smaller here, the spread over seeds being 0.007 to 0.012).

## 2026-10-24 — declared before the run: the inputs of the small encoder at every stay of the half

**Question.** The inputs an encoder from nothing reads (how many bins, rows for the channels a
stay never measured, a learnt time, several queries) were chosen at 50 and 200 stays, where the
encoder's size moved the reading by 0.04 to 0.08 and every input by 0.01 or less. From 500
stays up it is the other way round: the cells carry 0.023 and 0.030 and the shape nothing. Which
input does the small encoder want at every stay of the half, where the input is the lever, and
does any input lift it to the trees? The reading names the variant that goes to set B.

**Why this step.** The reading above ended *below the margin* by its own levels, whose bound on
the half (the lower edge against the trees above −0.015) was stricter than the register's: the
registered rule lets one configuration go on when its mean at every stay of the half lies within
0.015 of the trees', and the small shape's 0.008 does. The user chose (2026-10-24) to spend the
single reading on set B, and to read the inputs first, where they matter, so that the variant
named for set B is chosen in the regime it is tested in. The step costs 70 cells of a few seconds
to a minute each.

**Conditions.** As above, at the commit of this declaration; campaign
`campaigns/inputs-all-physionet2012.toml` at every stay of the half, seven candidates under the
registered recipe (a stop with a fifth held out, a patience of ten epochs, a rate of 0.000333):
the small shape on six bins (the candidate as read, the control of this campaign), on twelve and
on twenty-four bins; on six bins with a row for every channel the corpus measures
(`cells=corpus`), with a learnt time encoding, and with four learnt queries in an attention
pooling; and the 4.8M encoder on twelve bins. The trees (`learning_rate=0.1,rounds=300`), the
network over the table and MiniRocket at every stay are read from the gap's campaign
(`105c9081…`) on the same stays and seeds; the small shape on six bins and the 4.8M encoder on six
bins as read above (`892f50a0…`). Network cells on one Colab G4 under CUDA MPS, four lanes; the
trial on the M1 Pro (MPS).

**Readings.** B: six → twelve bins and six → twenty-four bins, the small shape, the one family
under Holm (two). Descriptively: twelve → twenty-four; each of the rows, the learnt time and the
queries against six bins; the 4.8M encoder on twelve bins against the small shape on twelve
(shape, in this campaign) and against the 4.8M encoder on six bins (bins at 4.8M, across
campaigns); the variant with the highest mean against the trees, the table's network and
MiniRocket. Rule as above. **The variant for set B** is the one with the highest mean at every
stay of the half; on a tie within 0.005, six bins under the registered recipe, the candidate
already read and the cheaper one. It is named in the register before the set B campaign is
defined, with its own declaration. The best of seven chosen on the stays it is scored on reads
high by one to two standard errors of a mean (0.002 to 0.004); set B is where it is tested.

**Predictions.**

1. Six → twelve bins lies within [−0.009; +0.018]: the lower edge is the upper edge of the
   same pair's loss at 200 under the 4.8M encoder (−0.021 [−0.034; −0.009]), where the cost was
   capacity, which the small shape at every stay no longer pays; the upper edge is T.
   Twelve → twenty-four lies within [−0.016; +0.023], the pair's intervals at 200 and 50.
2. The rows, the learnt time and the queries lie within ±T of six bins, as at 200 (A
   [−0.014; +0.013], T [−0.002; +0.018], Q [−0.013; +0.010]).
3. The 4.8M encoder on twelve bins lies within [−0.003; +0.007] of the small shape on twelve,
   the shape's pair at every stay on six bins.
4. The variant with the highest mean lies against the trees within [−0.023; +0.007], the pair's
   interval above, and its point between −0.008 and +0.002: the standing as read plus what the
   best of seven adds.

**Levels of the outcome.** *Finer bins carry*: B is carried for twelve or twenty-four bins.
*Noise*: B is not carried and no input is carried against six bins. *Level with the trees*: in
either, some variant's pair against the trees has its point at or above zero.

**What follows.** In every level the named variant goes to set B at every stay, learning from
every stay of set A, against the trees at their chosen setting, ten seeds, under the registered
rule (non-inferiority at δ = 0.015, then superiority), with a register row naming it before the
campaign is defined. The forecast of that rule's pass, read from the half: at a true difference
of −0.008 one time in four, at −0.004 one in two, at zero four in five (the half's half-width of
0.015 over √2 for 3,994 stays, before any gain from learning on twice the stays). *Finer bins
carry*: the carried bins become the layout the pretraining path reads (the next ticket). *Noise*:
six bins stay, and the inputs are settled at every budget read. *Checks before reading*: the
small shape on six bins within 0.01 of its reading above (0.831), as the control's rerun at every
stay was (−0.003 [−0.006; −0.001]); every stored description names its knobs.

**Limitations.** The recipe (rate, class weight) is not turned here; it was read at every stay
above for six bins and is taken as it stands for twelve and twenty-four, whose cells hold two and
four times the tokens. The bar against the trees is read at the trees' setting chosen on these
same stays. The forecast for set B assumes the half's spread scales with the number of stays and
no change of standing from twice the training stays; the network's and the trees' gains from 500
to every stay of the half were alike (0.056 and 0.052), so the forecast is not tilted either way.
A set B that fails the rule for the named variant does not close set B to a configuration of
another kind (a pretrained encoder): the register confirms one configuration once, though
every reading of set B spends it.

## 2026-10-24 — Colab G4, fp32: at every stay no input moves the small shape beyond noise, and rows for unmeasured channels cost 0.009

**Question.** The one declared above: which input the small shape wants at every stay of the
half, whether any lifts it to the trees, and which variant goes to set B.

**Conditions.** Campaign `52b9f184…` defined and ordered at `0546196f`; 70 network cells on one
Colab G4 under CUDA MPS in four lanes, 10 minutes wall-clock (the card 90 % busy); the four
results accepted lane by lane (`data/report/l32b2/accept-inputs.sh`). Pairs by
`scripts/campaign_pairs_report.py` (`pairs-inputs.sh`), CSV under
`data/report/l32b2/inputs-pairs/`. The reused cells as declared. A cell on six bins took 13 s,
on twelve 26 s, on twenty-four 79 s; the 4.8M encoder on twelve bins 44 s.

**Checks before reading.** Both held. The control's rerun reads 0.831, identical at every seed
to its reading above: the two orders ran the same encoder code, and a G4 run under one seed
repeats itself, so the check holds but says nothing of run-to-run noise (the earlier rerun's
−0.003 was read between orders from different commits). Every stored description
names its knobs.

**Area under the ROC curve** at every stay of the half, mean over seeds 1 to 10, the small shape
under the registered recipe unless named:

| Candidate | every stay |
|---|---|
| **Six bins (the control, rerun)** | **0.831** |
| Twelve bins | 0.830 |
| Twenty-four bins | 0.827 |
| Six bins, a row for every channel the corpus measures | 0.822 |
| Six bins, a learnt time encoding | 0.832 |
| Six bins, four queries in an attention pooling | 0.828 |
| 4.8M encoder on twelve bins | 0.829 |
| 4.8M encoder on six bins (reused) | 0.829 |
| Network over the trees' table (reused) | 0.810 |
| Trees per channel, `learning_rate=0.1,rounds=300` (reused) | 0.839 |
| MiniRocket (reused) | 0.809 |

**The pairs** (gain of the second over the first in area, 95 % interval over stays, seed by
seed mean ± SE; **bold** = carried):

| Pair | every stay |
|---|---|
| B: six → twelve bins (family, Holm over two) | −0.001 [−0.005; +0.002] ± 0.002 |
| B: six → twenty-four bins | −0.004 [−0.009; +0.001] ± 0.003 |
| Twelve → twenty-four bins | −0.003 [−0.006; +0.000] ± 0.002 |
| A: six bins → a row for every channel | **−0.009** [−0.018; −0.001] ± 0.001 |
| T: six bins → a learnt time | +0.001 [−0.001; +0.002] ± 0.001 |
| Q: six bins → four queries | −0.003 [−0.006; +0.001] ± 0.003 |
| Shape on twelve bins: small → 4.8M | −0.001 [−0.006; +0.004] ± 0.005 |
| Bins at 4.8M: six (reused) → twelve | −0.001 [−0.006; +0.004] ± 0.002 |
| Check: six bins as read above → rerun | +0.000 [+0.000; +0.000] ± 0.000 |
| Bar: six → trees | +0.008 [−0.007; +0.023] ± 0.002 |
| Bar: a learnt time → trees | +0.007 [−0.008; +0.022] ± 0.003 |
| Bar: twelve / twenty-four → trees | +0.009 [−0.006; +0.025] ± 0.003; +0.012 [−0.004; +0.028] ± 0.004 |
| Bar: four queries / 4.8M on twelve → trees | +0.011 [−0.005; +0.026] ± 0.002; +0.011 [−0.004; +0.025] ± 0.003 |
| Bar: a row for every channel → trees | **+0.018** [+0.001; +0.034] ± 0.002 |
| Bar: every variant → the table's network | −0.012 to −0.022; carried for all but twenty-four bins and the rows |
| Bar: every variant → MiniRocket | −0.013 to −0.023; carried for all but the rows |

**Predictions.** All four held. 1: six → twelve is −0.001, within [−0.009; +0.018]; twelve
→ twenty-four −0.003, within [−0.016; +0.023]. 2: the rows −0.009, the learnt time +0.001 and
the queries −0.003 lie within their intervals at 200. 3: the 4.8M encoder on twelve bins
reads −0.001 from the small shape there, within [−0.003; +0.007]. 4: the highest mean is the
learnt time (0.832), which reads 0.007 below the trees, within [−0.023; +0.007] and its point
within [−0.008; +0.002].

**Level of the outcome.** *Noise*, in its consequence: B is not carried and no input is carried
as a gain. The levels did not foresee a carried loss; the rows for every channel are one
(−0.009), so the outcome is *noise* with one input ruled out rather than left open.

**Conclusions.**

1. **The inputs are settled at every budget read.** At every stay, as at 50 and 200, no input
   moves the small shape by more than 0.004 in either direction, except the rows for the
   channels a stay never measured, which cost 0.009, carried. Twelve and twenty-four bins cost
   two and six times the time of six and gain nothing; the layout the pretraining path reads is
   six bins over the channels of the window.
2. **The gain from the cells does not depend on their count between six and twelve at either
   size.** The 4.8M encoder on twelve bins reads level with the small shape on twelve (−0.001)
   and with itself on six (−0.001).
3. **The variant for set B is the small shape on six bins under the registered recipe.** By the
   rule declared above the highest mean, the learnt time, lies within 0.005 of six bins (0.832
   against 0.831), and the tie goes to six bins, the candidate already read and the cheaper one.
   Its standing against the trees at every stay is the one read above, 0.008 below, not
   distinguishable; the forecast of the register's rule at that standing is one pass in four.
4. **A row for every channel the corpus measures hurts.** Cells for channels a stay never
   measured carry nothing but their absence, and every stay gets the same number of tokens (a cell
   took 21 s against 13); the pair against the trees for this variant is the one carried bar
   (+0.018 in the trees' favour). Why the rows cost is not read here.
5. A rerun of unchanged code on the G4 repeats to the bit, so an in-campaign control and a
   reused one are equal when the code is unchanged between the orders; the cross-campaign
   differences read earlier (0.003 at every stay, 0.018 at 200) were between orders from
   different commits. What a different card or driver adds is not read.

**Limitations.** The recipe was read for six bins and taken as it stands for twelve and
twenty-four, whose cells hold more tokens; a rate or patience of their own was not run. The
seven variants were read once each over ten seeds; the best of seven reads high by one to two
standard errors of a mean (0.002 to 0.004), which the tie rule absorbs by naming the candidate
read before this campaign. The bar is read at the trees' setting chosen on these same stays.

## 2026-10-24 — declared before the run: the small shape on six bins against the trees on set B

**Question.** Is the small encoder from nothing on cells of six bins, learnt from every stay of
set A, no worse than the trees per channel on set B by the registered margin of 0.015 in area,
and if it is, better than them?

**The configuration named.** The register's row of 2026-10-24 names
`from_scratch@bins=6,feedforward_width=128,heads=16,layers=2,learning_rate=0.000333,patience=10,stop_share=0.2,width=64`
for set B, chosen on the half by the rule declared above: the highest mean at every stay, the
learnt time's, lies within 0.005 of six bins', and the tie goes to six bins. Its mean at every
stay of the half is 0.831 against the trees' 0.839, within the 0.015 the register's gate asks.
It is the configuration the ticket read at 500 and every stay without a choice, and the one
chosen at 50 and 200.

**Conditions.** Campaign `campaigns/parity-b-physionet2012.toml`, at the commit of this
declaration: `purpose = "tuning"`, so every candidate learns from every stay of set A (3,997,
the network holding a fifth out for its stop) and is scored on set B (3,994 stays, 568 deaths);
seeds 1 to 10; the trees at `learning_rate=0.1,rounds=300` as the control, the named network as
the endpoint, the trees at their default setting read beside them. The network's ten cells on
one Colab G4 under CUDA MPS in four lanes (the ml order); the trees' twenty cells on the M1 Pro's
processor (the general order), the same commit in both. Trial on the M1 Pro before this
declaration: seed 1 of each family under this purpose, three cells (the trees in 24 s on the
processor, the network in 82 s on MPS), counts only; the trial's campaign is never read and is
deleted with the other trials.

**Reading.** The registered rule (2026-10-22): from the paired comparison of the network
against the trees at `learning_rate=0.1,rounds=300`, the network is *no worse* when the 95 %
interval of its area less the trees' lies above −0.015 and that difference plus 0.015 exceeds
twice its standard error over seeds; only then is it read for superiority, *better* when the
interval lies above zero and the difference exceeds twice its standard error. The pairs report
(`scripts/campaign_pairs_report.py --pair`) gives the trees' gain over the network; the rule is
read from its interval with the sign reversed. The pair against the trees at their default
setting is reported beside, not judged. Nothing else on set B is read.

**Predictions.** The network's area less the trees' on set B lies within [−0.023; +0.007], the
pair's interval at every stay of the half, with its point near −0.008: the network's and the
trees' gains from 500 to every stay of the half were alike (0.056 and 0.052), so learning from
twice the stays tilts the pair neither way. The interval on set B has a half-width of about 0.011
(the half's 0.015 over √2); the spread over seeds is about 0.01, as on the half.

**Levels of the outcome.** *No worse*: the rule's first stage passes; forecast from the half,
one time in four at a true −0.008, one in two at −0.004, four in five at zero. *Better*: both
stages pass; forecast one in forty at a true zero and one in four only at the interval's upper
edge (+0.007). *Not shown*: the first stage fails; forecast three times in four.

**What follows.** *No worse* or *better*: a network from nothing on cells is the first encoder
of this project to reach the trees at every stay on the validation side; the ticket closes with
that standing, and the pretraining path reads the same cells (the next ticket). *Not shown*: the
ticket closes with the half's reading as the encoder's standing, 0.008 below the trees and not
distinguishable from them on the half; the pretraining path is read on the half; whether set B
is spent again on a configuration of another kind is the register's decision, not this note's.
In every level the register's row stands as the configuration confirmed or not. *Checks before
reading*: the campaign's grid is complete (ten network cells, twenty of the trees); every stored
description names its knobs; the trees at `learning_rate=0.1,rounds=300` read above the trees
at their default setting on set B as they did on the half (0.839 against 0.832 in
`105c9081…`), which guards the control's setting against a copying error.

**Limitations.** One reading of set B, by one configuration, as the register allows; the chance of
a pass at the predicted standing is one in four, and the reading is spent knowing it (the user's
decision of 2026-10-24). The configuration was chosen on the half among seven at every stay and
among thirteen at 50 and 200, so its standing on the half reads high by the selection; set B is
where it is read without one. The trees' setting was chosen on the half too, so both sides of the
pair carry a selection of the same kind.
