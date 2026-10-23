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
