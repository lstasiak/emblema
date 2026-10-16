# The width of the states and the probe's reading

Whether a backbone's reading of the intensive-care stays follows how many directions its
encoder's states fill. Read across the backbones of the stage already read at 500 stays, with no
new pretraining and no new campaign. The rule in force is in `docs/preregistration.md`, row "The
reader of the stage"; this note changes no rule and no verdict.

## 2026-10-16 — declared before the run: the effective rank against the reading at 500 stays

**Question.** Over the eleven backbones whose states were measured so far, the probe's reading
at 500 stays rises with the effective rank of the states: the decoder of four blocks (2.4, 3.0)
and the objective of states (3.7, 5.1) read lowest, the mixture (7.6 to 7.8) in the middle, the
intensive-care corpora (11.3 to 12.5) highest; Spearman +0.71 over the eleven. Does the order
hold on backbones whose rank nobody has looked at?

**The data.**

- *The rank.* `scripts/state_spread_report.py --backbone <id> --device mps` with its defaults,
  as in every reading so far: the effective rank of the top states on 4,500 tokens of 2012's
  validation windows, row `physionet2012`, `top`. Run on this machine for the held-out
  backbones only; about 30 s each.
- *The reading.* The area of `frozen_ridge@layer=concat` at 500 stays on the half held out by
  seed 101, mean over seeds 1 to 10, from campaigns already accepted (2026-10-13 to 2026-10-15,
  `data/report/l16/`, `l19/`, `l12/`).
- *Held out*, width 256, rank never measured; the test runs on these thirteen:
  `backbone-scale-1e8new-m`, `-3e8-m`, `-1e9-m`, `-1e9-4ep-m`,
  `-1e9-without-physionet2012-m`, `-1e9-without-cmapss-m`, `-1e9-without-physionet2019-m`,
  `backbone-physionet2019-m`, `backbone-mixed4-without-physionet2012-m`,
  `backbone-mixed4-stays-x4-m`, `backbone-mixed4-m-forecast`, `backbone-mixed4-m-harder`,
  `backbone-mixed4-m-seed4`.
- *Seen*, the eleven the hypothesis was formed on: the mixture at seeds 1 to 3, the objective of
  states twice, the decoder of four blocks twice, the intensive-care corpora four times.
- *Descriptive only*: the four backbones of width 512 (25.3M), whose rank is of another space;
  the small backbone on the stays; all 24 of width 256 together; the untrained encoder, which
  fills about 28 directions and reads 0.693, since it carries no pretraining.

**The test.** Spearman's ρ between rank and reading over the thirteen held out, its one-sided
p from 100,000 permutations; ρ above zero with p below 0.05 (ρ about 0.48 or more) supports the
order.

- *Faults, read first.* Every report reads 4,500 tokens over 38 channels of 2012. A backbone
  whose report refuses 2012's windows leaves the test and is named; fewer than ten left stops
  the reading.

**Predictions.**

1. *The order*: over the thirteen ρ lies above zero with p below 0.05, within 0.19 to 0.92,
   the Fisher 95 % interval of +0.71 over the eleven seen.

No order is predicted among the three points of the data curve: they read within 0.006 of
each other at 500 stays (0.673 to 0.680), under the spread of pretraining.

**What follows.**

- If the prediction holds: the rank becomes a screen read after every pretraining, minutes
  before any campaign, descriptively, beside the rule. The next objective is chosen to keep
  the states at or above the intensive-care corpora's width, and its declaration predicts its
  rank.
- If it fails (ρ at or below zero, or p at or above 0.05): the rank is not a screen; a term
  that widens the states has no reason from this stage. The next step reads the instrument,
  fine-tuning against the frozen probe at 500 stays, before another objective.

**Limitations.**

- An order across backbones is not a cause. Corpora, objective and scale differ between them;
  only a recipe that widens the states and nothing else tests the cause.
- Within one recipe the order does not hold over the seen: among the intensive-care corpora's
  four the narrowest reads best. The claim is about recipes, not pretrainings.
- The rank is of the top states; the probe reads every block side by side.
- Most of the thirteen share the mixture's objective; if their ranks span a narrow range the
  test has little power, and a failure says less than a success.

## 2026-10-16 — measured: the effective rank against the reading at 500 stays

**Question.** The one declared above: over the backbones held out, does the probe's reading at
500 stays rise with the effective rank of the top states on 2012's validation windows?

**Conditions.** Commit `91ab8940`, the declaration; MacBook Pro M1 Pro, MPS, fp32, torch
2.14.0. Each report by `scripts/state_spread_report.py --backbone <id> --device mps` with its
defaults (`data/report/l20/spread.sh`, 18 backbones one after another); the small backbone of two
blocks with `--layers 2`, since the default averages four. The readings at 500 stays are the
accepted campaigns the declaration names. The reading of the test is `data/report/l20/reading.py`:
Spearman's ρ over the held out, its one-sided p from 100,000 permutations at seed 0. The thirteen
took 88 min, the five descriptive 109 min.

**Faults, read first.** Three of the thirteen held out hold no row of 2012: the point of about
10⁹ values without PhysioNet 2012, PhysioNet 2019 alone, and the mixture of four without 2012.
None was pretrained on 2012, and the report reads a backbone's own corpora only. They leave the
test as declared and are read in the next section; ten remain, exactly the number under which
the reading stops. Every other report holds 4,500 tokens over 38 channels of 2012. The reading
script stopped at the first missing row at first and was changed to name the fault as declared;
no number changed.

**The held out, the test.** Rank of the top states on 2012; area of `frozen_ridge@layer=concat`
at 500 stays, mean over seeds 1 to 10 on the half held out by seed 101.

| Backbone | Rank | Area |
| --- | ---: | ---: |
| scale 10⁸, new publication | 2.0 | 0.680 |
| scale 3·10⁸ | 4.1 | 0.677 |
| mixture, harder masks | 6.0 | 0.706 |
| scale 10⁹ | 8.3 | 0.673 |
| mixture, forecast tail | 8.5 | 0.678 |
| 10⁹ without PhysioNet 2019 | 8.7 | 0.699 |
| scale 10⁹, four epochs | 8.9 | 0.688 |
| mixture of four, seed 4 | 9.0 | 0.693 |
| 10⁹ without C-MAPSS | 9.1 | 0.693 |
| mixture, stays ×4 | 10.7 | 0.709 |

Spearman **ρ = +0.552, one-sided p = 0.052** over ten. At ten, p below 0.05 needs ρ of about
0.56.

**Beside the test, descriptively.**

| Set | n | Spearman |
| --- | ---: | ---: |
| Seen, the eleven the hypothesis was formed on | 11 | +0.709 |
| Width 256, seen and held out together | 21 | +0.681 |
| Width 512 (four) and the small backbone | 5 | +0.900 |

The seen, rank and area: decoder of four blocks 2.4 (0.654), 3.0 (0.668); objective of states
3.7 (0.642), 5.1 (0.643); mixture 7.6 (0.695), 7.8 (0.692), 7.8 (0.710); intensive-care corpora
11.3 (0.729), 12.3 (0.719), 12.4 (0.707), 12.5 (0.693). The width 512: 4.3 (0.665), 4.5 (0.676),
4.7 (0.702), 6.3 (0.695); the small backbone over its two blocks 13.5 (0.708).

**Conclusions.**

1. The prediction fails: p lies at or above 0.05. By the declaration the rank is not a screen,
   and a term that widens the states has no reason from this stage. What follows is the branch
   declared for a failure: the instrument is read next, fine-tuning against the frozen probe at
   500 stays, before another objective.
2. The direction repeats on backbones nobody had looked at: ρ lies inside the predicted 0.19 to
   0.92, and the widest held out, the stays ×4, reads best. The rule was set before the run and
   the result is not read as a near success.
3. The held-out ranks span 2.0 to 10.7, six of ten within 8.3 to 9.1; the seen span 2.4 to 12.5.
   With that span the test had little power, as the declaration's last limitation said, and the
   pooled +0.68 over 21 draws its strength from the seen.
4. Across the three points of scale the rank rises, 2.0, 4.1, 8.3, and the area does not, 0.680,
   0.677, 0.673. Width without a better reading: the rank is at most one of the things a backbone
   that reads well has.
5. The three that never read 2012 are of another kind, read in the next section.

**Limitations.**

- The list of thirteen was wrong twice. Three of them cannot be read on 2012 by this report, and
  "about 30 s each" was 11 to 12 min for a backbone of 30 corpora and 35 min for the 25.3M shape,
  3.3 h in all. Both facts were in the registry and one trial run before the declaration; neither
  was checked. The test lost power for it: p below 0.05 needed ρ of about 0.48 at thirteen and
  0.56 at ten.
- The small backbone's top reading averages two blocks, the others' four; its rank is not of the
  same reading and stays out of every set but the descriptive.
- The declaration's limitations stand: an order across backbones is not a cause, the order does
  not hold within a recipe, and the rank is of the top states while the probe reads every block.
