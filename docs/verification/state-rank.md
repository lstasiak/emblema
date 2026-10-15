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
