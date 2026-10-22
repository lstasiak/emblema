# The network trained from nothing against the trees on the intensive-care stays

## 2026-10-22 — measured: the power of a non-inferiority rule against the trees on set B

**Question.** An encoder trained from nothing is to be judged against the trees by one
confirmation on set B, learning from every stay of set A. How likely is a rule of
non-inferiority to pass a network that truly matches the trees, for each margin, number of seeds
and number of scored stays?

**Conditions.**

| | |
|---|---|
| Code | `302fb316`; no new run |
| Machine | MacBook Pro M1 Pro, CPU |
| Noise taken from | campaign `87cbc0e7…` (`campaigns/curve-physionet2012-mixed5.toml`): learnt from every stay of set A, scored on set B (3,994 stays, 568 deaths), five seeds; `boosted_trees_per_channel` against `from_scratch` at every stay (`intensive-care-curve.md`, 2026-09-30) |
| Rule simulated | the paired interval of the network's area less the trees' lies above −δ, and that difference plus δ exceeds twice its standard error; the rule of `scripts/comparison_power_report.py` applied to the difference shifted by δ, at a true difference of zero |
| Draws | stays resampled within the two outcomes; 3,994 stays as set B, 8,000 as every stay of sets A and B scored once |

    uv run scripts/campaign_pairs_report.py --out data/report/l32b1/power-b \
      --campaign 87cbc0e7-806d-4062-bb05-3779ca53100f
    uv run scripts/comparison_power_report.py \
      --predictions data/report/l32b1/power-b/predictions.csv --out data/report/l32b1/power-b/sim \
      --budgets all --stays 3994 8000 --seeds 5 10 20 --gains 0.005 0.01 0.015 0.02 0.025 0.03 \
      --pair NI trees-vs-scratch 87cbc0e7-… boosted_trees_per_channel 87cbc0e7-… from_scratch

The simulation reads the spread of the difference, not its mean.

**Components of the difference** (standard deviations in area):

| Stays | Seed | Stays | Interaction |
|---|---|---|---|
| 3,994 | 0.0108 | 0.0047 | 0.0058 |
| 8,000 | 0.0107 | 0.0034 | 0.0041 |

**Power at a true difference of zero** (normal model of the components; at five seeds the model
agrees with the draws within 0.07):

| Stays | Seeds | δ = 0.005 | δ = 0.01 | δ = 0.015 | δ = 0.02 | δ = 0.025 |
|---|---|---|---|---|---|---|
| 3,994 | 5 | 0.18 | 0.40 | 0.67 | 0.86 | 0.96 |
| 3,994 | 10 | 0.21 | 0.51 | **0.80** | 0.95 | 0.99 |
| 3,994 | 20 | 0.20 | 0.54 | 0.84 | 0.97 | 1.00 |
| 8,000 | 10 | 0.29 | 0.67 | 0.92 | 0.99 | 1.00 |
| 8,000 | 20 | 0.32 | 0.76 | 0.97 | 1.00 | 1.00 |

**Conclusions.**

1. A margin of 0.01, near the top of the gains the literature reports from pretraining at every
   label (0.004 to 0.013 in area), has power 0.51 on set B at ten seeds. It stays below 0.80 on
   every stay that can be scored: 0.76 at 8,000 stays and twenty seeds. Set C is frozen until the
   final run.
2. More seeds help little: past ten, the stays and not the seeds carry the error.
3. On set B at ten seeds, 0.80 is reached at δ = 0.015. A network that passes is then no more
   than 0.015 below the trees by the 95 % interval, a one-sided bound at 97.5 %.
4. The margin of the confirmation is therefore 0.015, registered on 2026-10-22.
5. The same table, read with no margin, is the power of the superiority test that follows a
   passed confirmation: 0.21 at a true lead of 0.005 and 0.51 at 0.01 (set B, ten seeds). A lead
   of the size the literature reports over trees would most likely go unseen.

**Limitations.** The network of `87cbc0e7` ran without the stop; on the held-out fifth the stop
cut the spread over seeds from 0.018 to 0.006 (`intensive-care-curve.md`, 2026-10-02). The power
above is therefore likely conservative for a network under the stop, though not certainly so. Scoring 8,000 stays assumes a second
side as noisy as set B.
