# Full fine-tuning at few labels against the network from nothing

## 2026-10-22 — declared before the reading: fine-tuning at 200, 50 and 20 stays under four pretrainings of the mixture

**Question.** Full fine-tuning under the mixture of four (`backbone-mixed4-m`) has beaten the
same network trained from nothing at 200 stays in every reading so far, and nowhere else. All of
those readings were made under the same two pretrainings. Does the gain hold under two
pretrainings it has never been read under, and how large is it under all four?

**Why this reading, and why fresh pretrainings decide it.** The gain at 200 stays was found by
reading four budgets (200, 500, 1,000 and every stay; `verdict-statistics.md`, 2026-10-13) and
keeping the one that was positive. A maximum chosen that way is biased upwards. Its two
pretrainings (seeds 1 and 2) have also been read at 200 stays three times:

| Reading | Side | Seed 1 | Seed 2 |
|---|---|---|---|
| `intensive-care-curve.md`, 2026-10-05, Colab G4 | validation, 3,994 stays | +0.036 [+0.027; +0.046] | +0.030 [+0.020; +0.040] |
| `verdict-statistics.md`, 2026-10-13, Kaggle T4 | half of tuning, seed 101 | +0.032 [+0.018; +0.046] | +0.033 [+0.019; +0.047] |
| `fine-tuning-recipe.md`, 2026-10-16, Colab G4 | the same half | +0.012 [−0.001; +0.026] | +0.028 [+0.013; +0.043] |

So seeds 1 and 2 cannot confirm the gain they suggested. The confirmatory reading is made under
seeds 3 and 4, whose cells were run and accepted on 2026-10-20 and have not been read: the
acceptance printed only counts of cells. The mean over all four pretrainings estimates the
gain's size; it is not the test.

**Conditions.**

- *Task and side.* In-hospital death on PhysioNet 2012 (`a6a9c653-…`). The half of the tuning
  stays held out by seed 101 (`one_in = 2`, 2,000 stays scored). Seeds 1 to 10 draw the same
  labels under every pretraining. The validation side is not read.
- *Arms.* `full_fine_tuning` and the network from nothing, under the settings in force at each
  budget (`docs/preregistration.md`, "Recipe").
- *Campaigns.* All four pretrainings (`sha256:0d81c01e…`, `94f00d72…`, `50b46711…`, `5a45aac9…`)
  have these campaigns on the registry:

| File | Budgets | Seeds 1, 2 | Seeds 3, 4 |
|---|---|---|---|
| `reader-200-physionet2012.toml` | 200 | `65c6a55e`, `7a44614d` (Kaggle T4, order from `08a764f5`) | `a801d0d4`, `3d2062a3` (Kaggle T4, order from `8aced485`) |
| `reader-low-physionet2012.toml` | 20, 50 | `d4e39bae`, `41edb9e5` (Kaggle T4, `8aced485`) | defined and ordered from the commit of this section; Kaggle T4 ×2, one order per GPU |

  All six existing campaigns are `finished`, each with every cell (20 at 200 stays, 40 at 20
  and 50), and the registry names the expected backbone digest for each one.
- *The code between the two orders.* Between `08a764f5` and `8aced485` the evaluation path
  changed in one respect only: the encoder's share of the rate (ADR-0055). Under a share of
  one, the optimiser receives one group of every trainable parameter at the plain rate, and
  `test_groups_at_the_whole_rate_repeat_the_plain_parameters_bit_for_bit` holds that this
  matches the plain run bit for bit.
- *Trial.* `data/report/l30/trial.sh` ran the exact pairing command on the campaigns already
  read (seeds 1 and 2 at 200, one side per pretraining and the two pooled by `+`). It repeated
  the reading of 2026-10-13 to the third decimal (+0.032 ± 0.016, +0.033 ± 0.011). Pooled
  over the two, the result was +0.033 ± 0.010 and fine-tuning's own pair +0.002. No unread
  campaign was exported.

**Reading.** All pairs come from `scripts/campaign_pairs_report.py` (10,000 resamples), paired
over stays and seed by seed. Δ is fine-tuning less the network from nothing, in area. A side of
several pretrainings is their mean (`+`). Every rule needs both parts: the paired 95 % interval
lies above zero, *and* the mean over seeds exceeds the threshold.

- *Threshold.* The network from nothing has no pretraining, so the term for the spread
  between pretrainings comes from one side only. This is the registered rule for a side
  without pretraining:
  T_n = 2·√(SE² + (s²/2)/n). Here n is the number of pretrainings read, and SE is the
  standard error over seeds of the pooled difference. s is the larger of 0.014 and the spread
  between pretrainings that full fine-tuning's own four give at the budget. That spread uses
  the registered formula over the six pairs: the squared mean difference less its noise,
  averaged, floored at zero, then the root. From the trial's SE, T₂ ≈ 0.024 and T₄ ≈ 0.021.
- *Primary, confirmatory.* Δ₂₀₀ under seeds 3 and 4 pooled, against T₂.
- *Size.* Δ₂₀₀ under all four pooled, against T₄, set beside the registered least gain at
  200 stays, 0.050 (`docs/preregistration.md`, "Thresholds").
- *Few-shot.* Δ₂₀ and Δ₅₀ under all four pooled, each against its own T₄. The four
  pretrainings are fresh at these budgets. The two budgets form one family under Holm: the
  smaller bootstrap p must be below 0.025 and the larger below 0.05, each with its mean above
  T₄. These are the budgets the narrowed claim names. Their registered least gains are 0.055
  and 0.060.
- *The null pair.* The network from nothing does not read the backbone. At each budget, its
  six pairs between cards are read as a check, and each must lie within ±0.02. A card moved it
  by 0.016 at 200 stays (`intensive-care-curve.md`, 2026-10-05) and by 0.001 in the trial. If
  any pair falls outside, the reading stops until the difference is explained.
- *The classical bar.* Boosted trees per channel read the same half at 0.736 at 200 stays
  (`head-and-representation.md`, 2026-10-20), and fine-tuning reads about 0.68. No outcome
  here reaches that bar, and the note states this beside every level.

**Outcome levels.** Each level is defined by the rule it meets, not by the sign of Δ.

| Level | Condition | Meaning |
|---|---|---|
| 1. Not replicated | the primary fails, or it passes while Δ₂₀₀ under four does not pass T₄ | the only gain pretraining has shown does not hold under fresh pretrainings |
| 2. Real, small | the primary passes; Δ₂₀₀ under four passes T₄ but lies below 0.050 | pretraining gives the network a warmer start at 200 stays that labels overtake from 500 on; the registered endpoint is not met |
| 3. At the endpoint | as level 2, with Δ₂₀₀ under four at 0.050 or above | a gain of the size the endpoint was registered for, put forward to settle the backbone and recipe the transfer matrix is computed on |
| 4. Harm | Δ₂₀₀ under seeds 3 and 4 below −T₂ | recorded, and its mechanism named |

**Predictions** (from the intervals above, the lower edge for a gain).

1. Δ₂₀₀ under seeds 3 and 4 lies within [+0.005, +0.035]. The six earlier readings have lower
   edges from −0.001 to +0.027 and means from +0.012 to +0.036. The threshold is about 0.024,
   so whether the primary passes is close to even, and no level is predicted.
2. Δ₂₀₀ under all four lies below 0.050. No earlier interval reaches it: the highest upper
   edge is +0.047. Level 3 is not expected.
3. At 20 stays fine-tuning loses to the network from nothing: Δ₂₀ within [−0.04, 0.00]. On
   the validation side it lay at −0.026 [−0.037; −0.016] and −0.020 [−0.030; −0.010]. At 50
   stays Δ₅₀ lies within [−0.01, +0.03] and does not pass. It lay at +0.007 [−0.003; +0.017]
   and +0.016 [+0.006; +0.025]. Those readings were on another side, so this is a judgement.
4. Every null pair lies within ±0.02.
5. Full fine-tuning's own spread between the four pretrainings at 200 stays lies below 0.014,
   so the floor sets s. Seeds 1 and 2 lie 0.002 apart.

**What follows.**

- *Level 1:* the gain at 200 stays was the two pretrainings' draw. Pretraining helps the
  adapted network at no budget the instrument reads. The next step changes what the encoder
  learns, to a forecasting objective, because the masks' features are what the labels outgrow.
- *Level 2:* the gain is real but below the endpoint, and it vanishes by 500 stays. The next
  step is the same as for level 1. An ensemble of the four pretrainings may be read beside
  it, but only against an ensemble of four networks from nothing at the same cost.
- *Level 3:* the result is put forward as the backbone and recipe to settle on. The decision
  weighs it against the classical bar.
- *Few-shot:* a pass at 20 or 50 stays is the first reading of the narrowed claim under
  fine-tuning. Without a pass, the claim is not supported by fine-tuning at these budgets.

**Limitations.**

- One task, on half the tuning side. Seeds 1 and 2 at 200 stays are not independent of the
  hypothesis.
- Cells run on Kaggle T4 from two commits; the null pair checks the card, not every source of
  drift.
- 20 stays hold two or three deaths in a draw, so a seed's area moves by 0.03 to 0.04 there.
- s is floored at a value pooled over the probe, not over fine-tuning. Fine-tuning's own four
  set s only if they spread more than the floor.
- Every number in this section is a validation number in the sense of the registered test
  set: the test stays are not read.

## 2026-10-22 — Kaggle T4 and M1 Pro: fine-tuning at 200, 50 and 20 stays under four pretrainings, read by the rule

**Question.** Does full fine-tuning's gain over the network from nothing at 200 stays hold under
seeds 3 and 4, how large is it under all four pretrainings, and does fine-tuning gain at 20 or
50 stays?

**Conditions.** Orders placed from `545c29cd` and run on Kaggle T4 ×2, one order per GPU:
`reader-low-physionet2012.toml` under seed 3 (`d5871e20-…`, backbone `50b46711…`, 10,385 s)
and seed 4 (`00079504-…`, `5a45aac9…`, 10,700 s), 40 cells each, accepted. Reading on this
machine: `data/report/l30/pairs.sh` exports the eight campaigns and pairs them with
`scripts/campaign_pairs_report.py`; `verdict.py` beside it applies the rule declared above.
Files: `data/report/l30/pairs/`.

**The primary and the size, at 200 stays.** Δ is fine-tuning less the network from nothing, in
area; s = 0.014 (fine-tuning's own spread is 0.000 after its noise is removed, so the floor sets
s).

| Side | Δ | 95 % interval | Threshold | Rule |
|---|---|---|---|---|
| Seed 1 | +0.032 | [+0.018; +0.046] | T₁ 0.037 | (read before) |
| Seed 2 | +0.033 | [+0.019; +0.047] | T₁ 0.030 | (read before) |
| Seed 3 | +0.010 | [−0.005; +0.025] | T₁ 0.030 | |
| Seed 4 | +0.019 | [+0.006; +0.031] | T₁ 0.029 | |
| **Seeds 3 + 4 (primary)** | **+0.014** | [+0.003; +0.026] | T₂ **0.021** | **fails** |
| All four (size) | +0.024 | [+0.014; +0.033] | T₄ 0.017 | passes T₄; least gain 0.050 |

**Level 1, not replicated.** The interval of the primary lies above zero, but its mean does not
reach T₂. The size under four passes T₄, but by the declared table that does not count without
the primary.

**Few-shot, one family under Holm.** Fine-tuning's own spread is 0.010 at 20 stays (the floor
sets s) and 0.025 at 50 stays, where it sets s.

| Budget | Δ under four | 95 % interval | p | Holm α | T₄ | Rule |
|---|---|---|---|---|---|---|
| 20 | −0.017 | [−0.029; −0.004] | 0.010 | 0.025 | 0.029 | does not pass |
| 50 | −0.003 | [−0.015; +0.009] | 0.636 | 0.050 | 0.030 | does not pass |

At 20 stays fine-tuning loses to the network from nothing under every pretraining (−0.037,
−0.009, −0.005, −0.016). The loss is within its threshold, so it is not read as harm. At 50 stays
the pretrainings disagree (−0.008, −0.029, +0.024, +0.002). Two of the six pairs between them
differ by more than T₄ (0.049 and 0.031).

**Against the classical bar.** Area of the mean over four pretrainings and ten seeds, beside
boosted trees per channel on the same half (`head-and-representation.md`, 2026-10-20):

| Stays | From nothing | Full fine-tuning | Trees per channel |
|---|---|---|---|
| 20 | 0.599 | 0.582 | — |
| 50 | 0.622 | 0.619 | — |
| 200 | 0.659 | 0.683 | **0.736** |

**The null pair.** All 18 pairs lie within ±0.02. At 200 stays they lie from −0.002 to +0.014.
At 20 and 50 stays the network from nothing is the same network under every card: the same
seed draws the same weights and labels and never reads the backbone. Its scores agree to 10⁻⁵
at 20 stays, and to 10⁻³ on eight seeds of ten at 50. On the other two, a difference at the
device's precision grows through training to 0.03 and 0.08. The null pair therefore checks
repeatability there, not the card.

**Configuration checked before the reading.** The two new campaigns were defined under
`50b46711…` and `5a45aac9…`, as declared, and each holds its 40 cells; the export read 240
cells over the eight campaigns, every cell declared. `verdict.py` first
stopped at the pooled pair of seeds 3 and 4 at 20 and 50 stays, which `pairs.sh` builds and the
rule does not read. The check was narrowed to print those pairs as descriptive, with no change
to the rule.

**Conclusions.**

1. By the declared rule the gain at 200 stays is not replicated (level 1). Under the two
   pretrainings it was found on, it was +0.032 and +0.033; under the two never read it is +0.010
   and +0.019.
2. The best estimate of the gain at 200 stays is +0.024 [+0.014; +0.033] over four pretrainings.
   That is half the registered least gain, and it includes the two pretrainings that selected it.
3. Fine-tuning does not gain at 20 or 50 stays. At 20 stays it loses slightly under every
   pretraining; at 50 stays the choice of pretraining moves the result as much as pretraining
   does.
4. At 200 stays the network from nothing lies 0.077 below the trees, and fine-tuning closes 0.024
   of that gap. The network trails the classical bar by more than any gain pretraining has shown.

**Predictions.** 1 held (+0.014 within [+0.005, +0.035]); 2 held (+0.024 < 0.050); 3 held
(−0.017 within [−0.04, 0.00], and −0.003 within [−0.01, +0.03] without a pass); 4 held; 5
held (0.000 at 200 stays). At 50 stays, which it did not name, fine-tuning's own spread is
0.025.

**Limitations.**

- One task, on half the tuning side; every number is a validation number.
- At 20 and 50 stays the null pair checks repeatability, not the card (above).
- Seeds 3 and 4 at 200 stays ran from `8aced485`, seeds 1 and 2 from `08a764f5`. Between them
  the evaluation path differs only in a change that the declaration's test holds bit for bit.
- The trees were read at 200 and 500 stays only, so the bar at 20 and 50 stays is not known.
