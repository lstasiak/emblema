# Pretraining over cells

Whether the small encoder over cells of channel and time bin, pretrained without labels on
intensive-care stays, fine-tunes to a lower error at 20, 50 and 200 labelled stays than the same
shape trained from nothing; which of its recipes; and whether grids added to the stays help or
harm. The claim and its outcome levels are the register's, "Pretraining in the tasks' regime"
([preregistration](../preregistration.md)); the objective is
[ADR-0064](../adr/0064-a-pretext-over-cells-masked-before-the-layout.md).

## 2026-10-25 — declared before the run: ten pretrainings over cells read by fine-tuning at 20, 50 and 200 stays

**Questions.** (1) Does fine-tuning a backbone pretrained over cells on the 2012 and 2019 stays
gain over the small shape from nothing at 20, 50 and 200 stays of the half? (2) Do a forecast
tail, the count term or added grids move that gain? (3) Is the gain larger on the scored stays
the backbone read in pretraining? Beside them, descriptively: the probe at 500 stays, the pretext
on unread stays and the classical bar.

**Conditions.**

| | |
| --- | --- |
| Code | the commit that adds this section; pretrainings ordered from `841e30de` |
| Task | in-hospital death, `a6a9c653…`; half of the tuning side held out by seed 101 (1,997 stays learnt from, 2,000 scored); seeds 1 to 10; tier M; the validation side is not read |
| Pretrainings | the small shape (2 blocks × 64, 16 heads) over six-bin cells; half the 2012 stays of set A ranked by the run's seed, five passes an epoch, and every 2019 training stay; 19 epochs, read off a run at seed 0 by a rule written before it (`scripts/pretraining_budget_report.py`; `5655a49f`, read on no task) |
| Recipes | *base* `backbone-cells-clinical-m` (the pilot's masks, count weight 0.5); *tail* the same plus a forecast tail for half the windows; *no count* count weight 0; *tail, no count*; *grids* base plus C-MAPSS, SKAB and ESA-AD as sampled, the same exposure to the stays; each at seeds 1 and 2 (`-seed2`) |
| Platform | every pretraining on one Colab G4 (RTX PRO 6000) in six lanes under CUDA MPS, fp32; campaigns on the same card in lanes |
| Campaigns | `cell-pretraining-physionet2012` once per backbone (the small shape from nothing, fine-tuning at the rate in force and at 0.0001, 0.000333 and 0.003); `cell-pretraining-scratch-physionet2012` once (the network from nothing at the four rates); `cell-pretraining-probes-physionet2012` once per backbone (the closed-form probe over the blocks side by side, under the backbone and at its initialisation, at 500) |
| Worker | `data/report/t42c/worker-env.sh` (the rate in force 0.001), the backbone's weights per campaign |
| Classical bar | cells already read on the same half and seeds (`gap-to-the-trees.md`, 2026-10-23): trees per channel 0.621 / 0.628 / 0.736 at 20 / 50 / 200; the logistic head over the summaries 0.730 at 200; at 20 and 50, where that head was not read, the network over the same summaries, 0.644 / 0.674, a diagnostic and not a candidate |

The backbones and their weights:

| Recipe | Seed 1 | Seed 2 |
| --- | --- | --- |
| base | `ef2674dd` (`e9bf831a…`) | `e02dce87` (`b23be234…`) |
| tail | `8e9e5340` (`0b2a37e7…`) | `f7576057` (`4751c1aa…`) |
| no count | `eb1996e0` (`2f9f3fa8…`) | `8bb1dfbc` (`ac42481b…`) |
| tail, no count | `e60bdcf8` (`616d121e…`) | `f94c8154` (`f0ed74aa…`) |
| grids | `8b592bcc` (`8b079edd…`) | `34bc095b` (`f9f4851a…`) |

**What was seen before this section.** The ten pretrainings were run and accepted before it was
written, and none was read on a task. Their logs were read: every term of the pretext reached a
plateau except the count in one grids run, and the two seeds of a recipe agree within 0.011 in
every term on the stays. The trial ran every campaign at seed 1 under the budget run's backbone
and printed only the cells accepted (2; 5 and 4 a budget; 2,000 answers a side); the rate choice
and the transduction ran on them to files left unopened. The screen's rule, run on campaigns already read, repeats
+0.032 and +0.033 at 200 stays and T₁ of 0.037 and 0.030 (`few-label-fine-tuning.md`, 2026-10-22).

**Reading.** Pairs by `scripts/campaign_pairs_report.py` (10,000 resamples); a side of several
pretrainings is their mean, joined by `+`. The rule is `scripts/pretraining_screen_report.py`.

- *Spread.* s is the larger of 0.014 and fine-tuning's spread between pretrainings, by the
  registered formula over the five pairs that differ by seed alone, averaged over pairs and the
  three budgets.
- *Gains, screened.* Fine-tuning less the network from nothing, per recipe (n = 2) and over the
  eight clinical pretrainings (n = 8), against T = 2·√(SE² + (s²/2)/n), each given its level.
  Read descriptively: a screen confirms nothing.
- *Contrasts.* The tail (tail and tail-no-count against base and no-count, 4 against 4), the
  count (base and tail against no-count and tail-no-count) and the grids (grids against base,
  2 against 2), each against T = 2·√(SE² + (s²/2)·(1/n_a + 1/n_b)). The interaction is read
  descriptively.
- *Null pairs.* The network from nothing under the first campaign against each of the other
  nine at every budget, within ±0.02, or the reading stops.
- *Rates.* One rate per arm by the registered rule (`scripts/rate_choice_report.py`, 2,000
  resamples, seed 1): the network from nothing over its campaign; fine-tuning over all ten
  backbones together, since a rate chosen recipe by recipe would add the choice to its reading.
  Where several rates clear the bar, the largest gain is chosen.
- *Transduction.* `scripts/transduction_report.py` over the ten backbones: each one's gain on the
  scored stays it read and on those it never read (about 1,000 each), the mean gap per budget
  and its interval from one draw of stays shared by every backbone (2,000 resamples, seed 1).
  Descriptive; the interval holds the seeds as they ran.
- *Probe at 500.* The probe under the backbone less at its initialisation, per recipe, against
  the stage reader's threshold (s = 0.0057).
- *Pretext on unread stays.* `scripts/cell_pretext_audit.py --model trained --unread-units` and
  `--model initialisation` per backbone, the model beside `stacked`; 2019 on its validation side
  and the grids zero-shot for the clinical recipes (`--unset-rows mean`), descriptively.
- *Power for the confirmation.* `scripts/comparison_power_report.py` on these cells, at a gain of
  0.02 with this s, gives the pretrainings and seeds the confirming run needs.

**Predictions** (from earlier intervals, the lower edge where a gain is claimed).

| Reading | Prediction | Source |
| --- | --- | --- |
| Gain over eight, 20 and 50 | noise, inside [−0.029, +0.009] | fine-tuning under four pretrainings over raw tokens: −0.017 [−0.029; −0.004] at 20, −0.003 [−0.015; +0.009] at 50 (`few-label-fine-tuning.md`, 2026-10-22) |
| Gain over eight, 200 | inside [0.000, +0.033]: noise or real and small | +0.024 [+0.014; +0.033] there, against a network from nothing that read 0.666 on this half where the small shape reads 0.733 (`encoder-candidates.md`, 2026-10-23), so less is left to gain |
| Spread s | own spread at or below 0.025, so s in [0.014, 0.025] | fine-tuning's own spread between pretrainings 0.010 at 20 and 0.025 at 50 (`few-label-fine-tuning.md`) |
| Tail, count | within ±T at every budget | neither pretext variant replaced the mixture's masks by the probe at 50 (`pretext-variants.md`) |
| Grids | at or below zero, harm possible | the stays-alone backbone's probe gained up to 0.055 where the mixture's did not (`intensive-care-curve.md`) |
| Transduction | gap inside [−0.02, +0.02] at every budget | the pretext sees no label; continuing a backbone over both clinical corpora moved the probe at 500 by −0.000 [−0.007; +0.007] (`domain-continuation.md`) |
| Rates | none declared | no rate of either arm was read at 200 stays or below |
| Probe at 500 | within ±0.02 of the probe at initialisation | the probe gained nothing over an untrained encoder at 500 (`head-and-representation.md`) |
| Pretext, unread stays | the model above `stacked` on the presence (0.44) and on the mean (0.12); at initialisation below both | the audit of 2026-10-24 (`cell-pretext.md`) and the runs' validation, presence at 0.24 of the trivial cost (seen) |
| Classical bar | fine-tuning above the trees but below the network over the summaries at 50; within ±0.02 of the trees at 200 | the small shape from nothing 0.656 / 0.733 against the trees' 0.628 / 0.736 and the summaries' 0.674 at 50 on this half (`encoder-candidates.md`) |

**What follows.** The register's levels decide each gain. A factor moves off the base recipe for
the confirming run where its contrast exceeds T at 50 or 200 stays and lies below −T at none;
otherwise the base level stays. Grids join the confirming corpora by the same rule; otherwise the
stays alone. The 4.8M shape over cells is pretrained for the confirmation only if the gain over
eight is real and small or better at 50 or 200 stays. Fine-tuning's chosen rate is the one
confirmed. The gain on the unread stays is reported beside every gain; the confirming run reads
the 2019 stays in part as well, whatever the gap.

**Limitations.** The budget is a bound from above: the run that set it decayed its rate over 24
epochs, the recipes over 19. Under grids the kept epoch follows the mean validation over five
corpora, so it answers the grids in part. Two pretrainings a recipe give a coarse spread, bounded
below by the floor. The logistic head over the summaries was read at 200 and 500 stays only; at 20
and 50 the network over the same summaries stands in for it. The pretraining read about half the
scored stays.
