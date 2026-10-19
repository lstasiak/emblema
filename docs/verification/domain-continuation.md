# Continuing the pretraining on the task's corpora

Whether a broad base, continued for a short while on the unlabelled stays of the task's own
corpora, reads the task better than the base as it is and than the same corpora pretrained
alone. The base is the point of about 10⁹ observed values (`backbone-scale-1e9-m`); the
continuation is a registered pretraining that starts from its weights (ADR-0056). Its hard
baseline is the same continuation with the channel table alone free, since the rows of a corpus
fitted without labels already carry a part of the task (`task-transfer.md`, 2026-10-17). The
reader at 500 stays is the row of `docs/preregistration.md` registered for the stage; nothing
else here is a registered rule.

## 2026-10-18 — declared before the campaigns: the point continued over the intensive-care corpora, and the rows of C-MAPSS under the point without it

**Question.** Does the point of about 10⁹ values, continued over PhysioNet 2012 and 2019 with
every weight free, read the intensive-care task above the point as it is, above the same
continuation with the rows alone free, and above the two corpora pretrained alone? On FD001,
does the point without C-MAPSS read the task once C-MAPSS's rows are fitted?

**The backbones**, pretrained before this was written as the trial of the knobs: ordered, run
and accepted from `d6efffc5` on this machine's accelerator (MPS, fp32;
`data/report/l23b/pretrain.sh`, 2026-10-17 23:29 to 2026-10-18 02:08). Every loss fell through
every epoch, so each run kept its last. The validation masks are drawn from the run's seed, so
the column compares (A) with (B) and not with (A2); the pretext below reads all three alike.

| Backbone | Start | Trained | Corpora, epochs | Steps | Peak rate, schedule | Validation kept | Time |
|---|---|---|---|---|---|---|---|
| (A) `backbone-scale-1e9-continued-clinical-m`, `c8a32156-…`, `sha256:a976ca35…` | the point, `803b82eb-…` | everything | 2012 + 2019, 4 | 6,272 at 8 × 4 | 1e-4, warm-up 0.25 epoch, decay to 1 % | 0.1684 / 0.1960 | 2,330 s |
| (A2) `…-continued-clinical-m-seed2`, `d15da082-…`, `sha256:f3466a39…` | the point | everything | as (A), seed 2 | 6,272 | as (A) | 0.1721 / 0.1977 | 2,321 s |
| (B) `…-continued-clinical-rows-m`, `7ccc254a-…`, `sha256:1e632e33…` | the point | the channel table | as (A) | 6,272 | as (A) | 0.1705 / 0.1981 | 1,892 s |
| (G) `backbone-scale-1e9-without-cmapss-rows-m`, `f6193117-…`, `sha256:8689e61d…` | the point without C-MAPSS, (H) `12fd5139-…` | the channel table | C-MAPSS, a quarter, 12 | 900 at 16 × 4 | 1e-2, constant | 0.0822 | 2,959 s |

(A) adds four passes over corpora the base read eight times inside its mixture: two sevenths of
the clinical recipe's steps at its batch and the shape of its schedule, the peak ten times lower
since the base ended at a hundred-thousandth. (B) is (A) with the encoder frozen. The point and
the point read for four epochs (`fd1c4c88-…`) are the controls for the training added. (G) is
the fit of `task-transfer.md`.

**The pretext and the states, read before this was written** (`data/report/l23b/pretext-*`,
`spread-*`; the model's error on hidden tokens, channel / block / token):

| Corpus | the point | (A) | (A2) | (B) | matched baseline |
|---|---|---|---|---|---|
| 2012 | 0.243 / 0.142 / 0.113 | 0.240 / 0.141 / 0.111 | 0.240 / 0.140 / 0.111 | 0.242 / 0.142 / 0.113 | 0.258 / 0.228 / 0.145 |
| 2019 | 0.271 / 0.162 / 0.125 | 0.269 / 0.161 / 0.122 | 0.269 / 0.161 / 0.122 | 0.271 / 0.162 / 0.125 | 0.288 / 0.226 / 0.157 |
| C-MAPSS | 0.110 / 0.100 / 0.101 | (G) 0.114 / 0.102 / 0.103 | (H), rows at initialisation: 0.326 / 0.321 / 0.326 | | 0.137 / 0.167 / 0.160 |

Every kind stays learnt. (A) and (A2) lower the point's error by 0.002 to 0.003, a hundredth of
it; (B) repeats the point to 0.001; (G) repeats the diagnostic's fit (`zero-shot-pretext.md`,
2026-10-17). The states fill 7.7 and 7.8 directions of 256 on 2012's stays under (A) and (A2)
against the point's 8.3, and 5.8 and 6.2 on 2019's against 6.4; (B) repeats the point; the
clinical recipe fills 12.5 and 11.9; the four-epoch run, which gained +0.015 at 500 stays, 8.9
and 7.5, with its pretext error on the stays 0.001 to 0.002 above the point's. The continuation
narrowed the states below the point's and lowered the pretext; the training that gained before
did the opposite. Neither diagnostic has ordered backbones on the task.

**The reading.**

- *The reader of the stage*: `campaigns/reader-probes-physionet2012.toml` under (A), (A2) and
  (B) on this machine, `frozen_ridge@layer=concat` and `untrained_ridge@layer=concat` at 200,
  500, 1,000 and every stay of the half held out by seed 101, seeds 1 to 10. Beside them, read
  before with the same file: the point (`10eee25b-…`), the mixture's four and the clinical four
  (`layer-readout.md`, 2026-10-14; `pretext-variants.md`, 2026-10-15), the four-epoch run at
  500 (`5f277fdf-…`). (A) and (A2) are one side read as their mean, by the registered
  threshold with s the larger of 0.014 and the value the two give. Two pretrainings screen;
  nothing replaces.
- *Fine-tuning and the few-shot budgets*, descriptively, on the validation side:
  `campaigns/yardstick-low-physionet2012.toml` (20 and 50 stays) and
  `campaigns/yardstick-200-physionet2012.toml` under (A), (A2) and (B) on Kaggle T4s, seeds 1
  to 10, paired with the same files under the point (`59bf4ee0-…`, `86cb4026-…`), the four-epoch
  run (`98f13d13-…`, `1b1eb43c-…`) and the mixture's seeds (`7101cdda-…`, `4785cac8-…`;
  `ced85fec-…`, `30ba06fe-…`).
- *FD001 at 50 windows*: `campaigns/yardstick-50-fd001.toml` under (G) and (H), paired over
  engines with the point (`36546f32-…`), the four-epoch run (`11baf6ed-…`) and the mixture's
  seeds (`ba0afd72-…`, `37e58082-…`); under (A), (A2) and (B) too, for what four epochs on the
  clinic cost the engines.
- *The initialisation.* (A), (A2) and (B) share the point's vocabulary of 329, so every untrained
  cell under them must equal the point's to the bit; any other outcome stops the reading. (G)'s
  table has 126 rows and (H)'s the point's less C-MAPSS: on FD001 the probe and fine-tuning are
  compared across the two, the initialisation within each alone.

**What is known at 500 stays** (area under the ROC curve on the half; `layer-readout.md`,
`pretext-variants.md`, 2026-10-14 and 2026-10-15): the point reads 0.673, below each of the
mixture's four by 0.019 to 0.036; the four-epoch run gains +0.015 ± 0.004 on it and stays
−0.010 ± 0.004 under the mixture's four; the clinical four read 0.693 to 0.729, +0.015
[+0.005; +0.025] over the mixture's four; the initialisation reads 0.680 over the point's
vocabulary. The yardsticks' values under the point, the four-epoch run and the mixture's seeds
stand in `intensive-care-curve.md` (2026-10-07 and 2026-10-15) and in the campaigns paired
above; the predictions quote the ones they need.

**Predictions**, from the intervals read before, the lower edge for a gain; a judgement says so.

1. *The initialisation.* Every untrained cell under (A), (A2) and (B) equals the point's to the
   bit, in every campaign.
2. *The continuation gains on the point* (the main reading, 500 stays). The mean of (A) and (A2)
   less the point lies at or above +0.007, the lower edge of four epochs less two, and below
   +0.036, the point's largest deficit to the mixture. The sign from the four-epoch pair, the
   size a judgement.
3. *The rows alone do not move the point.* (B) less the point lies within ±0.014 (s) at 500
   stays; within ±0.01 as a judgement from a pretext and states that repeat the point.
4. *The base plus the clinic does not reach the clinic alone.* The mean of (A) and (A2) less the
   clinical four lies below zero at 500 stays, and less the mixture's four within −0.03 and
   +0.01, under the threshold either way. A judgement from the pretext and the states.
5. *The yardsticks.* The network from nothing under (A), (A2) and (B) repeats the point's within
   ±0.013 at 20 and 50 stays (its interval under the point, 2026-10-07) and ±0.016 at 200 (the
   card's). Fine-tuning at 200 under (A) and (A2) lies within ±0.016 of the point's 0.671; the
   probe at 50 and 200 above the point's by 0 to +0.038 (the four-epoch gain). Nothing on the
   validation side decides anything.
6. *FD001.* The probe under (G) lies below (H)'s in RMSE by the interval, and within ±2.0 of the
   point's 22.6 (the seeds' spread); fine-tuning under (G) within ±3.0 of the point's 24.6; the
   network from nothing under (G) between 16 and 17. Under (A) and (A2) the probe and
   fine-tuning on FD001 lie within ±2.0 of the point's.

**What follows.**

- If 2 and 4 hold: the base gains from its own domain and still reads under the domain alone;
  the continuation is a cheap step, not a thesis, and the work turns to the pretext itself. The
  register pretrains a screened recipe to four when its mean gain lies above zero; this recipe's
  own corpora are the clinic, so the gain declared here is over the clinical four.
- If 4 fails, (A) and (A2) above the clinical four: the first reading of a base that carries
  something beyond the domain; pretrained to four and read by the rule against both fours.
- If 2 fails: the rate is a choice, not a result; a second peak, 1e-3 under the same schedule,
  is pretrained once and read before anything is concluded.
- If 3 fails, (B) away from the point by more than s: the schedule moved rows the base knew;
  (A) less (B) is read beside (A) less the point before either is named.
- If 6 fails on (G) against (H): the rows carry the pretext on the engines and not a linear
  reading of the remaining life.
- A cell that refuses, or a campaign that does not run whole, is named and its backbone left
  out.

**Limitations.**

- The pretrainings preceded the declaration as the trial of the knobs; their pretext and states
  are reported, not predicted. Two pretrainings screen and cannot replace.
- The peak rate is a choice; one other rate is named and no sweep is run. The continuations ran
  in fp32 on this machine, the base and the clinical recipe in fp16 on CUDA; the difference is
  below s.
- The yardsticks read the validation side again, descriptively. FD001 is read at its registered
  rung of 50 windows, where the arms differ most; from 200 on the network from nothing matches
  them (`label-efficiency-curve.md`, 2026-09-22). Under every backbone of about 10⁹ values the
  probe at 50 reads below the initialisation, so the rows are read for what they move, not for
  whether the base reads the engines.
- Comparisons to the initialisation hold within a backbone alone: (G), (H) and the point draw
  it over three vocabularies.

## 2026-10-19 — declared before the run: the continuation at the clinical recipe's peak

**Question.** The continuation at a peak of 1e-4 left the reader where the point stands: (A) and
(A2) less the point read −0.0003 [−0.007; +0.007] at 500 stays, prediction 2 of the section
above failing (its measured section follows the yardsticks). Did the rate hold the base in
place? The rule declared for that outcome is run: one pretraining at 1e-3, the clinical
recipe's own peak, read before anything is concluded.

**Conditions.** `experiments/backbone-scale-1e9-continued-clinical-high-rate-m.toml`, (A) to the
step but the peak, seed 1, so its masks and its order of stays are (A)'s; pretrained on this
machine's accelerator (MPS, fp32) and read with `campaigns/reader-probes-physionet2012.toml` as
(A) was. One pretraining: a judgement, not a screening.

**Predictions.**

1. Every untrained cell equals the point's to the bit.
2. The high rate less the point lies within −0.014 and +0.039 at 500 stays: from the point's
   threshold below to the clinical four's lead over it (0.712 against 0.673) above. A judgement.
3. The high rate less (A) lies above zero at 500 stays: a rate that moves the weights more moves
   the reader towards the clinic. A judgement.

**What follows.** If the high rate less the point exceeds the registered threshold for one
pretraining against one, 2·√(SE² + s²) with s = 0.014, about 0.03, a second seed is pretrained and the two are screened by the registered
rule. If it does not, the continuation over the task's corpora is closed at both rates: the
domain does not move the reader of this base, and the work turns to the pretext.

## 2026-10-19 — measured: both peaks on the reader, the yardsticks and FD001

**Question.** The two declared above, read together.

**Conditions.** Orders from `b1047304` (the first declaration, `12ebc03d`, with the register's
row) and `dff7a45d` (the second); every campaign ran whole and was accepted. The reader's four
campaigns on this machine (MPS, fp32; 2026-10-18 18:43 to 21:25 and 2026-10-19 01:19 to
02:14, 53 min each). The yardsticks under (A) and (A2) in one Kaggle session of two T4s, cut
by its twelve hours with two orders part-way; the other seven on one Colab G4 in five lanes
under CUDA MPS, 2026-10-19 13:05 to 14:27, those two resumed from the stored results. The
high rate (`61f82c53-…`, `sha256:9965fe48…`) ran 2,865 s here, kept its last epoch (0.1624 /
0.1913), reads the pretext 0.005 to 0.011 below the point on both corpora, four times (A)'s
move, and fills 7.0 directions on 2012 (the point 8.3, (A) 7.7) and 6.7 on 2019 (6.4, 5.8).
Pairs by `scripts/campaign_pairs_report.py` (`data/report/l23b/pairs.sh`, `pairs-high.sh`),
paired over stays or engines, seeds 1 to 10, 2,000 resamples; the yardsticks with the floors
of 0.035 and 0.03.

**The reader** (area under the ROC curve on the half, `frozen_ridge@layer=concat`; the
difference, candidate less control, with its interval).

| Candidate less control | 200 | 500 | 1,000 | all |
|---|---|---|---|---|
| (A)+(A2) − the point | +0.003 [−0.004; +0.009] | −0.000 [−0.007; +0.007] | +0.006 [−0.005; +0.016] | +0.012 [−0.001; +0.025] |
| (A)+(A2) − (B) | +0.003 [−0.003; +0.010] | +0.001 [−0.006; +0.008] | +0.006 [−0.004; +0.017] | +0.013 [+0.000; +0.026] |
| (A)+(A2) − the mixture's four | −0.015 [−0.029; −0.002] | −0.025 [−0.039; −0.011] | −0.024 [−0.042; −0.006] | −0.021 [−0.041; +0.000] |
| (A)+(A2) − the clinical four | −0.026 [−0.042; −0.010] | −0.039 [−0.055; −0.023] | −0.043 [−0.062; −0.024] | −0.041 [−0.063; −0.019] |
| (B) − the point | −0.001 [−0.002; −0.000] | −0.001 [−0.002; +0.000] | −0.001 [−0.002; +0.001] | −0.001 [−0.003; +0.001] |
| the high rate − the point | +0.018 [+0.005; +0.031] | **+0.023 [+0.007; +0.038]** | +0.035 [+0.015; +0.055] | +0.034 [+0.011; +0.057] |
| the high rate − (A) | +0.011 [+0.000; +0.021] | +0.023 [+0.009; +0.037] | +0.025 [+0.006; +0.044] | +0.017 [−0.007; +0.040] |
| the high rate − the mixture's four | −0.000 [−0.014; +0.013] | −0.002 [−0.016; +0.013] | +0.006 [−0.012; +0.024] | +0.001 [−0.019; +0.021] |
| the high rate − the clinical four | −0.010 [−0.026; +0.005] | −0.016 [−0.032; −0.001] | −0.013 [−0.032; +0.006] | −0.019 [−0.042; +0.003] |
| the probe − the initialisation, under (A) | +0.018 [−0.007; +0.042] | −0.008 [−0.029; +0.014] | −0.044 [−0.067; −0.021] | −0.047 [−0.073; −0.021] |
| the probe − the initialisation, under the high rate | +0.028 [+0.006; +0.051] | +0.015 [−0.005; +0.035] | −0.020 [−0.042; +0.003] | −0.031 [−0.055; −0.005] |

(A) less (A2) at 500 is +0.000 [−0.008; +0.008]; (A)+(A2) less the four-epoch run −0.015
[−0.035; +0.005], the high rate less it +0.008 [−0.013; +0.028]. The reader reads 0.673 under
the point, (A), (A2) and (B) alike, 0.696 under the high rate, 0.697 under the mixture's four
and 0.712 under the clinical four. The threshold for the high rate against the point, one
pretraining against one, is 2·√(SE² + s²) = 0.032 with SE 0.008.

**The yardsticks** (validation side, the mean of ten seeds; fine-tuning / the probe; the
network from nothing reads 0.585 to 0.591, 0.602 to 0.607 and 0.657 to 0.669 under every
backbone of the point's vocabulary, the initialisation's probe 0.524, 0.586 and 0.622).

| Backbone | 20 stays | 50 stays | 200 stays |
|---|---|---|---|
| the point | 0.568 / 0.524 | 0.605 / 0.564 | 0.671 / 0.596 |
| the four-epoch run | 0.555 / 0.526 | 0.600 / 0.602 | 0.679 / 0.635 |
| the mixture's seeds 1, 2 | 0.565, 0.571 / 0.533, 0.520 | 0.620, 0.626 / 0.591, 0.595 | 0.700, 0.678 / 0.639, 0.661 |
| (A) | 0.567 / 0.514 | 0.616 / 0.580 | 0.664 / 0.606 |
| (A2) | 0.566 / 0.529 | 0.606 / 0.575 | 0.656 / 0.603 |
| (B) | 0.562 / 0.524 | 0.608 / 0.561 | 0.666 / 0.596 |

Paired with the point, fine-tuning under (A) moves −0.001, +0.011 [+0.003; +0.019] and −0.007
[−0.015; +0.000] at 20, 50 and 200 stays; under (A2) −0.002, +0.001 and −0.015 [−0.023;
−0.007]; under (B) −0.007, +0.003 and −0.006. At 200 it lies below the mixture's seeds by
0.014 to 0.043. The probe under (A) lies above the point's by +0.016 [+0.010; +0.022] at 50
and +0.009 at 200, below it by 0.010 at 20; (B)'s repeats the point's within 0.003.

**FD001 at 50 windows** (RMSE on the validation engines, the mean of ten seeds; the network
from nothing / the probe / fine-tuning).

| Backbone | from nothing / probe / fine-tuning |
|---|---|
| the point | 16.8 / 22.6 / 24.6 |
| the four-epoch run; the mixture's seeds 1, 2 | — / 24.0; 25.5, 23.5 / 24.0; 22.4, 20.9 |
| (A), (A2), (B) | 16.8, 17.1, 16.6 / 22.7, 22.2, 22.6 / 24.8, 24.2, 24.8 |
| (G), the point without C-MAPSS with its rows fitted | 16.7 / 22.6 / 23.1 |
| (H), the point without C-MAPSS, rows drawn | 17.6 / 22.3 / 23.9 |

The initialisation's probe reads 20.5 over the point's vocabulary and 19.3 over (G)'s. (G) less
(H): the probe 0.34 worse [−0.70; +1.37], fine-tuning 0.85 better [−0.25; +2.02]. (G) less the
point: the probe −0.01 [−0.71; +0.67], fine-tuning 1.55 better [+0.84; +2.35]. (B)'s probe on
FD001 equals the point's to the bit, since (B) moved only the clinic's rows.

**The predictions of 2026-10-18.**

1. *The initialisation to the bit*: **fails in the letter.** Under (A) and (A2) on the Kaggle
   T4s, 10 of the 30 untrained cells differ from the point's by 2.6 × 10⁻⁷, the same cells by
   the same amount under both; under (B) on a G4, where the point's own ran, all 30 equal it
   to the bit, as does every reader and FD001 cell. The device rounds, the configuration is
   the same; the reading went on by that judgement.
2. *The continuation gains on the point*: **fails.** −0.000 [−0.007; +0.007] at 500 stays,
   below +0.007; nor at any other budget.
3. *The rows alone do not move the point*: **holds.** −0.001 at every budget.
4. *The base plus the clinic does not reach the clinic alone*: **holds.** −0.039 to the
   clinical four; −0.025 [−0.039; −0.011] to the mixture's four, inside −0.03 to +0.01.
5. *The yardsticks*: **holds**: the network from nothing within 0.007 of the point's,
   fine-tuning at 200 within 0.015 ((A2) at the edge), the probe at 50 and 200 above the
   point's by 0.006 to 0.016.
6. *FD001*: **fails on (G) against (H)**, the probe 0.34 worse where it was to be better by
   the interval; every other edge holds.

**The predictions of 2026-10-19.** 1 **holds** (40 of 40 cells). 2 **holds**: +0.023 inside
−0.014 to +0.039. 3 **holds**: +0.023 [+0.009; +0.037] over (A). The gain stays under the
threshold of 0.032: by the rule declared, no second seed.

**Conclusions.**

- Four epochs over the task's corpora at a tenth of the recipe's peak leave the reader at the
  point, at every budget, as do the same epochs with the encoder frozen: the pretext fell by
  a hundredth and the states narrowed, and neither moved the probe.
- At the recipe's own peak the continuation gains +0.023 [+0.007; +0.038] at 500 stays,
  reaches the mixture's four (−0.002) and stays below the clinical four (−0.016 [−0.032;
  −0.001]), under the threshold for one pretraining. The continuation is closed at both
  rates: the base carries nothing the reader sees beyond the domain, and the work turns to
  the pretext.
- On the validation side fine-tuning under the continuation lies within 0.015 of the point's
  at every budget and below the mixture's seeds at 200; the probe at 50 gains 0.011 to 0.016.
- On FD001 the rows of C-MAPSS fitted without labels read the probe as the point does and no
  better than rows drawn; fine-tuning gains 1.5 RMSE on the point and 0.9 [−0.3; +2.0] on
  the drawn rows. The rows carry the pretext on the engines, not a linear reading of the
  remaining life; four epochs on the clinic cost the engines nothing.
- The high rate's probe reads above the initialisation at 200 stays, +0.028 [+0.006; +0.051],
  where every backbone of about 10⁹ values read before lay at or below it; from 1,000 it lies
  below as the others do.

**Limitations.**

- Two pretrainings at 1e-4 and one at 1e-3; the rule declared stopped at one. The rates are
  two choices, no sweep.
- Prediction 1's letter was not met on the T4s, its substance was; the next declaration of
  its kind says "to the device's precision".
- The yardsticks and FD001 read the validation side; their comparators are not a rule.
