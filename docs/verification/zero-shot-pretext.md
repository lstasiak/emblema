# The pretext on a corpus the backbone never read

Whether anything a backbone learnt carries to a corpus it was not pretrained on. Masked
reconstruction is forecasting and imputation, so the pretext's own loss, read on a corpus the
backbone never saw against baselines fitted on that corpus alone, measures transfer of
structure without labels, without training and without a task. The backbones that left one
task corpus out (`backbone-scale-1e9-without-*`, 2026-10-07) make the reading possible; the
diagnostic of `pretext-variants.md` (2026-10-06) supplies the baselines; the reading of a
corpus through another backbone's windows is the one `state-rank.md` (2026-10-16) added.
Nothing here is a registered rule; the reading decides whether a few-shot forecasting task is
worth building.

## 2026-10-16 — declared before the run: the three backbones that left a corpus out, on it

**Question.** On the corpus it never read, does a backbone's pretext still beat the baseline
matched to each kind of mask — a line through the channel's own visible tokens for a block or a
single token, the cross-channel regression for a channel hidden whole — and if it does not,
is that because the rows its channels pass through were drawn at random, or because the
backbone carries nothing an untrained encoder does not?

**The data.**

- *The backbones.* `backbone-scale-1e9-without-physionet2019` (`4028de56-…`), `-without-
  physionet2012` (`ba2912dd-…`) and `-without-cmapss` (`12fd5139-…`), each read on the corpus
  it left out; `backbone-scale-1e9-m` (`803b82eb-…`), which read all three, on each; and the
  backbones of the task's own domain beside them, descriptively: on PhysioNet 2019 the corpus
  alone (`300d00dc-…`) and the two intensive-care corpora (`1fa0ef3a-…`); on PhysioNet 2012
  the mixture of four (`2aaca0ba-…`), the intensive-care pair and the stays four times over
  (`78c27106-…`); on C-MAPSS the mixture of four.
- *The reading.* `scripts/pretext_triviality_report.py --backbone ID --corpus NAME
  --windows-of 803b82eb-…`: the omitted corpus's windows are the ones the 10⁹ backbone read,
  2,000 fitted on and 2,000 scored, masks drawn from seed 1 under the backbone's own strategy,
  the baselines fitted under seed 2 on that corpus alone, 2,000 resamples over units. Nothing
  is normalised anew: the published block already is. Every backbone is read at one commit,
  the seen ones again (`data/report/l25/run.sh`).
- *The rows.* The channels of a corpus the backbone never read pass through rows of the
  channel table no pretraining set. Three readings of each omitted corpus: the rows drawn, as
  the evaluation draws them; the rows set to the mean of the learnt rows (`--unset-rows
  mean`), a channel the backbone could have seen without saying which; and the encoder at its
  initialisation, before a step of pretraining, with drawn rows (`--at-initialisation`), the
  floor of what a trained encoder can carry.
- *The trial, seen before this was written.* The three readings of the backbone without 2019
  on PhysioNet 2019 ran first, as the trial of the command (1:40 for the first, 24 s of it the
  reading; `data/report/l25/trial-*`). They are reported in the measured section as the
  trial, not as a test of the predictions, which are made for the other two corpora from
  them. Two checks of the path ran with them: the same backbone on C-MAPSS, which it read,
  through `--windows-of` repeats its own reading of 2026-10-07 to four places, and the 10⁹
  backbone on 2019 repeats its own.
- *What is read.* Per corpus and kind: the model's error, the matched baseline's, the excess
  (baseline less model) with its interval, the strongest linear answer, and the channel-mean
  baseline. A kind is *learnt* where the whole interval of the excess lies above zero.

**What the trial showed on PhysioNet 2019** (huber loss on hidden tokens, normalised units;
channel / block / token): rows drawn 0.432 / 0.410 / 0.382; rows at the mean 0.483 / 0.419 /
0.382; the initialisation 0.477 / 0.459 / 0.433; the matched baselines 0.288 / 0.226 / 0.157;
the channel mean 0.375 / 0.401 / 0.404; the 10⁹ backbone that read the corpus 0.271 / 0.162 /
0.125. Every kind is beaten zero-shot, by 0.14 to 0.23, the model sitting at the channel-mean
baseline, which is the prior: without a learnt identity for the channel it does not use the
channel's own course in time.

**Predictions**, for PhysioNet 2012 and C-MAPSS, from the trial and from the seen backbones'
intervals (`data/report/pretext/`, 2026-10-07: on 2012 the 10⁹ backbone's excess 0.015
[0.009; 0.021], 0.086 [0.081; 0.091], 0.032 [0.028; 0.036]; on C-MAPSS 0.027 [0.009; 0.052],
0.067 [0.064; 0.070], 0.059 [0.055; 0.063]).

1. *Every kind is beaten zero-shot on both corpora*, the excess interval under drawn rows
   lying wholly below zero, and the model's error within 0.10 of the channel-mean baseline
   (on the trial, within 0.06): the backbone predicts the prior where it has no identity.
2. *The rows at the mean of the learnt rows change no sign*: every kind stays beaten on both
   corpora, within 0.10 of the drawn reading. The failure is the missing identity, not the
   draw.
3. *The pretraining carries something, and little*: under drawn rows the zero-shot error lies
   below the initialisation's for every kind, by 0.02 to 0.08 (the trial: 0.045 to 0.051),
   and above the matched baseline by more than the seen backbone's whole excess on that
   corpus. What is carried is less than a tenth of the gap to the baseline.
4. *The backbones that read the corpus read it as before*: the seen readings at this commit
   repeat the ones of 2026-10-07 within their intervals.

**What follows.**

- If predictions 1 and 2 hold on both corpora: nothing the pretext learnt carries to a corpus
  it never read beyond the prior; the objective learns a course in time bound to the channels
  it saw. The few-shot forecasting task is not built on this backbone. The next readings are
  cheap and named: whether rows alone, fitted on the omitted corpus's unlabelled windows with
  the encoder frozen, restore the pretext below the baselines (then the encoder's machinery
  transfers and only identity is missing), and the objective itself.
- If prediction 1 fails on a corpus (a kind learnt zero-shot there), structure carries across
  corpora on that corpus's kind, and the forecasting task is read there first.
- If prediction 2 fails (the mean rows restore a kind), the draw is the failure, and the
  evaluation's grown rows are the first thing to change.
- A backbone that refuses the corpus, or a corpus whose windows the 10⁹ backbone did not
  read, is named and left out.

**Limitations.**

- One pretraining per backbone; the spread between pretrainings on the pretext loss is not
  read, so differences below about 0.005 in error are not read as differences.
- The drawn rows are one draw; prediction 2 reads whether the draw matters, not which draw.
- The excess is in the loss's units on hidden tokens; it says what the pretext carries, not
  what a task reads from it. A linear probe on a corpus never read (`layer-readout.md`) reads
  states, which are not these predictions.
- PhysioNet 2019's reading was seen before the predictions were written; the test is on the
  two other corpora.

## 2026-10-16 — M1 Pro, MPS: the three backbones that left a corpus out, on it, measured

**Question.** The one declared above.

**Conditions.** Commit `1163c3ca`, the declaration's commit plus one fix named below;
`data/report/l25/run.sh`: eighteen readings on this machine (MPS, fp32), 20 to 80 s each,
eleven minutes in all; CSV under `data/report/l25/<label>/`, the table by
`data/report/l25/reading.py`.

**Faults, named before the reading.**

- *The trial's controls were not controls.* The loop that ran the trial's "mean rows" and
  "initialisation" readings built their options from a string under a shell that does not
  split words, so the options never reached the command: the three trial readings on
  PhysioNet 2019 were three plain readings with rows drawn afresh each time. The
  declaration's paragraph "what the trial showed" reports them as controls, and prediction
  3's range (0.02 to 0.08 below the initialisation) was taken from them. The three draws stand
  as a reading of the draw's own effect (below); prediction 3's range is void and its
  direction is read descriptively.
- *The mean-row control set every row the manifest names.* A published manifest carries the
  whole vocabulary its chain had reached, so the control set the rows of every earlier corpus
  to the mean too. Fixed before the measured run (`1163c3ca`: only the corpus's own channels;
  the drawn rows from seed 1, which the declaration did not fix). The fix changed no number: a
  corpus's windows carry its own channels only, and the readings before and after it are
  equal to four places. The channels of PhysioNet 2019 lie past the table of the backbone
  without it and are drawn; those of PhysioNet 2012 and C-MAPSS lie within the tables of the
  backbones without them, rows set at the pretraining's own initialisation and never moved
  (no weight decay, no gradient), so those two readings do not depend on the seed.
- The run script's empty option list stopped it under bash 3.2 after the zero-shot readings;
  fixed, the seen readings ran in the second pass.

**The reading** (huber loss on hidden tokens, normalised units; the excess is the matched
baseline's error less the model's, 95 % interval over units).

| Corpus | Reading | Channel | Block | Token |
|---|---|---|---|---|
| 2019 | matched baseline / channel mean | 0.288 / 0.375 | 0.226 / 0.401 | 0.157 / 0.404 |
| 2019 | 10⁹, which read it | 0.271 (+0.017) | 0.162 (+0.063) | 0.125 (+0.032) |
| 2019 | 2019 alone / intensive-care pair | 0.250 / 0.249 | 0.151 / 0.151 | 0.114 / 0.114 |
| 2019 | **without 2019, rows drawn** | **0.469** (−0.181 [−0.192; −0.171]) | **0.404** (−0.178 [−0.189; −0.169]) | **0.359** (−0.202 [−0.212; −0.191]) |
| 2019 | without 2019, rows at the mean | 0.464 (−0.176) | 0.472 (−0.246) | 0.473 (−0.316) |
| 2019 | without 2019, at initialisation | 0.456 (−0.168) | 0.486 (−0.260) | 0.489 (−0.332) |
| 2019 | three unseeded draws of the rows | 0.432 to 0.483 | 0.410 to 0.459 | 0.382 to 0.433 |
| 2012 | matched baseline / channel mean | 0.258 / 0.361 | 0.228 / 0.350 | 0.145 / 0.371 |
| 2012 | 10⁹ / mixture of four | 0.243 (+0.015) / 0.239 | 0.142 (+0.086) / 0.144 | 0.113 (+0.032) / 0.122 |
| 2012 | intensive-care pair / stays ×4 | 0.222 / 0.222 | 0.130 / 0.134 | 0.103 / 0.107 |
| 2012 | **without 2012, rows as set** | **0.404** (−0.146 [−0.155; −0.138]) | **0.348** (−0.120 [−0.130; −0.111]) | **0.306** (−0.161 [−0.168; −0.154]) |
| 2012 | without 2012, rows at the mean | 0.374 (−0.116) | 0.352 (−0.123) | 0.355 (−0.210) |
| 2012 | without 2012, at initialisation | 0.471 (−0.214) | 0.464 (−0.236) | 0.487 (−0.343) |
| C-MAPSS | matched baseline / channel mean | 0.137 / 0.327 | 0.167 / 0.322 | 0.160 / 0.328 |
| C-MAPSS | 10⁹ / mixture of four | 0.110 (+0.027) / 0.105 | 0.100 (+0.067) / 0.099 | 0.101 (+0.059) / 0.100 |
| C-MAPSS | **without C-MAPSS, rows as set** | **0.344** (−0.207 [−0.234; −0.180]) | **0.332** (−0.165 [−0.189; −0.143]) | **0.330** (−0.170 [−0.193; −0.148]) |
| C-MAPSS | without C-MAPSS, rows at the mean | 0.294 (−0.157) | 0.290 (−0.123) | 0.290 (−0.131) |
| C-MAPSS | without C-MAPSS, at initialisation | 0.418 (−0.281) | 0.416 (−0.249) | 0.422 (−0.262) |

**The predictions.**

1. *Every kind beaten zero-shot on both corpora, within 0.10 of the channel mean*: **holds**.
   On 2012 the excess lies at −0.12 to −0.16, on C-MAPSS at −0.17 to −0.21, every interval
   wholly below zero; the model sits 0.00 to 0.07 from the channel-mean baseline (on 2012 the
   token kind 0.065 below it, on C-MAPSS 0.00 to 0.02 above). On 2019, the trial's corpus,
   the same: −0.18 to −0.20.
2. *The rows at the mean change no sign, within 0.10 of the drawn reading*: **holds in sign
   on every cell**; the size holds on 2012 and C-MAPSS (at most 0.05) and fails on 2019's
   block and token (0.07 and 0.11 worse than drawn: one row for every channel leaves the
   model unable to tell the corpus's channels apart). The failure is the identity, not the
   draw; on C-MAPSS the mean row reads 0.04 to 0.05 better than the rows as set.
3. *Carried something, and little*: the direction holds, the size does not. Zero-shot lies
   below the initialisation by 0.07 to 0.18 on 2012 and 0.07 to 0.09 on C-MAPSS (on 2019 by
   0.01, 0.08 and 0.13), not by 0.02 to 0.08; and the part carried is a fifth to a half of
   the untrained encoder's gap to the matched baseline (2012's token: 0.181 of 0.342), not
   under a tenth. What stands is the second half of the prediction: the zero-shot reading lies
   above the matched baseline by 0.12 to 0.21, more than the whole excess (0.015 to 0.086) of
   the backbone that read the corpus.
4. *The seen backbones read as before*: **holds**, every number equal to four places.

**Conclusions.**

- Nothing the pretext learnt reaches a corpus the backbone never read: on every corpus and
  every kind of mask the zero-shot reading loses to a line through the channel's own visible
  tokens, and to the cross-channel regression, by 0.12 to 0.21, where the backbone that read
  the corpus beats them by 0.02 to 0.09. The predicted value sits at the prior.
- The pretraining does carry something past an untrained encoder — a fifth to a half of the
  way to the baseline — so the encoder's machinery is not empty; what is missing is what the
  corpus's channels are. The row the channel passes through decides little: drawn, at the
  pretraining's initialisation, or at the mean of the learnt rows, the reading stays beaten
  by more than 0.1.
- On the task corpora the broad backbone's pretext is worse than the domain's own: 2019 alone
  and the intensive-care pair read 2019 by 0.01 to 0.02 less loss than 10⁹, and on 2012 the
  pair and the stays four times over by 0.01 to 0.02 less than 10⁹ and the mixture. Scale
  does not buy the task corpus a better pretext either.
- As declared for this outcome: the few-shot forecasting task is not built on this backbone.
  The next cheap reading is whether the rows alone, fitted on the omitted corpus's unlabelled
  windows with the encoder frozen, restore the pretext below the baselines; then the
  objective and the identity of channels across corpora.

**Limitations.**

- One pretraining per backbone, one seeded draw of the rows on 2019; the three unseeded draws
  span 0.05 in error, below every gap read here.
- PhysioNet 2019 was seen before the predictions were written, and the trial's controls were
  not what the declaration said they were; predictions 1 and 2 were tested on 2012 and
  C-MAPSS, prediction 3's size was not a test.
- The reading is of the pretext's predictions, not of the states; a linear probe reads a
  corpus never seen at 0.665 AUROC on 2012 (`state-rank.md`), so the states carry more than
  the predictions show.
