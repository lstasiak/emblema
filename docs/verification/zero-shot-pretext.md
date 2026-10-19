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
   below the initialisation by 0.07 to 0.18 on 2012 and 0.07 to 0.09 on C-MAPSS (on 2019 the
   block and token by 0.08 and 0.13, the channel 0.01 above it), not by 0.02 to 0.08; and the
   part carried is up to a half of the untrained encoder's gap to the matched baseline (2012's
   token: 0.181 of 0.342; a quarter to a half elsewhere, nothing on 2019's channel kind), not
   under a tenth. What stands is the second half of the prediction: the zero-shot reading lies
   above the matched baseline by 0.12 to 0.21, more than the whole excess (0.015 to 0.086) of
   the backbone that read the corpus.
4. *The seen backbones read as before*: **holds**, every number equal to four places.

**Conclusions.**

- Nothing the pretext learnt reaches a corpus the backbone never read: on every corpus and
  every kind of mask the zero-shot reading loses to a line through the channel's own visible
  tokens, and to the cross-channel regression, by 0.12 to 0.21, where the backbone that read
  the corpus beats them by 0.02 to 0.09. The predicted value sits at the prior.
- The pretraining does carry something past an untrained encoder — a quarter to a half of
  the way to the baseline, except for a channel hidden whole on 2019, where nothing — so the encoder's machinery is not empty; what is missing is what the
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

## 2026-10-16 — declared before the run: the rows alone, fitted on the corpus never read

**Question.** The reading above left the backbone's machinery and the identity of the
corpus's channels entangled: the pretext failed zero-shot, but the pretraining carried up to
half of an untrained encoder's gap. If the rows of the corpus's own channels, and nothing
else, are fitted on the corpus's unlabelled training windows under the pretext, does the
backbone that never read the corpus reach the baselines, and the backbone that read it? And
does the encoder at its initialisation, given the same fit, not?

**The data.**

- *The fit.* `--fit-rows 1000`: Adam at 0.01 over the two channel tables' weights with the
  gradient of every row outside the corpus's own channels zeroed before each step, every
  other weight frozen; batches of 64 of the 2,000 training windows in an order seed 2
  decides, the strategy's masks drawn per step from the same seed, the loss on hidden
  tokens as the pretraining scores it. The losses of the steps go to `fit.csv`. The reading
  afterwards is the one of the section above, on the same 2,000 validation windows.
- *The backbones.* The three that left a corpus out, each on it, through the 10⁹ backbone's
  windows; beside each, the encoder at its initialisation (`--at-initialisation`) given the
  same fit, the control that tells the machinery from the rows. The seen readings of the
  section above are the comparison and are not run again. `data/report/l25c/run.sh`.
- *The trial, seen before this was written*, from `data/report/l25c/trial.sh` and
  `trial2.sh`, their output directories listed and their settings checked for the options:
  the backbone without 2019 on PhysioNet 2019, rows fitted 300 steps (4:02) and 1,000 steps
  (9:05), and the initialisation fitted 1,000 steps (9:21). The predictions are made for the
  other two corpora from it.
- *What is read.* Per corpus and kind as above; the fit's loss over its steps; a kind is
  *learnt* where the whole interval of the excess lies above zero.

**What the trial showed on PhysioNet 2019** (channel / block / token; the matched baselines
0.288 / 0.226 / 0.157, the 10⁹ backbone that read it 0.271 / 0.162 / 0.125, the same backbone
zero-shot 0.469 / 0.404 / 0.359): rows fitted 300 steps 0.302 / 0.194 / 0.161; 1,000 steps
0.292 / 0.180 / 0.145 — the block and the token learnt (+0.046 [+0.041; +0.051] and +0.012
[+0.007; +0.017]), the channel matched (−0.005 [−0.009; +0.000]), every kind beyond the
strongest linear answer or matched to it, 0.02 from the backbone that read the corpus. The
initialisation fitted 1,000 steps: 0.371 / 0.396 / 0.400, beaten by 0.08 to 0.24. The fit's
loss on 2019 fell from 0.27 over the first 200 steps to 0.217 over steps 600 to 1,000, and
the initialisation's from 0.40 to 0.38.

**Predictions**, for PhysioNet 2012 and C-MAPSS, from the trial and the section above (2012:
zero-shot 0.404 / 0.348 / 0.306, the backbone that read it 0.243 / 0.142 / 0.113, baselines
0.258 / 0.228 / 0.145; C-MAPSS: zero-shot 0.344 / 0.332 / 0.330, read 0.110 / 0.100 / 0.101,
baselines 0.137 / 0.167 / 0.160).

1. *The block and the token are learnt after the fit on both corpora*: the excess over
   interpolation has its whole interval above zero (2019: +0.046 and +0.012).
2. *The fitted backbone reads within 0.03 of the backbone that read the corpus on the block
   and the token* (2019: 0.018 and 0.020 behind), *and within 0.05 on a channel hidden whole*
   (2019: 0.021).
3. *The encoder at its initialisation, given the same fit, stays beaten on every kind by more
   than 0.05* (2019: 0.08 to 0.24): the rows alone do not carry it, the machinery does.
4. *The fit converges within the steps given*: the mean loss over the last 400 steps lies
   within 0.01 of the mean over steps 400 to 600 (2019: 0.217 and 0.219).

**What follows.**

- If predictions 1 to 3 hold on both corpora: the encoder's machinery transfers to a corpus it
  never read once the corpus's channels have identities fitted without labels — the first
  reading on this backbone where a corpus never seen is read better than its own baselines.
  The few-shot forecasting task reopens with the rows' fit as its adaptation, and the identity
  of channels across corpora becomes the direction: what the fitted rows are, against the
  rows the backbone learnt for like channels, is the next reading. For the evaluation, a
  frozen probe on a corpus the backbone never read should fit its grown rows the same way
  first; whether that moves the probe at 500 stays is a declared reading of its own.
- If prediction 1 or 2 fails on a corpus: what transfers is bound to that corpus's kind of
  structure, and the claim narrows to the corpora on which it holds.
- If prediction 3 fails (the initialisation reaches the baselines too): the rows alone do
  it, and nothing is claimed of the machinery.
- If prediction 4 fails: the steps were too few, the reading stands as a lower bound, and
  the fit is run longer before anything is claimed about the gap to the seen backbone.

**Limitations.**

- One seed of the fit, one rate and one number of steps; the trial chose them on 2019.
- The fit sees the corpus's training windows without labels, as unsupervised adaptation in
  the target domain does; it is not zero-shot, and the section above is.
- PhysioNet 2019 was seen before the predictions were written; the test is on the two other
  corpora.

## 2026-10-17 — M1 Pro, MPS: the rows alone, fitted on the corpus never read, measured

**Question.** The one declared above: with the rows of the corpus's own channels, and nothing
else, fitted on its unlabelled training windows, does the backbone that never read the corpus
reach the baselines and the backbone that read it, and does the encoder at its initialisation,
given the same fit, not?

**Conditions.** Commits `5617203f` (PhysioNet 2019) and `5b8dcb51` (the rest); MacBook Pro
M1 Pro, MPS, fp32; `data/report/l25c/run.sh`, output under `data/report/l25c/`. The fit as
declared: 1,000 steps of Adam at 0.01 on batches of 64 of the 2,000 training windows, the
other weights frozen; 9 minutes a fit on 2019, 29 to 31 on 2012, 49 to 56 on C-MAPSS. The
reading afterwards on the same 2,000 validation windows as the section above. The script of
these commits fitted with the model in training mode; every experiment sets a dropout of 0.0,
so that is the evaluation pass the script fits in since the review that followed the run.

*Fault, named.* The first fit on 2012 stopped at its first step: the backward pass over a batch
of 64 windows asked MPS for 40 GiB (`loco2012-on-2012-fit.oom.log`). A window of 2012 holds
about 2.6 times the tokens of one of 2019 and a window of C-MAPSS 6.4 times, and attention's
activations grow with the square. From `5b8dcb51` the script scores each batch a few windows
at a time under the masks drawn for the whole batch, each part's summed error divided once by
the batch's count of hidden tokens, so the step's gradient and loss are the whole batch's up to
the order of a floating-point sum (`--fit-chunk`; 8 windows on 2012, 4 on C-MAPSS; a test on
CPU holds the chunked fit to the whole one within 10⁻⁶). The check: the fit on 2019 repeated in
chunks of 8 reads 0.292 / 0.180 / 0.145 against the whole batch's 0.292 / 0.180 / 0.145, and
its loss over the first 200, steps 400 to 600 and the last 400 is 0.271 / 0.219 / 0.217 in
both. The 2019 errors of the run reproduce the trial's to three places; the interval of the
channel's excess moves by 0.001 with the resamples.

**The reading** (channel hidden whole / block / token; the model's error, and in brackets the
matched baseline's less the model's, with its 95 % interval over 2,000 resamples of units; a
kind is learnt when the interval lies above zero).

| Corpus | Zero-shot, rows drawn | Rows fitted 1,000 steps | The backbone that read it | At initialisation, rows fitted |
|---|---|---|---|---|
| 2019, channel | 0.469 (−0.181) | 0.292 (−0.004 [−0.008; +0.001]) | 0.271 (+0.017) | 0.370 (−0.083) |
| 2019, block | 0.404 (−0.178) | **0.180 (+0.046 [+0.041; +0.051])** | 0.162 (+0.063) | 0.396 (−0.171) |
| 2019, token | 0.359 (−0.202) | **0.145 (+0.012 [+0.007; +0.017])** | 0.125 (+0.032) | 0.400 (−0.242) |
| 2012, channel | 0.404 (−0.146) | **0.248 (+0.010 [+0.004; +0.017])** | 0.243 (+0.015) | 0.354 (−0.096) |
| 2012, block | 0.348 (−0.120) | **0.145 (+0.083 [+0.078; +0.088])** | 0.142 (+0.086) | 0.341 (−0.113) |
| 2012, token | 0.306 (−0.161) | **0.119 (+0.026 [+0.022; +0.030])** | 0.113 (+0.032) | 0.365 (−0.220) |
| C-MAPSS, channel | 0.344 (−0.207) | **0.114 (+0.022 [+0.005; +0.048])** | 0.110 (+0.027) | 0.326 (−0.189) |
| C-MAPSS, block | 0.332 (−0.165) | **0.102 (+0.065 [+0.062; +0.068])** | 0.100 (+0.067) | 0.321 (−0.154) |
| C-MAPSS, token | 0.330 (−0.170) | **0.103 (+0.057 [+0.053; +0.061])** | 0.101 (+0.059) | 0.326 (−0.166) |

The matched baselines: 2019 0.288 / 0.226 / 0.157, 2012 0.258 / 0.228 / 0.145, C-MAPSS
0.137 / 0.167 / 0.160.

**The fit's loss**, mean over the first 200 steps, steps 400 to 600 and the last 400.

| Fit | 2019 | 2012 | C-MAPSS |
|---|---|---|---|
| the backbone that left the corpus out | 0.271 / 0.219 / 0.217 | 0.228 / 0.183 / 0.178 | 0.124 / 0.081 / 0.080 |
| the encoder at its initialisation | 0.402 / 0.387 / 0.384 | 0.363 / 0.346 / 0.342 | 0.249 / 0.239 / 0.236 |

**Predictions.**

1. *Block and token learnt on both corpora.* Holds: 2012 +0.083 and +0.026, C-MAPSS +0.065
   and +0.057, every interval above zero. On both the channel hidden whole is learnt too
   (+0.010 [+0.004; +0.017] and +0.022 [+0.005; +0.048]), which 2019 did not give.
2. *Within 0.03 of the backbone that read the corpus on block and token, 0.05 on the
   channel.* Holds, by a wide margin: 0.003 to 0.006 behind on 2012, 0.002 to 0.004 on
   C-MAPSS, against 0.018 to 0.021 on 2019.
3. *The initialisation, given the same fit, stays beaten by more than 0.05.* Holds: by 0.11
   to 0.25 on 2012 and 0.21 to 0.22 on C-MAPSS. On C-MAPSS the fit moves the initialisation
   by 0.09 to 0.10, from 0.42 to 0.32, and leaves it at the channel-mean baseline.
4. *Convergence.* Holds on every fit: the last 400 steps lie 0.001 to 0.005 below steps
   400 to 600.

**Conclusions.**

- The encoder's machinery transfers to a corpus it never read. Once the corpus's channels
  have identities fitted without labels, the backbone that left the corpus out reads it
  within 0.006 of the backbone that read it on 2012 and C-MAPSS, and beats every baseline
  there; on 2019 it stays 0.02 behind and matches the cross-channel baseline on a channel
  hidden whole. The rows that no pretraining set were what failed zero-shot.
- The rows alone do not do it: the same fit on the encoder at its initialisation stays 0.08
  to 0.25 behind, at or below the channel-mean baseline on every kind.
- The fit is cheap and converges: the corpus's own rows (39 on 2019), 1,000 steps on 2,000
  unlabelled windows, 9 to 56 minutes on this machine's accelerator.
- The few-shot forecasting task reopens with the rows' fit as its adaptation, and the
  identity of channels across corpora is the next direction: what the fitted rows are,
  against the rows the backbone learnt for like channels. For the evaluation, a frozen probe
  on a corpus the backbone never read should fit its grown rows the same way first; whether
  that moves the probe at 500 stays is a declared reading of its own.

**Limitations.**

- One seed, rate and number of steps of the fit, chosen on 2019; one pretraining per
  left-out corpus, so the 0.006 gap to the seen backbone carries no error between pretrainings.
- The fit sees the corpus's training windows without labels; it is unsupervised adaptation,
  not zero-shot.
- The check of the chunked fit is one reading on 2019; the 2012 and C-MAPSS fits were not run
  whole, since they cannot be here.

## 2026-10-19 — declared before the run: what the fitted rows are

**Question.** The section above restored the pretext on a corpus the backbone never read by
fitting the rows of that corpus's channels alone. Is a fitted row the identity of a quantity,
so that the row of 2019's heart rate lies near the row the backbone learnt for 2012's, or the
corpus's own? If the quantity's, a backbone reads a new corpus through a map of counterparts
instead of a fit, and a channel embedding shared across corpora is a knob of the pretraining;
if the corpus's, the fit without labels stays the adaptation.

**The data.**

- *The tables.* `scripts/channel_identity_report.py`: for each pair of
  `docs/verification/channel-counterparts.csv` (24 channels of PhysioNet 2019 and the channel
  of 2012 that is the same quantity; six pairs carry a note that says why they are doubtful),
  the counterpart's rank by cosine among every measured row of its corpus, in three readings —
  the rows as they are, the rows centred by the candidates' mean, and the fitted row's shift
  from where its run started it — with a percentile bootstrap over pairs and a permutation
  that gives every pair a random candidate; the share of counterparts within the first three
  (`TOP`), since a corpus's pressures lie near one another. The fitted rows are those of the
  registered continuations of 2026-10-17 (`task-transfer.md`), whose pretext repeats the
  section above's fit to 0.002: 2019's rows under the backbone without 2019 (`7d9beac7-…`),
  2012's under the one without 2012 (`8d29b0e6-…`), and C-MAPSS's under the one without it
  (`f6193117-…`, 2026-10-18). Beside them, within one corpus: the cuff against the line in
  2012 (`channel-counterparts-within-2012.csv`, three pairs) and each turbofan sensor against
  itself under the other five flight conditions (`--family`, 630 pairs among 125 rows); and
  C-MAPSS's fitted rows against 2012's learnt ones, where no counterpart exists, read by the
  largest cosine a row reaches, beside 1,000 rows drawn as the table draws them.
- *The pretext under the map.* `pretext_triviality_report.py --unset-rows map` on the backbone
  without 2019, on 2019 through the point's windows: the 24 mapped rows copied from the rows
  learnt for their counterparts, the other rows at the mean (`map-mean`) or drawn (`map`);
  the same under the map permuted (`data/report/l25d/counterparts-permuted.csv`, every
  channel given another's counterpart from seed 1), the control that says what a learnt row
  of the wrong quantity carries; and, for the first time, the tokens of the mapped channels
  tallied apart (`kinds-mapped.csv`) under every reading, the mean rows, the drawn rows, the
  fitted rows of `7d9beac7-…` and the point that read the corpus among them, so that the
  comparison is on the tokens the map reaches. The same on 2012 with the map read the other
  way. `data/report/l25d/run.sh`; about twenty readings, each under a minute.
- *Read before this was written.* The point `803b82eb-…`, which read both corpora, as the
  reference for what the learnt rows do (`data/report/l25d/point-*`); the script's trial on
  five pairs of `7d9beac7-…` (`trial-rows2019-5pairs`); and the map mode's trial on the
  backbone without 2019 under a permuted map of the same five pairs
  (`trial-loco2019-permuted5`).

**What was read before**, the point's own learnt rows (share within three [bootstrap], median
rank, p of the permutation):

| Pairs | Candidates | Raw | Centred |
|---|---|---|---|
| 2019 → 2012, 24 | 37 | 0.04 [0.00; 0.12], 21.5, 0.88 | 0.04 [0.00; 0.12], 21.5, 0.86 |
| 2012 → 2019, 24 | 34 | 0.12 [0.00; 0.25], 21.5, 0.37 | 0.12 [0.00; 0.29], 22.0, 0.36 |
| 2012, cuff → line, 3 | 36 | 1.00, 1.0, 0.002 | 1.00, 1.0, < 0.001 |
| C-MAPSS, sensor → the same under another condition, 630 | 125 | 0.21 [0.17; 0.24], 11, < 0.001 | 0.21 [0.18; 0.24], 11, < 0.001 |

Chance puts 3 of 37 within three, 0.08, and 3 of 125, 0.024. The largest centred cosine a
drawn row reaches among 2012's rows: median +0.13, 95th percentile +0.19. So the rows the
backbone learnt put the same quantity together within a corpus — the cuff beside the line,
a sensor beside itself under another condition — and not across the two intensive-care
corpora: 2019's heart rate is no nearer 2012's than 2012's other channels. The trial on five
fitted rows of 2019 (HR, Temp, SBP, MAP, DBP) placed their counterparts 16th to 42nd of 44
under every reading, the rows having moved 11 to 13 units from draws of 16 to 17. The permuted
five-pair map on the backbone without 2019 read 0.465 / 0.366 / 0.337 over the corpus, and on
the five channels alone 0.454 / 0.285 / 0.228 against 0.483 / 0.478 / 0.485 on the rest at the
mean: a learnt row of the wrong quantity already reads far better than the mean row.

**Predictions.** No interval read before covers a rank or a cosine; the thresholds below are
judgements anchored on the reference, and say so.

1. *Across corpora the fitted rows find no counterpart.* On 2019 → 2012 and 2012 → 2019 the
   share within three lies at chance under every reading — its bootstrap interval covers
   0.08 (2019 → 2012) or 0.09 (2012 → 2019) and the permutation gives p > 0.05 — as the
   point's own rows do (0.04 to 0.12). A judgement from the reference and the five-pair trial.
2. *Within a corpus the fit recovers the structure the learnt rows have.* The fitted rows of
   C-MAPSS put a sensor beside itself under another condition at a share above 0.10 under
   the shifted reading (chance 0.024, the point 0.21), p < 0.01; the fitted rows of 2012 put
   the cuff within three of the line for at least two of the three pairs. A judgement at half
   the reference's strength.
3. *A fitted row with no counterpart reaches no learnt row.* The median largest centred cosine
   of C-MAPSS's fitted rows among 2012's learnt rows lies at or below +0.19, the drawn rows'
   95th percentile.
4. *The map carries the row, not the quantity.* On the 24 mapped channels' tokens of 2019, the
   right map reads within ±0.02 of the permuted map on the block and the token, and both lie
   more than 0.05 above the fitted rows of `7d9beac7-…` on the same tokens; over the whole
   corpus `map-mean` reads at least 0.05 below the mean rows on the block and the token
   (0.472 / 0.473), as the permuted trial did (0.366 / 0.337). The same on 2012. A judgement
   from the trial.

**What follows.**

- If 1, 3 and 4 hold: a channel's identity in this backbone is the corpus's, not the
  quantity's; a map of counterparts is no way to read a new corpus and a shared channel
  embedding is no knob of the next pretraining; the fit without labels stays the adaptation
  (ADR-0057). Prediction 2 then says whether a fitted row is a row of the same kind as a
  learnt one or a nuisance parameter: if 2 fails, the cosine readings of fitted rows say
  nothing about them, and only the pretext does.
- If 1 fails — the fitted rows find counterparts the point's own rows do not: the fit on a
  frozen encoder lands on a quantity's direction the joint pretraining spread; the map
  reading is then the test of use, and 4 is read first.
- If 4 fails — the right map beats the permuted one by more than 0.05 on the mapped tokens:
  the quantity's identity carries across corpora in use though not in cosine; the rows of a
  new corpus start from their counterparts before the fit, a knob declared for the next
  pretraining.
- A reading that refuses is named and left out.

**Limitations.**

- One fitted table per corpus, one seed of each fit; the C-MAPSS rows were fitted under a
  shorter schedule than the clinical ones (900 steps at 16 × 4 against 6 epochs).
- The map is one reading of "the same quantity": six pairs are doubtful and are read with and
  without; a counterpart in the other corpus may be sampled on another cadence, which the
  row may encode, so a failure of 1 does not say whether the identity is the corpus's or its
  cadence's.
- The reference and two trials were seen before the predictions; the thresholds are
  judgements, not intervals.
- 24 pairs: a share within three of 0.25 or more is told from chance, one of 0.15 is not.
