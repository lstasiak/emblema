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
