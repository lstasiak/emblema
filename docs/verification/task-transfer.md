# The task on a corpus the backbone never read

Whether what carries a backbone's pretext to a corpus it never read also carries the task on
that corpus. The rows of the corpus's channels, fitted on its unlabelled training windows with
every other weight frozen, restore the pretext to within 0.006 of a backbone that read the
corpus (`zero-shot-pretext.md`, 2026-10-17); that reading is of what the encoder predicts, not
of what a linear head reads from its states. Here the same fit is a registered pretraining
(ADR-0056), so the evaluation's probe reads the backbone it leaves as it reads any other. The
reader within a backbone on the sepsis task is the row of `docs/preregistration.md` registered
for it on 2026-10-17; nothing else here is a registered rule.

## 2026-10-17 — declared before the run: the rows fitted, read by the probe on sepsis and on the stays

**Question.** Under a backbone that never read a corpus, does the closed-form probe on that
corpus's task read the pretraining's share differently once the corpus's channel rows are
fitted on its unlabelled training windows, the encoder frozen — and does an encoder at its
initialisation, given the same fit, not?

**The data.**

- *The backbones on PhysioNet 2019.* Under the point of about 10⁹ values without the 2019
  stays (`backbone-scale-1e9-without-physionet2019-m`, `4028de56-…`): (a) the point as it is,
  2019's channels passing through rows the evaluation draws for each seed; (b) the point
  continued over the 2019 stays with the channel table alone free
  (`backbone-scale-1e9-without-physionet2019-rows-m`); (c) the same fit on an encoder the seed
  drew (`…-rows-untrained-m`), the control that tells the machinery from the rows. Beside them,
  read in `sepsis-reader.md` (2026-10-16, 2026-10-17): (d) the point that read 2019
  (`backbone-scale-1e9-m`) and (e) 2019 alone (`backbone-physionet2019-m`).
- *The backbones on PhysioNet 2012.* Under the point without the 2012 stays
  (`backbone-scale-1e9-without-physionet2012-m`, `ba2912dd-…`): (a) as it is, 2012's rows
  inside its table at their initialisation, read at 500 stays in `layer-readout.md`
  (2026-10-13, campaign `87c68b49-…`); (b) `backbone-scale-1e9-without-physionet2012-rows-m`;
  (c) `…-rows-untrained-m`. Beside them the point that read 2012 and the mixture of four's
  pretrainings at 500 stays (`verdict-statistics.md`, 2026-10-13).
- *The fit.* Adam at 0.01, no warm-up and no decay, the pretext's masks and loss, the channel
  table the only parameter the optimiser is handed, every other weight the point's. On 2019 a
  quarter of the training stays over six epochs at an effective batch of 64 (16 × 4), about
  1,080 steps; on 2012 the whole training side over sixteen epochs at 8 × 8, about 1,000
  steps, since a batch of 64 stays of 2012 does not fit this machine's accelerator at once.
  Precision fp32, as the diagnostic read the fit; the best epoch kept by the pretext's
  validation loss, as every run keeps it. The diagnostic's fit read 2,000 windows spaced evenly
  over the training side for 1,000 steps and kept the last step; one step of the two over the
  same windows and masks agrees to rounding (`tests/scripts/test_pretext_triviality_report.py`),
  and from the second step on each draws its masks by its own protocol.
- *The pretext check, before any campaign.* `scripts/pretext_triviality_report.py --backbone`
  on (b) and (c) of each corpus, on the corpus itself, which is now the backbone's own input.
- *The reading.* On sepsis, `campaigns/reader-probes-physionet2019.toml` under (a), (b) and
  (c): `frozen_ridge@layer=concat` and `untrained_ridge@layer=concat` at 400, 4,000 and every
  patient of the half of the tuning side held out by seed 101, seeds 1 to 10, sixty cells a
  backbone on this machine's accelerator. On the stays, `campaigns/review-concat-500-physionet2012.toml`
  under (b) and (c) at 500 stays, twenty cells each. Pairs by `scripts/campaign_pairs_report.py`
  over stays and seed: the probe less its initialisation within each backbone, and (b) less (a)
  and (c) less (a) across backbones of one pretraining. The rule within a backbone: the paired
  interval above zero and the mean over seeds above twice its standard error; at 4,000 patients
  it resolves 0.02 in 0.97 and 0.01 in 0.43, at 500 stays 0.02 in 0.35.
- *The trial, seen before this was written.* The continuation (b) on 2019 was ordered, run
  and accepted from the commit of the knobs (`10751086`; `data/report/l23a/pretrain.sh`):
  backbone `7d9beac7-…`, parent `4028de56-…` in the registry, 181 steps an epoch, 655 s on
  this machine's accelerator, the pretext's training loss 0.277 in the first epoch and 0.215
  in the sixth, the validation loss 0.216 (0.559 of the trivial predictor's), the sixth epoch
  kept. Its pretext read on 2019 by `scripts/pretext_triviality_report.py --backbone`
  (`data/report/l23a/pretext-rows2019`): channel / block / token 0.293 / 0.179 / 0.143 against
  the diagnostic's 0.292 / 0.180 / 0.145, the block and the token learnt (+0.047 [+0.042;
  +0.052], +0.014 [+0.009; +0.018]), the channel matched (−0.005 [−0.009; −0.000]). The
  registered run is the diagnostic's fit on 2019; prediction 1 is tested on (c) and on 2012.
  No campaign has run.

**What is known** (area under the ROC curve, mean over seeds 1 to 10; the pair in brackets is
the probe less its initialisation, paired over stays, with its 95 % interval).

| Task, budget | at initialisation | (d) the point that read the corpus | (e) the corpus alone | the mixture of four |
|---|---|---|---|---|
| sepsis, 4,000 | 0.728 | 0.680 (−0.047 [−0.064; −0.030]) | 0.725 (−0.003 [−0.013; +0.008]) | not readable: its vocabulary ends before 2019's channels |
| sepsis, 400 / all | 0.647 / 0.743 | −0.065 / −0.042 | +0.005 / −0.007 | — |
| stays, 500 | 0.680 | 0.673 (−0.007 [−0.029; +0.014]) | — | 0.692 to 0.710 (−0.001 to +0.017) |
| stays, 500, (a) the point without 2012 | 0.680 | 0.665 (−0.015 [−0.047; +0.008], seeds −0.015 ± 0.015) | | |

The initialisation is drawn per seed from the backbone's shape and vocabulary, so under (a),
(b), (c), (d) and (e) of one task it is one reading: every pair below is against the same
numbers.

**Predictions.**

1. *The pretext.* (b) reads 2019 within 0.02 of the diagnostic's fitted reading on every kind
   of mask (channel / block / token 0.292 / 0.180 / 0.145) and (c) stays beaten by more than
   0.05 on every kind; on 2012 the same against 0.248 / 0.145 / 0.119. The registered run
   reproduces the diagnostic's fit; where it does not, the run is named and read before
   anything is claimed of the task.
2. *The rows carry the task* (the main reading, sepsis at 4,000). The probe under (b) less its
   initialisation lies above the same difference under (a) by more than the rule within one
   pretraining resolves — (b) less (a), paired over stays and seed, has its interval above zero
   and its mean over seeds above twice its standard error — and lies in the band of the
   backbones that read the corpus, between (d) and (e): −0.047 to −0.003. Read descriptively at
   400 and at every patient, and at 500 stays on 2012, where (b) less (a) is expected above
   zero and (b) in the band of the point that read 2012 and the mixture, −0.029 to +0.017.
3. *The rows alone do not.* (c) less its initialisation lies within the rule's resolution of
   zero: its interval covers zero or its mean over seeds is below twice its standard error, and
   its absolute value is below 0.02 at 4,000 patients.
4. *Under (a) the point without 2019 is no better than its initialisation*: the probe less its
   initialisation at 4,000 lies at or below zero, as under (d), since the corpus's channels pass
   through rows that carry nothing.

**What follows.**

- If predictions 2 and 3 hold: the rows fitted without labels carry the task as they carry the
  pretext — the first transfer in a task on this backbone to a corpus it never read — and the
  fit becomes the adaptation step of every reading on such a corpus: the few-shot reading on
  FD001 under the point without C-MAPSS, and the readings of the next objective. A
  continuation with everything free is then read against this one as its baseline.
- If prediction 2 fails, (b) no better than (a) by the rule: the pretext and the task part
  ways at the rows. The machinery restores what the encoder predicts and not what a head reads
  from its states, and that boundary is what the next readings start from.
- If prediction 3 fails, (c) above its initialisation by the rule: the fitted rows carry task
  information of their own, and (b)'s share is read against (c), not against (a).
- If prediction 1 fails on a corpus: the registered run is not the diagnostic's fit there, and
  the task is not read on it until the difference is named.
- A cell that refuses, or a campaign that does not run whole, is named and its backbone left
  out.

**Limitations.**

- One pretraining of each point, one seed of each fit: nothing here carries the error between
  pretrainings, and the reading is within one pretraining by design.
- The fit reads the corpus's training windows without labels, as unsupervised adaptation in the
  target domain; the task's labels stay unseen by it, and the test and validation sides unread.
- At 500 stays the rule within a backbone resolves 0.02 in 0.35 of readings: the reading on
  2012 is descriptive beside the one on sepsis.
- The probe is the reader; a network fine-tuned on either task is not read.
