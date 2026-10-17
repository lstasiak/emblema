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

## 2026-10-17 — measured: the rows fitted carry a part of the task at 4,000 patients, and nothing on an encoder at its initialisation

**Question.** As declared above: whether the probe on sepsis and on the stays reads the
pretraining's share differently once the corpus's channel rows are fitted without labels, and
whether the same fit on an encoder at its initialisation reads as nothing.

**Conditions.** Commit `07953136`, this machine's accelerator (M1 Pro, MPS, fp32), the
registry's handoff for every run (`data/report/l23a/pretrain.sh`, `define.sh`, `orders.sh`,
`run.sh`, `pairs.sh`). The pretrainings:

| Backbone | Id | Parent | Steps | Wall time | Validation loss, first → last epoch (of the trivial predictor's) |
|---|---|---|---|---|---|
| (b) 2019 rows | `7d9beac7-…` | `4028de56-…` | 181 × 6 | 655 s | 0.237 → 0.216 (0.56) |
| (c) 2019 rows, untrained | `dcedf337-…` | — | 181 × 6 | 640 s | 0.392 → 0.377 (0.97) |
| (b) 2012 rows | `8d29b0e6-…` | `ba2912dd-…` | 63 × 16 | 1,859 s | 0.222 → 0.174 (0.49) |
| (c) 2012 rows, untrained | `5a76a37b-…` | — | 63 × 16 | 1,777 s | 0.373 → 0.347 (0.99) |

The validation loss fell every epoch under all four, so the last epoch is the one kept. The
campaigns: sepsis under (a) `4b8276d8-…`, (b) `3f3229ae-…`, (c) `47d2891d-…`, sixty cells
each in 66 to 72 minutes; the stays under (b) `49d7fffb-…` and (c) `2d7e3b74-…`, twenty cells
each in 10 minutes. Every cell answered.

**Prediction 1, the pretext** (`scripts/pretext_triviality_report.py --backbone`, error on
hidden tokens, channel / block / token; the diagnostic's fit of `zero-shot-pretext.md` in
brackets). Held on both corpora: (b) within 0.002 of the diagnostic's fit on every kind of
mask, (c) beaten by the matched baseline by 0.08 to 0.24.

| Backbone | channel | block | token | learnt |
|---|---|---|---|---|
| (b) 2019 rows | 0.293 (0.292) | 0.179 (0.180) | 0.143 (0.145) | block, token |
| (c) 2019 rows, untrained | 0.364 | 0.391 | 0.396 | none |
| (b) 2012 rows | 0.247 (0.248) | 0.145 (0.145) | 0.118 (0.119) | all three |
| (c) 2012 rows, untrained | 0.357 | 0.343 | 0.367 | none |

**The probe on sepsis** (area under the ROC curve, mean over seeds 1 to 10; the pair is the
probe less its initialisation, paired over patients, with its 95 % interval; then the mean over
seeds of the same difference with its standard error). The initialisation is one reading under
(a), (b) and (c) and the campaigns of `sepsis-reader.md`: 0.647 / 0.728 / 0.743 at 400 / 4,000
/ every patient, identical to four decimals.

| Backbone | 400 | 4,000 | every patient (9,761) |
|---|---|---|---|
| (a) the point without 2019 | 0.567, −0.080 [−0.096; −0.064], −0.080 ± 0.017 | 0.642, −0.085 [−0.105; −0.066], −0.085 ± 0.010 | 0.687, −0.056 [−0.070; −0.041], −0.056 ± 0.005 |
| (b) its rows fitted | 0.565, −0.082 [−0.099; −0.065], −0.082 ± 0.017 | 0.673, −0.054 [−0.074; −0.034], −0.054 ± 0.003 | 0.691, −0.052 [−0.072; −0.032], −0.052 ± 0.001 |
| (c) the same fit, untrained | 0.654, +0.007 [+0.004; +0.011], +0.007 ± 0.003 | 0.725, −0.003 [−0.008; +0.002], −0.003 ± 0.001 | 0.739, −0.004 [−0.008; −0.000], −0.004 ± 0.001 |
| (d) the point that read 2019 | −0.065 | 0.680, −0.047 [−0.064; −0.030] | −0.042 |
| (e) 2019 alone | +0.005 | 0.725, −0.003 [−0.013; +0.008] | −0.007 |

Across backbones, the probe under one less the probe under (a), paired over patients and seed:

| Pair | 400 | 4,000 | every patient |
|---|---|---|---|
| (b) less (a) | −0.002 [−0.011; +0.006], −0.002 ± 0.007 | **+0.031 [+0.015; +0.047], +0.031 ± 0.009** | +0.004 [−0.012; +0.020], +0.004 ± 0.005 |
| (c) less (a) | +0.087 [+0.071; +0.104], +0.087 ± 0.016 | +0.082 [+0.063; +0.102], +0.083 ± 0.010 | +0.052 [+0.036; +0.066], +0.052 ± 0.005 |

**The probe on the stays at 500** (the same columns; (a) and the band from the campaigns named
in the declaration).

| Backbone | at initialisation | probe | probe less initialisation | less (a) |
|---|---|---|---|---|
| (a) the point without 2012 | 0.680 | 0.665 | −0.015 [−0.047; +0.008], −0.015 ± 0.015 | — |
| (b) its rows fitted | 0.693 | 0.670 | −0.023 [−0.043; −0.002], −0.023 ± 0.010 | +0.005 [−0.006; +0.016], +0.005 ± 0.003 |
| (c) the same fit, untrained | 0.693 | 0.668 | −0.025 [−0.033; −0.016], −0.025 ± 0.011 | +0.003 [−0.024; +0.030], +0.003 ± 0.016 |
| the point that read 2012 | 0.680 | 0.673 | −0.007 [−0.029; +0.014] | |
| the mixture of four | 0.680 | 0.692 to 0.710 | −0.001 to +0.017 | |

The initialisation under (b) and (c) is not the reading under (a): a fresh encoder is drawn
over the larger of the task's vocabulary and the backbone's, and the continuation's table
holds 2012's channels alone where its parent's holds the mixture's, so the draw differs
(0.693 against 0.680). Each pair within a backbone is against that backbone's own
initialisation; the pairs across backbones compare probes and are unaffected. On sepsis the
tables are of one size and the reading is one, as the declaration assumed for both tasks.

**Predictions.**

1. *The pretext.* Held on both corpora, as above.
2. *The rows carry the task.* Held by the rule at 4,000 patients: (b) less (a) is +0.031
   [+0.015; +0.047], its mean over seeds 3.6 standard errors above zero. The band clause held
   at the interval and not at the point: (b) less its initialisation is −0.054, 0.007 below
   (d)'s −0.047, with its interval covering (d). Descriptively the gain is of 4,000 patients
   alone: at 400 and at every patient (b) and (a) are within 0.004 of each other. On the
   stays at 500, (b) less (a) is +0.005 [−0.006; +0.016], above zero at the point and by no
   rule, and (b) at −0.023 lies in the band of the backbones that read 2012.
3. *The rows alone do not.* Held at 4,000: (c) less its initialisation is −0.003 [−0.008;
   +0.002], the interval covering zero. At 400 it is +0.007 [+0.004; +0.011] and at every
   patient −0.004 [−0.008; −0.000], each below 0.02 in size. On the stays the same fit on an
   untrained encoder reads 0.025 below its initialisation, within the seeds' spread at 500.
4. *Under (a) the point without 2019 is no better than its initialisation.* Held, and by far:
   −0.085 [−0.105; −0.066], the worst reading of any backbone on sepsis, 0.038 below (d).

**Conclusions.**

- The channel rows fitted on a corpus's unlabelled windows, every other weight of a backbone
  that never read the corpus frozen, carry a part of the task the probe reads on it: at 4,000
  patients the probe gains 0.031 [+0.015; +0.047] over the same backbone with rows drawn,
  0.031 of the 0.038 to the backbone that read the corpus. The first transfer in a task to a
  corpus a backbone never read, within one pretraining.
- The fit carries no task of its own: on an encoder at its initialisation the fitted rows read
  as drawn ones at 4,000 patients. (b)'s share is read against (a), as declared.
- What the rows restore is the share of the backbone that read the corpus, not a gain over
  the initialisation: every pretrained backbone reads sepsis below the encoder at its
  initialisation, as `sepsis-reader.md` found, and the point without 2019 reads worst of all.
- The gain is read at 4,000 patients and not at 400 or at every patient. Which budget reads
  the transfer is a fact of this reading, not a rule.
- On the stays at 500 nothing resolves, as the declaration expected of that budget.

**What follows.** Predictions 2 and 3 hold, so the declaration's first branch applies: the fit of
the rows is the adaptation step of every reading on a corpus the backbone never read, and a
continuation with everything free is read against this one as its baseline. The branch is
taken with its limits named: one budget of one task resolved it, and the rows restore the
share of a backbone that read the corpus, which on sepsis is a loss against the initialisation.

**Limitations.**

- One pretraining of each point and one seed of each fit, as declared: the interval carries the
  patients and the seeds of the probe, not the error between pretrainings.
- The transfer is read by the rule at one budget of one task; at the other budgets and on the
  stays it is a point estimate inside an interval that covers zero.
- On the stays the initialisation under the continuations is a draw of a smaller table than
  under their parent, so the pairs within a backbone there are against different
  initialisations; the declaration's assumption of one reading held on sepsis only.
- The probe is the reader; a network fine-tuned on either task is not read.
