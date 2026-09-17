# Corpus saturation: does more of a corpus still lower the loss?

Purpose: before the first full pretraining run, replace the rule of thumb that admitted the
reference encoder — twenty unique observed values per parameter, a ratio fitted to language
models — with a measurement on the corpora themselves. A small model is trained over a tenth, a
quarter, half and all of a corpus's training units at one budget of optimiser steps, and its
validation loss at the end of each run is read against the share. The decisions are in ADR-0027
(how the measurement is made and judged) and in the dated section of ADR-0008 (what it did to
the parameter budget); this note records what was observed, where, and with which versions.

What must hold everywhere is a test, not a number here: that a share of a corpus is a share of
whole units, nested for one seed and honoured the same by every reader of a published corpus
(`tests/pretraining/ports/test_training_corpus_reader_contract.py`); that every share of a curve
spent one budget of steps and the verdict rules earn each verdict on a curve that should and
refuse it to one that should not (`tests/pretraining/domain/saturation`); that a run is written
as it goes, left alone once finished and picked up from its last checkpoint when a session drops
(`tests/scripts/test_corpus_saturation_report.py`). What this note adds is the curves of real
corpora on a real machine, and what an evening of them costs.

Every number here is measured on validation units. The test portions of every corpus are untouched.

Method, on a machine that holds the published corpora and the local stack:

```sh
uv sync --all-extras
uv run pytest tests/pretraining/domain/saturation tests/scripts/test_corpus_saturation_report.py
uv run scripts/corpus_saturation_report.py --corpus esa_ad <manifest key> <checksum> \
    --experiment saturation-esa_ad-s --device mps --track http://127.0.0.1:5000
uv run scripts/corpus_saturation_report.py --report-only          # the tables from the stored runs
uv run scripts/corpus_saturation_figures.py data/report/saturation/runs --figures docs/verification/figures
```

Each experiment in `experiments/saturation-*.toml` states the budget a run over the whole corpus
gets; the report derives for each share the epochs that spend the same steps, reads the share's
units, trains through the pretraining use case and stores each run under
`data/report/saturation/runs/<experiment>/<share>/` as it goes. A dropped session is resumed by
running the same command again: finished runs are skipped, the interrupted one is picked up from
the checkpoint its last recorded epoch reported. A smoke run (`--smoke`, `--epochs 1`, a share of
a few per cent) trains every share for the stated epochs instead and measures what an epoch costs.

## 2026-09-17 — the small tier over four corpora, one seed (macOS arm64, M1 Pro, MPS, fp32)

The first leg: the small tier's shape (192 wide, 4 blocks, 3 heads, feed-forward 768, twelve time
frequencies; 1.79M parameters over each corpus's own vocabulary, 1.81M over the mixture's) trained
over a tenth, a quarter, half and all of the training units of each published corpus, at the budget
of steps its experiment gives the whole corpus, seed 1, dropout 0, effective batch 32 (SMD in
micro-batches of 16 accumulated twice), fp32 on MPS. Validation was scored on every fourth window of
the held-out side, the same windows for every share: 1,290 C-MAPSS, 192 SKAB, 3,076 SMD and 3,846
ESA windows. The corpora are the versions in the local catalog: C-MAPSS at window 50 / stride 5
(block `bbee7fcd…`, 21 channels), SKAB at 100 / 10 (block `56141548…`, 8 channels), SMD at 50 / 10
(block `b686fdcf…`, 38 channels — republished at this window for the measurement, beside the
100 / 20 version) and ESA at one hour / one hour (block `20cb7969…`, 17 channels). The runs were
made from the ticket's own tree before it was committed, which the revision below records as dirty;
the code is the branch's, reviewed and committed after the measurement. MLflow experiments 6–9 on the
local server hold the same curves.

|  |  |
| --- | --- |
| Machine | macOS-26.6.2-arm64-arm-64bit-Mach-O, arm |
| Python | 3.14.7 |
| torch | 2.14.0 |
| Device | mps |
| Revision | 49cb280e6a801fc76eba2e3fc275bfc07082d559-dirty |
| Rule | every share of a corpus spends the optimiser steps its experiment gives the whole corpus, rounded to whole epochs; every share is scored on the same validation windows |

Commands, in the order run (the second and third read the stored runs and train nothing):

```sh
caffeinate -is uv run scripts/corpus_saturation_report.py \
    --corpus esa_ad <manifest> <checksum> --corpus smd <manifest> <checksum> \
    --corpus cmapss <manifest> <checksum> --corpus skab <manifest> <checksum> \
    --experiment saturation-esa_ad-s --experiment saturation-smd-s \
    --experiment saturation-cmapss-s --experiment saturation-skab-s \
    --validation-stride 4 --device mps --track http://127.0.0.1:5000 2>&1 | tee data/report/saturation/leg-1.log
uv run scripts/corpus_saturation_report.py --report-only --experiment saturation-cmapss-s \
    --experiment saturation-esa_ad-s --experiment saturation-skab-s --experiment saturation-smd-s
uv run scripts/corpus_saturation_figures.py data/report/saturation/runs --figures docs/verification/figures
```

### What it cost

Sixteen runs in 16.0 hours of wall clock (23:32 to 15:30), against 14.8 hours projected from the
smoke run; the difference is one epoch of the SMD half share that took 36 minutes instead of 24
while the machine was in use. Seconds per optimiser step at the whole share, validation passes
included: ESA 1.32, SMD 1.53, C-MAPSS 1.01, SKAB 0.41. By corpus: ESA 5.8 h, SMD 6.8 h, C-MAPSS 3.0 h,
SKAB 0.24 h. Every share of a corpus costs the same, as the rule intends, to within the epoch
rounding and the validation passes a share with more epochs makes.

### Curves and verdicts

#### saturation-cmapss-s

| Share | Training windows | Epochs | Steps | Last training | Last validation | Relative training | Relative validation | Best validation (epoch) | Gap | Minutes | Finished |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10% | 1841 | 44 | 2552 | 0.0050 | 0.0051 | 0.005 | 0.005 | 0.005 at 44/44 | 1.01× | 50 min | yes |
| 25% | 4768 | 17 | 2533 | 0.0056 | 0.0051 | 0.006 | 0.005 | 0.005 at 17/17 | 0.89× | 45 min | yes |
| 50% | 10057 | 8 | 2520 | 0.0045 | 0.0045 | 0.005 | 0.004 | 0.004 at 8/8 | 0.98× | 43 min | yes |
| 100% | 20237 | 4 | 2532 | 0.0049 | 0.0047 | 0.005 | 0.005 | 0.005 at 4/4 | 0.93× | 43 min | yes |

Relative losses are against the trivial predictor's on the same side — the channel mean, whose error is 1.022 on the validation side and 1.013 on the training side of the smallest share; the gap is their ratio. Best validation is the lowest of the run's epochs, with the epoch it fell at; the verdict reads the last epoch, where every share has spent its budget.

Gains in validation loss share to share: 10%→25% +1.2%, 25%→50% +10.8%, 50%→100% -4.4%.

**saturated** — the step from 50% to 100% of the training units lowered validation by -4.4%, under 5%: the corpus is learnt at this size and budget, and a larger model over it gains nothing — other corpora would.

#### saturation-esa_ad-s

| Share | Training windows | Epochs | Steps | Last training | Last validation | Relative training | Relative validation | Best validation (epoch) | Gap | Minutes | Finished |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10% | 6515 | 19 | 3876 | 0.0846 | 5.0541 | 0.193 | 1.028 | 1.022 at 7/19 | 5.33× | 94 min | yes |
| 25% | 15154 | 8 | 3792 | 0.1080 | 5.1210 | 0.227 | 1.042 | 1.003 at 2/8 | 4.59× | 86 min | yes |
| 50% | 30700 | 4 | 3840 | 0.2386 | 5.2678 | 0.343 | 1.072 | 1.030 at 1/4 | 3.12× | 85 min | yes |
| 100% | 61299 | 2 | 3832 | 0.3426 | 6.3533 | 0.343 | 1.293 | 1.240 at 1/2 | 3.77× | 84 min | yes |

Relative losses are against the trivial predictor's on the same side — the channel mean, whose error is 4.915 on the validation side and 0.438 on the training side of the smallest share; the gap is their ratio. Best validation is the lowest of the run's epochs, with the epoch it fell at; the verdict reads the last epoch, where every share has spent its budget.

Gains in validation loss share to share: 10%→25% -1.3%, 25%→50% -2.9%, 50%→100% -20.6%.

**not learnt** — at the whole share validation is 1.29× the trivial predictor's: nothing carried over to the held-out side, so the curve reads nothing about the data — the held-out side, or the loss it is scored by, comes before the corpus is judged.

#### saturation-skab-s

| Share | Training windows | Epochs | Steps | Last training | Last validation | Relative training | Relative validation | Best validation (epoch) | Gap | Minutes | Finished |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10% | 290 | 49 | 490 | 0.2771 | 1.3257 | 0.315 | 1.054 | 0.993 at 21/49 | 3.35× | 4 min | yes |
| 25% | 738 | 20 | 480 | 0.2770 | 0.4785 | 0.303 | 0.381 | 0.379 at 17/20 | 1.26× | 4 min | yes |
| 50% | 2391 | 7 | 525 | 0.2838 | 0.4904 | 0.311 | 0.390 | 0.387 at 3/7 | 1.25× | 4 min | yes |
| 100% | 3896 | 4 | 488 | 0.2991 | 0.3127 | 0.297 | 0.249 | 0.249 at 4/4 | 0.84× | 3 min | yes |

Relative losses are against the trivial predictor's on the same side — the channel mean, whose error is 1.257 on the validation side and 0.880 on the training side of the smallest share; the gap is their ratio. Best validation is the lowest of the run's epochs, with the epoch it fell at; the verdict reads the last epoch, where every share has spent its budget.

Gains in validation loss share to share: 10%→25% +63.9%, 25%→50% -2.5%, 50%→100% +36.2%.

**data-limited** — the step from 50% to 100% of the training units lowered validation by 36.2%: the data, not the model, is the limit at this budget — the corpus suits pretraining and more of it would help.

#### saturation-smd-s

| Share | Training windows | Epochs | Steps | Last training | Last validation | Relative training | Relative validation | Best validation (epoch) | Gap | Minutes | Finished |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 10% | 7597 | 15 | 3570 | 0.1599 | 1.4632 | 0.144 | 1.011 | 0.890 at 3/15 | 7.01× | 100 min | yes |
| 25% | 15195 | 8 | 3800 | 0.1400 | 1.2323 | 0.128 | 0.851 | 0.790 at 3/8 | 6.66× | 105 min | yes |
| 50% | 30387 | 4 | 3800 | 0.1523 | 1.0230 | 0.148 | 0.707 | 0.616 at 1/4 | 4.78× | 111 min | yes |
| 100% | 58411 | 2 | 3652 | 0.2253 | 0.8418 | 0.238 | 0.582 | 0.571 at 1/2 | 2.44× | 93 min | yes |

Relative losses are against the trivial predictor's on the same side — the channel mean, whose error is 1.447 on the validation side and 1.109 on the training side of the smallest share; the gap is their ratio. Best validation is the lowest of the run's epochs, with the epoch it fell at; the verdict reads the last epoch, where every share has spent its budget.

Gains in validation loss share to share: 10%→25% +15.8%, 25%→50% +17.0%, 50%→100% +17.7%.

**overfitting** — at the whole share validation is 2.44× training, each against its own trivial predictor, past 1.5×: the model fits windows it will not see again, so the data is too little for it — a smaller model, or more units.

![Validation loss against the share of each corpus](figures/corpus-saturation-validation.png)

![Relative validation over relative training loss at the last epoch](figures/corpus-saturation-generalisation.png)

![Validation loss over each run, every share at one budget of steps](figures/corpus-saturation-epochs.png)

### Reading

**C-MAPSS is saturated at the objective's floor.** Every share ends at half a per cent of the
channel variance on both sides, the gap is one, and every share reaches that floor within about
1,500 steps regardless of how much of the corpus it read (the first panel of the third figure).
Hiding tokens of a 50-cycle window of these sensors leaves them nearly determined by the tokens
left visible — the sensors follow a few latent states — so the floor is the objective's, not the
data's: more C-MAPSS, or more passes over it, teaches the encoder nothing at this size. The corpus
remains the host of the remaining-useful-life task; as pretraining data it is learnt from a quarter
of its engines.

**SMD is the clean curve, and it says data-limited and overfitting at once.** Each doubling of the
units lowers the validation loss by 16–18 %, with no sign of flattening at the whole corpus; the gap
shrinks from 7.0× at a tenth to 2.4× at the whole; and every run's best epoch is its first or
third, after which validation rises while training keeps falling (the fourth panel). The rules
fixed beforehand call it overfitting, because the gap at the whole share is past 1.5×, and what
that verdict prescribes — more units — is what the curve shows working: the overfitting recedes
as the units come. Read at each run's best epoch instead of its last, the shares order the same
way (0.89, 0.79, 0.62, 0.57).

**SKAB is data-limited on a corpus of 3,896 windows.** A tenth of its experiments (290 windows)
learns nothing of the held-out side; a quarter and a half tie at 0.38; the whole reaches 0.25, with
a gap under one — the held-out experiments are easier than the training ones for this model. The
curve is not monotone and rests on one seed, so the verdict stands on its last step and on nothing
else. A small ingredient of the mix, as the corpus arithmetic had said.

**ESA's held-out months were not learnt, at any share.** The validation loss never falls under the
channel mean's — 1.00 at best, 1.02–1.07 at the last epoch of the three smaller shares, 1.24–1.29
at the whole — while the training side is learnt to 0.19–0.34 of its variance. The 21 held-out
months carry excursions of up to 96 standard deviations, the channel mean errs 4.9 there against
0.44–1.0 on the training sides, and a squared error over such months is decided by a few windows on
which no model trained on other months does better than the mean. Measured on the published block:
one held-out month of the 21, `ESA-Mission1/2000-04` — the first weeks of the mission — holds 88 %
of the side's squared magnitude; the 0.3 % of its tokens past ten standard deviations hold 91 % of
it, and 1 % of its windows 92 %; the other 99.7 % of the tokens have a mean square of 0.45, easier
than the training side's ordinary tokens at 0.56. What the model does on those tokens is invisible
in this loss. The training side is half the same story: 0.13 % of its tokens hold 45 % of its
squared magnitude, so the objective's gradient over this corpus is half from excursions. The run
over the whole corpus does worse than the small ones, not better, because its training months
hold excursions of their own (three months with a mean square of 10–14) and a model that learns
to reproduce them is punished where they do not recur. The rules fixed beforehand read this as overfitting
(3.77×) and prescribed a smaller model or more units, which is the wrong remedy for a side that was
never learnt; the fourth reading, "not learnt", was added for this case after the curve was drawn
and is declared as such in ADR-0027. What this leg says about the satellite corpus is therefore
about its held-out side and its loss, not about its volume.

**Within every run but C-MAPSS, validation is best after one to three epochs.** The rule of equal
steps repeats a tenth of a corpus fifteen to forty-nine times, and at 1.79M parameters and about
3,800 steps of batch 32 the small tier overfits each classical corpus on its own within one or two
passes — the whole of SMD is best after its first epoch of two. The best-validation column reads
what early stopping would have kept; it changes no verdict and no ordering of shares.

### Limitations

- One seed. Shares of one seed are nested, so the curve of a corpus carries no between-subset
  variation, but a second seed would show how much of SKAB's non-monotone curve is the draw of
  three experiments.
- The budget is an evening's, not convergence: C-MAPSS converged, the other three did not, and
  every verdict is read at this budget. Whether the reading holds at the budget of a full
  pretraining run is what that run's own curve shows.
- Validation on every fourth window, so the validation numbers carry the sampling error of a
  quarter of the held-out side; the same windows for every share, so the comparison between
  shares does not.
- The loss is an unbounded squared error in units normalised on the training side, and the split
  holds out whole months at random. On an anomaly benchmark both are the wrong instruments for a
  saturation curve: a bounded loss (or excursions clipped in the objective, never in the
  tokeniser, which the anomaly task needs raw) and a held-out side that spreads the excursion
  months are what the satellite curve needs before it can be read; the concentration figures above
  are from the block's own values (tokens past ten standard deviations, per month), not from any
  model, and are to be regenerated by the diagnostic that scores these runs on ordinary tokens.
- ESA's ragged windows are padded to the longest in each batch (a median of 720 tokens against a
  maximum of 1,331), so the satellite runs cost about a third more per step than their tokens; the
  cost table above is what the padding cost, not what the tokens would.
- The tree was dirty: the code of the measurement was the branch's own, uncommitted; the revision
  column names the commit it was built on.
- The small tier's shape only. The reference shape over the half and the whole of ESA and SMD, at
  the same budget of steps and on the same machine, is the second leg; its section follows when it
  ends.

### What it decided

Recorded in the dated section of ADR-0008: the reference shape of the published tier stands for
the mixed backbone and the sweep past it is closed at this stage; the third satellite mission is not
added; corpora are mixed from the first full run as before, with C-MAPSS weighted as data that is
learnt from a quarter of it, and with the satellite corpus's held-out months scored apart from the
mix's and their loss reconsidered before its share of the mix is judged.

## 2026-09-17 — the reference shape over the satellite corpus, second leg (macOS arm64, M1 Pro, MPS, fp32)

The size axis, on the satellite corpus alone: the published tier's shape (256 wide, 6 blocks,
4 heads, feed-forward 1024, twelve time frequencies; 4.75M parameters over ESA's 17 channels)
over the half and the whole of ESA's training units, at the same steps as the small shape's runs
(3,840 and 3,832), seed 1, dropout 0, micro-batches of 16 accumulated twice, on the same machine
and the same validation windows. The experiment file declares the small tier's machine and states
the reference shape field by field, which is how the figures label it: "tier S, 256 wide, 6 deep".
The SMD half of the leg was stopped before its first epoch ended, so the size axis on the largest
classical corpus is unmeasured; a run directory with settings and no epochs remains under
`runs/saturation-smd-m/0.5/`, and a later run of the same command starts it afresh.

Same commands as the first leg with `--experiment saturation-esa_ad-m --experiment saturation-smd-m`
and `--fractions 0.5 1`; the report rendered with `--report-only --experiment saturation-esa_ad-m`;
the figures redrawn from all stored runs, so the reference shape joins the three figures above
rather than getting a set of its own.

### What it cost

108 and 104 minutes: 1.69 s per optimiser step at the half share and 1.63 s at the whole, validation
passes included, against 1.32 s for the small shape over the same corpus — the reference shape
costs about a quarter more per step on this corpus, less than its 2.7× parameters, because the
padded attention over ragged windows is the larger part of a step at either shape.

### Curve and verdict

#### saturation-esa_ad-m

| Share | Training windows | Epochs | Steps | Last training | Last validation | Relative training | Relative validation | Best validation (epoch) | Gap | Minutes | Finished |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 50% | 30700 | 4 | 3840 | 0.2384 | 4.9164 | 0.343 | 1.000 | 0.944 at 2/4 | 2.92× | 108 min | yes |
| 100% | 61299 | 2 | 3832 | 0.4547 | 5.4981 | 0.455 | 1.119 | 0.971 at 1/2 | 2.46× | 104 min | yes |

Relative losses are against the trivial predictor's on the same side — the channel mean, whose error is 4.915 on the validation side and 0.696 on the training side of the smallest share; the gap is their ratio. Best validation is the lowest of the run's epochs, with the epoch it fell at; the verdict reads the last epoch, where every share has spent its budget.

Gains in validation loss share to share: 50%→100% -11.8%.

**not learnt** — at the whole share validation is 1.12× the trivial predictor's: nothing carried over to the held-out side, so the curve reads nothing about the data — the held-out side, or the loss it is scored by, comes before the corpus is judged.

### Reading

The reference shape lowers the held-out loss against the small shape at both shares — 1.000
against 1.072 at the half, 1.119 against 1.293 at the whole — and its best epochs (0.944 at the
half, 0.971 at the whole) are the only satellite runs of either leg that fell under the channel
mean at all. It is still read as not learnt at the last epoch, by the same 0.3 % of tokens that
decide the small shape's runs, and within each run validation is best after one or two epochs and
rises while training falls, as at the small shape. The size axis on this corpus therefore says
that a larger model reproduces the training months' excursions a little better on the held-out
side, and no more: it does not change the reading, which belongs to the loss and the split. What
the reference shape earns on the largest single corpus is not measured here, and the decision on
the shape of single-corpus backbones in ADR-0008 stays provisional.
