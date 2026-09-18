# ADR-0028: Bounding what one excursion costs — the objective's loss is a parameter of the experiment, a mixed run reports validation per corpus, and the satellite corpus is split so that its excursions lie on both sides

- Status: accepted (2026-09-18, on the measurement in the dated section below)
- Date: 2026-09-17

## Context

The saturation measurement (ADR-0027, `docs/verification/corpus-saturation.md`) read the
satellite corpus as *not learnt*: at every share, at both shapes, the validation loss stayed at
or above the channel mean's, while every share learnt its training months. The diagnostic that
followed (the third section of the same note) scored the six stored backbones on the tokens their
runs were judged on and split the loss. It found three things.

The ordinary behaviour of the corpus was learnt: over the twenty held-out months other than the
one holding the excursions, the model's error is a quarter to a half of the channel mean's at
every share. The verdict belonged to the loss and the split.

The loss on the ordinary tokens nevertheless rises with the share of the corpus, from the quarter
share up, on every ordinary month of the larger mission. Under a squared error the training
gradient of this corpus is 45 % from 0.13 % of its tokens — excursions of ten to two hundred
standard deviations that no month repeats — and, the shares being nested, it can be said which
share brings them: the four heaviest training months enter at the whole corpus and at no smaller
share, which is where the ordinary-token loss jumps. The whole-corpus model reproduces
excursions of 10–20 deviations seven times better than the mean and errs half again as much as the
tenth-share model on the nine tenths of tokens within one deviation. Read under a Huber loss, the
same backbones sit flat across the shares: what the squared error moves between shares is the
excursions' doing. The curve was measuring the objective, not the data.

The month holding 88 % of the held-out side's squared magnitude, the first weeks of a mission, is
not learnable from the other months under this split even on its ordinary tokens: the model reads
the excursions in a window's visible context and predicts excursions for the hidden ordinary
tokens beside them, four and a half times worse than the channel mean at the whole share and
worse at each larger share. No training month looks like it.

The first mixed run is next, and the satellite corpus enters it as an ingredient (ADR-0008).
Three things follow for that run, and each is a decision about where a parameter lives rather
than what its value is: the objective's loss, what a run over several corpora reports, and how
the satellite corpus is split.

## Decision

**The objective's loss is a parameter of the experiment: a squared error, or a Huber loss with
its knee stated.** `ExperimentConfiguration` gains the reading the reconstruction is scored by —
`mse`, or `huber` with `huber_delta` in normalised units — rendered in `parameters()` like every
other parameter of the method, so it reaches the tracker, the signature, and the order and result
documents; the experiment file states it in an `[objective]` section. The training runtime scores
the hidden tokens by it and the validation loss it reports is under the same reading. The mixed
run and the satellite curve use the Huber loss with the knee at one deviation: within a deviation
of the target the loss is the squared error the trivial baselines are least-squares estimators
for, so on the nine tenths of tokens that lie there the objective is unchanged, and past it the
gradient of a token is bounded by the knee — an excursion of a hundred deviations pulls as hard as
a miss of one. The control experiments and the classical corpora keep the squared error, so their
verdicts and the numbers already recorded are not moved by this record.

**The target is what the tokeniser produced; the bound is in the objective.** The Catalog's
windows keep their raw normalised values — the anomaly task on this corpus is scored on excursions
and needs them — and the objective bounds what one costs to learn. Clipping the target was the
other candidate and is rejected below.

**A run over several corpora reports its validation loss per corpus, each against its own trivial
predictor.** `EpochOutcome` carries, beside the loss over every validation token, one entry per
corpus of the mix: the corpus's name, the loss over its hidden tokens under the run's reading, and
the trivial predictor's loss over the same tokens, so the relative loss the saturation note reads
is a division and not a lookup. The tracker logs each as a metric of its own. The held-out months
of the satellite corpus err 4.9 under the channel mean where the classical corpora err about one;
a mixed loss would be theirs. Where a run stops or keeps its best epoch by validation, it reads a
stated aggregate of the per-corpus relative losses — the mixed run's experiment states which, and
the mean over corpora is the default, so that no corpus's magnitude decides it.

**The satellite corpus is republished with its held-out months stated, chosen so that the months
holding the excursions lie on both sides.** The Catalog's split accepts an explicit list of
held-out units beside the seeded fraction it draws today; the manifest already lists the
validation units, and a version split by list records no split seed. The list for the satellite
corpus is derived by a rule from the block's own values, as the diagnostic weighs them: the months
are ranked by the mean square of their tokens under the current normalisation; the months past one
— the unit the training side is normalised to, which today names fifteen of the hundred and five,
four of them past ten — alternate between the sides in rank order, the heaviest on the training
side, so that a model still meets the kind of behaviour it will be asked about; and the remaining
months are drawn by the seed so that the held-out side keeps the stated fraction of the corpus.
The rule, the list it produced and the version it produced are recorded with the publication,
before any run over the new version. It is a new version of the
corpus, with a new manifest; runs over the old version stay comparable with each other and with
nothing else.

**The satellite curve is run again under the bounded loss before the corpus's share of the mix is
judged.** The verdict rules of ADR-0027 are unchanged; what they are given is a curve whose points
differ by data rather than by excursions.

## Consequences

- The signature of every configuration changes once more: the reading of the loss is rendered in
  the parameters. The order and result documents move to their next version. The one backbone
  registered from a dirty tree was to be redone in any case.
- The reconstruction loss takes its reading from the configuration rather than being the squared
  error by construction; the trivial baselines and the assessment (ADR-0019, ADR-0020) are scored
  under the run's reading, since a model and the baselines it is measured against are measured the
  same way — the rule stands, the reading is what the run says. The interval machinery sums
  per-unit losses and holds for any additive loss.
- The epoch outcome, the tracker port and its adapters, the result document and the persisted
  run gain the per-corpus entries; a single-corpus run has one entry, which is its loss.
- The Catalog's split gains a second constructor and the publish command a way to state the list;
  the manifest's split seed becomes optional. The domain invariant is unchanged: both sides hold a
  unit and share none.
- The satellite curve costs another evening on the development machine; it is optional until the
  corpus's weight in the mix is decided, and required before it is.
- A backbone trained under the Huber loss reports a loss in Huber units; the saturation note's
  relative losses are read against the trivial predictor's Huber loss, as the diagnostic already
  reads them, so one is still nothing learnt.

## Alternatives considered

**Clipping the target at k standard deviations in the objective.** Keeps the squared error and
the baselines' units, and bounds an excursion's error at the clip. Rejected: the bound is 2k on
the error and 4k² on the loss — at ten deviations an excursion still weighs four hundred ordinary
tokens — and it asks the model to predict a plateau that is not in the data; the diagnostic's
clipped reading of the stored backbones was uninformative for the same reason. The Huber loss
bounds the gradient at the knee and asks nothing of the target.

**Clipping or winsorising the values in the tokeniser.** Rejected: the tokeniser's windows serve
the anomaly task on this corpus, which is scored on the excursions; a bound in the tokeniser
would be a second corpus under one name.

**Dropping the commissioning months from the corpus.** Rejected: which months are commissioning
is a judgement about the data the reader was written not to make (ADR-0025); the split and the
loss are where a month that looks like nothing else is handled, and the anomaly benchmark's own
protocol keeps its windows.

**A single mixed validation loss, weighted by tokens.** Rejected: the satellite corpus's held-out
tokens err five times worse than a classical corpus's under the trivial predictor and would decide
the number, and a loss that one corpus decides cannot stop a run over four.

**A hold-out stratified by month mass drawn inside the Catalog.** The mass of a month in
normalised units depends on statistics fitted on the training side, which depends on the split;
a stratification computed inside the publication would either fit statistics twice or rank on raw
values with a unit per channel. Rejected in favour of a stated list, derived by a stated rule from
a published version's block, so that the rule is reproducible and the list is data of the
publication.

**Keeping the squared error and reading the satellite corpus off the ordinary tokens only.**
Rejected: a loss read apart from what was trained under it says what the model happened to learn,
not what the objective asked of it; the diagnostic shows the objective spending half its gradient
on tokens no held-out month repeats, and a reading does not change what the gradient does.

## Revisit when

- The satellite curve under the Huber loss is drawn: if it still reads *not learnt* on a split
  that spreads the excursions, the corpus's difficulty is not its loss and its share of the mix is
  reconsidered under ADR-0008.
- The mixed run's per-corpus validation shows the knee mattering: the knee is then a dial of the
  ablation, not a constant of this record.
- The classical corpora's own curves are re-run under the Huber loss and their verdicts move: the
  squared error then stops being the default for anything.
- A corpus arrives whose excursions are its signal rather than its anomalies: the bound is then
  per corpus in the mix, not per run.

## 2026-09-18 — measured

The curve the first revisit condition waited for is drawn (`docs/verification/corpus-saturation.md`,
section of 2026-09-18): the small shape over the satellite corpus, at the first leg's budget to
the step, under the Huber loss with its knee at one standard deviation, over the version whose
held-out months were named. It reads **data-limited**: 0.555, 0.474, 0.397 and 0.270 of the
channel mean's loss on the held-out months at a tenth, a quarter, half and the whole of the
training months, each share lower than the one before, the gap between the sides closing from
3.8× to 1.2×, and the whole share still falling when its two epochs end. The same shape at the
same steps under the squared error over the drawn split read *not learnt* at every share.

What it settles: the satellite corpus's difficulty was its loss and its split, not its data, and
the corpus enters the mixed run as an ingredient that is still learning at the whole of itself.
The record's three decisions stand as made; the first revisit condition is spent. What it does
not settle: the knee is one value, measured once, and its weight in the mix is the mixed run's
own decision. The status moves to accepted on this measurement.

## Sources

- Huber, P. J. (1964). Robust Estimation of a Location Parameter. Annals of Mathematical
  Statistics 35(1), 73–101.
- Kotowski, K. et al. (2024, v2 2025). European Space Agency Benchmark for Anomaly Detection in
  Satellite Telemetry. arXiv:2406.17826.
