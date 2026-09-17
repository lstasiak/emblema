# ADR-0027: Measuring corpus saturation — one budget of steps over nested shares of whole units, the share a parameter of the run, and a verdict read by rules fixed beforehand

- Status: accepted
- Date: 2026-09-16

## Context

The corpus arithmetic (ADR-0008) admitted the reference encoder on a ratio borrowed from language
models: twenty unique observed values per parameter, with the satellite corpus carrying the mix
over that line and nothing to spare. A ratio fitted to text trained for one epoch says little
about telemetry trained for fifty, so the budget was accepted on the condition that a measurement
corrects it before the first full pretraining run: does more of a corpus still lower the loss of a
model trained on it, or has the model learnt what the corpus holds?

Three things had to be settled before that measurement could be run honestly. What a "fraction of
the corpus" is, given that windows of one unit overlap and a split through them leaks. What two
runs over different fractions are compared at, given that a run over a tenth of the data trained
for the same epochs has had a tenth of the compute. And where the fraction lives, given that every
other parameter of scale is written into the experiment and reported with the run (ADR-0022).

The measurement is also the first that costs an evening rather than minutes on the development
machine, so a run had to survive a dropped session and a finished one had to stay finished.

## Decision

**A share of a corpus is a share of its training units, whole, ranked by the run's seed.** The
`CorpusShare` value object takes the lowest-ranked units that fit the fraction, rounded up, each
unit ranked on its own by the same digest the Catalog splits with (ADR-0013), framed apart from
the split so a run seeded like its corpus does not read the units just past the validation cut.
Shares of one seed are nested — a tenth inside a quarter inside a half — so each point of a curve
adds data to the one before rather than drawing afresh. The validation side is never shared: a run
over a tenth is scored on the whole of it, or its loss would not be comparable with a run over all.

**The share is a parameter of the experiment, inherited from the tier and overridable in the
file.** `corpus_fraction` sits beside the shape in the configuration, renders into the parameters
a tracker logs and a signature digests, and travels in the order and result documents. The reader
of a published corpus takes the share with the manifest (`read(manifest, share)`), so a use case
that orders, fulfils or accepts a run reads exactly what the configuration says. The tier table
already stated a share per tier; it now means what it says, and the control experiments state
the whole corpus explicitly because the small tier's tenth is sized for the real corpora.

**Every share spends the same optimiser steps.** An experiment states the budget for the whole
corpus; the report derives, for each share, the epochs that spend the same steps over it — twenty
epochs over a tenth against two over the whole — and the warmup keeps its share of the run rather
than its count of epochs, which is why a warmup may now be a fraction of an epoch. The curve is
then a question about data alone: at one compute budget, does more distinct data lower the loss,
or does repeating less of it do as well? Epochs are whole, so the steps agree to the nearest epoch
and the curve refuses points further apart than fifteen per cent.

**One curve per corpus, on its own vocabulary, at the small tier.** The mix would blur what is
being asked: a curve over four corpora concatenated entangles saturation with the transfer between
them, which is the thesis and not the measurement, and would force decisions about batch weighting
and padding across corpora that belong to the first mixed run. The two corpora that carry
ninety-four per cent of the mix's unique values answer for the mix; the size axis — the reference
shape against the small one at the largest shares — is measured on the accelerator the published
results train on, as that run's rehearsal.

**The verdict is code, with thresholds fixed before any curve was drawn.** `judge` reads a curve in
order of what invalidates what: a run over the whole corpus whose validation loss exceeds its
training loss by more than one and a half times is overfitting, and says nothing about the data
beyond that there is too little of it for the model; a last step up in data that lowers the
validation loss by less than five per cent is a plateau, and the corpus is learnt at this size
and budget; otherwise the loss still falls and the data, not the model, is the limit. Both
thresholds are the ones the masked-reconstruction assessment already reads a run against
(ADR-0020), so one convention judges both.

**Each side is read against its own trivial predictor.** The two sides of a unit-level split are
not equally hard: the satellite corpus's held-out months carry excursions of eighty standard
deviations, and the channel-mean predictor errs five times worse there than on the training side.
A raw ratio of validation to training loss would call a perfect model overfitting. Every point
therefore carries the error of the channel mean on each side, the gap is the ratio of the relative
losses, and the figures draw the relative validation loss, where one is "nothing learnt".

**A run is stored as it goes and picked up where it stopped.** Each run has a directory: what it
was, then each epoch as it is measured, then its outcome. A session that drops leaves the epochs
it managed and the checkpoint the last of them reported; the next session resumes from it through
the training runtime's own mid-epoch resume (ADR-0021), because the run's share is in its
configuration and therefore in the signature the checkpoint carries — no special case is needed to
tell a tenth's checkpoint from a half's. A directory holding an outcome is never trained again,
and the report renders whatever has finished. The figures are drawn from the files alone.

## Consequences

- The signature of every configuration changed: the share and the warmup are now rendered in the
  parameters. The one backbone registered before this, from a dirty tree, is to be redone from a
  clean commit in any case; the order and result documents moved to version 2.
- The port of the published-corpus reader takes the share; both adapters honour it over the
  manifest's training units, the in-memory one over the units it is told about, empty ones
  included, so the contract holds them to one answer.
- The tier table's `corpus_fraction` is read by the training path, not only by the budget
  arithmetic: a smoke run at the small tier reads a tenth of a real corpus by stating nothing.
- The fraction runs of a corpus do not repeat the same windows the same number of times: a tenth is
  seen twenty times, the whole twice. The curve is read at this budget and says so; whether the
  verdict holds at the budget of a full pretraining run is what that run's own curve will show.
- The report runs at the small tier, on the development machine, in one evening; the size axis
  waits for the accelerator and the note carries both legs, dated.

## Alternatives considered

**A share of windows rather than of units.** Simpler, and exact to the window. Rejected: windows
of one unit overlap by eighty per cent, so a share of windows is a share of every unit with its
neighbours held out — the leak the split exists to prevent, now inside the training side.

**Equal epochs for every share.** What "train on ten per cent" most often means. Rejected: a run
over a tenth for the same epochs has had a tenth of the steps, and a loss that keeps falling with
the fraction then measures compute, not data.

**The fraction as a flag of the report.** No change to the core. Rejected: a run over part of a
corpus that reports a configuration saying nothing of it is a run whose provenance lies, and the
next process to read a share would have written the selection again.

**Independent draws per share.** Each point its own random subset of units. Rejected: the curve
would carry the variation between subsets on top of the effect of their size; nesting removes it
and costs nothing, since one seed steers everything.

**A curve over the mixed corpus.** What the first full run trains on. Rejected for this
measurement, kept for the mixed run's own curve: it would answer a question about transfer that
the project asks later and elsewhere, with decisions about weighting not yet taken.

**Reading the verdict off the figure.** Rejected as every verdict in this project is: a threshold
stated after the curve is a verdict written in advance.

## Revisit when

- The size axis on the accelerator disagrees with the small tier's curve: the reference shape at
  the largest share does no better than the small one, or is still data-limited where the small
  one had saturated. The tier table's reference shape is settled by that leg.
- A corpus arrives whose units differ in length by more than an order of magnitude: a share of
  units is then a poor proxy for a share of the data, and the share would need to rank by
  observations rather than count units.
- The mixed run's own curve is wanted: the share then applies per corpus of the mix, which is a
  tuple of shares beside a tuple of inputs.

## 2026-09-17 — a fourth reading, added after the first curves

The first leg (`docs/verification/corpus-saturation.md`, the section of this date) produced a case
the three readings had not foreseen. On the satellite corpus no share, at no epoch, did better on
the held-out months than the channel mean — 1.00 at best, 1.29 at the whole corpus — while every
share learnt its training months to a fifth or a third of their variance. The rules read it as
overfitting, at 3.77×, and prescribed a smaller model or more units. Both would be wrong: the
training side was learnt, and the tenth was no better on the held-out side than the whole. What
the curve shows is a held-out side no model trained on the other months reaches through this loss
— months with excursions of ninety-six standard deviations, where a squared error is decided by
a few windows: one of the twenty-one held-out months holds 88 % of the side's squared magnitude,
three tokens in a thousand hold 91 % of it, and the other tokens are easier than the training
side's.

`judge` therefore reads a fourth verdict before the other three: **not learnt**, when the relative
validation loss at the largest share is at or above one, the channel mean's own error. It is a
precondition, not a threshold chosen — the number is the trivial predictor's — and the three
readings and their thresholds stand as fixed. The addition is made after a curve was drawn, which
the rules said never to do for a threshold; it is declared here and in the note, which keeps the
gap the original rules read beside the new verdict, so that a reader sees what changed and why.
The remedy the new reading prescribes is to look at the held-out side and the loss before judging
the corpus: for the satellite corpus, a validation split that does not put the wildest months on
one side, or a loss that bounds what one excursion can cost.

Two smaller things changed with the same leg. The report's tables carry each run's best epoch
beside its last — what early stopping would have kept — because every run but C-MAPSS's was best
after one to three epochs; the verdict still reads the last epoch, where every share has spent its
budget. And the size axis ran on the development machine after all, in the small tier's budget of
steps, rather than on the accelerator: the accelerator path has not run yet and the curve was
wanted before the first mixed run. The revisit condition above stands; the leg's section is in the
note for the satellite corpus, whose SMD half was stopped before its first epoch.
