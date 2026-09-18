# ADR-0029: Pretraining over a mixture of corpora — one vocabulary chained through the publications, optimiser steps of one corpus each weighed by their count, the best epoch kept by the mean relative validation, and a run that says where it can be picked up from

- Status: proposed
- Date: 2026-09-18

## Context

The first backbone of the published tier was trained over one corpus (`docs/verification/manual-handoff.md`,
section of 2026-09-18), and the corpus arithmetic (ADR-0008) admits the reference shape on the
value of the mix alone: the small shape already learns or overfits every classical corpus on its
own, and only the four corpora together hold the values the shape is priced for. The mixed run is
the first full pretraining, and everything below is what a run over several corpora has to
decide that a run over one never met.

What the four corpora are, as published at the windows the saturation measurement used
(`docs/verification/corpus-saturation.md`; the satellite corpus at the version whose held-out
months are named, ADR-0028):

| Corpus | Window | Training windows | Tokens a window | Tokens an epoch | Share of tokens | Share of windows |
| --- | --- | --- | --- | --- | --- | --- |
| C-MAPSS | 50 / 5 cycles | 20,237 | 1,050 | 21.2M | 11 % | 14 % |
| SKAB | 100 / 10 s | 3,896 | 800 | 3.1M | 2 % | 3 % |
| SMD | 50 / 10 min | 58,411 | 1,900 | 111.0M | 60 % | 41 % |
| ESA-AD | 1 / 1 h | 61,202 | 839 on average, 1,331 at most | 51.3M | 27 % | 43 % |
| together | | 143,746 | | 186.6M | | |

The measurements behind the decisions: C-MAPSS is learnt to the objective's floor from a quarter
of its engines, so passes over it are compute rather than information; SKAB, SMD and the
satellite corpus are data-limited, the last still falling when its budget ends; SMD and SKAB are
best after one to three passes at the small shape, so a run over them needs a rule that keeps
the epoch worth keeping; the satellite corpus's held-out months err five times worse than a
classical corpus's under the trivial predictor, so a mixed validation loss would be theirs
(ADR-0028); the satellite corpus has two independent units under its 61,202 windows, so a count
of its windows is not a count of what it can teach; and windows of the four corpora differ in
length by a factor of 2.4, so a batch that mixes them carries the padding of its longest.

On the platform's accelerator the reference shape spends 0.31 s an optimiser step of thirty-two
C-MAPSS windows in half precision, and the step's cost grows with the tokens and with their
square; an epoch of the mixture prices at about forty minutes and a session there ends after
twelve hours. A run of hours on a session that can drop needs to be picked up, and the platform
has no tracking server to ask which checkpoint to pick up from.

## Decision

**A mixture is a value of the domain: the corpora in the order their vocabulary was chained, and
a run over one corpus is a mixture of one.** `TrainingMixture` holds the `TrainingCorpus` of
each published corpus in a stated order; its vocabulary is the last corpus's, since the chain
below makes every earlier vocabulary a prefix of it; its shape is the tuple of the corpora's
shapes, and a run's signature digests the configuration over those shapes in order. A mixture of
one corpus digests to what a run over that corpus digests to today, so the backbones already
registered and the known-answer signatures are untouched. The backbone records its inputs as a
tuple in the same order, the order handed to the machine that trains names the manifests in that
order, and the result reports the shapes in that order. The reader port is unchanged — it reads
one manifest — and the use cases that order and fulfil compose one read per manifest.

**The vocabulary of the mixture is chained through the publications, and a mixture that is not
a chain is refused.** Each corpus after the first is published with the manifest of the one
before it as the vocabulary to continue, so that channel identifiers never collide: C-MAPSS
holds 1–21, SKAB 22–29, SMD 30–67 and the satellite corpus 68–84, and the mixture's channel
table has 84 rows. Statistics are fitted per corpus and travel in that corpus's manifest. A
`TrainingCorpus` carries the names of the channels its manifest lists, and the mixture's
invariant is that each corpus's names are a prefix of the next's in the stated order: two
corpora published from scratch would both carry identifiers from one, and a run over them
would learn a collision without an error. The names are not part of the corpus's shape, so
they reach no signature. A mixture that leaves a corpus out is still a chain — the names of a
corpus before the gap are a prefix of the names of the one after — so a leave-one-corpus-out
backbone reads the same publications with one manifest fewer, and the rows of the missing
corpus's channels stay at their initialisation.

**Every optimiser step is taken over micro-batches of one corpus, the steps of an epoch are
interleaved in the order the seed gives them, and an epoch is every window of every corpus
once.** A batch of one corpus pads nothing on the three regular corpora and pads the satellite
corpus to its own longest window only; the validation pass scores each corpus on its own loader,
which is the per-corpus entry ADR-0028 asks for, under masks drawn per corpus from its own
first batch, so that a corpus is scored on the same hidden tokens whichever mixture it is in;
and the micro-batch that fits the device is set by the longest window of the mix, 1,900 tokens
of SMD, at sixteen windows with two accumulated into a step of thirty-two, the effective batch
of the single-corpus backbones. The two accumulated are of one corpus: the interleaving takes
turns of as many micro-batches as a step accumulates, because the gradient of a step is the mean
over the step's tokens, and a step that accumulated a 1,900-token SMD batch with an 839-token
satellite batch would weigh the two by their tokens inside the step — over the mix, the
satellite corpus at 35 % and SMD at 49 % rather than the shares below. What is left of a corpus
once its full turns are taken, fewer micro-batches than a step, closes the epoch after every
full turn, so at most two of an epoch's four and a half thousand steps mix corpora. **What follows is the weight of each
corpus, and it is stated rather than implied: under an optimiser that normalises the gradient,
a corpus weighs its share of the optimiser steps, which is its share of micro-batches — 43 %
the satellite corpus, 41 % SMD, 14 % C-MAPSS, 3 % SKAB — and not its share of tokens.** The
satellite corpus's two units take the largest share; the per-corpus validation and the rule
below are what guard it, and no weight is a parameter of this run: a weight would be the first
parameter of the mix, and the first run measures the mix before any dial on it is earned.

**The run trains a fixed number of epochs and keeps the epoch whose mean relative validation over
the corpora is lowest.** The rule reads the aggregate ADR-0028 declared as the default — the mean
over corpora of each corpus's loss against its trivial predictor's — so that no corpus decides
it by the size of its values or its windows. The weights of the best epoch so far are written
durably when an epoch improves on it, and the backbone the run reports is the best epoch's,
named on the last epoch with the epoch it came from. The rule is the runtime's, not the
mixture's: a run over one corpus keeps its best epoch by the same reading, which over one corpus
is its lowest validation loss, and the run's signature does not carry it, since the signature
names the configuration and the data and the rule is the code's, which the backbone names by its
commit. The three C-MAPSS backbones registered before the rule fell through every one of their
four epochs (`docs/verification/manual-handoff.md`, section of 2026-09-18), so their last epoch
is the epoch this rule keeps and their weights are the weights it would have kept. The run is
not stopped early: the learning rate decays to its floor over the stated epochs, and a run cut
off by a patience rule would end at whatever rate the cut fell on, which is another budget for
every run that stops elsewhere. The first mixed run states eight epochs, priced at about six
hours on the platform with the estimate's factor of three either way; a run that outlives its
session is picked up in the next.

**The reading of the loss is the bounded one, with the knee at one deviation, over the whole
mixture, and dropout is zero.** The reading is ADR-0028's; one reading for the run, since the
bound was decided per run there. Dropout is zero for the reason the shape is common (ADR-0008,
section of 2026-09-18): the single-corpus backbones the mixed one is compared against were
trained without it, and a transfer matrix whose backbones differ in regularisation reads it in
every cell. What overfitting the classical corpora show is handled by the epoch kept, not by a
parameter the comparison would have to carry.

**The window of each corpus lives in its manifest, and the experiment names corpora alone.** The
four corpora are windowed in four units — cycles, seconds, minutes, hours — and each manifest
states its window in its own unit; the experiment file lists the corpora in chain order, the
order names the manifests, and the verification note signs each corpus's window beside its
numbers. SMD enters at 50 / 10 minutes, the window the saturation curve was measured at and the
cheaper of the two published, as the first dial of scale ADR-0008 named.

**The run says where it stands and where it can be picked up from.** The training runtime logs,
through the standard library's logging, a line every so many optimiser steps — the step of the
epoch, the loss since the last line, the seconds a step and the time left in the epoch and in
the run at this session's pace, an interval the command line sets — one line for every
checkpoint it writes — the reference, the step and the
seconds since the session began, which a resumed run counts afresh — and one line at the end of
every epoch with the loss of each corpus. The command line sends the log to standard error. The platform keeps the log of a
session that ended on its limit, so the reference a dropped run is resumed from is read off the
log and passed as the checkpoint to resume from, and the epoch lines are the curve's copy if the
result never arrives. The checkpoint interval of the mixed run is two thousand steps, about
twenty minutes and sixty megabytes each, so that a session that drops loses at most that and the
transient prefix holds a few dozen states rather than a few hundred.

## Consequences

- The domain gains `TrainingMixture` and `TrainingCorpus` gains its channel names; the backbone's
  input becomes a tuple and the table of inputs gains a position in its key, with the rows
  already there moved to position zero. The order and result documents move to their next
  versions; the backbones registered before this record decode as they are, since the
  configuration document does not change.
- `TrainingRuntime.train` takes a mixture; the replaying runtime, the in-memory runtime, the
  contract tests and the report scripts wrap a corpus in a mixture of one. The runtime holds one
  training and one validation loader per corpus and interleaves the training ones in turns of a
  step's micro-batches; the processes it is given to collate batches are per loader, so a run
  over four corpora with two workers each runs eight.
- Every run keeps its best epoch from here on, the report scripts' runs included: a report that
  diagnoses the weights a run left diagnoses the best epoch's, where before it diagnosed the
  last epoch's, and a run whose validation rises writes an artifact for every epoch that
  improved on the one before.
- An epoch of the outcome names the weights it kept, where it kept any, and the outcome says
  which epoch its backbone came from; the invariant that the backbone is stated on the last
  epoch alone stands, and a resumed run may end with weights an epoch before its checkpoint
  wrote, which no epoch of its own names. A checkpoint carries the best so far, and one due on
  an epoch's last step is written once the epoch is scored, so that a run picked up at the
  boundary keeps what that epoch kept.
- The experiment file names `corpora` in chain order; the files already written name one corpus
  each and change in that field alone, which no signature covers.
- The standard library's logging is used for the first time in the package, by the runtime; the
  structured logging the stack names arrives with the infrastructure it is for. The order
  command takes one manifest per corpus the file names, in the file's order.
- The publications for the mixed run are a chain against the remote bucket — SKAB, SMD at
  50 / 10, the satellite corpus with its named months — continued from the C-MAPSS manifest
  already there, so the C-MAPSS backbone and the mixed one share channel identifiers and the
  first comparative run reads both on the same tokens.
- Every number of the satellite corpus in the mixed run's note is a number over two satellites,
  as its weight in the steps makes plain.
- The refusal of a micro-batch that cannot fit the device before the first step is not made here:
  the estimate has no calibration for the platform's accelerator in half precision, and the mixed
  run is the measurement that calibrates it.

## Alternatives considered

**Mixed micro-batches, weighed by tokens.** A batch drawn from every corpus gives each corpus
its share of tokens in every gradient, which is what a loss summed over tokens means. Rejected:
a batch that holds an 800-token SKAB window beside a 1,900-token SMD window pads the first to
the second, and buckets by length that avoid the padding group by corpus anyway; the weighting
by tokens would then be nominal, and the padding real.

**Sampling with a temperature, as multilingual pretraining does.** A corpus drawn with
probability proportional to its size to the power of α flattens the mix — at α = 0.5 SKAB rises
to about 6 % and C-MAPSS to 18 %. Rejected for the first run: it draws with replacement and
loses the epoch as a unit, C-MAPSS is learnt from a quarter of itself and gains nothing from
more of its passes, and SKAB is three million values whatever its draw. The record's first
revisit condition is where a dial like it enters.

**A weight per corpus in the experiment file.** Rejected for the same reason: no run has yet
measured the mix at its natural weight, and a parameter added before its first measurement
would be a dial nobody has turned. A weight later is a multiplier on a corpus's micro-batches an
epoch and enters the parameters the signature digests.

**Stopping on validation with a patience.** Rejected: it ends the run where the learning rate
happens to be, so two runs that stop at different epochs trained under different schedules; a
fixed budget with the best epoch kept compares runs at one schedule and still keeps what the
patience rule would have.

**A budget in optimiser steps rather than epochs.** Rejected: the epoch is what says every
window was seen once, and a mixture whose corpora differ in size by a factor of fifteen needs
that unit more, not less; the saturation measurement had reason to fix steps across shares of
one corpus, and this run has none.

**A checkpoint pointer at a known key, or a hosted tracker.** The store is addressed by content
and a pointer would need a named key beside it, a second way to address the store for one use;
a hosted tracking server is a service and a credential more on a platform that has neither.
Rejected in favour of the log the platform already keeps.

**A reader adapter over several manifests.** Rejected: the port reads what a manifest describes,
and a mixture is a composition of reads the use cases already make one of; an adapter that read
several would have to know the chain, which is the domain's invariant to check.

**A unit label on the window specification.** Rejected: the specification is the Catalog's, in
the corpus's own time unit, and the mix never compares windows across corpora; a label would be
carried by every window for a comparison nothing makes.

**Dropout at a tenth, the usual default.** Rejected for this run: it is a convention without a
measurement here, and it would separate the mixed backbone from the single-corpus ones by a
parameter the comparison cannot subtract.

**SMD at 100 / 20 minutes.** Rejected: twice the tokens a window at four times the attention,
a micro-batch of eight, and no curve measured at it.

## Revisit when

- The per-corpus validation of the mixed run shows a corpus rising while the mean over corpora
  still falls: a weight per corpus is then the first parameter of the mix, and the satellite
  corpus's share of the steps the first value to turn.
- The mean relative validation is still falling at the eighth epoch over the whole mix: the
  budget grows before the shape does, and if it still falls at the larger budget the sweep past
  the reference shape reopens (ADR-0008).
- The mixed run measures the device's memory at the reference shape in half precision: the
  refusal of a micro-batch that does not fit is then written with that calibration.
- A corpus arrives whose windows are not published under the chain — a second vocabulary for
  one corpus, or a corpus tokenised elsewhere: the mixture's invariant is then a mapping between
  vocabularies rather than a prefix.

## Sources

- Kingma, D. P. and Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.
  arXiv:1412.6980 — the per-parameter normalisation that makes a corpus's weight its share of
  steps rather than of tokens.
- Arivazhagan, N. et al. (2019). Massively Multilingual Neural Machine Translation in the Wild:
  Findings and Challenges. arXiv:1907.05019 — temperature-based sampling over corpora of
  unequal size.
- Conneau, A. et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale.
  ACL 2020 — the same sampling in self-supervised pretraining, with α = 0.3.
