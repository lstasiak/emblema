# ADR-0018: The positive control is a corpus reader — one latent process, two sensor layouts, disjoint trajectories

- Status: accepted
- Date: 2026-09-13

## Context

The thesis of this project is that a backbone pretrained without labels on heterogeneous sensor
streams learns something that transfers to a sensor layout it has never seen. A negative result is
publishable — real data may simply hold no transferable structure, or not enough of it — but only
if one alternative has been ruled out first: that the pipeline could not find the structure even
when it was there. Without that, every disappointing number is ambiguous, and the ambiguity is the
project's largest risk, because it is resolved by months of looking rather than by an experiment.

A positive control resolves it. A corpus in which shared latent structure is present by
construction gives a question with a known answer: if transfer fails there, the fault is in the
implementation and nothing below it is worth running. The control is also the cheap gate before
every expensive run, because it needs no download and no accelerator.

What existed before this record: a corpus reader port whose adapters read files, a tokenisation
scheme, a published-corpus format, a loader and an encoder. What did not exist: any corpus whose
answer we know, any data with timeless tokens (the one real corpus has no static features), and any
pair of corpora with disjoint channel sets — the subsets of the one real corpus share all of theirs.

The constraints this record works inside: a corpus version is identified by the checksum of the data
read, so the same specification must produce the same bytes on every machine; observations of one
unit arrive in non-decreasing time order; the block a published corpus is written into rejects two
observations closer together than a small fraction of the window length; every scale parameter lives
in configuration rather than in code; the domain, ports and application layers import no third-party
library.

## Decision

**The generator is an adapter of the corpus reader port, not a script beside the pipeline.**
`SyntheticCorpusReader` describes, lists units and streams observations like any other reader, so
the control travels the same road as real data — registered, frozen as a version, tokenised, split,
archived, published, loaded, encoded — through the same use cases and the same command line. A
failure anywhere along that road is a failure the control catches, which a generator that wrote
tensors into a test would not. It also means the control needs no new concept: it is a corpus.

**Nothing is stored; a unit is generated when it is asked for.** The specification is the data, so
the corpus exists on any machine that has the code, costs no download and runs in continuous
integration. Each unit's randomness is addressed rather than streamed — a seed and the names of what
the numbers belong to are hashed into the state they are drawn from, through the same rule the
project turns a seed into an order with — so a unit can be regenerated without the units before it
and a reader that streams one unit at a time stays possible for a corpus of any size.

**Two layouts share the latent process and no trajectory.** The hidden factors are sums of
harmonics whose frequencies are drawn once, from the seed of the process; their amplitudes and
phases are drawn per unit. Two layouts built on one process therefore share the structure — the
frequency content, the way the factors behave — while every unit of each is its own realisation.
Sharing the realisations instead would have made transfer easier to demonstrate and impossible to
interpret: a unit of the second layout would carry a latent signal the model had already seen in
the first, so a model could recognise a memorised trajectory rather than carry structure across. The
project refuses that leak elsewhere — the real corpus's reader registers the training trajectories
only, so a version's checksum is also the proof that no evaluation unit took part in pretraining —
and refusing it here costs one seed. The diagnostic variant is still one edit away: equal
trajectory seeds make the two layouts watch the same realisations, which is the ceiling of what
transfer could achieve and worth reaching for only if the control fails.

**The factors are defined for every instant, not over a grid.** The layouts sample at instants of
their own, so the process has to answer at any time; a grid would have forced the two layouts onto
a common cadence and quietly removed the axis of heterogeneity the control exists to exercise.

**Coupling is mixed variance-preservingly against private factors.** A channel's signal is
`√s · shared + √(1 − s) · private + noise`, where the private factors are built exactly like the
shared ones but drawn under the layout's own seed. At `s = 1` a channel follows the shared factors;
at `s = 0` it follows factors no other layout sees, and it has the same amount of signal to follow.
Setting the shared part to zero without replacing it would have made the null corpus a corpus with
less signal as well as less shared signal, and a failure to transfer there would have had two
explanations. The null pair is generated from the control pair by turning that one dial: the same
sensors, the same instants, the same observations dropped, the same noise.

**A channel responds to a few factors, with weights scaled to unit norm.** No single channel carries
the whole state, so a model has to assemble the picture from several — which is the structure
transfer is supposed to carry — and every channel's shared signal has the same size whichever
factors it was given.

**Time runs on a grid of whole steps and values are emitted at a fixed precision.** An instant is a
step index times a step length, which is exact floating-point arithmetic and identical on every
machine; a value passes through a sine, which need not agree to the last bit between processors and
library builds, so it is rounded to six decimals — the precision of the corpus, as a file format
would have one. Together these make the checksum of a generated corpus a property of its
specification rather than of the machine that generated it, which is what the version identity of a
corpus requires. The grid has a second effect: two channels either share an instant exactly or lie a
whole step apart, so a window can always be archived at the precision the block is written in.

**Each unit carries one timeless channel, a gain that scales its shared signal.** The corpus is the
first in the project to have a static feature, so it is the first to exercise the timeless token end
to end on data — the round-trip check now holds a static feature to the same tolerance as a
measurement. The gain being informative rather than decorative is deliberate: a static feature the
model can use is a static feature whose handling can be wrong in a way that shows.

**Four named corpora, not command-line dials.** `control-a` and `control-b` are the coupled pair;
`null-a` and `null-b` are their uncoupled twins. A control whose parameters can be turned from a
command line is not a control — two runs would not be comparable without recording the flags — so
the presets are constants beside the generator, like the subset names of a file-backed reader, and a
new variant is a new name rather than a new flag. The layouts differ on both axes of heterogeneity
at once: eight channels sampled on every step against five sampled irregularly and apart from each
other, with more noise and more lost.

**Which adapter reads which corpus is wired in the composition root.** The registry of corpus facts
keeps what no adapter can know — publisher and licence — and the root keeps the choice of adapter,
because the root exists so that the adapter a use case was given can be read in one place. The
directory a corpus is read from became a default under the corpus's own name rather than a required
argument, since a generated corpus has nothing to unpack.

**The control is certified by a permutation test before anything uses it.** For every channel of
every unit, the channel's values are fitted by least squares on its own unit's factors, and again on
another unit's. The second fit is not zero — every unit shares the frequencies of the process, so a
sinusoidal basis explains part of any channel — and the statistic is the excess of the first over the
second: a channel following its own realisation specifically. Measured on the presets, a coupled
corpus recovers about +0.73 and an uncoupled one about ±0.01, against thresholds of 0.4 and 0.1. The
argument the control exists to make depends on the structure really being there; asserting it in a
test is what keeps the claim from being a comment.

## Consequences

- The control runs everywhere. No download, no accelerator, no marker: the generator's tests and the
  certificate are part of the ordinary suite, and the whole synthetic package is covered.
- The published-corpus path is exercised by two corpora with **disjoint channel sets** for the first
  time. The second continues the first's vocabulary, so one model holds both, and a test puts
  windows of both layouts in one batch through the encoder and checks that the gradient reaches the
  channel embeddings of each. That is the pretraining leg: the road is open, and what remains for
  the self-supervised objective to add is the objective.
- The timeless token now has coverage on data. The sanity report fits the statistics of static
  features, includes them in its channel diagnostics, and holds their round trip to the tolerance;
  the largest error over a drawn window is at the resolution of the double.
- The corpus-saturation budget file is untouched: the control is not part of the pretraining mixture
  and has no business in arithmetic about it. The sanity report says so instead of failing on a
  missing key, and takes its window from the command line for a corpus that has none.
- A numpy upgrade would change the bit stream the corpus is drawn from and therefore its checksum.
  That is the behaviour a changed corpus should have — a new version, not a silent substitution —
  and the lock file pins the version, so it happens deliberately. Only uniform draws are taken from
  the bit generator and the normal draws are built from them here, so no change to a distribution's
  algorithm can move the corpus underneath a pinned bit stream.
- The control does not prove the thesis and is not evidence for it. It makes a negative result
  interpretable, and it is the gate before an expensive run rather than a result of one.

## Alternatives considered

- **A script that writes synthetic files a file-backed reader then reads.** Rejected: it buys
  nothing the port does not already give, and it puts a corpus on disk that has to be fetched,
  versioned and kept in step with the code that wrote it.
- **Fixtures inside the tests of the pretraining context.** Rejected: the control would then test
  the model rather than the pipeline, and the road from a corpus to a batch — the part most likely
  to be wrong — would be the part it skipped.
- **Shared trajectories between the layouts.** Rejected above: a stronger signal bought with a leak.
  Kept reachable as a diagnostic ceiling, at the cost of nothing.
- **A null made by setting the shared part to zero.** Rejected: it changes the amount of signal as
  well as its origin, so a failure to transfer would have two explanations.
- **Private factors from a different frequency band.** Rejected: it would let a coupled channel be
  told from an uncoupled one by its spectrum alone, and a model pretrained on one null layout would
  stop resembling a model pretrained on the other.
- **Hashing the specification instead of the generated data.** Rejected: the checksum of a corpus
  version means the bytes that were read, and a specification hash would not change when the code
  that turns it into data changes.
- **Certifying the control by fitting the factors from the channels rather than the other way
  round.** Rejected: it needs the channels to share instants, which the irregular layout refuses by
  design, and it would have had to be abandoned for exactly the corpus that matters most.
- **Reporting the raw share of variance the shared factors explain, without the permutation.**
  Rejected on measurement: an uncoupled channel reaches 0.6 on a short unit purely because a
  five-parameter sinusoidal basis fits any smooth curve, and individual channels reach higher still
  by chance alignment of frequencies. The excess over a matched shuffle is the statistic that
  separates the two populations cleanly.
- **Generator parameters as command-line flags.** Rejected: a control has to be the same control
  twice.
- **A polymorphic registry entry that knows both a corpus's provenance and its reader.** Rejected:
  it moves the choice of adapter out of the composition root, whose stated purpose is to be the one
  place that choice can be read, and it would have introduced the first inheritance hierarchy in the
  source tree to express what a wiring table expresses plainly.

## Revisit when

- The control fails. Then the shared-trajectory pair is generated as a ceiling — if transfer fails
  even there, the fault is upstream of anything to do with structure — and the dials of the presets
  (fewer factors, less noise, more units) are the next thing to move, not the thresholds.
- The self-supervised objective arrives. The certificate answers "is the structure there"; the
  objective adds "does the model find it", and the trivial-baseline diagnostics it brings are the
  natural place to measure the control against an interpolation baseline per mask type.
- A layout is wanted whose channels are neither a linear projection of the factors nor noisy in a
  Gaussian way. Then the projection becomes a component with alternatives rather than a matrix, and
  the record says which non-linearity and why.
- A second generated corpus is added for a different purpose, such as verifying the statistical
  procedure. Then the shared parts of these three modules are worth a second look;
  until then the roles are disjoint and so is the code.

## Sources

- Ojala, M. and Garriga, G. C. (2010). Permutation Tests for Studying Classifier Performance. JMLR
  11, 1833–1863.
- Zhang, C. et al. (2017). Understanding Deep Learning Requires Rethinking Generalization. ICLR.
- Adebayo, J. et al. (2018). Sanity Checks for Saliency Maps. NeurIPS.
- Bai, J. and Ng, S. (2002). Determining the Number of Factors in Approximate Factor Models.
  Econometrica 70(1), 191–221.
- Shukla, S. N. and Marlin, B. M. (2021). Multi-Time Attention Networks for Irregularly Sampled
  Time Series. ICLR.

### 2026-09-20 — the specification moves to shared code, the reader stays

The transfer leg needs the exact reading of a sensor at an instant, answered on the Evaluation
side, and Evaluation may not import the Catalog's adapters. The latent process, the layouts,
the draws and the presets now live in `shared/adapters/synthetic`, with the noiseless signal
model extracted from the reader as `SensorSignal`; `SyntheticCorpusReader` stays in the Catalog
and adds the noise, the gaps and the rounding over it. The decision above holds in every
particular — the generator is still an adapter of the corpus reader port, nothing is stored,
the presets are still constants — and the bytes of every corpus are unchanged, which the pinned
checksums assert. The forecasting task the leg poses, and why the truth is read where it is,
are in ADR-0033.

### 2026-09-21 — the transfer leg read: the window, the family of the signals, and nothing from noise

The transfer leg ran at the endpoint's budget on wide second layouts (ADR-0033). What bears on this
decision is below; the runs, their intervals and the order they were made in are in
`docs/verification/synthetic-transfer.md` and `docs/preregistration.md`.

- **The window was the fault.** At a window of 32 against factor periods of 24 to 300 both pairs
  failed, and so did the ceiling — `control-b-shared`, the second layout over the first's own
  trajectories — which put the fault above the data. At 128 the coupled pair passes its rule:
  full fine-tuning 0.094 below a fresh encoder, interval [+0.089, +0.099], floor 0.053. The window
  must span the process's time scales in the pretext and the task together: a backbone pretrained
  at 128 is a worse start than none on a task at 32.
- **The null pair shares the family of the signals.** At a coupling of zero a channel follows a
  private factor built like the shared ones, in the same band with the same harmonics, and a
  backbone carries that family. Swapping the backbones between the pairs changes the advantage by
  at most 0.012, the shared frequencies' own share (+0.011) is a fifth of the floor, and nothing
  leaks between the pairs. The null pair's equivalence rule was withdrawn after its measurement,
  post hoc and named so; the pair controls leakage and pairing, which it passes.
- **Nothing is found where nothing was put.** `noise-a`, the null pair's first layout with its
  signal drowned, pretrains a backbone that is a worse start than a fresh encoder on both tasks
  (−0.025 and −0.011).
- **One model holds both vocabularies only once it is grown to them.** The backbone is pretrained
  on the first layout alone, so its channel table grows rows for the channels a task's corpus
  adds (ADR-0033).

The positive control holds: what transfers at this tier and budget is the family of the signals,
the shared frequencies adding a fifth of the floor. A rebuilt null pair needs a different family,
and two asymmetries of the present layouts: a coupled channel is six sinusoids and a private one
three, and the fresh encoder's spread over seeds at 128 is four times larger on the coupled task.
