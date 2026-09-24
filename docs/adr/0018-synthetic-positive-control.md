# ADR-0018: The positive control is a corpus reader — one latent process, two sensor layouts, disjoint trajectories

- Status: accepted
- Date: 2026-09-13; amended 2026-09-20, 2026-09-21
- Full text before condensation: commit `5f14447`

## Context

The thesis: a backbone pretrained without labels on heterogeneous sensor streams learns something
that transfers to an unseen sensor layout. A negative result is publishable only if we rule out
that the pipeline cannot find structure even when it is there. A corpus with shared latent
structure by construction gives a question with a known answer, needs no download or accelerator,
and gates every expensive run. No such corpus existed, nor data with timeless tokens, nor two
corpora with disjoint channel sets.

## Decision

- **The generator is an adapter of the corpus reader port** (`SyntheticCorpusReader`), so the
  control travels the same road as real data: registered, frozen, tokenised, split, published,
  loaded, encoded.
- **Nothing is stored.** A unit is generated on request; its randomness is addressed by hashing the
  seed and names (the `seeded_rank` rule), so any unit is regenerated alone.
- **Two layouts share the latent process, not trajectories.** Factor frequencies are drawn once per
  process; amplitudes and phases per unit. Shared realisations would let a model recognise a
  memorised trajectory. Equal trajectory seeds remain a diagnostic ceiling.
- **Factors are defined for every instant**, so each layout samples at its own instants.
- **Coupling is variance-preserving**: `√s · shared + √(1 − s) · private + noise`, private factors
  built like shared ones. The null pair turns one dial (s = 0) and keeps everything else.
- **Each channel responds to a few factors**, weights at unit norm.
- **Time on whole steps, values rounded to six decimals**, so the checksum is a property of the
  specification, not the machine.
- **One timeless channel per unit**, a gain on its shared signal — the first static feature on data.
- **Four named presets**, not flags: `control-a`/`control-b` (coupled), `null-a`/`null-b`. The
  layouts differ in both axes: eight regular channels against five irregular, noisier and gappier.
- **The composition root chooses the reader**; the registry keeps publisher and licence.
- **A permutation test certifies the control**: per channel, least-squares fit on its own unit's
  factors minus the fit on another unit's (Ojala and Garriga, 2010). Coupled ≈ +0.73, uncoupled
  ≈ ±0.01, against thresholds 0.4 and 0.1.

## Consequences

- The control runs everywhere, in the ordinary suite.
- Two corpora with **disjoint channel sets** share one vocabulary; a test puts both layouts in one
  batch and checks gradients reach each channel embedding.
- Timeless tokens are covered end to end.
- A numpy upgrade changes the bit stream and so the checksum — a new version, deliberately. Only
  uniform draws come from the generator.
- The control does not prove the thesis; it makes a negative result interpretable.

## Alternatives considered

- *A script writing files*: a corpus on disk to fetch and keep in step with its code.
- *Test fixtures*: would skip the road from corpus to batch.
- *Null by zeroing the shared part*: less signal as well as less shared signal.
- *Private factors in another band*: coupling readable from the spectrum.
- *Hashing the specification instead of the data*: the checksum means bytes read.
- *Raw explained variance without permutation*: an uncoupled channel reaches 0.6 by curve fitting.
- *Generator flags on the CLI*: a control must be the same twice.

## Revisit when

- The control fails → the shared-trajectory ceiling first, then fewer factors or less noise, not
  the thresholds.
- Non-linear or non-Gaussian channels are wanted → the projection becomes a component.

## Amendments

- **2026-09-20** — the specification (process, layouts, draws, presets, noiseless `SensorSignal`)
  moved to `shared/adapters/synthetic` so Evaluation can read the truth (ADR-0033). The reader
  stays in the Catalog and adds noise, gaps and rounding. Corpus bytes unchanged.
- **2026-09-21 — the transfer leg read** (`docs/verification/synthetic-transfer.md`).
  - The window was the fault: at 32 against factor periods of 24–300 everything failed, including
    the ceiling. At 128 the coupled pair passes: full fine-tuning 0.094 below a fresh encoder,
    [+0.089, +0.099], floor 0.053.
  - The null pair shares the family of the signals; swapping backbones between pairs moves the
    advantage by ≤ 0.012, the shared frequencies add +0.011. The null equivalence rule was withdrawn
    post hoc, named so; the pair still controls leakage and pairing.
  - `noise-a` (signal drowned) pretrains a worse start than a fresh encoder (−0.025, −0.011).
  - The backbone's channel table grows rows for a task's new channels (ADR-0033).

## Sources

Ojala and Garriga (2010), JMLR; Zhang et al. (2017), ICLR; Adebayo et al. (2018), NeurIPS; Bai and
Ng (2002), Econometrica; Shukla and Marlin (2021), ICLR.
