# ADR-0049: The scale of pretraining as a stage — a curve of the gain over the data, read at one or two per cent of the labels, before the transfer matrix

- Status: accepted (2026-10-04)
- Addendum (2026-10-04): the shape one step up, width 512 and eight layers, holds 25.3M parameters, not ~20M.
- Date: 2026-10-04

## Context

Every backbone so far was pretrained on about 10⁸ observed values: four or five corpora, 79M
tokens an epoch for the mixture of four. Under that data:

- the registered endpoint, full fine-tuning against the network from nothing at 200 labels, was
  not confirmed on either task (`docs/preregistration.md`, standing); on the turbofans the
  control led every candidate at every budget, down to 50 windows;
- leaving any one corpus out of the mixture moved the relative pretext loss of each other corpus
  by at most 0.015 (`docs/verification/manual-handoff.md`, 2026-10-04): the corpora are learnt
  beside each other, not from each other;
- a backbone pretrained for longer, with a lower pretext loss, did not replace a shorter one on
  either task.

Published work reports a gain from self-supervised pretraining on clinical series mostly at one
per cent of the labels or fewer, and close to nothing with every label; general time-series
models that transfer across domains are pretrained on 10¹⁰–10¹¹ points. The project's data rule
(unique values at least twenty times the parameters) bounds fitting, not transfer.

The transfer matrix is computed once; on backbones that show no gain anywhere it would spend that
once on a configuration already read as negative.

## Decision

- **A stage between the leave-one-corpus-out backbones and the transfer matrix measures whether
  the gain grows with the pretraining data.** The answer is a curve over points half an order of
  magnitude apart: ~10⁸ (today), ~3·10⁸, ~10⁹, and ~3·10⁹ under a rule registered before the 10⁹
  point is read.
- **The claim is narrowed to the regime where a gain is reported**: 20 and 50 stays on the
  intensive-care task, 50 windows on FD001, and 40 and 400 patients on a sepsis task over
  PhysioNet 2019. Cells at 200 labels and above are secondary.
- **The primary measure is the gain of the probe solved in closed form at 50 stays, as a function
  of the observed values in the pretraining mixture, at a fixed shape.** The closed-form probe has no
  optimiser, so its spread over seeds is the draw of labels and its folds alone. Two shapes are
  read: today's (4.8M parameters) and one step up (~20M); both curves start at the encoder's
  initialisation.
- **A larger mixture adds data beside the task's corpus, not instead of it**: every corpus a task
  reads keeps the same exposure (epochs times passes) at every point, so the curve does not
  confuse scale with dilution. A second pretraining seed at 10⁸ bounds what a step must exceed.
- **New data comes from open corpora**: UTSD (Apache-2.0), PhysioNet 2019 (CC BY 4.0), Tennessee
  Eastman, and for the last point LOTSA subsets under CC BY, CC0, Apache or MIT. Credentialed
  clinical data stays out: weights trained on it could not be published.
- **The recipe of adaptation is fixed before the curve is read**: a probe phase before full
  fine-tuning and patience counted in steps after the warm-up, chosen by the replacement rule on
  the tuning side. The same recipe reads every point.
- **The thresholds follow the reading already registered**: the least gain is one seed's spread
  at that budget on a fixed fifth of the tuning side, measured under the recipe before the first
  reading on the validation side.
- **The transfer matrix runs once, after the stage**, on its largest backbones.

## Consequences

- The claim is narrower than registered on 2026-09-16. A flat curve is a
  publishable result: two or three points with zero slope bound what more data of this kind buys
  at this model size.
- The new data is almost all regularly sampled; transfer from regular to irregular series is part
  of what the curve measures.
- Large corpora no longer fit a block in memory: the catalogue publishes a block per subset,
  written as a stream, and a mixture weights a corpus by the fraction read as well as by passes.
- The stage costs about two weeks and some tens of accelerator hours. A third order of magnitude
  (~10¹⁰) is out of reach and out of scope.

## Alternatives considered

- **Closing the project with the negative result at 10⁸.** It cannot separate "the method does
  not transfer" from "the data was two orders of magnitude short"; one curve can.
- **Replacing the tasks or corpora with ones known to transfer.** It changes the question after
  seeing the answer.
- **Only a larger model on today's data.** Trained from nothing, the smaller shape beat the
  larger one on the intensive-care task; more parameters without more data repeat that regime.
