# ADR-0068: A detector scored by the pretext's error, against a linear answer from the same cells

- Status: accepted (2026-10-31)
- Date: 2026-10-29

## Context

The detection tasks (ADR-0067) read nearest-neighbour detectors over pooled states, summaries
and counts. None asks the part of a backbone that learnt to fill in hidden cells (ADR-0064),
though published pretrained models score anomalies by the error of what they reconstruct.

The initialisation cannot be such a detector's control: its decoder is random, so any trained
one beats it. The question is whether pretraining knows more about normal telemetry than an
answer that learnt nothing, and that answer must see what the model sees. The masks hide whole
channels and the window's tail as well as single cells; interpolation over a channel's own bins
cannot answer the first, where a model reading the other channels wins by construction.

The decoder is stored with every backbone, but the port through which Evaluation loads weights
returns the encoder only; the pretext loss exists only summed over a batch, and one draw of masks
spans the batch. The masks and the loss a backbone was trained under are in Pretraining's register,
not in its artifact, and a cell run from an order reaches no database.

## Decision

- **The score of a window is the error of its values.** Each window is masked by the backbone's
  own strategy K times (K = 8, a parameter of the detector's name, set before any reading); the
  decoder answers the five summaries of every hidden timed cell that held a reading, scored by
  the backbone's loss in normalised units, and the score is the mean over those cells and draws.
  Presence and count are not scored: their weights, set for training, would rank the windows.
  The divisor is the hidden cells that held a reading, not every hidden cell as in training,
  lest a sparser window score lower.
- **Masks are drawn per window**, from a generator seeded by the campaign's seed and the
  window's unit, position and draw (`seeded_rank`), so a score does not depend on its batch.
- **The control is a linear answer from the same visible cells** under the same masks: a ridge
  per channel over its interpolated and carried values, the other channels' visible means in the
  same bin and the bin, a feature with nothing visible held at the channel's mean with an
  indicator; fitted on the fitting side's masked windows without a label, its penalty chosen by
  the error on the other half. Interpolation alone is reported beside; the nearest-neighbour
  detector over the summaries stays the classical bar.
- **The same alarm.** Every detector answers its score less the alarm at 2 % of the fitting
  windows, against its own unit under `reference=unit`. The alarm and the unit's normalisation
  move out of the nearest-neighbour detector into one place both kinds use; existing answers do
  not change.
- **A seam beside the factory of encoders**, implemented where processes are assembled: it
  rebuilds the backbone's whole pretext from its artifact, draws the masks and returns per
  window the visible cells, the hidden ones and the decoder's answers. Evaluation computes both
  errors and never sees a checkpoint. The artifact now stores the masks and the loss it was
  trained under; one written before does not, and the detector refuses it.
- **Refusals.** A backbone without cells; a corpus with channels beyond the vocabulary the
  backbone learnt, whose rows would be random. Cells without a time are not scored. A window
  with no hidden cell holding a reading in any draw has no error to read: it scores the fitting
  side's median, and their number is logged.

## Consequences

- The detectors are `pretext_error` and `linear_answer_error`, with `interpolation_error`
  beside; each takes `draws`, `contamination` and `reference`, and each is tied to one
  backbone's masks.
- They are read on the first mission alone, secondary, registered first.
- One window costs K passes through encoder and decoder. A fit keeps its alarm and the ridge's
  weights.
- The backbone and the ridge both learnt from the fitting side, so the share flagged at their
  alarm is optimistic; the area under the precision-recall curve, which ranks, is not moved by
  the alarm.
- A gain over the linear answer is what pretraining learnt beyond a linear map of the same cells.

## Alternatives considered

- *The whole pretext loss*: its ranking would rest on weights set for training.
- *Masks within a channel only*: the model still reads the other channels, as a regression would.
- *The pretext audit's stacked linear answer*: two stages for the same features.
- *Masks named in the campaign's order*: the order would then speak of one backbone's training.
- *A published language of Pretraining*: a contract package for one detector.
