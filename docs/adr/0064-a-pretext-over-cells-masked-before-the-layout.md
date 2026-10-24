# ADR-0064: A pretext over cells — masks drawn on the cell grid before the layout, presence and count beside the hidden cells' summaries

- Status: accepted
- Date: 2026-10-24

## Context

An encoder trained from nothing over cells of channel × time bin (ADR-0062) matches boosted
trees at 200 intensive-care stays (`docs/verification/encoder-candidates.md`). Every pretraining
read so far ran over one token per reading, where the pretext was solved by interpolation or by
the stays' clock of measurements (`pretext-variants.md`); none has run over cells. DuETT, the
published network with this layout, hides whole time bins and whole variables and predicts each
hidden cell's value and whether anything was measured there; its ablations lose most without
the presence loss.

A cell is not a reading. An empty cell carries the channel's last value forward and counts the
bins since the channel was last observed. Hiding a cell by removing its token, as ADR-0019
removes a reading, would leave its value in the cells after it and its absence in their
"since": the target would stand in the encoder's input.

## Decision

- **Masks are drawn over the full cell grid and applied before the layout.** A window is laid
  out twice on the host: from every reading, which gives the grid the masks are drawn on
  (`MaskingStrategy` as it is, a "token" being a cell timed at its bin's centre, empty cells
  included) and the targets; and from the readings outside the hidden cells, on the same rows
  and bins, which is what the encoder reads (`BinnedTokens.without`). A hidden cell leaves the
  encoder's input as a hidden reading did; its neighbours are recomputed as if the hidden
  readings had never been taken.
- **The target of a hidden cell is what the full layout holds there.** The decoder of ADR-0019,
  its head seven wide, answers the cell's presence as a logit, its `log(1 + count)` and its mean,
  minimum, maximum, first and last. The five value features are scored under the run's reading
  of a miss, averaged over the five, only where the cell held a reading; the presence at every
  hidden cell by the binary cross-entropy; the count where the cell held a reading. The values'
  term is named `reconstruction`, so validation and the kept epoch follow ADR-0060; the presence
  and count terms enter the training loss by the experiment file's `presence_weight` and
  `count_weight` and are logged beside it.
- **Trivial predictors without a model**: the channel's mean for the values, as before; for
  presence, the channel's share of visible cells that held a reading in the same window, the
  window's share where the channel is hidden whole, each smoothed by one imagined cell of
  either kind, since a share read off five cells otherwise wagers certainty; for the count, the
  visible mean of the channel's `log(1 + count)` the same way, unsmoothed.
- **The input part of the architecture.** `EncoderArchitecture.bins` (0 for one token per
  reading) enters the backbone's identity: the signature, the stored document, the artifact and
  the parameter count, whose value projection is nine wide over cells. No migration: the
  configuration is a document, and one written before the field existed reads as `bins = 0`.
  The experiment file states it under `[input]`. Cells exclude the latent target, the contrast
  and the summaries, which read the raw value column.
- **The Evaluation lays cells for a backbone pretrained over them** from the backbone's own
  architecture, through the factory that serves the encoder; a plan stating bins beside a
  backbone stays refused. Such a candidate is kept in its measured form alone.

## Consequences

- A run over cells lays every batch out twice per step on the host; the pilot measures the
  loader's throughput before any accelerator trains (ADR-0062's revisit).
- On corpora read on a grid every cell holds a reading, so the presence is constant and the
  pretext a coarse interpolation. The audit before pretraining (`cell-pretext.md`) reads each
  corpus apart, and the mixture is composed accordingly.
- A tail drawn over cells asks which channels will be measured in the window's last bins and
  what they will hold: the schedule becomes a target, where ADR-0058 met it as a shortcut.
- Backbones over readings and over cells do not continue one another; the stored shape refuses.

## Alternatives considered

- **Hiding the cell's token alone.** The target leaks through carry-forward and "since".
- **An empty cell in place of the hidden one.** The encoder would meet, at inference, cells
  that mean "nothing measured" after training on cells that meant "hidden": the `[MASK]`
  mismatch ADR-0019 rejected.
- **Masking the raw readings, then laying out.** A cell that lost some readings and kept others
  would show part of its own target; drawing on the grid hides a cell whole.
- **Recomputing the visible layout on the device.** Every window's grid would have to be
  recovered from the padded token axis; the host layout is the Evaluation's already.

## Sources

Labach et al. (2023), DuETT; Zhang et al. (2024), t-PatchGNN; He et al. (2022), masked
autoencoders.
