# ADR-0057: The rows of a corpus the backbone never read — fitted without labels, as the adaptation of a reading

- Status: accepted (2026-10-19)
- Date: 2026-10-19

## Context

A backbone gives every channel of every corpus one row of its channel table, keyed by the
channel's identifier in the published vocabulary; a corpus published after the backbone
carries identifiers past its table, and a corpus it left out carries rows its seed drew and
nothing moved. Three readings of `docs/verification/zero-shot-pretext.md` say what those rows
cost and what sets them.

- Zero-shot (2026-10-16): on a corpus the backbone never read, its pretext loses to a line
  through the channel's own visible tokens and to the cross-channel regression by 0.12 to 0.21
  in loss, where a backbone that read the corpus beats them by 0.02 to 0.09. The row the
  unseen channels pass through — drawn, at the initialisation, or the mean of the learnt rows
  — changes the reading by at most 0.11 and never its sign; one row for every channel is worse
  than a drawn row for each, since the model can no longer tell the channels apart.
- The rows alone (2026-10-17): with the rows of the corpus's own channels, and nothing else,
  fitted on 2,000 of its unlabelled training windows for 1,000 steps, the backbone that never
  read the corpus beats every baseline on 2012 and C-MAPSS and reads within 0.006 of the
  backbone that read it; the encoder at its initialisation, given the same fit, stays 0.08 to
  0.25 behind. What was missing was the channels' identity, not the encoder's machinery.
- In the evaluation (`task-transfer.md`, 2026-10-17): the same fit, registered as a
  continuation under `trained = "channel_rows"` (ADR-0056), carries +0.031 [+0.015; +0.047]
  of the task at 4,000 patients of the sepsis corpus to a backbone that never read it.

The alternative was a channel embedding shared across corpora by the quantity measured — the
heart rate of one corpus and of the other under one row — which would let a backbone read a
new corpus through a map of counterparts, without a fit.

## Decision

- **A channel's row belongs to its corpus.** The vocabulary keeps one identifier per (corpus,
  channel), and the backbone keeps one row per identifier; no row is shared by quantity.
- **A corpus the backbone never read is adapted by fitting its rows without labels**, every
  other weight frozen, on its unlabelled training windows: in a diagnostic through
  `pretext_triviality_report.py --fit-rows`, and for the evaluation as a registered
  continuation of the backbone with `trained = "channel_rows"`, whose provenance names its
  parent. A reading on such a corpus — pretext, probe or fine-tuning — takes that fit as its
  first step and says so.
- **The map of counterparts is a diagnostic, not a reading mode.** Whether a fitted row is the
  quantity's or the corpus's is measured (`channel_identity_report.py`, the declaration of
  2026-10-19 in `zero-shot-pretext.md`), and the measurement is what would change this
  record.

## Consequences

- A backbone reads a corpus it never saw only after a fit; the fit is cheap (nine to fifty-six
  minutes on the host's accelerator for 1,000 steps) and converges, but it sees the corpus's
  windows, so no reading on a new corpus is zero-shot.
- A claim of transfer is a claim about the encoder's machinery under fitted rows; the rows
  themselves carry a part of the task (+0.031 of +0.038 at 4,000 patients), so every such
  reading has the rows-alone fit as its hard baseline, as `domain-continuation.md` does.
- The row of a channel the backbone never read is left as drawn where it is not fitted; the
  mean of the learnt rows is not used, since one row for every channel reads worse.

## Revisit when

- The declared reading finds the fitted rows of one corpus beside the learnt rows of the same
  quantity in another (a share within three of the candidates well above chance), or the map
  of counterparts reads the pretext closer to the fit than to a permuted map by more than
  0.05 on the mapped channels: then a row shared by quantity, or a start from the counterpart
  before the fit, becomes a knob of the next pretraining and this record is superseded.
