# ADR-0081: An interval of the window hidden in every channel at once, as a fifth draw of the masking strategy

- Status: accepted
- Date: 2026-11-15

## Context

The masking strategy hides whole channels, a block of each channel's time, single tokens and,
where stated, the window's tail (ADR-0019, ADR-0064). The block of one channel is placed
independently of every other channel's, so a hidden cell usually stands beside visible cells of
other channels in the same span of time. Only the tail hides a moment in every channel, and
only at the window's end.

Two facts make that a gap.

- On the intensive-care stays, the answer that reads a hidden cell from the other channels in
  the same bin carries most of what any answer without a model reads of the cell's presence:
  0.54 of the stack's 0.59 on the 2012 stays, 0.62 of 0.65 on the 2019 stays
  (`docs/verification/cell-pretext.md`, 2026-10-24). Which channels are measured together is the
  ward's protocol; a pretext that can read it off the same bin need not learn when a
  measurement comes.
- DuETT, the published network over cells, hides whole time bins and whole variables. Its
  ablation on in-hospital death keeps 0.612 of the full 0.627 in area under the precision-recall
  curve with the masks over time alone and 0.577 with the masks over variables alone, against
  0.556 without pretraining (Labach et al. 2023, Table 2). The masks over time carry most of
  what its pretraining gives, and ours has no draw of that kind inside the window.

## Decision

- **A fifth draw, the interval.** `MaskingStrategy` takes `interval_rate`, `interval_min_span`
  and `interval_max_span`. A window loses, with that probability, one interval of its time in
  every channel at once; the interval's length is drawn uniformly between the two spans, as a
  fraction of the window, and its start uniformly so that it lies inside the window. It is
  defined on time, not on a count of tokens or bins, as the block and the tail are: over
  readings it hides every reading in the interval, over cells every cell whose bin's centre
  falls in it, so over cells a span shorter than a bin can hide nothing, as a block's can.
- **Invariants as the tail's.** A drawn interval states `0 < min <= max < 1`; one never drawn
  states no spans. A timeless token is never inside it. Every window keeps a visible token.
- **A kind of its own.** `MaskKind.INTERVAL` is a hidden token of the interval whose channel
  still shows something outside it. A token is counted under one kind: channel first, then
  the tail, then the interval, then block, then token. Its matched trivial answer is the
  interpolation of its own channel, as for a block; what sets it apart is that no other channel
  shows the moment, so the cross-channel answers have nothing to read there.
- **Drawn last and only where stated.** The interval's numbers are drawn after the tail's, by a
  strategy that has one, so every run without it draws the masks it drew before. The stored
  configuration, the experiment document and the trained model write the three fields only
  where an interval is drawn: nothing written before changes its bytes or its signature, and a
  model stored without them reads as drawing none.
- **The experiment file states it** under `[masking]`; the rate and spans a pretraining uses
  are the file's business.

## Consequences

- A recipe can hold the share of hidden tokens and move only their alignment across channels:
  an interval at the block's rate and span in place of the block. That is the contrast the
  first pretrainings with it are read on.
- Over cells, the second layout recomputes what the encoder sees without the hidden cells
  (ADR-0064), so across an interval every channel carries its last value forward and counts the
  bins since: the encoder reads a gap, not a mark.
- The audit of a pretext over cells and the triviality diagnostic report the interval as a
  kind beside the others.

## Alternatives considered

- **The tail alone.** It hides every channel, but only the window's end, so the encoder is
  never asked about a moment with context on both sides; over readings it lost to the plain
  masks (`docs/verification/layer-readout.md`).
- **Aligning the block draw across channels by a flag.** One knob fewer, but a recipe could
  then not draw both, and a stored strategy would change the meaning of its block fields.
- **Hiding whole bins by index.** It would tie the draw to the cell layout; the strategy is
  shared with pretexts over readings, where there is no bin.

## Sources

Labach et al. (2023), DuETT: Dual Event Time Transformer for Electronic Health Records.
