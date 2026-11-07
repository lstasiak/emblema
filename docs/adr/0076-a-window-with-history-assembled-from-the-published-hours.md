# ADR-0076: A window with history, assembled from the published hours as cells

- Status: proposed
- Date: 2026-11-07

## Context

The satellite tasks score one-hour windows, and a task refuses a corpus published at any other
length. A pretext that hides a channel's next hour and asks for it from the hours before needs
those hours in its window: one-hour cells over 48 hours, measured with no label
(`telemetry-pretext-input.md`).

Checked on one month of each mission (`telemetry-history-window.md`):
- the 48 hours concatenated equal the window the publication cuts whole, to the bit: identifiers,
  values, times, gaps and the cells of 48 bins;
- the cells of the hours, joined as the context report joins them, equal those of the whole window
  within 2·10⁻¹⁵;
- laying a whole window from its readings gives 40 to 56 windows a second on the host, while
  joining the hours' cells gives 2,812 to 6,727.

## Decision

- **One construction of a window with history, in `shared/adapters/arrays/`.** Each published
  window is laid once into cells, over the same channels and `bins / history` bins each; a
  window spanning `h` of them joins their cells in time, recomputing the carry-forward, the count
  since the last reading and the last reading's offset over the joined bins. This is
  `JoinedCells`, moved out of the context report; pretraining, evaluation and the report call
  it, and a test holds it to the concatenated readings' layout.
- **History is part of the encoder's input.** `EncoderArchitecture.history` gives how many
  consecutive published windows one window spans: one, as every encoder so far, or 48 here. It
  enters the identity as `bins` does, through the signature, the stored document, the artifact
  and the experiment file's `[input]`. `bins` must be a whole multiple of `history`.
- **Pretraining draws one window per published hour.**
  - The reader hands each side's windows with their unit and extent.
  - A window ends at every published window. The windows before it on its axis are its
    history, whichever side holds them; an hour with no window is cells of nothing. The axis is
    the unit's part, a mission, as the manifest keys units `<part>/<name>`, or the unit itself;
    overlapping windows of one axis are refused, and a window more than a hundredth of a
    window off its place fills none. A validation window may read training hours;
    no training window reads the block, which comes last.
  - The hours are laid once a run, the windows joined per batch.
  - Masks are drawn over the joined cells. Hiding a cell empties the published cells under it,
    and the carry-forward and counts are recomputed after it, as `BinnedTokens.without` does from
    the readings; a known-answer test holds the two equal.
  - A tail of the last hour changes no earlier cell, since every cell depends only on the bins
    before it.
- **The pretext hides the last hour.** `MaskKind.HORIZON` at rate one, with a span of exactly
  one bin in 48, the cell the evaluation's tail reading hides.
- **The evaluation joins the same history from the windows it is handed.** It ends at each scored
  window and reaches back by end time within its mission. Interleaved units are refused, as
  in ADR-0075. A frozen month never handed in is
  history of nothing.
- **The publication holds out a validation block and excludes the task's months.**
  - The ESA corpus is published hourly from each mission's first half.
  - Its last three months are the validation side, which chooses the epoch.
  - The second half is excluded by unit name. `--exclude-units` is extended to `esa_ad_raw`, whose
    units are named months.

## Consequences

- A window is 2,304 cells on the first mission; a step at batch 8 takes 0.73 s on the Mac's
  MPS. On the published corpus, 153,000 hours are laid in 285 s once a run, and a batch of 32 is
  joined, masked and joined again in 25 ms on one host process.
- The first halves' 76,700 hours, one cell each over 59 channels, hold about 0.3 GB in float64,
  so they are joined in float64, as `BinnedTokens` lays them, and only the batch is narrowed.
- Neighbouring windows share 47 of their 48 hours; whether so dense a draw helps is read in the
  pilot, not assumed.
- `history` reads as one in every stored document, so no encoder so far changes.
- The arm that perturbs readings (ADR-0077) re-lays only the hour it perturbs, before the join.

## Alternatives

- **Publish 48-hour windows.** At a step of 6 hours: eight times the data, one target hour in
  six, 47 hours lost at each month's start, two constructions of one window, 40 to 56 windows a
  second. At 48: about 1,500 windows.
- **48-hour task windows labelled by their last hour.** New tasks, which no reading so far
  compares with.
