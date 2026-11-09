# ADR-0077: A corpus cut from a task's data adopts the task corpus's channel statistics

- Status: accepted
- Date: 2026-11-07

## Context

A published corpus normalises each channel by a mean and a standard deviation fitted on its own
training side (ADR-0011). ADR-0076 publishes the satellite telemetry for pretraining from each
mission's first half, holds its last three months out as the validation side, and excludes the
second half. The tasks read the corpus published before: the same first halves as its training
side, so its statistics come from all of them.

The two corpora would then normalise the same reading differently, and a backbone trained on one
would be asked over the other. Measured on the raw archives, the statistics fitted without the
last three months against those of the whole first half:

| Mission | Largest mean shift, in standard deviations | Standard deviation ratio |
|---|---|---|
| Mission1, 48 channels | 0.023 | 1.001 to 1.018 |
| Mission2, 11 channels | 0.29 (channel 27), 0.09 (channel 22) | 0.76 to 1.07 |

The task corpus's statistics match the whole first half to 1e-11, so they hold no validation or
frozen month of a task. Only the pretrained arm would read shifted values; the bars beside it
read the task corpus.

## Decision

- **A publication may adopt the statistics of an earlier manifest** (`--statistics-from`). Each
  channel of the corpus takes the statistics the earlier scheme fitted for the channel of the
  same corpus, name, kind and unit; one with none there is refused rather than left raw. The
  training side then fits nothing.
- **The manifest says so.** `statistics_from` names the manifest the statistics came from. It is
  written only where it is set, so every manifest that fitted its own keeps its bytes and its
  checksum.
- **The pretraining corpus of ADR-0076 adopts the task corpus's statistics.** A month it holds
  is then the same windows, token for token, as that month in the task corpus. This is checked
  on the published blocks before the first pretraining.

## Consequences

- The three validation months' mean and spread enter the normalisation, with no label. They
  choose the epoch only, by a comparison between epochs, and every arm reads them alike.
- A corpus's training side no longer always fitted its statistics. The manifest names the
  exception, so a reader that assumes otherwise can tell.
- `--vocabulary-from` keeps its meaning. The two options name one manifest here, and may name
  two elsewhere.

## Alternatives

- **Fit the pretraining corpus's own statistics.** The shift above is built in, and it weighs
  on the pretrained arm alone.
- **Renormalise the task's windows to the backbone's statistics in evaluation.** The same values
  up to rounding, but in every runtime that reads a backbone.
- **Republish the task corpus.** Its statistics would change, and no earlier reading would
  compare with the new ones.
- **No validation side.** The epoch would again be chosen on the task's months, which ADR-0076
  removed.
