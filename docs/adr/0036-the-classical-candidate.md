# ADR-0036: The classical candidate — a port of its own, features that outlive a channel layout, and a process that declares a comparison without being able to run it

- Status: accepted
- Date: 2026-09-23
- Full text before condensation: commit `5f14447`

## Context

Every arm of a campaign learns from the same labels, so the grid says which way of using the
weights is best and nothing about whether the weights were needed. Without a competitor that
starts from no representation learning, "a window summary and a few hundred trees would do" stands
beside the headline. A fit of trees is not an adaptation with blank fields: it has a feature
scheme, rounds, depth and the other tasks it may learn from. With two worker kinds, two processes
can write over one grid. And on macOS the training stack and XGBoost each bring a libomp whose weak
symbols the loader coalesces: importing the second crashes the first's next parallel region.

## Decision

- **A second driven port**, `ClassicalRuntime.fit(recipe, task, budget, other tasks' labels,
  windows)`, beside `AdaptationRuntime`. One `CandidateProvider` reaches both.
- **Labels are drawn once for either kind** (fetch task, draw budget, read the purpose's side,
  label it) — one use case, one value, so a cell means the same whichever kind ran it.
- **Two feature schemes.** Ten statistics per window (count, mean, deviation, extremes, first,
  last, slope, mean step, mean gap). *Per channel*: width = channel count. *Aggregated across
  channels*: mean, deviation and extremes of each statistic over channels plus the whole-window
  statistics — layout-independent width, so one fit can take in another corpus's labels. A recipe
  naming other tasks under the per-channel scheme is refused. A channel absent from a window yields
  missing values, not imputed ones; boosting learns a direction for missing.
- **XGBoost `hist`, seed in every draw, thread count in the recipe** — a histogram sums in the
  order work was split, so runs agree only on equal threads; a worker configured otherwise is
  refused the cell. Linux uses `xgboost-cpu` (the default wheel carries 347 MB of NCCL).
- **The artifact is a document**: the model in XGBoost's own format with column names, target
  scale and recipe — not a pickle.
- **A campaign is written over only by a process that read it**: the caller states the revision
  it read; the row is locked while compared. Losing the race means re-reading and adding the cell
  beside the one that overtook it.
- **Describing a candidate and running it are two ports.** A catalogue answers descriptions from
  arms and schedules or baselines and knobs, with no runtime; providers hold a catalogue and add
  running. The declaring process carries neither stack — on macOS the condition for it to exist.

## Consequences

- A baseline-only campaign runs end to end without the training context; the classical image is
  under half the size of the ML one.
- A campaign records what each candidate was set to (weights, schedule, knobs, threads); a process
  set otherwise is refused, by text comparison, never by interpretation.
- The transfer baseline has the capability (layout-independent scheme, recipe with other tasks)
  but not yet the measurement; source tasks must be the same value in declaring and running
  processes.
- A classical cell takes seconds where an adaptation takes minutes, so it runs on the `general`
  queue.
