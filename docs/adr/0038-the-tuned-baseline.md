# ADR-0038: The tuned baseline — spectra without a grid, a grid in the corpus's time, knobs chosen by a declared selection

- Status: accepted
- Date: 2026-09-24

## Context

Three more classical baselines were owed: gradient boosting over each channel's spectrum,
MiniRocket, and (later) a patch model. The last two need a series on equal steps, which a window
of tokens is not. Each has knobs, and a first measurement showed a knob moving MiniRocket by
several per cent. A knob that matters is either chosen by a procedure declared before the run,
or it is chosen by whoever looks at the comparison.

## Decision

- **Spectra are read from the readings, not from a grid.** A floating-mean Lomb-Scargle
  periodogram (Zechmeister and Kürster, 2009), at whole cycles per window, summarised as the
  share of power per octave, centroid, entropy and peak. A channel read `n` times is read up to
  `n / 2` cycles; above that, even sampling holds only aliases. The spectrum is its own scheme
  and part of the channel-aggregated reading, so the transfer baseline sees it too.
- **The grid is counted in the corpus's time, over the channels a task holds.** One step per unit
  of time (a cycle, an hour) times a `grid_resolution` that defaults to 1 — a regular series is
  read as recorded, as MiniRocket was published. Rows are the channels the fitted windows hold;
  a mask row is added only where those windows miss steps. Times are binned with a tolerance for
  single precision.
- **MiniRocket is implemented here in NumPy** (Dempster, Schmidt and Webb, 2021) to keep a JIT
  compiler out of the worker image. Under aeon's own random draws it reproduces aeon's features
  to float32 precision.
- **Knobs are named and a variant is named by them** (`minirocket@grid_resolution=2`). Each method
  lists the knobs a selection may turn; the specification that already validates the method
  validates the value. The catalogue reads a variant from its name alone, so the declaring
  process and the worker cannot disagree. A variant equal to its default is refused. Threads and
  the ridge penalty are not knobs.
- **Knobs are chosen by a campaign whose purpose is selection.** Variants are scored on tuning
  units held out per repeat (one in five), never on the validation side. At each budget, the rule
  of one standard error (Breiman et al., 1984; Hastie, Tibshirani and Friedman, §7.10), with the
  variance corrected for overlapping repeats (Nadeau and Bengio, 2003), picks the variant closest
  to the published default among those as good as the best. A selection keeps no artifact and
  announces nothing.
- **A chosen variant is checked, not trusted.** A comparison names each choice and its selection.
  Declaring it re-reads the selection by its rule and requires the variant, and its base, to be
  described exactly as the selection ran them.

## Alternatives considered

- *A fixed number of grid steps in configuration:* fits one window length, distorts every other.
- *aeon or sktime as dependencies:* numba and llvmlite in an image whose point is being small.
- *Tuning on the validation side:* the winner's reading there would be inflated by the luck that
  made it win.

## Consequences

A baseline in a comparison is either the method as published or the output of a procedure dated
before it ran, and the stored design says which. Tuning costs a second campaign — minutes for the
classical candidates, hours for networks. Until the networks' arms follow the same protocol,
baselines are tuned at every budget and the arms at one, an asymmetry that can only make the
network's claim harder. Anomaly-detection tasks have no labels on the tuning side and will need
their own selection rule.
