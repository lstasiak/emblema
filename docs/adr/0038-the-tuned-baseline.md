# ADR-0038: The tuned baseline — readings that need no grid, a grid counted in the corpus's time, and knobs chosen by a declared selection rather than by hand

- Status: accepted
- Date: 2026-09-24

## Context

The classical candidates of ADR-0036 read a window as statistics. Three more were owed: gradient
boosting over the spectrum of each channel, MiniRocket, and later a patch model — the last two
convolve a series on equal steps, which a window of tokens is not. Each of them has knobs, and a
first measurement made the question of who sets them concrete: MiniRocket over the turbofans erred
several per cent less on one resolution of the grid than on another. A knob that moves a baseline
by that much is either chosen by a procedure declared before it runs or it is chosen, knowingly or
not, by whoever looks at the comparison. (That first measurement was itself distorted by the
grid's rounding, corrected before any number here was read; the question it raised stands.)

A comparison that tunes its own method and leaves the baseline at whatever it was typed as says
more about the effort spent on each than about the methods. One that tunes the baseline on the
very numbers it then reports inflates the baseline by the luck that made a setting win.

## Decision

**The spectrum is read off the readings, not off a grid.** A generalised Lomb-Scargle periodogram
with a floating mean (Zechmeister and Kürster, 2009) is computed at the instants each channel was
observed, at whole cycles per window, and summarised as the share of power per octave, its
centroid, its entropy and its peak. Resampling an irregular channel first would invent readings
between the real ones and move power towards low frequencies — the loss this project's encoder
exists to avoid, charged to a baseline that has no need of it. A channel read `n` times is read up
to `n / 2` cycles: above that an evenly sampled channel holds only aliases of what lies below, and
counting them would count the same power twice. The spectrum is a scheme of its own for the trees
and a part of the reading summarised across channels, so the only baseline that can cross a
channel layout is offered what an engineer compares two machines by.

**A grid is counted in the corpus's own time, over the channels a task holds.** A window is laid
on one step per unit of the corpus's time — a cycle of an engine, an hour of a stay — times a
resolution that defaults to one, and over the channels the fitted windows hold rather than the
corpus's whole vocabulary, where a sensor read per operating condition is six channels and a task
over one condition holds one of them.
At the default a series recorded on that cadence is read as it was recorded, which is how
MiniRocket was published; a count of steps fixed for every corpus would fit one window length and
distort every other. A step with no reading carries the last one forward and a second channel per
channel marks the steps really observed, so a method on a grid sees both what the grid says and
how much of it was made up.

**MiniRocket is implemented here, in NumPy.** The kernel family, the dilations, the biases drawn
from quantiles of one fitted series and the proportion of positive values follow Dempster, Schmidt
and Webb (2021); the channels a kernel reads are convolved as their sum, which is the reference's
arithmetic reordered, since a convolution is linear. The reference library would have brought a
JIT compiler into an image whose worth is being small. What makes the reimplementation safe to use
is measured, not assumed: given aeon's own random draws, the fit and the transform reproduce aeon's
features to the precision aeon computes in, and its errors to a few ten-thousandths of the target.

**A candidate's knobs are named, and a variant is named by them.** Each classical method states
which knobs a selection may turn and turns them through the value object that already judges it,
so a value it cannot take is refused where the method is. A variant is its name —
`minirocket@grid_resolution=2` — and every process that holds the catalogue reads the same variant
out of the same text, so the process that declares a campaign and the one that runs its cells
never have a setting to keep in step. The threads and the ridge penalty are not knobs: the first
changes an answer's last bits and not how well a method fits, and the second is already chosen by
leave-one-out error inside every fit.

**Knobs are chosen by a campaign whose purpose is selection.** It is an ordinary grid whose
candidates are variants, scored on units held out of the tuning side — each repeat ranks the
tuning units under its seed and holds out one in five — so the side the comparison is later read
from is never read. A variant is chosen at each budget separately, because the setting that suits
fifty labels does not suit a thousand, by the rule of one standard error (Breiman, Friedman,
Olshen and Stone, 1984; Hastie, Tibshirani and Friedman, §7.10): every variant within one
standard error of the best is as good, and among them the one departing least from the published
setting wins. The repeats divide the same units over and over, so the variance is corrected as
Nadeau and Bengio (2003) correct it for repeated random holdout. The default of every method is
the library's or the paper's own, so the untuned candidate is the method as published and the rule
leans towards it.

**A chosen variant is checked, not trusted.** A comparison names, per budget, the variant it runs
and the finished selection that chose it; declaring the comparison reads that selection again by
its rule, and a design naming any other variant is refused. The pairing keeps the candidate's own
name, so the curve reads one candidate across budgets, while each cell runs the variant chosen for
its budget and is checked against that variant's description.

## Consequences

A baseline in a comparison is either the method as published or the output of a procedure dated
before it ran, and the record says which. The price is a second campaign before every comparison
that tunes — minutes for the classical candidates, hours for the networks — and a file per
selection that has to be committed before it runs.

The arms that adapt a backbone are not yet held to the same protocol: their peak rates were chosen
on the validation side at one budget. Until they are, the baselines are tuned at every budget and
the arms at one, an asymmetry that can only make the network's claim harder to confirm. The
grammar of variants lives at the level of the catalogue, so the arms join by stating their knobs,
not by a second mechanism.

A task without labels on its tuning side — anomaly detection — has nothing to score a variant on
this way and will need a rule of its own when its protocol is written.
