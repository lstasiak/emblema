# ADR-0036: The classical candidate — a port of its own, features that outlive a channel layout, and a process that declares a comparison without being able to run it

- Status: accepted
- Date: 2026-09-23

## Context

A campaign compares ways of using a pretrained backbone against each other. Every arm in it
learns the task from the same budget of labels, so the grid answers which way of using the
weights is best — and says nothing at all about whether the weights were needed. Without a
competitor that starts from no weights and no representation learning, the headline result has
an obvious alternative explanation standing beside it: that a summary of each window and a few
hundred trees would have answered the question just as well.

A classical fit is not an adaptation with some fields left blank. An adaptation is a transfer
mode over named weights, a schedule, a rank of low-rank update and a seed; a fit of trees is a
feature scheme, a count of rounds, a depth, and the other tasks it was allowed to learn from.
Carrying both in one value would mean every runtime ignoring half of what it was handed, and a
stored campaign that could not say what its candidate had actually been set to.

Two more things pressed on the design once the second kind of candidate existed. A grid worked
by more than one process is a grid two processes can write over each other in. And on the
platform this is developed on, the training stack and the library the baselines are fitted with
cannot share a process: they each bring a copy of the same OpenMP runtime, the loader coalesces
the weak symbols the two copies share, and a thread started under one runs the other's code over
the other's state. Merely importing the second makes the first crash in its next parallel
region; fitting with it makes the first hang instead, which leaves nothing behind to read.

## Decision

**A second driven port, not a widened one.** `ClassicalRuntime.fit` takes a recipe, the task,
the budget of labels, the other tasks' labels and the windows to answer. It sits beside
`AdaptationRuntime` and shares no value with it beyond the task and the labels. Both are reached
through one `CandidateProvider`, so a campaign is made of either kind without knowing which.

**The labels a run works from are drawn once, for whichever kind runs it.** Fetching the task,
drawing the budget, reading the side the run's purpose names and labelling it say nothing about
what a candidate is made of, and both kinds need all four. They are one use case and one value,
so a cell of the grid means the same thing whichever kind stood in it.

**Two feature schemes, and only one of them can cross a corpus.** A window is summarised by ten
statistics — how many tokens, their mean, deviation, extremes, first and last value, the slope
against time, the mean step between consecutive readings and the mean gap between them. Per
channel, that vector is as wide as the corpus has channels and means nothing in a corpus laid
out differently. Aggregated across channels — the mean, deviation and extremes of each statistic
over the channels, beside the statistics of the whole window — it is the same width whatever the
layout, which is the whole of what lets one fit take in another corpus's labels. A recipe that
names other tasks under the channel-bound scheme is refused where it is built rather than left
to fail on ragged rows.

A channel with no token in a window yields no statistic, and the gap is left as it is. Gradient
boosting learns a direction for a missing value at each split, so imputing a number would be
replacing evidence of absence with a guess the model then cannot tell from a reading.

**XGBoost, with the thread count as part of the recipe.** Histogram-built trees, the seed of the
cell in every place the library draws, and the number of threads stated by the recipe rather
than by the machine that picked the cell up: a histogram is summed in the order the work was
split in, so two runs agree only if they agree on that. Stating it in the recipe means a
campaign records it beside the rounds and the depth, and a worker configured otherwise is
refused the cell rather than quietly answering it.

The distribution differs by platform for one reason. On Linux the default one requires an NVIDIA
collective communications library, which is 347 MB of CUDA inside an image whose entire claim is
that it needs no accelerator; the `-cpu` build is the same version without it. On macOS there is
no GPU support to leave out, and no `-cpu` build published.

**The artifact is a document, not a pickled estimator.** What a fit leaves behind is the model in
the format the library writes models in, beside the feature scheme's column names, the scale the
target was fitted under and the recipe as scalars. A row means what the scheme says it means and
an answer is in the unit the task asks its question in, so both travel with the model; and a
document of plain data survives a release of the library in a way a pickled object does not.

**A campaign is written over only by a process that read it.** Two workers that each read a
grid, ran a cell and wrote the whole thing back would have the second delete the first's cell.
What a caller may write over is stated by the caller — the revision it read, which the aggregate
counts off its own state, since a campaign gains cells and then closes and nothing else about it
moves. The row is locked while that is compared, so the check and the write cannot be
interleaved. Losing the race is not losing the work: the campaign is read again and the result
put beside the cell that overtook it, which also settles who closes the grid.

**Saying what a candidate is, and being able to run it, are two ports.** Declaring a campaign
asks each competitor what it is and never runs one; a catalogue answers that from the arms and a
schedule, or from the baselines and the knobs they fit by, and needs no runtime. The providers
hold a catalogue and add the running, so the description a campaign was designed against and the
one a cell is checked against are one object rather than two readings that have to agree. The
process that declares a comparison then carries neither stack — which on this platform is not a
saving but the condition under which it can exist at all.

## Consequences

A campaign made only of baselines runs end to end with the context that trains backbones absent,
and the image that fits them is less than half the size of the one that adapts. The two are
separate processes serving separate queues, and neither carries what the other needs.

The comparison a campaign records is identified by what it recorded: the weights an arm started
from, the schedule it learnt under, the knobs a baseline fitted by and the threads it fitted on.
A process set to anything else is refused the cell. That is a text comparison and never an
interpretation — a provider that decided two settings were near enough would be making the
judgement a comparison declared in advance exists to take away.

What is not here yet is the measurement the transfer baseline was built for. The capability is:
a scheme whose width survives a change of layout, a recipe that names other tasks, and a fit
that takes their labels in. Which tasks a campaign draws on, and the row of the matrix that
reads out of it, arrive with that measurement, because the tasks must be the same value in the
process that declares the campaign and in the one that runs its cells or the check above refuses
every cell.

The comparison also costs a third of the arithmetic it looks like it should. Four statistics of
a window are read off the same sorted view of it, and a fit of a few hundred shallow trees over
a few thousand rows is seconds where an adaptation is minutes — which is why the queue that
carries it is the one everything else without a special need goes to.
