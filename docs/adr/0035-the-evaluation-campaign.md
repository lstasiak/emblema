# ADR-0035: The evaluation campaign — a grid of candidates declared before it runs, a verdict only once it is whole, and one narrow message out of it

- Status: accepted
- Date: 2026-09-22

## Context

The first label-efficiency curve was produced by a report script: a directory of CSV files, a
cell appended once it finished, a shard per seed, and a reader that turned the directory back
into a comparison. It worked, and it made three things impossible to promise. A run that was
interrupted left cells that had to be spotted by hand. Nothing stopped a curve being read while
part of it was missing, and which cells are missing is never independent of what they found — a
diverging arm is the one that crashes. And nothing but convention kept the design fixed: the
budgets, the seeds and the rule for what counts as a difference were command-line flags, so
every one of them could have been chosen once the numbers were visible.

This context also has to produce something another context can act on. Serving may promote only
an artifact that appeared in a finished comparison, identified by checksum, and it learns of one
through an event rather than through a foreign key. What that event carries decides how much of
this context's reasoning leaks into the next one.

## Decision

**A campaign is a design plus the record of running it.** `CampaignDesign` states who competes,
over which budgets, under which seeds, which candidate is the control, which single comparison
is the endpoint, and the rules a verdict is read by. It is written once, before anything runs,
and the aggregate is afterwards only the cells that have been recorded. Everything a verdict
depends on is therefore fixed before the first number exists.

**A candidate carries what it was set to, in words this context does not interpret.** The
compute budget says how much arithmetic every neural arm was held to in common; what each of
them did with it — the rate, the decay, the warm-up, the layers a low-rank update touched — is a
bag of named strings the provider states and the campaign only records. A campaign that stored
the budget alone would say how long its arms ran and nothing about how they learnt, and a
reading of the curve a month later would rest on whichever copy of the configuration still
existed. Interpreting them is out of the question: a gradient boosting candidate has a depth and
a shrinkage, and a grid that understood either would have to know what it was comparing.

**The grid's axis is the candidate, not the transfer mode.** The four ways of using a backbone —
from scratch, frozen probe, low-rank, full fine-tuning — are four competitors, not a dimension
beside the competitors. That is what lets a classical method enter the same grid on the same
terms later: a campaign with a transfer-mode axis would have had nowhere to put a gradient
boosting model. Candidates reach the campaign through `CandidateProvider`, which describes what
a candidate is and answers one cell with an error per unit. Nothing about how a candidate is
built crosses that port.

**A campaign does not finish while a cell is pending, and has no verdict before it finishes.**
Both are invariants of the aggregate rather than checks in a service. Recording a cell twice is
refused, and so is recording anything after the campaign closed.

**Resuming is asking the campaign what is pending.** The state is the rows in the database, never
the messages in the queue, so a broker that lost everything costs a resubmission rather than
progress. A cell already recorded is answered from what is stored instead of being run again,
which is what makes the work safe to deliver twice.

**The candidate's artifact is the fitted candidate, kept at one cell.** What Serving may promote
is the thing that was measured, not the weights it started from: a classical candidate is
registered as the model it fitted, and a neural one carrying pre-fit weights would put two
different things under one name. Keeping every cell of a grid would store eighty sets of weights
to promote one, so the design names the cell whose artifact is kept — the endpoint's budget
under the first seed, chosen before anything ran, so that keeping one is not a selection made on
the results.

**What leaves the context is narrow on purpose.** `CampaignCompleted` carries the campaign, the
task, a sentence, and one entry per candidate: what it is called, what kind it is, the artifact
where one was kept, one figure per budget, and where it stands against the control in a
three-valued published vocabulary. The context's own verdict taxonomy stays inside: its
distinctions are written against a registration, they mean nothing to a reader who has not seen
that document, and publishing them would make every refinement of the statistics a breaking
change to a contract.

## Consequences

- A partial grid produces no number at all. This is the intended cost: the alternative is a
  curve whose missing cells are correlated with what they would have shown.
- A campaign that contains a permanently failing cell never finishes. That is the same invariant
  seen from the other side, and the answer is to fix the cell or to design a campaign without it,
  not to read the grid early.
- The design is stored as a document and the results as rows. Nothing queries a design a field
  at a time; the errors per unit are queried, pooled and resampled, so they are a long table.
- Two workers finishing the last two cells at the same moment would both see a whole grid. The
  worker that trains runs one job at a time, so this cannot happen today; a second concurrent
  consumer needs an optimistic lock on the campaign row, the same one the pretraining registry
  will need.
- A candidate records the weights it starts from, and a provider serving other ones refuses the
  cell. Without it a worker wired to the wrong backbone would answer quietly and wrongly, and
  the comparison would be unidentifiable afterwards.
- A method's parameters are held in name order and compared as text. The design is stored as a
  document, and a document store keeps no order of its own, so a canonical one is what makes a
  campaign read back the campaign that was written.
- A final campaign opens the frozen side once per cell rather than once per campaign, so a grid
  of eighty cells leaves eighty records of the reading that the project promises happened once.
  The promise is about the campaign, and the record will have to be read as such.

## Revisit when

- A second kind of task arrives. The campaign spreads candidates over budgets of labels, which
  exists only for the supervised protocol; an unsupervised detection campaign has no budget axis
  and is a different shape.
- A candidate has to be fitted once and scored on several tasks, which the one-cell-one-fit
  shape does not express.
- More than one process consumes cells at a time.
