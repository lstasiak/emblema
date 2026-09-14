# ADR-0020: Judging a masked-reconstruction run — rules in the domain, an interval per comparison, a noise floor, and a probe for the diagnostic the control cannot answer

- Status: accepted
- Date: 2026-09-15

## Context

ADR-0019 settled what the objective is and that each kind of mask is measured against a trivial
baseline. It did not settle how a run is *judged*. The first measured leg exposed three ways that
gap lets a reading go wrong.

**An excess is not a verdict.** That leg reported single-token masks as trivial on the dense
layout from one number, 0.0167 against interpolation's 0.0135. Nothing said whether a difference
that size survives a different draw of validation units. A margin read off two means over tokens
that are not independent — windows of one unit overlap, and a unit is one realisation of the
process — can be noise and look like a finding.

**A baseline can be beaten and still leave nothing learnt.** Where measurement noise dominates a
channel, no predictor goes below it. A model that "beats" a baseline already sitting at that floor
has been asked a question with no answer in it, and a model that scores *below* the floor has been
given the answer: a leak, not a result.

**Beating each source separately is not beating both.** Interpolation reads a channel's own line;
the cross-channel ridge reads the other channels. A model could beat each while a single linear
regression over both together beat the model.

Two more things the first leg could not do. Its final losses were still falling steeply, so
nothing distinguished a model that had stopped learning from a schedule that had stopped it. And
its spectral diagnostic was mute: the control's factors are slower than its window by design, so
99.4 % of the truth's energy sits at one cycle per window and every higher frequency holds noise
alone. A diagnostic that cannot separate a model that learnt structure from one that learnt
smoothness says nothing about either.

## Decision

**The rules live in `pretraining/domain/assessment`, not in the report.** A verdict is a claim
about the system, so it is held to the architecture rules, the type checker and coverage like any
other domain code. Each rule is one function taking what it reads and returning a `Check` —
measured, expected, status, and a reading that says what the outcome means and what to do about
it. `decide` folds the checks into one `Outcome` by the precedence of what a failure invalidates:
an implementation fault first, because it voids every other number; then a kind of mask whose
baseline sits at the floor, which no amount of training changes; then a kind not beaten in a run
that has not been shown to have stopped learning, which is a question for a longer run and not for
the strategy. Only a run shown to have stopped learning that still loses condemns the kind.

The thresholds — how far the realised share of hidden tokens may stray, how far below the floor is
a leak, how close to the floor leaves a baseline no room, what loss reduction still counts as
learning — are constants in that module with the reasoning beside each. They are fixed before the
runs they judge. A threshold turned until a run passes is a verdict written in advance.

**Every comparison carries a bootstrap interval over validation units, and a kind is `learnt` only
when the whole interval lies above zero.** The unit is the resampling level because that is the
level at which the data are independent. Errors are ratios of sums, so groups combine by addition:
the diagnostic tallies summed squared errors per unit and hands them over, and the statistics that
turn tallies into intervals sit behind the `MaskKindSummariser` port. Both of a kind's intervals
are read off the same resamples, so the two comparisons are not two experiments.

The masks are drawn once over the validation windows. An interval therefore holds the variation
between units under that draw, not the variation between draws or between pretraining seeds. The
note says so.

**A noise floor is stated where the corpus knows one, and it decides where the linear baseline
decides.** For a generated corpus the measurement noise per channel is known exactly; in
normalised units it is the variance no predictor recovers. Three rules read it. A model below
0.8 × the floor is a leak, not a good model — the tolerance is not one because the floor is an
expectation and the noise of a few thousand tokens scatters about it. A matched baseline within
1.5 × the floor recovers all a predictor can, so that kind of mask has nothing to teach on this
corpus however long the model trains, and the outcome is to change the strategy, not the training.
And the second, linear baseline only decides where the floor leaves room above it: at the floor
there is nothing beyond linear left to learn, and with no stated floor the comparison informs
without deciding.

**The second baseline is a single ridge over both sources at once** — the other channels' nearest
visible values and the channel's own interpolated line, one regression per kind of mask and
channel. It stands beside the matched baselines that ADR-0019 names rather than replacing them: a
model is reported against the baseline the kind is matched to *and* against the strongest linear
answer on the same sources. For a channel hidden whole the two coincide, nothing of the channel
being left, and the second comparison is not made.

Both ridges are fitted on training windows **under masks the strategy draws**, not on intact
windows. A baseline fitted where every neighbour is present leans on neighbours it will not have
when it is asked, and loses for a reason that has nothing to do with what the model learnt. The
penalty is 1.0 on every coefficient but the bias, and is not tuned: see the consequences.

**Convergence is read off two runs, not off one curve.** A decaying learning rate flattens the last
epochs by construction, so a flat end would pass a run that stopped learning because its schedule
stopped it. A run passes when a stored run of the same configuration with at most half its epochs
gives the same verdicts and a final validation loss no more than 5 % higher. "The same
configuration" is decided by everything the run states about itself — corpus, code digest, machine,
model shape, schedule, seed — its date aside. Where no such run is stored the rule warns and says
which command produces one.

**A learning-rate schedule is a value object of the domain**: linear warmup, cosine decay to a
floor, as a factor of the peak per optimiser step. It is a pure function of the step, so it is
tested without an optimiser, and it is what makes the convergence rule readable — a run whose last
epochs are flat is flat because the model stopped learning or because the schedule did, and the
schedule is stated rather than inferred.

**A third corpus, the spectral probe, answers the question the control cannot.** It is the dense
control layout with the factor periods compressed from 24–300 steps to 6–32, so the truth completes
several cycles within a window and the spectrum has more than one frequency to report. It lives in
the report's own module, not in the control registry: the control is a fixed pair whose dials must
not be turnable, and the probe is an instrument for one diagnostic. The assessment states in its
own words when a spectrum is uninformative rather than letting a reader read recovery into noise.

## Consequences

- The assessment is exercised from both sides: the rules have their own tests, and the report is
  driven end to end in `tests/scripts`. A stored run is read back from what it *measured* and never
  from what was concluded, so a changed rule reassesses old runs rather than inheriting the old
  rule's verdict. Replaying 23 stored runs through the rules after they moved into the package
  reproduced every printed assessment byte for byte.
- On the published tier every kind of mask is `learnt` on both control layouts and on the probe,
  against both baselines. Single-token masks, trivial on the dense layout in the first leg, clear
  zero by +0.0041 there now. What changed between the legs is the tier, the budget and the warmup;
  what did not change is the code that decides. Both legs are in
  `docs/verification/masked-reconstruction.md`.
- The convergence rule warns on all three corpora at the default budgets: doubling from half still
  lowers the final loss by 28–43 %. The warning is true and is kept rather than tuned away. Each
  default budget sits one doubling past the first budget at which every kind was learnt, so a
  verdict does not rest on the epoch it was first reached at.
- **The ridge penalty is a fixed 1.0 and nothing guards against a baseline that is too weak.** The
  `room` rule guards the opposite case. A penalty chosen by a sweep on training windows would close
  this, and it is deliberately not done here: it would change every number in a note whose runs are
  already recorded, for a baseline whose margin of defeat is five- to fortyfold. The sweep belongs
  with the training runtime, which re-measures anyway.
- The intervals are percentile bootstrap intervals over 20 to 40 units. At that count they are
  approximate, and a verdict whose interval barely clears zero may not survive a repeat — which is
  why the budget is chosen so that none of them barely clears it.
- `Verdict`, `Decision` and `Interval` in Pretraining are not the Evaluation context's words of the
  same spelling. Here they judge a kind of mask against a baseline within one pretraining run;
  there they will judge a candidate on a downstream task. No type crosses between the contexts and
  neither imports the other, so the collision is in the dictionary and not in the code — but it is
  a collision, and the Evaluation context names its own types without looking here.
- `MaskKindSummariser` has one adapter, `UnitBootstrap`. A fake would pass no test with content in
  it: the port's whole obligation is the statistics. The contract is therefore parametrised over a
  list of one, as `Tokeniser` is for the same reason (ADR-0012), and states what any resampling
  scheme owes — ratios of sums, no room for zero where every unit agrees on the sign, and the same
  answer twice.

## Alternatives considered

- **Report the excess and let the reader judge.** What the first leg did. Rejected: it produced the
  one reading of that leg which the next leg overturned, and no number on the page said how firm it
  was.
- **A paired test over units instead of a bootstrap.** Rejected: the quantity compared is a ratio of
  sums, weighted by how many tokens a unit contributed, and a paired test over per-unit means is a
  different quantity. The bootstrap resamples exactly what is reported.
- **Bootstrap over windows.** Rejected: windows of a unit overlap and share one realisation of the
  process, so their excesses are not independent and the interval would be too narrow — the error
  that makes a margin look firmer than it is.
- **Read convergence off the shape of one loss curve.** Rejected: a decaying schedule makes the end
  of any run flat. The rule would pass exactly the runs it is meant to catch.
- **Put the spectral probe in the control registry beside `control-a` and `control-b`.** Rejected:
  a control whose dials can be turned is not a control, and the probe exists for one diagnostic.
- **Widen the control's factor band so the one corpus serves both purposes.** Rejected: it would
  change the corpus every earlier measurement was taken on, to make one diagnostic talk.
- **Keep the rules in the report script.** Rejected: a verdict that decides whether the objective
  works is not a detail of a script that will be deleted; it would sit outside the architecture
  rules, the type checker and the coverage gate, and it would go when the script goes.

## Revisit when

- A real corpus is assessed. It states no noise floor, so three rules skip and the linear baseline
  informs without deciding. Either a floor is estimated from the data — repeated measurements, or
  the residual of a saturated model — and the estimate is defended here, or the rules that need one
  are honest about being unavailable and the report says which verdicts rest on less.
- A run is assessed on more than one pretraining seed. Then the interval can hold the variation
  between seeds as well as between units, and the note's first limitation goes.
- The training runtime owns the loop. Convergence stops being a comparison of two stored CSV
  directories and becomes a property of a tracked run; the rule survives, its input changes.
- A second summariser arrives — a paired test, a BCa interval, a permutation test. Then the
  contract has two adapters and the single-adapter exception above lapses.

## Sources

- Efron, B. and Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. The
  percentile interval and its accuracy at small numbers of resampled groups.
- Bouckaert, R. and Frank, E. (2004). Evaluating the Replicability of Significance Tests for
  Comparing Learning Algorithms. PAKDD. Why a difference measured once, without resampling, is not
  a finding.
- Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets. JMLR 7.
