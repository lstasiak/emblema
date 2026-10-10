# Preregistration: what counts as success in the label-efficiency comparison

- Registered: 2026-09-16, commit `e5ffafb`
- Rules in force as of: 2026-10-04, the last row of the register
- Applies to: the label-efficiency curve on the turbofan task, its single test run, the
  transfer leg of the synthetic control, the protocol of the intensive-care task, and the
  curve of the gain over the scale of pretraining

A curve read after it is drawn can be made to say almost anything: a threshold is chosen, a metric
is swapped, a budget is called the interesting one. This document fixes, before the numbers it
judges exist, which comparison decides the question, how large a difference has to be, and what is
reported when the answer is partial or negative.

**How this document is kept.** The sections below state the rules and the configuration in force,
and nothing else: no measurement, no diagnostic, no cost. They are edited in place, and every edit
is one dated commit and one row of the [register of amendments](#register-of-amendments) at the
end, which says what changed, whether it changed before or after the result it concerns, and what
had been measured at the time. The rule in force on any earlier day is the file at that day's
commit. Measurements live in `docs/verification/`, decisions about the system in `docs/adr/`, and a
diagnostic declared before it runs is written into the note that will hold its result, with its
prediction, before the run. A row of the register says in a sentence or two what changed and
points to the note that holds the declaration and its result; it does not restate either, and
the rules above are edited in the same commit as the row. A reader who wants to know how much
moved, and when, reads the register; it is meant to be counted.

## The claim under test

An encoder pretrained without labels lowers the number of labels a new task needs. The comparison
that tests it is the same architecture, on the same task, with the same label budget, pretrained
against trained from scratch. Nothing here claims the encoder beats a classical baseline on a
single task; that is a separate question, answered by the evaluation harness with its own
candidates, and this document does not preregister it.

## The task

Remaining useful life on the first C-MAPSS subset (FD001), a supervised regression with a label
budget.

| | |
|---|---|
| Units | 100 engines with run-to-failure histories; 100 further engines in the official test set |
| Split | by unit, before windowing, inherited from the published corpus: 79 engines tune, 21 are held out and validate; the 21 are engines no backbone was pretrained on |
| Window | 50 cycles, stride 5 |
| Label | remaining cycles, piecewise-linear with a ceiling of 125 cycles |
| Label budget unit | one labelled window |
| Budgets | 50, 200, 1,000, all (2,568 on the corpus in force) |
| Sampling | windows drawn from tuning units, stratified over four bins of the target, driven by a seed |
| Methods | from scratch, frozen backbone with a linear head (trained under the schedule, or solved in closed form as ADR-0044 adds), low-rank updates, full fine-tuning; the harness's classical baselines and patch model beside them |
| Repeats | five seeds per cell, 1 to 5; one pretraining seed |
| Resampling unit | the engine |

The ceiling of 125 cycles is the convention this task is usually reported under, and it is fixed
here because it moves the error materially: about a quarter of the windows carry the ceiling as
their label, so a quarter of the evaluation mass is a constant that any model predicts. The
ceiling is not a parameter to be tuned once the numbers are in.

The budget ladder is uneven at the top on purpose: 1,000 labelled windows is already over a third
of every label the tuning units hold, so the last rung is a factor below three where the first is
a factor of four. The low rung, 50 windows, is about two per cent of the available labels.

## The configuration the claim is judged under

Everything a run depends on is named here, so that none of it is chosen once an error is visible.
A different corpus, backbone, arm or schedule enters only by an amendment written before the run
that measures it.

**The corpus.** `cmapss` read with a channel per sensor and operating condition, scaled within
the condition (ADR-0034), all four subsets, window 50 at stride 5, manifest
`durable/sha256/d63f8e1b…`. The units of FD001 and FD003 are held out exactly as the earlier
version of those two subsets held them out, so the task's 79 tuning and 21 validation engines are
unchanged; a seeded fifth of FD002 and FD004 is held out beside them.

**The backbone.** `backbone-cmapss-m-8` of run `colab-cond`, weights `sha256:6283c210…`: eight
epochs of batch 32 in half precision on an NVIDIA A100, seed 1, over the corpus above. It was
named by the doubling rule: the ladder 4, 8, 16, 32 and 64 epochs, and 128 if the last doubling
still gains five per cent or more, is run over one experiment file, and the backbone is the run
at the end of the first doubling that lowers the best epoch's validation loss over the whole
held-out side by less than five per cent. The rule is applied by
its letter, since a choice between two runs it calls equal, made after seeing which is lower, is
the kind of choice this document exists to prevent.

**How a candidate is made.** Every method answers through the same linear head over the mean of
the token states in the tail of the window, a fifth of it unless a selection chooses another
share (ADR-0041); the first curve, of 2026-09-22, ran under the mean over the whole window. A
run trains for a stated budget of steps and is scored after the
last, never stopped on the validation error. Targets are learnt in units of the label ceiling. The
head's bias starts at the mean of the labels the run holds; its weights are drawn from the seed.
Low-rank updates go beside the attention's projections and the feed-forward network's linears of
every block, rank 8, α = 16, no dropout, and start at zero. Two seeds are kept apart, the seed of
the draw fixing which labels and the seed of the run fixing the head, the fresh weights of the
control arm, the updates and the order of windows; the five repeats of a cell are five values of
one seed that drives both, so the two methods of a cell are compared on the same labels.

**The schedule, one shape for every arm.** A linear warm-up over the first tenth of the run's
optimiser steps, then a cosine decay to one per cent of the peak by the last step; batches of
sixteen; no weight decay. Every run stands on a floor of 2,000 optimiser steps: thirty epochs, or
as many whole epochs as reach 2,000 steps, whichever is more, with the warm-up and the decay
spanning the run so lengthened. The floor gives every cell the same chance to leave the plateau
of the mean predictor and leaves the budget of labels as the only thing that differs between
cells of one arm.

**The stop, at the budget of every labelled unit (from 2026-10-03).** At that budget every arm
of the backbone holds one unit in five of its labelled sample out, ranked by a digest of the
run's seed and the unit, learns from the rest, scores the held-out units after each epoch (the
areas under the ROC and precision-recall curves summed for an outcome, the negative squared
error for a quantity) and keeps the weights of the best epoch, giving up after ten epochs
without a better one; the schedule's epochs are its cap. The validation side is never read for
the stop. At smaller budgets a held-out fifth is a handful of units, and every run keeps the
schedule above with its last weights, as before. The patch model keeps its fixed epochs at every
budget until it has the stop. Read on the intensive-care task, where the stop alone gained the
network from nothing 0.027 in area at every stay from fewer labels and halved its spread over
seeds (`intensive-care-curve.md`, 2026-10-03; ADR-0047). Nothing read before this date changes.

**The peak rate of each arm**: from scratch 1e-3, frozen probe 3e-2, low-rank updates 1e-4,
full fine-tuning 1e-3. It is the arm's default, the setting a selection departs from; a
comparison runs each trained arm at the variant its selection chose at each budget
(`campaigns/curve-fd001.toml` names them). Peaks are chosen on the validation side at
the endpoint's budget, by a rule fixed before the sweep: three peaks per arm, 200 labelled
windows under seeds 1, 2 and 3, the peak with the lowest mean validation RMSE over the three. A
peak chosen at an edge of its grid is followed by one peak beyond that edge, half a decade away,
at the same seeds and budget, and the rule chooses again, at most twice per arm. Every arm is
swept the same way, so no arm is tuned more than another, and the peaks are swept on each task's
own validation side rather than carried over from another task.

**How a candidate is tuned.** A candidate runs at its default unless a selection declared
before it runs chooses otherwise. The default of a classical baseline is the setting its method
was published with; the default of an arm or of the patch model is the schedule in force above,
with the arm's registered peak. The knobs a selection may turn, and the values it may turn them
to, are named in a committed campaign file whose purpose is *selection*. For a baseline they are
the knobs of its method, the ridge penalty of the convolution baseline excepted, since every fit
already chooses it, by leave-one-out error on a regression and by folds on outcomes. For a
network they are the knobs that leave the compute budget as it is — the peak rate, the weight
decay, the share of the run the warm-up takes, the fraction of the peak the rate decays to, and
the head's pooling and the share of its tail — and never the epochs, the
floor of steps or the batch, so a tuned network spends what its base spends. A selection never
reads the validation side: in each repeat the seed ranks the task's tuning units, one in five is
held out, the budget is drawn from the rest and every variant is scored on the held-out units. A
variant is chosen at each budget of the curve separately, by the rule of one standard error:
every variant whose mean RMSE over the repeats is within one standard error of the best one's is
as good, the standard error corrected for the overlap of the repeats as Nadeau and Bengio (2003)
do for repeated random holdout — the variance scaled by 1/J + n_test/n_train rather than 1/J —
and among those the one that departs least from the default is chosen: fewest knobs turned, then
the smallest ratio on a log scale. A comparison runs the variants a finished selection chose and
names that selection; one naming any other variant is not declared. Every candidate of a
comparison is tuned by this one protocol, or none is; a comparison that tuned its baselines per
budget and its arms at the endpoint alone is read with that asymmetry stated, in the baselines'
favour.

**Where a comparison runs.** Every comparison is made within one budget, and every cell of one
budget runs on one kind of accelerator; a budget begun on one and resumed on another is run again
whole on the second. Between accelerators an arm has drifted by up to 0.7 RMSE in a seed, the
size of the effect the rules look for.

**When a configuration replaces the one in force.** Both must hold on the validation side: the
endpoint is confirmed under the candidate configuration by the endpoint's rule; and full
fine-tuning under the candidate beats full fine-tuning under the configuration in force, paired
on the same validation engines and the same windows, pooled over the five seeds, with the whole
95 per cent interval of a bootstrap over the engines (10,000 resamples) above zero, both sides
run on one kind of accelerator. Comparing the two reductions instead would choose whichever run was luckier.

## Metrics

**Endpoint metric: RMSE over the validation windows**, accumulated as summed squared error per
engine so that a comparison is paired on units and groups combine by addition. The repeats of a
cell are pooled per engine: the squared errors of every repeat are added per engine before the
pair is made and resampled, so a cell's RMSE is the root of the mean squared error over its
repeats, and the spread of RMSE over the repeats is reported beside it.

Reported with every result, and carrying no threshold:

- the asymmetric prognostics score, as a mean per window rather than the benchmark's sum; it
  penalises a late prediction more heavily than an early one and its exponential tail is
  dominated by a handful of engines, so it is read, not thresholded;
- α-λ accuracy: the share of windows whose prediction lies within ±20 % of the true remaining
  life;
- RMSE over the last window of each engine, which is the protocol published numbers are computed
  under; on the validation side every engine runs to failure, so this reading is the error at the
  end of life and is not set beside published errors, which are compared on the test side, once;
- RMSE restricted to windows below the ceiling, the regime in which the task is a task.

**No metric may take the endpoint's place after the numbers are seen.** Swapping the quantity that
decides the question is the failure mode this document exists to prevent, and it is not repaired
by reporting the other metrics honestly alongside.

## The primary endpoint

**Full fine-tuning against training from scratch, at 200 labelled windows.** One comparison, named
in advance, and therefore carrying no correction for multiplicity.

The claim is confirmed when all three hold:

1. the relative reduction in RMSE is at least **10 %**;
2. the whole 95 % confidence interval of the paired difference lies above zero: a bootstrap over
   the validation engines, 10,000 resamples, the same engines for both methods;
3. the reduction is not smaller than the practical floor defined below.

Ten per cent is one full step between published methods on this subset: reported errors run from
about 11.8 to about 19.8 RMSE, and consecutive published results differ by roughly 0.5 to 2 RMSE,
which is 4–12 % of the base. A difference of that size is the smallest one that a reader of this
literature would recognise as a result rather than as a repetition.

Two hundred labels is the low-budget regime without being the noisiest rung: about eight per cent
of the available labels, four times the smallest budget, and small enough that a model which
needs labels to learn the task has not yet had them.

The bootstrap p-value counts the observed reduction as one resample on its own side,
`(k + 1) / (B + 1)` per side, twice the smaller; no p-value is zero. The endpoint is measured once
and read once: the cell at 200 labelled windows is not run again for the grid.

## Secondary comparisons

Every other candidate at every budget, against the control, less the primary, is secondary. The
family's size is stated in the campaign's committed file before it runs: eleven for the three
transfer modes first registered, four budgets by three less the primary; thirty-five for the ten
candidates of the curve repeated on 2026-09-27. Secondary cells are tested at the 5 % level with
a Holm correction over the family, and they are labelled secondary wherever they appear. The
family is the declared size whatever has run: a cell that has not run enters the correction with
a p-value of one, so a
partial grid is read more strictly than the whole one, never less, and the conclusion states when
the grid is incomplete. A secondary cell's verdict follows the family's word, not its own
interval. A secondary result does not confirm the claim on its own; it describes the shape of the
curve around the endpoint that does. The correction a campaign reads its family under is named in
its committed file before it runs; the family of this claim is under Holm, and a campaign that
names the step-up correction of Benjamini and Hochberg controls the share of false discoveries
among its rejections rather than the chance of any, and says so wherever its cells appear.

## When a difference is too small to matter

A difference is reported as *statistically distinguishable, practically nil* when it is smaller
than

    max(3 % of the from-scratch RMSE at that budget,
        the standard deviation of that RMSE over its five seeds)

The fixed part keeps the floor from collapsing when a run happens to be quiet; the measured part
keeps it from sitting below the noise the experiment itself generates. Both are fixed here; the
measured part is read off the same run it judges, not chosen afterwards. The floor binds the
endpoint as well as the secondary cells: a reduction the floor swallows is not confirmed, whatever
its interval says.

## Partial and negative outcomes

- **Confirmed at some budgets, not at others.** The result is the crossing point, reported as
  such: the budget beyond which the advantage is no longer distinguishable. This is an expected
  shape, not a failure, and it is the honest form of a label-efficiency claim.
- **Confirmed on the primary endpoint, indistinguishable everywhere else.** Reported as a single
  positive cell with the interval widths beside it, and explicitly not generalised.
- **Not confirmed.** The negative result is published with the same prominence a positive one
  would have had. Before it is interpreted as a statement about pretraining, three things are
  read in order: the synthetic control below, whether the pretraining run was shown to have
  stopped learning, and whether the from-scratch baseline at that budget is already at the floor
  of what the data support. A negative result with a failing control is a statement about the
  implementation, not about the method.

## Power, declared before the fact

Twenty-one validation engines are the binding constraint of this design, not the five seeds. For
a paired difference over 21 units, the half-width of a 95 % interval is about 0.46 standard
deviations of the per-engine differences; under the Holm correction over eleven secondary
comparisons it rises to roughly 0.8. Wide intervals are therefore a property of the experiment as
designed, known now, and they will not be reinterpreted later as a finding about the method.
Over 21 units the percentile interval falls short of its nominal coverage, and the endpoint's
confirmation carries about twice its nominal one-sided error
(`docs/verification/verdict-statistics.md`). The rule stands as registered; the shortfall is
reported beside every reading made under it.

## The synthetic control

The control corpora exist to make a negative result readable. The generator and the certificate
that a pair carries the structure it claims are recorded in ADR-0018; the transfer leg, in
ADR-0033. The leg is measured at the endpoint's budget alone, 200 labelled windows under seeds 1
to 5, the four arms, under the schedule, the floor and the head's start of the curve, on windows
of 128 time units at stride 12, a window that spans the process's time scales. The task is the
reading of one sensor twelve time units past the window's end, its error the RMSE in the
sensor's own units; the pretraining pairs are `control-a` and `null-a`, the task's the wide
second layouts `control-b-wide` and `null-b-wide`, the peaks swept on each pair's own task by the
curve's rule. Backbones, sides and peaks are in `docs/verification/synthetic-transfer.md`.

- **The positive control** is the coupled pair, judged by the rule the curve is judged by: the
  advantage of full fine-tuning over training from scratch must clear the practical floor with
  its whole interval above zero.
- **The null pair controls leakage and pairing, not structure.** Its two layouts share the family
  of their signals by the generator's design, so a backbone carries something across them
  whatever the coupling. Its reading is that the advantage on the null task does not depend on
  which pair's backbone is fine-tuned, by more than the floor.
- **The structure's own share** is the coupled task's advantage under its own backbone less that
  under the null pair's backbone, paired over the same units; it is reported as measured. A
  backbone pretrained on noise bounds what the mechanics of any pretraining give.

The first form of the null rule, an equivalence within ±the floor, was withdrawn after its
measurement and replaced by the reading above; the register records that as the post hoc change
it is. The leg is complete and read as passed.

## Protocols of the other tasks

Declared now, so that no task's protocol is chosen once its numbers are visible.

| Task | Protocol | Primary metric | Label-budget axis |
|---|---|---|---|
| Remaining useful life, turbofan (FD001) | supervised, label budget | RMSE | yes |
| Anomaly detection, industrial testbed | unsupervised: fitted on normal data, scored on the rest | PR-AUC, F1 at a fixed threshold | no |
| Anomaly detection, server machines | unsupervised | PR-AUC | no |
| Binary classification, intensive-care stays | supervised, label budget | AUROC with calibration | yes |
| Binary classification, sepsis after the first day of a stay | supervised, label budget | AUROC with calibration | yes |
| Anomaly detection, satellite telemetry | unsupervised, the benchmark's own protocol | the benchmark's metrics | no |

The anomaly-detection tasks measure whether pretraining improves detection, which is a different
quantity from label efficiency. They never share an axis, a panel or a summary sentence with the
curve.

## The intensive-care task

Death in hospital after a stay in intensive care (PhysioNet/CinC Challenge 2012,
`physionet2012-in-hospital-death`), read over the first 48 hours of the stay. The protocol is
registered here before any run on the task. The budgets, the endpoint and the selections are
registered before any selection runs; the least gain that counts and the floor's fixed part
before the first cell of the grid, by the reading declared for a campaign of the control and the
endpoint over ten seeds on one fixed fifth of the tuning side (`verdict-statistics.md`,
2026-09-29). None of them is chosen from anything read on the validation side.

| | |
|---|---|
| Units | Stays; one window per stay, the whole 48 hours and a minute. A stay with no measurement has no window and is not in the pool. |
| Sides | Tuning: set A as published (3,997 stays with a window, 554 deaths). Validation: set B (3,994, 568). Frozen: set C (4,000), named from the challenge's listing and opened once, in the final run. The corpus's statistics and every backbone's pretraining read set A only. |
| Backbones | Two at tier M, seed 1, pretrained without labels on set A's stays: `backbone-physionet2012-m-32` over the stays alone (weights `sha256:c5878c03…`), named by the doubling rule over 8, 16, 32 and 64 epochs, and `backbone-mixed5-m` over the four corpora of the mixed backbone and the stays (weights `sha256:869ed545…`). A campaign runs under one backbone, so each has its own. |
| Label | `In-hospital_death`, zero or one, from `Outcomes-a.txt` and `Outcomes-b.txt`. |
| Budgets | 50, 200 and 1,000 stays, and every stay of the tuning side. |
| Endpoint | Full fine-tuning against the network trained from nothing, at 200 stays, under the mixed backbone; one comparison, carrying no correction. The campaign under the stays-alone backbone reads its own family; the two backbones set side by side are a declared diagnostic, not a claim. |
| Draw | A budget is a count of stays, drawn in proportion to the two outcomes, every prefix within one stay of each outcome's share, nested across budgets and seeded; a draw holding one outcome is refused. |
| Measure | One minus the area under the ROC curve, ties counting half; reported as the area. A reduction is a gain in area. |
| Thresholds | Stated in area, not as shares: the endpoint's least gain is 0.045 and the floor's fixed part 0.01. The floor is the larger of that part and the control's spread over seeds. |
| Interval | Paired bootstrap over stays, both sides on the same draw, in two strata — stays that died and stays that did not — each resampled to its own size; 10,000 resamples, 95 %, two-sided p-value as for the turbofans. Repeats pooled by the mean of their areas over the same resample. If the percentile interval's coverage on known answers falls short of its level at these sizes, the bias-corrected interval is registered before the grid instead. |
| Family | Every other candidate and budget against the control, under Holm, as for the turbofans: 35 under the mixed backbone; 15 under the stays alone, whose campaign holds the four ways of using the backbone and the control only, since the baselines and the patch model read no backbone. |
| Calibration | The Brier score of every cell is reported beside the area and judged by nothing. |
| Heads | Every network ends in one linear head read as log-odds, trained by binary cross-entropy from the log-odds of the sample's prevalence, no class weighted; the ridge probe and MiniRocket fit an L2-penalised logistic regression on their grid of penalties, chosen by the mean log-loss of five folds in the sample's proportion of outcomes; trees grow under the logistic objective (ADR-0045). MiniRocket's grid over outcomes is the published ridge grid, 0.001 to 1,000 in ten steps, followed by 4,640, 21,500, 100,000, 464,000 and 2,150,000; the turbofan task keeps the published grid. |
| Selection | Every candidate is selected at each budget before the grid, under each backbone it runs on: one stay in five held out of the tuning side per repeat, three repeats, one minus the area under the rule of one standard error, and the edge rule. A network departs from its default — the turbofan's registered peak of its mode as the rate (the patch model at the schedule's 1e-3), pooled by the mean — by a third and three times the rate, the tail of a fifth or of half the window, and attention; the probe solved in closed form turns the two tails alone, having no query to train. A tail departs from the mean by two knobs, attention by one. The peaks are where the search starts, not a choice: a default is kept only on a tie. The classical baselines turn the knobs their turbofan selections turned. |
| Final run | A stay of set C with no measurement is answered with the prevalence of the task's own sample the candidate learnt from, the answer of a predictor that has seen nothing; none is dropped. Every candidate answers those stays alike, so the paired comparison is unmoved except through the ties the constant answer makes with each candidate's other answers; the areas and Brier scores reported move by the same stays. The harness does not answer such a stay yet: it is built and tested before the final run, and this row is amended with the commit. |

## The sepsis task

Whether sepsis follows the first day of a stay in intensive care (PhysioNet/CinC Challenge 2019,
`physionet2019-sepsis`), asked at the end of that day. Registered before the corpus's
publication was pretrained on and before any campaign on the task; the least gain and the
floor's fixed part follow by the reading below, before the first cell is read on the validation
side.

| | |
|---|---|
| Units | Stays; the first day of each, counted from its first recorded row: the first window of 24 hours the corpus publishes for the stay, read alone (ADR-0052). A stay shorter than a day has no such window and is not in the pool. A stay whose first day holds no measurement (2 stays) or whose label turns within it (281) is not in the pool either; every stay the task does not read is named in its listing of ineligible stays. 30,378 of the 40,336 stays answer: 15,645 of hospital A, 14,733 of hospital B. |
| Label | Whether `SepsisLabel` turns to one after the first day, read from the stay's own file: one if it does, zero if it never does. The challenge sets the label six hours before the onset it dates, so a positive stay's onset falls at least 30 hours after its first row. 1,563 positives, 5.1 % (hospital A 6.1 %, hospital B 4.2 %). |
| Sides | Frozen: one stay in five of those the task reads, in each hospital and outcome apart, ranked by seed 1 on a stream of its own (so that the publication's draw of the rest by the same seed is independent of it) and rounded up, 6,077 stays (313 septic), named in the task's listing and cut out of the corpus before its publication, so no statistic and no backbone reads them. Validation: the corpus's held-out side, a fifth of the remaining stays drawn by the publication's seed 1, of which the task reads 4,779. Tuning: the corpus's training side, of which it reads 19,522. The publication is `durable/sha256/bf7de2fa…`, continuing the vocabulary of the scale chain (channels 291–329); the tasks over it are `03c3a159-…` and, over hospital B (tuning 9,398, validation 2,388, frozen 2,947), `fa926d86-…`. The corpus's statistics and every backbone's pretraining read the training side only. |
| Tasks | `physionet2019-sepsis` over both hospitals, and `physionet2019-sepsis-hospital-b` over hospital B alone, on that hospital's stays of the same sides: hospital A is the centre the stays of 2012 come from, so a backbone pretrained on those stays is read on another hospital only in hospital B. |
| Budgets | 40, 400 and 4,000 patients and every patient of the tuning side. In hospital B alone 400, 4,000 and all: 40 patients hold under two septic stays there. |
| Draw, measure, interval, heads | As for the intensive-care task: a budget is a count of stays drawn in proportion to the two outcomes; one minus the area under the ROC curve; the paired bootstrap over stays in the two strata of the outcome; one linear head read as log-odds. |
| Endpoint | The closed-form probe against the network from nothing at 400 patients, under the mixture of about 10⁹ observed values in today's shape; one comparison, carrying no correction, confirmed by the intensive-care endpoint's three conditions. 40 patients are read descriptively; 4,000 and every patient are secondary, as is the task over hospital B. |
| Thresholds | By the reading registered for the intensive-care task on 2026-09-29, at 40, 400 and 4,000 patients, under the recipe in force, before the first cell is read on the validation side. |
| Recipe | The intensive-care task's settings in force, unchanged; no setting is selected on this task. |
| Reported beside | The area per hospital on the same validation side, descriptively. |
| What the task does not separate | Hospital A and the stays of 2012 come from one centre, and patients common to both cannot be ruled out after anonymisation; hospital B is therefore read beside the whole. The first day is counted from the first recorded row, not from admission: 37 % of hospital A's stays start after their first hour, most within three. A quarter of the stays last less than a day and fewer of them turn septic; the task predicts for patients still in intensive care after a day. |
| Final run | Every frozen stay has a measurement in its first day by construction. The frozen stays are in no publication; how they are tokenised under the task's publication, with its vocabulary and statistics, is built and tested before the final run, and this row is amended with the commit. |

## The scale of pretraining

Whether the gain grows with the data a backbone is pretrained on (ADR-0049). Registered on
2026-10-04, before any run of the stage; the least gain and the floor's fixed part follow by the
reading below, before the first cell is read on the validation side.

**The claim, narrowed.** An encoder pretrained without labels on a mixture of at least 10⁹
observed values from at least thirty corpora lowers the error of a task at about one or two per
cent of its labels or fewer — 20 and 50 stays on the intensive-care task (0.5 and 1.25 % of the
tuning side), 50 windows on FD001 (2 %), 40 and 400 patients on the sepsis task — against the
same network trained from nothing under the same recipe, read by the probe and by full
fine-tuning. The claim says nothing at 200 labels and above; those cells are secondary. The
claim above this section stands as registered; this one is read beside it.

| | |
|---|---|
| Points of the curve | Point zero: the encoder at its initialisation, pretrained on nothing. Then mixtures of about 10⁸ observed values (the mixture of four, `backbone-mixed4-m`), about 3·10⁸, about 10⁹, and about 3·10⁹ under the rule below. The axis is the count of observed values in the mixture's training units, overlapping windows counted once, as `docs/verification/data-spike.md` counts them; the epochs a backbone ran are reported beside it. Each point is pretrained at seed 1, its best epoch kept by the pretext's validation loss, as at 10⁸; the mixture of four in today's shape is pretrained again at seed 2. A descriptive point of about 1.5·10⁸ values (`backbone-scale-1e8new-m`) reads the corpora of the points of about 3·10⁸ and 10⁹ with the new ones at a tenth, so that three points share one set of sources and differ in the amount of data alone, beside the mixture of four, which holds a similar amount from four sources; it enters neither the slope, the largest point's rule nor an endpoint. |
| Exposure | Every corpus a task reads (the stays, C-MAPSS, PhysioNet 2019 once published) is planned at the same exposure at every point: epochs times passes equal to the mixture of four's eight. The other corpora's passes and fractions are free. A larger mixture therefore adds data beside the task's corpus and does not read it less often. |
| Shapes | Today's (width 256, six layers, 4.8M parameters) at every point, and one step up (width 512, eight layers, 25.3M) at point zero, 10⁸, 10⁹ and, if run, 3·10⁹. |
| Tasks and budgets | The intensive-care task over the publication continuing the turbofan vocabulary (`a6a9c653-…`) at 20, 50 and 200 stays; FD001 at 50 windows; a sepsis task over PhysioNet 2019 at 40, 400 and 4,000 patients and every patient, its protocol registered here before its first campaign. Seeds 1 to 10 in every cell. |
| Primary measure | The gain in area of the probe solved in closed form over the network trained from nothing, at 50 stays, as a function of the observed values of the mixture, for each shape. The trained probe and full fine-tuning under the recipe are read beside it. |
| Endpoint | The closed-form probe against the network from nothing at 50 stays, under the mixture of about 10⁹ observed values in today's shape; one comparison, carrying no correction, confirmed as the intensive-care endpoint is: a gain of at least the least gain, the whole interval above zero, and a gain the floor does not swallow. 20 stays are read descriptively: no confirmation is read there. Every other cell of the stage is secondary, its family stated in its campaign's file. FD001 has its own endpoint, the same comparison at 50 windows under the same mixture and shape, confirmed by the turbofan endpoint's three conditions; the sepsis task's is registered with its protocol. |
| Recipe | One recipe of adaptation reads every point, chosen on the tuning side by the replacement rule declared for it (`intensive-care-curve.md`, 2026-10-04): a variant replaces an arm's setting where its paired interval lies above zero and its mean over seeds exceeds twice its standard error. On the intensive-care task at 20, 50 and 200 stays the network from nothing and full fine-tuning keep their settings in force (`campaigns/recipe-*-physionet2012.toml`); on FD001 at 50 windows full fine-tuning starts its head solved in closed form (`full_fine_tuning@head_start=solved,learning_rate=0.000333,pooling=tail,tail_share=0.2`) and the network from nothing keeps its setting. The trained probe runs at its setting in force. In the larger shape full fine-tuning's setting is chosen before anything is read under a backbone of about 10⁹ values: on the intensive-care task at 50 and 200 stays, under the mixture of four in that shape at 5e-4, on the same fifth and seeds and by the same replacement rule, against `full_fine_tuning` at its peak, the variants are half the peak (`learning_rate=0.0005`), the head solved first (`head_start=solved`) and both. The setting is chosen at each budget apart, as the recipe in force was; where more than one variant replaces it, the larger mean gain there is chosen, and 20 stays take the setting chosen at 50. No variant replaced it at 50 or 200 stays, so full fine-tuning keeps its setting in force in the larger shape at 20, 50 and 200 stays (`intensive-care-curve.md`, 2026-10-07). Today's shape keeps its setting. |
| The pretext | A variant of the pretext replaces the masks of the mixture of four when, on the intensive-care task at 50 stays, on the half of the tuning side held out by seed 101 (`one_in = 2`) over seeds 1 to 10, the closed-form probe under the variant's backbone against the probe under `backbone-mixed4-m` at seed 1 has its paired 95 % interval above zero and its mean over seeds above twice the error between backbones on the same half (the row below): the interval pairs draws of labels, not pretrainings. Where two variants replace it, the larger mean gain is chosen. 20 and 200 stays and FD001 at 50 windows are read descriptively; the validation side is not read. Under this design a variant that does not replace says nothing of a gain below 0.03 to 0.04 (`verdict-statistics.md`, 2026-10-08). `pretext-variants.md`. |
| The error between backbones | A rule that compares two backbones, each pretrained once, reads its mean over seeds against twice √(SE² + s²): SE is the mean's standard error over the draws of labels, s the spread a difference between two backbones takes from their pretraining alone. s is read from the mixture of four's pretrainings, by the probe the rule reads and on the side it reads: for every pair of pretraining seeds at 20, 50 and 200 stays, the squared mean difference less its noise (the squared standard error over seeds plus the square of the paired interval's half-width over 1.96), averaged over pairs and budgets, floored at zero, then its root. From the mixture's four pretraining seeds (`experiments/backbone-mixed4-m.toml`, `-seed2.toml` to `-seed4.toml`) by the blocks side by side it is 0.0060 on half the tuning side and 0.0104 on the validation side (`layer-readout.md`, 2026-10-10). |
| Thresholds | By the reading registered for the intensive-care task on 2026-09-29, at each of 20, 50 and 200 stays: one seed's spread over ten seeds on one fixed fifth of the tuning side, under the recipe, for the network from nothing and the closed-form probe; the least gain at a budget is the larger of the two standard deviations there, rounded up to 0.005, and the floor's fixed part the smallest standard deviation of the network from nothing over the three budgets, rounded down to 0.005 and not below 0.01. The run is declared before it runs and reads nothing on the validation side. Read so: the least gain is 0.055 at 20 stays, 0.060 at 50 and 0.050 at 200, and the floor's fixed part 0.035. |
| Interval | As for the intensive-care task: paired bootstrap over units in the two strata of the outcome, 10,000 resamples, 95 %, the ten seeds pooled by the mean of their areas over the same resample. A step of the curve is the paired difference between two backbones' probes on the same draws and units. |
| The slope | The curve rises when, at 50 stays in a shape, the paired interval of the closed-form probe under the mixture of about 10⁹ values less the probe under the mixture of about 10⁸ lies above zero and its mean over seeds exceeds twice the error between backbones on the validation side; it is flat when the interval lies within the floor at 50 stays on either side of zero; otherwise it is not settled. The error between backbones read in today's shape bounds both shapes. |
| Length of training | Beside the point of about 10⁹ values in today's shape, the same publication, shape, masks and schedule trained four epochs instead of two (`backbone-scale-1e9-4ep-m`): the corpora a task reads keep their eight reads, and every other corpus is read four times instead of twice. Read by the yardstick at 50 stays: the point of about 10⁹ values was limited by its training when the closed-form probe under the longer run less under `backbone-scale-1e9-m` has its paired interval above zero and its mean over seeds above twice the error between backbones on the validation side (today's shape was read under the earlier threshold, 0.004 by the last block). It reads no endpoint and enters neither the slope nor the largest point's rule; where it holds, a slope that is flat or not settled is not read as data that do not help. The larger shape is read the same way: `backbone-scale-1e9-512x8-4ep-m`, its publication, shape, masks, rate and schedule trained four epochs instead of two, against `backbone-scale-1e9-512x8-m` by the closed-form probe in force, at 50 stays on the validation side (`intensive-care-curve.md`, 2026-10-09). |
| The probe's reading | From 2026-10-09 the closed-form probe of every reading of the stage not yet made reads the encoder's blocks side by side (`frozen_ridge@layer=concat`, ADR-0053): the larger shape's longer run, the pretext rule, the weight of the tasks' data and the objective. It replaced the last block by the rule declared before its reading: at 50 stays on half the tuning side it gained 0.020 to 0.036 under all three backbones of about 10⁹ values, where a reading of every block over the untrained encoder gained nothing. Points already read are not read again. Where a rule compares backbones by this probe, the pretraining seeds' difference it names is read with this probe too, before the first reading under it (`layer-readout.md`, 2026-10-09). |
| Regimes | Adaptation without labels (the target corpus's tuning side in the mixture) and zero-shot (the mixture without the target corpus) are read separately at every point and never pooled. |
| The largest point | Run when, at 20 or 50 stays in either shape, the closed-form probe under the mixture of about 10⁹ values has its paired interval above zero against the network from nothing, or against the probe under the mixture of about 10⁸ values; or when a variant of the pretext has replaced the current one by the rule of the pretext above. Not run when neither holds. Eight comparisons enter this rule uncorrected: it is a liberal rule for spending compute, not a test, since running the point needlessly costs time and not running it could miss a curve that rises. A fourth order of magnitude is not run under any outcome. |
| After the stage | The transfer matrix is computed once, on the backbones of the largest point run, under the recipe. |
| What the curve does not separate | Data and diversity grow together: each larger mixture adds domains as well as values. The epoch kept is chosen by the pretext's loss, which has not ordered backbones on these tasks, because the point at 10⁸ was chosen so. How a zero-shot backbone treats the channels of the corpus it left out is decided, and recorded in an ADR, before the first zero-shot reading. |

**Predictions, declared before the first point is read.**

1. At point zero the closed-form probe's interval against the network from nothing at 50 stays
   holds zero or lies below it.
2. Under the mixture of about 10⁸ values the closed-form probe's interval against the network
   from nothing holds zero at 20, 50 and 200 stays.
3. At about 3·10⁸ values the probe's gain at 50 stays lies within 0.01 of its gain at 10⁸, and
   the two seeds at 10⁸ differ by less than 0.01.
4. At about 10⁹ values in today's shape the gain at 50 stays lies between +0.01 and +0.03 in
   area; the endpoint is not confirmed.
5. At about 10⁹ values the larger shape gains at most 0.01 over today's at 50 stays; at 10⁸ it
   gains nothing over it.
6. At about 3·10⁹ values, if run, the gain at 50 stays lies between +0.02 and +0.04: the curve
   rises, and rises slowly.
7. On FD001 at 50 windows no point shows a gain whose interval excludes zero.
8. At 200 stays the interval holds zero at every point of the curve, which are all mixtures; a
   backbone over the stays alone is not a point of it.
9. At every point the zero-shot regime gains no more than adaptation without labels.

## The test set

The official test engines are frozen when the task is created and are used **once**, at the end,
for the configuration in force at that time; no hyperparameter, no variant and no decision to
continue is taken on them. Every number produced before that run is labelled *validation*
wherever it appears.

Seven of the 100 test engines are shorter than the 50-cycle window, the shortest holding 31
cycles. They are scored on the observations they have: the model reads a window as a set of
tokens and does not require it to be full. None of them is dropped, and no padding is invented for
them.

## Standing

- Turbofan endpoint, validation side: **not confirmed** by the curve repeated on 2026-09-27
  under the configuration in force, the wrong way: full fine-tuning 10.5 % above the arm from
  nothing at 200 labels, and the control ahead of every candidate at every budget. The first
  curve's confirmation of 2026-09-22 is read as the control's handicap
  (`docs/verification/label-efficiency-curve.md`, `docs/findings.md`).
- The synthetic control's transfer leg: complete, passed
  (`docs/verification/synthetic-transfer.md`).
- Intensive-care endpoint, validation side: **not confirmed** by the grid of 2026-09-30 under the
  mixed backbone: full fine-tuning 0.024 in area above the arm from nothing at 200 stays, below
  the floor of 0.040 and the least gain of 0.045
  (`docs/verification/intensive-care-curve.md`, `docs/findings.md`).
- The scale of pretraining: registered, no point read.
- Sepsis endpoint: registered, nothing read.
- The single test run: not made.

## Register of amendments

One row per change, in the order made. *Kind*: **criterion** changes a rule; **configuration**
names or replaces what the rules run over; **reading** makes a registered rule precise without
changing it; **diagnostic** declares a run with its prediction and changes nothing by itself;
**measured** records a result under the registration it belongs to. *When* says what had been
measured when the change was made. The title in italics is the heading the change was registered
under, so a citation elsewhere in the repository of the form "`docs/preregistration.md`, date,
title" resolves to a row here and to the commit the row names, where the full text stands.

| Date | Commit | Kind | When | What changed |
|---|---|---|---|---|
| 2026-09-16 | `e5ffafb` | registered | before any run | The document as first registered: the claim, the task on 80/20 engines drawn by the task, the endpoint at 200 with 10 % and the interval, the family of eleven under Holm, the floor, the outcomes, the power, the synthetic control's two rules, the protocols, the test set. |
| 2026-09-16 (committed 2026-09-17) | `dfab961` | configuration | before any run | *the held-out engines are the ones the backbone never saw.* The validation side becomes the corpus's own held-out FD001 engines; endpoint, thresholds and floor unchanged. |
| 2026-09-18 | `4d01929` | configuration | before any run on it | *the backbone the curve is drawn from, and how a candidate is made.* `backbone-cmapss-m` named; the shared linear head, scoring after the last epoch, targets in ceiling units, the low-rank placement, the two seeds. ADR-0008, ADR-0030. |
| 2026-09-19 | `3eb3769` | configuration | after a sweep on the validation side, before the grid | *the schedule of every arm, fixed on the validation side before the grid.* The schedule's shape, epochs and batch; the first peaks; repeats pooled per engine; the two-sided p-value. `label-efficiency-curve.md`, 2026-09-19. |
| 2026-09-19 | `3eb3769` | reading | before the grid | *how the registered rules are read, made precise before the grid.* The floor binds the endpoint; the family is the registered eleven whatever ran; the p-value's form; the asymmetric score as a mean per window; the last-window RMSE comparable on the test side only. |
| 2026-09-20 | `605a5e0`, `96da833` | criterion, configuration | **after the first grid**, endpoint not confirmed | *a floor of optimiser steps, the head's start, and a sweep over three seeds.* The floor of 2,000 steps, the head's bias at the mean label, peaks chosen again over seeds 1–3, the backbone retrained to its plateau by the doubling rule; corrections made after a result was seen, applied to every arm alike, with the endpoint, threshold, floor and family unchanged. `label-efficiency-curve.md`, 2026-09-20. |
| 2026-09-20 | `6b7b191` | configuration | by the rule, before any grid under the floor | *the peaks under the floor, from the sweep over three seeds.* The four peaks; the control's stands from here. |
| 2026-09-21 | `0e9b0f5` | configuration | after a pilot on the null pair's narrow corpus, before the leg's grid | *the synthetic control's transfer leg: corpora, sweep and power, before its grid.* The forecast task, the two backbones, the wide second layouts, the grid, peaks per pair, power from the pilot. ADR-0033. |
| 2026-09-21 | `0e9b0f5` | configuration | by the rule, after the ladder of 4 to 64 epochs | *the backbone named again: sixty-four epochs, the plateau reached by the rule.* `backbone-cmapss-m-64`. `manual-handoff.md`, 2026-09-21. |
| 2026-09-21 | `3426e0c` | configuration | after the leg's sweeps, before its grid | *the synthetic leg's peaks and its power, read off the sweep, before its grid.* The same four peaks on both pairs. |
| 2026-09-21 | `3426e0c` | configuration | by the rule, before any grid under the floor | *the peaks of the three pretrained arms under the backbone of sixty-four epochs.* `label-efficiency-curve.md`, 2026-09-21. |
| 2026-09-21 | `3426e0c` | criterion | before the leg's numbers under five seeds; twelve cells at fifty had run and are not read | *the synthetic leg is measured at the endpoint's budget alone.* The other budgets answer nothing the control's rules ask. |
| 2026-09-21 | `b463ed1` | diagnostic | after both pairs failed their rules at the endpoint | *the ceiling of the synthetic transfer: the second layout over the first's trajectories.* Outcome: the fault lay above the data. `synthetic-transfer.md`, "the ceiling". |
| 2026-09-21 | `36c3c19` | diagnostic | after the ceiling | *the window against the factors' periods: the coupled pair at a window of 128.* Outcome: the window was the fault. `synthetic-transfer.md`, "the window". |
| 2026-09-21 | `74fffe4` | configuration, criterion | after the window diagnostic, before any run at 128 under a rule | *the synthetic control moves to a window of 128: sweep, null pair, and the reading of both rules.* Both pairs republished at 128, backbones retrained, peaks swept again, both rules read at 128 in that order. |
| 2026-09-21 | `74fffe4` | measured, diagnostic | under the registration above | *the control at 128 read: the coupled pair passes, the null pair fails, and the family's share is measured by swapping the backbones.* `synthetic-transfer.md`, "the control closed at a window of 128". |
| 2026-09-21 | `3311640` | measured | under the swap's declaration | *the swapped backbones measured: the first prediction held, the second did not.* The remedy left to a registered decision. `synthetic-transfer.md`, "the backbones swapped". |
| 2026-09-21 | `7a8627c` | criterion, diagnostic | **post hoc**: after the swap's measurement | *the null pair read as the control of leakage it is, and a backbone pretrained on noise to bound what any pretraining gives.* The equivalence rule withdrawn after its measurement and replaced by the leakage reading; the structure's share reported; the leg read as passed; the noise backbone declared. |
| 2026-09-21 | `7a8627c` | measured | under the declaration above | *the noise backbone measured: the mechanics alone are worse than a fresh encoder.* The prediction held; the leg complete. `synthetic-transfer.md`, "a backbone pretrained on noise". |
| 2026-09-21 | `c1c8a9c` | diagnostic, criterion | after the sweep under the 64-epoch backbone, before any run | *the turbofan backbone's pretext window, and peaks at the edge of their grids, before any run.* The pretext-window cell, prediction failed (`synthetic-transfer.md`, "the pretext window apart from the task's"); a ladder at 100 cycles ordered and withdrawn; the edge rule for peaks adopted. |
| 2026-09-21 | `3b1d5de` | configuration | after the normalisation finding, before any run under it | *the turbofan corpus normalised within one operating condition: a backbone on FD001 and FD003, before any run.* The corpus, its ladder, a sweep of all four arms under the edge rule, the endpoint, and the decision rule for what follows. ADR-0034. |
| 2026-09-22 | `d5a181e`, `8ed23cc` | reading, measured | reading settled after the sweep and before the endpoint; then the endpoint measured | *the endpoint on FD001 and FD003 read by the endpoint's own rule, and where it runs, before any of it runs.* The stricter reading, all three conditions. Measured: confirmed; the two-subset corpus and its 8-epoch backbone became the configuration. `label-efficiency-curve.md`, 2026-09-22, the edges and the endpoint. |
| 2026-09-22 | `69aaa71`, `f0161e7`, `fcdfe35` | configuration, diagnostic, measured | grid registered before it ran; the check exploratory, choosing nothing | *the grid under the floor on FD001 and FD003, and whether a longer backbone helps the task, before either runs.* The grid at 50, 1,000 and all; the longer backbone not better, prediction failed. `label-efficiency-curve.md`, 2026-09-22, both sections. |
| 2026-09-22 | `fdf8053`, `0523683`, `bad25e9` | configuration, measured | after the endpoint on two subsets, before any run on four | *the four subsets read per operating condition: a backbone over all of C-MAPSS, and when it replaces the one over FD001 and FD003, before any run.* The corpus, its ladder, sweep and endpoint, the replacement rule. Measured: endpoint confirmed, the replacement rule met, the configuration replaced. `label-efficiency-curve.md`, 2026-09-22, the A100 section. |
| 2026-09-22 | `7c543bc`, `fcdfe35` | configuration, measured | grid registered before it ran | *the grid under the floor on the four subsets read per operating condition, before it runs.* One accelerator per budget. `label-efficiency-curve.md`, 2026-09-22, the last section. |
| 2026-09-22 | `060394b` | diagnostic | before it runs; settles nothing | *the endpoint read again on seeds no sweep has seen, and whether a longer backbone helps this corpus, before either runs.* Neither reading changes the configuration. `label-efficiency-curve.md`, 2026-09-22, the endpoint on seeds no sweep has seen. |
| 2026-09-22 (committed 2026-09-23) | `d4933fa` | editorial | after the readings above | The rules in force rewritten in place from the register as it stood at `060394b`; no criterion, threshold, configuration or reading changed. |
| 2026-09-24 | `ca5c767` | criterion | before any selection runs | *how a classical baseline is tuned.* A baseline runs as published unless a declared selection on held-out tuning units chooses a variant per budget by the rule of one standard error. |
| 2026-09-25 | `97b0b2a` | criterion, reading, measured | before any selection of a network runs | *how a candidate is tuned.* The selection protocol extends to the arms and the patch model; a campaign names its family's correction in its file. Measured: the interval's coverage on known answers; the rule stands (`verdict-statistics.md`). |
| 2026-09-26 | `591277d` | configuration, diagnostic | before either selection runs; the pooling campaign `cb5ed115…` had been read on the validation side | *the networks' default head, and a pilot selection of their knobs before the curve is repeated.* Every network pools by the tail of the window; two pilot selections at 200 labels declared with their predictions. `head-and-representation.md`, 2026-09-26. |
| 2026-09-27 | `38fe056` | diagnostic | before it runs; nothing read on the validation side since the pilot | *the floor of steps asked again at the rate the pilot chose, before the curve's selections.* Two selections under the floor and twice it, with the reading that would change it. `head-and-representation.md`, 2026-09-27. |
| 2026-09-27 | `a0a8970` | measured | under the declaration above | *the floor read: 2,000 steps stand.* No configuration changes. `head-and-representation.md`, 2026-09-27, the A100 section. |
| 2026-09-27 | `a0a8970` | configuration, diagnostic | before any of the three selections runs | *the three trained arms selected per budget before the curve is repeated.* One selection per trained arm at the four budgets, declared with predictions; the curve runs each arm at what its selection chose. `head-and-representation.md`, 2026-09-27. |
| 2026-09-27 | `98be0ab` | measured, diagnostic | under the declaration above; before the two follow-ups run | *the three arms selected per budget, and two rates beyond the edge declared.* Two choices at 50 lie at an edge, so two selections one rate beyond are declared by the edge rule. `head-and-representation.md`, 2026-09-27, the two last sections. |
| 2026-09-27 | `bb09023` | measured | under the declaration above | *the rates beyond the edge read: the chosen rates stand.* `head-and-representation.md`, 2026-09-27, the G4 section. |
| 2026-09-27 | `bb09023` | configuration, diagnostic | before the curve runs; nothing read on the validation side since the pilot's confirmation campaign | *the curve repeated by the harness: five arms and five baselines under the protocol, before it runs.* `campaigns/curve-fd001.toml`: the trained arms at their selected variants, the probe solved in closed form added (ADR-0044), the patch model as published, Holm over thirty-five; predictions declared. Endpoint, threshold, interval and floor unchanged. `label-efficiency-curve.md`, 2026-09-27. |
| 2026-09-27 | `2fb9063` | measured | under the declaration above; budgets 50 and 200 ran on an A100 rather than the G4 named, one kind per budget as the rule requires | *the repeated curve read: the endpoint is not confirmed, the wrong way.* The first curve's confirmed endpoint is read as the control's handicap. `label-efficiency-curve.md`, 2026-09-27, the G4 and A100 section. |
| 2026-09-27 (committed 2026-09-28) | `589790b` | configuration | before any run on the task | *the intensive-care task.* The protocol of `physionet2012-in-hospital-death`; the budgets, the endpoint, the least gain and the floor left to a row before the grid. ADR-0045, ADR-0046. |
| 2026-09-28 | `91bedc0` | configuration | before any counted run on the task; one smoke campaign at the small tier, whose numbers are not cited | *the closed-form heads over outcomes.* The ridge probe and MiniRocket fit an L2-penalised logistic regression chosen by the log-loss of five stratified folds, after the smoke run showed calibration on leave-one-out answers reversing the ranking. ADR-0045. |
| 2026-09-29 | `db39a13` | configuration | before any selection or grid on the task; measured on the tuning side only | *MiniRocket's penalties over outcomes.* The grid over outcomes extended past the published one by five steps. `classical-baselines.md`, 2026-09-28. |
| 2026-09-29 | `e427dfe` | configuration | by the rule, after the ladder of 8 to 64 epochs; before any selection or grid on the task | *the intensive-care backbones named.* The rung of 32 epochs over the stays alone, and the mixture of five beside it. `manual-handoff.md`, 2026-09-29. |
| 2026-09-29 | `dc41dd3` | editorial, reading | after every row above | *the rules brought in line with the register, and the register cut to what changed.* The rows of 2026-09-26 and 2026-09-27 had changed the configuration without the text above; the tail, the arms' variants per budget, the closed-form probe and the standing are now written in. Reading: the secondary family is every other candidate at every budget, its size stated in the campaign's file, as the repeated curve and the intensive-care task already read it. Rows keep their titles; the predictions and results they restated stay in the notes they name. |
| 2026-09-29 | `dc41dd3` | measured | **post hoc**: after every campaign it concerns had been read | *the floor's fixed part declared at 2 % in the harness's campaign files.* Every turbofan campaign file since 2026-09-23 declares 2 % where 3 % is registered. Read again at 3 %, no verdict changes: each distinguishable cell is worse than the control or clears the larger floor by a wide margin, and one reading of the pilot's budget moves from at the floor to below it, with the same conclusion. The registered 3 % stands. |
| 2026-09-29 | `69354d5` | configuration | before any selection on the task; the probes' penalties read on the tuning side, nothing on the validation side | *the intensive-care budgets, endpoint and selections, before any selection runs.* Budgets of 50, 200, 1,000 and every stay; full fine-tuning against the network from nothing at 200 under the mixed backbone as the endpoint; every candidate selected per budget and backbone, the pooling among the networks' knobs, in `campaigns/selection-*-physionet2012.toml`. The least gain and the floor wait for the selections' spread over seeds. |
| 2026-09-29 | `c9a0493` | diagnostic | after the selections, read on the tuning side only; before any grid | *the spread over seeds read on a fixed fifth before the thresholds.* The selections' repeats score other stays each and cannot separate a seed from them; the least gain and the floor's fixed part follow the reading declared for `campaigns/seed-spread-physionet2012.toml`. `verdict-statistics.md`, 2026-09-29. |
| 2026-09-29 | `4bbf491` | configuration | by the reading declared beforehand; nothing on the validation side | *the intensive-care least gain and the floor's fixed part.* One seed's area moved by 0.041 at 200 on a fixed fifth, so the least gain rises from 0.02 to 0.045; the fixed part is 0.01. `verdict-statistics.md`, 2026-09-29. |
| 2026-09-29 | `933425f` | configuration | after every selection, before either grid runs; nothing on the validation side | *the intensive-care grids.* `campaigns/curve-physionet2012-mixed5.toml` and `campaigns/curve-physionet2012-stays.toml`: every candidate at its selection's choice per budget, seeds 1 to 5, the thresholds above; the campaign under the stays alone holds the backbone's four ways and the control. |
| 2026-10-03 | `e8aad4e` | criterion | after the diagnosis of the network from nothing on the intensive-care task, read on a fifth of the tuning side; nothing on the validation side; no registered verdict changes | *the stop, at the budget of every labelled unit.* Every arm of the backbone keeps the weights of its best epoch on a fifth of its labels held out by unit, patience ten, the schedule's epochs as a cap; smaller budgets and the patch model unchanged. `intensive-care-curve.md`, 2026-10-03; ADR-0047. |
| 2026-10-03 | `3eec184a` | configuration | after the mixture without SMD was read on a fifth of the tuning side (`intensive-care-curve.md`, 2026-10-02); nothing on the validation side; no registered verdict changes | *the stays republished continuing the turbofan vocabulary read per operating condition, and the task defined over them.* SKAB, the satellite telemetry and the stays are published again continuing `durable/sha256/d63f8e1b…` (channels 127–134, 135–151 and 152–195), the same data and the same held-out units; the stays' window is 48 hours and a minute, as the protocol states, so the 928 readings stamped 48:00 enter it. The task over that publication is `a6a9c653-…`; the backbones of the mixture of four and its leave-one-corpus-out variants are pretrained on it and read under it, the earlier backbones under the earlier publication and task. `manual-handoff.md`, 2026-10-03. |
| 2026-10-04 | `d24431a4` | configuration, criterion | before any run of the stage; the probe and the network from nothing at 50 and 200 stays already read on the validation side under the mixture of five and the stays alone (grids of 2026-09-30), and the seeds' spread at those budgets on a fixed fifth (`verdict-statistics.md`, 2026-09-29); nothing read under the recipe, the new publication or any backbone of the curve | *the scale of pretraining: a narrowed claim, a curve of the probe's gain over the mixture's data, and when its largest point runs.* A new section; the claim above unchanged. The least gain and floor at 20, 50 and 200 stays follow by the registered reading, under the recipe, before the first validation reading. ADR-0049. |
| 2026-10-04 | `b8a3d938` | configuration | by the readings declared beforehand; nothing on the validation side; nothing of the curve read | *the recipe and the thresholds of the scale of pretraining.* No variant replaces a setting on the intensive-care task; on FD001 full fine-tuning starts its head solved. The least gain at 20, 50 and 200 stays and the floor's fixed part are read off the same campaigns. `intensive-care-curve.md`, 2026-10-04. |
| 2026-10-04 | `4ae30acc` | editorial | after the larger shape's run was ordered, before it was read | *the larger shape's parameters.* Width 512 and eight layers, as registered, hold 25.3M parameters, not about 20M; the shape is unchanged. |
| 2026-10-05 | `5eb41f36` | criterion | before either variant of the pretext is pretrained; nothing read under a variant; the probe under the mixture's two seeds read on the validation side (`intensive-care-curve.md`, 2026-10-05), not on the fifth | *when a variant of the pretext replaces the mixture's masks.* By the closed-form probe at 50 stays on a fifth of the tuning side, and only past the difference between two pretraining seeds of the mixture. `pretext-variants.md`, 2026-10-05.
| 2026-10-06 | `11600d46` | configuration | before any campaign on the task; the corpus published and both tasks defined, nothing drawn but a smoke's budgets of 40 and 400 on the tuning side | *the sepsis task.* Its units, sides and protocol over PhysioNet 2019, the endpoint at 400 patients; the least gain and floor follow by the registered reading before the first validation cell. ADR-0052. |
| 2026-10-06 | `cad30ff3` | configuration, criterion | while the point of about 3·10⁸ values and the 2019 stays alone are pretrained; nothing read under a backbone beyond 10⁸; full fine-tuning under the larger shape at 10⁸ already read on the validation side (`intensive-care-curve.md`, 2026-10-05) | *a third point at one set of sources, and how full fine-tuning is set in the larger shape.* The tenth of the new corpora as a descriptive point; the setting of full fine-tuning under 25.3M chosen on the tuning side before the first reading at 10⁹. |
| 2026-10-07 | `d8e237aa` | configuration | by the rule registered beforehand, on the tuning side; nothing read under a backbone beyond 10⁸ | *full fine-tuning's setting in the larger shape.* No variant replaces it at 50 or 200 stays; the setting in force reads the larger shape at 20, 50 and 200 stays. `intensive-care-curve.md`, 2026-10-07. |
| 2026-10-07 | `00f529dc` | configuration, criterion | after the points of about 10⁹ values were pretrained and their campaigns ordered, before any of them was read; the pretext loss of both points still falling by 8 % between their two epochs (`intensive-care-curve.md`, 2026-10-07) | *a longer run at 10⁹.* The point trained four epochs, read only for whether the point was limited by its training. |
| 2026-10-08 | `e951d12f` | configuration | after the power of the rule was simulated from answers already read (`verdict-statistics.md`, 2026-10-08), before any reading under it | *the half of the tuning side.* The pretext rule scores the half held out by seed 101 instead of the fifth, ten seeds kept; the pretraining seeds' difference is read again on the half before the first reading under it. |
| 2026-10-09 | `cab97e84` | configuration | by the rule declared before the reading (`layer-readout.md`, 2026-10-08), after it; no later reading of the stage made | *the probe's reading.* The closed-form probe reads the blocks side by side in every reading of the stage not yet made; the pretraining seeds' difference is read again with it. |
| 2026-10-09 | `1f1aa9b5` | criterion | after the slope and the longer run in today's shape were read (`intensive-care-curve.md`, 2026-10-07) and the seeds' share of the noise was measured (`verdict-statistics.md`, 2026-10-08); before any reading by the blocks side by side | *the seeds in the curve's rules.* The slope and the length of training also require a mean over seeds above twice its standard error, and take the pretraining seeds' difference read by the probe that compares; readings already made stand. The largest point's rule keeps the interval alone: it spends compute, claims nothing, and has been met. |
| 2026-10-09 | `66a41c20` | criterion, configuration | after the pretraining seeds' difference was read by the blocks side by side at every budget (`layer-readout.md`, 2026-10-09); before any reading under it | *the error between backbones.* A comparison of two backbones reads its mean against twice √(SE² + s²), s the spread pretraining alone gives a difference, read from the mixture's pretrainings at every budget, instead of one difference at 50 stays; the mixture is pretrained at seeds 3 and 4 to read s from four. Readings already made stand. |
| 2026-10-09 | `b76eb833` | configuration | after the longer run in today's shape and the larger shape at two epochs were read (`intensive-care-curve.md`, 2026-10-07); before the larger shape's longer run was pretrained | *the larger shape's longer run.* `backbone-scale-1e9-512x8-4ep-m`, read by the length of training's rule against the larger shape at two epochs, by the blocks side by side. |
| 2026-10-10 | `<hash>` | measured | after the mixture was pretrained at seeds 3 and 4 and read by the blocks side by side; before any reading under the error between backbones | *the error between backbones, from four seeds.* s is 0.0060 on the half and 0.0104 on the validation side; the two-seed values first stated, 0.0072 and 0.0095, came from rounded figures (0.0076 and 0.0099 by the formula). |
