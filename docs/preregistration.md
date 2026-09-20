# Preregistration: what counts as success in the label-efficiency comparison

- Registered: 2026-09-16
- Applies to: the first comparative run of the label-efficiency curve, and to the transfer leg of
  the synthetic control
- Amendments: appended as a dated section at the end, never by editing a registered criterion

A curve read after it is drawn can be made to say almost anything: a threshold is chosen, a metric
is swapped, a budget is called the interesting one. This document fixes, before any of those
numbers exist, which comparison decides the question, how large a difference has to be, and what
is reported when the answer is partial or negative. Its date in the repository history is part of
the claim.

## The claim under test

An encoder pretrained without labels lowers the number of labels a new task needs. The comparison
that tests it is the same architecture, on the same task, with the same label budget, pretrained
against trained from scratch. Nothing here claims the encoder beats a classical baseline on a
single task — that is a separate question, answered by the evaluation harness with its own
candidates, and this document does not preregister it.

## The task

Remaining useful life on the first C-MAPSS subset (FD001), a supervised regression with a label
budget.

| | |
|---|---|
| Units | 100 engines with run-to-failure histories; 100 further engines in the official test set |
| Split | by unit, before windowing: 80 engines for training, 20 for validation |
| Window | 50 cycles, stride 5 — 2,766 training windows, 420 validation windows |
| Label | remaining cycles, piecewise-linear with a ceiling of 125 cycles |
| Label budget unit | one labelled window |
| Budgets | 50, 200, 1,000, all (2,766) |
| Sampling | windows drawn from training units, stratified over bins of the target, driven by a seed |
| Methods | from scratch, frozen backbone with a linear head, LoRA, full fine-tuning |
| Repeats | five fine-tuning seeds per cell; one pretraining seed |
| Resampling unit | the engine |

The ceiling of 125 cycles is the convention this task is usually reported under, and it is fixed
here because it moves the error materially: 24.1 % of the windows carry the ceiling as their
label, so a quarter of the evaluation mass is a constant that any model predicts. The ceiling is
not a parameter to be tuned once the numbers are in.

The budget ladder is uneven at the top on purpose — 1,000 labelled windows is already 36 % of
every label the training units hold, so the last rung is a factor of 2.8 where the first is a
factor of 4. The low rung, 50 windows, is 1.8 % of the available labels.

## Metrics

**Endpoint metric: RMSE over the validation windows**, accumulated as summed squared error per
engine so that a comparison is paired on units and groups combine by addition.

Reported with every result, and carrying no threshold:

- the asymmetric prognostics score, which penalises a late prediction more heavily than an early
  one — it is standard for this task, but its exponential tail is dominated by a handful of
  engines, so it is read, not thresholded;
- α-λ accuracy: the share of windows whose prediction lies within ±20 % of the true remaining
  life — bounded, and closer to the decision a maintenance schedule actually makes;
- RMSE over the last window of each engine, which is the protocol published numbers for this task
  are computed under, and therefore the only one of these comparable with them;
- RMSE restricted to windows below the ceiling, the regime in which the task is a task.

**No metric may take the endpoint's place after the numbers are seen.** Swapping the quantity that
decides the question is the failure mode this document exists to prevent, and it is not repaired
by reporting the other metrics honestly alongside.

## The primary endpoint

**Full fine-tuning against training from scratch, at 200 labelled windows.** One comparison, named
in advance, and therefore carrying no correction for multiplicity.

The claim is confirmed when both hold:

1. the relative reduction in RMSE is at least **10 %**, and
2. the whole 95 % confidence interval of the paired difference lies above zero — bootstrap over
   the 20 validation engines, 10,000 resamples, the same engines for both methods.

Ten per cent is one full step between published methods on this subset: reported errors run from
about 11.8 to about 19.8 RMSE, and consecutive published results differ by roughly 0.5 to 2 RMSE,
which is 4–12 % of the base. A difference of that size is the smallest one that a reader of this
literature would recognise as a result rather than as a repetition.

Two hundred labels is the low-budget regime without being the noisiest rung: 7 % of the available
labels, four times the smallest budget, and small enough that a model which needs labels to learn
the task has not yet had them.

## Secondary comparisons

The remaining eleven cells of the comparison — four budgets by three transfer modes, less the
primary — are secondary. They are tested at the 5 % level with a Holm correction over the family
of eleven, and they are labelled secondary wherever they appear. A secondary result does not
confirm the claim on its own; it describes the shape of the curve around the endpoint that does.

## When a difference is too small to matter

A difference is reported as *statistically distinguishable, practically nil* when it is smaller
than

    max(3 % of the from-scratch RMSE at that budget,
        the standard deviation of that RMSE over its five seeds)

The fixed part keeps the floor from collapsing when a run happens to be quiet; the measured part
keeps it from sitting below the noise the experiment itself generates. Both are fixed here; the
measured part is read off the same run it judges, not chosen afterwards.

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

Twenty validation engines are the binding constraint of this design, not the five seeds. For a
paired difference over 20 units, the half-width of a 95 % interval is about 0.47 standard
deviations of the per-engine differences; under the Holm correction over eleven secondary
comparisons it rises to roughly 0.8. Wide intervals are therefore a property of the experiment as
designed, known now, and they will not be reinterpreted later as a finding about the method.

## The synthetic control

The control corpora exist to make a negative result readable: a pair that shares latent structure
must transfer, and a pair that shares none must not. The generator and the certificate that the
pair carries the structure it claims are recorded in ADR-0018.

- **The coupled pair** is judged by the rule the curve is judged by: the advantage over training
  from scratch must clear the practical floor with its whole interval above zero.
- **The null pair** is an equivalence claim, so it is stated as one: the whole interval must lie
  *within* ±the practical floor. Failing to reject zero is not evidence of zero, and it will not be
  reported as if it were. The number of generated units is chosen before the run so that the
  expected interval is narrower than the floor — units are a parameter of the generator here, so
  the power of this leg is a decision, not an accident.

## Protocols of the other tasks

Declared now, so that no task's protocol is chosen once its numbers are visible.

| Task | Protocol | Primary metric | Label-budget axis |
|---|---|---|---|
| Remaining useful life, turbofan (FD001) | supervised, label budget | RMSE | yes |
| Anomaly detection, industrial testbed | unsupervised: fitted on normal data, scored on the rest | PR-AUC, F1 at a fixed threshold | no |
| Anomaly detection, server machines | unsupervised | PR-AUC | no |
| Binary classification, intensive-care stays | supervised, label budget | AUROC with calibration | yes |
| Anomaly detection, satellite telemetry | unsupervised, the benchmark's own protocol | the benchmark's metrics | no |

The anomaly-detection tasks measure whether pretraining improves detection, which is a different
quantity from label efficiency. They never share an axis, a panel or a summary sentence with the
curve.

## The test set

The official test engines are frozen when the task is created and are used **once**, at the end,
for the variants finally chosen — no hyperparameter, no variant and no decision to continue is
taken on them. Every number produced before that run is labelled *validation* wherever it appears.

Seven of the 100 test engines are shorter than the 50-cycle window, the shortest holding 31
cycles. They are scored on the observations they have: the model reads a window as a set of
tokens and does not require it to be full. None of them is dropped, and no padding is invented for
them — dropping them silently is the usual way the published numbers for this task stop being
comparable with each other.

## Amendments

An amendment is appended here as a dated section, stating what changed, why, and what had already
been measured when it changed.

### 2026-09-16 — the held-out engines are the ones the backbone never saw

**Changed.** The validation side is no longer 20 engines drawn by the task. It is the 18 FD001
engines the published corpus already holds out, and the tuning side is the remaining 82.

**Why.** The corpus is published with a split over units, and the statistics that normalise every
token are fitted on its training side only. A task that drew its own validation engines would draw
most of them from that training side — engines whose unlabelled windows the backbone was pretrained
on and whose values contributed to the normalisation. The comparison would then be transductive,
and the first question a reader asks would be about the design rather than the result. Inheriting
the corpus's own held-out units costs two engines and removes both objections.

**Consequences.** The paired interval is now over 18 units, so its half-width is about 0.50 standard
deviations rather than 0.47; under the Holm correction over the secondary family it is
correspondingly wider. The tuning side holds 2,651 windows and the validation side 535, so the
largest budget, *all*, is 2,651 rather than 2,766 and the ceiling still carries a quarter of the
tuning windows. The endpoint, the thresholds and the floor are unchanged. The
split is recorded when the task is created and never recomputed, so a later publication of the
corpus under another seed cannot move it.

**Measured when this changed.** Nothing. No model has been fine-tuned on this task and no comparison
has been run; the unit counts come from replaying the published split over the raw unit keys.

### 2026-09-18 — the backbone the curve is drawn from, and how a candidate is made

**Changed.** Nothing registered above; this names what was left open. The pretrained arm of every
comparison starts from one backbone: `backbone-cmapss-m`, registered as
`de3815c9-6ff4-4fed-9878-df1cbcd204bd` — 256 wide, 4 heads, 6 blocks, 4,751,872 encoder
parameters over the 21 channels of the turbofan corpus, pretrained in half precision on a T4
under seed 1 at commit `061fd05`, weights
`sha256:6830e117d06f9882a240e3e763f792f8d6dca5c9c58efab462130a40c9664930`, over the corpus whose
manifest is `sha256:a00c3865aba466147cc2fb731cc1903afcb91bdc94c2027379fab5232ca789c1` (window 50,
stride 5, the split of 2026-09-16). The two smaller backbones of the same corpus stand as a
reference of precision and are not on the curve.

How a candidate is made is fixed with it. Every method answers through the same linear head over
the mean of the observed token states; a run trains for a stated number of epochs and is scored
after the last, never stopped on the validation error; targets are learnt in units of the label
ceiling. Low-rank updates go beside the attention's projections and the feed-forward network's
linears of every block, and start at zero. The budget of labels is spread over four strata of the
target. Two seeds are kept apart: the seed of the draw fixes which labels, the seed of the run
fixes the head, the fresh weights of the control arm, the updates and the order of windows. The
five repeats of a cell — the table's "fine-tuning seeds" — are five values of one seed that
drives both: each repeat draws its own labels and learns from them, so the spread over repeats
includes which labels a budget happened to hold, and the two methods of a cell are compared under
the same seed and therefore on the same labels.

**Why.** The choice of backbone is recorded in ADR-0008 and the procedure in ADR-0030; a
registration that named the methods without naming the weights and the procedure would leave
both to be chosen once the first errors were visible.

**Measured when this changed.** Nothing on this backbone. The procedure has run on a two-engine
sample of the corpus under a small untrained encoder, to show that each method trains and
answers; no number from it is a result.

### 2026-09-19 — the schedule of every arm, fixed on the validation side before the grid

**Changed.** Nothing registered above; this fixes what the first run left open and names how
the seeds enter the endpoint. Every arm learns under one shape of the learning rate: a linear
warm-up over the first tenth of the run's optimiser steps, then a cosine decay to one per cent
of the peak by the last step, thirty epochs, batches of sixteen, no weight decay. The peak
rate per arm, one value for the whole grid, is: from scratch 3e-4, frozen probe 1e-2, low-rank
updates 3e-3, full fine-tuning 3e-4. The low-rank updates keep rank 8, α = 16, no dropout,
beside the attention's projections and the feed-forward network's linears.

In the paired comparison of a cell, the five repeats of each arm are pooled per engine — the
squared errors of every repeat are added per engine before the pair is made and resampled — so
a cell's RMSE is the root of the mean squared error over its repeats and the interval is over
the 18 engines as registered; the spread of RMSE over the five repeats is reported beside it
and is the measured part of the practical floor. The p-value the Holm correction ranks is
two-sided, twice the smaller share of bootstrap resamples on either side of zero. The grid runs
on the free GPU platform in single precision, two accelerators in one session sharded by seed.

**Why.** The first run of each arm (ADR-0030, 2026-09-18) used a constant rate and no warm-up,
and the control arm trained from scratch ended above the mean predictor: the primary endpoint
is measured against that arm, so a control at the trivial predictor would confirm the claim for
the wrong reason. The rate's shape and each arm's peak were therefore chosen on the validation
side at the endpoint's budget, one seed, before any cell of the grid — the one set per mode this
registration allows — and by a rule fixed before the sweep: the lowest validation RMSE among
three peaks per arm, the shape chosen for the control applying to every arm. The sweep first
covered the control and the probe alone, with the other two arms keeping their first-run rates;
when the control landed level with full fine-tuning, the other two arms were swept over three
peaks the same way, so that no arm is tuned more than another. Tuning the control is
conservative for the claim; tuning every arm equally is fair to it, and both are stated here
with every number.

**Measured when this changed.** The sweep, 200 labelled windows drawn under seed 1 from 73 of
the 82 tuning engines, validation over the 18 held-out engines (535 windows), M1 Pro, MPS,
fp32 (`docs/verification/label-efficiency-curve.md`); validation RMSE, the chosen cells in bold:

| arm | peak rate | constant rate | warm-up 10 % + cosine to 1 % |
| --- | --- | --- | --- |
| from scratch | 1e-3 | 44.64 | 25.75 |
| from scratch | 3e-4 | 39.14 | **22.38** |
| from scratch | 1e-4 | 34.85 | 29.69 |
| frozen probe | 1e-2 | 38.92 | **38.73** |
| frozen probe | 3e-3 | 39.31 | 40.14 |
| frozen probe | 1e-3 | 40.08 | 40.81 |
| low-rank updates | 3e-3 | — | **20.70** |
| low-rank updates | 1e-3 | 24.07 (first run) | 21.84 |
| low-rank updates | 3e-4 | — | 22.90 |
| full fine-tuning | 3e-4 | — | **21.10** |
| full fine-tuning | 1e-4 | 22.30 (first run) | 22.08 |
| full fine-tuning | 3e-5 | — | 22.16 |

No cell of the grid had run. What these numbers already say is recorded so that the grid's
reading cannot be mistaken for a surprise: at this budget and seed the control arm, once given
a rate it can start at and a decay to settle under, stands within a few per cent of the two
arms that update the pretrained encoder, and the mean predictor scores 41.11 on the same
windows. Whether the endpoint holds is what the grid measures, over five seeds and with its
interval; nothing here changes the endpoint, the threshold or the floor.

### 2026-09-19 — how the registered rules are read, made precise before the grid

**Changed.** Nothing registered above; this fixes, before any cell of the grid, five readings
the registered text left to the code, so that none is chosen once the numbers are visible.

1. *The floor binds the endpoint.* The claim is confirmed only when the reduction takes at
   least 10 % off, keeps its whole interval above zero **and** is not smaller than the practical
   floor. The registration calls a reduction under the floor practically nil; the same document
   cannot call it confirmed. The floor's measured part is the standard deviation of the
   from-scratch RMSE over its five seeds, so this binds only when the control's spread over
   seeds exceeds a tenth of its error — the case in which five seeds do not agree on what the
   control scores, and a difference of that size is not a result.
2. *The secondary family is the registered eleven whatever has run.* A cell that has not run,
   or whose control has not, enters the Holm correction with a p-value of one: it is never
   rejected and holds the cells that ran to the levels the family of eleven sets. A partial grid
   is therefore read more strictly than the whole one, never less; and the conclusion states
   when the grid is incomplete — how many of the twelve compared cells and how many of the five
   seeds it holds — because the reading of a partial grid is not the registered one. A secondary
   cell's verdict follows the family's word, not its own interval, so the two never disagree.
3. *The bootstrap p-value counts the observed reduction as one resample on its own side*,
   ``(k + 1) / (B + 1)`` per side, twice the smaller. No p-value is zero; the smallest is two in
   ten thousand and one, which says how many resamples were drawn.
4. *The asymmetric score is reported as a mean per window*, not the benchmark's sum. The sum
   over five hundred overlapping validation windows does not reproduce the benchmark, which sums
   one answer per engine at a cut-off, and it would scale with the number of windows so that the
   validation and the test side could not be read together; the mean removes only that factor.
   Inside one grid the two differ by a constant, so no ranking or ratio changes. The score
   stays read, not thresholded: on the smoke run ten of the 535 windows carry a fifth to a
   quarter of it.
5. *The last-window RMSE is the benchmark's protocol on the test side only.* The published
   numbers are computed on trajectories cut short of failure, whose last window carries a
   remaining life of roughly 7 to 145 cycles. On the validation side every engine runs to
   failure, and its last window ends within four cycles of it (0 to 4 on the smoke run's 18
   engines), so the reading there is the error at the end of life. It is reported with that
   caption and is not to be set beside published errors; the comparison the registration
   promised is made on the test side, once.

**Why.** Each of the five was decided by the code rather than by the registered text, and a
reviewer found each before the grid: the endpoint's rule and the floor's rule could conflict
without the text saying which wins; the family the correction ran over was whatever lay on
disk; a p-value of zero was printed for a bootstrap; the score's aggregation was unstated; and
a caption invited a comparison the validation side cannot bear. Fixing them now costs nothing
and leaves nothing to choose later.

**A limit known now, recorded now.** At the smallest budget, fifty labelled windows in batches
of sixteen make four optimiser steps an epoch and 120 in the run. On the smoke run at that
budget the arm trained from scratch scored 41.11, the mean predictor's own number on these
windows; the probe 40.82 and the low-rank updates 40.78. The registration's reason for tuning
the control's schedule — that a control at the trivial predictor confirms the claim for the
wrong reason — applies to the secondary cells at fifty as well: what they measure is as much a
budget of optimisation as a budget of labels. The grid is not changed for it; the cells at fifty
are read knowing it, and the crossing point, if there is one, is looked for above them.

**Measured when this changed.** No cell of the grid. The smoke run of the grid's path — the
four modes at fifty windows under one seed, M1 Pro, MPS, fp32
(`docs/verification/label-efficiency-curve.md`) — had scored from scratch 41.11, probe 40.82,
low-rank 40.78, full fine-tuning 34.81; it is a check of the path, and under the rules as read
here its one secondary rejection, the probe at fifty (p = 0.0084), is rejected among three cells
and not among the registered eleven, which is what item 2 is for.

### 2026-09-20 — a floor of optimiser steps, the head's start, and a sweep over three seeds

**Changed.** Three parts of the procedure, after the first grid and before any run under them.

1. *Every run stands on a floor of 2,000 optimiser steps.* A run trains for thirty epochs or for
   as many whole epochs as reach 2,000 steps, whichever is more, and the warm-up and the decay
   span the run so lengthened. At the registered budgets, in batches of sixteen, that is 500
   epochs at 50 labelled windows, 154 at 200, 32 at 1,000 and the thirty already stated at
   2,651: no cell trains for fewer steps than before, and no cell trains for fewer than the cell
   at 1,000 did in the first grid.
2. *The head's bias starts at the mean of the labels the run holds*, in units of the ceiling,
   under every mode. Its weights are drawn as before.
3. *The peak rate of every arm is chosen again under this floor*, by the rule of 2026-09-19 with
   one change: the sweep runs at 200 labelled windows under three seeds, 1, 2 and 3, and an arm
   takes the peak with the lowest mean validation RMSE over the three. The three peaks per arm
   are the three swept on 2026-09-19; the shape stays a warm-up over the first tenth and a cosine decay to one per cent.
   The peaks so chosen replace the ones registered on 2026-09-19 for every run from then on,
   including the transfer leg of the synthetic control.

Nothing else moves. The endpoint, its threshold and its interval, the practical floor and its
measured part, the secondary family and its correction, the budgets, the seeds, the metrics and the
backbone stand as registered. The backbone is the next thing to be named again: the one the first
grid used had trained for 2,532 steps, four epochs, with its validation loss still falling by seven
per cent an epoch. It is trained again under the same configuration with the budget doubled three
times over, eight, sixteen and thirty-two epochs, each a run of its own with the warm-up and the
decay spanning it, and the runs are read by the rule the saturation curve was read by: the first
doubling that lowers the best epoch's validation loss over the whole held-out side by less than five
per cent is the plateau, and the run at the end of that doubling is the backbone. If the last
doubling still lowers it by five per cent or more, the budget is doubled again before anything is
named. The backbone so chosen will be registered here by its weights, in a further dated section,
before any cell is trained from it; the peaks of the three arms that start from it are then swept
again under it, by the rule of point 3, and the control's peak stands, since the control never sees
the backbone.

**Why.** The first grid (`docs/verification/label-efficiency-curve.md`, 2026-09-20) was read
under thirty epochs whatever the budget, so a cell had about twice as many optimiser steps as it
had labels: 120 at 50 windows, 390 at 200, 1,890 at 1,000 and 4,980 at 2,651. No cell had
converged when it was scored, and at 200 the endpoint was decided by how often thirty epochs got
an arm off the plateau of the mean predictor — two seeds of five for the control, four for the
pretrained arm — which is a fact about optimisation under a small number of steps, not about
what the labels teach. The registration's own reason for tuning the control's schedule, that a
control at the trivial predictor confirms the claim for the wrong reason, applies to a control
that thirty epochs leave on the plateau under three seeds of five. A floor in steps gives every
cell the same chance to leave it and leaves the budget of labels as the only thing that differs
between cells of one arm. Two thousand is the number of steps the cell at 1,000 had, where the
three arms that step the encoder had become indistinguishable, rounded up; it is not a claim of
convergence, which the next grid measures as the first did, from the losses per epoch.

The head's default bias is a draw, and under three seeds of five in the first grid the first
epoch's loss stood at five to twelve times the variance of the target: a run of 390 steps spent
a share of them walking the bias back to the mean. Starting it there is what a predictor with no
information says and costs no seed its comparability, because every arm of a cell starts from
the same value.

The peaks are chosen again because the ones registered on 2026-09-19 were chosen over 390 steps
and one seed, and one seed does not see how often an arm leaves the plateau: the control's peak
was chosen under seed 1, one of the two seeds it left the plateau under. Three seeds and the
mean over them see it. The rule, the three peaks per arm and the shape are otherwise unchanged,
and every arm is swept the same way, so no arm is tuned more than another.

**What this is, stated plainly.** These are corrections made after a result was seen, and they
are recorded as such. What keeps them from being a choice made on the numbers is that they are
fixed here before any run under them, that they apply to every arm alike, and that the next
grid is read by the same endpoint, threshold, floor and family as the first. The first grid's
reading stands in its note as the reading under the rules of its day.

**Measured when this changed.** The first grid in full, 80 cells over two accelerators: the
endpoint at 200 not confirmed (reduction 7.17 RMSE, interval [5.92, 8.57], floor 7.25); the
low-rank arm at 200 distinguishable above the floor (11.12, [9.07, 13.50]); at 1,000 the three
arms that step the encoder indistinguishable; at 2,651 the arm trained from scratch better than
both pretrained arms. No run had been made under the floor, the head's start or the new peaks.

**Cost, declared now.** Under the floor the grid spends about 165,000 optimiser steps in the
three arms that step the encoder, against 110,000 in the first grid; at the 0.39 s per step the
platform showed, about eighteen hours of one T4, two sessions of the platform or one over two
accelerators. The sweep spends 2,000 steps per run over three arms, three peaks and three seeds,
about six hours of one T4, and the probe's runs are minutes. The three runs of the backbone
spend fifty-six epochs together, about three hours of one T4 at the 0.31 s per step its first
run showed, and the second sweep of the three pretrained arms about four and a half hours more.
