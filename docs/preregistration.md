# Preregistration: what counts as success in the label-efficiency comparison

- Registered: 2026-09-16, commit `e5ffafb`
- Rules in force as of: 2026-09-22, after the amendment `060394b`
- Applies to: the label-efficiency curve on the turbofan task, its single test run, and the
  transfer leg of the synthetic control

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
prediction, before the run. A reader who wants to know how much moved, and when, reads the
register; it is meant to be counted.

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
| Methods | from scratch, frozen backbone with a linear head, low-rank updates, full fine-tuning |
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
the observed token states. A run trains for a stated budget of steps and is scored after the
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

**The peak rate of each arm**, one value for the whole grid: from scratch 1e-3, frozen probe
3e-2, low-rank updates 1e-4, full fine-tuning 1e-3. Peaks are chosen on the validation side at
the endpoint's budget, by a rule fixed before the sweep: three peaks per arm, 200 labelled
windows under seeds 1, 2 and 3, the peak with the lowest mean validation RMSE over the three. A
peak chosen at an edge of its grid is followed by one peak beyond that edge, half a decade away,
at the same seeds and budget, and the rule chooses again, at most twice per arm. Every arm is
swept the same way, so no arm is tuned more than another, and the peaks are swept on each task's
own validation side rather than carried over from another task.

**How a classical baseline is tuned.** A baseline runs as its method was published unless a
selection declared before it runs chooses otherwise. The knobs a selection may turn, and the
values it may turn them to, are named in a committed campaign file whose purpose is
*selection*; the ridge penalty of the convolution baseline is not among them, since every fit
already chooses it by leave-one-out error. A selection never reads the validation side: in each
repeat the seed ranks the task's tuning units, one in five is held out, the budget is drawn from
the rest and every variant is scored on the held-out units. A variant is chosen at each budget
of the curve separately, by the rule of one standard error: every variant whose mean RMSE over
the repeats is within one standard error of the best one's is as good, the standard error
corrected for the overlap of the repeats as Nadeau and Bengio (2003) do for repeated random
holdout — the variance scaled by 1/J + n_test/n_train rather than 1/J — and among those the one
that departs least from the published setting is chosen: fewest knobs turned, then the smallest
ratio on a log scale. A comparison runs the variants a finished selection chose and names that
selection; one naming any other variant is not declared. Until the arms are tuned by the same
protocol, the baselines are tuned at every budget and the arms at the endpoint's alone — an
asymmetry in the baselines' favour, which can make the claim harder to confirm and never easier.

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

The remaining eleven cells of the comparison, four budgets by three transfer modes less the
primary, are secondary. They are tested at the 5 % level with a Holm correction over the family of
eleven, and they are labelled secondary wherever they appear. The family is the registered eleven
whatever has run: a cell that has not run enters the correction with a p-value of one, so a
partial grid is read more strictly than the whole one, never less, and the conclusion states when
the grid is incomplete. A secondary cell's verdict follows the family's word, not its own
interval. A secondary result does not confirm the claim on its own; it describes the shape of the
curve around the endpoint that does.

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

## The synthetic control

The control corpora exist to make a negative result readable. The generator and the certificate
that a pair carries the structure it claims are recorded in ADR-0018; the transfer leg, in
ADR-0033. The leg is measured at the endpoint's budget alone, 200 labelled windows under seeds 1
to 5, the four arms, under the schedule, the floor and the head's start of the curve, on corpora
published with windows of 128 time units at stride 12, a window that spans the process's time
scales. The task is the exact reading of one sensor twelve time units past the window's end,
four strata of the target, its error the RMSE in the sensor's own units. The corpora are
`control-a` and `null-a` for the pretraining and the wide second layouts `control-b-wide` and
`null-b-wide` for the task, half their 800 units held out under seed 1, one in three of those
frozen, 266 validation units on either pair; the backbones `control-a-s` (weights
`sha256:488be6bd…`) and `null-a-s` (`sha256:30f71255…`), 24 epochs of batch 32 at tier S; the
peaks, swept on each pair's own task by the curve's rule, from scratch 1e-3, frozen probe 1e-2,
low-rank updates 3e-3, full fine-tuning 1e-3 on both pairs.

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
| Anomaly detection, satellite telemetry | unsupervised, the benchmark's own protocol | the benchmark's metrics | no |

The anomaly-detection tasks measure whether pretraining improves detection, which is a different
quantity from label efficiency. They never share an axis, a panel or a summary sentence with the
curve.

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

- Endpoint on the validation side, under the configuration in force: **confirmed** on 2026-09-22
  (full fine-tuning +12.3 %, interval [+1.49, +3.30], floor 0.57;
  `docs/verification/label-efficiency-curve.md`). Over four seeds of the five the reduction stays
  between 10.2 and 15.3 %; over the two seeds the sweep did not see, 5.0 %.
- Grid under the floor, validation side: measured on 2026-09-22 on the same configuration
  (same note). The synthetic control's transfer leg: complete, passed
  (`docs/verification/synthetic-transfer.md`).
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
| 2026-09-16 (committed 2026-09-17) | `dfab961` | configuration | before any run | *the held-out engines are the ones the backbone never saw.* The validation side becomes the published corpus's own held-out FD001 engines (then 18) and the tuning side the rest (82), so no validation engine was pretrained on; endpoint, thresholds and floor unchanged. |
| 2026-09-18 | `4d01929` | configuration | before any run on it | *the backbone the curve is drawn from, and how a candidate is made.* `backbone-cmapss-m` (weights `sha256:6830e117…`, four epochs); the shared linear head, scoring after the last epoch, targets in ceiling units, the low-rank placement, the two seeds. ADR-0008, ADR-0030. |
| 2026-09-19 | `3eb3769` | configuration | after a sweep on the validation side, before the grid | *the schedule of every arm, fixed on the validation side before the grid.* Warm-up over a tenth and cosine to 1 %, thirty epochs, batches of sixteen; peaks 3e-4 / 1e-2 / 3e-3 / 3e-4; repeats pooled per engine; the two-sided bootstrap p-value. Sweep in `label-efficiency-curve.md`, 2026-09-19. |
| 2026-09-19 | `3eb3769` | reading | before the grid | *how the registered rules are read, made precise before the grid.* The floor binds the endpoint; the family is the registered eleven whatever ran; `(k + 1) / (B + 1)`; the asymmetric score as a mean per window; the last-window RMSE comparable on the test side only. A limit noted: at fifty windows a cell measures a budget of optimisation as much as of labels. |
| 2026-09-20 | `605a5e0`, `96da833` | criterion, configuration | **after the first grid** (endpoint not confirmed: 7.17 RMSE, [5.92, 8.57], floor 7.25) | *a floor of optimiser steps, the head's start, and a sweep over three seeds.* The floor of 2,000 steps; the head's bias at the mean label; peaks to be chosen again over seeds 1–3; the backbone to be retrained to its plateau by the doubling rule. Recorded as corrections made after a result was seen, applied to every arm alike, with the endpoint, threshold, floor and family unchanged. Grid in `label-efficiency-curve.md`, 2026-09-20. |
| 2026-09-20 | `6b7b191` | configuration | by the rule, before any grid under the floor | *the peaks under the floor, from the sweep over three seeds.* 1e-3 / 1e-2 / 3e-4 / 3e-5 under the four-epoch backbone; the control's peak stands from here. |
| 2026-09-21 | `0e9b0f5` | configuration | after a pilot on the null pair's narrow corpus, before the leg's grid | *the synthetic control's transfer leg: corpora, sweep and power, before its grid.* The forecast task, the backbones `control-a-s` and `null-a-s`, the wide second layouts of 800 units, the curve's grid, peaks swept on each pair's own task, power from the pilot. ADR-0033. |
| 2026-09-21 | `0e9b0f5` | configuration | by the rule, after the ladder of 4 to 64 epochs | *the backbone named again: sixty-four epochs, the plateau reached by the rule.* `backbone-cmapss-m-64` (weights `sha256:78b3c201…`), the fourth doubling lowering nothing. Ladder in `manual-handoff.md`, 2026-09-21. |
| 2026-09-21 | `3426e0c` | configuration | after the leg's sweeps, before its grid | *the synthetic leg's peaks and its power, read off the sweep, before its grid.* 1e-3 / 1e-2 / 3e-3 / 1e-3 on both pairs; 266 validation units suffice. |
| 2026-09-21 | `3426e0c` | configuration | by the rule, before any grid under the floor | *the peaks of the three pretrained arms under the backbone of sixty-four epochs.* 1e-2 / 3e-4 / 3e-4. Sweep in `label-efficiency-curve.md`, 2026-09-21. |
| 2026-09-21 | `3426e0c` | criterion | before the leg's numbers under five seeds; twelve cells at fifty had run and are not read | *the synthetic leg is measured at the endpoint's budget alone.* The leg's grid reduced to 200 labelled windows: the other budgets answer nothing the control's rules ask. |
| 2026-09-21 | `b463ed1` | diagnostic | after both pairs failed their rules at the endpoint | *the ceiling of the synthetic transfer: the second layout over the first's trajectories.* A leak layout as an upper bound. Prediction: if full fine-tuning beats the control here the pair's design is at fault, otherwise the fault is above the data. Outcome: it did not; the fault was above the data. `synthetic-transfer.md`, "the ceiling". |
| 2026-09-21 | `36c3c19` | diagnostic | after the ceiling | *the window against the factors' periods: the coupled pair at a window of 128.* Prediction: if full fine-tuning beats the control at 128 the window was the fault and the control moves to it. Outcome: it did. `synthetic-transfer.md`, "the window". |
| 2026-09-21 | `74fffe4` | configuration, criterion | after the window diagnostic, before any run at 128 under a rule | *the synthetic control moves to a window of 128: sweep, null pair, and the reading of both rules.* Both pairs republished at 128, backbones retrained, peaks re-swept, the null pair's equivalence and the coupled pair's rule read at 128 in that order. |
| 2026-09-21 | `74fffe4` | measured, diagnostic | under the registration above | *the control at 128 read: the coupled pair passes, the null pair fails, and the family's share is measured by swapping the backbones.* Coupled +0.094 [+0.089, +0.099], floor 0.053; null +0.025 [+0.020, +0.029], above its floor of 0.013. The swap declared with two predictions. `synthetic-transfer.md`, "the control closed at a window of 128". |
| 2026-09-21 | `3311640` | measured | under the swap's declaration | *the swapped backbones measured: the first prediction held, the second did not.* Coupled backbone on the null task +0.027 as predicted; null backbone on the coupled task +0.082, within the floor of the pair's own +0.094. The remedy left to a registered decision. `synthetic-transfer.md`, "the backbones swapped". |
| 2026-09-21 | `7a8627c` | criterion, diagnostic | **post hoc**: after the swap's measurement | *the null pair read as the control of leakage it is, and a backbone pretrained on noise to bound what any pretraining gives.* The equivalence rule withdrawn after its measurement and replaced by the leakage reading; the structure's share reported (+0.011 [+0.008, +0.015]); the leg read as passed. The noise backbone declared with its prediction. |
| 2026-09-21 | `7a8627c` | measured | under the declaration above | *the noise backbone measured: the mechanics alone are worse than a fresh encoder.* −0.025 and −0.011; the prediction held; the leg complete. `synthetic-transfer.md`, "a backbone pretrained on noise". |
| 2026-09-21 | `c1c8a9c` | diagnostic, criterion | after the sweep under the 64-epoch backbone, before any run | *the turbofan backbone's pretext window, and peaks at the edge of their grids, before any run.* The pretext-window cell at 32, prediction failed (−0.044, `synthetic-transfer.md`, "the pretext window apart from the task's"); a ladder at 100 cycles ordered, later withdrawn; the edge rule for peaks adopted. |
| 2026-09-21 | `3b1d5de` | configuration | after the normalisation finding, before any run under it | *the turbofan corpus normalised within one operating condition: a backbone on FD001 and FD003, before any run.* A corpus of the two subsets (`a9c73709…`), its ladder, a sweep of all four arms with the edge rule, the endpoint, and the decision rule for what follows. ADR-0034 records the finding. |
| 2026-09-22 | `d5a181e`, `8ed23cc` | reading, measured | reading settled after the sweep and before the endpoint; then the endpoint measured | *the endpoint on FD001 and FD003 read by the endpoint's own rule, and where it runs, before any of it runs.* The stricter reading, all three conditions, settled. Measured: +10.5 % [+0.98, +3.13], floor 0.74, confirmed; the two-subset corpus and `backbone-cmapss-m-8` (`sha256:259fdc70…`) became the configuration, peaks 3e-3 / 3e-1 / 1e-4 / 1e-3. `label-efficiency-curve.md`, 2026-09-22, the edges and the endpoint. |
| 2026-09-22 | `69aaa71`, `f0161e7`, `fcdfe35` | configuration, diagnostic, measured | grid registered before it ran; the check exploratory, choosing nothing | *the grid under the floor on FD001 and FD003, and whether a longer backbone helps the task, before either runs.* The grid at 50, 1,000 and all on two T4s at `d5a181e`. The check: prediction failed, 32 epochs not better than 8 on each seed. Measured grid: +12.5 % at 50 and 1,000, −3.6 at all, the last column no comparison. `label-efficiency-curve.md`, 2026-09-22, both sections. |
| 2026-09-22 | `fdf8053`, `0523683`, `bad25e9` | configuration, measured | after the endpoint on two subsets, before any run on four | *the four subsets read per operating condition: a backbone over all of C-MAPSS, and when it replaces the one over FD001 and FD003, before any run.* The corpus `d63f8e1b…`, its ladder, sweep and endpoint, the replacement rule, and where the endpoint may run. Measured: ladder 8 epochs; peaks 1e-3 / 3e-2 / 1e-4 / 1e-3; endpoint +12.3 % confirmed; replacement +3.2 % [+0.05, +1.03]; the configuration replaced. `label-efficiency-curve.md`, 2026-09-22, the A100 section. |
| 2026-09-22 | `7c543bc`, `fcdfe35` | configuration, measured | grid registered before it ran | *the grid under the floor on the four subsets read per operating condition, before it runs.* The grid at `fdf8053` on an A100, one accelerator per budget. Measured: +16.0 % at 50, +9.0 at 1,000, −5.1 at all; full fine-tuning at all not settled under two seeds. `label-efficiency-curve.md`, 2026-09-22, the last section. |
| 2026-09-22 | `060394b` | diagnostic | before it runs; settles nothing | *the endpoint read again on seeds no sweep has seen, and whether a longer backbone helps this corpus, before either runs.* Seeds 6–10 at 200 under the peaks in force; the 16-epoch backbone (`sha256:8cd60452…`) against the one in force by the replacement arithmetic. Neither reading changes the configuration. Outcome: pending. |
| 2026-09-22 | the commit that adds this row | editorial | after the readings above | The rules in force rewritten in place from the register as it stood at `060394b`; no criterion, threshold, configuration or reading changed, which a diff against that commit shows. Measurements, cost declarations and the full text of diagnostics stay in the commits the rows name and in the verification notes. |
| 2026-09-24 | the commit that adds this row | criterion | before any selection runs | *how a classical baseline is tuned.* Baselines run as published unless a declared selection, scored on held-out tuning units and never on the validation side, chooses a variant per budget by the rule of one standard error with the Nadeau–Bengio correction, ties broken towards the published setting; a comparison runs only what a finished selection chose. Measured when it was written: nothing under this rule. |
