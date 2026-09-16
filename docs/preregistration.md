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
