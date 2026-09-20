# ADR-0033: The transfer leg of the synthetic control — a forecasting task whose truth is generated, one ground-truth port for every task, and the generator's specification shared by both sides

- Status: accepted
- Date: 2026-09-20

## Context

The synthetic control exists to make a negative result readable (ADR-0018): a pair of layouts
that shares latent structure must transfer, and a pair that shares none must not. Its
pretraining leg has run; its transfer leg had not, and the first label-efficiency grid came
back negative on the registered endpoint (ADR-0030, `docs/verification/label-efficiency-curve.md`),
so the leg is now the thing that decides what that reading says about the encoder.

The Evaluation context knew one kind of task. Its aggregate read a label from the moment a
unit failed under a ceiling, its one ground-truth port answered a failure time per unit, and
its one real adapter counted cycles in the turbofan files. A generated corpus has no failures.
It has hidden factors and sensors that follow them, and what a task on it can ask has to be
something the pipeline can learn from labels alone and that pretraining on the other layout of
the pair could make cheaper to learn.

The adapter that answers such a task's truth lives in Evaluation and needs the generator, which
lived in the Catalog's adapters; the contract that keeps bounded contexts apart forbids that
import.

## Decision

**The task is a forecast of one sensor's exact reading, a fixed time past the window.** Under
`ForecastScheme(channel, horizon)` a window's target is the noiseless value of the named
sensor at `ends_at + horizon`, in the sensor's own units, with no ceiling and no transform. The
two synthetic tasks name the first sensor of the second layout of each pair, twelve steps past
the window: half the shortest period the factors are given. A forecast rather than the reading
at the window's end, because the latter is close to what the reconstruction objective already
teaches and would make the leg a tautology; a forecast needs the frequencies behind the signal,
which is the structure the coupled pair shares and the null pair does not, as well as the phase
and amplitude the window reveals. The same task is posed on both pairs: on the null pair the
sensor follows factors private to its layout, so the task is as learnable from labels as on the
coupled pair and only the pretraining is uninformative, which is the condition an equivalence
claim needs.

**One ground-truth port for every task, keyed by window.** `UnitLifetimes.failure_times(units)`
becomes `GroundTruth.truths_of(windows)`: one number per window, whose meaning is the task's
label scheme's business — the failure of the unit for a remaining-life task, the exact reading
for a forecasting one. The aggregate holds a closed set of schemes,
`RemainingLifeScheme | ForecastScheme`, and turns each truth into a target itself; each scheme
also states the unit targets are learnt in, the ceiling or one, so the runtime scales targets
without knowing which task it serves. A truth per window rather than per unit because a
forecast varies along a unit and a failure does not, and the port that fits both is the one
that takes windows; the turbofan adapter answers every window of an engine with the same
moment. This is the revisit ADR-0026 named: a second shape of task arrived, and one port with
a richer result was chosen over two.

**The generator's specification moves to `shared/adapters/synthetic`; the reader stays in the
Catalog.** The latent process, the sensor layouts, the draws and the presets are the facts both
sides read — the Catalog to emit observations, Evaluation to answer what a sensor exactly was —
as the run-to-failure files of the turbofan corpus are read by the Catalog's reader and by
Evaluation's ground truth alike. The noiseless signal model is extracted from the reader as
`SensorSignal`: the reader adds the measurement noise, drops what the layout fails to report
and rounds; the ground truth reads the same signal at any instant and rounds to the corpus's
precision. The bytes of every corpus are unchanged, which the pinned checksums assert. It is an
adapter package without a port, like the array codec and the tensor helpers already there, not
an extension of the kernel.

**The frozen test side of a generated corpus is a share of its held-out units.** A generated
corpus publishes no test set, so a third of the units the corpus holds out, ranked under a
seed, is frozen when the task is defined, and the task's tuning and validation sides are the
corpus's own sides less those units. The turbofan task keeps naming its official test engines.

## Consequences

- The transfer report and the curve report take a task by name; a stored grid records the
  task it was run on, and shards of two tasks are refused as one curve, as shards of two plans
  are. The readings beside the endpoint — the error below the ceiling, at the end of life, the
  benchmark's score — are the remaining-life task's own and are left blank for a forecast; the
  endpoint is the RMSE under either scheme, and the trivial predictor read against it is the
  mean, the ceiling predictor being the remaining-life task's alone.
- The forecasting task travels the whole road on the miniature control corpora in a test: both
  layouts published as the command line publishes them, the task defined over the second, its
  labels read off the generator, every mode learning from a budget of them and answering every
  validation window. The road exists before anything expensive runs on it.
- A third kind of task — a classification with per-stay labels — enters through the same port
  and a third scheme in the closed set, with its own adapter reading the outcomes file; nothing
  in the aggregate or the use cases has to learn about it beyond the scheme.
- The synthetic leg's budgets, unit counts, seeds and the size of its equivalence floor are
  registered in `docs/preregistration.md` before it runs, as the curve's were; this record fixes
  the task and the machinery, not the reading.

## Alternatives considered

- **The hidden factor's value as the label.** Rejected: on the null pair the sensors do not
  follow the shared factors, so the task would be unlearnable there and the equivalence would
  hold for a trivial reason.
- **The reading at the window's end.** Rejected above: too close to the pretext task.
- **A second port beside the lifetimes one, chosen by the use case per scheme.** Rejected: two
  ports for one question, and every process would wire an adapter of each whether or not it
  serves a task of that kind.
- **A ground-truth adapter parametrised by a function the composition root takes from the
  Catalog's adapters.** Rejected: it keeps the letter of the context boundary and breaks its
  sense, and it hides the shared fact — the generator's specification — inside a closure.
- **Leaving the generator where it was and duplicating the signal model in Evaluation.** Rejected:
  two definitions of what a sensor exactly reads would be the place the corpus and its truth
  could quietly disagree.

## Revisit when

- A task's truth is not a number per window — a class, a set, a curve. Then the port's result
  becomes a value object and the schemes read it, rather than a second port appearing.
- The evaluation harness's campaigns take over the scripts' task registry; the frozen share of
  a generated corpus then becomes a field of the task definition rather than a rule of the
  composition root.
