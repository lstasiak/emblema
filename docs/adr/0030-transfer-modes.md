# ADR-0030: Transfer modes — four arms of one procedure, low-rank updates as a cap on degrees of freedom, and the encoder reached through a seam the process wires

- Status: accepted
- Date: 2026-09-18; amended 2026-09-19, 2026-09-20 (twice)
- Full text before condensation: commit `5f14447`

## Context

The label-efficiency curve compares one architecture on one task at one label budget, pretrained
against from scratch. Missing was the procedure that turns a backbone into a candidate and answers
the validation windows. Constraints: the preregistration fixes four methods (from scratch, frozen
probe, LoRA, full fine-tuning; the endpoint compares the last with the first); Evaluation may not
import Pretraining's adapters (ADR-0005, ADR-0017); the model is small (4.75M parameters, ≤ 2,651
labelled windows).

## Decision

- **`TransferMode`** is one closed vocabulary with the control as a member: `FROM_SCRATCH`,
  `FROZEN_PROBE`, `LORA`, `FULL_FINE_TUNING`. Same architecture, same head; they differ in which
  weights train and where they start. `AdaptationPlan` pairs a mode with its starting artifact and
  refuses the two nonsensical combinations.
- **LoRA is an ablation of regularisation, not an economy** (Hu et al., 2022). The update starts at
  zero; `LoraSpec` states rank, scale, dropout and targets; a target that reaches no layer is
  refused before anything is wrapped. The layer is forty lines; no library.
- **One linear head over masked-mean-pooled states, fixed epochs, scored after the last** — no
  early stopping, which would spend the validation side on each run. Targets are learnt in units of
  the label ceiling. `AdaptationSchedule` states epochs, batch, rate and weight decay.
- **The outcome is one prediction per validation window** (`AdaptationOutcome`), so per-unit errors
  and every metric are arithmetic over stored rows.
- **Two seeds**: `run_seed` (head, fresh weights, updates, order) and `sample_seed` (which labels).
  The head is drawn first, so every mode starts it identically.
- **The encoder is reached through a seam the process wires**: a structural `BackboneFactory`
  (`pretrained(weights)`, `fresh()`, `width`) in Evaluation's torch adapters, implemented by
  `RestoredBackbones` in the entrypoints (ADR-0017 option a). Moving the encoder to shared code
  would drag its architecture into the kernel; Pretraining's contracts cannot name torch.
- **`MaskedMeanPooling` moved to `shared/adapters/tensors/`**: export, task heads and inference pool
  the same way.
- **`WindowBlock.at(positions)`** reads windows without copying; a block is verified once per
  workspace and opened once per run.

## Consequences

- Domain in `evaluation/domain/transfer/`; port `AdaptationRuntime` with torch and in-memory
  adapters under one contract; `RunAdaptation` never asks for the frozen side.
- Tests: the probe keeps backbone weights bit for bit; LoRA's wrapped layers too, only the update
  gets gradients; the control never asks for pretrained weights; a run repeats bit for bit on host.
- Trainable counts at rank 8: head 257, LoRA ~197k, full 4.75M — reported with every run.

## Alternatives considered

- *A separate axis for where weights come from*: the curve has one axis of four methods.
- *A separate head per mode*: the modes would differ in two things.

## Revisit when

- A third encoder consumer (Serving export) → reconsider moving the encoder to shared code.
- Budget needs fp16 adaptation → precision joins the schedule.
- A per-unit-label task → a second head kind.
- The probe far below fine-tuning → a two-stage arm (Kumar et al., 2022).

## Amendments

- **2026-09-18 — first run** (`docs/verification/transfer-modes.md`, M1, 200 labels, one seed,
  30 epochs, untuned): RMSE from scratch 44.64, probe 38.92, LoRA 24.07, full 22.30; mean predictor
  41.11. The control learnt the mean, so its schedule is fixed on validation before the grid.
- **2026-09-19 — the rate has a shape.** Warm-up share and cosine floor, via `LearningRateSchedule`
  (moved to the shared kernel). Peaks per arm chosen by a rule registered before the sweep (lowest
  validation RMSE of three peaks, one seed, 200 labels): control 22.38 at 3e-4, LoRA 20.70 at 3e-3,
  full 21.10 at 3e-4, probe 38.73 at 1e-2.
- **2026-09-20 — the grid read** (`docs/verification/label-efficiency-curve.md`, 2 × T4, fp32,
  4 modes × 4 budgets × 5 seeds). **The endpoint is not confirmed**: full fine-tuning at 200 takes
  21.5 % off the control's error with the interval above zero, but the practical floor (7.25)
  swallows the 7.17 gain. The control left the mean-predictor plateau under 2 of 5 seeds, the
  pretrained arm under 4. At 1,000 labels, from scratch beat both (14.4 vs 18.1 and 21.1). The probe
  is far below fine-tuning beyond 50 labels. Epoch-measured runs gave small budgets too few steps.
- **2026-09-20 — a floor of steps and the head's start**, registered before any run: a run takes its
  epochs or enough whole epochs to reach a step floor, whichever is more; the head's bias starts at
  the sample's label mean.

## Sources

Hu et al. (2022), LoRA; Alain and Bengio (2017); Kornblith et al. (2019); Kumar et al. (2022);
Loshchilov and Hutter (2019).
