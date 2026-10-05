# ADR-0050: A head solved first, a patience in steps and a probe at initialisation — the knobs of the adaptation recipe at one or two per cent of the labels

- Status: accepted (2026-10-04)
- Date: 2026-10-04

## Context

The curve over the scale of pretraining (ADR-0049) is read at 20 and 50 stays and 50 windows,
under one recipe of adaptation for every point. Three things the runs could not do stood in the
way:

- **Full fine-tuning starts from a head drawn at random.** Its first gradients reach the encoder
  through a head that answers nothing yet, the mechanism Kumar et al. (2022) name for
  fine-tuning that loses to the probe.
- **The stop counts its patience in epochs.** At 200 stays an epoch is about ten steps, so a
  patience of ten epochs runs out inside the warmup; at 50 stays an epoch is two or three steps.
  The stop was registered only at every stay for that reason (ADR-0047).
- **The held-out units of a stop are drawn without regard to outcome.** A fifth of 50 stays is
  ten; with 7 deaths among the 50, those ten hold none of them with probability 0.19, and four
  of 20 stays with 3 deaths hold none with probability 0.49. A stop scored on one outcome reads
  an area of one half after every epoch and stops on noise.

The curve also needs a point at zero pretraining, to separate the probe's closed form from what
pretraining gives.

## Decision

- **`head_start = solved`, a knob of the training regime.** Before the first step the head is
  solved in closed form over the encoder's states as the run receives it, frozen: ridge for a
  quantity, penalised logistic regression for an outcome, the penalty chosen by the same folds as
  the closed-form probe (ADR-0044, ADR-0045). Then the run takes its steps under its schedule.
  - It spends no optimiser step, so the budget is unchanged and no phase length is searched.
  - It is open to every arm that trains its head, the network from nothing included: the
    comparison holds the recipe fixed across arms.
  - It needs a pooling with no weights of its own, as the closed-form probe does, and the
    campaign's penalties. Every arm holds them, and they reach an arm's description only where
    a head is solved, so the descriptions of stored campaigns stay what they were.
- **`patience_steps`, a second patience, exclusive with the one in epochs.** The stop is still
  scored after each epoch; the wait is counted in optimiser steps and begins no earlier than the
  end of the warmup. A best epoch inside the warmup is kept like any other. A patience in epochs
  is counted exactly as before.
- **`stop_division = outcomes`.** Each outcome's units are ranked by the digest of the run's seed
  and the unit, and each outcome holds out its own share, so both sides of the stop hold both
  outcomes. The default division by units is unchanged, so the stop registered at every stay
  draws the same units. A quantity refuses the division.
- **The arm `untrained_ridge`.** The probe solved in closed form over the encoder at the control
  arm's initialisation, the same architecture and the run's seed. The frozen modes may start
  from no weights; the modes that adapt weights still need pretrained ones. It is an arm of its
  own name, so a campaign cannot read it for the pretrained probe by leaving a backbone out.

## Consequences

- The recipe is chosen by the replacement rule on the tuning side, before the curve is read;
  this record adds the knobs, not the choice.
- The probe phase is a closed form; a head trained under its own rate first would be a second
  knob, if the solved head is not enough.
- An untrained probe and the control arm start from the same initialisation under one seed, so
  they are compared on the same terms.
- The flattened plan has three more columns (`patience_steps`, `stop_division`, `head_start`), and
  its penalties are filled where a head is solved first; candidates kept before read as before.

## Alternatives considered

- **A head-only phase of k steps before full fine-tuning.** It is the published method's
  first stage taken literally, but it spends steps from the budget or beyond it, needs
  a rate for the head inside the warmup, and adds a length to search; the closed form reaches
  the head's optimum without any of them.
- **One patience, in steps.** Simpler, but it changes the stop registered at every stay.
- **A backbone pretrained for zero steps as point zero.** It would register a backbone no
  pretraining made, and the loop would have to accept a run of no steps.
