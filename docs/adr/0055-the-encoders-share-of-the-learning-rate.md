# ADR-0055: The encoder's share of the learning rate — a knob of the regime for the modes that train the backbone's weights

- Status: accepted (2026-10-16)
- Date: 2026-10-16

## Context

Full fine-tuning of a pretrained backbone starts from a head that answers nothing yet, and the
head and the encoder learn at one rate. Kumar et al. (2022) show that fine-tuning this way
distorts the pretrained features: the encoder moves along the directions the labels pull on
before the head has found its bearing, and what the pretraining made of the states is lost in
the first steps. Their remedy is a head fitted first and a fine-tuning at a small rate.

The head solved first is a knob of the regime since ADR-0050. Read on the intensive-care task
it replaced nothing under full fine-tuning (0.000 at 50 stays, −0.017 at 200), and a smaller
rate for the whole network moved fine-tuning by at most 0.012 at 200 stays
(`docs/verification/intensive-care-curve.md`). Neither reading slowed the encoder alone while
the head learnt at the schedule's rate, which is the recipe's second half. From 500 stays up
fine-tuning reads the stays worse than the network from nothing (`verdict-statistics.md`,
2026-10-13): the pretrained weights are a worse start than a draw under this recipe.

## Decision

- **`encoder_rate`, a knob of the training regime**, in `(0, 1]`, one by default: the share of
  the schedule's rate the encoder's weights learn at, while the head, the pooling's own
  weights and the rows grown for a task's new channels learn at the whole rate. The grown rows
  hold nothing pretrained, so nothing in them is protected by a smaller step.
- The optimiser takes the parameters in two groups, each at its rate, and the schedule's
  shape scales both by one factor at every step. Under the whole rate the loop is handed one
  group of everything trainable, as before, so every stored campaign repeats as it ran.
- The plan takes the knob only where the mode trains the backbone's weights: full fine-tuning
  and the network from nothing, so a comparison holds the recipe fixed across arms. The frozen
  modes refuse it, since it would change nothing; low-rank updates refuse it, since the
  pretrained weights stay fixed under them and the updates hold nothing pretrained.
- The flattened plan has one more column, `encoder_rate`; a candidate's description names the
  knob only where it is turned, so the descriptions stored before stay what they were.

## Consequences

- The recipe of Kumar et al. can now be read whole: the head solved first and the encoder at
  a tenth or a hundredth of the rate, against fine-tuning's setting at each budget, declared
  before it runs.
- Layer-wise decay of the rate, which the literature reads beside this knob, is not here; it
  would be a second knob if one share over the whole encoder is not enough.
- A slowed encoder under the network from nothing is a legitimate variant of the control arm,
  not an error: whether it helps that arm is read in the same campaign.
