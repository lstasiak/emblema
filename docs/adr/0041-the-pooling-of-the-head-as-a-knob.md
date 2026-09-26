# ADR-0041: The pooling of the head as a knob — free of the channel layout, turned by name on every network

- Status: accepted
- Date: 2026-09-25

## Context

Every network in the comparison pooled the states of a window by their mean before a linear head:
the adapted arms over about a thousand token states, the patch model over each channel's patches.
Read on frozen states with the backbone in force
([`head-and-representation.md`](../verification/head-and-representation.md)), that mean is the
largest identified loss against the tuned trees: a linear head over the mean of the last fifth of
the window sits within the trees' interval at 200 labels, the same head over the whole window
19 % behind them and distinguishable. The last reading alone is not the task; the movement within
the window is. The trees also read one block per channel, and a pooling laid out per channel did as
well as the tail.

The networks' knobs so far were the schedule's rate and decay (ADR-0038); the patch model's shape
was not turnable (ADR-0039).

## Decision

- **A value in the domain, `HeadPooling`**, shared by the adapted arms and the patch model: a
  scheme and the share of the window a tail keeps. Three schemes — the mean over the window, the
  mean over its tail, and a weighted mean under a learnt query (attention) — and no other.
- **Only poolings free of the channel layout.** A pooling laid out per channel is as wide as the
  corpus has channels and binds the head to one layout, which is what the transfer claim cannot
  afford. It stays a diagnostic. The tail matched it on the turbofans and keeps the layout out.
- **Two knobs of the variant grammar, `pooling` and `tail_share`**, turned on the arms and on the
  patch model alike, so one campaign names `full_fine_tuning@pooling=tail,tail_share=0.2` and
  `patch_transformer@pooling=attention` by one rule. Neither touches the compute budget. A tail
  over the whole window computes the mean and is allowed, because a variant turns one knob at a
  time and the tail has to exist before its share is set; a share under any other scheme is
  refused, since it would be recorded as turned and change nothing.
- **The patch model's shape is turnable by name** as well: none of its fields touches the budget,
  and a selection that may turn the rate but not the depth would tune half a network. This is the
  resolution ADR-0039 was left open for.
- **The pooling travels with the candidate**: in the plan's parameters, in the campaign's record of
  the method, in the kept state and in the inference graph, which pools as the run did.
- **A learnt pooling keeps the encoder in the loop under the frozen probe.** The probe's shortcut
  of encoding once and training the head over stored states holds only when the pooling has no
  weights; attention reads the states per token, so its query trains with the encoder run under no
  gradient.
- **Confirmation is a campaign**, `campaigns/pooling-fd001.toml`: the control arm, full fine-tuning
  and the frozen probe under each pooling, paired with the trees on the same engines.

## Consequences

- The label-efficiency curve can be repeated with the head that the diagnostics single out, and the
  networks' hyperparameter search has the pooling among its knobs.
- The mean stays the default, so every campaign recorded before this decision reads as it was run;
  a kept patch model without a pooling in its parameters is read as pooled by the mean.
- The best share of the tail is a fact about the window of one corpus. It is chosen by a declared
  selection, as every knob is, not set by hand.
- Attention pooling adds `width` weights to the head and starts at the mean, so a run under it
  begins where a run under the mean would.

## Alternatives considered

- **A pooling per channel in production.** The strongest reading of one corpus and the end of
  transfer across layouts. Rejected; the classical baselines already hold that reading.
- **The pooling as a field of the schedule.** It is not how a run spends its budget but what its
  head reads, and the patch model's schedule is the arms'. Rejected.
- **A fifth transfer mode.** A mode says what the backbone's weights do; the pooling is the same
  question for every mode. Rejected.
