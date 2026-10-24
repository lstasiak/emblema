# ADR-0063: Several learnt queries in the attention pooling — a readout per aspect of the window that stays free of the channel layout

- Status: accepted
- Date: 2026-10-23

## Context

A head over one state per channel reads the intensive-care task 0.037 above the probe under the
mean over the window, and pretraining has no share in that lift (`head-and-representation.md`,
2026-10-21). That head is as wide as the corpus has channels, so ADR-0041 keeps it out of the
candidates: a backbone must read a corpus of eight channels as it reads one of forty-four.
Pooling the channel states by a mean or by their extremes loses the lift again
(`head-and-representation.md`, 2026-10-22). The attention pooling of ADR-0041 reads the window
under one learnt query: one weighted mean, so one aspect of the window. A network that needs the
last lactate and the lowest pressure cannot read both through one softmax.

## Decision

- **`queries`, a knob of the head's pooling**, one by default, more than one only under the
  attention scheme. The pooling holds `queries` learnt queries, scores every observed state
  against each, takes a softmax per query and lays the weighted means side by side, so the head
  reads `queries` states of the encoder's width. This is pooling by multihead attention over a
  set (Set Transformer), without the attention's projections: the states are scored as they are.
- **Every query starts at zero**, so before the first step each computes the mean the head
  computed without the knob; the head's own weights over the slots differ at initialisation, so
  the queries part from one another as they learn.
- **Free of the channel layout.** A query weighs tokens by what their states hold, never by which
  channel they came from; the state's width is `queries` times the encoder's, whatever the corpus.
  The head laid out per channel stays a diagnostic.
- **Turned on every network that pools**: the arms and the patch model, which read one
  `HeadPooling`. The patch model's head is sized by the pooling's width factor, which it did not
  read before; at one query nothing changes.
- **Refused where a head is solved in closed form**, as the attention pooling already is, since
  a closed form has nothing to train the queries with.
- The knob is named in a candidate's description only where it is turned, so stored descriptions
  stay what they were.

## Consequences

- At one query the module computes what it computed before, bit for bit, which a test holds.
- The head widens with the queries: eight queries over a 256-wide encoder give a head over 2,048
  features, trained from the labels alone. Few labels may not afford many queries; the selection
  reads how many.
- The pooling is still one layer of attention over the final states. A pooling with its own
  projections and several layers (a PMA block) is a further knob if several queries help.

## Alternatives

- **One state per channel, then a pooling over the channels** (`head-and-representation.md`,
  2026-10-22): measured under frozen
  states and a closed-form head, it kept nothing of the per-channel lift.
- **A tail of the window** (ADR-0041): reads where the window ends, not which tokens matter.

## Sources

Lee et al. (2019), Set Transformer, pooling by multihead attention; Vaswani et al. (2017).
