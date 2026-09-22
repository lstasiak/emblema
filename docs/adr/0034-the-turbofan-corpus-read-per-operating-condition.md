# ADR-0034: The turbofan corpus read per operating condition — a channel per sensor and condition, scaled within it, and the default reading left as it was

- Status: accepted
- Date: 2026-09-22

## Context

The C-MAPSS reader reads the 21 sensors as channels and leaves the three operational settings
out; a published version fits one scale per channel over its training units. Two of the four
subsets, FD002 and FD004, were flown at six flight conditions, and in them the condition explains
a median of 100 per cent of a sensor's variance and never less than 88. Under one scale per sensor
over the four subsets, FD001's spread is 0.5 per cent of the scale for the median sensor, the
masked reconstruction is solved by reading the condition off the other channels (the backbone's
loss fell to 0.7 per cent of the trivial predictor's), and the pretrained arms lost the turbofan
task to training from scratch (`docs/verification/label-efficiency-curve.md`).

Publishing FD001 and FD003 alone, one condition each, restored both: FD001 spans 76 per cent of the
scale, the pretext is a task again, and full fine-tuning takes 10.5 per cent off training from
scratch at the endpoint. It also leaves out 509 of the 709 engines — 71 per cent of the rows —
that a corpus for pretraining is meant to offer.

On the released files the settings of every row lie within 0.009 thousand feet of altitude and
0.002 Mach of one of the six conditions, the throttle angle exactly on it, and FD001 and FD003 fly
the first of them, at sea level.

## Decision

**Read per operating condition, a sensor is a channel per condition.** The channel is named
`<sensor>@<altitude>kft-M<Mach>-TRA<throttle>`, and a cycle's 21 readings land on the channels of
the condition its settings name. The scale fitted per channel on the training units is then a
scale within the condition; the tokeniser, its statistics and the manifest are unchanged, and the
statistics still come from the training side alone.

**The six conditions are the reader's constants**, after Saxena, Goebel, Simon and Eklund (2008).
A row names a condition when it lies within half a thousand feet, a hundredth of Mach and half a
degree of throttle of it — five times the released scatter at the least, and eight times under
the distance between the closest two conditions. A row that names none of its subset's conditions is
malformed. The schema is the subset's conditions times the sensors, whatever conditions the rows
happen to fly.

**The single-condition subsets fly sea level**, so their channels are the channels of the sea-level
cycles of FD002 and FD004: the turbofan task's engines keep the readings they had, scaled over more
engines flown at the same condition.

**The default reading is unchanged.** The mode is an argument of the reader and a flag of the
publishing command line, `--per-operating-condition`, which the composition root refuses for any
other corpus. A version read per condition has the files' checksum and a different channel schema,
and a version's identity is its checksum, schema and regime, so the registry keeps the two
readings apart instead of taking one for the other.

## Consequences

- A window holds as many tokens as before, 50 cycles of 21 readings. The vocabulary grows to 126
  channels over the four subsets and stays at 21 over the single-condition ones. In an FD002 or
  FD004 window a channel is observed only at the cycles flown at its condition, so its sampling is
  irregular; the encoder takes that without change.
- Twenty-eight channels are constant within their condition: the five sensors a condition sets
  (T2, P2, farB and the two demanded fan speeds) hold still on 28 of their 30 channels. They take
  a scale of one and carry the condition through their identity.
- Measured on the four subsets read per condition, through the reader, over the 98 channels that
  vary: the units of the most compressed subset move over 0.56 of a channel's scale for the median
  channel, against 0.004 read by sensor and 0.53 for FD001 and FD003 alone, and consecutive
  readings within a unit correlate at 0.63, against 0.02 and 0.75. The single-condition subsets
  read per condition give the values they give by sensor, renamed.
- The default reading's description, units and all 3,367,539 observations of the four subsets,
  hashed before the change and after it, are identical, so every version published so far reads
  as it did.
- The reader of the official test engines, not yet written, reads in the mode of the corpus the
  backbone it serves was pretrained on.
- Whether the four subsets pretrain a better backbone for the turbofan task is measured, not
  assumed: a configuration changes only by a registration of its own, read on the same
  validation engines.

## Alternatives considered

- **Statistics per condition inside the tokenisation scheme.** A condition would become a concept
  of the Catalog's domain and of the manifest's contract for the sake of one corpus; the channel's
  identity gives the same scaling with nothing changed outside the reader.
- **The settings as channels.** The encoder would read the condition from them, and the pretext
  would stay solvable by it.
- **FD001 and FD003 alone.** Published and measured; it gives up 71 per cent of the rows.
- **A scale per unit.** It removes an engine's own level, which is part of what tells wear apart.

## Revisit when

- another corpus turns out to be flown at discrete regimes that set its levels cycle by cycle;
- the reader of the test engines is written.

## Sources

- A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine
  Run-to-Failure Simulation", International Conference on Prognostics and Health Management, 2008.
