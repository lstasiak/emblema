# ADR-0031: Reading the clinical stays — a stay is the unit on the protocol's 48-hour axis, descriptors are timeless tokens, and the test set stays out

- Status: accepted
- Date: 2026-09-18

## Context

The irregular corpus of the programme is the PhysioNet/CinC Challenge 2012: 12,000 ICU stays in
three sets of 4,000, each stay a file of `Time,Parameter,Value` rows — six general descriptors at
`00:00`, then up to 37 clinical variables at the minutes they were measured, over the first 48
hours. The data spike priced it with set A pretraining, set B validating and set C as the frozen
test set, and counted 1,733,980 observations in set A with the admission weight left out. It is the
first corpus with static descriptors on real units, the first whose units may hold no measurement
at all — 3 of the 4,000 stays of set A — and the sparsest by far: a stay holds a median of 422
values over 37 channels, and the rarest variable, cholesterol, is measured 315 times in 4,000
stays.

Measured on sets A and B before the reader was written: all six descriptors are present in every
file at `00:00`, but not always in the first six rows — in 49 and 60 files a measurement taken at
admission sits between them; height is unrecorded (`-1`) in 47 % of stays, weight in 8 %, gender in
a handful; 455 and 473 rows carry the stamp `48:00`; `Weight` is at once a descriptor and a series
of 125 thousand later rows, and one file of set B records it twice at `00:00`; 3,744 rows repeat a
(time, variable) pair, two measurements in one minute; and a few series values are physiologically
impossible — temperatures below zero, a pH in the hundreds — as the challenge's own notes warn.

## Decision

**A stay is the unit**, keyed `set-a/132539`; the record identifier is the key and nothing else.
**The extent is the protocol's, not the stay's**: `[0, 48 h + 1 min)` for every stay, because a
stamp names the minute it starts and the protocol's last stamp is `48:00`. A window of the whole
stay is then one window per unit, and a stay that ended early is a stay whose later hours hold
nothing rather than a shorter unit. **Time is hours since admission**, as the budget file prices
this corpus, at the minute the source records.

**Descriptors become timeless tokens.** Age, height and gender as the numbers the files carry — a
z-scored binary channel is ±1 — and the ICU type as a presence token of one of four ward channels
(`ICUType/coronary_care`, `ICUType/cardiac_surgery_recovery`, `ICUType/medical`,
`ICUType/surgical`), because the wards have no order a z-score could respect and one channel with
the codes 1–4 would invent one. A descriptor recorded as `-1`, or not carried by the file, is no
feature. **The weight at admission is the first measurement of the `Weight` series**, at hour
zero: it is a weight, later weights follow in the same series, and a channel named twice would
put one quantity under two vocabulary entries. The reader therefore counts 3,674 observations more
in set A than the spike did, and the full-corpus test states the difference.

**Only sets A and B are read.** Set C is the challenge's test set and the frozen side of the
classification task — a task names its frozen side and does not materialise it (ADR-0026) — and a
corpus a backbone is pretrained on must not hold it. **Values are read raw.** The impossible
temperatures and pH values stay as published: a cleaning rule would be a second corpus under the
same checksum, as a second subsampling policy would be for the satellite corpus (ADR-0025), and
what a model does with a value 88 deviations out is a fact for the sanity note, not a decision
hidden in a reader. **Rows are trusted in file order**, and a row that goes back in time is
refused as in the SKAB reader; the released files never do.

## Consequences

- Set A reads as 4,000 units, 44 channels — 37 timed, 7 timeless — and 1,737,654 observations.
  Describing it, which parses every file for the checksum and the counts, takes 10 s on the
  development machine; the sanity pass, which reads every stay again to fit the scheme, 58 s
  (`docs/verification/window-sanity.md`, 2026-09-18).
- The extra minute of the extent is what lets the last stamp be read, not what puts it in a
  window: a whole-stay window covers `[0, 48)`, so the rows stamped `48:00` — 455 in set A, 473 in
  set B — are read, counted and checksummed but fall inside no window. Every corpus leaves such a
  tail; here it is one minute wide.
- Five channels never vary in the fitted data: the four ward tokens and `MechVent`, which the
  challenge records only while ventilation is on. Their statistics fall back to a scale of one and
  their normalised value is zero; what they carry is the channel identity, which is what a presence
  token is for.
- Twenty-three channels hold values beyond eight fitted deviations, the widest `pH` (88, values on
  another scale), `Urine`, `Temp` (below zero) and `SaO2`. They are the source's, and one more
  argument, after SMD, for statistics fitted per corpus rather than across the mixture.
- A channel measured only on the held-out side has no statistics and the tokeniser refuses its
  token. Among thousands of stays this does not happen; among the four of the sample it does, so
  the publication test names the stays it holds out rather than drawing them.
- The sample in `tests/data/physionet2012/` is four stays copied byte for byte, chosen for what
  the format allows: unrecorded descriptors, a measurement between the descriptors, a stay of
  descriptors alone, a weight recorded twice at admission.
- The reader is the fourth text reader on the shared line helpers and the fifth downloaded corpus
  through the unchanged publication path; nothing of the domain, the tokeniser or the archive
  changed. Publishing the full corpus, pretraining on it and the classification task are the
  second half of the ticket, on the machine that publishes.

## Alternatives considered

- **The stay's own extent**, one minute past its last stamp. Rejected: the protocol frames every
  stay as 48 hours, and a whole-stay window would differ in length per unit.
- **Minutes as the time axis**, exact integers. Rejected: the budget file and the satellite corpus
  price windows in hours, and a sixtieth of an hour is exact enough in a double.
- **The ICU type as one channel with the codes 1–4.** Rejected: an order the wards do not have.
- **One-hot gender.** Rejected: a z-scored binary channel already says it; two presence channels
  would say it twice.
- **The admission weight as a timeless channel of its own.** Rejected: one quantity under two
  entries, learnt twice.
- **Dropping the admission weight.** Rejected: for 3,674 stays it is the weight, and for most of
  them the only one.
- **Cleaning impossible values in the reader.** Rejected above.
- **Set C as a third subset.** Rejected: a frozen test set inside a pretraining corpus.

## Revisit when

- The classification task arrives — in-hospital death from the outcome files: a per-unit label
  port beside `UnitLifetimes`, and set C as the frozen side the task names.
- A second reading of the same files is wanted, cleaned values or another encoding of the
  descriptors: a second known corpus, not a parameter.
- The mixture's normalisation is fitted across corpora: the tails here and in SMD are the case
  against it.

## 2026-09-19 — the held-out side is the challenge's own set B

The publication holds out set B whole, and states it as a part rather than as units: a third
split policy beside the drawn and the named one, `SubsetSplit`, holds out every unit read from
one part of a corpus, and the command line asks for it with `--hold-out-subset set-b`.

Why a part and not the two policies already there. The challenge divided these stays itself —
set A to train on, set B to validate against, set C to score — and a publication that drew its
own fifth would report numbers no reader could hold beside the ones published under the
challenge, while blurring the line the downstream task's validation side is drawn on. Naming the
units states the same division, but four thousand keys on a command line state it less legibly
than the name the challenge gave them, and the manifest records both sides in full either way.

Holding out a part asks the same of every corpus: that a unit's key says which part it was read
from. Four of the five readers already keyed their units that way; the server corpus keyed a
machine by its file name alone, which names its group in another shape (`machine-1-1`), and a
part held out of it would have matched nothing. Its machines are now keyed within their group
(`1/machine-1-1`), as the key of every other corpus that arrives in parts is. A seeded draw
ranks units by their keys, so a republication of that corpus draws a different fifth than the
versions published before today; the published versions themselves, and every number measured on
them, are untouched — they are artifacts, not derivations.

What the record raised and this settles: every one of the 44 channels is measured in both sets,
so holding out set B leaves no channel without statistics. The rarest in set A are `Cholesterol`
(315 measurements) and `TroponinI` (435), in set B `TroponinI` (376) and `Cholesterol` (356);
reading both sets through the reader to compare the two sides takes 38 s on the development
machine. Among the four stays of the sample it is still false — the one stay of set B measures
`pH`, which none of the three stays of set A does — so the sample keeps naming its held-out
stays, and the refusal that follows from holding out its set B is asserted as a test.

## Sources

- Silva, I., Moody, G., Scott, D. J., Celi, L. A. and Mark, R. G. (2012). Predicting In-Hospital
  Mortality of ICU Patients: The PhysioNet/Computing in Cardiology Challenge 2012. Computing in
  Cardiology 39, 245–248.
- Goldberger, A. L. et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
  Research Resource for Complex Physiologic Signals. Circulation 101(23), e215–e220.
