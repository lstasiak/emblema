# The positive control has the structure it claims

Purpose: settle, before any expensive run leans on it, that the synthetic corpora really do carry
the shared latent structure they were built to carry, and that their uncoupled twins carry none.
The control's whole value is an argument by elimination — if transfer fails on a corpus whose
structure was put there on purpose, the fault is in the implementation — and that argument is void
if the structure was never there. This note is the evidence, and the thresholds behind it are
asserted in `tests/scripts/test_synthetic_control_report.py`, so a regression fails a build rather
than a reading.

The statistic is a permutation test, not a raw share of variance. Every channel of every unit is
fitted by least squares on its own unit's latent factors, and again on another unit's. The second
fit is far from zero: every unit of a corpus shares the frequencies of the process, so a sinusoidal
basis explains part of any channel, coupled or not. What only shared structure produces is the
**excess** of the first fit over the second — a channel following its own realisation specifically.
A raw share would not separate the populations: an uncoupled channel of a short unit reaches 0.6
simply because five parameters fit any smooth curve.

Method, on any machine with the repository:

```sh
uv sync --all-extras
uv run pytest tests/catalog/adapters/synthetic tests/scripts/test_synthetic_control_report.py
uv run scripts/synthetic_control_report.py --units 12
```

Nothing is downloaded and nothing is stored: the corpora are generated from their specification, so
this runs in continuous integration and on any machine identically. What a second machine adds here
is the checksum agreement — the corpus must be the same bytes everywhere, or a version frozen over
it on one machine is a different version on another.

## 2026-09-13 — Windows AMD64

|  |  |
| --- | --- |
| Machine | Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Python | 3.14.5 |
| numpy | 2.5.3 |
| Process | 4 factors of 3 harmonics, seed 101 |
| Periods drawn | 29.7 to 294.8 steps |
| Units fitted | 12 per corpus |

### Factor recovery

| Corpus | Coupling | Channel fits | Own factors | Another unit's | Excess | Weakest own |
| --- | --- | --- | --- | --- | --- | --- |
| control-a | 1 | 96 | 0.997 | 0.396 | +0.602 | 0.991 |
| control-b | 1 | 60 | 0.994 | 0.231 | +0.763 | 0.986 |
| null-a | 0 | 96 | 0.202 | 0.197 | +0.004 | 0.007 |
| null-b | 0 | 60 | 0.186 | 0.210 | -0.024 | 0.003 |

### Verdict

- `control-a` came out as designed: excess +0.602, at least 0.4.
- `control-b` came out as designed: excess +0.763, at least 0.4.
- `null-a` came out as designed: excess +0.004, no more than 0.1 either way.
- `null-b` came out as designed: excess -0.024, no more than 0.1 either way.
- The control carries the structure it claims, and the null carries none. A pipeline that finds
  nothing here is at fault.
- Figures: `docs/verification/figures/synthetic-control-control-a.png`,
  `synthetic-control-control-b.png`, `synthetic-control-null-a.png`,
  `synthetic-control-null-b.png` — the hidden factors of one unit above the sensors that watch
  them. In the coupled pair the sensors are visibly the factors seen partially and noisily; in the
  null pair they are signals of the same character with nothing behind them.

Read the excess column against the two beside it rather than on its own. The coupled corpora fit
their own factors to 0.997 and 0.994 — within 0.003 of each other, from different widths, cadences
and noise levels, which is what the variance-preserving mix is for: how much of a channel is shared
structure is a dial, not a side effect of how often the layout reports. Their excesses differ by
0.161 all the same, and the whole of that difference is in the baseline: a fixed frequency basis
explains 0.396 of a `control-a` channel and 0.231 of a `control-b` one, because how much of a
sinusoid a smooth curve can absorb depends on how long the unit is and how often it reports. The
excess is a margin over a floor that moves, not a measurement of signal strength.

That floor, 0.20 to 0.40 across the four corpora, is also roughly the part of the structure carried
by the process rather than by a realisation — the part a model could in principle learn from one
layout and reuse on another. And the weakest coupled channel at 0.986 says no channel of either
layout is a passenger: every one of them is a genuine view of the factors, so a failure to transfer
cannot be blamed on a sensor that watches nothing.

The null pair reaches this table as a matched twin of the coupled one, which it was not in the
first implementation: the draws were addressed by a unit's key, and a key carries the layout's
name, so renaming a layout re-rolled every unit of it. Unit 0 of `control-a` spanned 347 steps with
2714 observations, unit 0 of `null-a` 595 steps with 4685 — nothing matched except the
specification. Separately, the unit's gain scaled the shared signal alone, so switching the
coupling off also removed the gain and left the null the fainter corpus as well as the unstructured
one. Draws are now addressed by a unit's position and the gain multiplies the mixture; pooled
variance within each pair is -4.1% and +2.0%, and over twenty layout seeds the gap straddles zero
(-1.0% mean, 2.8% sd), so what remains is the band each seed happened to draw rather than the dial.

## Cross-machine agreement — measured, on both architectures

The reader emits values at six decimals and instants as exact multiples of a step so that a second
machine agrees. Three checksums are pinned in
`tests/catalog/adapters/synthetic/test_synthetic_corpus_reader.py`, one per code path the generator
has; they were recorded on arm64 (macOS, Apple silicon) and the ordinary suite passes on them here
on Windows AMD64, so a corpus version frozen on one machine is the same version on the other. The
suite either agrees or fails, so nothing has to be compared by hand.

The agreement goes further than the bytes: the table above was produced independently on both
machines and every figure in it matches to the three decimals it is printed at. That covers the
least-squares fits and the transcendental functions under them, not only the reader's rounding.

## What this note does not establish

- **That the model finds the structure.** The certificate is a linear fit with the answer key in
  hand. Whether a self-supervised objective recovers the same structure without it is the next
  measurement, and the control is what makes its failure interpretable.
- **That transfer works.** The transfer leg — pretrain on one layout, adapt to the other, and
  confirm that the same procedure gains nothing on the null pair — needs a training loop and the
  transfer modes, and lands with them.
