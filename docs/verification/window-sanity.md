# Windows beside their token reconstruction

Purpose: see, before any GPU hour is spent, that a token window still holds the measurements it was
cut from, and that the corpus behind it looks the way its documentation says. The arithmetic of the
round trip is a test, not a number here — the reader, the tokeniser and the scheme are held to it on
synthetic data in the port contract (`tests/catalog/ports/test_tokeniser_contract.py`), on the
miniature sample in `tests/catalog/test_round_trip_on_the_sample.py` and property by property in
`tests/catalog/domain`. What only a machine with the raw corpus can add is the picture, the residual
on real values and the shape of every channel.

Method, on any machine with the corpus unpacked under `data/raw/`:

```sh
uv sync --all-extras
uv run pytest tests/catalog                       # the assertions
uv run scripts/window_sanity_report.py            # the figures and the numbers
uv run scripts/window_sanity_report.py --subset FD001
```

Where the raw corpus is absent the script falls back to the miniature sample in `tests/data/`, and
its output says which one it used. Statistics are fitted over every unit read, because this is a
check on the data rather than a training run: no number here is a result.

The judgement the ticket asks for is made by looking at the figures: every reconstruction marker has
to sit on the raw curve, in every channel, at the raw scale of that channel.

## Runs

The whole corpus, all four subsets. The unit drawn is the longest one, which is where a time axis
that drifts would show most.

## 2026-09-12 — Windows AMD64, whole corpus

|  |  |
| --- | --- |
| Machine | Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Python | 3.14.5 |
| matplotlib | 3.11.2 |
| Corpus | cmapss (raw corpus, `data/raw/cmapss/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData`) |
| Window | length 50, stride 5 |
| Units | 709, of which 0 shorter than a window |

### Windows drawn

| Unit | Window | Observations | Largest value error | Largest time error | Figure |
| --- | --- | --- | --- | --- | --- |
| FD004/118 | [1, 51) | 1,050 | 2.84e-14 | 3.55e-15 | `docs/verification/figures/cmapss-FD004-118-w1.png` |
| FD004/118 | [246, 296) | 1,050 | 2.84e-14 | 0.00e+00 | `docs/verification/figures/cmapss-FD004-118-w2.png` |
| FD004/118 | [491, 541) | 1,050 | 2.84e-14 | 0.00e+00 | `docs/verification/figures/cmapss-FD004-118-w3.png` |

### Channels

| Channel | Values | Mean | Spread | Lowest, deviations | Highest, deviations | Beyond 8 deviations |
| --- | --- | --- | --- | --- | --- | --- |
| BPR | 160,359 | 9.055 | 0.7516 | -1.20 | 2.68 | 0 |
| NRc | 160,359 | 8089 | 80.62 | -3.02 | 2.54 | 0 |
| NRf | 160,359 | 2350 | 111.2 | -2.90 | 0.37 | 0 |
| Nc | 160,359 | 8678 | 374.7 | -1.85 | 1.51 | 0 |
| Nf | 160,359 | 2274 | 142.4 | -2.52 | 0.81 | 0 |
| Nf_dmd | 160,359 | 2274 | 142.5 | -2.52 | 0.80 | 0 |
| P15 | 160,359 | 14.42 | 6.444 | -1.36 | 1.12 | 0 |
| P2 | 160,359 | 9.895 | 4.266 | -1.40 | 1.11 | 0 |
| P30 | 160,359 | 359.7 | 174.1 | -1.28 | 1.21 | 0 |
| PCNfR_dmd | 160,359 | 98.39 | 4.656 | -2.89 | 0.35 | 0 |
| Ps30 | 160,359 | 44.21 | 3.426 | -2.39 | 1.26 | 0 |
| T2 | 160,359 | 485.8 | 30.42 | -1.34 | 1.08 | 0 |
| T24 | 160,359 | 597.4 | 42.48 | -1.46 | 1.12 | 0 |
| T30 | 160,359 | 1467 | 118.2 | -1.90 | 1.27 | 0 |
| T50 | 160,359 | 1261 | 136.3 | -1.74 | 1.32 | 0 |
| W31 | 160,359 | 25.94 | 11.69 | -1.35 | 1.19 | 0 |
| W32 | 160,359 | 15.57 | 7.015 | -1.36 | 1.20 | 0 |
| epr | 160,359 | 1.154 | 0.1421 | -1.57 | 1.17 | 0 |
| farB | 160,359 | 0.02519 | 0.004997 | -1.04 | 0.96 | 0 |
| htBleed | 160,359 | 360.7 | 31.02 | -1.89 | 1.27 | 0 |
| phi | 160,359 | 338.8 | 164.2 | -1.28 | 1.21 | 0 |

### Verdict

- The round trip holds: the largest error over 3 window(s) is 2.84e-14, against a tolerance of 1e-06.
- Channels that never varied in the data fitted: none.
- Channels holding values beyond 8 deviations: none.
- Whether the two curves agree is a judgement made by looking at the figures.

FD001 alone, the subset a downstream task is built on. Its engines run under one operating
condition, so six of the twenty-one sensors report the same value in every row of it: the six
the table below marks as never varied. They are not an error and they are not dropped — the
scheme keeps a channel that never varied in raw units rather than dividing by a vanishing
spread — but a backbone trained on FD001 alone learns six embeddings from a constant.

## 2026-09-12 — Windows AMD64, FD001 only

|  |  |
| --- | --- |
| Machine | Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Python | 3.14.5 |
| matplotlib | 3.11.2 |
| Corpus | cmapss (raw corpus, `data/raw/cmapss/6. Turbofan Engine Degradation Simulation Data Set/CMAPSSData`) |
| Window | length 50, stride 5 |
| Units | 100, of which 0 shorter than a window |

### Windows drawn

| Unit | Window | Observations | Largest value error | Largest time error | Figure |
| --- | --- | --- | --- | --- | --- |
| FD001/69 | [1, 51) | 1,050 | 0.00e+00 | 3.55e-15 | `docs/verification/figures/cmapss-FD001-69-w1.png` |
| FD001/69 | [156, 206) | 1,050 | 0.00e+00 | 0.00e+00 | `docs/verification/figures/cmapss-FD001-69-w2.png` |
| FD001/69 | [311, 361) | 1,050 | 0.00e+00 | 0.00e+00 | `docs/verification/figures/cmapss-FD001-69-w3.png` |

### Channels

| Channel | Values | Mean | Spread | Lowest, deviations | Highest, deviations | Beyond 8 deviations |
| --- | --- | --- | --- | --- | --- | --- |
| BPR | 20,631 | 8.442 | 0.0375 | -3.13 | 3.80 | 0 |
| NRc | 20,631 | 8144 | 19.08 | -2.30 | 7.86 | 0 |
| NRf | 20,631 | 2388 | 0.07192 | -3.01 | 6.45 | 0 |
| Nc | 20,631 | 9065 | 22.08 | -1.97 | 8.12 | 1 |
| Nf | 20,631 | 2388 | 0.07098 | -2.77 | 6.53 | 0 |
| Nf_dmd | 20,631 | 2388 | 0 (never varied) | 0.00 | 0.00 | 0 |
| P15 | 20,631 | 21.61 | 0.001389 | -7.06 | 0.14 | 0 |
| P2 | 20,631 | 14.62 | 0 (never varied) | 0.00 | 0.00 | 0 |
| P30 | 20,631 | 553.4 | 0.8851 | -3.97 | 3.04 | 0 |
| PCNfR_dmd | 20,631 | 100 | 0 (never varied) | 0.00 | 0.00 | 0 |
| Ps30 | 20,631 | 47.54 | 0.2671 | -2.59 | 3.70 | 0 |
| T2 | 20,631 | 518.7 | 0 (never varied) | 0.00 | 0.00 | 0 |
| T24 | 20,631 | 642.7 | 0.5 | -2.94 | 3.70 | 0 |
| T30 | 20,631 | 1591 | 6.131 | -3.18 | 4.30 | 0 |
| T50 | 20,631 | 1409 | 9 | -2.96 | 3.62 | 0 |
| W31 | 20,631 | 38.82 | 0.1807 | -3.74 | 3.40 | 0 |
| W32 | 20,631 | 23.29 | 0.1082 | -3.65 | 3.04 | 0 |
| epr | 20,631 | 1.3 | 0 (never varied) | 0.00 | 0.00 | 0 |
| farB | 20,631 | 0.03 | 0 (never varied) | 0.00 | 0.00 | 0 |
| htBleed | 20,631 | 393.2 | 1.549 | -3.36 | 4.38 | 0 |
| phi | 20,631 | 521.4 | 0.7375 | -3.69 | 2.67 | 0 |

### Verdict

- The round trip holds: the largest error over 3 window(s) is 3.55e-15, against a tolerance of 1e-06.
- Channels that never varied in the data fitted: Nf_dmd, P2, PCNfR_dmd, T2, epr, farB.
- Channels holding values beyond 8 deviations: Nc.
- Whether the two curves agree is a judgement made by looking at the figures.

## What the figures show

Every reconstruction marker sits on the raw curve in all six figures, at each channel's own
scale — read and confirmed on 2026-09-12, which is what closes the visual half of this check. The FD004 engine drawn for the whole-corpus run steps between six discrete levels in
every channel, which is what six operating conditions look like; the FD001 engine drifts
smoothly, and its six constant channels are flat lines. The largest error over six windows is
2.84e-14 on a value and 3.55e-15 on an instant, both far inside the tolerance the tests hold to.

The two values the FD001 table flags as far out — one reading of Nc at 8.12 deviations, the
lowest reading of P15 at −7.06 — are the end of an engine's life, not a fault in the reading:
a channel whose spread over a hundred healthy engines is small is bound to look extreme once
one of them degrades.

Two things this run does not cover. There is no timeless token in C-MAPSS — the operational
settings vary per cycle and are channels — so the leg that reads static features back is
exercised only by the port contract and the property tests; the first reader with unit
descriptors closes it on data. And nothing here is drawn for the other corpora of the mixture,
because they have no reader yet: each one adds a dated section to this note as it arrives.

## 2026-09-13 — Windows AMD64, the generated control corpora

The first corpora in the project with a **timeless channel**, and the first sampled irregularly and
asynchronously, so this run closes the leg the section above left open: a static feature now goes
through a window as a token of its own, comes back with it, and is held to the same tolerance as a
measurement. The script fits the statistics of static features, counts them among the channels it
diagnoses, and stops the report if one fails to come back. They need no download: both corpora are
generated from their specification, and the window comes from the command line because a generated
corpus is not part of the pretraining budget the default window is read from.

```sh
uv run scripts/window_sanity_report.py --corpus control-a --length 32 --stride 12 --windows 2
uv run scripts/window_sanity_report.py --corpus control-b --length 32 --stride 12 --windows 2
```

| Corpus | Sampling | Units | Unit drawn | Observations per window | Largest value error | Largest time error |
| --- | --- | --- | --- | --- | --- | --- |
| control-a | every step, channels together | 160 | control-a/121 | 253 and 255 | 4.44e-16 | 0.00e+00 |
| control-b | every third step, channels apart | 120 | control-b/13 | 57 and 63 | 2.22e-16 | 0.00e+00 |

The round trip holds at the resolution of the double on both, the static feature included — the
figures carry the unit's gain in their title, and the value is the one the tokens gave back. No
channel of either corpus was constant or held a value beyond eight fitted deviations, and the
fitted spreads sit at 1.00–1.05 against a design of unit variance, which is the generator agreeing
with its own specification from the outside.

Figures: `figures/control-a-control-a-121-w1.png`, `-w2.png`,
`figures/control-b-control-b-13-w1.png`, `-w2.png`.

What this adds to the mixture corpora is nothing: the real readers still owe this note a section
each. What it removes is the doubt about the timeless leg, which had no coverage on data at all.

