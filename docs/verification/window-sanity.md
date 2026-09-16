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

## 2026-09-16 — Windows AMD64, SKAB and SMD

The first two downloaded corpora after C-MAPSS, and the two that bracket it on channel count: 8
against 21 against 38. Nothing behind the readers changed to accommodate either, which is the
claim this section exists to support — the same tokenisation scheme, the same window arithmetic
and the same archive, driven from the same command line.

```sh
uv run scripts/fetch_corpora.py skab smd
uv run scripts/window_sanity_report.py --corpus skab
uv run scripts/window_sanity_report.py --corpus smd
```

|  |  |
| --- | --- |
| Machine | Windows-11-10.0.26200-SP0, Intel64 Family 6 Model 140 Stepping 1, GenuineIntel |
| Python | 3.14.5 |
| matplotlib | 3.11.2 |

| Corpus | Window | Units | Unit drawn | Observations per window | Largest value error | Largest time error |
| --- | --- | --- | --- | --- | --- | --- |
| skab | length 100 s, stride 10 s | 35 | anomaly-free/anomaly-free | 752, 752, 768 | 1.42e-14 | 7.11e-15 |
| smd | length 100 min, stride 20 min | 28 | machine-2-6 | 3,800 each | 1.11e-16 | 7.11e-15 |

Both round trips sit eight orders of magnitude inside the 1e-06 the tests hold to, and every
reconstruction marker sits on the raw curve in all six figures, at each channel's own scale — read
and confirmed on 2026-09-16.

### SKAB: a nominal cadence is not a fixed one

The testbed writes one row a second, and the files do not keep to it. The unit drawn holds 9,405
rows across an extent of 9,961 seconds. Over the corpus 44,222 gaps are of one second, 2,479 of
two, and 70 longer ones run from three seconds to 247. The reader therefore places each
observation by the seconds elapsed since its experiment's first row rather than by its row index;
a reader that had counted rows would have put every observation after the first gap in the wrong
second, and the figures would still have looked right, because the reconstruction would have
agreed with the mistake. The three windows drawn hold
752, 752 and 768 observations for a nominal 800, which is that loss made visible.

| Channel | Values | Mean | Spread | Lowest, deviations | Highest, deviations | Beyond 8 deviations |
| --- | --- | --- | --- | --- | --- | --- |
| Accelerometer1RMS | 46,806 | 0.1293 | 0.1267 | -0.90 | 4.68 | 0 |
| Accelerometer2RMS | 46,806 | 0.1628 | 0.1447 | -1.02 | 4.41 | 0 |
| Current | 46,806 | 1.653 | 0.7924 | -1.90 | 2.10 | 0 |
| Pressure | 46,806 | 0.07935 | 0.2597 | -5.15 | 6.22 | 0 |
| Temperature | 46,806 | 79.34 | 9.393 | -1.52 | 1.67 | 0 |
| Thermocouple | 46,806 | 26.54 | 2.576 | -1.75 | 2.67 | 0 |
| Voltage | 46,806 | 229.6 | 10.95 | -2.63 | 2.35 | 0 |
| Volume Flow RateRMS | 46,806 | 75.53 | 45.46 | -1.65 | 1.28 | 0 |

No channel is constant and none holds a value beyond eight fitted deviations. Pressure is the
widest-tailed of the eight and takes exactly ten distinct values in the whole corpus, spaced
0.327927 bar apart: that is the resolution of the gauge, not anything about the loop.

Figures: `figures/skab-anomaly-free-anomaly-free-w1.png`, `-w2.png`, `-w3.png`.

### SMD: the first constant channel met on real data

`metric_08` holds the same value in all 708,405 minutes of the corpus. A channel with no spread
has no scale to normalise by, and the scheme falls back to a scale of one rather than dividing by
zero; until now that fallback had only property tests behind it, and this run is the first time a
real corpus has exercised it end to end. The round trip holds on that channel like any other.

Seventeen of the 38 metrics hold values beyond eight fitted deviations, up to 735 on `metric_27`.
That is not a fault in the reading: these are bounded rates that sit at or near zero for most of a
week and spike, so a spread fitted over the whole corpus is small and any spike is far outside it.
It is worth knowing before a normalised value is taken at face value, and it is an argument for
fitting statistics per corpus rather than across the mixture.

| Channel | Values | Mean | Spread | Lowest, deviations | Highest, deviations | Beyond 8 deviations |
| --- | --- | --- | --- | --- | --- | --- |
| metric_01 | 708,405 | 0.1356 | 0.1377 | -0.98 | 6.28 | 0 |
| metric_02 | 708,405 | 0.07032 | 0.1108 | -0.63 | 8.39 | 20 |
| metric_03 | 708,405 | 0.08122 | 0.1285 | -0.63 | 7.15 | 0 |
| metric_04 | 708,405 | 0.09548 | 0.1575 | -0.61 | 5.74 | 0 |
| metric_05 | 708,405 | 0.2638 | 0.4158 | -0.63 | 1.77 | 0 |
| metric_06 | 708,405 | 0.7226 | 0.2949 | -2.45 | 0.94 | 0 |
| metric_07 | 708,405 | 0.3942 | 0.3234 | -1.22 | 1.87 | 0 |
| metric_08 | 708,405 | 0 | 0 (never varied) | 0.00 | 0.00 | 0 |
| metric_09 | 708,405 | 0.02001 | 0.05146 | -0.39 | 19.05 | 2,186 |
| metric_10 | 708,405 | 0.001013 | 0.011 | -0.09 | 90.85 | 1,196 |
| metric_11 | 708,405 | 0.05069 | 0.08676 | -0.58 | 10.94 | 809 |
| metric_12 | 708,405 | 0.0608 | 0.07861 | -0.77 | 11.95 | 518 |
| metric_13 | 708,405 | 0.0173 | 0.04214 | -0.41 | 23.32 | 1,000 |
| metric_14 | 708,405 | 0.07983 | 0.1051 | -0.76 | 8.76 | 25 |
| metric_15 | 708,405 | 0.05488 | 0.07756 | -0.71 | 12.19 | 199 |
| metric_16 | 708,405 | 0.06195 | 0.08415 | -0.74 | 11.15 | 186 |
| metric_17 | 708,405 | 2.495e-05 | 0.003756 | -0.01 | 266.21 | 56 |
| metric_18 | 708,405 | 7.869e-05 | 0.004943 | -0.02 | 202.29 | 266 |
| metric_19 | 708,405 | 0.1656 | 0.1825 | -0.91 | 4.57 | 0 |
| metric_20 | 708,405 | 0.159 | 0.1749 | -0.91 | 4.81 | 0 |
| metric_21 | 708,405 | 0.1807 | 0.1842 | -0.98 | 4.45 | 0 |
| metric_22 | 708,405 | 0.1854 | 0.1857 | -1.00 | 4.39 | 0 |
| metric_23 | 708,405 | 0.2152 | 0.2747 | -0.78 | 2.86 | 0 |
| metric_24 | 708,405 | 0.3317 | 0.3113 | -1.07 | 2.15 | 0 |
| metric_25 | 708,405 | 0.1209 | 0.1514 | -0.80 | 5.81 | 0 |
| metric_26 | 708,405 | 0.3393 | 0.3203 | -1.06 | 2.06 | 0 |
| metric_27 | 708,405 | 2.47e-06 | 0.001361 | -0.00 | 734.67 | 3 |
| metric_28 | 708,405 | 0.1822 | 0.1818 | -1.00 | 4.50 | 0 |
| metric_29 | 708,405 | 1.58e-05 | 0.003382 | -0.00 | 295.67 | 25 |
| metric_30 | 708,405 | 0.1004 | 0.1392 | -0.72 | 6.46 | 0 |
| metric_31 | 708,405 | 0.1917 | 0.1869 | -1.03 | 4.32 | 0 |
| metric_32 | 708,405 | 0.1053 | 0.1488 | -0.71 | 6.01 | 0 |
| metric_33 | 708,405 | 0.03576 | 0.08363 | -0.43 | 11.53 | 158 |
| metric_34 | 708,405 | 0.05773 | 0.1075 | -0.54 | 8.77 | 11 |
| metric_35 | 708,405 | 0.2149 | 0.2045 | -1.05 | 3.84 | 0 |
| metric_36 | 708,405 | 0.2107 | 0.2068 | -1.02 | 3.82 | 0 |
| metric_37 | 708,405 | 0.01667 | 0.07108 | -0.23 | 12.27 | 775 |
| metric_38 | 708,405 | 0.009591 | 0.05937 | -0.16 | 14.80 | 3,182 |

Figures: `figures/smd-machine-2-6-w1.png`, `-w2.png`, `-w3.png`. Several metrics are flat
inside the window drawn, most of them at zero, which is the machine being idle in that respect
rather than a channel that never varies; `metric_08` is the only one of the latter kind.

## 2026-09-16 — macOS arm64 (M1 Pro), the satellite telemetry

The ESA Anomaly Dataset, read through the subsampling the corpus decision fixed: the benchmark's
lightweight channels, the first half of each mission by time, one observation per channel per
thirty-second bin with its native instant. Two things are new here and nothing behind the reader
changed for either: the channels arrive as pickled data frames rather than text, and a unit is a
calendar month of a mission rather than a machine or an engine, because two missions with channels
of their own cannot be split on (ADR-0025). The window is the one the budget prices, one hour.

```sh
uv sync --all-extras                               # the corpora extra brings pandas
uv run scripts/fetch_corpora.py esa_ad             # 11.6 GB
uv run scripts/window_sanity_report.py --corpus esa_ad
```

|  |  |
| --- | --- |
| Machine | macOS-26.6.2-arm64-arm-64bit-Mach-O, Apple M1 Pro |
| Python | 3.14.7 |
| matplotlib | 3.11.2 |
| pandas | 3.0.5 |

| Corpus | Window | Units | Unit drawn | Observations per window | Largest value error | Largest time error |
| --- | --- | --- | --- | --- | --- | --- |
| esa_ad | length 1 h, stride 1 h | 105 (84 months of Mission1, 21 of Mission2) | ESA-Mission1/2000-03 | 720 each | 0.00e+00 | 0.00e+00 |

The round trip is exact to the bit on this corpus: the values are single-precision in the source
and come back as the same doubles, and an instant placed in hours since the mission's start comes
back as the same hour. Every reconstruction marker sits on the raw curve in all six channels of the
three figures, at each channel's own scale — read and confirmed on 2026-09-16.

### The satellite: synchronous channels, wide tails, and a corpus read from pickles

The six lightweight channels of Mission1 share their instants — the publisher samples the subsystem
as one — so every hour of the month drawn holds exactly 720 observations, 120 per channel at the
thirty-second cadence the binning leaves. Mission2's eleven channels run at eighteen seconds and
the binning halves them; its months hold 1,838,869 values per channel against Mission1's
7,350,161.

Every channel but one holds values beyond eight fitted deviations, and the tails are wider than
anything the machine corpora showed: Mission1's channels sit within a hundredth of their mean for
years and drop to −89 deviations, Mission2's `channel_20` spans −194 to +73. These are the
excursions the benchmark labels as anomalies and rare events, over a baseline whose spread is tiny
— the spread is the right one for the baseline, and a normalised value far outside it is the
telemetry saying something happened. Four of Mission2's channels (`channel_25` to `channel_28`)
live at 10⁻¹¹: the scale fitted per channel is what makes them the same kind of token as a
channel at 0.8. No channel is constant.

Reading the corpus is cheap next to tokenising it: describing both missions — hashing a gigabyte of
archives and unpickling every channel — takes 5.7 s, and streaming the 64.3 million observations
38 s. The reader keeps the mission it last read, so that a month is served from memory; without
that each of the 105 months would unpickle its mission again.

| Channel | Values | Mean | Spread | Lowest, deviations | Highest, deviations | Beyond 8 deviations |
| --- | --- | --- | --- | --- | --- | --- |
| ESA-Mission1/channel_41 | 7,350,161 | 0.8116 | 0.00913 | -88.89 | 18.68 | 20,032 |
| ESA-Mission1/channel_42 | 7,350,161 | 0.7852 | 0.01043 | -75.25 | 17.35 | 19,001 |
| ESA-Mission1/channel_43 | 7,350,161 | 0.7725 | 0.01579 | -48.91 | 11.29 | 14,338 |
| ESA-Mission1/channel_44 | 7,350,161 | 0.7969 | 0.02375 | -33.56 | 7.52 | 5,649 |
| ESA-Mission1/channel_45 | 7,350,161 | 0.8138 | 0.009948 | -81.80 | 18.72 | 19,995 |
| ESA-Mission1/channel_46 | 7,350,161 | 0.769 | 0.01073 | -71.66 | 17.75 | 19,156 |
| ESA-Mission2/channel_18 | 1,838,869 | 0.4576 | 0.02969 | -15.41 | 18.23 | 6,411 |
| ESA-Mission2/channel_19 | 1,838,869 | 0.456 | 0.002313 | -48.08 | 58.29 | 1,117 |
| ESA-Mission2/channel_20 | 1,838,869 | 0.4561 | 0.0009217 | -193.71 | 73.19 | 3,290 |
| ESA-Mission2/channel_21 | 1,838,869 | 0.1762 | 0.02121 | -8.31 | 9.05 | 2,655 |
| ESA-Mission2/channel_22 | 1,838,869 | 0.8564 | 0.02166 | -8.88 | 6.46 | 504 |
| ESA-Mission2/channel_23 | 1,838,869 | 0.8754 | 0.02886 | -8.14 | 4.32 | 87 |
| ESA-Mission2/channel_24 | 1,838,869 | 0.1552 | 0.02833 | -4.96 | 6.88 | 0 |
| ESA-Mission2/channel_25 | 1,838,869 | 5.551e-11 | 1.932e-12 | -8.56 | 5.04 | 2 |
| ESA-Mission2/channel_26 | 1,838,869 | 2.008e-11 | 8.06e-13 | -10.96 | 23.44 | 18 |
| ESA-Mission2/channel_27 | 1,838,869 | 9.247e-12 | 3.164e-12 | -2.92 | 9.39 | 2 |
| ESA-Mission2/channel_28 | 1,838,869 | 5.976e-11 | 1.168e-12 | -17.81 | 11.45 | 18 |

Figures: `figures/esa_ad-ESA-Mission1-2000-03-w1.png`, `-w2.png`, `-w3.png`.

### The same road, end to end

The corpus was then published through the command line unchanged, against the local stack, in one
run of 189 s including both reads of the data and the upload:

```sh
uv run python -m emblema.entrypoints.cli.publish_corpus --corpus esa_ad --window 1 --stride 1
```

105 units, 76,682 windows, 64,323,742 tokens in a block of 1,095,655,006 bytes; 84 months fit the
scheme and 21 validate it, months of both missions on each side. The budget file priced the same
window at 76,685 windows and 64,326,491 tokens from bin-start instants laid over each mission as one
series; the reader lays hours inside months, so the fraction of an hour at the start of a mission
and at its cut is the 2,749 tokens between the two counts.

### What this note still owes

One corpus of the mixture has no reader yet and therefore no section here: the clinical stream,
which is the one irregular corpus with static descriptors on real patients. It adds a dated section
as it arrives.
