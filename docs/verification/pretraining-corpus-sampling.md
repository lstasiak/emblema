# How the pretraining corpora are sampled

## 2026-10-22 — M1 Pro, CPU: the sampling of every pretraining corpus, and each regime's share of a pretraining

**Question.** The intensive-care tasks read each sensor at its own irregular times, as does the
telemetry the encoder is meant for. How much of what pretraining reads is sampled that way? Each
reader declares its corpus regular or irregular. This section measures the sampling itself and
weighs each corpus by its share of a pretraining. It is descriptive: no rule was declared, and
nothing here is read against a threshold.

**Conditions.**

- *Commit* `91429482`. `scripts/regime_coverage_report.py`, CPU, on this machine.
- *Sampling* (`measure`). Every corpus the mixture of four and the curve over the scale of
  pretraining read: 31 corpora. Each is read through its reader and windowed as published
  (C-MAPSS per operating condition, 50 cycles; SKAB 100 s; the satellite telemetry 1 h; the stays
  of 2012 one window of 48 h; the stays of 2019 24 h with a stride of 12 h; Tennessee Eastman 50
  samples; the 25 datasets of UTSD-12G 144 to 1,024 samples). Per corpus, up to 100 units are
  drawn by seed 1, all units where there are fewer, and up to 20 windows per unit by the same
  seed. Measures come from the instants of the observations, not their values.
- *Exposure* (`exposure`). Each corpus's published manifest gives its windows, the share of its
  units on the training side and its observed training values. The experiment file gives its
  passes and its fraction of units. A step is taken over windows of one corpus, so a corpus's
  share of the windows is its share of the steps.
- *Files.* `data/report/l32a/coverage/` (`units.csv`, `windows.csv`, `corpora.csv`,
  `exposure.csv`).

**What a unit and a window hold.** Median over units or windows, 10th and 90th percentiles in
brackets.

| Corpus (declared regime) | Co-observed | Gap variation | Rate spread | Readings per channel in a window | Channels read once in a window |
|---|---|---|---|---|---|
| PhysioNet 2012 stays (irregular) | 0.22 [0.18; 0.34] | 0.46 [0.29; 0.66] | 1.38 [1.06; 1.89] | 4 [2; 10] | 0.07 [0.00; 0.24] |
| PhysioNet 2019 stays (irregular) | 0.32 [0.27; 0.38] | 0.33 [0.00; 0.64] | 1.15 [0.60; 1.40] | 2 [1; 4] | 0.36 [0.04; 0.65] |
| Satellite telemetry (irregular) | 1.00 [1.00; 1.00] | 0.08 [0.08; 0.29] | 0.00 | 120 [120; 120] | 0.00 |
| C-MAPSS FD001, FD003 (regular) | 1.00 | 0.00 | 0.00 | 50 | 0.00 |
| C-MAPSS FD002, FD004 (regular) | 0.17 | 0.85 to 0.87 | 0.30 to 0.34 | 8 | 0.00 |
| SKAB (regular) | 1.00 [1.00; 1.00] | 0.21 [0.20; 1.62] | 0.00 | 95 [94; 96] | 0.00 |
| Tennessee Eastman (regular) | 1.00 | 0.00 | 0.00 | 50 | 0.00 |
| UTSD-12G, 25 datasets (regular) | 1.00 in every dataset | 0.00 in every dataset | 0.00 or one channel | 144 to 1,024 | 0.00 |

*Co-observed* is the mean, over a unit's distinct instants, of the share of its channels read at
that instant. It is one on a grid, and one over the number of channels where no two channels
share an instant. *Gap variation* is the median over channels of the coefficient of variation of
the spacing between readings, zero on a grid. *Rate spread* is the decimal logarithm of the ratio
between the slowest and the fastest channel's median spacing. A stay of 2012 measures 73 % of
the channels its set measures, and a stay of 2019 71 %. Every other corpus measures all of them.

**Each regime's share of a pretraining.** The regime is as each reader declares it. *Stays* is
the share of the two sets of stays, the only corpora sampled like the intensive-care tasks
(above).

| Pretraining | Irregular, windows | Irregular, values | Stays, windows | Stays, values |
|---|---:|---:|---:|---:|
| Mixture of four (`backbone-mixed4-m`) | 73.1 % | 94.7 % | 4.5 % | 3.1 % |
| Mixture, stays four times as often | 76.3 % | 95.1 % | 15.8 % | 11.4 % |
| Intensive-care corpora alone | 100 % | 100 % | 100 % | 100 % |
| Scale curve, about 1.5·10⁸ values | 34.4 % | 34.6 % | 4.0 % | 2.2 % |
| Scale curve, about 3·10⁸ values | 20.8 % | 17.9 % | 2.4 % | 1.1 % |
| Scale curve, about 10⁹ values (with 2019) | 21.4 % | 8.0 % | 16.4 % | 3.2 % |

In the mixture of four the satellite telemetry takes 68.6 % of the windows and 91.6 % of the
values, C-MAPSS 22.8 % and 4.8 %, SKAB 4.2 % and 0.6 %. These shares match what the experiment
file states (about 69, 23, 4 and 4 % of the steps).

**Conclusions.**

1. Only the two sets of stays are sampled the way the intensive-care tasks are. At one instant a
   stay reads 22 % (2012) or 32 % (2019) of its channels. Its channels' rates lie 14 to 24 times
   apart. A window holds 2 to 4 readings of a channel, and a third of its channels once in 2019.
2. The satellite telemetry is declared irregular, but it is read as a grid. Every channel is read
   at every instant, the spacing varies by 8 %, and a window holds 120 readings of every channel.
   By declaration the mixture of four is 73 % irregular. Sampled like the intensive-care tasks,
   it is 4.5 % of its windows and 3.1 % of its values.
3. Everything else the encoder pretrains on, C-MAPSS read per condition aside (4), is a dense
   grid of 50 to 1,024 readings per channel per window, where a hidden value lies between
   readings of the same channel. Along the scale curve the stays' share of the values falls from
   2.2 % to 1.1 %, and at 10⁹ it is 3.2 %.
4. C-MAPSS read per operating condition looks asynchronous to the encoder on FD002 and FD004
   (co-observed 0.17), because each sensor carries six names that alternate with the condition.
   That comes from the reading, not from the sensors.
5. The readings that weighed the regime are consistent with this, but none of them tests it. The
   intensive-care corpora alone gained 0.015 at 500 stays (`pretext-variants.md`), the stays four
   times as often 0.012 (`layer-readout.md`), and about 10⁹ values lost to the mixture
   (`verdict-statistics.md`). No reading compared mixtures of different regimes at one size.

**Limitations.**

- Up to 100 units per corpus by one seed. A unit's share of channels is counted against the
  channels the sampled units of its part measure, not the reader's schema.
- Only instants are measured. How well a hidden value can be answered from its neighbours is
  not measured here (`pretext-variants.md` reads that for the masks).
- The satellite telemetry is measured as its reader keeps it: the earliest reading of each
  channel in each 30 s bin. The raw telemetry is not measured.
- A fraction of units is applied to windows and values as to units. That holds on average over
  the units a seed draws.
