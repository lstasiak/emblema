# A telemetry window with history, joined from the published hours

A pretext that hides a channel's next hour reads the 48 hours before it, while the satellite
tasks are published as one-hour windows. This note checks that a window joined from the hourly
publication is the window cut whole, and how fast a host builds either.

## 2026-11-07 — the joined window against the window cut whole, on one month of each mission

**Question.** A 48-hour window can be built three ways: cut whole from the readings, built from
the readings of 48 hourly windows concatenated, or joined from the cells each hourly window is
laid into once. Are these one window? How many a second does the host build each way?

**Conditions.**

| | |
| --- | --- |
| Code | the commit that adds this note; `scripts/telemetry_history_window_check.py`, `JoinedCells` |
| Machine | M1 Pro, CPU, float64 |
| Data | ESA-AD as the publication reads it (`EsaAdRawCorpusReader`, `SlidingWindowTokeniser`), one month in the middle of each mission's span, normalised over that month |
| Windows | 48 hours at a step of 48, cut whole; the same month at 1 hour and a step of 1 |
| Cells | 48 bins over the window cut whole or concatenated; 6 bins an hour, joined into 48 |
| Throughput | 8 windows laid from their readings; 320 windows joined from hours laid once, an anchor each hour |
| Command | `telemetry_history_window_check.py --mission ESA-Mission2`, then `ESA-Mission1` |

**Results.**

| | Mission 2, 2001-10 | Mission 1, 2007-01 |
| --- | --- | --- |
| Windows, readings | 15, 1,900,404 | 15, 2,330,061 |
| Concatenated against cut whole: channels, values | equal | equal |
| Concatenated against cut whole: worst time, gap, cell | 0, 0, 0 | 0, 0, 0 |
| Joined cells against the whole window's, worst | 1.1 × 10⁻¹⁵ | 1.8 × 10⁻¹⁵ |
| Windows a second, laid from readings | 56 | 40 |
| Windows a second, joined from cells | 6,727 | 2,812 |
| One month's hours as cells | 3.5 MB | 15.4 MB |

**Conclusions.**
1. A 48-hour window concatenated from the hourly publication is the window the publication cuts
   whole, to the bit. The joined cells match within the arithmetic's rounding.
2. Joining cells laid once is 70 to 120 times faster than laying each window from its readings.
   A Mac step at batch 8 takes 0.73 s, or 11 windows a second (`telemetry-pretext-input.md`).
3. A pretraining over 48-hour windows can read the hourly publication the tasks read, joined as
   the input measurement joined its history.

**Limitations.**
- One month a mission, both from the middle of the span. Hours off the whole hour, as the second
  mission's first month starts 16 s off, are keyed to a ten-thousandth of an hour here, and the
  pretraining reader must key them as the context report does.
- Throughput is on this host's CPU, one process, one timing each; an earlier run of the same
  command read 41 and 58 laid, 2,883 and 7,278 joined. A GPU's step time at batch 32 is not
  measured.
