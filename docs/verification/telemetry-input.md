# How the satellite telemetry's values are prepared for the network

## 2026-11-12 — measured: the fitting side's scale is set by the events, and the channel-local detector reads the scale only through its readings count

**Question.** Does the standard score the raw ESA telemetry was published under let the encoder
read its channels, as the robust score now lets it read the intensive-care stays
(`clinical-input.md`), and does the product's detector (`channel_local`, ADR-0075), which reads
each channel against its own recent hours, depend on how the publication scales a channel?

**Conditions.**

| | |
| --- | --- |
| Code | the working tree over `7f86eb7b`; the audit scripts, local and untracked, under `data/report/l49/audit/` (`esa_scale.py`, `count_floor_check.py`), the Mac's CPU |
| Data | the raw ESA telemetry as published `durable/sha256/f1d0b0f2…` (59 channels, 2026-10-27): the fitting side of each mission, Mission1 2000-01 to 2006-12 (84 months), Mission2 2000-01 to 2001-09 (21 months); the validation months unread. The count and standard deviation of every channel equal the manifest's to 10⁻⁶ |
| Events | the benchmark's labelled intervals per channel on the fitting side, anomalies and rare events, as the task scores them; communication gaps left out |
| Detector | the stored summaries, context statistics and departures of the channel-local reading on Mission1 (`telemetry-channel-local.md`, 2026-11-06; local run directory, refit section: 61,324 fitting windows, 41,088 scored); every departure recomputed to the bit before any map was applied; labels untouched |

**The scale of a channel on the fitting side** (the channels the events move; the other 29
channels of Mission1, channels 12 to 40, have SD within 0.7 to 0.9 of the robust deviation and no
reading beyond 5 SD):

| Channel | SD / robust deviation | p1–p99 width in SD | Events on the fitting side (anomalies + rare) | Share of readings in events | SD / SD without the events | Share of the sum of squares in events |
| --- | --- | --- | --- | --- | --- | --- |
| M1 channel_44 | 3.4 | 1.03 | 48 (22 + 26) | 1.1 % | 4.01 | 93.8 % |
| M1 channel_43 | 2.7 | 1.25 | 48 (22 + 26) | 1.1 % | 3.19 | 90.3 % |
| M1 channel_74 | 1.9 | 2.00 | 16 (11 + 5) | 0.4 % | 1.99 | 74.8 % |
| M1 channels 41, 42, 45, 46 | 1.2–1.3 | 2.1–2.8 | 48–49 | 1.1 % | 1.50–1.84 | 56–71 % |
| M1 channels 73, 75, 76 | 1.1–1.8 | 2.6–2.8 | 14–17 | 0.3–0.4 % | 1.34–1.54 | 44–58 % |
| M1 channels 47–52 | 0.9–1.0 | 3.2–3.4 | 33–35 | 2.8–4.9 % | 1.12–1.20 | 22–34 % |
| M1 channels 70–72 | 0.9–1.2 | 3.5–3.7 | 8–10 | 0.3–0.4 % | 1.02–1.03 | 4–6 % |
| M2 channel_18 | **534** | 0.02 | 260 (10 + 250) | 2.4 % | **510** | 99.7 % |
| M2 channel_19 | **22** | 0.32 | 251 (5 + 246) | 1.3 % | 18.2 | 99.7 % |
| M2 channel_20 | **9.0** | 0.68 | 261 (11 + 250) | 2.2 % | 8.4 | 98.6 % |
| M2 channels 21–28 | 1.1–2.6 | 3.7–7.4 | 159–186 | 3.9–4.2 % | 1.02–1.29 | 7–42 % |

A Gaussian channel's bulk spans 4.65 SD between its first and 99th percentile. The farthest reading
of every channel the events move lies inside a labelled event, and the five channels whose
farthest reading lies outside one have it within 3 SD; no reading of the fitting side is a
source error. Mission1's channels take 48 to 403 distinct values each (quantised telemetry), Mission2's
45,000 to 370,000; every channel has a positive robust deviation.

**Where a bound on the robust score bites** (|x − median| over 1.4826 MAD; normal = outside the
events; the normal readings beyond 50 are counts, of 50 million):

| Channel | Normal readings beyond 10 | beyond 50 | Farthest normal | Event readings beyond 10 | beyond 100 | beyond 1,000 | Farthest event |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1, every channel | 0.000 % | 1 reading (channel_74) | ≤ 5 (channel_74: 208) | 0–27 % | 0–7 % | 0 | 358 |
| M2 channel_18 | 0.013 % | 1 reading | 51 | 54 % | 31 % | 20 % | 9,814 |
| M2 channel_19 | 0.007 % | 3 readings | 124 | 19 % | 6 % | 3 % | 1,397 |
| M2 channel_20 | 0.003 % | 1 reading | 520 | 29 % | 6 % | 0 | 1,840 |
| M2 channel_22 | 0.902 % | 0 | 21 | 9 % | 0 | 0 | 23 |
| M2 other channels | ≤ 0.013 % | 0 | ≤ 12 | ≤ 7 % | 0 | 0 | ≤ 26 |

**What the channel-local detector reads of the scale.** Each channel's hour is five summaries
(the mean, the least and the largest value, the last reading, and the log of the readings
counted) read against the median of its recent hours over their scaled MAD, floored at a tenth of
the summary's scaled MAD over the fitting side. Two maps of the values were applied per channel,
the exact map from the standard to the robust score with the fitting side's statistics (a = SD /
robust deviation, median 0.87 over the 59 channels), and a stress map (a log-uniform in
[0.05, 50], b in [−5, 5]). The verdict on the first mission reads the `bins-24h` mean arm and the
`bins-72h` max arm (`telemetry-channel-local.md`, 2026-11-11):

| Arm | Value-summary departures under either map | Scored channel-hours whose departure the count summary carries | Top-1 % scored windows the count carries, max / mean | Top-1 % overlap under the robust map, max / mean | Under the stress map |
| --- | --- | --- | --- | --- | --- |
| hour-24h | equal to 2 × 10⁻⁸ relative (10⁻¹⁰ absolute) | 0.3 % | 55 % / 68 % | 0.998 / 0.995 | 0.448 / 0.968 |
| hour-72h | equal | 0.7 % | 30 % / 59 % | 0.825 / 0.998 | 0.696 / 0.891 |
| bins-24h (the verdict's mean arm) | equal | 36 % | 67 % / 71 % | 0.966 / **0.995** | 0.905 / 0.978 |
| bins-72h (the verdict's max arm) | equal | 35 % | 50 % / 64 % | **0.981** / 0.990 | 0.844 / 0.793 |

The count summary has no spread on the fitting side of any channel (the sampling is regular), so
its floor is a tenth of the median spread of the value summaries over the channels, 0.114 in the
standard score's units and 0.096 in the robust score's: the count's departures are measured in a
unit the values' scale sets. Spearman correlations of the scores between the two scales are 0.997
to 1.000.

**Conclusions.**

- On Mission1 the standard score is not heavy-tailed, but the labelled events set it: in ten of
  48 channels 0.3 to 1.1 % of the readings hold 44 to 94 % of the variance. On Mission2 the rare
  events set the scale of channels 18 to 20 entirely; under the standard score the encoder reads
  them as constant.
- The robust score held to 100 clips five normal readings of 50 million (one on Mission1's
  channel 74, four on Mission2's channels 19 and 20) and flattens 0 to 7 % of Mission1's event
  readings and up to 31 % of Mission2's, all beyond any alarm; a bound of 10 would flatten a
  quarter of the event readings of channels 41 to 46.
- The channel-local detector is invariant to the publication's scale in its value summaries, to
  rounding. It is not invariant in its readings-count summary, whose floor is in the values'
  units: that summary carries 50 to 71 % of the scored side's top-1 % windows, and the exact
  change of score moves 0.5 to 2 % of the verdict arms' top-1 % windows. The floor is a flaw of
  the instrument, queued as its own change; whether the count's alarms find events is unread.
- Decision: the telemetry is published under the robust score held to 100 (ADR-0080); the
  intensive-care corpora keep ADR-0079.

**Limitations.** The audit reads the fitting side with the benchmark's labels, which the
detectors taught a few events read too; rare events are counted as events, as the task scores
them. The invariance is shown on the recorded run's arrays under the exact map, not on a
republished block; the bins' departures were recomputed from the published cells. The validation
months, whose readings reach 26,000 SD (`composition-screen.md`, 2026-10-30), were not read. Tier
M, Mac CPU, no label of a validation month touched.
