# A satellite read as a new system: forecasts from its first days

The product's third claim, since 2026-11-19, is the onboarding of a system with no history and no
labels: from its first days, a model forecasts its normal behaviour better than the classical
answers that can be given then, and no worse than a published zero-shot forecaster. This note
reads the two ESA missions as if each had just been switched on, before any model of ours is
built for it, to set the bar such a model must beat and to see whether another satellite's data
helps the published forecaster.

## 2026-11-20 — declared before the run: the published forecaster against the classical answers in a satellite's first month, and what another satellite's series adds

**Question.** In the first 30 days of a satellite's data, does Chronos-2 zero-shot forecast each
channel's next hour better than the best classical answer that could be given then, on both
missions? On the second mission, does handing it the other satellite's most similar recent series
of the same kind of channel, with the hour that followed it, improve its forecast?

**Conditions.**

| | |
|---|---|
| Commit | the one that adds this section; the instrument at `55d91cf5` |
| Machine | Mac (M1 Pro): Chronos-2 on MPS, the local forecasters on CPU; validation numbers |
| Data | Raw ESA telemetry, robust score held to 100 (ADR-0080): Mission1 from the publication of 2026-11-12 (`490c1547…`, 48 channels), Mission2 from the counterparts' publication of 2026-11-19 (`f5f675bb…`, 33 channels: the 11 lightweight ones and the 22 targets whose subsystem and physical unit a read channel of Mission1 shares); each channel's mean per hour (`telemetry_hourly_means.py`). No frozen test month lies in either mission's first 13 months. |
| Days and cells | Days N = 2, 7, 14, 30, 60, 90, 180, 365, from each mission's first hour; day N's hours run to the next day (the last for 30 days), all of them or 300 drawn by seed 1 + N. A cell is a channel that read in the hour and in at least 24 of the 47 before (`cold_start_cells.py`). |
| Classical answers | The last hour, the median of the 47 hours, the hour one period back; AutoETS, AutoARIMA, AutoTheta and seasonal naive on each cell's 47 hours, as on 2026-11-19; a ridge per channel over its 47 hours and one also reading the other channels' latest hour, fitted on the hours before day N and kept until the next (penalty by two-fold cross-fitting over 10⁻⁴ to 10⁶, inputs clipped to the fitted range), answering from day 7 and day 14. An answer joins the bar only where it answers every cell. |
| Published forecaster | Chronos-2 zero-shot (`amazon/chronos-2`, `chronos-forecasting` 2.3.2): `chronos2_47`, the 47 hours, a window's channels handed together, as on 2026-11-19. Beside: each channel alone (`_each`); every hour so far, up to the 8,192 the model reads (`_own`), more than the ridges see (`chronos2_ladder.py`). |
| The other satellite's series | For each channel with a counterpart (Mission2: 22, Mission1: 41), `chronos2_47` plus one series of the other mission's fitting side as a covariate over the same 47 hours, its next hour handed as known: a random channel of another kind (`unrelated`), a random one of the same kind (`kind`), the same kind's 47 hours nearest the target's by z-normalised Euclidean distance (`analog`). Beside, the oracle: the target's own 47 hours and true next hour by that route, the most it can carry. |
| Measure | As on 2026-11-19: each window's mean squared error of the hour's mean; the geometric mean over windows of one answer's over the other's; against the bar, the answer led least in every draw. 10,000 draws of whole days (a day's hours share their context), seed 1, 95 % interval. Beside: the median over cells of the cell's ratio. |
| Reading | `cold_start_reading.py`: each day descriptively, and days 2, 7 and 14 pooled, which are the first 30 days; `data/report/l61/run.sh` runs the whole. |

**Rules.**

- *Verdict A, the first month.* On each mission, `chronos2_47` over the best classical answer on
  the first 30 days pooled (the ridges, silent on day 2, are not in that bar); levels as on
  2026-11-19, least gain 0.80. The claim of a lead from a satellite's first days needs at least
  the least gain on both missions, so no correction applies.
- *Verdict B, the other satellite.* On Mission2, `chronos2_analog` over `chronos2_47` on the same
  days, the same levels. On Mission1 the oracle shows the route carries nearly nothing (below), so
  it is read there descriptively.
- *Everything else is descriptive.*

| Verdict A | Verdict B | What follows |
|---|---|---|
| At least the least gain on both | any | the published forecaster is the cold-start bar a domain model must beat from day 2 |
| Less on either | any | in a satellite's first month the local classical answers are the bar; a domain model has room from day 2 |
| any | At least the least gain | the nearest series of another system is the onboarding layer's first gain; the multi-system corpus is built as a library |
| any | Real and small, noise or harm | another satellite's series adds nothing Chronos-2 uses on ESA; the library is read next on the turbine fleet, systems of one design side by side, before the corpus builds on it |

**Predictions**, from the trials' intervals (below), the edge nearer one for a lead:

| | Predicted | Source |
|---|---|---|
| Verdict A, Mission1 | 1.0 to 1.4: noise or harm | days 5 and 100, 1.18 [0.98; 1.42]; days 400 to 430, 1.11 [1.06; 1.16] |
| Verdict A, Mission2 | 0.85 to 1.5: noise | 1.13 [0.84; 1.49]; 0.97 [0.90; 1.04] |
| The claim of a lead from the first days | not met | the two rows above |
| Verdict B, Mission2 | 0.8 to 1.45: noise or harm | 1.04 [0.76; 1.45]; 1.16 [1.07; 1.26] |
| Mission2's oracle over the 47 hours | 0.03 to 0.07: the route carries the hour | 0.045 [0.029; 0.073]; 0.050 [0.044; 0.056] |

**Power.** On every hour of days 400 to 430, 720 windows over 30 days, the intervals over
whole days span about ±5 % for Mission1 and ±7 % for Mission2 on the log scale against the
classical bar, and ±8 % for the nearest series on Mission2. The first 30 days hold about 590
windows over 28 days, so their intervals are of that size: an upper edge at 0.80 needs a point
at or under about 0.75, a true ratio of one reads noise, and a true 1.10 on Mission1 reads harm.

**Trial, before this declaration**, on days outside the grid. `data/report/l61/trial.sh`: days 5
and 100, 20 hours each per mission, every script and the reading whole, each branch (Chronos-2
17 to 30 s a day, the local forecasters 56 to 63 s). `data/report/l61/trial-block.sh`: every hour
of days 400 to 430, 720 per mission, for the intervals (Chronos-2 1,673 s and 2,389 s, nearly all
of it `_own`; the local forecasters 607 s and 432 s). Changed after it:

- The rungs' base was the own history up to 2,048 hours with the series over its last 47 alone,
  which Chronos-2 set aside; the base is now the 47 hours, which the series covers whole.
- The cap of 2,048 hours was taken for the model's default context; the default is 8,192.
- The search floored each 47 hours' deviation at 0.05, flattening 55 % of Mission1's queries; it
  now z-normalises as subsequence search does.
- Each day read the 30 days after it, so early days overlapped; they now run to the next day.
- The draws resampled windows; consecutive hours share their context, so they resample days.
- An independent review found no look-ahead, by perturbing the hours from the forecast hour on;
  on its findings the ridges' grid was widened, the window's ridge reads the other channels'
  latest hour rather than a fitted median, and missing files and empty libraries are refused.

The trials, the measure above (Chronos-2 over each, 95 % interval over whole days):

| | Mission1, days 5 and 100 | Mission1, days 400 to 430 | Mission2, days 5 and 100 | Mission2, days 400 to 430 |
|---|---|---|---|---|
| `chronos2_47` over the best classical answer | 1.18 [0.98; 1.42], AutoETS | 1.11 [1.06; 1.16], AutoETS | 1.13 [0.84; 1.49], AutoETS | 0.97 [0.90; 1.04], ridge |
| `unrelated` over `chronos2_47` | 1.09 [0.98; 1.20] | 1.10 [1.07; 1.12] | 1.06 [0.93; 1.21] | 1.06 [1.01; 1.12] |
| `kind` over `chronos2_47` | 1.03 [0.93; 1.13] | 1.06 [1.04; 1.08] | 1.01 [0.84; 1.18] | 1.07 [1.02; 1.13] |
| `analog` over `chronos2_47` | 1.02 [0.90; 1.14] | 1.04 [1.02; 1.06] | 1.04 [0.76; 1.45] | 1.16 [1.07; 1.26] |
| the oracle over `chronos2_47` | 1.05 [0.99; 1.11] | 0.98 [0.97; 1.00] | 0.045 [0.029; 0.073] | 0.050 [0.044; 0.056] |
| `_own` over `chronos2_47` | 0.99 [0.81; 1.19] | 1.37 [1.31; 1.44] | 0.91 [0.67; 1.24] | 0.98 [0.92; 1.05] |

On Mission1 the route carries nearly nothing even with the hour itself: its channels move about
a hundredth of a unit an hour in their first year, and Chronos-2 is already that close; handed
alone, a channel with its own next hour reads 0.83 to 0.86 there, 0.03 to 0.11 on Mission2.

**Limitations declared.** Two missions, each verdict about one satellite. Each mission's time is
scaled by its own undisclosed factor, so the other satellite's series is matched by shape, not
time. The normalised units were fitted on later months; they weigh the channels in a window's
mean and set the bound of 100 for every answer alike. One published forecaster; the hour's mean,
one hour ahead. A reading held at the bound can throw a ridge; the bar takes the answer led least.
