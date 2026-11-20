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

## 2026-11-20 — measured: in a satellite's first month the published forecaster on 47 hours trails the local classical answers on the first mission and leads them a little on the second; another satellite's series makes it worse

**Question.** As declared above: in each satellite's first 30 days, does Chronos-2 zero-shot lead
the best classical answer, and on the second mission does the other satellite's nearest series
of the same kind improve its forecast?

**Conditions.** Instrument `55d91cf5`, declaration `abc5cdf4`, `data/report/l61/run.sh` on the Mac
(M1 Pro) on 2026-11-20. The first start ran out of memory after Chronos-2 had answered Mission2's
days 2 to 90 (Docker Desktop held 13.6 GB of 32); the run resumed with Chronos-2 handed 2 windows
a call rather than 32 and the local forecasters after it. Answers differ by at most 10⁻⁵ between
2 and 32 windows a call (Mission2, day 14). Chronos-2 took 37 to 874 s a day, the local
forecasters 131 to 457 s. The first 30 days hold 588 windows per mission: 28,224 cells on
Mission1, 19,404 on Mission2.

**The verdicts**, the first 30 days pooled, geometric mean over windows, 95 % interval over
whole days:

| Verdict | Mission | Against | Ratio [95 %] | Median over cells | Level |
|---|---|---|---|---|---|
| **A** | **Mission1** | AutoETS | **1.293 [1.221; 1.368]** | 1.24 | **harm** |
| **A** | **Mission2** | AutoETS | **0.752 [0.665; 0.863]** | 0.96 | **real and small** |
| | | | | | the claim of a lead from the first days: **not met** |
| **B** | **Mission2** | `chronos2_47` | **1.146 [1.069; 1.232]** | 1.10 | **harm** |

The same days, descriptively (over `chronos2_47` unless named):

| | Mission1 | Mission2 |
|---|---|---|
| `_own` over the best classical answer | 0.915 [0.860; 0.977] | 0.635 [0.543; 0.756] |
| `_own` | 0.707 [0.665; 0.749] | 0.843 [0.767; 0.924] |
| `_each` | 1.083 [1.042; 1.125] | 1.002 [0.942; 1.063] |
| `unrelated` | 1.137 [1.104; 1.172] | 1.058 [1.000; 1.121] |
| `kind` | 1.095 [1.071; 1.120] | 1.044 [0.988; 1.101] |
| `analog` | 1.091 [1.061; 1.122] | 1.146 [1.069; 1.232] |
| `analog` over `kind` | 0.996 [0.980; 1.012] | 1.098 [1.029; 1.173] |
| the oracle | 1.029 [1.006; 1.051] | 0.057 [0.049; 0.067] |

Each day, against the best classical answer (AutoETS but where named), descriptive:

| Day | Mission1, `chronos2_47` | Mission1, `_own` | Mission2, `chronos2_47` | Mission2, `_own` |
|---|---|---|---|---|
| 2 | 1.31 [1.20; 1.48] | 1.09 [0.97; 1.25] | 0.94 [0.76; 1.24] | 0.89 [0.71; 1.21] |
| 7 | 1.22 [1.13; 1.32] | 0.83 [0.79; 0.87] | 0.68 [0.61; 0.84], median of the 47 hours | 0.54 [0.45; 0.72] |
| 14 | 1.33 [1.21; 1.45] | 0.90 [0.83; 0.99] | 0.75 [0.65; 0.89] | 0.63 [0.52; 0.78] |
| 30 | 1.08 [0.91; 1.25] | 0.83 [0.73; 0.94] | 0.86 [0.74; 0.99] | 0.60 [0.53; 0.67] |
| 60 | 1.23 [1.07; 1.41] | 1.05 [0.83; 1.39] | 0.80 [0.72; 0.91] | 0.64 [0.56; 0.76] |
| 90 | 1.09 [1.00; 1.23], AutoARIMA | 1.45 [1.24; 1.81] | 0.85 [0.77; 0.94] | 0.89 [0.77; 1.05] |
| 180 | 1.00 [0.92; 1.13], AutoARIMA | 1.12 [1.02; 1.26] | 0.88 [0.77; 1.01] | 0.84 [0.76; 0.93] |
| 365 | 1.25 [1.17; 1.33] | 1.61 [1.52; 1.72] | 1.05 [0.96; 1.22] | 0.91 [0.84; 1.03] |

On Mission1 the median over cells reads `_own` against `chronos2_47` at 1.44 on day 180 and 2.57
on day 365: past three months, its longer context costs the typical cell too.

**Predictions against the outcome.**

| Predicted | Read |
|---|---|
| A, Mission1: 1.0 to 1.4, noise or harm | 1.293, harm: held |
| A, Mission2: 0.85 to 1.5, noise | 0.752, real and small: missed; the trials' days were not the first month |
| The claim of a lead from the first days: not met | not met: held |
| B, Mission2: 0.8 to 1.45, noise or harm | 1.146, harm: held |
| Mission2's oracle: 0.03 to 0.07 | 0.057: held |

**Checked after the reading**, descriptive (pooled first 30 days):

| Check | Result |
|---|---|
| Recomputed by independent code from the files | every verdict and interval; Chronos-2 re-answered on 12 windows to within 5×10⁻⁶ |
| Where Mission1's deficit lies | 88 % of cells hold a period within the 47 hours, handed to the local forecasters as their season (18 hours in 68 %): `chronos2_47` over AutoETS 1.42 there, 0.73 elsewhere; `_own` 0.99 there. Leaving out any one channel: 1.26 to 1.33; jumps carry 3 % of its squared error |
| The local forecasters without a season, Mission2 | they miss 15 to 20 % less than with the period the same-phase answer chose; `chronos2_47` over AutoETS, AutoARIMA, AutoTheta so fitted 0.88, 0.80, 0.84 (on Mission1, AutoETS without a season misses 11 times more) |
| Mission2's channels | 14 of 33 move together (median correlation of hourly changes 0.996) and carry about 80 % of the windows' squared error |
| What the library carries | the nearest series' next hour moves with the target's at 0.02 on Mission2, yet Chronos-2's answer follows it at 0.23; the mean of the 200 nearest of the same kind moves with the target at −0.07 (Mission2) and 0.30 (Mission1), and with `chronos2_47`'s error at 0.04 and 0.07 |
| The rungs' series | each goes to the whole window's group, so Chronos-2 does not pair a series with its channel; the row above bounds any pairing |

**Conclusions.**

1. In a satellite's first month, Chronos-2 zero-shot on the 47 hours does not lead the classical
   answers on both missions: on the first, AutoETS fitted on the same 47 hours misses less (1.29,
   harm), on the second Chronos-2 leads (0.75, real and small; 0.88 against the same forecasters
   without a season, above). By the declared rule the claim is not met, and the local classical
   forecasters are the bar a model must beat from day 2; on the first mission they are ahead or
   level at every day read, while on its later years Chronos-2 led them (0.475, 2026-11-19).
2. Handed every hour so far, up to 8,192, Chronos-2 leads the best classical answer in the
   first month on both missions (0.92 and 0.64, descriptive; the local forecasters fit 47
   hours). On the first mission the long context turns against it from day 90 (1.12 to 1.61).
3. Another satellite's series never helps. On the second mission the nearest one makes the
   forecast worse (1.15, harm), and worse than a random series of the same kind (1.10), though
   the route carries the hour itself (oracle 0.06); on the first the route carries nothing. The
   library itself holds no next hour that Chronos-2 lacks (above), so no pairing or number of
   neighbours would help one hour ahead.
4. By the declared table: a domain model has room from day 2 against the local classical
   answers, and a library of other systems' series is read next on the turbine fleet, systems of
   one design side by side, before a multi-system corpus builds on it.

**Limitations.** As declared. The second mission's lead was not predicted: the trials drew days
5, 100 and 400, not the first month. `_own` sees more hours than the local forecasters, so its
lead is not a like-for-like comparison. The local forecasters take the same-phase answer's period
as their season, which weakens them on Mission2; a season each forecaster chooses per cell is the
bar's next form. Mission2's verdict rests on few independent signals. One draw of hours per day;
the descriptive rows were not declared as verdicts.
