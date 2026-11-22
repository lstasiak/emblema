# A wind turbine read as a new system: forecasts from its first days

The product's third claim is the onboarding of a system with no history and no labels. The two
satellites (`telemetry-cold-start.md`) gave two systems; a wind farm gives many of one design side
by side. This note reads each published turbine of two farms as if it had just been switched on,
the farm's other turbines kept, to set the bar a model must beat on a new turbine, with and
without the farm's history, and to see what the published forecaster does with the hour's
operating conditions.

## 2026-11-21 — declared before the run: the published forecaster on a new turbine, from its past and with the hour's operating conditions, against the classical answers with and without the farm

**Question.** In the first 30 days of a turbine read as new, does Chronos-2 zero-shot forecast
its component temperatures' next hour better than (A) the best classical answer from the past,
the farm's models included; (B) itself, when handed the other turbines' recent hours; (C) the
best classical answer that also knows the hour's operating conditions, without and with the
farm's models?

**Conditions.**

| | |
|---|---|
| Commit | the one that adds this section; the instrument at `34c3892b` |
| Machine | Mac (M1 Pro), Chronos-2 on MPS; validation numbers |
| Data | Kelmarsh `95596b21…` (turbines 1, 3, 4, 5) and Penmanshiel `451ab8f5…` (01, 02, 04, 06, 07, 08, 10, 13, 14, 15), robust score held to 100, statistics fitted on 2016 to 2019 (ADR-0085); hourly means per turbine (`telemetry_hourly_means.py --shared-channels`). Six turbines and 2023 to 2024 are held unread. |
| Starts | Every 35 days through 2020 to 2022 from an offset drawn by seed 1 and the farm's name: Kelmarsh 31 (2020-01-14 to 2022-11-29), Penmanshiel 30 (2020-02-02 to 2022-11-13). Each turbine is read as new from each start, its earlier hours unknown. The 20 starts per farm whose 395 days end by 2022 also make the curve. |
| Days and cells | Days 2, 7 and 14 (to day 30) on every start; 30, 60, 90, 180 and 365 on the curve's. Each day, 8 hours drawn per start, shared by its turbines. A cell is one of 17 component temperatures read in the hour and in 24 of the 47 before; a window is one turbine's hour. |
| The past | The last hour, the median of the 47 hours and the hour one period back; AutoETS, AutoARIMA and AutoTheta with that period as season and with none, and seasonal naive; ridges on the turbine's own hours since the start; the farm's ridges, fitted on the other turbines' hours before the start, and boosted trees (XGBoost, rate 0.1, 300 rounds, at most 100,000 rows) on the window ridge's rows of the other turbines' hours before 1 January of the start's year. |
| Normal behaviour | Also knowing six signals of the forecast hour, as the turbine measures them: wind, power, rotor speed, blade angle A and two ambient temperatures. The cell's own regression of each of its 47 hours on the hour before and the hour's signals (penalty 0.1, the best by its own error on the first trial, both farms pooled). A ridge on the turbine's own hours since the start, and the farm's ridge and trees, on the channel's 47 hours, the power's last 24 and the six signals. Read on the cells whose six signals were read in the hour. |
| Published forecaster | Chronos-2 zero-shot (`amazon/chronos-2`, `chronos-forecasting` 2.3.2). `chronos2_47`: the window's 21 channels, 47 hours. `chronos2_neighbours`: with each temperature's mean over the other turbines across the same hours. `chronos2_nbm`: the 17 temperatures' 47 hours, the six signals as covariates over them and, known, in the forecast hour. Beside: each channel alone; every hour since the start (`_own`, `chronos2_nbm_own`); the neighbours' mean in the hour itself; the pitch motors' currents as three more signals. |
| Bars | Without the farm: the answers from the new turbine's data alone, the satellites' bar with the local forecasters also fitted with no season. With the farm: those and the farm's ridges and trees. The ridges on the turbine's own hours join only where they answer every cell, so never on the pooled days, as on the satellites; every other answer keeps its place and the comparison reads the cells all of them answer. |
| Measure | Each window's mean squared error of the hour's mean over its cells; the geometric mean over windows of the forecaster's over the bar's answer it leads least, taken in every draw. 10,000 draws of whole starts, seed 1, 95 % interval. Beside: the median over cells. |
| Reading | `fleet_cold_start_reading.py` per farm on days 2, 7 and 14 pooled; every day of the curve and each group of temperatures descriptively; `data/report/l62b/run.sh` runs the whole. |

**Rules.** Each verdict per farm; least gain 0.80; levels as on 2026-11-19. A claim needs at least
the least gain on both farms, so no correction applies.

| Verdict | Of | Against |
|---|---|---|
| **A** | `chronos2_47` | the past, with the farm |
| **B** | `chronos2_neighbours` | `chronos2_47` |
| **C without the farm** | `chronos2_nbm` | normal behaviour without the farm |
| **C with the farm** | `chronos2_nbm` | normal behaviour with the farm |

| Outcome | What follows |
|---|---|
| C without the farm: the least gain on both | a zero-shot forecaster of normal behaviour is the onboarding layer of a system with no fleet from day 2 |
| C without the farm: less on either | on 47 hours the forecaster adds nothing to the cell's own regression; an onboarding model must beat it and approach the farm's trees |
| C with the farm: the least gain on both | the forecaster beats what an operator with a fleet already has |
| A: the least gain on both | from the past alone the forecaster beats the fleet's models |
| A and C with the farm: less on either | the farm's history is the bar for a new turbine of a known fleet; the stage's model is read against the farm's trees |
| B: the least gain on both | the other turbines' hours carry the next hour into the forecaster; the corpus is built as a fleet |
| B: less on either | the forecaster takes nothing near the least gain from its neighbours' past |

**Predictions** are the second trial's intervals (below), and the level each would read:

| Verdict | Kelmarsh | Penmanshiel | Level predicted |
|---|---|---|---|
| A | 1.09 to 1.28 | 1.05 to 1.32 | harm |
| B | 0.99 to 1.00 | 0.98 to 1.00 | noise, or real and small |
| C without the farm | 0.92 to 1.04 | 1.00 to 1.12 | noise, or harm on Penmanshiel |
| C with the farm | 1.64 to 1.90 | 1.92 to 2.41 | harm |
| Beside: `chronos2_nbm_own` without the farm | 0.80 to 0.89 | 0.87 to 1.01 | real and small |
| Beside: the farm's normal-behaviour trees, without the farm | 0.51 to 0.61 | 0.44 to 0.56 | the room a model of the turbine's kind has |

**Power.** Scaled from the trial's intervals by the square root of the starts, the log half-width
of each verdict's interval falls to 0.004 to 0.06. A true ratio at or under 0.73 to 0.79 reads at
least the least gain with power 0.9; the verdicts predicted less have lower edges far above 0.80,
so each can close its direction.

**Trials, before this declaration**, on 2018 to 2019, years the reading does not touch, 8 starts
per farm and 8 hours a day (`data/report/l62b/trial`, `trial2`, `trial.sh`). Changed after them:

- The oracle that knows the hour's power read its level alone (0.98 against the farm's ridge);
  with the hour before too it sees the change of load, 0.87 to 0.92 against the farm's window
  ridge. It still cannot reach 0.80, since the hour's temperature change follows the hour's
  wind, so the normal-behaviour verdicts were added (user decision, 2026-11-21).
- The farm's trees halve the farm's ridge's error with the hour's signals and cut it by a fifth
  from the past alone, on each of the 14 turbines of the fitting years; they joined both bars.
- A bar answer missing a few cells dropped out whole; now only the ridges on the turbine's own
  hours may.
- The first normal-behaviour bar held no answer that knew the hour's signals on the first days:
  Chronos-2 read 0.65 and 0.69 against answers from the past on the first trial. With the cell's
  own regression, which reads what the forecaster reads, the second trial reads 0.98 and 1.06
  (an independent review found this).
- Starts are every 35 days rather than drawn, 31 and 30 against at most 23.
- A Chronos-2 start that waits over five minutes to load prints its stacks and runs again.

Checked: every ratio of the first trial recomputed from the files by independent code; Chronos-2's
inputs equal the hourly means, and carrying its missing hours changes its error by under 1 %;
16 deliberate breakages of the scripts each fail a test (`breakages.log`).

| Second trial, first month | Kelmarsh | Penmanshiel |
|---|---|---|
| A | 1.171 [1.086; 1.276] | 1.186 [1.054; 1.318] |
| B | 0.995 [0.987; 1.003] | 0.993 [0.983; 1.002] |
| C without the farm | 0.977 [0.923; 1.035] | 1.057 [0.996; 1.122] |
| C with the farm | 1.768 [1.638; 1.899] | 2.132 [1.916; 2.409] |
| `chronos2_47` without the farm | 0.919 [0.877; 0.955] | 0.952 [0.895; 1.015] |
| `chronos2_nbm_own` without the farm | 0.846 [0.802; 0.892] | 0.929 [0.870; 1.010] |
| Normal-behaviour trees, without the farm | 0.551 [0.506; 0.608] | 0.498 [0.437; 0.556] |

**Limitations declared.** Two farms of one maker, the same years. On the trials the trees had less
history than the reading's: on Penmanshiel, whose six temperatures begin in April 2018, the
normal-behaviour trees answered no 2018 start, so C with the farm read 4 starts and C without the
farm 7. The normalised units were fitted on 2016 to 2019. The converter's ambient temperature may
be warmed in its cabinet; its hourly change moves with the other turbines' at 0.92 and 0.88, so
it is read as weather. Between 1 January and the start the farm's ridges see hours its trees do
not. One published forecaster; the hour's mean, one hour ahead.

## 2026-11-22 — measured: on a new turbine the published forecaster reaches the least gain in no verdict, and the farm's trees about halve the best classical error

**Question.** As declared above: in the first 30 days of each turbine read as new, does Chronos-2
lead the best classical answer from the past with the farm (A), gain from its neighbours' past
(B), or lead the best classical answer of normal behaviour without and with the farm (C)?

**Conditions.** Instrument `34c3892b`, declaration `5fc4e1b9`, `data/report/l62b/run.sh` on the
Mac (M1 Pro) on 2026-11-22: Kelmarsh 2.7 hours, Penmanshiel 6.3 hours, no step run twice. The first
30 days hold 2,959 windows over 31 starts on Kelmarsh and 7,111 over 30 on Penmanshiel; the
normal-behaviour verdicts read 2,903 and 7,077 windows, those whose six signals were read less 4
and 1 with a cell their bar left unanswered. Every answer of the bars answered every cell, except
the ridges on the turbine's own hours, left out as declared; the normal-behaviour bars missed
0.1 % and 0.01 % of their cells.

**The verdicts**, days 2, 7 and 14 pooled, geometric mean over windows, 95 % interval over whole
starts:

| Verdict | Farm | Against | Ratio [95 %] | Median over cells | Level | Predicted |
|---|---|---|---|---|---|---|
| **A** | Kelmarsh | the farm's trees | **1.270 [1.189; 1.352]** | 1.31 | **harm** | harm |
| | Penmanshiel | the farm's trees | **1.262 [1.200; 1.327]** | 1.24 | **harm** | harm |
| **B** | Kelmarsh | `chronos2_47` | **0.994 [0.991; 0.998]** | 0.99 | **real and small** | noise or real and small |
| | Penmanshiel | `chronos2_47` | **0.981 [0.977; 0.985]** | 0.99 | **real and small** | noise or real and small |
| **C without the farm** | Kelmarsh | the cell's own regression | **0.988 [0.954; 1.024]** | 1.04 | **noise** | noise |
| | Penmanshiel | the cell's own regression | **1.062 [1.020; 1.103]** | 1.07 | **harm** | noise, or harm |
| **C with the farm** | Kelmarsh | the farm's trees | **1.842 [1.714; 1.973]** | 1.93 | **harm** | harm |
| | Penmanshiel | the farm's trees | **2.107 [1.975; 2.250]** | 2.02 | **harm** | harm |

No claim holds: no verdict reads the least gain on either farm. Every level and every point,
the two rows beside included, fell where the trials put them.

The same days, descriptively:

| | Kelmarsh | Penmanshiel |
|---|---|---|
| `chronos2_47` against the past without the farm | 0.921 [0.897; 0.947] | 0.984 [0.950; 1.018] |
| `chronos2_own` against the past with the farm | 1.183 [1.120; 1.252] | 1.191 [1.137; 1.248] |
| `chronos2_nbm` against `chronos2_47`: what the hour's signals give it | 0.696 [0.676; 0.717] | 0.719 [0.692; 0.746] |
| `chronos2_nbm_own` against normal behaviour without the farm | 0.833 [0.797; 0.872] | 0.905 [0.866; 0.946] |
| `chronos2_nbm_own` against normal behaviour with the farm | 1.553 [1.450; 1.657] | 1.797 [1.690; 1.916] |
| The farm's normal-behaviour trees against normal behaviour without the farm | 0.537 [0.495; 0.584] | 0.504 [0.477; 0.531] |
| The farm's trees against its window ridge, from the past | 0.724 [0.693; 0.750] | 0.721 [0.699; 0.745] |
| The neighbours' mean in the hour itself, against `chronos2_47` | 0.765 [0.735; 0.795] | 0.670 [0.641; 0.701] |
| The oracle that knows the hour's power, against the farm's window ridge | 0.894 [0.876; 0.911] | 0.879 [0.855; 0.902] |
| The pitch motors' currents, against `chronos2_nbm` | 0.873 [0.854; 0.893] | 0.846 [0.831; 0.862] |

**The curve**, each day on the 20 starts whose year is read (19 on day 2, and for the ridge on
Penmanshiel's day 7), against normal behaviour without the farm unless named:

| Day | `chronos2_nbm` K | P | `chronos2_nbm_own` K | P | the farm's trees K | P | own ridge over the farm's K | P |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.95 | 0.98 | 0.90 | 0.87 | 0.56 | 0.50 | – | – |
| 7 | 0.96 | 1.10 | 0.80 | 0.96 | 0.51 | 0.50 | 2.73 | 3.02 |
| 14 | 1.02 | 1.10 | 0.81 | 0.84 | 0.51 | 0.51 | 1.49 | 1.69 |
| 30 | 0.91 | 1.03 | 0.75 | 0.79 | 0.54 | 0.48 | 1.28 | 1.38 |
| 60 | 1.03 | 1.12 | 0.77 | 0.79 | 0.57 | 0.46 | 1.13 | 1.16 |
| 90 | 0.99 | 1.08 | 0.83 | 0.80 | 0.60 | 0.47 | 1.13 | 1.17 |
| 180 | 0.92 | 0.99 | 0.84 | 0.80 | 0.55 | 0.50 | 1.15 | 1.10 |
| 365 | 0.97 | 1.17 | 0.83 | 1.03 | 0.50 | 0.59 | 0.98 | 1.01 |

K: Kelmarsh, P: Penmanshiel. The ridge on the turbine's own hours from the past, over the farm's.

**Within the groups of temperatures**, descriptively, C without the farm leads on the stator (0.91
and 0.77) and the generator bearings (0.88 and 0.90), trails on the transformer (1.42 and 1.35)
and Penmanshiel's gear oil (1.33; Kelmarsh 1.08 [0.99; 1.18]). With the farm it trails in every
group (1.16 to 3.63).

**Checked before reading.** The run's conditions name the declared starts, turbines, hours and
seed; the four verdicts' points recomputed from the files by independent code
(`data/report/l62b/recheck_run.py`) agree to the third decimal.

**Conclusions.**

1. On 47 hours the published forecaster adds nothing to the regression of the cell's own hours
   on the hour's operating conditions: it matches it (0.99, noise) or trails it (1.06, harm).
   From the past it trails the farm's trees by about a quarter (1.27 and 1.26, harm). An
   onboarding model has to beat that regression and approach the farm's trees.
2. For a turbine of a known fleet, the farm's history is the bar. With the hour's conditions the
   farm's trees miss about half as much as the best classical answer from the turbine alone
   (0.46 to 0.60 from day 2 to day 365); the forecaster misses 1.84 and 2.11 times as much as
   they do. The ridge on the turbine's own hours reaches the farm's ridge only at day 365 (0.98,
   1.01); the window ridge reaches it within noise at day 60 and stays there on Penmanshiel.
3. The neighbours' past adds 0.6 and 1.9 % to the forecaster, real but far from the least gain.
   Their same hour would add 24 and 33 %, which suggests that what the turbines share is the
   hour's weather rather than its history.
4. Descriptively, the forecaster gains with context. On every hour since the start, with the
   conditions, it leads the regression by 17 and 9 % on the first month, and on every day of the
   curve but Penmanshiel's days 7 and 365 (0.96, 1.03) by 10 to 25 %. It still trails the farm's
   trees by 55 and 80 %, the gap a model adapted to the domain has to close.

**Limitations.** Two farms of one maker in the same years; Kelmarsh has four turbines. Turbines
read as new share their farm's weather, so a start is one sample, not four or ten. The trees and
the ridges on the farm read the turbine's kind, which a new system of a new design lacks. One
published forecaster, zero-shot; the hour's mean, one hour ahead. The normalised units were fitted
on 2016 to 2019 and weigh the channels in a window's mean. The descriptive rows were not declared
as verdicts.
