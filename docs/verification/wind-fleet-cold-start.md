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
