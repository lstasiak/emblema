# A fleet of wind turbines: what three open sources hold

The onboarding of a system with no history is to be read on many systems of one design, and wind
turbines are the open data that offers them: farms of identical machines under one weather, and a
benchmark of labelled faults. This note records what the three sources hold, measured from their
files before any reading was built on them, and what the two farms' publications hold.

## 2026-11-20 — measured: two farms are a clean fleet on a ten-minute grid whose latest tables repeat every instant

**Question.** What do the Kelmarsh and Penmanshiel farm data hold, as files: which signals, on
what grid, with what errors, and how far do a farm's turbines move together? What must a reader
and a reading know before they touch them?

**Conditions.**

| | |
|---|---|
| Commit | the one that adds this note |
| Machine | Mac (M1 Pro), CPU |
| Kelmarsh | Zenodo 10.5281/zenodo.16807551, CC BY 4.0: nine yearly archives, 3.69 GB |
| Penmanshiel | Zenodo 10.5281/zenodo.16807304, CC BY 4.0: twenty archives, 7.36 GB |
| Checks | every archive against the publisher's MD5 (`scripts/fetch_corpora.py`) |
| Commands | `scripts/wind_fleet_audit.py farm` per farm, the values on the turbines and years a first publication reads (below); `scripts/regime_coverage_report.py measure --corpus kelmarsh penmanshiel` |

Six turbines and the years 2023 and 2024 are held unread from here on
([preregistration](../preregistration.md), "The fleet of wind turbines"). The tables' structure,
their instants and each signal's coverage were measured on everything; every statistic of a
value on the other fourteen turbines before 2023.

**The two farms.**

| | Kelmarsh | Penmanshiel |
|---|---|---|
| Turbines | 6 Senvion MM92 | 14 Senvion MM82 (no turbine 03) |
| First reading | 21 January to 5 February 2016 | 2 June to 27 July 2016, turbine by turbine |
| Last reading | 31 December 2024 | 31 December 2023 (nine turbines, without January 2023) or 2024 (five) |
| Tables | 54 | 144 |
| Rows, and distinct instants | 28,305,504 and 2,839,104 | 36,912,986 and 5,758,976 |
| Tables that repeat an instant | 12, all of 2023 and 2024 | 46, all of 2023 and 2024 |
| Most rows of one instant | 81 | 92 |
| Instants whose rows disagree | 0 | 0 |
| Readings of the sixty signals | 157,169,305 | 308,715,149 |
| Steps of ten minutes, by turbine | 99.998 % or more | 99.993 % or more |

- A table holds 299 to 372 columns. Up to sixty are the averages of the signals the
  manufacturer's mapping names for both farms; the rest are their minima, maxima and
  deviations, a few averages the mapping does not name, and what the export derives (lost
  production, availability, potential power).
- The tables of 2023 and 2024 are cumulative. A table restarts its period every few hundred rows
  and repeats every instant before, with values in its newest rows alone: on average 41 rows an
  instant at Kelmarsh and 34 at Penmanshiel. Taken together the rows of an instant hold each
  signal once and never two values.
- The signals grow with time. At Kelmarsh 44 of the sixty read in 2016, 14 more from 2017 and
  two from 2018; power and wind begin in January and February 2016, the temperatures on 3 May
  2016 at 09:40 in all six turbines at once. At Penmanshiel 38 read from the start and 22 from
  2018, among them the gear oil and the front and rear bearing temperatures.
- One signal stands under two spellings of its unit (`mm/ss`, then `mm/s2`) and one title is
  cased otherwise in the tables than in the mapping.

**Sampling**, as the coverage report measures every corpus (100 months of a turbine, seed 1,
windows of an hour; median [10th; 90th percentile]):

| Corpus | Co-observed | Gap variation | Rate spread | Tokens / window |
|---|---|---|---|---|
| Kelmarsh | 1.00 [0.95; 1.00] | 0.05 [0.00; 1.56] | 0.00 [0.00; 0.00] | 360 [264; 360] |
| Penmanshiel | 1.00 [0.97; 1.00] | 0.02 [0.00; 1.24] | 0.00 [0.00; 0.00] | 360 [228; 360] |

Both are grids: every signal of a turbine is read at the same instants.

**Values**, on the turbines read, before 2023.

- For 50 of the sixty signals at Kelmarsh and 51 at Penmanshiel the standard deviation is 0.7
  to 3.0 times the robust deviation (1.4826 times the median absolute deviation). The others
  sit at one value most of the time: the three blade angles (17 to 242 times), the energy
  imported (57 to 59) and the reactive energy exported (341 to 8,982), the grid voltage (24 to
  26) and frequency (6 to 14), the power factor (7), the drive train acceleration at Kelmarsh
  (5) and the metal particle count, whose robust deviation is nought. Under a robust score held
  to 100, 15 % of Kelmarsh's blade angle readings lie at the bound.
- The ranges Penmanshiel's mapping states are limits of storage (−275 to 200 °C for every
  temperature), and one is wrong (0 to 360° for a vane that reads −180 to 180°); Kelmarsh's
  states none. They bound nothing.
- A failed sensor writes a sentinel. Readings outside a wide physical range (−50 to 150 °C,
  0 to 60 m/s, −10 to 2,500 RPM; `*-implausible.csv`):

| Signal | Kelmarsh: readings, commonest value, hours touched | Penmanshiel |
|---|---|---|
| Transformer temperature | 6, −273.2 °C, 4 | 122, −273.2 °C, 25 |
| Transformer cell temperature | 6, −273.2 °C, 4 | 122, −273.2 °C, 25 |
| Temperature motor axis 1, 2 and 3 | 3, 2 and 3; 215.0 °C; 2, 1 and 2 | 27, 16 and 3; 215.0 °C; 10, 7 and 1 |
| Ambient temperature (converter) | 1, −113.3 °C, 1 | 2, −104.6 °C, 2 |
| Stator temperature 1 | 0 | 1, 264.0 °C, 1 |
| Wind speed Sensor 2 | 409, −15.0 m/s, 87 | 4,327, −15.0 m/s, 769 |
| Wind speed Sensor 1 | 6, −1.5 m/s, 6 | 1, −1.0 m/s, 1 |
| Generator RPM | 430, −576.6, 170 | 403, −576.0, 298 |

  Of the 167,000 to 559,000 hours a temperature reads on a farm, its sentinels touch at most
  25; twelve of the nineteen temperatures hold none at either farm. Values between a sentinel
  and the ordinary range are ten-minute averages over a sensor that failed part-way. Outside
  these units an energy of 230,667 kWh in ten minutes and a particle count of −13,223 stand at
  Penmanshiel.

**The fleet**, hourly means on the turbines read, before 2023 (`*-hourly.csv`): the correlation
of a signal's change from one hour to the next between two turbines of a farm, median over pairs.

| Signal | Kelmarsh | Penmanshiel |
|---|---|---|
| Wind speed | 0.72 | 0.86 |
| Power | 0.71 | 0.77 |
| Nacelle ambient temperature | 0.97 | 0.94 |
| Transformer temperature | 0.79 | 0.76 |
| Rotor bearing, front and rear bearing temperatures | 0.64 to 0.67 | 0.60 to 0.68 |
| Gear oil temperature | 0.57 | 0.56 |
| Generator bearing temperatures | 0.30 to 0.32 | 0.46 to 0.50 |
| Nacelle temperature | 0.30 | 0.44 |

- The 19 temperatures of a turbine behave as 6.4 independent signals at Kelmarsh (6.2 to 6.6 by
  turbine) and 6.1 at Penmanshiel (6.0 to 6.5): the participation ratio of the correlations of
  their hourly changes (`*-independent.csv`).
- Only the ambient signals carry a day's rhythm: a change correlates with the change a day
  before at 0.29 to 0.67 for the two ambient temperatures and Kelmarsh's transformer cell, and
  below 0.21 for every other temperature. A temperature's level correlates with the hour
  before at 0.88 to 1.00.

**Conclusions.**

1. Kelmarsh and Penmanshiel are a fleet as the readings need one: twenty turbines of one maker
   over up to nine years at real instants on one grid, sixty signals named alike. Their reader
   keeps the averages of those signals and takes the rows of an instant as one reading
   (ADR-0084); read otherwise, the cumulative tables lose nearly every value of 2023 and 2024.
2. A turbine's neighbours carry much of the same hour: wind and power move together at 0.7 to
   0.9, the component temperatures at 0.3 to 0.8. Nineteen temperatures are about six signals,
   which is what a reading's power counts.
3. All of a farm's turbines start within weeks, Kelmarsh's temperatures within the same ten
   minutes: the real first days are one stretch of weather per farm.
4. The values need two decisions before a publication, which ADR-0085 takes. A few hundred
   readings are sentinels of failed sensors with no published bound to remove them by, and nine
   or ten signals of a farm have no robust spread to be scored by.

**Limitations.** The values and the fleet's correlations are read on fourteen of the twenty
turbines, before 2023. The physical ranges are this note's, wide by choice, and no publisher's.
The status logs, which name each turbine's stops, were not read.

## 2026-11-20 — measured: the fault benchmark's excerpts overlap and share real time only in nine groups

**Question.** CARE to Compare offers what the farms lack, turbines with annotated faults. How
are its excerpts laid out, and can they be read as a fleet?

**Conditions.** As above; Zenodo 10.5281/zenodo.15846963 (version 6), CC BY-SA 4.0, one archive
of 5.50 GB checked against the publisher's MD5; `scripts/wind_fleet_audit.py care`, which
reads every excerpt's stamps and wind speed and no other sensor (`care-excerpts.csv`,
`care-groups.csv`, `care-sensors.csv`).

95 excerpts of 36 turbines on three farms, each a year of training followed by the window a
detector is scored on.

| | Farm A | Farm B | Farm C |
|---|---|---|---|
| Excerpts, turbines | 22, 5 | 15, 9 | 58, 22 |
| With an annotated fault, without | 12, 10 | 6, 9 | 27, 31 |
| Sensors, of them temperatures | 54, 25 | 63, 25 | 238, 77 |
| Training, days | 350 to 363 | 361 to 365 | 353 to 365 |
| Scored window, days | 8 to 34 | 9 to 98 | 4 to 73 |
| Fault's annotated lead-up, days | 7 to 32 (median 7) | 20 to 61 | 2 to 66 (median 15) |
| Groups sharing real time, by size | 20, 2 | 10, 4, 1 | 26, 16, 15, 1 |
| Pairs of one turbine with the same rows, with other rows at equal stamps | 30, 8 | 6, 0 | 40, 24 |

- The publisher shifted every excerpt by whole years to begin in 2022. Excerpts of different
  turbines share real time where their wind speeds at equal stamps correlate above 0.7: within
  the groups found they do at 0.89 or more, between them at 0.55 at most.
- Two excerpts of one turbine either hold the same rows, so the scored window of one can lie
  inside the training year of the other, or hold another year's rows under the same stamps.
- The version read counts 45 excerpts with a fault and 50 without; its README still says 44 and
  51. The "86, 257 and 957 features" of its description count five descriptive columns and the
  statistics; the sensors are 54, 63 and 238.

**Conclusions.**

1. The benchmark is no fleet as published: its excerpts overlap, share time in nine groups that
   nothing marks, and each ends in its own scored window. A neighbour can be read only inside a
   group found from the data.
2. It is read from its files for a detection reading, its excerpts split by turbine so that no
   scored window lies in another excerpt's training year, and it joins no corpus as it is.
3. Its faults are slow: an annotated lead-up of two days to two months, a week or more at each
   farm's median.

**Limitations.** Shared time is inferred from wind alone, at a threshold the gap between 0.55
and 0.89 leaves free. No sensor but the wind speed was read, so nothing here says what a
detector finds.

## 2026-11-21 — measured: the two publications hold the audit's readings less the ones it listed as a failed sensor's

**Question.** Published as decided (ADR-0085), does each farm hold what the audit counted on the
turbines and years it reads, and does it remove the readings the audit listed and no others?

**Conditions.** Commit `c9a47a0f`, the reader at `7f643708`; Mac (M1 Pro), CPU; 2026-11-20 to 21.
Per farm, the turbines not held unread and the months before 2023:

    uv run --env-file .env.r2 python -m emblema.entrypoints.cli.publish_corpus --corpus kelmarsh \
        --subset 1 --subset 3 --subset 4 --subset 5 --exclude-units <the months of 2023 and 2024> \
        --window 1 --stride 1 --normalisation robust --robust-bound 100 \
        --cleaning corpora/cleaning/wind_farms.toml --vocabulary-from <the publication before> \
        --hold-out <the months of 2020 to 2022>

Kelmarsh continues the vocabulary of the raw ESA counterparts' publication (`f5f675bb…`),
Penmanshiel that of Kelmarsh. The counts below are read from the manifests.

| | Kelmarsh | Penmanshiel |
|---|---|---|
| Manifest | `95596b21…` | `451ab8f5…` |
| Turbines | 4 | 10 |
| Months of a turbine: fitted on, held out | 191, 144 | 426, 360 |
| Windows of an hour | 242,582 | 565,407 |
| Tokens | 79,658,907 | 180,052,542 |
| Readings the audit counted there | 79,659,773 | 180,057,566 |
| Readings removed, channels they lie in | 866, 9 | 5,024, 10 |
| Block | 1.36 GB | 3.08 GB |
| Time to publish | 10 min | 22 min |

**Conclusions.**

1. Each publication holds exactly the audit's readings less those it removed (79,659,773 − 866
   and 180,057,566 − 5,024), so the reader and the audit's own count agree on every turbine and
   year published.
2. The readings removed are the audit's inventory, channel by channel: at Penmanshiel 122 each
   of the two transformer temperatures, 27, 16 and 3 of the three axis motors', 4,327 of the
   second anemometer, 403 of the generator's speed. Nothing was clipped.
3. All sixty channels have statistics from the fitting side, 2016 to 2019, under the robust
   score held to 100.

**Limitations.** A check of counts, not of values: no reading has yet read these publications.
The stray readings between a range's edge and the ordinary values stay (ADR-0085).
