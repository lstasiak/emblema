# ADR-0085: A wind farm's publication removes readings outside a wide physical range and fits its statistics on the fleet's first four years

- Status: accepted
- Date: 2026-11-20
- Extends ADR-0078 to a corpus with no published bounds, and ADR-0080 to the wind farms

## Context

The reader of the two wind farms (ADR-0084) reads values raw. Measured on the turbines and years
a first publication reads ([`wind-fleet-sources.md`](../verification/wind-fleet-sources.md)):

- A failed sensor writes a sentinel: a transformer at −273.2 °C, an axis motor at exactly
  215.0 °C, an anemometer at −15.0 m/s. Seven of the nineteen temperatures hold such readings,
  at most 122 in 2.4 million, with ten-minute averages over a sensor that failed part-way
  between them and the ordinary range. An hour's mean over one of them is wrong by tens of
  degrees.
- No published bound removes them. The ranges the manufacturer's mapping states are limits of
  storage (−275 to 200 °C), and one is wrong.
- The signals a forecast reads (the temperatures, wind speed, power, rotor speed) have a
  standard deviation 0.8 to 2.7 times their robust deviation, and none of their readings lies
  beyond 100 robust deviations. Ten other signals sit at one value most of the time and have
  next to no robust spread.
- Six of Penmanshiel's temperatures are first read in April 2018.

## Decision

- **A publication of a wind farm removes readings outside one wide physical range per unit**:
  −50 to 150 °C, 0 to 60 m/s, −10 to 2,500 revolutions a minute
  (`corpora/cleaning/wind_farms.toml`, by the mechanism of ADR-0078). A removed reading is one
  never taken; nothing is clipped. The manifest states how many each channel lost.
- **The ranges are this project's**, set after the audit and before any publication or reading.
  They use no label and no outcome, and were measured without the turbines and years held
  unread. They are wide by choice, so what a publication removes is the audit's inventory and
  no weather or load.
- **Channels in other units are published as read**, their faults with them (an energy of
  230,667 kWh in ten minutes): no range separates a fault from load there.
- **The robust score held to 100**, as the satellite telemetry takes it. The bound touches no
  reading of a signal a forecast reads.
- **The statistics are fitted on 2016 to 2019**, the first years in which every signal has a
  full year, of the turbines read; 2020 to 2022 are held out by name. A turbine read as new in
  the held-out years is then scored in units the fleet's past gave, as an operator joining a
  turbine to a fleet would have them, and not in units from its own future.

## Consequences

- The counts a manifest states can be held against the audit's inventory.
- A few stray readings between a range's edge and the ordinary values stay (a hub at −33 °C,
  where ordinary readings begin near −7 °C): ranges tight enough to take them would be fitted
  to the data.
- The ten signals with no robust spread saturate at the bound (15 % of Kelmarsh's blade angle
  readings). No forecast reads them; a pretraining over the farms decides their score first,
  and a publication may then be made again, since the score is the publication's.
- The turbines' real first months lie on the fitting side, so a reading of them is scored in
  units that saw them.
- Three years, not four, are left to read turbines as new in.

## Alternatives considered

- *Publishing raw and leaving the sentinels to each reading*: every later consumer would have
  to know them, and an hourly mean would already hold them.
- *Removing only the exact sentinel values*: leaves the averages over a sensor failing part-way.
- *The standard score*: unbounded, so one faulty counter reads in the thousands.
- *The normal score of a quantile*: distorts the distances a forecast's error is measured in.
- *Fitting on 2016 to 2018*: six target temperatures of Penmanshiel would take their scale from
  eight months without a winter.

## Revisit when

- A pretraining reads the farms → the score of the signals with no robust spread.
- A publisher states valid ranges → those replace the ranges set here.
