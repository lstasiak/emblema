# What a telemetry pretext's window should hold, read from the fitting months with no label

A pretext that hides a window's last hour and asks for it from the hours before needs two
choices: how many hours the window spans and how wide its cells are. This note fixes both
choices before any pretraining. It also fixes how many fitting months a pretraining holds out
as its own validation. Every number here comes from the fitting months; no label is read.

## 2026-11-07 — declared before the run: the forecast of the hidden hour from its history, by resolution and length

**Question.** How much does each channel's history tell about its next hour beyond the median of
the five hours before, the package's channel-local detector (ADR-0075)? From what length of
history and what width of cells does more stop helping? Do the channels' errors repeat after a
year?

**Conditions.**

| | |
| --- | --- |
| Code | the commit that adds this section; `scripts/telemetry_pretext_input_report.py` |
| Machine | M1 Pro, CPU, float64 |
| Tasks | `esa-mission1-events` (84 fitting months, 48 channels), `esa-mission2-events` (21, 11), fitting months only |
| Cells | the context report's published one-hour cells (`telemetry-context.md`), six ten-minute bins an hour |
| Grid | cells of 60, 30 and 10 minutes; windows of 6, 12, 24 and 48 hours, the hidden hour included |
| Target | the hidden hour's summary per channel: mean, least, largest and last reading, and the log of the count, as the detector reads it |
| Bar | the median of each target over the five hours before, with at least three of them holding a reading; the hours clipped as the targets are |
| Forecast | ridge per channel and target, over the history's cells (`own`), or over those and every other channel's last hour (`cross`); inputs and targets clipped at the teaching rows' 0.1 and 99.9 % quantiles, inputs standardised there; the intercept not penalised |
| Sides | the first two thirds of the fitting months by time fit the forecasts; the last quarter of those chooses each penalty among 10⁻⁵ to 10¹ times the rows fitted, kept for the refit on every teaching month; the last third of the fitting months is read |
| Rows read | channel-hours whose hidden hour holds a reading and whose history holds a reading in at least half the hours before under every length (3 of 5, 6 of 11, 12 of 23, 24 of 47), so every arm reads the same rows |
| Commands | `data/report/l38/input/measure.sh m1`, `m2`; `telemetry_pretext_input_report.py read --dir …`; `… cost --experiment experiments/backbone-cells-composition-m.toml --device mps` (and `cpu`) |

**Measure.** Each target's error over a read row is the absolute error over that target's
mean absolute error under the bar on the teaching rows. A target that never moves there is
left out, from the skill and from the monthly errors alike. An arm's skill is one less its forecast's summed error over the bar's, over every
channel, target and read row:

    skill = 1 - sum |forecast - target| / scale  /  sum |median - target| / scale

The interval comes from 10,000 resamples of the read months, paired across the arms.

**The rule.** The rule applies to the `cross` inputs, since the encoder sees every channel.

1. At each width, the chosen length is the shortest no longer length beats. Here "beats" means
   the paired difference's 2.5 % quantile lies above zero.
2. The chosen width is the widest one that no narrower width beats at its own chosen length.

Ties go to the wider cells and the shorter window, which are cheaper. Cost does not rule an arm
out. A layout too large for the host takes its pilot and its campaign cells to a GPU.

The rule makes up to five one-sided tests at 2.5 % each, with no correction for their
multiplicity. It therefore leans slightly towards longer windows and finer cells.

**The held-out block.**
- A pretraining holds out the last fitting months as its validation, so no month a task scores
  chooses its epoch.
- The block holds at least three months and at least 2,000 hours. A median over n normal
  readings has a standard error of about 1.25 / √n of their spread, which is under 3 % of it
  at 2,000 hours.
- The block takes twelve months if the errors repeat after a year. That holds if the median over
  the channels of the lag-12 autocorrelation of each channel's monthly median error under the bar
  lies above the white-noise band, 1.96 / √months. Each series is detrended first.
- The yearly lag is asked only of a mission with at least three years of fitting months, so
  the first mission only.

**Read beside, not ruled.**
- `own` against `cross`.
- Each channel's hourly autocorrelation of its clipped mean to 72 hours.
- The share of chosen penalties at an edge of their grid.
- The skill of the chosen arm itself. This is the margin a forecast has over the median, which
  is the bar a pretext must clear before a GPU.
- The forward and backward cost of the small shape over a window of each layout on MPS and the
  CPU, with the memory it held.

**Predictions.**
- No interval exists to predict the skill from: nothing forecast these hours before.
- Direction only. Longer history helps where a channel's hourly autocorrelation shows a cycle
  within 48 hours; finer cells help less than longer history; `cross` reads above `own`.
- What follows from each outcome:
  - a chosen arm reading near zero skill means a learned forecast has little over the median to
    learn, and the pretext's case rests on arm B, which asks which channels depart;
  - a 60-minute choice keeps the pretext's window at most 48 × 48 cells;
  - a finer choice takes the pilot to a GPU.

**Limitations declared.**
- The ridge is linear and fitted per channel. It shows what history tells about this target
  under this loss, not what an encoder's pretext would learn from it. A layout it does not
  need may still help an encoder.
- The target is the hour's summary under every width. What ten-minute cells add inside the
  hidden hour, its shape and its onset, is not measured.
- The fitting months hold events and are read without their labels. An event in the read months
  raises the error of the bar and of the forecasts alike.
- The second mission has 21 fitting months, seven of them read, so its intervals are wide.
