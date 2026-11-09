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

## 2026-11-07 — the declared measure read the median's precision and the level's drift, corrected before its rerun

**What happened.** The run of the section above, on `ac65ca5c`, gave every arm a skill far below
zero on both missions. On the first, the `cross` skills ran from −37 to −79 and the `own` ones
from −1.7 to −4.2. On the second, they ran from −0.2 to −1.1. A quarter to over a third of the
chosen penalties lay at the grid's lower edge. No choice is taken from that run. Its numbers were
seen only to find the fault, and the trial before it had already shown one: a skill of −11.6.

**The fault, on one channel.** Channel 394 of the first mission, at 60-minute cells and 6 hours:

| Target | Scale | Forecast / median error, taught rows | `own`, read | `cross`, read |
| --- | --- | --- | --- | --- |
| mean | 0.0070 | 0.98 / 1 | 1.01 / 0.88 | 8.1 / 0.88 |
| least | 0.0039 | 1.54 / 1 | 2.17 / 1.05 | 51 / 1.05 |
| log of the count | 0.0025 | 1.57 / 1 | 19.6 / 1.89 | 762 / 1.89 |

The mean target sat at −0.20 over the taught months and at +0.58 over the read ones.
- The ridge forecast the hour's level, which drifts between months. The median of the hours
  before follows such a drift. A ridge fitted on earlier months extrapolates it, and with every
  channel's last hour as inputs, it does so along every drifting channel.
- The forecast and the bar were not nested. A ridge shrunk to nothing answered the taught
  months' mean, not the median, and a fit under squared loss could lose to the median under
  absolute loss on its own rows.
- Where the median is nearly exact, on stepped values and a fixed count of readings an hour,
  its error is nearly zero. As a scale, it weighed each error of a forecast hundreds of times.

**The corrected measure.** Everything else above stands: the arms, the sides, the rows read, the
skill over the read months, the rule, the block and what is read beside.

| | |
| --- | --- |
| Code | the commit that adds this section |
| Target | the hidden hour's departure from the bar, per target; the forecast is the bar plus the forecast departure, so a forecast of nothing is the bar |
| Own history | the cells with the level taken out: the five values less the bar's mean, the log of the count less the bar's; a bin before the history's first reading holds nothing, and a bin without a reading holds no count |
| Other channels | each one's last hour as its departure from its own median over the five hours before, beside whether it was held; nothing where either is missing |
| Fit | ridge on inputs standardised over the taught rows, the departures not centred; a constant column, penalised as every other; penalties 10⁻⁶ to 10³ times the rows and an infinite one, which forecasts nothing, so the grid holds the bar itself |
| Scale | the bar's mean error on the taught rows, at least a tenth of the target's mean absolute deviation there, as the channel-local detector floors a history's spread; a target without deviation is left out |
| Sides written | the errors on the taught rows beside the read ones |

**Checks that the measure is believable.** These are read before the rule, on the fitting side, and
they choose nothing. If either fails, the run is a configuration fault: `choice.json` then carries
no layout and no block.
- On the rows it was fitted on, no arm's skill is below −0.05. The grid holds a forecast of about
  nothing, so a fit that loses more than this fights the measure rather than lacks signal.
- On the rows read, no arm's skill is below −1, meaning no forecast errs twice as much as the bar.

The shares of penalties at the grid's least and at its infinite end are read beside. Many at the
infinite end mean that history adds nothing over the median for those targets. That is a
finding, not a fault.

**Trial.** On the first mission, channels 377, 394 and 407, over every fitting month (167 s):
- both checks hold: the least skill is 0.21 on the taught rows and 0.07 on the rows read;
- 3 % of the penalties lie at the least and 42 % at the infinite end;
- channel 394, the one in the table above, reads 0.00 on the taught rows and −0.04 on the rows
  read.

The trial read skills of 0.07 to 0.17 over the median. The ten-minute arms read lowest: hundreds
of columns under one penalty a target fit the taught rows better than they forecast the read
ones. This tilts the rule towards wider cells. It is a limit of the linear forecast, already
declared above, not of what the history holds. Three channels of 48 choose nothing, and the
rule stands as declared.

**Limitations added.**
- The penalty is still chosen under absolute error for a fit under squared loss.
- The inputs carry no level, so a forecast cannot read a channel acting otherwise near one level
  than near another.

## 2026-11-07 — measured: one-hour cells over 48 hours, with history adding a fifth over the median on the first mission and nothing on the second

**Question.** The one declared on top, under the corrected measure: what a window of each width
and length tells about its hidden hour beyond the median, which layout the rule chooses, and how
many months the held-out block takes.

**Conditions.**

| | |
| --- | --- |
| Code | `0b841319` for the run; the commit adding this section for `pool` |
| Machine | M1 Pro, CPU, float64; the first mission 2,554 s at 12.6 GB, the second 220 s |
| Commands | `data/report/l38/input/rerun.sh` (`measure.sh m1`, `m2`, then `read` on each); `telemetry_pretext_input_report.py pool --dirs run-m1 run-m2 --names m1 m2 --out pooled.json` |
| Cost | the `cost` run of `ac65ca5c`, whose code the correction did not touch |
| Read | 48 channels over 28 read months on the first mission, 11 over 7 on the second |

**Checks.** Both hold on both missions. The least skill on the taught rows is 0.23 on the first
mission and 0.28 on the second. On the rows read it is 0.02 and −0.16. Penalties at the grid's
least: 5 % and 6 %. At its infinite end, the median itself: 40 % and 26 %.

**Skill over the median, `cross` inputs, with 95 % intervals over the read months.**

| Cells | Hours | First mission | Second mission | Both, pooled |
| --- | --- | --- | --- | --- |
| 60 min | 6 | 0.185 [0.134; 0.266] | −0.004 [−0.086; 0.073] | 0.179 [0.131; 0.255] |
| 60 min | 12 | 0.188 [0.137; 0.270] | 0.001 [−0.028; 0.027] | 0.183 [0.134; 0.260] |
| 60 min | 24 | 0.211 [0.151; 0.304] | −0.002 [−0.031; 0.027] | 0.205 [0.148; 0.293] |
| **60 min** | **48** | **0.214 [0.153; 0.310]** | −0.013 [−0.040; 0.016] | **0.207 [0.149; 0.298]** |
| 30 min | 48 | 0.210 [0.150; 0.304] | −0.022 [−0.055; 0.011] | |
| 10 min | 6 | 0.098 [0.068; 0.143] | −0.073 [−0.177; 0.007] | |
| 10 min | 48 | 0.137 [0.096; 0.200] | −0.120 [−0.256; −0.013] | |
| 60 min, `own` | 48 | 0.198 [0.142; 0.287] | −0.035 [−0.090; 0.010] | |

The full grid is in each run's `read` log.

**Paired differences on the first mission (95 %).**

| | |
| --- | --- |
| 6 → 48 hours, 60 min | +0.019 to +0.045 |
| 24 → 48 hours, 60 min | +0.001 to +0.006 |
| 60 → 30 min, 48 hours | −0.007 to −0.002 |
| `own` → `cross`, 60 min, 48 hours | +0.010 to +0.024 |

**The rule's choice.**
- The first mission chooses 60 min over 48 hours. The second chooses 60 min over 6 hours, where
  the lengths' points lie within 1.4 points of each other.
- The declaration did not say how to combine the missions. One pretraining reads both, so the
  layout is the rule's over their union. In that union, a month of one mission is drawn apart
  from the same calendar month of the other. This combination is set after the missions' readings
  and is open about it. It could be foreseen, because 97 % of the bar's error is the first
  mission's.
- The union chooses **60 min over 48 hours**, 48 cells a channel, and the pretext's window is
  laid so.

**The block.** Three months, 2,208 hours, on both missions. The first mission's lag-12
autocorrelation is −0.09 against a band of 0.21, so no yearly structure is seen.

**Cost of the chosen layout.** On 48 channels, 2,304 cells a window.

| Device | Batch 1 | Batch 8 | Batch 32 |
| --- | --- | --- | --- |
| MPS | 0.17 s, 2.2 GB | 0.73 s, 12.1 GB | not tried: 11 GB for attention laid out whole |
| CPU | 0.10 s | 0.71 s | not tried |

**Predictions against the outcome.**
- *Longer history helps where a channel shows a cycle within 48 hours.* Held. Over the first
  mission's channels, the gain from 6 to 48 hours rises with the hourly autocorrelation at a day
  less that at half a day (Spearman 0.54, p < 10⁻⁴).
- *Finer cells help less than longer history.* Held. They help not at all here.
- *`cross` reads above `own`.* Held on both missions.
- *A 60-minute choice keeps the window at most 48 × 48 cells.* Held. A host fits batch 8, and a
  campaign at batch 32 goes to a GPU.

**Conclusions.**
1. The pretext's window is one-hour cells over 48 hours, every channel in it, with a three-month
   held-out block at the end of each mission's first half.
2. On the first mission, the hours before tell a fifth of the median's error away. Arm A has a
   margin to learn. It lies mostly in a few channels: the median channel reads 0.11, 69 % of
   channels read above zero, and the top tenth reads above 0.50.
3. On the second mission, history adds nothing over the median on the months read (0.29 on the
   taught ones). There, the pretext's case rests on arm B.

**Limitations.**
- 48 hours is the grid's edge. The gain flattens: +0.3, +2.3 and +0.3 points over its three
  doublings on the first mission, as a daily cycle met by 24 hours would make it. Whether longer windows add more is
  not measured.
- The rule has no least effect worth its cost. 24 and 48 hours differ by 0.3 points, which is
  significant over 61,000 windows and small in use.
- One window holds one hidden hour. At a step of its length, 48 hours give half the hidden hours
  an epoch that 24 hours give, at about twice the attention cost a cell. The window's step and
  its count of hidden hours are left to the window's decision.
- The width is a linear forecast's verdict. Ten-minute cells give it hundreds of columns that it
  fits better than it forecasts. What they add to an encoder within the hour is not measured.
