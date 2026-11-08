# A pretext that forecasts each satellite channel's next hour from the hours before it

Arm A of the telemetry pretext hides every channel's last hour of a window joined with the 47
before it (ADR-0076) and asks the backbone to fill it in. This note sets how long that pretext
trains and checks that its forecast beats the forecasts that need no model, before any of its
pretrainings is read on a task.

## 2026-11-08 — declared before the run: the budget, and the forecast against the history's median on each mission

**Question.** Over the whole pretraining corpus, after how many epochs has every term of the
pretext stopped learning? At the epoch the run keeps, does the decoder's answer for the hidden
hour beat each channel's median over the hours before, on each mission's validation months? The
first sets the budget of the arm's pretrainings; the second decides whether they are ordered.

**Conditions.**

| | |
|---|---|
| Commit | the one that adds this section |
| Machine | Colab, one G4 (RTX PRO 6000 Blackwell); CUDA, fp32; tier M |
| Experiment | `experiments/backbone-history-esa-budget-m.toml`: seed 0, eight epochs at most, batches of 32 |
| Corpus | `esa_ad_raw` as published for pretraining (`durable/sha256/e926af17…`): both missions' first halves, 99 months train and the last three of each mission validate, the second halves cut out, the task corpus's statistics (ADR-0077); its 76,698 windows are the task corpus's windows of those months, to the token |
| Budget | `scripts/pretraining_budget_report.py --log <run.log>`, over a log that states each term to eight places: the cells' presence and count on Mission2 lie near a ten-thousandth and a hundred-thousandth, where the five places logged before quantised the cost the rule reads to whole steps |
| Forecast | `scripts/history_pretext_answers_report.py --result <result> --corpus <corpus> --side validation --out <dir>`: 1,500 validation windows drawn by seed 1, the hidden last hour of each channel that held a reading, Huber loss over its five values in normalised units; the decoder against the last reading carried forward and the median of the hourly means before it, on the same cells; 95 % intervals from 2,000 resamples of whole windows; each mission on its own |

**Rules.**

- *Budget.* A term stops learning at the first epoch whose cost none of the next three lowers by
  more than 1 % of itself; the budget is the latest such epoch over the reconstruction, the
  presence and the count. A term still learning at the cap leaves no budget: the cap was too
  short, and a longer run is agreed before any pretraining is ordered.
- *Forecast.* Read on the backbone the run keeps, the epoch of lowest validation reconstruction.
  A mission passes when the whole 95 % interval of decoder minus median of history lies below
  zero. Both missions must pass for the arm's pretrainings to be ordered. A mission that fails
  sends the pretext back to its target before any further run on a card: the decoder would
  then answer the hour's departure from the history's median rather than its value.
- The decoder against the last reading is reported beside it and decides nothing.

**Predictions**, from the trial's intervals at the edge nearer zero:

| | Predicted | Source |
|---|---|---|
| Mission1, decoder − median | at most −0.022 | the pilot over a tenth of the months, [−0.024; −0.022] |
| Mission2, decoder − median | below zero, short of −0.096 | the pilot over Mission2 alone, [−0.110; −0.096]; here the mission is a fifth of the windows, so a fifth of the steps |
| Budget | at most five epochs | the pilot over Mission2 alone still fell 0.5 % over its last two epochs, 3,288 steps in; an epoch here is about 2,260 steps |

**Trial, before this declaration.** Two pilots on the Mac (M1 Pro, MPS, fp32, tier S), the
recipe of the run but for the months read and the batch, and the forecast report run on each
in full:

| Pilot | Run | Mission | Decoder | Last reading | Median of history | Decoder − median |
|---|---|---|---|---|---|---|
| a tenth of both missions' months, 2 epochs, batches of 8 × 4 (`backbone-history-esa-s.toml`, commit `205df311`) | 2 × 1,250 s, 4.5 s a step | Mission1 | 0.098 | 0.154 | 0.121 | −0.023 [−0.024; −0.022] |
| | | Mission2 | 0.468 | 0.132 | 0.149 | +0.320 [+0.305; +0.333] |
| Mission2's 18 training months, 8 epochs, batches of 32 (`backbone-history-esa-m2-s.toml`, commit `259d157b`) | 8 × 137 s | Mission2 | 0.046 | 0.130 | 0.149 | −0.103 [−0.110; −0.096] |

- Over a tenth of the months, about two of them Mission2's, the decoder lost on Mission2 on the
  channels that barely move (18, 21 to 24 and 27): there the last reading missed by 0.006 to
  0.08 and the decoder by 0.2 to 1.5, and it missed them on its own training months too. Over
  all of Mission2's months it matched the last reading on channels 21 to 24, and beat both
  answers three to six times on channels 25 to 28; its training months read 0.055, so it did
  not learn them by heart. The first pilot's
  failure was too few of the mission's months, not a target that cannot carry a level.
- The validation loss the runs logged equals the report's decoder loss (0.168 against 0.167 on
  the first pilot), the report checks that only the last hour was hidden in every window, and
  it refuses a corpus other than the one the run read.
- A first reading of the first pilot pooled the missions and resampled single cells; it showed
  the decoder losing overall and was replaced by the per-mission, window-resampled report.

**Limitations.** One seed. The validation months are the last three of each first half, part
of the tasks' fitting side; no task's validation or test month is read. A channel that reads
only in the hidden hour still has a row in the window, which tells the decoder it reads there:
11 of 44,128 cells in the first pilot. The pilots' numbers are tier S and validation numbers.

## 2026-11-08 — measured: the forecast beats the history's median on both missions; eight epochs set no budget

**Conditions.** As declared, at commit `dfeaf86a`: order `durable/sha256/11a5a455…`, result
`durable/sha256/f3d16c86…`. The stored configuration was checked against the experiment file
before the reading: tier M, fp32, 2 × 64, 48 bins over 48 hours, the last hour the only mask, 8
epochs, batches of 32, seed 0, 72,282 training and 4,416 validation windows. On the G4 an epoch
took 456 s, a step 0.2 s, with the card 84 % busy; 3,811 s in all. The forecast was read on the
backbone of epoch 8, the lowest validation reconstruction, on the Mac (M1 Pro, MPS, fp32).

**Budget.** The values' cost, validation loss over the trivial predictor's, from the result:

| Epoch | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|
| Cost | 0.0870 | 0.0839 | 0.0805 | 0.0772 | 0.0760 | 0.0754 |
| Largest fall over the next three epochs | 11.3 % | 9.3 % | 6.3 % | 2.3 %, two left | 0.8 %, one left | — |

The presence and the count were not read: the budget is the latest epoch over the terms, so one
term still learning leaves none whatever the others do.

**Forecast**, 1,500 validation windows. *Same phase* was added after the run and decides
nothing: each channel's hour one period back, the period being the lag of 1 to 24 hours that
best repeated the window's own earlier hours, over at least six pairs of hours that both read.

| Mission | Windows | Cells | Decoder | Last reading | Median of history | Same phase | Decoder − median | Decoder − same phase |
|---|---|---|---|---|---|---|---|---|
| Mission1 | 747 | 35,856 | 0.0082 | 0.154 | 0.121 | 0.020 | −0.113 [−0.115; −0.112] | −0.012 [−0.015; −0.011] |
| Mission2 | 752 | 8,272 | 0.048 | 0.132 | 0.149 | 0.433 | −0.100 [−0.110; −0.091] | −0.384 [−0.442; −0.331] |

The period chosen most on Mission1 was 18 hours (38 % of cells), then 23 (12 %); 24 hours was
not among the five chosen most on either mission. On Mission2 it was one hour (47 %), and 33
cells had no period and took the latest hour read.

**Conclusions.**

- The forecast gate passes on both missions: arm A's pretrainings may be ordered once a budget
  is set.
- Mission1 beat its prediction (−0.113 against at most −0.022). Mission2 went just past its
  predicted edge: −0.100 against short of −0.096, with the interval across it. Its decoder did
  not gain on the pilot over Mission2 alone (0.048 against 0.046).
- No budget: the values still fell 6.3 % over the three epochs after the fifth, the last epoch
  the rule could judge. The cap was too short, and the prediction of at most five epochs failed.
- Mission1's channels repeat: one period back misses by a sixth of what the median misses by.
  Most of the decoder's gain over the median is that cycle, yet it misses by 40 % of what the
  cycle misses by. It forecasts past the repetition, and nothing in its gain needs a leak to
  explain it.
- On Mission2, with no period, same phase misses by more than the last reading and is no bar
  there.

**Limitations.** One seed, validation months. Same phase picks among 24 lags on the window's
own hours and does not adjust for level. A seasonal answer that carries the change since the
last cycle could narrow the gap on Mission1.
