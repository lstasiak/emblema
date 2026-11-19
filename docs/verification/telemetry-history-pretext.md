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

## 2026-11-08 — declared before the run: the budget over sixteen epochs

**Question.** With the cap doubled, after how many epochs has every term of the pretext stopped
learning? That number sets the epochs of arm A's pretrainings.

**Conditions.** As in the first declaration, except for these:

| | |
|---|---|
| Commit | the one that adds this section |
| Experiment | `experiments/backbone-history-esa-budget-m.toml` capped at 16 epochs, about 36,000 steps; nothing else changed |
| Cost | about 2 h on the G4, about 21 units; a dropped session resumes from the last checkpoint |
| Forecast | the same report on the backbone the run keeps; reported, decides nothing: the gate passed above |

**Rules.** The budget rule above, unchanged, judges the reconstruction, the presence and the
count. A term still learning at the sixteenth epoch leaves no budget. Then no pretraining is
ordered until a longer cap, or another schedule, is agreed.

**Predictions.**

| | Predicted | Source |
|---|---|---|
| Budget | from 6 to 13 epochs | at eight epochs the values still fell 6.3 % over the three after the fifth, so fewer than six is unlikely. Here the rate at the eighth epoch is still about half its peak, against its floor before. Thirteen is the last epoch the rule can judge under this cap |

## 2026-11-08 — measured: the pretext's budget is thirteen epochs

**Conditions.** As declared, at commit `68622066`: order `durable/sha256/288e1da0…`, result
`durable/sha256/2d6aba00…`, not resumed. The stored configuration was checked against the
experiment file before the reading: 16 epochs, otherwise as the eight-epoch run. On the G4 an
epoch took 456 s, a step 0.2 s, with the card 86 % busy and 1.9 GB of its memory used; 7,510 s
in all. The log names no error and no NaN. Budget:
`scripts/pretraining_budget_report.py --log budget.log`. Forecast: read as before, on the
backbone of epoch 16, the lowest validation reconstruction.

**Budget.** Each term's validation cost, its loss over the trivial predictor's:

| Epoch | 5 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| Reconstruction | 0.0827 | 0.0715 | 0.0705 | 0.0701 | 0.0694 | **0.0689** | 0.0689 | 0.0685 | 0.0684 |
| Presence | 0.0002 | 0.0002 | 0.0003 | 0.0003 | 0.0002 | 0.0002 | 0.0002 | 0.0002 | 0.0002 |
| Count | **29.0** | 19.2 | 13.0 | 45.7 | 20.1 | 16.0 | 13.8 | 12.2 | 11.4 |

The rule stops the count at epoch 5, the presence at 6 and the reconstruction at 13: the budget
is 13 epochs, at the edge of the predicted 6 to 13.

**Forecast**, 1,500 validation windows; it decides nothing:

| Mission | Decoder at 8 → 16 epochs | Median of history | Same phase | Decoder − median | Decoder − same phase |
|---|---|---|---|---|---|
| Mission1 | 0.0082 → 0.0070 | 0.121 | 0.020 | −0.114 [−0.116; −0.113] | −0.013 [−0.016; −0.012] |
| Mission2 | 0.048 → 0.045 | 0.149 | 0.433 | −0.103 [−0.113; −0.094] | −0.387 [−0.445; −0.334] |

**Conclusions.**

- Arm A's pretrainings train for 13 epochs, by the declared rule.
- The reconstruction set the budget, at the last epoch this cap could judge; past it the values
  fell 0.6 % in three epochs.
- The count's cost is the decoder's loss over a guess that copies the channel's mean count from
  the window's visible hours. That guess misses by 4 × 10⁻⁶: a satellite channel's count barely
  moves from hour to hour. The decoder never came near it (4 × 10⁻⁵ to 1 × 10⁻³), so the ratio
  swings with noise. The rule's letter stopped the count at a trough in epoch 5 that it later fell
  below, and at the cap it was still falling, by 7 % over the last epoch: by the rule's intent
  the count never stopped within the cap. The budget stands on the letter, declared before the
  run, and on the term the pretext is for. The count weighs 4 to 5 % of the training loss from
  the fifth epoch on and at most 2.2 % of the validation loss from the third.
- The decoder gained a little over the eight further epochs on both missions. It stays past the
  repetition on Mission1.
- The two caps compare the schedule with the model: at epoch 8 the sixteen-epoch run, at half
  its peak rate, read 0.0745 where the eight-epoch run, annealed to its floor, read 0.0754. The
  annealing brought no gain, and the rate decays over the whole cap, so the plateau the rule read
  at epochs 13 to 16, at a tenth of the peak rate and under, is at least partly the schedule's.
  The budget is the budget under this schedule and this cap.

**Limitations.** One seed. Validation months. The budget lies at the cap's last judgeable epoch,
which was also the prediction's upper edge, so the result reads as "thirteen or more": a longer
cap could set it later. The budget rule reads a ratio to a trivial predictor, which is noise
where that predictor is near zero, as the count's is here; the next declaration of the rule
should judge such a term by its loss beside the values' loss instead. The presence's validation
loss rose by a third after epoch 6 while its training loss fell, a small head overfitting at
3 × 10⁻⁴ of the validation loss.

## 2026-11-08 — declared before the run: arm A's forecast error, read by the channel that departs most, against the departures with no model

**Questions.** (1) On the first mission's validation months, does the pretrained forecaster's
error for the scored hour, each channel over its typical miss and the window scored by its
largest channel, read above the departures with no model and no label (`channel_local`, the
leg's bar)? (2) Does it read above the answer that learnt nothing under the same reading, each
channel's last visible hour carried forward? Both must pass for the pretext to have shown it
reads more than the departures themselves. Beside, descriptively: the mean and the raw error
in place of the largest, the cells' farthest feature, the events found at 1 % of normal
windows, the second mission, and the forecaster's error on the months it read against the
months it never read.

**Conditions.**

| | |
|---|---|
| Commit | the one that adds this section; the knob at `d4ab754b`, the campaign and the pretrainings' files at `4af3c884` |
| Pretrainings | `experiments/backbone-history-esa-m.toml` and `-seed2.toml`: the budget run's recipe over 13 epochs, seeds 1 and 2, on the pretraining corpus `e926af17…`; tier M, one G4 each, about 1 h 40 min |
| Tasks | `esa-mission1-events` `8a6b1c18…` (61,363 fitting windows; 56 validation months, 41,088 windows, 62 events), `esa-mission2-events` `09368ab9…` (15,335 fitting windows; 14 validation months, 10,223 windows, 236 events); the task corpus `f1d0b0f2…` |
| Campaign | `campaigns/telemetry-history-pretext.toml`, once per pretraining and mission: the arm `pretext_error@error=largest,hidden=tail,reference=unit`, beside it `error=channel` and the raw error, the control `interpolation_error@error=largest,hidden=tail,reference=unit`; the backbone asked over each window joined with the 47 before it among those handed in (ADR-0076), the tail hidden once |
| Bars | the cells of `telemetry-mission1-local` and `telemetry-mission2-local` (`channel_local@reference=unit` on the first mission, `@departure=bins` on the second; `cells_knn@distance=farthest,reference=unit`), fitted on the same months and paired across campaigns |
| Platform | pretrainings on Colab G4; the first mission's pretext cells on the G4 (2.7 h each on the Mac, which scores 10.6 of its windows a second); every other cell and the trial on the Mac |
| Reading | `data/report/l38/reading/read.sh`: `campaign_pairs_report.py`, 10,000 resamples over months in two strata, seed 1, `--recall-at 0.01`; a window whose 47 hours before reach into a month the task never loaded is dropped from every side before the pairing |

**Rules.**

- *The knob, chosen on the fitting side before this declaration.* The arm's reading is the
  largest channel, from the trees' finding that the missed events lie in which channel departs
  (`telemetry-channel-local.md`). It stands unless, on the fitting side's later months under the
  budget run's backbone (seed 0, which no pretraining of the arm uses), the mean reads at least
  0.01 above it in area; then the mean is the arm. The pilot's number anchors the prediction and
  decides nothing else.
- *The reading.* Two readings, one family under Holm, each the arm less its control on the
  validation months, the pretrainings' cells pooled by the mean of their areas over the same
  resample, the interval over months: (1) less `channel_local`; (2) less the interpolation under
  the same knobs. T = 2·s/√2 over the two pretrainings, s the larger of 0.014 and their own
  spread; the control draws nothing. The least gain that counts is the detection tasks'
  registered 0.05; the floor's fixed part 0.01.
- *Truncated windows.* A validation window whose 47 hours before reach into a frozen month is
  read by no side, bars included: the arm would see less history there and the bars would not.
- *The second mission* is read descriptively, as the register has it: its bar reads 0.885 and
  its control spreads.
- *Read and unread months*, descriptively: `history_pretext_answers_report.py` on each
  pretraining, the training side against the validation side of the pretraining corpus. Where the
  decoder's loss on the months it read is under half its loss on the months it never read, the
  arm "pretrained without event windows" returns to the stage's table.

| Outcome | Condition | What follows |
|---|---|---|
| Above the departures | (1) at least +0.05 and T, interval above zero, under Holm; (2) the same | the pretext reads more than the departures: arm A's pretrainings are the leg's candidate, arm B is read as what its term adds to A |
| Above its no-model answer only | (2) passes, (1) within ±max(0.05, T) or below | the forecaster is a better per-channel reader than the last hour, but no better a detector than the departures with no model; arm B's term is the next lever, read against A |
| Around or below | (1) and (2) within ±max(0.05, T), or below | the forecast's sharpness does not reach the events; the pretext's target is talked through before any further card |
| Undecided | T above 0.05 | the pretrainings spread more than the least gain; whether more seeds are worth their cards is decided with the size read |

**Predictions**, from the pilot's intervals below at the edge nearer zero:

| | Predicted | Source |
|---|---|---|
| (1) arm − `channel_local`, area, Mission1 | between −0.02 and +0.07, under the least gain of 0.05 | the pilot, +0.014 [−0.022; +0.066] |
| (2) arm − interpolation, area, Mission1 | at least +0.045, the point near +0.15: passes on the point, the lower edge a hair under the least gain | the pilot, +0.156 [+0.045; +0.280] |
| Outcome | above its no-model answer only | the two rows above |
| Recall at 1 %, arm − `channel_local`, Mission1 | at least +0.02, the point near +0.17 | the pilot, +0.172 [+0.017; +0.315] |
| The mean beside the largest, Mission1 | area about 0.02 under the arm's, recall about 0.10 above | the pilot, −0.021 [−0.075; +0.100] and +0.103 [0.000; +0.237] |
| Mission2, arm − `channel_local`, area | within ±0.03 | the pilot, +0.006 [−0.022; +0.031] |
| Read against unread months | the decoder's loss on the read months not under half that on the unread | the pilot's arm read the unread months no worse, below |

**Power.** The paired difference (1) spreads 0.022 over resamples of the pilot's 36 months and
29 events, so about 0.015 over the validation months' 56 and 62, before the pretrainings' own
spread; T is 0.020 at the least. A true gain of 0.05 has its interval above zero with a
probability of 0.5 to 0.9 (the first of the two readings held at 97.5 % under Holm) and reads
at or above the least gain about half the time, so the
reading can miss a gain at the least gain; a gain of 0.10 reads above zero with a probability
over 0.99. The pilot's +0.014 predicts a miss for a reason other than power. The gain predicted
for (2) stands two and a half of its pilot spreads (0.060) from zero.

**Pilot on the fitting side, before this declaration.** The budget run's backbone (seed 0, 16
epochs), `scripts/history_pretext_reading_pilot.py` on the Mac (M1 Pro, MPS, fp32): each
mission's fitting months split at a date, the months before it fitting the scales and the
alarm, those from it scored; intervals from the script's `read` over its CSVs, 2,000 resamples
of the scored months, seed 1.

| Mission | Fit | Scored | Reading | Area | Recall at 1 % |
|---|---|---|---|---|---|
| Mission1 | 35,020 windows to 2003-12 | 26,124 from 2004-01: 36 months, 29 events | arm, largest | 0.184 [0.061; 0.333] | 0.759 [0.604; 0.899] |
| | | | mean | 0.163 | 0.862 |
| | | | raw | 0.172 | 0.724 |
| | | | interpolation, largest | 0.027 [0.004; 0.120] | 0.690 |
| | | | `channel_local` | 0.170 [0.060; 0.300] | 0.586 [0.424; 0.748] |
| Mission2 | 10,199 windows to 2001-02 | 5,136 from 2001-03: 7 months, 66 events | arm, largest | 0.931 [0.892; 0.971] | 0.970 |
| | | | mean | 0.919 | 0.970 |
| | | | raw | 0.898 | 0.939 |
| | | | interpolation, largest | 0.942 [0.897; 0.983] | 0.970 |
| | | | `channel_local@departure=bins` | 0.925 [0.886; 0.967] | 1.000 |

- *The knob.* The mean reads 0.021 under the largest on Mission1 and 0.012 under it on
  Mission2: the largest stands. It finds more of Mission1's events at 1 % of false alarms (0.86
  against 0.76), so the recall is read beside the area.
- *The control.* On Mission1 the last hour carried forward, read by its largest channel, reads
  0.027 in area, a sixth of the bar's (a random order reads 0.0015), yet finds 0.69 of the
  events at 1 % of false alarms, as many as the raw pretext: it reaches most events by some
  hour of theirs and ranks the rest of their hours low. An answer that carries the last hour
  forward departs where a channel turns, at an event's onset as at the turns of the channels'
  18-hour cycle, not inside an event of hours (their median span is 8 hours, the longest 357).
  Question (2) therefore asks whether the forecaster reads past the cycle more than whether it
  finds the events; (1) is the reading that counts.
- *Unread months.* The arm read the three months the pretraining never read (2006-10 to 12,
  four events) at 0.70, the mean at 0.69 and `channel_local` at 0.51; on Mission2 the arm fell
  from 0.931 to 0.890 there and the bar from 0.925 to 0.881. Over four events this says little
  either way; the declared reading of read against unread months decides.
- *Cost.* A Mission1 window is 2,304 cells against 528 on Mission2, and the attention's cost
  grows with their square: the Mac scores 10.6 of them a second against 146. The reading now
  batches windows by a budget of cells, not by count: at 64 windows a batch the attention's
  scores over Mission1 took the whole memory.

**Trial, before this declaration.** The campaign file defined, ordered, run and accepted
through the campaign's own commands on the Mac, on the second mission under its pilot backbone
of tier S (`86c3edbb…`, which no reading uses): four cells in 622 s, the pretext's about 190 s
each, 1.9 GB. The reading script then ran whole on that campaign and the channel-local
campaign of 2026-11-06: 235 windows in the 47 hours after an unloaded month dropped from every
side, seven pairs, the area and the recall tables written, no number read.

**Limitations declared.** Two pretrainings. The detectors over the encoder's states refuse a
backbone over a history, so the arm is read by its pretext's error alone and no "states against
their initialisation" reading exists for it. The channels' scales are fitted on months the
pretraining read. The validation months have been read before; the test months stay frozen.
The pilot read the budget run's backbone of 16 epochs, where the pretrainings stop at 13, on
fitting months the pretraining read in part, with months resampled in one stratum where the
reading uses two.

## 2026-11-09 — measured: the forecast error reads the events no better than the departures with no model, and no better than its own no-model answer

**Conditions.** As declared, at commit `5b15beed`. Pretrainings: backbones `5a0ffd31…` (seed 1,
result `durable/sha256/aafb98f8…`, weights kept at epoch 13 `4f89bca9…`) and `03a53a6d…` (seed
2, result `3af4817a…`, weights `1214d12a…`), side by side on one Colab G4: 11,342 s and 11,397 s,
862 s an epoch each, twice the 456 s of one run alone; 3.8 GB of its memory. The stored configuration of both results matches the experiment files (48 bins
over 48 joined hours, 2 × 64, 13 epochs, batches of 32, fp32, the hour hidden at a rate of
one); the validation reconstruction at epoch 13 read 0.0156 and 0.0150, 0.071 and 0.068 of the
trivial predictor's. Campaigns `telemetry-history-pretext`: first mission `fdd31528…` and
`ec4cead7…`, second `4e7db42a…` and `48755751…`; the first mission's eight cells on the G4 in
eight processes (a pretext cell 855–877 s, the interpolation 273 s, 1,779 s in all), the second
mission's on the Mac (631 s and 655 s an order, 1.8 GB). Reading: `read.sh` as declared; 1,128
windows of the first mission and 235 of the second dropped in the 47 hours after an unloaded
month (24 and 5 boundaries); 39,960 windows, 62 events over 4,618 of them, 35,342 normal
windows remain on the first mission, so a random order reads 0.0018 in area. The kept detectors record `max` and `mean` as their reductions; the interpolation's cells of
the two campaigns agree to the bit; `channel_local` reads 0.176 here, 0.179 over every window
on 2026-11-06.

**Mission1, validation months.** Area: the mean of the two pretrainings' areas; in brackets
seed 1, seed 2.

| Candidate | Area | Recall at 1 % |
|---|---|---|
| arm, largest channel | 0.158 (0.138, 0.178) | 0.589 (0.565, 0.613) |
| mean over channels | 0.178 (0.152, 0.203) | 0.621 (0.548, 0.694) |
| raw error | 0.206 (0.198, 0.215) | 0.669 (0.645, 0.694) |
| interpolation, largest | 0.122 | 0.548 |
| `channel_local` | 0.176 | 0.597 |
| `cells_knn` farthest | 0.157 | 0.419 |

**The declared readings**, 10,000 resamples of months, one Holm family; T = 0.040 (the
pretrainings' own spread 0.028 over the floor 0.014):

| | Area | Interval | p | Recall at 1 % |
|---|---|---|---|---|
| (1) arm − `channel_local` | −0.018 | [−0.074; +0.039] | 0.67 | −0.008 [−0.112; +0.085] |
| (2) arm − interpolation | +0.036 | [−0.033; +0.112] | 0.27 | +0.040 [−0.090; +0.154] |

Both lie within ±0.05 and within ±T: the outcome is **around or below**. Beside: the mean reads
+0.020 [+0.001; +0.049] over the largest and the raw error +0.048 [+0.007; +0.098]; the arm
+0.001 [−0.090; +0.076] over the farthest feature; the interpolation −0.054 [−0.104; −0.015]
under `channel_local`.

**Mission2**, 14 validation months (232 events after the drop), descriptively: arm 0.889 (0.883, 0.895), mean 0.898, raw 0.829, interpolation 0.928, `channel_local` 0.886, farthest
0.624; recall at 1 % 0.98–1.00 for every candidate but the farthest (0.86). Arm −
`channel_local` +0.003 [−0.012; +0.023]; arm − interpolation −0.039 [−0.054; −0.017]: the last
reading carried forward is the best reader of this mission's events.

**Read against unread months**, `history_pretext_answers_report.py`, 1,500 windows a side,
Huber loss of the decoder on the hidden hour: Mission1 0.0094 and 0.0091 on the months the
pretraining read against 0.0073 and 0.0070 on the months it never read; Mission2 0.038 and
0.040 against 0.047 and 0.045. Neither ratio falls under one half, so the arm "pretrained
without event windows" does not return. The decoder beats the median of the history by 0.114
on Mission1 and 0.10 on Mission2 under both pretrainings, as the forecast gate found.

**Predictions against the outcome.**

| | Predicted | Read |
|---|---|---|
| (1) area | −0.02 to +0.07 | −0.018 ✓ |
| (2) area | at least +0.045, the point near +0.15 | +0.036 ✗ |
| Outcome | above its no-model answer only | around or below ✗ |
| Recall, arm − `channel_local` | at least +0.02 | −0.008 ✗ |
| The mean beside the largest | area 0.02 under, recall 0.10 over | area 0.020 over, recall 0.032 over ✗ |
| Mission2, arm − `channel_local` | within ±0.03 | +0.003 ✓ |
| Read against unread | not under one half | 1.3 and 0.8 to 0.9 ✓ |

**Post hoc, from the reading's exported predictions.** For each event, its windows in time
order; each window's rank among the mission's 39,960 (0 the highest score), and the share of a
detector's top 1 % that falls in the event:

| Detector | Median rank of the event's first hour | Median rank of its later hours | Median share of the event's hours in the top 1 % |
|---|---|---|---|
| arm, seed 1 / seed 2 | 0.12 / 0.19 | 0.43 / 0.48 | 0.02 / 0.03 |
| raw error, seed 2 | 0.07 | 0.45 | 0.03 |
| interpolation | 0.12 | 0.48 | 0.01 |
| `channel_local` | 0.09 | 0.50 | 0.01 |
| farthest | 0.16 | 0.42 | 0.00 |

Every detector referenced to the window's own past ranks the first hour of an event high and
its later hours where chance would, since the history absorbs the event. The events last 11
hours at the median, 113 at the third quartile, 571 at most, and the area spreads each event's
weight over all its hours, so it reads their tails, not their alarms: a detector that ranked
every event's first hour at the top and nothing else would read 0.30 here, and these read 0.12 to 0.21 because they also miss onsets, 22 to 26 of
the 62 within their top 1 %. The pilot saw the tail in the control only; it belongs to the
class. Of the 62 events, 27 are the benchmark's rare nominal events, counted as positives as it
counts them; 33 are local to a few channels and 29 global. The detectors referenced to a channel's
history find 21 to 26 of the 29 global events at 1 % and 12 to 17 of the 33 local ones (the raw
error 17, the arm 13, `channel_local` 12); the farthest feature 18 and 8. The two pretrainings' areas differ by 0.040 under the largest and 0.052
under the mean, as much as the least gain. The interpolation read 0.027 on the pilot's 36
fitting months and 0.122 here, just above the upper edge of its pilot interval [0.004; 0.120].

**Conclusions.**

- The forecaster's error, read by its largest channel, reads the first mission's events no better
  than the departures with no model (−0.018 [−0.074; +0.039]) and not demonstrably better than
  the last hour carried forward (+0.036 [−0.033; +0.112]). Arm A, as declared, adds nothing to
  the detectors with no model on either mission.
- The forecast itself is good: 0.07 of the trivial predictor's loss, 17 times better than the
  history's median on the first mission and 3 times on the second, as sharp on months it never
  read. A better forecast of a channel's hour is
  not a better reader of these events.
- The registered area reads the hours after an alarm, which every detector referenced to its
  own history ranks at chance; a detector that ranked every onset first and nothing else would
  read 0.30 here, and these read 0.12 to 0.21 because they also miss onsets. For detectors of
  this class the recall at a false-alarm budget is the reading that follows their mechanism, and under it the arm reads 0.59 against the
  bar's 0.60 and its raw error 0.67, all within one interval.
- The knob chosen on the fitting side (the largest channel) reversed on the validation months,
  and two pretrainings spread as much as the least gain: a pilot over 29 events chooses no
  knob, and the spread between two pretrainings (0.040 under the largest, 0.052 under the mean)
  is as large as the least gain.
- What stays out of reach are the local events: every detector here referenced to a channel's
  history finds 12 to 17 of the 33 (the farthest feature 8), and the label-taught trees' advantage on them comes from recurring
  event classes the labels name, which no pretext learns from the telemetry alone.

**Limitations.** Two pretrainings, one reading each, validation months. The channels' scales
were fitted on months the pretraining read. The onset-only oracle and the tail ranks are post hoc, on
these detectors; the recall at 1 % was declared
beside, not first. The per-month reference (`reference=unit`) shares one spread across a month,
so a month holding a 571-hour event compresses its own scores; every candidate bears it alike,
and its size is not measured. The rare nominal events are counted as the benchmark counts
them. The Mission2 reading is descriptive. Hourly cells were chosen for the forecast's
residual, not for the events' length; whether the local events the detectors miss survive the
hour's summary is not read here.

## 2026-11-15 — declared before the run: the forecast against a ridge per channel on months the pretraining never read, the published forecaster beside

**Question.** On the first mission's months the pretraining never read, does the decoder's
loss for the hidden hour lie under a ridge per channel over the same 47 hours by at least the
least gain the register sets, a ratio of 0.80? Beside, descriptively: the read and held months,
the second mission, the answers with no model, a ridge that also reads the window's other
channels, and Chronos-2 zero-shot on the hour's mean.

**Conditions.**

| | |
|---|---|
| Commit | the one that adds this section; the instrument at `d5e474f7`, the register's task and row at `3da677bd` |
| Machine | Mac (M1 Pro, MPS, fp32); Chronos-2 in its own environment (`chronos-forecasting` 2.3.2, torch 2.15, MPS) |
| Pretrainings | arm A of 2026-11-09: `5a0ffd31…` (seed 1, result `aafb98f8…`, weights `4f89bca9…`) and `03a53a6d…` (seed 2, result `3af4817a…`, weights `1214d12a…`); 2 × 64, 48 bins over 48 joined hours, 13 epochs, Huber knee one; both `ready` and in the store on 2026-11-15 |
| Months | read: the training months of the pretraining corpus `e926af17…` (81 of Mission1, 18 of Mission2); held: its validation months, the last three of each first half; unread: the second-half months the detection tasks validate on, from `f1d0b0f2…` (Mission1 2007-01 to 2013-12 less 28 frozen months, 56 left; Mission2 2001-10 to 2003-06 less 7, 14 left), the frozen months listed in `data/report/l51/frozen-units.txt` and left unread |
| Draw | 1,500 windows per mission and set, seed 1, among windows whose 47 hours hold no hour in which no channel read (188 training windows and no held one were such in the trial) |
| Ridges | `ChannelRidge` in `history_pretext_answers_report.py`: 424 coefficients per channel (every feature of the 47 hours and an intercept), fitted on up to 20,000 training windows per mission drawn by seed 2 outside the 1,500 read (the second mission has fewer, so all of its remaining ones), the penalty chosen among 0.01 to 10,000 by decades between halves of the fitting months on their rows unclipped, each feature it answers clipped to the fit's range; fitted once, on seed 1's run, read by both pretrainings. Beside it the ridge over the window: the same features and the hourly means of the window's other channels (2,633 coefficients on the first mission, 894 on the second), fitted the same way on the same rows; a window lacking a channel gives it zeros, as the layout gives an absent hour |
| Published forecaster | `chronos2_answers.py`: `amazon/chronos-2` (119M parameters), the channels of a window handed together, its median for the 48th hour's mean from the 47 hourly means in the corpus's units, on the decoder's cells |
| Reading | `history_forecast_reading.py`: 10,000 resamples of whole windows, seed 1, least gain 0.80, floor 0.02, Huber knee 1.0; `data/report/l51/run.sh` runs the whole |

**Rules.**

- *The verdict* is the arm's ratio on the first mission's unread months, decoder over ridge in
  mean Huber loss over the five values on the same cells, the arm the mean of the two
  pretrainings; T = 2·√(SE² + (s²/2)/2), s the larger of 0.02 and the two ratios' spread. An
  interval covering one is less than about T either way; an upper edge under one but above 0.80
  real and small; at or under 0.80 at least the least gain; a lower edge above one harm.
- *Everything else is descriptive*: the read and held months, the second mission, the answers
  with no model, the ridge over the window, the median over windows of decoder over ridge, the
  mean's squared error, and Chronos-2 by that error alone. The ridge over the window says
  whether a gain over the bar is the other channels' hours; the median over windows whether the
  typical window agrees with a mean a few windows can carry.

| Outcome | What follows |
|---|---|
| At least the least gain | the forecaster is the leg's candidate; the onboarding of a third mission is read next |
| Real and small, noise, harm | the pretext is no foundation forecaster on this telemetry; the leg stays the channel-local detector with its few-shot readouts, and no third mission is read |

**Predictions**, from the trial's intervals at the edge nearer one, the trial's ridges fitted
on a tenth of the windows the run fits on:

| | Predicted | Source |
|---|---|---|
| Mission1, unread, decoder over ridge | at most 0.96; the outcome real and small or noise, not the least gain | the trial: read months 0.92 [0.80; 0.96], held 0.69 [0.55; 0.86] with two of 100 windows carrying 0.15 of it |
| Mission1, unread, median over windows of decoder over ridge | 0.80 to 0.96 | the trial: 0.85 [0.81; 0.90] read, 0.88 [0.82; 0.96] held |
| Mission1, unread, decoder over the ridge over the window | at most 0.97, nearer one than against the bar | the trial: 0.94 [0.83; 0.97] read, 0.83 [0.77; 0.89] held |
| Mission2, unread, decoder over ridge | at most 1.02 | the trial: held 0.95 [0.79; 1.02], read 0.69 [0.57; 0.83] |
| Mission1, unread, decoder − median of history, Huber | at most −0.10 | 2026-11-09, −0.114 and −0.114 on the held months |
| Ridge on the held months, Mission1 | under 0.0099 | the trial's ridge on 2,000 windows; ten times the windows fit it better, not worse |
| Decoder over Chronos-2, squared error of the mean, Mission1 unread | 1.0 to 2.5: the published forecaster no worse | the trial: 1.06 [1.03; 1.26] read, 1.65 [1.07; 2.48] held |

**Power.** The first mission's ratio spread 0.041 (its resampling standard deviation) over 100
read windows and 0.081 over 100 held windows; over 1,500 windows about 0.011 to 0.021, so T,
with s floored at 0.02, is 0.03 to 0.05, and the least gain lies four to seven T under one. A
true ratio near one reads noise and closes the direction; a true 0.75 reads at least the least
gain in most worlds, a true 0.80 about half the time.

**Trial, before this declaration.** `data/report/l51/trial.sh` on seed 1: the exact commands
on 100 windows per mission of the read months, both ridges fitted on 2,000 training windows per
mission outside them, then the held months under those ridges; Chronos-2 on every cell; the
reading script whole on both sets. The read and held months were read on 2026-11-09, so their
numbers are shown. The trial ran twice: the first ridge, unclipped and on a narrower grid,
answered one held window's channel at −9.3 against −0.13, and two such windows carried the held
months' ratio from 0.82 to 0.58; the ridge below is the one declared.

Huber loss over the five values, 100 windows a set and mission:

| Set | Mission | Decoder | Last reading | Median of history | Same phase | Ridge | Ridge over the window |
|---|---|---|---|---|---|---|---|
| read | Mission1 | 0.0158 | 0.1633 | 0.1271 | 0.0275 | 0.0172 | 0.0168 |
| read | Mission2 | 0.0172 | 0.0998 | 0.2046 | 0.3518 | 0.0248 | 0.0263 |
| held | Mission1 | 0.0069 | 0.1558 | 0.1211 | 0.0191 | 0.0099 | 0.0082 |
| held | Mission2 | 0.0522 | 0.1509 | 0.1363 | 0.4057 | 0.0548 | 0.0535 |

The ratios read, 95 % intervals from 2,000 resamples of whole windows:

| Set | Mission | Decoder over ridge | Median over windows | Decoder over the ridge over the window | Decoder over Chronos-2, squared error of the mean |
|---|---|---|---|---|---|
| read | Mission1 | 0.92 [0.80; 0.96] | 0.85 [0.81; 0.90] | 0.94 [0.83; 0.97] | 1.06 [1.03; 1.26] |
| read | Mission2 | 0.69 [0.57; 0.83] | 0.77 [0.72; 0.80] | 0.65 [0.53; 0.81] | 0.91 [0.56; 1.64] |
| held | Mission1 | 0.69 [0.55; 0.86] | 0.88 [0.82; 0.96] | 0.83 [0.77; 0.89] | 1.65 [1.07; 2.48] |
| held | Mission2 | 0.95 [0.79; 1.02] | 1.01 [0.92; 1.06] | 0.98 [0.84; 1.04] | 2.65 [1.91; 3.97] |

- The halves were 48 and 51 months. The bar chose a penalty of 10 for 35 of the 59 channels
  and the grid's edges for three (0.01 for channel 408, 10,000 for 426 and 427, whose weights
  beyond the intercept stay near 0.1 there, so a wider grid would move them little); the ridge
  over the window chose 100 for 40. The second mission has 12,933 training windows outside the
  ones read, so the run fits on all of them. The bar is the strongest answer with no
  pretraining by far: a ninth of the last reading's miss and 0.6 of the cycle's on the first
  mission's read months. On the held months two windows, a channel falling to −1.5 and back,
  still carry 0.15 of the ratio after the clipping, which the median over windows does not
  follow; the ridge over the window closes about half the decoder's gap there. The decoder's
  held-month loss (0.0069 [0.0065; 0.0073]) matches 2026-11-09's 0.0073 on another draw.
- Chronos-2 loaded in 2 s and answered 5,900 cells in 5 s; a context with a missing hour is
  answered, one with no hour read is refused, and no cell of the reading is such. On the mean's
  squared error it matches the decoder on the months the decoder trained on and beats it on the
  held months of both missions, from the same 47 hourly means and no training on this telemetry.
- Cost: loading and joining a publication's windows about 100 s; scoring 200 windows about
  15 s; both ridges fitted on 4,000 windows about two minutes, the ridge over the window holding 11 to 12 GB at its peak over two runs. The whole run, six readings of 3,000 windows, the fit on 33,000 windows,
  three passes of Chronos-2 and the tables, stays under two hours.

**Limitations declared.** Two pretrainings. The bar reads one channel at a time where the
backbone reads the window together; the ridge over the window is descriptive. Chronos-2 scales
every series inside besides the corpus's scaling and is read on the mean alone. The unread
months are the detection tasks' validation months, read before by detectors though by no
pretraining; the test months stay frozen. One loss, normalised units. The ridges' 20,000
windows per mission are a size chosen for cost. The second mission has eleven channels and
fourteen months left to read.

## 2026-11-15 — measured: on the later years the forecast reads level with the ridge by the mean and behind it on the ordinary window

**Question.** As declared above: on the first mission's months the pretraining never read, does
the decoder's Huber loss lie under the ridge per channel by at least a ratio of 0.80?

**Conditions.** Instrument `d5e474f7`, declaration `17195580`, `data/report/l51/run.sh` on the
Mac (M1 Pro, MPS, fp32) on 2026-11-15, about 50 minutes in all: six readings of 288 to 553 s,
Chronos-2 69 to 75 s a set, the tables as declared (least gain 0.80, floor 0.02, 10,000
resamples). Draws of 1,500 windows per mission and set: 188 training windows dropped for an hour
no channel read, none of the held, 1,363 of the unread; the 35 frozen test months left unread.
The ridges fitted on 20,000 Mission1 and all 11,533 remaining Mission2 training windows
(1,086,575 cells); the bar chose a penalty of 1 for 30 channels and 10 for 14, three at the
grid's upper edge of 10,000 (423, 424, 427); the ridge over the window chose 100 for 30.

**The verdict and the ratios**, the arm's ratio decoder over ridge in mean Huber loss over the
five values, its interval ± T with T = 2·√(SE² + s²/4), the declared (s²/2)/2 written out:

| Set | Mission | Seed 1 | Seed 2 | Arm [± T] | Level |
|---|---|---|---|---|---|
| read | Mission1 | 0.89 | 0.86 | 0.87 [0.82; 0.92] | real and small |
| held | Mission1 | 0.86 | 0.83 | 0.84 [0.81; 0.88] | real and small |
| **unread** | **Mission1** | 0.99 | 1.00 | **1.00 [0.94; 1.06]** | **noise** |
| read | Mission2 | 0.83 | 0.83 | 0.83 [0.77; 0.89] | descriptive |
| held | Mission2 | 0.92 | 0.87 | 0.89 [0.84; 0.94] | descriptive |
| unread | Mission2 | 0.95 | 0.90 | 0.92 [0.88; 0.97] | descriptive |

The declared rule reads noise on the first mission's unread months: the pretext is no
foundation forecaster on this telemetry, and no third mission is read.

Mean losses, Huber over the five values (decoder seed 1 / seed 2; the other answers have no
seed):

| Set | Mission | Decoder | Last reading | Median of history | Same phase | Ridge | Ridge over the window |
|---|---|---|---|---|---|---|---|
| read | Mission1 | 0.0079 / 0.0076 | 0.153 | 0.135 | 0.026 | 0.0089 | 0.0091 |
| held | Mission1 | 0.0074 / 0.0072 | 0.154 | 0.121 | 0.022 | 0.0087 | 0.0081 |
| unread | Mission1 | 2.77 / 2.79 | 0.67 | 3.90 | 0.53 | 2.79 | 2.80 |
| read | Mission2 | 0.043 / 0.043 | 0.130 | 0.181 | 0.365 | 0.052 | 0.054 |
| held | Mission2 | 0.045 / 0.042 | 0.130 | 0.147 | 0.452 | 0.049 | 0.051 |
| unread | Mission2 | 0.048 / 0.046 | 0.149 | 0.159 | 2.5·10⁷ | 0.051 | 0.053 |

Descriptively, over every window of a set. The median over windows of decoder over ridge reads
0.88 / 0.87 (seed 1 / seed 2) on the read months and 0.90 / 0.88 on the held, but **1.96 [1.91;
2.03] / 2.95 [2.85; 3.03]** on the unread. The decoder over the ridge over the window reads 0.85
read, 0.90 held and 1.00 unread. On the mean's squared error the decoder reads 1.68 [1.54; 1.82]
/ 1.33 [1.23; 1.43] times Chronos-2 on the held months and the ridge 3.09 [2.35; 4.33] times; on
the read months the decoder reads 0.94 / 0.83 and the ridge 0.46 times Chronos-2, with upper
edges at 8.6 and 3.8; on the unread months both read 17 times Chronos-2. On the second mission's
unread months the decoder reads 0.92 of the ridge. There Chronos-2's squared error is 0.0130
against the decoder's 0.0125 once eight cells of channel 432 are set aside: their history holds a
reading of 2·10⁹ in the normalised units, which Chronos-2 answers in the millions (its mean over
every cell 1.6·10⁹) and the same-phase answer meets too (2.5·10⁷).

**The unread months read, post hoc.** The first mission's unread losses are carried by a few
cells: 706 of 71,958 (1 %) with a Huber loss above one carry 99 % of the decoder's loss and
98.5 % of the ridge's, mostly (96 %) on channels 412 to 414, whose values reach 26,000 in the
normalised units, before the bound of 100 the telemetry now takes. There every answer, Chronos-2
included, misses alike, so the ratio of means is one whatever the skill. Laid out by the window's
largest value (history or target), the decoder as seed 1 / seed 2:

| Window's largest value | Windows | Decoder | Ridge | Ridge over the window | Last reading | Same phase | Chronos-2, squared error of the mean, against the decoder's |
|---|---|---|---|---|---|---|---|
| held months, all (≤ 2) | 1,500 | 0.0074 / 0.0072 | 0.0087 | 0.0081 | 0.154 | 0.022 | 0.0008 against 0.0014 / 0.0011 |
| unread ≤ 2 | 1,003 | **0.0201 / 0.0302** | 0.0098 | 0.0118 | 0.162 | 0.021 | 0.0007 against 0.0124 / 0.0254 |
| unread 2 to 5 | 357 | 0.020 / 0.029 | 0.014 | 0.015 | 0.162 | 0.018 | 0.0006 against 0.015 / 0.025 |
| unread 5 to 10 | 119 | 0.28 / 0.40 | 0.66 | 0.65 | 0.16 | 0.025 | 0.0007 against 1.10 / 1.30 |
| unread 10 to 100 | 14 | 1.6 / 1.9 | 2.1 | 2.1 | 0.29 | 0.28 | 3.4 against 23 / 26 |
| unread above 100 | 7 | 635 | 634 | 634 | 120 | 120 | 8·10⁵ against 1.4·10⁷ |

On the 1,003 windows whose every value lies where the training months lay, the answers with no
model, the ridges and Chronos-2 read as on the held months. The decoder reads 2.7 to 4.2 times
its held-month loss, and 2.2 / 3.3 times the ridge on the median of those windows. On the 119
windows of larger excursions the ridges fail (0.66) and the decoder less (0.28 / 0.40). Over the
years the decoder's loss on the 1,003 windows grows from 0.013 / 0.019 (2007) to 0.024 / 0.036
(2012 to 2013) while the ridge's grows from 0.0084 to 0.011. It grows mildly with the window's
level drift from the held months (0.018 to 0.023 from the least to the most drifted quarter), so
drift of the levels explains little. Fourteen channels report 176 readings an hour on the unread
months against 40 on the held, and the decoder's loss grows most on them (4.8 / 11.6 times).
Re-reading the first 200 unread windows with every value within 2, their count, offset and
recency features clipped to each channel's training range, leaves the decoder's loss at 0.0128
either way; the training months already hold hours of thousands of readings. The change in
sampling is therefore not the cause through those features (`data/report/l51/clipped_features.py`
and `sampling.py`, kept with the run's files). No hour of either set is read by no channel; four
unread windows hold seven silent channels for 2 to 44 hours, and no held window does.

**Predictions against the outcome.**

| Predicted | Read |
|---|---|
| Mission1 unread ratio at most 0.96, real and small or noise | 1.00: the level as predicted, the ratio above the predicted edge |
| Median over windows 0.80 to 0.96 | 1.96 / 2.95: missed |
| Decoder over the ridge over the window at most 0.97 | 1.00: missed |
| Mission2 unread at most 1.02 | 0.92: held |
| Decoder − median of history at most −0.10 | −1.1: held, by the extreme cells alone |
| Ridge on the held months under 0.0099 | 0.0087: held |
| Decoder over Chronos-2 on the mean, 1.0 to 2.5 | 17 over every window, 18 to 36 on the ordinary ones: missed |

The trial's held months lie next to the training months and did not stand for the years later;
the reading of 2026-11-09 called "read against unread" had compared only those adjacent months.

**Conclusions.**

1. The declared rule reads noise, and the direction closes: the forecasting pretext on this
   input is not the leg's foundation forecaster; the onboarding of a third mission is not read.
2. The ratio of one is mechanical: 1 % of cells at values the publication now bounds carry 99 %
   of the decoder's loss and 98.5 % of the ridge's. On the ordinary window of the later years the decoder reads two to three times
   the ridge and grows worse with the years; the ridge, the ridge over the window, the answers
   with no model and Chronos-2 read as on the held months.
3. On the months the pretraining read or held out the decoder beats the bar by 13 to 16 %, real
   and small, and about half of that gain is the other channels' hours (the ridge over the window).
4. On the first mission Chronos-2 zero-shot, from 47 hourly means, beats the decoder's mean by
   1.3 to 1.7 times on the held months and holds its loss on the later years; the pretraining
   does not. On the second mission the two read level, and a reading of 2·10⁹ in a history
   sends Chronos-2 into the millions as it does every other answer. What drifts in the first
   mission's later years is not the levels, the count of readings or the hours missing; the cause
   is left open.
5. The two pretrainings differ by half on the later years (1.96 against 2.95 on the median
   window) where on the held months they differ by 3 %.

**Limitations.** Two pretrainings. The publication read predates the bound of 100 the telemetry
now takes (ADR-0080), which the post-hoc bands stand in for. The reading by bands, by years, by
drift and the clipped re-read are post hoc and descriptive; the one that was declared is the
first table. 1,363 unread windows were not drawn for an hour no channel read, 188 of the
training ones. The cause of the decoder's loss on the ordinary later window is not established.

## 2026-11-19 — declared before the run: the published forecaster against the best classical answer, window by window

**Question.** On the first mission's later years, which no pretraining read, does Chronos-2
zero-shot forecast the hour's mean better than the best classical answer by at least the least
gain, a fifth, when the classical answers include local statistical forecasters fitted on each
window's own hours, which follow drift as the ridge fitted once does not? On the second mission's
unread months, is it no worse?

The reading of 2026-11-15 compared Chronos-2 with the ridge per channel fitted once on the
training months, and only by the mean over cells, which a few cells of extreme values carry.
Its lead there, about three times on the ordinary windows of the later years, may be the
forecaster's or only its following of the windows' drift. These months were read on
2026-11-15, so this reading is a gate with a stronger bar, not a confirmation: a lead that
passes is confirmed on a mission no reading has touched before any claim rests on it.

**Conditions.**

| | |
|---|---|
| Commit | the one that adds this section; the instrument at `abbfe363` |
| Machine | Mac (M1 Pro, CPU for the local forecasters, MPS for Chronos-2); validation numbers |
| Cells | the 1,500 windows per mission and set of 2026-11-15 (seed 1), their contexts written again by the same command (`data/report/l54/contexts.sh`), each set's cells and targets identical to that reading's; held months (the three per mission the corpus kept back) and unread months (Mission1 2007-01 to 2013-12 and Mission2 2001-10 to 2003-06, the 35 frozen test months left unread) |
| Published forecaster | `amazon/chronos-2` (119M parameters), zero-shot, the 47 hourly means of a window's channels handed together, its median for the hidden hour's mean (`chronos2_answers.py`, `chronos-forecasting` 2.3.2 in its own environment) |
| Bar, nine classical answers | the ridge per channel and the ridge over the window of 2026-11-15, fitted once on the training months; the hour one period back, the last reading and the median of the history; and four forecasters fitted on each cell's own 47 hours, `statsforecast` AutoETS, AutoARIMA, AutoTheta and seasonal naive, the season the period the same-phase answer chose from those hours (one where it is under two hours or over 23), a missing hour carrying the hour before it and the hours before the first reading taking that reading (`local_forecaster_answers.py`, `statsforecast` 2.1.1 in its own environment) |
| Beside | the decoder of the forecasting pretext's first pretraining (seed 1 of 2026-11-09) |
| Measure | each cell's squared error of the hour's mean in normalised units; each window's mean over its cells; for an answer, the geometric mean over windows of Chronos-2's window mean over the answer's, as forecasting benchmarks aggregate relative errors, so a few windows of extreme values weigh no more than their share |
| Reading | `history_forecast_reading.py --verdict-of chronos2 --verdict-set unread --bar` the nine, only the unread rows reading a level; 10,000 resamples of whole windows, seed 1; the bar in every draw is the answer Chronos-2 leads least, so it is never weaker than any of its members; 95 % percentile interval; `data/report/l54/read.sh` runs the whole |

**Rules.**

- *The verdict* is the ratio over the best of nine on the first mission's unread months. An
  interval covering one is less than the threshold either way; an upper edge under one but above
  0.80 is real and small; an upper edge at or under 0.80 is at least the least gain; a lower edge
  above one is harm. Chronos-2 is one model with no seeds or pretrainings, so no spread is added.
- *The second condition* is the same ratio on the second mission's unread months, read by its
  level: anything but harm meets it. Both are needed, so no correction for two readings applies.
- *Everything else is descriptive*: the held months, the ratio of mean losses, Chronos-2 against
  each of the nine and against the decoder, and which answer is the best.
- The least gain stays 0.80, as the register set it for the forecast on 2026-11-15.

| Outcome on the first mission | Second mission | What follows |
|---|---|---|
| At least the least gain | not harm | forecasting is ground for a foundation model on this telemetry; the multi-mission corpus and adaptation is read next, confirmed on a mission no reading touched |
| At least the least gain | harm | the lead is the first mission's; the corpus is read only with that named limitation |
| Real and small, noise or harm | any | the lead over the ridge was the following of drift, which a local classical forecaster also does; forecasting is no foundation model's ground here unless an operations task is named for it on its own |

**Predictions**, from the trial's intervals (below), the edge nearer one for a lead:

| | Predicted | Source |
|---|---|---|
| Mission1 unread, Chronos-2 over the best of nine | at most 0.90: real and small or at least the least gain | the trial on training months, 0.66 [0.49; 0.90] |
| Mission1, the best of nine | the ridge over the window or a local forecaster (AutoARIMA, AutoETS) | the trial: window ridge 0.66, ridge 0.50, AutoETS 0.48, AutoARIMA 0.52 |
| Mission2 unread, Chronos-2 over the best of nine | 0.89 to 1.09: noise; the best a local forecaster | the trial, 0.97 [0.89; 1.09] against AutoETS; AutoTheta 0.96, AutoARIMA 0.88 |
| Ratio of mean losses, Mission1 | above one or wide, carried by few cells | the trial, 28 [0.46; 104] against AutoARIMA |

**Power.** On 100 trial windows the first mission's interval spans a factor of 1.8 (0.30 each
way on the log scale); on 1,500 windows that falls to about 0.08, an interval of roughly ±8 %
around the point, a little wider for the best taken in every draw. The upper edge reaches 0.80
when the point is at or under about 0.74: a true ratio of 0.66 reads at least the least gain in
nearly every world, 0.70 in about nine in ten, 0.74 in half, 0.80 in about one in forty; a true
1.0 reads noise.

**Trial, before this declaration.** `data/report/l54/trial.sh` on the first 100 windows per
mission of the training months, which the pretraining trained on and the reading of 2026-11-15
read: the local forecasters (198 s for 5,893 cells on eight cores), Chronos-2 (54 s with its
loading) and the reading script whole. The measure was chosen after the trial and is stated as
such: on the first mission the ratio of mean losses read 28 [0.46; 104] against AutoARIMA,
carried by a few cells of extreme values, as the reading of 2026-11-15 found for the ridge (1 %
of cells, 99 % of the loss), while the geometric mean over windows reads 0.66 [0.49; 0.90] on
the same cells. It is the aggregate forecasting benchmarks use, and the ratio of means stays
beside it. No window had an answer missing by nothing, which leaves no ratio and is refused.

Chronos-2 over each answer, geometric mean over windows, 95 % interval:

| Answer | Mission1 | Mission2 |
|---|---|---|
| ridge per channel | 0.50 [0.37; 0.69] | 0.26 [0.21; 0.32] |
| ridge over the window | 0.66 [0.48; 0.90] | 0.21 [0.17; 0.26] |
| AutoETS | 0.48 [0.39; 0.61] | 0.97 [0.87; 1.07] |
| AutoARIMA | 0.52 [0.39; 0.74] | 0.88 [0.75; 1.01] |
| AutoTheta | 0.21 [0.17; 0.27] | 0.96 [0.86; 1.07] |
| seasonal naive | 0.23 [0.15; 0.34] | 0.35 [0.26; 0.46] |
| hour one period back | 0.23 [0.15; 0.34] | 0.30 [0.23; 0.41] |
| median of the history | 0.04 [0.03; 0.06] | 0.11 [0.08; 0.14] |
| last reading | 0.007 [0.005; 0.010] | 0.008 [0.006; 0.010] |
| decoder, beside | 0.67 [0.47; 0.97] | 0.40 [0.33; 0.48] |
| **best of nine** | **0.66 [0.49; 0.90]**, window ridge | **0.97 [0.89; 1.09]**, AutoETS |

**Limitations declared.** The unread months were read on 2026-11-15, so this is a gate. One
published forecaster; TimesFM and TiRex are not read. The hour's mean only. The publication is
the one of 2026-11-15, before the bound of 100 the telemetry now takes, so extreme values remain
in some windows. The local forecasters see missing hours filled and one season per cell. One draw
of windows.
