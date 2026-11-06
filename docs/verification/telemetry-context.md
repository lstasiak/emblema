# A longer input on the first mission

Every detector on the first mission reads one hour of telemetry. Taught every event of the fitting
side, the frozen direction over the states of an untrained encoder and the same encoder trained on
every label meet at 0.26 and miss the same events
([telemetry-supervised-reference.md](telemetry-supervised-reference.md)); without a label the
whitened distance over those states reads about 0.08 and the best classical bar 0.15
([telemetry-readout.md](telemetry-readout.md)). Half of the validation events last longer than
twelve hours, and a quarter never enter the top 5 % of windows under any one-hour readout. Before
a pretext aims at this task, this note asks whether the input is the limit: the same detectors,
the same encoder at its initialisation, a window of several hours ending at the scored hour.

## 2026-11-05 — declared before the run: the detectors at the initialisation over windows of 6, 24 and 72 hours, chosen on later fitting months

**Questions.** (1) Without a label, does the whitened distance over the states of the encoder at
its initialisation read the validation months higher when the encoder sees the hours before the
scored one? (2) Taught every event of the fitting side, does the frozen direction? Beside them,
descriptively: whether the gain, if any, needs the encoder to see the context, or a detector over
one-hour states pooled over the previous hours reads the same; whether smoothing the one-hour
detector's own scores over the previous hours, the answer with no model, reads the same; and where
the classical bars over the cells themselves stand with the same context. Nothing is pretrained or
trained; the scored unit and its label stay the hour.

**Conditions.**

| | |
| --- | --- |
| Code | `scripts/telemetry_context_report.py`, at the commit of this declaration |
| Machine | M1 Pro, MPS for the encoder, the CPU for the detectors; validation numbers |
| Task | `esa-mission1-events` as in the supervised reference: the fitting side is 2000-01 to 2006-12, 84 months, 61,324 windows, 99 events; the validation months are 56, 41,088 windows, 62 events; the frozen test months are not touched, and a window's context never reads one. The validation months interleave with the frozen ones, 24 of the 56 follow one, so a window in the first hours of such a month sees a context cut short while no selection window does: under one rule for every arm, the control included, a validation window whose widest context of 72 hours reaches into a month the task never loaded is not read, about 1,700 windows of 41,088 (4 %) touching 3 events, counted by the run. An hour the task does not score, over a gap in the downlink inside a loaded month, is an hour of nothing in a context, as the data has it |
| Cut for the choice | as in the supervised reference: the 48 months before 2004-01 fit, the 36 months from 2004-01 choose; an event on both sides fits with its earlier windows and is not read |
| Encoder | the small shape over cells (width 64, two blocks), drawn at its initialisation under each seed; the pretrained weights are not read, since they read the fitting months |
| Input | a layout `<hours>x<bins>`: the window the encoder sees ends at the scored hour and is cut into bins, each bin's cell derived from the published one-hour cells it gathers, exactly as the layout would cut the concatenated tokens (tested against it). The control is the published layout, 1x6, re-measured by the same code. Span at the pretrained shape's resolution: 6x6, 24x6, 72x6. Span at a finer resolution: 6x36 (ten-minute cells, as the control's) and 24x24 (one-hour cells) |
| Detectors | over the states, read as the campaigns read them (every block side by side, mean over the cells, each score against its month): the whitened distance without a label, and Fisher's direction taught the fitting events (70 in the choice, 99 in the refit); the recipes are the campaigns' |
| Beside | `pooled`: the hour's one-hour state beside the mean of the loaded previous hours' states, over 6, 24 and 72 hours, its own state again where no previous hour is loaded, under both detectors; `smoothed`: the control's own scores taken as the maximum or the mean over the 6, 24 or 72 scored hours ending at and including the scored one; the cells over the corpus's channels at every layout of six bins, under the farthest single feature without a label, once, since it draws nothing, and under trees taught the same events, per seed |
| Seeds | 1, 2, 3 draw the initialisation and the trees; the encoder has no parameter per bin, so one seed draws the same weights under every layout and the two sides of a pair share their encoder (checked) |
| Reading | each month's scores read against that month; area and recall at 1 % of normal windows; 10,000 resamples over months; the validation months stay shut until the choice is made |
| Command | `data/report/l38/context/night.sh` (`measure --section select` in three lanes, each resumed until whole), then `choose.sh`, `refit.sh`, `check.sh`, `read.sh`; the choice and the reading refuse a grid not yet whole |
| Cost | from the trial, one lane on the accelerator: laying the published cells 222 s once; the states of a layout of six bins 130 to 140 s a seed, of 24x24 1,665 s, of 6x36 about 64 minutes from the timing on 200 windows; the detectors seconds (the direction and the whitened distance under a second, the farthest feature 4 to 17 s, the trees 14 to 28 s, the pooled states about a second); the refit reads the states kept by the selection, so it is minutes. The selection is about 5.5 hours of accelerator time in two lanes, a night |

**The rule, stated before the run.**

- *The choice*, for each family and detector from the selection months alone: the arm with the
  highest mean area over the three seeds; of every arm within 0.01 of it the shortest, then the
  one with the fewest bins. The control is among the arms, so a family whose context never beats
  it by that margin chooses the control; its best arm of context is then read on the validation
  months beside, marked so, and decides nothing.
- *The reading*, for the two declared families, the states under the whitened distance and under
  the direction: the chosen arm less the control on the validation months, seeds paired, with its
  interval over months and T = 2·SE over seeds; nothing here was pretrained, so no spread between
  pretrainings enters. The two readings are one family under Holm, and "clearly above" needs
  the correction passed. The least gain that counts is the detection tasks' registered 0.05.
  Where T exceeds it the reading is undecided: a difference inside ±T then closes nothing.
  Everything beside is read the same way, descriptively.
- *The check*, before any area is read: the control's refit cells under the direction must score
  the supervised reference's campaign-made direction cells' windows to the device's precision;
  if they do not, the run is stopped and the code is wrong.

| Outcome | Condition | What follows |
| --- | --- | --- |
| Clearly above | difference at least +0.05 and T, interval above zero, passes under Holm, for either declared reading | the input was a limit: the pretext sensitive to anomalies is pretrained and read over that layout, which needs the layout as a knob of the package and a publication of the telemetry in windows of that length for the pretraining |
| Real, small | interval above zero, difference below 0.05 | context helps by less than the least gain; the pretext keeps the one-hour input unless the size is talked through |
| Around | the interval holds zero and T is at most 0.05, for both | at these spans the input is not what holds the detectors at 0.26 and 0.08; a pretext on the one-hour input gains little from the blocker, and the direction of the pretext is talked through before any GPU |
| Undecided | the interval holds zero and T exceeds 0.05 | the seeds spread more than the least gain; nothing closes, and whether more seeds are worth their night is decided with the size read |
| Below | difference below −T, interval below zero | context harms at this resolution; the layouts at finer resolution are read beside to say whether it is the span or the coarser cells |
| No context chosen | both families choose the control on the selection months | as "around", with the best arms of context read beside |

Beside, three readings shape what a positive outcome asks for, and none decides it: if `pooled`
gains as much as the layouts, the encoder need not see the context and a detector over one-hour
states can pool it; if `smoothed` gains as much, the gain is not of the input at all but of
reading the score over time, and the pretext has nothing to aim at; and the cells' bars say
whether the gain is of the encoder or of the longer input itself.

**Power, stated before the run.** Under the direction at the initialisation the supervised
reference's interval over months had a half-width of 0.04 with three seeds paired, and its seeds
spread with a standard error of 0.01 to 0.04, so T may reach 0.08 and exceed the least gain: a
reading under the direction may come out undecided. Under the whitened distance the
initialisation's area spreads 0.009 over the ten seeds of the screen's cells
([telemetry-readout.md](telemetry-readout.md), `untrained_knn@distance=whitened,reference=unit`),
so T at three seeds is about 0.01, and the screen's interval over months for a pairing of two
encoders had a half-width of 0.018; here the two sides share their encoder. A true gain of +0.05
is read above zero with probability about 0.7 under the direction and above 0.95 under the
whitened distance; +0.10 above 0.95 under both. So "around" under the direction leaves a gain of
about 0.03 open; under the whitened distance it leaves little.

**Size from above.** Sixteen of the 62 validation events never enter the top 5 % of windows under
either one-hour readout, and 11 are found by no arm at all. Were half of them to become readable
with context while the rest held, the event-weighted area would rise by about 0.1; that is the
largest gain this run can realistically show, and there is no precedent for it in our data. The
literature says only the direction: a detector without history raises false alarms (Telemanom),
and the benchmark's methods read windows longer than an hour.

**Prediction.** No earlier interval measured context, so the predictions come from the mechanism
and from the widths above, not from a lower edge. Under the direction the difference lies in
[−0.05; +0.10], near +0.02: the hours before an event's onset may show a drift the hour hides, but
a 24-hour window at six bins has four-hour cells, and 11 of the 62 events last an hour or less.
Under the whitened distance it lies in [−0.03; +0.05], near +0.01: a longer window averages the
readings the whitened distance flags. The finer layouts read at or above the coarse ones of the
same span. `smoothed` gains at most what the control's recall at 1 % leaves, and `pooled` lies
between the control and the layouts. The cells' bars move with the layouts.

**Trial, before this declaration.** Every section ran end to end at seed 1 over the layouts 1x6,
6x6 and 24x24 and the span of 6 hours, on the tree before this declaration's commit: the cells
laid, the selection with every family, the choice, the refit of the chosen and best arms with the
controls, the check and the reading with the grid not whole. The check held: the control's refit
cell under the direction scores the supervised reference's 41,088 windows with a largest gap of
6·10⁻⁵, a mean gap of 2·10⁻⁸ and no rank moved by more than ten. The choice's and the reading's
tables were written and deleted unopened, as were the cells and the states; only the cell names,
their seconds, the counts of windows not read (0 in the selection, 552 on the validation months
at a widest context of 24 hours) and the check were read. One line of the reading's last table
slipped through the filter on the trial's log: the control's cells bar under the farthest feature
finds 17 of the 62 validation events at 1 % of normal windows, a count of a bar already measured
in the screen, not an area and not a context arm.

**Beside the run, checked before it.** The published cells of 300 windows laid by the derivation
equal the campaign runtime's cells to the last bit, and the states embedded from them are equal
to the last bit; one seed draws the same weights under six and under thirty-six bins. The cost of
the arms on 200 windows: a layout of six bins embeds in about 2 minutes a seed, 24x24 in 32, 6x36
in 64, with the batch scaled to the cells a window holds.

**Limitations.** Three seeds. The pretrained encoders are not read, so the note says nothing of
whether a pretrained encoder would read context better than its initialisation. The derivation
gives the cells the layout would make of the concatenated tokens, but the published windows'
per-channel normalisation and the hour-aligned cut stay as published. The classical bars over the
cells are read at six bins only: at finer resolution the cells run to 12 to 19 thousand features
and do not fit beside the accelerator. The validation months have been read before by the frozen
detectors, the few-event pilot and the supervised reference; the test months stay frozen.

## 2026-11-06 — measured: no layout of context is chosen; read beside, every context arm over the states falls below the hour

**Questions.** As declared: whether the whitened distance and the frozen direction over the
states of the encoder at its initialisation read the validation months higher when the encoder
sees the hours before the scored one, the layout chosen on the later fitting months.

**Conditions.** As declared, at commit `aa99bb1c`; `night.sh 2` on the M1 Pro (MPS for the
encoder), two lanes sharing the accelerator, then `choose.sh`, `refit.sh 2`, `check.sh`, `read.sh`.
The selection took 16,855 s (4 h 41 min), the refit 209 s; a seed of 24x24 took 3,000 to
3,200 s in a shared lane, of 6x36 5,800 to 6,900 s, of six bins 130 to 400 s. The check
held under every seed: the control's refit cells under the direction score the supervised
reference's 41,088 windows with a largest gap of 3.7 to 6.1·10⁻⁵ and no rank moved by more than
ten. Windows not read on the validation months, their context of 72 hours reaching into a month
the task never loaded: 1,704 of 41,088 under every arm alike, so 39,384 windows are read, 4,569
of them event windows, 61 of the 62 events; one event lies wholly in those first hours.

**The choice, on the months 2004 to 2006, 70 events taught; mean area over three seeds.**

| Family | Detector | 1x6 | 6x6 | 6h | 6x36 | 24h | 24x6 | 24x24 | 72h | 72x6 | Chosen | Best context |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| states | direction | **0.206** | 0.103 | | 0.097 | | 0.036 | 0.020 | | 0.022 | 1x6 | 6x6 |
| states | whitened | **0.034** | 0.036 | | 0.025 | | 0.027 | 0.023 | | 0.034 | 1x6 | 6x6 |
| pooled | direction | **0.206** | | 0.179 | | 0.197 | | | 0.192 | | 1x6 | 24h |
| pooled | whitened | 0.034 | | **0.046** | | 0.039 | | | 0.029 | | 6h | 6h |
| smoothed, max | direction | **0.206** | | 0.082 | | 0.063 | | | 0.077 | | 1x6 | 6h |
| smoothed, max | whitened | 0.034 | | 0.039 | | 0.038 | | | **0.060** | | 72h | 72h |
| smoothed, mean | direction | **0.206** | | 0.094 | | 0.106 | | | 0.170 | | 1x6 | 72h |
| smoothed, mean | whitened | **0.034** | | 0.038 | | 0.034 | | | 0.032 | | 1x6 | 6h |
| cells | farthest | **0.167** | 0.076 | | | | 0.021 | | | 0.009 | 1x6 | 6x6 |
| cells | trees | 0.036 | 0.058 | | | | **0.078** | | | 0.069 | 24x6 | 24x6 |

Both declared families choose the control: under the direction every layout of context reads
below the hour, by half at six hours and by ten times at 24 and 72; under the whitened distance
6x6 is within 0.01 of the hour and the shorter wins. The outcome is **"no context chosen"**, read
as "around"; the best arm of context of each family is read beside.

**The reading, on the validation months, 99 events taught; the arm less the control, seeds
paired, 10,000 resamples over months, T = 2·SE over seeds.**

| Family | Detector | Arm | Area per seed | Area | Recall @1 % | Control | Difference [interval] | p | T | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| states | direction | 6x6, beside | 0.118 0.159 0.157 | 0.144 | 0.426 | 0.275 | **−0.130** [−0.195; −0.066] | 0.000 | 0.040 | below, beside |
| states | whitened | 6x6, beside | 0.032 0.031 0.048 | 0.037 | 0.158 | 0.096 | **−0.059** [−0.136; −0.008] | 0.025 | 0.011 | below, beside |
| pooled | direction | 24h, beside | 0.261 0.274 0.270 | 0.268 | 0.525 | 0.275 | −0.006 [−0.037; +0.021] | 0.603 | 0.022 | around |
| pooled | whitened | 6h, chosen | 0.045 0.038 0.035 | 0.040 | 0.180 | 0.096 | −0.056 [−0.111; −0.009] | 0.013 | 0.006 | below |
| smoothed, max | direction | 6h, beside | 0.106 0.089 0.095 | 0.097 | 0.541 | 0.275 | −0.178 [−0.262; −0.104] | 0.000 | 0.018 | below |
| smoothed, max | whitened | 72h, chosen | 0.021 0.022 0.022 | 0.022 | 0.098 | 0.096 | −0.074 [−0.163; −0.013] | 0.027 | 0.001 | below |
| smoothed, mean | direction | 72h, beside | 0.069 0.048 0.056 | 0.058 | 0.240 | 0.275 | −0.217 [−0.325; −0.106] | 0.000 | 0.019 | below |
| cells | farthest | 6x6, beside | 0.062 | 0.062 | 0.344 | 0.021 | +0.041 [−0.092; +0.097] | 0.222 | 0 | around |
| cells | trees | 24x6, chosen | 0.048 0.059 0.055 | 0.054 | 0.508 | 0.031 | +0.023 [−0.037; +0.060] | 0.474 | 0.026 | around |

The controls per seed: the direction 0.285, 0.282, 0.257 (recall 0.590 at 1 %), the whitened
distance 0.097, 0.095, 0.096 (0.180), the trees 0.018, 0.061, 0.015. The smoothed arms at every
span, 12 rows, read −0.055 to −0.230 with every interval below zero. The direction's control
reads 0.275 against 0.266 in the supervised reference, since 1,704 windows are not read.

**Checks, after the run; no reading depends on them.** Of the 348 normal windows an arm scores
highest, the share within six hours after an event's last window is 1 to 2 % under the hour's
direction, 15 to 20 % under 6x6 and 25 % under the scores smoothed by their maximum over six
hours; under the maximum over 24 hours all 348 lie within the day after an event. The 11
validation events that none of the supervised reference's seven arms found at 1 % under two
seeds are found by no arm of context under two seeds either. By the benchmark's annotations
those 11 are local in 10 cases against 23 of the 51 found, touch a median of 3 of the 48 channels
against 12, last a median of 5.5 hours against 15, and 8 are of one class that the fitting side
teaches with 30 events and the arms still miss in 8 of its 15 validation events. Against its
control, 24x6 under the trees finds 10 to 11 events the control does not and loses 6 to 11, per
seed; 6x6 under the direction gains 0 to 4 and loses 7 to 20.

**Conclusions.**

- **No layout of context is chosen, and read beside, every context arm over the states falls
  below the hour**: the direction −0.130 [−0.195; −0.066] at 6x6, the whitened distance −0.059
  [−0.136; −0.008], 24 and 72 hours lower still on the selection months. The outcome is "no
  context chosen", read as "around": at these spans the input is not what holds the detectors
  at 0.26 and 0.08. The predictions, [−0.05; +0.10] and [−0.03; +0.05], missed on the downside
  for the arms read beside.
- **Two mechanisms show, and neither is information the hours lack.** The state is the mean
  over the window's cells, so an event's hour weighs 1/6 of a six-hour state and recall at 1 %
  falls from 0.59 to 0.43; and a window ending at the scored hour carries the hours before it,
  so the hours after an event inherit its signature and take 15 to 20 % of the top false alarms
  against 1 to 2 % for the hour alone. The task labels the hour; a trailing window is punished
  for its trail.
- **Smoothing the hour's scores over time, the answer with no model, harms most**: −0.18 to
  −0.23 under the direction. Whatever a longer window could give is not a gain of reading the
  score over time.
- **Pooling the previous hours' states beside the hour's leaves the direction where it was**,
  −0.006 [−0.037; +0.021]: taught the events, the direction weighs the hour and ignores the
  context. The trees over the cells with 24 hours read +0.023 [−0.037; +0.060] over their
  one-hour control and find different events, not more.
- **What the detectors miss is local, few-channel and short, not long**: a deviation in 3 of
  48 channels is a small part of a state averaged over every channel's cells, and a longer
  window averaged the same way sees less of it, as this run found. The lever to test next is a
  channel-local readout against each channel's own recent hours, with no model first; a pretext
  on the one-hour input averaged over the window gains little from the blocker, and none goes
  to the GPU before that check.

**Limitations.** Three seeds; T reaches 0.04 under the direction. The hour stays the scored unit
and the label, so a window ending at it is read against labels its trail cannot match, and the
readout is the campaigns' mean over cells; no readout that keeps the last hour apart was tried.
The windows not read are 4 % of the validation months and one event, under every arm alike. The
classical bars with context are read at six bins only. The validation months have been read
before; the test months stay frozen.
