# A supervised reference on the first mission

Without a label the best detectors on the first satellite mission are classical, and the pretrained
encoder's states match their initialisation ([telemetry-readout.md](telemetry-readout.md)). Taught
every event of the fitting side, the frozen direction over the states reaches about 0.26 with or
without pretraining, and fine-tuning read at one length reaches 0.09
([taught-fine-tuning.md](taught-fine-tuning.md)). Neither says how much room the task leaves a
network of this shape. This note trains the same network on every label, with its length chosen
where a length may be chosen, to put a number on that room before a pretext aims at it.

## 2026-11-04 — declared before the run: the network trained on every event, its length chosen on later fitting months

**Question.** How far does the small encoder over cells reach on the validation months, in
event-weighted PR-AUC, when it is trained from its initialisation on every event of the fitting
side and its rate and length are chosen on the fitting side's later months? Read against the frozen
direction taught the same events, which reads 0.266 at the initialisation, and the trees, 0.154.
This is a reference, not a claim: it measures room, and says nothing about pretraining.

**Conditions.**

| | |
| --- | --- |
| Code | `scripts/telemetry_supervised_reference.py`, at the commit of this declaration |
| Machine | M1 Pro, MPS, fp32; three processes share the accelerator; validation numbers |
| Task | `esa-mission1-events`: the fitting side is 2000-01 to 2006-12, 84 months, 61,324 windows, 99 events; the validation months are 56, 41,088 windows, 62 events, which the pretraining never read; the frozen test months are not touched |
| Cut for the choice | the 48 months before 2004-01 teach (35,020 windows, 70 events, 3,696 taught windows); the 36 months from 2004-01 choose (26,124 windows, 29 events read); one event lies on both sides, its earlier windows teach and its later ones are not scored |
| Encoders | the initialisation, drawn under each seed, is the reference; `base1` and `base2` run beside it under the same rule |
| Learning | as in the few-event pilot: batch 16, half of it windows of taught events with an event drawn first, half the other windows of the teaching months, binary cross-entropy, AdamW, warm-up over a tenth of the steps and decay to a hundredth; rates 0.001 and 0.0003; lengths 500, 1,000, 2,000, 4,000, 8,000 and 16,000 steps, each a run of its own with its own schedule; seeds 1, 2, 3 draw the start and the batches alike for every encoder |
| Refit | at the rate and length chosen for each encoder, the whole fitting side teaches and the validation months are read: the reference itself |
| Bars | the frozen direction and the trees taught all 99 events, copied from the diagnostics of the few-event pilot at commit `6dec492f`, refused if they scored other windows; the direction taught the teaching months' events, scored on the selection months, measured here |
| Reading | each month's scores read against that month; area and recall at 1 % of normal windows; 10,000 resamples over months; the validation months stay shut until every refit cell is in |
| Command | `data/report/l40/night.sh` (`measure --section select` in three lanes and `--section bars`, each resumed until whole), then `choose.sh`, `refit.sh`, `adopt.sh`, `read.sh`; the choice and the reading refuse a grid not yet whole |
| Cost | from the trial: a selection cell 100 s at 500 steps and 120 s at 1,000, about 0.04 s a step and 80 s to lay and score both sides; a refit cell 70 s at 500 steps; a direction cell on the selection months 170 s. The 108 selection cells are about 8.7 hours of cells in three lanes, the nine direction cells 26 minutes, the refit at most 1.7 hours of cells: about one night |

**The rule, stated before the run.**

- *The choice*, for each encoder from the selection months alone: the rate and length with the
  highest mean area over the three seeds; of every run within 0.01 of it the shortest, since a cell
  repeated on the accelerator moves by about that much; among equals the higher mean, then the rate
  in force. If the chosen length is the grid's longest, the grid grows once by a doubling, to
  32,000 steps, before the refit. A choice whose mean lies below the frozen direction's on the same
  months is *untrainable*: no length reaches a linear readout of the frozen states there.
- *The reading*: the initialisation's refit less the frozen direction at the initialisation, seeds
  paired, with its interval over months and T = 2·SE over seeds. Neither side was pretrained, so no
  spread between pretrainings enters. The pretrained encoders' refits are read the same way beside,
  descriptively.

| Outcome | Condition | What follows |
| --- | --- | --- |
| Clearly above 0.26 | difference at least +0.05 and T, interval above zero | the pretext sensitive to anomalies gets a numeric target: the room between this reference and the best zero-label detector, 0.154 |
| Around 0.26 | the interval holds zero | weak evidence: labels or training may limit, not the encoder; what that pretext is expected to reach is revised before any GPU |
| Below 0.26 | difference below −T, interval below zero | this recipe extracts less than a linear readout of frozen states; the direction stays the readout of the few-label leg |
| Untrainable | the choice above, on the selection months | as below, with the curve of lengths as the mechanism |

**Power, stated before the run.** On the stored cells of the few-event pilot at 99 events, the same
comparison at 2,000 steps has intervals over months of half-width 0.07 to 0.12 under three seeds. A
true difference of +0.05 is read above zero with probability about 0.2 to 0.3, and +0.15 with
about 0.8. So "around" cannot rule a room of 0.05 out; it can only say the room is not large. The
seeds' own standard error at 99 events was 0.01 to 0.04, inside that.

**Prediction.** The difference lies in [−0.17; +0.07], near −0.07: the outcome is "around" or
"below". Sources: fine-tuning less the direction at the initialisation, −0.171 [−0.232; −0.084] at
2,000 steps, and the gain of 200 over 2,000 steps at 10 events, +0.105 [+0.064; +0.154], both in
[taught-fine-tuning.md](taught-fine-tuning.md), added as if the length effect carried to 99 events.
The pretrained refits lie within ±0.05 of the initialisation's, from −0.001 [−0.044; +0.040] at
2,000 steps. For the chosen length there is no interval to predict from: at 10 events the
shortest run won, and at 99 events the late loss at 2,000 steps, 0.25, says the run was not done
learning, so the choice may fall either way; the curve is reported whatever it shows.

**Trial, before this declaration.** Every section ran end to end at seed 1, one rate and the
lengths 500 and 1,000: the selection cells with both sides scored, the direction on the selection
months, the choice, the refit, the bars adopted (every window and target identical to the cells'),
the reading by class and the reading itself with the grid not yet whole. The tables were written
and deleted unopened, as were the cells; only the arms, the cost and the choice's flags were read.
The flags said the choice fell on 500 steps for every encoder, so the grid did not ask to grow, and
that the initialisation's choice was below the direction on the selection months at those two
lengths, which the prediction already allows. Before the trial, moving the pilot's training and
scoring into two functions changed nothing: at 30 steps on the CPU the code of `main` and the new
code wrote the same bytes in all three cells.

**Limitations.** The pretrained encoders read the selection months in pretraining, which is why
the reference is the initialisation and they stand beside. The refit teaches about a fifth more
events than the choice saw, at the length the choice picked, which is conservative. About a tenth
of the fitting windows lie in events nobody has marked and teach as normal, under every encoder.
The validation months have been read by the frozen detectors and by the few-event pilot; the test
months stay frozen. The direction reads every block's states side by side under a shrunk
covariance, the fine-tuning the mean of the last block's: a structural difference the harness does
not let a trained encoder close. Three seeds; a cell repeated on the accelerator moves by up to
0.03. Each window is one hour: a detector that needs longer context is not measured here.

## 2026-11-05 — measured: at its chosen length the network trained on every event meets the frozen direction at 0.26, and both miss the same events

**Question.** The declaration above, by its rule, at commit `e0763d5c`.

**Conditions.** As declared, on a clean tree at `e0763d5c`: 108 selection cells in three lanes
(7.9 hours, three processes on the accelerator, a cell of 500 steps 97 to 189 s and of 16,000
steps 1,244 to 2,316 s), nine direction cells on the selection months (159 s each), nine refit
cells (121 to 307 s), the bars on the validation months adopted from the few-event pilot. M1 Pro,
MPS, fp32; validation numbers. The validation months were opened after the refit was whole.

**The choice**, on the selection months (2004-01 to 2006-12), area as a mean over three seeds.
The frozen direction taught the same teaching months stands beside; no encoder was untrainable
and none asked for a longer grid.

| Encoder | Rate | Steps | Area | Per seed | Direction on the same months |
| --- | --- | --- | --- | --- | --- |
| initialisation | 0.0003 | 500 | 0.259 | 0.243 0.288 0.246 | 0.206 |
| `base1` | 0.0003 | 1,000 | 0.192 | 0.172 0.096 0.308 | 0.021 |
| `base2` | 0.001 | 500 | 0.114 | 0.117 0.099 0.124 | 0.035 |

The curve on the selection months, by length, at the chosen rate of each encoder (at 500 steps
the other rate reads lower for every encoder):

| Steps | 500 | 1,000 | 2,000 | 4,000 | 8,000 | 16,000 |
| --- | --- | --- | --- | --- | --- | --- |
| initialisation, 0.0003 | 0.259 | 0.167 | 0.160 | 0.190 | 0.073 | 0.105 |
| `base1`, 0.0003 | 0.171 | 0.192 | 0.150 | 0.085 | 0.036 | 0.050 |
| `base2`, 0.001 | 0.114 | 0.074 | 0.058 | 0.045 | 0.035 | 0.030 |
| late loss, initialisation | 0.47 | 0.40 | 0.33 | 0.27 | 0.16 | 0.07 |

**The reading**, on the 56 validation months: the refit on all 99 events against the frozen
direction taught the same events, seeds paired, interval over months, T = 2·SE over seeds.

| Encoder | Refit per seed | Refit | Recall @1 % | Direction | Difference [interval] | p | T | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| initialisation | 0.244 0.274 0.275 | 0.265 | 0.602 | 0.266 | −0.001 [−0.044; +0.040] | 0.899 | 0.035 | **around** |
| `base1` | 0.176 0.200 0.080 | 0.152 | 0.554 | 0.261 | −0.109 [−0.167; +0.016] | 0.102 | 0.074 | around |
| `base2` | 0.178 0.274 0.203 | 0.219 | 0.570 | 0.217 | +0.002 [−0.115; +0.132] | 0.949 | 0.058 | around |

The trees taught the same events read 0.154 (recall 0.640). The prediction, a difference in
[−0.17; +0.07] near −0.07, held in its interval and missed its centre by 0.07: the refit lost
nothing to the direction.

Beside, descriptively. The selection cells scored on the validation months, at the setting chosen
for each encoder, read 0.219 (initialisation), 0.169 (`base1`) and 0.120 (`base2`): the refit's
29 more events and 36 more months add 0.05 at the initialisation. For the initialisation and
`base1` the setting chosen on the selection months is also the one that reads highest on the
validation months; for `base2` the other rate reads 0.171 there. By class of event: a seen class
has an event on the fitting side (56 of the 62 validation events), an unseen class none (six):

| Arm | Encoder | Seen classes (56 events) | Unseen classes (6 events) |
| --- | --- | --- | --- |
| refit | initialisation | 0.255 | 0.275 |
| refit | `base1` | 0.129 | 0.265 |
| refit | `base2` | 0.189 | 0.335 |
| direction | initialisation | 0.225 | 0.572 |
| direction | `base1` | 0.194 | 0.835 |
| direction | `base2` | 0.120 | 0.833 |
| trees | | 0.166 | 0.016 |

**Checks, after the run; no reading depends on them.** At 1 % of normal windows flagged, counting
an event found under at least two seeds, the refit and the direction at the initialisation find 34
of the 62 validation events together, 2 and 6 apart, and 20 neither; 11 events are found by none
of the seven arms, the trees included, and 16 never enter the top 5 % of windows under either
readout at the initialisation, at 1 to 158 windows long. The events' best positions correlate 0.72 (Spearman) between
the two readouts, 0.42 between the refit and the trees. The initialisation's row recomputed from
the stored answers with a second resampling of the months (seed 7, 2,000 draws) gives
−0.0015 [−0.043; +0.042]; its interval prints the same as the few-event pilot's "pretrained less
initialisation" at 99 events, which is a coincidence at the third decimal: that row pools both
pretrained encoders and recomputes to −0.0012 [−0.0439; +0.0398] against this one's
−0.0015 [−0.0435; +0.0398]. Every refit cell taught 6,227 windows, the pilot's ceiling count, and
scored 41,088. The refit's late loss is 0.50 at the initialisation and 0.36 to 0.44 under the
pretrained encoders, the weights moved by 0.025 to 0.034: at 500 steps the network has drawn each
event about 80 times and is far from fitting them.

**Conclusions.**

- **The recipe reads highest when it has barely trained: 0.265 at 500 steps on the validation
  months, against 0.266 for the frozen direction at the same initialisation.** The outcome is
  *around*: a room of 0.05 above 0.26 is not ruled out, the power said so, but no room shows.
  Both readouts are near-linear readouts of untrained features, and they miss the same events,
  so the number measures what this recipe extracts from these windows, not what the task
  allows. What the pretext sensitive to anomalies is expected to reach is revised before any
  GPU: its target is not a number this reference sets but whether a pretrained encoder reads
  above its initialisation at all, under the direction or under short fine-tuning, and whether a
  pretext can change which events the windows show, as training has not.
- **The shortest run wins at every encoder and both rates**, and the loss keeps falling as the
  area falls: by 8,000 steps the late loss is below 0.2 and the area 0.03 to 0.07 on the later
  months. For two encoders of three the setting chosen on the selection months is also the best
  on the validation months, which is what the cut was for.
- **Pretraining gives this reading nothing.** The pretrained refits read 0.152 and 0.219, within
  their intervals of the initialisation's but not above it, and under the frozen direction the
  pretrained states read 0.02 to 0.04 on the selection months, which the pretraining read, against
  0.21 at the initialisation. A hypothesis, not a finding: the pretext learnt to reconstruct those
  months with their events, so their states do not stand out; the few-event pilot's control on
  read months pointed the same way, but drift between the years is not separated from it here.
- **The readouts share most of their events and part on the unseen classes.** At 1 % of normal
  windows flagged the refit and the direction at the initialisation find 34 events together and 2
  and 6 apart. By class, the refit gains on the seen classes, 0.255 against the direction's 0.225,
  and loses on the six unseen events, 0.275 against 0.572 to 0.835, where the direction over
  frozen states of either kind finds most of them and the trees none. A detector that must catch
  what nobody labelled keeps the frozen direction.
- **0.26 stays a lower bound of what the task allows, now for two readouts.** A linear direction
  over the states of an untrained encoder and the same encoder after 500 steps on every label meet
  there, with a recall of 0.60 at 1 % of normal windows for both, and most of the events neither
  finds are found by no other arm either.

**Limitations.** The grid's shortest length won; the few-event pilot saw 200 steps beat 500 at 10
events, so the curve may peak below the grid and the reference is a lower bound of this recipe
too. Three seeds; `base1` spreads 0.08 to 0.20 over them. The readouts differ in structure, all
blocks side by side under a shrunk covariance against the mean of the last block. The unseen
classes hold six events. The validation months had been read before by the frozen detectors and
by the few-event pilot.
