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
