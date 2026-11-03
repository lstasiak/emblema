# Fine-tuning on a few taught events

The frozen states of a pretrained encoder match their initialisation on the first satellite mission,
with or without taught events ([telemetry-readout.md](telemetry-readout.md)). One route stays open:
the encoder itself learning from the events. This note asks whether that gain is large, cheaply and
on the fitting side alone, before any code or campaign is built for it.

## 2026-11-03 — declared before the run: fine-tuning on five or ten events, read on later fitting months

**Question.** Does full fine-tuning on 5 or 10 taught events gain at least 0.05 in event-weighted
PR-AUC under a pretrained encoder, over the same fine-tuning from its initialisation?

**Conditions.**

| | |
| --- | --- |
| Code | `scripts/taught_fine_tuning_pilot.py`, at the commit of this declaration |
| Machine | M1 Pro, MPS, fp32; validation of the idea, not a campaign |
| Task | `esa-mission1-events`, fitting side only: 2000–2002 teach (26,260 windows, 62 events), 2003–2006 are read (35,064 windows, 37 events); no event lies on both sides; the validation side is not touched |
| Encoders | the small shape over six-bin cells, pretrained `base1` and `base2`, and its initialisation under each seed |
| Learning | the registered schedule's 2,000 steps, batch 16, at rates 0.001 and 0.0001, the screen's settings at 50 and 20 stays, both for each encoder; half of each batch from taught events, an event drawn first so each weighs alike, half from the teaching months' other windows, counted as normal |
| Grid | events 5 and 10, seeds 1, 2, 3, which draw the events and the batches alike for every encoder |
| Bars, beside | the direction over frozen states and trees over summaries, taught the same events |
| Reading | each month's scores read against that month; area and recall at 1 % of normal windows; 10,000 resamples over months |
| Command | `scripts/taught_fine_tuning_pilot.py measure`, then `read` |
| Cost | a fine-tuning cell 2 min, a direction cell 3 min, trees 2 min; about 2 hours in all |

**The rule.** The family is the four readings of fine-tuning (two rates by two counts) under Holm's
correction. A reading passes when its gain, the mean over both pretrained encoders, reaches both
the least gain of 0.05, as the first mission's campaigns count it, and the threshold
T = 2·√(SE² + s²/4), with s = max(0.014, the pretrainings' own spread), and its interval over months
lies above zero and stands under the correction.

| Result | Meaning | What follows |
| --- | --- | --- |
| A reading passes | a large gain, under two pretrainings | a runtime and a decision record, then a reading on fresh pretrainings |
| None passes, no harm | less than about T | the direction closes as a lever at this size |
| Gain below −T, interval below zero | harm | the direction closes, and the note names the mechanism if one is measured |

**Prediction.** No reading passes; gains between −0.03 and +0.04. Sources: the frozen direction's
pretrained less initialisation, −0.003 [−0.034; +0.023] at 5 events and +0.019 [−0.016; +0.043] at
10 ([telemetry-readout.md](telemetry-readout.md)), which are read on other months and without the
encoder learning. T is expected near 0.07, from s of about 0.06 in those readings. So this test
sees a large gain only: below T a real gain cannot be told from none.

**Trial, before this declaration.** The exact command at 5 events and seed 1 ran end to end; the
reading wrote its tables, unopened, and only the arms and cost were read. On those cells the loss
fell from about 0.4 to about 0.01 under every encoder, with no failure, and the weights moved about
5 %. Those cells stay in the pilot.

**Limitations.** Two pretrainings only. The two sides are consecutive years of one mission, and 40
of the fitting side's 99 events begin in 2000, on the teaching side. About a tenth of the windows
counted as normal lie in events nobody has marked. The areas are not comparable with those on the
validation side.

## 2026-11-03 — measured: the first reading settles nothing, because the pretrained encoders had read those months

**Question.** The declaration above, read by its rule at commit `5cc04705`.

**Conditions.** As declared: 60 cells, M1 Pro, MPS, fp32, about 2.2 hours of cells.

| Reading | `base1` | `base2` | Initialisation | Gain [interval] | T | Passes |
| --- | --- | --- | --- | --- | --- | --- |
| rate 0.001, 5 events | 0.128 | 0.026 | 0.073 | +0.005 [−0.038; +0.080] | 0.100 | no |
| rate 0.001, 10 events | 0.058 | 0.083 | 0.086 | −0.015 [−0.053; +0.034] | 0.062 | no |
| rate 0.0001, 5 events | 0.100 | 0.027 | 0.070 | −0.006 [−0.045; +0.092] | 0.094 | no |
| rate 0.0001, 10 events | 0.049 | 0.062 | 0.105 | −0.049 [−0.084; +0.048] | 0.059 | no |

Beside them, the direction over frozen states read 0.024 and 0.023 under the pretrained encoders and
0.086 at the initialisation at 5 events, 0.028 and 0.039 against 0.125 at 10; the trees 0.006 and
0.020.

**Why this does not close the direction.**

- The declaration missed that the base recipe pretrains on the first half of each mission's months,
  which is this task's fitting side. Both the teaching and the read months had been read by the
  pretrained encoders and not by their initialisation, so the comparison is not neutral.
- The direction under `base1` at 5 events reads 0.024 here and 0.170 on months it never read, while
  the initialisation reads 0.086 and 0.128. That fits events read in pretraining no longer standing
  out, but the months and events also differ, and the mechanism was not measured.

**Conclusions.**

- No reading passes, as predicted; one gain (−0.049) lies just below the predicted range.
- The rule's verdict stands as recorded and is not used: the design cannot answer its question.
- A failure to pass could not have closed the direction anyway, as the next declaration's power shows.
- Training behaves: the loss falls under every encoder with no failure.

**Limitations.** Three seeds and two pretrainings; T is 0.06 to 0.10.

## 2026-11-03 — declared before the run: fine-tuning on five or ten events, read on months the encoders never read

**Question.** The same: does full fine-tuning on 5 or 10 taught events gain at least 0.05 in
event-weighted PR-AUC under a pretrained encoder, over the same fine-tuning from its initialisation?

**Conditions.**

| | |
| --- | --- |
| Code | `scripts/taught_fine_tuning_pilot.py`, at the commit of this declaration |
| Machine | M1 Pro, MPS, fp32; a pilot, not a campaign |
| Task | `esa-mission1-events`: the whole fitting side teaches (61,324 windows, 99 events); the validation months are read (56 months, 41,088 windows, 62 events), which pretraining never read; the frozen test months are not touched |
| Events | drawn under the task's registered identity, so each seed teaches the events it taught the telemetry screen's detectors |
| Encoders, grid | `base1`, `base2` and the initialisation; events 5 and 10; seeds 1, 2, 3 |
| Learning | as in the first declaration, at rate 0.001 alone: at 0.0001 the first run's late loss stays at 0.03 to 0.15 against 0.006 to 0.02, and neither rate led in its reading |
| Reading | each month's scores read against that month; area and recall at 1 % of normal windows; 10,000 resamples over months; a run refused unless its grid is whole |
| Bars, beside | the screen's cells at the same seeds and events, copied from its export, refused if they scored other windows: the direction over frozen states, read against each month, and trees over summaries, read against the fitting side, the only form the screen ran at both counts |
| Command | `scripts/taught_fine_tuning_pilot.py measure`, `adopt`, `read` |
| Cost | 18 cells of 2 min, about 40 min |

**The rule.** Two readings of fine-tuning under Holm's correction, T taking one spread between
pretrainings over both. Each reading has one of four outcomes; the first declaration's "less than T"
is withdrawn, since no failure to pass rules a gain out.

| Outcome | Condition | What follows |
| --- | --- | --- |
| Large gain | gain at least 0.05 and T, interval above zero, standing under the correction | a runtime and a decision record, then fresh pretrainings |
| Harm | gain below −T, interval below zero | the direction closes |
| Closed | gain + T and the interval's upper end both below 0.05 | the direction closes at this size |
| Inconclusive | anything else | nothing closes; read again, descriptively, when fresh pretrainings exist |

**Power, stated before the run.** A normal approximation, with s = 0.06 to 0.09 and a spread over
seeds of 0.04, both from the frozen direction at one pair of pretrainings, so s could be several
times other than that and fine-tuning's own may differ. With two pretrainings and three seeds, T is
0.08 to 0.10. A true gain of 0.05 passes with probability 0.16 to 0.25, and 0.10 with 0.49 to 0.74.
With no true gain, the least gain is ruled out with probability 0.16 to 0.25. So the expected
outcome is inconclusive, whatever is true, and this run can only show a very large gain. About eight
pretrainings at ten seeds would reach power near 0.8 at a gain of 0.07.

**Prediction.** Both readings inconclusive; gains between −0.04 and +0.04. Sources: the frozen
direction on the same months, −0.003 [−0.034; +0.023] at 5 events and +0.019 [−0.016; +0.043] at 10
([telemetry-readout.md](telemetry-readout.md)). The first reading's areas do not enter the range,
being read on months the encoders had seen.

**Trial, before this declaration.** The command at 5 events and seed 1, at both rates, ran end to
end. Six cells took about 2 min each, with no failure and the loss falling under every encoder. The
bars were copied, every window and target identical to the measured cells', and the reading wrote
its tables, which were deleted unopened. The cells are deleted too: the code gained its record of
the commit after them, and each cell belongs to one commit.

**Limitations.** Two pretrainings. These validation months have been read before by frozen detectors,
not by fine-tuning. Three seeds against the screen's ten, and the draws differ widely: at 5 events
seeds 1, 2 and 3 teach 249, 354 and 12 windows. The rate comes from the clinical screen, not from
this task. About a tenth of the fitting windows (6,227) lie in events and all but the taught ones
are drawn as normal, under both encoders, which pairing does not remove if pretraining reacts to
them differently. Fine-tuning against the trees is not like for like, as they are read against the
fitting side.

## 2026-11-04 — declared before the run: what fine-tuning does here, and how far the task lets a detector go

**Question.** The reading on unread months leaves three things open: whether this recipe can learn
events at all, whether it overfits what it is taught, and how much room the task leaves when labels
are not the limit. These diagnostics judge no claim. They change no rule, select nothing, and the
readings above stand.

**Conditions.**

| | |
| --- | --- |
| Code | `scripts/taught_fine_tuning_diagnostics.py`, at the commit of this declaration |
| Machine | M1 Pro, MPS, fp32 |
| Encoders, seeds | `base1`, `base2` and the initialisation, seeds 1, 2, 3 |
| Learning | as in the pilot, at rate 0.001, except where stated |
| Control | the 36 fitting-side events of at least 20 windows: first halves teach, second halves are read (3,002 taught windows); a month holding a read window is read whole, so no event is read against itself, and normal windows teach only in the remaining months of even place (16,901 normal windows teach, 41,216 windows are read); every other event leaves both sides; the frozen direction beside |
| Curve | the pilot's cells on unread months, each a run of its own at 200 and at 500 steps, beside the pilot's 2,000; events 5 and 10 |
| Ceiling | all 99 events of the fitting side taught, read on the unread months: fine-tuning, the frozen direction and the trees |
| Classes | the pilot's cells on unread months, each read apart on the events of the classes it was taught and on the rest, each half as a gain over the same encoder untaught, from the telemetry screen's cells; no new training |
| Command | `scripts/taught_fine_tuning_diagnostics.py measure`, `classes`, `read` |
| Cost | about 2.8 hours in two launches: control and curve 87 min, ceiling 82 min |

**How each is read, stated before the run.** All descriptive: three seeds. Only pretrained less
initialisation carries an interval over months, and no rule below uses it.

| Diagnostic | Reading | What it says |
| --- | --- | --- |
| Control | on every encoder, fine-tuning's area at least the direction's less 0.05 | the recipe can learn events, so the pilot's result is about the task and its few events |
| Control | on every encoder, more than 0.05 below | the recipe or its code is at fault, and the pilot's comparison with the direction says little about fine-tuning as such |
| Control | encoders disagree | nothing about the recipe in general; the note names which |
| Curve | a shorter run above the full one by more than 0.03 at both counts and on at least two of three encoders | the full run overfits, and the pilot's recipe was too long |
| Curve | otherwise | length does not explain the result |
| Classes | gain on the taught classes above the gain on the rest by more than 0.03, as a mean over the seeds | it learns the classes taught, as the frozen detectors did |
| Ceiling | any value | the most these detectors reach with every label of this side |

**Prediction.** The control's fine-tuning stays within 0.05 of the direction on every encoder. The
curve shows no shorter run ahead. Gains on taught classes lie above those on the rest. No prediction
for the ceiling, which is measured to see the room there is. These have no earlier interval behind
them: none of this was read before.

**Trial, before this declaration.** Every section at seed 1 ran end to end, with the curve at 5
events and 200 steps and the ceiling at 99 events, and the class reading ran from the pilot's cells.
The first attempt found a step count breaking a file name, fixed before these cells. The tables were
written and only their arms and the cost read, and the cells were deleted. The control was run again
after its cut changed.

**Before the run: what a review changed.** A review of the code before this run found the first cut
of the control unreadable: 19 of its 61 read months held only event windows, each then read against
itself, so a perfect detector would have reached about 0.56. It also found the first class reading
biased by how many events a half holds, by up to about 0.20 with no class learnt. Both are fixed
above. Moving the pilot's code into functions changed nothing: at 30 steps on the CPU the
committed code and the new one wrote the same bytes in all three cells.

**What repeating a cell on MPS moves.** The same cell, seed and code, run three times on MPS (5
events, seed 3), differs by up to 0.023 and 0.028 in area and by 0.065 and 0.081 in recall at 1 %, under
`base1` and the initialisation, with rank correlations of 0.91 to 0.93 and 0.79 to 0.83 between
runs. The batches and the start are the same, so this is the accelerator's own arithmetic. It sits
inside the spread over seeds that every threshold here reads; a single cell's area is not exact to
better than about 0.02, and a gap of 0.03 between two means over three seeds is near what this noise
reaches.

**Limitations.** The control reads windows the pretrained encoders have read, so it says whether
fine-tuning can learn, not whether pretraining helps. An event's halves are close in time, so the
control is easy by design. The ceiling is these detectors' at this side's labels, not the task's
limit; reading each month against itself caps a perfect detector at about 0.90 on these months, as
three of them are half or more event windows, and the trees are read against the fitting side. The
best reading without a label is the cells' farthest neighbour at 0.154, and the states' is 0.090.
Three seeds; these areas guide what to try next and confirm nothing.
