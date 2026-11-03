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
