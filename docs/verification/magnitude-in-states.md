# Magnitude in the states

Whether the states of the encoder pretrained over the composition lose how large a reading is, or
are worse for the nearest-neighbour detector at every size. On the satellite missions the
pretrained states trail the same encoder at its initialisation on the events whose window holds a
reading beyond ten deviations, and on the second mission on the rest too
([composition-screen.md](composition-screen.md), 2026-10-31, read after the results). The test
here is under control rather than by the events' labels: normal windows are raised and each
encoder's distance is read before and after. A diagnostic: it changes no rule and no claim.

## 2026-10-31 — declared before the run: normal windows raised around their largest reading

**Question.** As a window's largest reading grows, does the pretrained encoder's distance rise less
and less against its initialisation's (the size is lost), or less at every size (a duller
response)? And on the second mission, do the validation's normal windows lie farther from the
fitting side under the pretrained states than under their initialisation?

**Conditions.**

| | |
| --- | --- |
| Code | the commit that adds this section |
| Machine | MacBook Pro M1 Pro, MPS, fp32; tier M backbones, validation side |
| Tasks | `esa-mission1-events` (each window read against its own month, as its campaign) and `esa-mission2-events` (against the fitting side), every fitting window, the validation side |
| Encoders | `base1` (`0a4dc540…`) and `base2` (`48175bbf…`) from the screen; the same shape at its initialisation under seeds 1, 2 and 3; each read at both blocks side by side, pooled by the mean, as `frozen_knn` and `untrained_knn` |
| Detector | the campaign's: five nearest fitting windows, features standardised over the fitting side |
| Raised windows | 300 normal validation windows of each mission whose largest reading lies within ten deviations, drawn by seed 2 (none of them among the trial's); around the window's largest reading, its cell (the channel's readings in its sixth of the window), its channel or the whole window multiplied by 3, 10 and 30 |
| Command | `scripts/magnitude_response_report.py measure` per task, then `read` |

**What was seen before this section.** Nothing of the rise. The trial ran the exact command on the
second mission with 5 raised windows, both backbones and all three initialisations (8 s an encoder),
and the measure step on the first mission with 5 windows, one initialisation and ×3 alone (61,363
fitting and 41,088 scored windows, about 3 minutes an encoder, 4.5 GB resident). Every encoder's
scores reproduced its campaign's on all 10,223 windows of the second mission, the largest
difference 8·10⁻⁶ and the correlation 1.000000. The trial's reading was printed with its digits
masked, but not its words: on those 5 windows, the first 5 of the 300 drawn here, the rule's
columns read "size lost: no; duller: yes" on the second mission, under a first version of the
rule over the distance. An independent review then found that rule could not tell the hypotheses
apart and let a negative rise pass. The rule now reads the displacement, against the
initialisations' own range, and the windows are drawn by another seed. The predictions below were
written before the trial and are left as they were.

**Reading.** `scripts/magnitude_response_report.py read`.

- *Measures.* The rule reads the **displacement**: how far a raised window's state moves from its
  unraised state, in the fitting side's spread per feature, its median over the 300 windows. No
  neighbour enters it. The detector's **distance** is read beside, descriptively: its rise over
  the unraised distance, in the median unraised distance. It also depends on how far the window
  lay from its neighbours before it was raised. When that is far, a small move adds almost
  nothing, so the two hypotheses could look alike on the distance.
- *Sensitivity.* An encoder's median displacement over the mean of the initialisations'. Each
  initialisation is read against the other two, as the reference for the draw of weights. The
  95 % intervals come from 2,000 resamples of the windows, the same draw for every encoder. The
  intervals do not carry the draw of weights; the initialisations' own points do.
- *Tilt.* The sensitivity at ×10 over that at ×3, from the same resamples.
- *Rule, on the channel scope, per mission.* A pretrained encoder **falls** when its tilt is at
  most 0.75, its interval below one, and its point below every initialisation's own. It is
  **duller** when its sensitivity at ×3 is at most 0.8, its interval below one, and its point
  below every initialisation's. The size is **lost** when both pretrained encoders fall. The
  response is **duller** when both are duller. The rule is void when a rise it divides is not
  above zero (at the point or in more than 2.5 % of resamples), or when an initialisation's own
  tilt is at most 0.75 with its interval below one. The cell and window scopes, ×30, the
  distance and the reference are descriptive. On the first mission the campaign reads each
  window against its own month; the rule does not, so the check alone is "as the campaign".
- *Reference.* Per encoder: the median distance of the validation's normal windows, over the
  median distance of 2,000 fitting windows to the other fitting windows. Beside it, the
  participation ratio of the fitting states, raw and standardised.
- *Check.* Each encoder's scores reproduce its campaign's, window by window, to the precision of
  the device. The events' areas by the largest reading reproduce the reading after the screen.

**Predictions.** No earlier reading measured a rise; the directions come from the window-level
areas of the reading after the screen, seeds 1 to 3.

| Reading | Prediction | Source |
| --- | --- | --- |
| M1, channel | size lost; not duller | events beyond ten deviations: initialisation 0.77–0.83, states 0.73–0.75; within ten, 0.49–0.51 and 0.50–0.52 |
| M2, channel | size lost and duller | beyond ten: 0.95–0.99 and 0.67–0.75; within ten: 0.75–0.79 and 0.55–0.59 |
| Initialisations against each other | tilt within [0.8, 1.25]; the rule not void | one shape, one layout; only the draw differs |
| M2 reference | the ratio higher under both pretrained encoders than under every initialisation | under backbones that read the second mission's fitting months its PR-AUC is 0.06 / 0.10, under four that never read ESA 0.31 to 0.59 (`anomaly-tasks.md`) |
| Check | scores within 10⁻⁴; areas within 0.005 | the trial |

**What follows.** If the size is lost on either mission, a decision record proposes an input that
keeps a reading's size (beside the normalised value in a cell) and it runs as an arm of the
screen, two pretrainings, in the session of the clinical confirmation, never straight to
confirmation. If the size is not lost, no input is changed for this reason. Whether the
telemetry leg is read again on the confirmation's pretrainings is then decided before the
confirmation is declared. If the states keep the size and are not duller, the trailing areas
point to the reference rather than the response. The reference is read as a candidate mechanism
for the second mission, not a verdict.

**Limitations.** A raised window is a model of an event, not an event: the readings of a channel
are scaled, not shifted, so a channel near its mean barely moves. The cell is one of a window's
hundreds under the mean pooling, so its rise is small for every encoder and its ratio noisy. The
trial's glimpse of the first rule's words on 5 windows is named above. Two
pretrainings, one recipe. Validation side, tier M.
