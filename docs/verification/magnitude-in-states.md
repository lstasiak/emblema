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

## 2026-10-31 — M1 Pro, MPS, fp32: the states keep a reading's size; the detector's distance loses it to the normal windows' spread

**Question.** As declared above.

**Conditions.**

| | |
| --- | --- |
| Code | `eab58ac0`, the declaration's commit, clean tree |
| Machine | MacBook Pro M1 Pro, MPS, fp32; about 8 s an encoder on the second mission, 170 s on the first |
| Windows | second mission: 15,335 fitting, 10,223 scored (475 in events); first: 61,363 and 41,088 (4,759); 300 normal windows of each raised 2,700 ways, drawn by seed 2 |
| Command | `data/report/l34a/run.sh` (both tasks' `measure`, then `read` with the campaigns' predictions) |

**Check.** On the second mission every encoder's scores match its campaign's on all 10,223
windows, the largest difference 8·10⁻⁶. On the first mission each score is read against its
month and scaled by the month's spread, up to 119. The largest difference is 2.5·10⁻³, on a score
of 114 (2·10⁻⁵ of it), the median 8·10⁻⁵, the correlation 1.000000 for every encoder. The
declared 10⁻⁴ did not allow for that scaling; the device's precision, scaled, accounts for it.
The events' areas match the reading after the screen on both missions (table below).

**Rule: displacement on the channel scope** (sensitivity against the initialisations, 95 %
interval over windows; an initialisation against the other two).

| Mission | Encoder | ×3 | ×10 | Tilt ×10 / ×3 |
| --- | --- | --- | --- | --- |
| M1 | base1 | 1.10 [1.07; 1.14] | 0.97 [0.95; 1.00] | 0.88 [0.86; 0.90] |
| M1 | base2 | 1.41 [1.36; 1.46] | 1.34 [1.30; 1.37] | 0.95 [0.93; 0.96] |
| M1 | initialisations | 0.63 to 1.42 | 0.63 to 1.51 | 0.93 to 1.07 |
| M2 | base1 | 0.85 [0.80; 0.91] | 0.93 [0.90; 0.95] | 1.09 [1.03; 1.14] |
| M2 | base2 | 0.97 [0.96; 0.99] | 1.20 [1.19; 1.22] | 1.24 [1.21; 1.25] |
| M2 | initialisations | 0.76 to 1.17 | 0.70 to 1.27 | 0.92 to 1.08 |

Verdict on both missions: **size lost: no; duller: no.** No tilt reaches 0.75, and every
pretrained point at ×3 lies within or above the initialisations' range. The rule was never void.

**Descriptive.** The cell scope: the pretrained states move further than their initialisation,
1.1 to 1.5 times on M1 and 1.8 to 2.7 on M2. The whole window: the pretrained states keep pace on
M1 (tilt 0.87 to 1.00). On M2 they lag, 0.59 and 0.76 at ×3, 0.43 and 0.48 at ×30 (tilt 0.84,
0.78). Raising the whole window is not what the events look like.

**Descriptive: the detector's distance on the channel scope**, with what lies behind it.

| Mission | Encoder | Distance sensitivity ×3 | ×10 | Fitting distance | Normal validation distance | Ratio | Directions, standardised |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | base1 | 0.14 | 0.13 | 1.010 | 3.567 | 3.53 | 2.0 |
| M1 | base2 | 0.38 | 0.34 | 0.906 | 2.992 | 3.30 | 2.1 |
| M1 | initialisations | 0.68 to 1.20 | 0.71 to 1.23 | 0.33 to 0.39 | 0.96 to 1.22 | 2.88 to 3.15 | 1.2 |
| M2 | base1 | 0.47 | 0.30 | 1.081 | 4.441 | 4.11 | 4.5 |
| M2 | base2 | 0.53 | 0.53 | 1.228 | 4.375 | 3.56 | 5.3 |
| M2 | initialisations | 0.72 to 1.17 | 0.69 to 1.22 | 1.10 to 1.26 | 2.17 to 2.25 | 1.76 to 2.04 | 2.3 to 2.6 |

**Events against normal windows** (window-level AUROC, the campaign's scores).

| Mission | Encoder | Beyond ten deviations | Within ten |
| --- | --- | --- | --- |
| M1 | base1 / base2 | 0.729 / 0.748 | 0.499 / 0.516 |
| M1 | initialisations | 0.774 to 0.829 | 0.492 to 0.510 |
| M2 | base1 / base2 | 0.671 / 0.747 | 0.549 / 0.594 |
| M2 | initialisations | 0.945 to 0.990 | 0.749 to 0.785 |

**Predictions.**

| Reading | Predicted | Observed | Held |
| --- | --- | --- | --- |
| M1, channel | size lost; not duller | not lost; not duller | no; yes |
| M2, channel | size lost and duller | neither | no |
| Initialisations against each other | tilt in [0.8, 1.25], rule not void | 0.92 to 1.08; not void | yes |
| M2 reference | ratio higher under both pretrained encoders | 4.11 and 3.56 against 1.76 to 2.04 | yes |
| Check | scores within 10⁻⁴; areas within 0.005 | M2 8·10⁻⁶; M1 2.5·10⁻³ on scores up to 119; areas as before | M2 yes; M1 no as worded |

**Conclusions.**

1. The pretrained states keep a reading's size. When one channel or one cell is raised they
   move as far as their initialisation's, or further, on both missions, and no more slowly as
   the reading grows. An input that keeps the size is not called for by this test.
2. What the detector loses is contrast, not response. The same move adds a seventh to a third as
   much distance on M1, and a third to a half on M2. Under the pretrained states the fitting
   windows lie 2.3 to 3 times as far apart on M1 (0.91–1.01 against 0.33–0.39). Their
   variation also spreads over more directions (2.0 against 1.2 on M1; 4.5–5.3 against 2.3–2.6
   on M2), so a raised channel stands out less among them.
3. On the second mission the held-out months lie twice as far from the fitting months under the
   pretrained states as under their initialisation (ratio 3.6–4.1 against 1.8–2.0). Every
   window there, normal or not, starts far from its neighbours. This matches the area within
   ten deviations falling from 0.75–0.79 to 0.55–0.59 there, and not on the first mission,
   whose windows are read against their own month.
4. The initialisation's states vary along about one standardised direction on M1 (1.2), so its
   distance reads little beyond one direction of variation, along which large readings stand
   out. Read after the results, not tested.

**Limitations.** The raised windows are a model of an event: readings scaled around their
largest, not the shapes the missions' events take. The intervals resample windows, not the draw
of weights; the initialisations' own range stands in for it. Two pretrainings, one recipe. The
mechanism in conclusions 2 and 3 is read after the results and is not tested here. Validation
side, tier M.
