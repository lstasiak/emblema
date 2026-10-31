# Where the anomaly signal sits in the states

Which distance over a frozen encoder's states reads the satellite missions' events, and why. The
raised windows ([magnitude-in-states.md](magnitude-in-states.md)) showed that the pretrained
states keep a reading's size and that the nearest-neighbour detector loses contrast instead: the
normal windows' variation spreads over more directions under pretraining, and on the second
mission the held-out months lie farther from the fitting months. This note reads the same states
through other distances, each aimed at one of those two mechanisms, to decide which detectors
enter the screen that reformulates the telemetry leg. A diagnostic: it changes no rule and no
claim, and nothing it finds is confirmed until fresh pretrainings read it.

## 2026-10-31 — declared before the run: the events along the components, and four distances

**Question.** (1) Along which principal components of the normal fitting states do the events
stand apart: the components that carry most of the variance, or the residual ones? (2) Does a
distance that whitens the states, or one that reads each window against its own month's centre,
gain area over the mission's own campaign detector under the pretrained encoders, beyond what
the draw of weights alone moves?

**Conditions.**

| | |
| --- | --- |
| Code | the commit that adds this section |
| Machine | MacBook Pro M1 Pro, MPS, fp32; tier M backbones, validation side |
| Tasks | `esa-mission1-events` and `esa-mission2-events`, every fitting window, the validation side, drawn by the campaign's use case |
| Encoders | `base1` (`0a4dc540…`) and `base2` (`48175bbf…`) from the screen; the same shape at its initialisation under seeds 1, 2 and 3; both blocks side by side, pooled by the mean, as `frozen_knn` and `untrained_knn` read them |
| States | 128 features a window, standardised over the fitting side as the campaign's detector standardises them; every scorer reads these coordinates |
| Command | `scripts/state_geometry_report.py measure` per task, then `read` with the campaigns' exported predictions |

**Scorers.** All over the same standardised states; the fitting side's covariance is the
Ledoit-Wolf shrinkage of the sample covariance.

| Name | What it reads | Mechanism it tests |
| --- | --- | --- |
| `knn` | the campaign's detector: mean distance to the 5 nearest fitting windows, alarm over the fitting side (the second mission's campaign, and its control) | drift, when read on the first mission |
| `knn_unit` | the same distance read against the window's own month's median and spread (the first mission's campaign, and its control) | drift of the months, on the distance |
| `knn_whitened` | the same nearest-neighbour distance after the states are whitened by the fitting covariance | the spread over directions |
| `mahalanobis` | squared Mahalanobis distance from the fitting mean | the spread, as one Gaussian |
| `mahalanobis_top` / `mahalanobis_residual` | the Mahalanobis distance split between the components that carry 90 % of the fitting variance and the rest | where the signal sits |
| `mahalanobis_month` | squared Mahalanobis distance from the window's own month's coordinate-wise median, under the fitting covariance | drift of the months, on the whitened states |

The month's centre is its median over all its scored windows, events among them, as the
campaign's own-month reference reads them: a median stands up to the few events a month holds. A
month's own covariance would not (the second mission holds about 700 windows a month in 128
dimensions, 4.6 % of them in events), so the covariance stays the fitting side's. A relative
Mahalanobis distance with a Gaussian per month was tried on synthetic states and read the events
below the normal windows for that reason; it is not run here.

**Reading.** `scripts/state_geometry_report.py read` (2,000 resamples, seed 1).

- *Profile.* Per encoder: the share of variance each principal component of the standardised
  fitting states carries, and the area of `|projection|` of the events' windows against the normal
  windows along it. The table reports the leading five, the best component of the top block and
  of the residual, and the areas of `mahalanobis_top` and `mahalanobis_residual`.
- *Areas.* Window-level AUROC of events against normal windows, all events and split by whether
  the window holds a reading beyond ten deviations; the register's event-weighted PR-AUC beside.
- *Control.* The mission's own campaign detector: `knn` on the second mission, `knn_unit` on the
  first, so a scorer never enters by re-admitting the detector the campaign already runs.
- *Gains.* For `knn_whitened`, `mahalanobis_month` and the other campaign detector: the area over
  all events less the control's, per encoder; 95 % intervals from resampling the months, the
  same draw for every encoder and scorer; a resample that leaves no event is dropped. The
  intervals carry the months, not the draw of weights; the initialisations' own gains stand in
  for that, as points.
- *Rule, per task and scorer.* A scorer **enters the screen** when both pretrained encoders gain
  at least 0.02 over the control with the interval clear of zero, and each gain exceeds every
  initialisation's gain under the same scorer (a point, not an interval). `knn_whitened` needs
  besides that `mahalanobis_residual` reads the events at least as well as `mahalanobis_top`
  under both pretrained encoders, since that is the mechanism by which whitening helps. Void
  with fewer than two pretrained encoders or two initialisations. The rule reads what a scorer
  does for the pretrained states beyond what it does for the drawn ones, not which scorer is the
  best detector. Everything else is descriptive.
- *Check.* `knn` reproduces the second mission's campaign scores and `knn_unit` the first's,
  window by window, to the precision of the device; the events' areas reproduce the earlier
  readings.

**What was seen before this section.** The scorers were checked on synthetic states of known
geometry (`tests/scripts/test_state_geometry_report.py`): events moved along a quiet direction
are read by the residual and by whitening, events along the loudest direction by the top block
and not helped by whitening, and a month drifted whole is read by the month-centred distance.
The trial ran `measure` and `read` on the second mission (5 to 6 s an encoder, 41 s to read,
126 MB of states); `knn` matched both campaigns on all 10,223 windows, largest difference
7·10⁻⁶. The trial's reading was printed with its digits masked but not its signs: under `base1`
the gains of `knn_whitened` and `knn_unit` over `knn` showed as positive with intervals above
zero, and the interval of `mahalanobis_month` as crossing zero. The predictions below were
written before the trial and are left as they were. An independent review then made the control
the mission's own campaign detector, since on the first mission a gain over `knn` would re-admit
`knn_unit`, and had a thin month read against the fitting centre instead of stopping the
reading.

**Predictions** (directions from the raised windows and the areas after the screen; no earlier
reading measured these quantities).

| Reading | Prediction | Source |
| --- | --- | --- |
| Profile, initialisations | the events stand apart along the first component; the residual's best component near 0.5 | participation ratio 1.2 (M1), 2.3–2.6 (M2): the drawn encoder reads a window's level |
| Profile, pretrained, events beyond ten deviations | in the top block, not the residual | a raised reading moves the states as the initialisation's, along directions of large variance |
| Profile, events within ten deviations, M1 | no component above 0.55 | every detector near chance there, 0.49–0.59 |
| `knn_whitened` | does not enter on either mission: within ±0.02 of the control or below | the profile prediction; Sun et al. (kNN resilient), Rippel Table IV (residual directions mis-calibrate) |
| `mahalanobis_month`, M2 | enters: both pretrained gain within [+0.02; +0.15] over `knn`; still below the initialisations' `knn` | drift 3.6–4.1 against 1.8–2.0; the gap within ten deviations 0.75–0.79 against 0.55–0.59 bounds the gain |
| `knn_unit`, M2 | enters, with a smaller gain than `mahalanobis_month` | rescales one distance a month; on M1 the own-month reference left the states 0.018 short of the initialisation |
| `mahalanobis_month` and `knn`, M1 | neither enters: within ±0.02 of `knn_unit` | the control already reads each window against its month; the first mission's states were not short of the initialisation by drift |
| Initialisations under the three scorers | gains smaller than the pretrained encoders' on M2 | their drift is half the pretrained states' |

**What follows.** A scorer that enters joins the telemetry screen of the reformulated leg beside
the detectors scored by the pretext's residual and the few-label grid; one that does not is left
out with its reading as the reason. The screen runs on `base1` and `base2`, so whatever it
chooses is confirmed on the fresh pretrainings of the clinical confirmation, under a rule written
before they run.

**Limitations.** Two pretrainings, one recipe; the diagnostic's choices are made on the same
backbones the hypotheses came from. Window-level AUROC is not the register's measure. The second
mission has about 14 validation months, so its intervals are wide. The month's centre is
transductive: the window's own month, without labels, as the campaign's reference is. The trial's
glimpse of the gains' signs on the second mission is named above. Validation side, tier M.

## 2026-10-31 — M1 Pro, MPS, fp32: no distance over the states enters; whitening closes most of the gap on the second mission, and every encoder reads the first mission's subtle events alike

**Question.** As declared above.

**Conditions.**

| | |
| --- | --- |
| Code | `17bedeb6`, the declaration's commit, clean tree |
| Machine | MacBook Pro M1 Pro, MPS, fp32; 5–6 s an encoder on the second mission, 130–165 s on the first; the reading 2,000 resamples |
| Windows | second mission 15,335 fitting, 10,223 scored (475 in events, 14 months); first 61,363 and 41,088 (4,759 in events) |
| Command | `data/report/l35/geometry/run.sh` |

**Check.** `knn` reproduces both second-mission campaigns on all 10,223 windows (largest
difference 8·10⁻⁶) and `knn_unit` both first-mission campaigns on all 41,088 (2.5·10⁻³ on
scores scaled per month up to 119), correlation 1.000000 throughout. The `knn` and `knn_unit`
areas by the largest reading match the readings after the screen and the raised windows.

**Rule.** No scorer enters on either mission.

| Mission | Scorer | base1 | base2 | Initialisations (range) | Enters |
| --- | --- | --- | --- | --- | --- |
| M2, over `knn` | `knn_whitened` | +0.119 [+0.059; +0.177] | +0.117 [+0.042; +0.180] | −0.010 to +0.023 | no: the top components separate the events better than the residual |
| M2 | `mahalanobis_month` | +0.045 [−0.083; +0.212] | +0.095 [−0.045; +0.256] | −0.060 to −0.046 | no (intervals cross zero) |
| M2 | `knn_unit` | +0.047 [+0.001; +0.097] | +0.026 [−0.018; +0.085] | +0.012 to +0.027 | no |
| M1, over `knn_unit` | `knn_whitened` | +0.125 [−0.015; +0.266] | +0.103 [−0.039; +0.257] | +0.121 to +0.142 | no |
| M1 | `mahalanobis_month` | +0.220 [+0.098; +0.337] | +0.201 [+0.072; +0.330] | +0.217 to +0.230 | no (the initialisations gain as much) |
| M1 | `knn` | +0.088 [−0.046; +0.222] | +0.109 [−0.040; +0.256] | +0.093 to +0.120 | no |

**Where the signal sits** (components of the standardised fitting states; areas of events against
normal windows).

| Mission | Encoder | Components to 90 % | First component's share | Best component, top / residual | Mahalanobis top / residual |
| --- | --- | --- | --- | --- | --- |
| M2 | base1 / base2 | 7 / 9 | 0.38 / 0.35 | 0.81 / 0.73, 0.81 / 0.76 | 0.820 / 0.737, 0.757 / 0.811 |
| M2 | initialisations | 3–4 | 0.57–0.62 | 0.82–0.89 (the third component) / 0.80–0.86 | 0.84–0.87 / 0.86–0.87 |
| M1 | base1 / base2 | 5 / 4 | 0.69 / 0.68 | 0.65 / 0.68, 0.65 / 0.67 | 0.635 / 0.665, 0.618 / 0.658 |
| M1 | initialisations | 1–2 | 0.89–0.92 | 0.50 / 0.67–0.72 | 0.50 / 0.66–0.69 |

**Areas** (window-level AUROC all events / beyond ten deviations / within; the register's
event-weighted PR-AUC last).

| Mission | Scorer | base1 | base2 | Initialisations |
| --- | --- | --- | --- | --- |
| M2 | `knn` (campaign) | 0.607 / 0.671 / 0.549; 0.042 | 0.666 / 0.747 / 0.594; 0.071 | 0.84–0.88 / 0.95–0.99 / 0.75–0.79; 0.31–0.64 |
| M2 | `knn_whitened` | 0.726 / 0.864 / 0.603; 0.129 | 0.783 / 0.915 / 0.664; 0.279 | 0.86–0.88 / 0.99 / 0.74–0.78; 0.60–0.66 |
| M2 | `mahalanobis` | 0.743 / 0.917 / 0.588; 0.123 | 0.811 / 0.985 / 0.655; 0.325 | 0.86–0.87 / 1.00 / 0.73–0.76; 0.60–0.67 |
| M2 | `mahalanobis_month` | 0.651 / 0.742 / 0.571; 0.086 | 0.761 / 0.900 / 0.637; 0.213 | 0.79–0.82 / 0.96–0.99 / 0.64–0.67; 0.30–0.45 |
| M1 | `knn_unit` (campaign) | 0.528 / 0.729 / 0.499; 0.086 | 0.545 / 0.748 / 0.516; 0.105 | 0.53–0.54 / 0.77–0.83 / 0.49–0.51; 0.07–0.09 |
| M1 | `knn` | 0.616 / 0.969 / 0.565; 0.033 | 0.654 / 0.993 / 0.605; 0.026 | 0.63–0.66 / 0.99 / 0.57–0.62; 0.02–0.04 |
| M1 | `mahalanobis` | 0.665 / 0.988 / 0.618; 0.026 | 0.659 / 0.991 / 0.610; 0.020 | 0.66–0.69 / 1.00 / 0.62–0.64; 0.02 |
| M1 | `mahalanobis_month` | 0.749 / 0.985 / 0.714; 0.027 | 0.747 / 0.985 / 0.712; 0.022 | 0.75–0.77 / 0.98 / 0.72–0.74; 0.03–0.04 |

**Predictions.**

| Reading | Predicted | Observed | Held |
| --- | --- | --- | --- |
| Profile, initialisations | first component; residual near 0.5 | M2: the third component (0.82–0.89), the first 0.42–0.52; M1: the first component 0.50 and the residual 0.67–0.72 | no |
| Profile, pretrained | top block | M2: top 0.81 against residual 0.73–0.76 (`mahalanobis_top` 0.82 under base1, but 0.76 against 0.81 under base2); M1: residual slightly ahead (0.67 against 0.65) | M2 partly; M1 no |
| M1 events within ten deviations | no component above 0.55 | up to 0.65 (pretrained) and 0.67 (initialisations) | no |
| `knn_whitened` | within ±0.02 of the control or below | M2 +0.12 for both pretrained, intervals clear of zero; M1 +0.10 to +0.14 for every encoder | no; the verdict "no" holds through the mechanism clause alone |
| `mahalanobis_month`, M2 | enters, [+0.02; +0.15] | +0.045 and +0.095, intervals crossing zero; the initialisations −0.05 | no |
| `knn_unit`, M2 | enters, below `mahalanobis_month` | +0.047 and +0.026, one interval touching zero | no |
| `mahalanobis_month` and `knn`, M1 | within ±0.02 of `knn_unit` | +0.20 and +0.09 to +0.11, for every encoder alike | no |
| Initialisations' gains, M2 | smaller than the pretrained encoders' | yes for all three scorers | yes |

**Conclusions.**

1. By the declared rule no distance over the states enters the screen. On the second mission the
   whitened distance gains 0.12 for both pretrained encoders and nothing for the drawn ones, but
   the gain does not come from the residual components the mechanism clause asked for. On the
   first mission every scorer lifts every encoder alike, so none shows an advantage of
   pretraining.
2. The campaign's Euclidean distance was a poor instrument for the pretrained states. On the
   second mission a whitened Gaussian reads them at 0.74 and 0.81 against 0.61 and 0.67, and on
   the events beyond ten deviations at 0.92 and 0.99, level with the initialisations' 1.00. The
   gap to the initialisations shrinks from 0.2 to 0.05–0.12 and stays within ten deviations
   (0.59–0.66 against 0.73–0.76). The pretrained states spread the events' signal over 7–9
   components where the drawn states hold it in one or two, and a Euclidean distance over
   correlated features weighs the first few directions alone.
3. The first mission's subtle events, 87 % of them without a reading beyond ten deviations, are
   read at 0.71–0.74 by a Mahalanobis distance from the window's own month's centre, under
   pretrained and drawn encoders alike, where every detector of the screen read them at
   0.49–0.59. The event-weighted PR-AUC does not follow (0.02–0.04 against 0.07–0.10 for the
   campaign's detector): the scorer separates the events in the middle of the ranking, not at
   its top, so it raises more false alarms before the first events.
4. The drawn encoder's dominant direction on the first mission (89–92 % of the variance) carries
   no signal (0.50) and reads the far events backwards (0.23); the drawn states' signal sits in
   their residual directions, the pretrained states' in both. Neither geometry is the one the
   declaration predicted.
5. Read after the results: the distance's reference and metric decide more than the encoder's
   weights. The pretrained encoder is not a worse feature extractor for these events than a drawn
   one once the metric suits its geometry, and not a better one either.

**Limitations.** Two pretrainings, one recipe; the diagnostic read the backbones the hypotheses
came from. Window-level AUROC is not the register's measure, and the two disagree in conclusion
3. The second mission's intervals span 14 months. The month's centre is transductive. Validation
side, tier M.
