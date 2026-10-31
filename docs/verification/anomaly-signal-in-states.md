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
