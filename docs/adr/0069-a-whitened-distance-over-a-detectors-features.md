# ADR-0069: A whitened distance over a detector's features, beside the nearest neighbours

- Status: accepted
- Date: 2026-10-31

## Context

The detection tasks (ADR-0067) score a window by its mean Euclidean distance to the nearest
fitting windows over standardised features: the states of a frozen encoder, the channels'
summaries or their counts. Under that distance the pretrained encoder's states trail the same
encoder at its initialisation on both satellite missions.

Two readings of the states explain why without blaming the states. Raised normal windows move
the pretrained states as far as the drawn ones, so no size is lost
([magnitude-in-states](../verification/magnitude-in-states.md)). But the pretrained states spread
the normal windows' variation over 7–9 principal components where the drawn states hold it in
1–2, and the events' signal with it. A Euclidean distance over 128 correlated features weighs
the few leading directions almost alone, so a departure spread over many directions barely moves
it. Scored instead by the squared Mahalanobis distance from the fitting mean under the fitting
side's covariance, the pretrained states read the second mission's events at 0.74 and 0.81
against 0.61 and 0.67, and the far events level with the initialisations (0.92–0.99 against
1.00); the drawn states hardly change
([anomaly-signal-in-states](../verification/anomaly-signal-in-states.md)). The whitened distance
is not a better detector of pretraining's advantage: by the declared rule no distance showed one.
It is a fairer instrument for both arms, and a fairer instrument is what the screen that
reformulates the telemetry leg needs.

A covariance of 128 features estimated on a side of 15,000 windows inverts; one estimated on
fewer windows than features does not, which is where TimeRep (arXiv 2509.12650) gave up the same
distance over a foundation model's states.

## Decision

- **The distance is a knob of every detector over features**, `distance`, with two values:
  `neighbours`, the mean Euclidean distance to the nearest fitting windows as before, and
  `whitened`, the squared Mahalanobis distance from the fitting mean. The default stays
  `neighbours`, so every campaign declared so far reads as it was stored, and a stored campaign
  names the distance only where it is whitened.
- **The whitened detector standardises as the nearest-neighbour one does**, then whitens by the
  inverse square root of the standardised fitting rows' covariance, shrunk towards a scaled
  identity by Ledoit and Wolf's closed form, so it inverts whatever the side's size and shrinks
  hardly at all where the side is large. The score is the squared length of the whitened row.
- **The same alarm and the same reference.** The alarm is set over the fitting rows' own scores
  at the declared contamination; `reference=unit` reads each score against its month as it does
  for the nearest neighbours. Neighbours are not named under the whitened distance; a detector
  scored by an error has no distance to choose, and both are refused.
- **One place fits a detector over features**, by the recipe's distance, for the runtime over
  states and the one over summaries alike, and keeps it under its own format.

## Consequences

- The screen of the reformulated telemetry leg reads the states detector as
  `frozen_knn@distance=whitened` against `untrained_knn@distance=whitened`, and the summaries
  may be read the same way. The choice was made on the two backbones the screen reads, so it is
  confirmed on the fresh pretrainings of the clinical confirmation, under a rule written before
  they run.
- The window-level areas and the register's event-weighted PR-AUC can disagree under a whitened
  distance, as they did on the first mission's subtle events; the register's measure decides.
- A new stored form, `whitened-gaussian-npz`; a campaign cell under the whitened distance costs
  one eigendecomposition of a 128 × 128 matrix more than the nearest neighbours, and no search.
- The alarm is set over fitting rows that sit inside the covariance they are scored under, so
  it flags a few more fresh normal windows than declared where the side is small: about the
  share of features over windows, 1 % at campaign scale and more in a tier-S pilot.
- The nearest-neighbour detector and every result read under it stand as they are.
