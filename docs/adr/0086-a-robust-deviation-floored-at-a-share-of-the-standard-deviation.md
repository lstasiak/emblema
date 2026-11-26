# ADR-0086: A publication may floor the robust deviation at a share of the standard deviation; the wind farms take a twentieth

- Status: accepted
- Date: 2026-11-26
- Extends ADR-0080; replaces the consequence of ADR-0085 that the signals with no robust spread
  saturate at the bound

## Context

A robust score divides a value's distance from the median by 1.4826 times the median absolute
deviation (MAD), and is held to a bound the publication names (ADR-0080). Where the MAD is
nought the standard deviation stands in. A signal held at one value most of the time, with
sensor noise around it, has a MAD just above nought, and the stand-in never applies.

Kelmarsh's blade angle sits near 0° below rated wind (median and MAD 0.0245°, standard
deviation 15.2°). Its robust deviation is 0.036°, so the bound of 100 cuts every angle beyond
3.7°: 11 % of the published hours sit at the bound, and pitching above rated wind cannot be told
from blades feathered at rest. ADR-0085 accepted this saturation because no forecast read the
signal. The cold-start readings since then read the blade angle as one of six operating
conditions of the forecast hour, for the farm's trees, the ridges and the fine-tuned forecaster
alike. Penmanshiel's blade angle has a MAD of 0.69° and is not cut.

## Decision

- **A publication may name a floor of the robust deviation as a share of the standard
  deviation** (`--robust-floor`): the deviation is the larger of 1.4826 · MAD and the floor times
  the standard deviation. Where the MAD is nought the standard deviation stands in, as before,
  and is never below the floor. The floor is kept with the channel's statistics in the manifest, as the
  bound is. A manifest that states one is written at a fourth version, which a reader that knows
  no floor refuses rather than denormalising wrongly; a manifest that states none keeps its
  version and its bytes.
- **The wind farms are published again with a floor of a twentieth.** The bound of 100 then
  lies at least five standard deviations from the median. Kelmarsh's blade angle is scored in
  units of 0.76° and is cut only beyond 76°, among blades feathered at rest.
- **Both farms take the same rule.** Besides Kelmarsh's three blade angles it changes Kelmarsh's
  grid voltage and Penmanshiel's two gear-oil pressures, which no reading uses; on Penmanshiel
  no signal a forecast reads changes.

## Consequences

- Every reading of the farms that reads Kelmarsh's operating conditions is made again on the
  new publication: its cells, the classical answers that read conditions, the trees and the
  forecaster's answers. The temperatures read as targets keep their values, so the cells keep
  their hours.
- The satellite telemetry and the intensive-care corpora are published with no floor and do not
  change.
- Hours of blades feathered beyond 76° still sit at the bound: that state reads as one value.

## Alternatives considered

- *The quantile score for the farms* (ADR-0079): no bound cuts it, but it changes every channel of
  both farms and every number read on them.
- *Leaving the blade angle out of the conditions*: no publication again, but it drops the signal
  the pitch motors' temperatures follow.
- *A floor of a tenth*: nothing would be cut on either farm, but Penmanshiel's blade angle, which
  is not cut, would change its units too.
