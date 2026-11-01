"""What a bootstrap over units resamples: a pairing a campaign reads by, or a difference of two.

A closed union, as the pairings are: the bootstrap needs the strata the units are drawn in and
the reduction over any draw of them, which every pairing states and which the difference of two
pairings read under one draw states too.
"""

from emblema.evaluation.domain.statistics.anomaly_interaction import AnomalyInteraction
from emblema.evaluation.domain.statistics.paired_units import PairedUnits

type Resampled = PairedUnits | AnomalyInteraction
