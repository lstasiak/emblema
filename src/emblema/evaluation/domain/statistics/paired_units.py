"""Two candidates' answers over the same units, paired, whichever measure reads them.

A closed union rather than a protocol: the measures a campaign may read by are a set this
context names. Each pairing states what a bootstrap resamples: the units, the strata they are
drawn in, the errors of both sides and their difference over any draw of the units. Squared
errors pool by addition within a unit; areas under the ROC curve do not split by unit and are
swept over the draw instead.
"""

from emblema.evaluation.domain.statistics.paired_unit_errors import PairedUnitErrors
from emblema.evaluation.domain.statistics.paired_unit_rankings import PairedUnitRankings

type PairedUnits = PairedUnitErrors | PairedUnitRankings
