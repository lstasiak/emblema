"""How a budget of labels is spread over a task's pool, whichever kind of target it has.

A closed union rather than a protocol, for the reason the label schemes are one: which
stratification a task uses follows from the kind of its target — ranks of a quantity, or the two
outcomes in proportion — and a codec and a task match on the members this context names.
"""

from emblema.evaluation.domain.labels.class_strata import ClassStrata
from emblema.evaluation.domain.labels.target_bins import TargetBins

type Stratification = TargetBins | ClassStrata
