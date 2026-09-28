"""How a window's target is read from what the ground truth says about it.

A closed set of schemes, because which one a task uses decides what the ground truth has to say:
the moment a unit failed, the exact reading a window is asked to forecast, or the outcome its unit
recorded. Every scheme states the kind of its target and the scale it is learnt in.
"""

from emblema.evaluation.domain.labels.forecast_scheme import ForecastScheme
from emblema.evaluation.domain.labels.outcome_scheme import OutcomeScheme
from emblema.evaluation.domain.labels.remaining_life_scheme import RemainingLifeScheme

type LabelScheme = RemainingLifeScheme | ForecastScheme | OutcomeScheme
