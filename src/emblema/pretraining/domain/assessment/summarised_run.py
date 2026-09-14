from dataclasses import dataclass

from emblema.pretraining.domain.assessment.kind_summary import KindSummary
from emblema.pretraining.domain.assessment.results import Results


@dataclass(frozen=True)
class SummarisedRun:
    """A run beside the summaries of its kinds of mask, which is what the rules read.

    The summaries carry the uncertainty of the run's comparisons, which only resampling its units
    gives; they are made outside the domain and kept with the run they were made from, so that a
    rule comparing two runs reads each run's verdicts beside that run's own measurements.

    Attributes:
        results: What the run measured.
        summaries: Its kinds of mask with their intervals, one per kind and side of the channels
            apart, the channels apart last.
    """

    results: Results
    summaries: tuple[KindSummary, ...]
