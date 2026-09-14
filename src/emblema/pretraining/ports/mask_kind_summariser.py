from collections.abc import Iterable
from typing import Protocol

from emblema.pretraining.domain.assessment.kind_summary import KindSummary
from emblema.pretraining.domain.assessment.mask_kind_tally import MaskKindTally


class MaskKindSummariser(Protocol):
    """Turns per-group tallies into one summary per kind of mask, with its comparisons' uncertainty.

    The seam between the rules, which read summaries, and the statistics that give a comparison
    its interval: how groups are resampled is the adapter's to decide and the rules never see it.
    """

    def summarise(self, tallies: Iterable[MaskKindTally]) -> tuple[KindSummary, ...]:
        """One summary per kind of mask and side of the channels apart, the channels apart last.

        A summary's errors are ratios of sums over the hidden tokens of every group; its intervals
        say how much lower the model's error is than the matched and the linear baseline's.
        """
        ...
