from emblema.pretraining.domain.assessment.assessment import Assessment, assess
from emblema.pretraining.domain.assessment.results import Results
from emblema.pretraining.domain.assessment.summarised_run import SummarisedRun
from emblema.pretraining.ports.mask_kind_summariser import MaskKindSummariser


class AssessReconstructionRun:
    """Judges what a masked-reconstruction run measured and says what, if anything, has to change.

    The numbers of a run become a decision by rules stated once and applied the same way to every
    run. Each run's tallies are summarised by the same summariser, the shorter run's too, so that a
    verdict that changed between the two changed in the run and not in the statistics.
    """

    def __init__(self, summariser: MaskKindSummariser) -> None:
        self._summariser = summariser

    def __call__(self, results: Results, shorter: Results | None = None) -> Assessment:
        """Every rule applied to ``results``, the convergence judged against ``shorter``.

        Raises:
            IncomparableRunsError: If ``shorter`` is not a run of the same configuration with at
                most half the epochs.
        """
        return assess(
            self._summarised(results),
            None if shorter is None else self._summarised(shorter),
        )

    def _summarised(self, results: Results) -> SummarisedRun:
        return SummarisedRun(results, self._summariser.summarise(results.tallies))
